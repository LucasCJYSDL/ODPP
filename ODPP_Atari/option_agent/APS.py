import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from learner import policy
from learner import APS_module
from utils.summary_tools import write_summary
from option_agent.base_option_agent import Option_Agent
from tqdm import tqdm

class APS_Agent(Option_Agent):

    def __init__(self, args):
        super(APS_Agent, self).__init__(args)

        ## policy
        if self.args.is_discrete:
            self.pi_func = policy.CategoricalPolicy(input_dim=(self.args.obs_dim + self.args.code_dim),
                                                    hidden_dim=self.args.pi_hidden_dim,
                                                    action_dim=self.args.act_dim)
        else:
            self.pi_func = policy.GaussianPolicy(input_dim=(self.args.obs_dim + self.args.code_dim),
                                                 hidden_dim=self.args.pi_hidden_dim,
                                                 action_dim=self.args.act_dim, output_activation=F.tanh,
                                                 act_range=self.args.act_range)  # danger

        ## value
        self.pi_value = APS_module.ValueSF(input_dim=(self.args.obs_dim + self.args.code_dim),
                                           hidden_dim=self.args.pi_hidden_dim, sf_dim=self.args.code_dim)

        self.pi_func_params = list(self.pi_func.parameters())
        self.pi_value_params = list(self.pi_value.parameters())
        self.pi_func_optim = Adam(params=self.pi_func_params, lr=self.args.lr)
        self.pi_value_optim = Adam(params=self.pi_value_params, lr=self.args.lr)

        ## decoder
        self.dec_func = APS_module.APS(obs_dim=self.args.obs_dim, sf_dim=self.args.code_dim, hidden_dim=self.args.dec_hidden_dim)

        self.dec_func_params = list(self.dec_func.parameters())
        self.dec_func_optim = Adam(params=self.dec_func_params, lr=self.args.lr)

        # particle-based entropy
        # the parameters are from: https://github.com/rll-research/url_benchmark/blob/main/agent/aps.yaml
        rms = APS_module.RMS(self.args.device)
        self.pbe = APS_module.PBE(rms, knn_clip=0.0001, knn_k=12, knn_avg=True, knn_rms=True, device=self.args.device)

    def eval_mode(self):
        self.pi_func.eval()

    def train_mode(self):
        self.pi_func.train()

    def cuda(self):
        self.pi_func.cuda()
        self.pi_value.cuda()
        self.dec_func.cuda()

    def save_models(self, path):
        torch.save(self.pi_func.state_dict(), "{}/pi_func.th".format(path))
        torch.save(self.pi_value.state_dict(), "{}/pi_value.th".format(path))
        torch.save(self.dec_func.state_dict(), "{}/decoder_func.th".format(path))

    def load_models(self, path):
        self.pi_func.load_state_dict(torch.load("{}/pi_func.th".format(path), map_location=lambda storage, loc: storage))
        self.pi_value.load_state_dict(torch.load("{}/pi_value.th".format(path), map_location=lambda storage, loc: storage))
        self.dec_func.load_state_dict(torch.load("{}/decoder_func.th".format(path), map_location=lambda storage, loc: storage))

    def sample_option(self, episode_id, init_state):

        task = torch.randn(self.args.code_dim)
        task = task / torch.norm(task)
        task = task.unsqueeze(0)
        task = task.to(self.args.device)
        c = torch.tensor([0], dtype=torch.long, device=self.args.device)

        return c, task  # (1,), (1, code_dim) # fit the outloop

    def sample_option_batch(self, episode_id, init_state):
        env_num = init_state.shape[0]
        task_list = []

        for i in range(env_num):
            task = torch.randn(self.args.code_dim)
            task = task / torch.norm(task)
            task_list.append(task)

        tasks = torch.stack(task_list, dim=0)
        tasks = tasks.to(self.args.device)

        c = torch.tensor([0 for _ in range(env_num)], dtype=torch.long, device=self.args.device)

        return c, tasks # (env,), (env, code_dim)

    def _get_return(self, rwd):
        rwd_array = rwd.squeeze(-1).detach().clone().cpu().numpy() # (bs, traj_len, )
        traj_len = rwd_array.shape[1]

        ret_array = []
        bootstrap = 0
        for i in range(traj_len-1, -1, -1):
            bootstrap = rwd_array[:, i] + self.args.discount * bootstrap
            ret_array.insert(0, bootstrap.copy())

        ret_array = np.transpose(ret_array)

        return (torch.tensor(ret_array, dtype=rwd.dtype, device=rwd.device)).unsqueeze(-1)

    def _get_advantage(self, rewards, v_func, next_v_func, filled):
        traj_len = rewards.shape[1]
        reward_array = rewards.detach().clone().cpu().numpy()
        v_func_array = v_func.detach().clone().cpu().numpy()
        next_v_func_array = next_v_func.detach().clone().cpu().numpy()
        filled_array = filled.detach().clone().cpu().numpy()

        delta_array = reward_array.copy() + self.args.discount * next_v_func_array.copy() - v_func_array.copy()
        delta_array = delta_array * filled_array # filter the fake data

        adv_array = []
        bootstrap = 0
        for i in range(traj_len-1, -1, -1):
            bootstrap = delta_array[:, i] + self.args.discount * self.args.lamda * bootstrap
            adv_array.insert(0, bootstrap.copy())
        adv_array = np.transpose(adv_array, (1, 0, 2))

        return torch.tensor(adv_array, dtype=rewards.dtype, device=rewards.device)

    def _compute_intr_reward(self, task, next_obs):
        # maxent reward
        with torch.no_grad():
            rep = self.dec_func(next_obs, norm=False)
        n_iter = rep.shape[0] // 5000
        if rep.shape[0] % 5000:
            n_iter += 1
        reward_list = []
        for i in range(n_iter):
            reward_list.append(self.pbe(rep[i*5000:(i+1)*5000]))
        reward = torch.cat(reward_list, dim=0)
        # print(reward.shape)
        intr_ent_reward = reward.reshape(-1, 1)

        # successor feature reward
        rep = rep / torch.norm(rep, dim=1, keepdim=True)
        intr_sf_reward = torch.einsum("bi,bi->b", task, rep).reshape(-1, 1)

        return intr_ent_reward + intr_sf_reward

    def train(self, episode_id, train_batch, traj_rwd):
        # set up the data
        option_embs = train_batch.get_item("option_emb").view(-1, self.args.code_dim)
        filled = train_batch.get_item("filled").view(-1, 1) # (bs, traj_len, 1)
        states = train_batch.get_item("state").view(-1, self.args.obs_dim) # (bs, traj_len, obs_dim)
        acts = train_batch.get_item("action") # (bs, traj_len, 1) or (bs, traj_len, act_dim)
        # rewards = train_batch.get_item("reward") # (bs, traj_len, 1)
        dones = train_batch.get_item("done") # (bs, traj_len, 1)
        next_states = train_batch.get_item("next_state").view(-1, self.args.obs_dim) # (bs, traj_len, obs_dim)
        log_info = {}

        # update the decoder
        for _ in tqdm(range(self.args.dec_iters*10)):
            inds = torch.randperm(next_states.size(0))[:512]
            dec_loss = -torch.einsum("bi,bi->b", option_embs[inds], self.dec_func(next_states[inds]))
            dec_loss = (dec_loss.unsqueeze(-1) * filled[inds]).sum() / filled[inds].sum()

            self.dec_func_optim.zero_grad()
            dec_loss.backward()
            self.dec_func_optim.step()

        # for _ in tqdm(range(self.args.dec_iters)):
        #     dec_loss = -torch.einsum("bi,bi->b", option_embs, self.dec_func(next_states))
        #     dec_loss = (dec_loss.unsqueeze(-1) * filled).sum() / filled.sum()
        #
        #     self.dec_func_optim.zero_grad()
        #     dec_loss.backward()
        #     self.dec_func_optim.step()

        log_info['decoder_loss'] = dec_loss.item()

        # get the intrinsic reward
        int_rwd = self._compute_intr_reward(option_embs, next_states) # (bs*traj_len, 1)
        int_rwd = int_rwd * filled
        log_info['objective'] = int_rwd.mean().item()
        int_rwd = int_rwd.view(self.args.traj_num, self.args.traj_length, 1).detach()

        # PPO part: working for both discrete and continuous settings
        pi_input = torch.cat([states, option_embs], dim=1)
        next_pi_input = torch.cat([next_states, option_embs], dim=1)
        if self.args.is_discrete:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length)  # (bs*traj_len, )
        else:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length, self.args.act_dim)  # (bs*traj_len, act_dim)

        ret = self._get_return(int_rwd)  # (bs, traj_len, 1) high variance due to the long horizon
        filled = filled.view(self.args.traj_num, self.args.traj_length, 1)

        ## update the baseline
        for _ in range(self.args.value_iters):
            ret_pre = self.pi_value.forward(pi_input, option_embs)  # (bs*traj_len, 1)
            pi_v_loss = (torch.square(
                ret_pre.reshape(self.args.traj_num, self.args.traj_length, 1) - ret) * filled).sum() / filled.sum()
            # pi_v_loss = F.mse_loss(input=ret_pre, target=ret.reshape(self.args.traj_num * self.args.traj_length, 1))
            self.pi_value_optim.zero_grad()
            pi_v_loss.backward()
            self.pi_value_optim.step()
        log_info['pi_v_loss'] = pi_v_loss.item()

        ## update the policy network
        v_func = self.pi_value.forward(pi_input, option_embs)  # (bs * traj_len, 1)
        v_func = (v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = self.pi_value.forward(next_pi_input, option_embs)  # (bs * traj_len, 1)
        next_v_func = (next_v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = next_v_func * (1.0 - dones)  # (bs * traj_len, 1) # danger

        adv = self._get_advantage(int_rwd, v_func, next_v_func, filled)  # (bs, traj_length, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        _, log_a, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs * traj_len, ) TODO: use the entropy based on the dist
        log_a = log_a.view(self.args.traj_num, self.args.traj_length, 1).detach()  # (bs, traj_len, 1)

        for pi_iter in range(self.args.pi_iters):
            _, log_a_new, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs*traj_len, )
            ratio = torch.exp(log_a_new.view(self.args.traj_num, self.args.traj_length, 1) - log_a)  # log_a_old
            clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            # pi_loss = - (torch.min(ratio * adv, clip_adv) * filled).sum() / filled.sum() # TODO: try not with the filled

            approx_kl = (log_a - log_a_new.view(self.args.traj_num, self.args.traj_length, 1)).mean().item()
            if approx_kl > 1.5 * self.args.target_kl:
                break

            self.pi_func_optim.zero_grad()
            pi_loss.backward()
            self.pi_func_optim.step()

        log_info['pi_loss'] = pi_loss.item()
        log_info['pi_iters'] = pi_iter

        # log to tensorboard
        write_summary(self.writer, info=log_info, step=episode_id)
        write_summary(self.writer, info=traj_rwd, step=episode_id)
        print("Training info: ", log_info)





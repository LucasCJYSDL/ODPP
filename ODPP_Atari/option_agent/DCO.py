import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from learner import policy, value
from utils.summary_tools import write_summary
from spectral_DPP_agent.laprepr import LapReprLearner
from option_agent.hierarchical_policy import get_final_state_value, get_return_array, get_advantage

# Deep Covering Option
class DCO_Agent(object):
    def __init__(self, args):
        self.args = args
        self.args.discount = 0.80  # TODO: fine-tuning
        self.args.code_dim = 1
        ## policy
        if self.args.is_discrete:
            self.pi_func = policy.CategoricalPolicy(input_dim=self.args.obs_dim, hidden_dim=self.args.pi_hidden_dim, action_dim=self.args.act_dim)
        else:
            self.pi_func = policy.GaussianPolicy(input_dim=self.args.obs_dim, hidden_dim=self.args.pi_hidden_dim,
                                                 action_dim=self.args.act_dim, output_activation=F.tanh, act_range=self.args.act_range)  # danger
        self.pi_value = value.ValueFuntion(input_dim=self.args.obs_dim, hidden_dim=self.args.pi_hidden_dim)

        self.pi_func_params = list(self.pi_func.parameters())
        self.pi_value_params = list(self.pi_value.parameters())
        self.pi_func_optim = Adam(params=self.pi_func_params, lr=self.args.lr)
        self.pi_value_optim = Adam(params=self.pi_value_params, lr=self.args.lr)

        log_dir = os.path.join('./option_agent', self.args.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def eval_mode(self):
        self.pi_func.eval()

    def train_mode(self):
        self.pi_func.train()

    def cuda(self):
        self.pi_func.cuda()
        self.pi_value.cuda()

    # dummy function to keep the interface as the same as others, always return option 0
    def sample_option(self, episode_id=None, init_state=None):
        c = torch.tensor(0, dtype=torch.int64, device=self.args.device)
        c_onehot = F.one_hot(c, self.args.code_dim).float()
        return c, c_onehot # (1,), (1, code_dim)

    def sample_action(self, state, option):
        with torch.no_grad():
            state = torch.tensor([state], device=self.args.device, dtype=torch.float32)
            act, _, _ = self.pi_func.forward(state)
        return act # (1, ) or (1, act_dim)
    
    def sample_option_batch(self, episode_id=None, init_state=None):
        env_num = init_state.shape[0]
        c = torch.zeros((env_num, ), dtype=torch.int64, device=self.args.device)
        c_onehot = F.one_hot(c, self.args.code_dim).float()
        return c, c_onehot  # (env_num,), (env_num, code_dim)

    def sample_action_batch(self, state, option):
        with torch.no_grad():
            state = torch.tensor(state, device=self.args.device, dtype=torch.float32)
            act, _, _ = self.pi_func.forward(state)
        return act # (env_num, ) or (env_num, act_dim)

    def save_models(self, path):
        torch.save(self.pi_func.state_dict(), "{}/pi_func.th".format(path))
        torch.save(self.pi_value.state_dict(), "{}/pi_value.th".format(path))

    def load_models(self, path):
        self.pi_func.load_state_dict(torch.load("{}/pi_func.th".format(path), map_location=lambda storage, loc: storage))
        self.pi_value.load_state_dict(torch.load("{}/pi_value.th".format(path), map_location=lambda storage, loc: storage))

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

    def train(self, episode_id, train_batch, lap_repr: LapReprLearner, traj_rwd):
        # set up the data
        filled = train_batch.get_item("filled")  # (bs, traj_len, 1)
        dones = train_batch.get_item("done") # (bs, traj_len, 1)
        horizons = train_batch.get_item("horizon")  # (bs, 1)
        states = train_batch.get_item("state")  # (bs, traj_len, obs_dim)
        acts = train_batch.get_item("action")
        next_states = train_batch.get_item("next_state")  # (bs, traj_len, obs_dim)
        log_info = {}
        # log_info['env_rwd'] = traj_rwd

        # get the reward function
        assert self.args.generalized # since we need the second dimension be the normalized fielder vector
        f_s = lap_repr.get_embedding_matrix(states.reshape(self.args.traj_num * self.args.traj_length, -1).cpu().numpy(), normalized=False)[:, 1:2]
        f_s_prime = lap_repr.get_embedding_matrix(next_states.reshape(self.args.traj_num * self.args.traj_length, -1).cpu().numpy(), normalized=False)[:, 1:2]
        f_s = f_s.view(self.args.traj_num, self.args.traj_length, -1).detach() # (bs, traj_len, 1)
        f_s_prime = f_s_prime.view(self.args.traj_num, self.args.traj_length, -1).detach() # (bs, traj_len, 1)
        rwd = (f_s - f_s_prime) * filled # based on Jinnai's paper
        ret = self._get_return(rwd) # (bs, traj_len, 1)

        # update the value function
        for _ in range(self.args.value_iters):
            ret_pre = self.pi_value.forward(states.reshape(self.args.traj_num * self.args.traj_length, -1))  # (bs*traj_len, 1)
            pi_v_loss = (torch.square(ret_pre.reshape(self.args.traj_num, self.args.traj_length, 1) - ret) * filled).sum() / filled.sum()
            self.pi_value_optim.zero_grad()
            pi_v_loss.backward()
            self.pi_value_optim.step()
        log_info['pi_v_loss'] = pi_v_loss.item()

        # # update the policy network
        v_func = self.pi_value.forward(states.reshape(self.args.traj_num * self.args.traj_length, -1))  # (bs * traj_len, 1)
        v_func = (v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = self.pi_value.forward(next_states.reshape(self.args.traj_num * self.args.traj_length, -1))  # (bs * traj_len, 1)
        next_v_func = (next_v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = next_v_func * (1.0 - dones)  # (bs, traj_len, 1) # danger

        adv = self._get_advantage(rwd, v_func, next_v_func, filled)  # (bs, traj_length, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.args.is_discrete:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length)  # (bs*traj_len, )
        else:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length, self.args.act_dim)  # (bs*traj_len, act_dim)

        _, log_a, _ = self.pi_func.forward(states.reshape(self.args.traj_num * self.args.traj_length, -1), a=pi_func_a)  # (bs * traj_len, ) TODO: use the entropy based on the dist
        log_a = log_a.view(self.args.traj_num, self.args.traj_length, 1).detach()  # (bs, traj_len, 1)

        for pi_iter in range(self.args.pi_iters):
            _, log_a_new, _ = self.pi_func.forward(states.reshape(self.args.traj_num * self.args.traj_length, -1), a=pi_func_a)  # (bs*traj_len, )
            ratio = torch.exp(log_a_new.view(self.args.traj_num, self.args.traj_length, 1) - log_a)  # log_a_old
            clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            # pi_loss = - (torch.min(ratio * adv, clip_adv) * filled).sum() / filled.sum()  # TODO: try not with the filled

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

    def train_with_env_rwd(self, episode_id, train_batch):
        # set up the data
        updated = train_batch.get_item("updated")  # (bs, epi_len, 1)
        filled = train_batch.get_item("filled")  # (bs, epi_len, 1)
        horizons = train_batch.get_item("horizon")  # (bs, 1)
        states = train_batch.get_item("state")  # (bs, epi_len, obs_dim)
        options = train_batch.get_item("option")  # (bs, epi_len, 1)
        option_onehots = train_batch.get_item("option_onehot")  # (bs, epi_len, code_dim)
        acts = train_batch.get_item("action")  # (bs, epi_len, 1) or (bs, epi_len, act_dim)
        rewards = train_batch.get_item("reward")  # (bs, epi_len, 1)
        dones = train_batch.get_item("done")  # (bs, epi_len, 1)
        next_states = train_batch.get_item("next_state")  # (bs, epi_len, obs_dim)
        bs = next_states.shape[0]
        epi_len = next_states.shape[1]
        log_info = {}

        # get target value
        target_v = self.pi_value.forward(next_states.reshape(bs * epi_len, -1))  # (bs * epi_len, 1)
        target_v = (target_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach()  # (bs, epi_len, 1)
        final_v = get_final_state_value(self.args, target_v, horizons)
        # get return value
        ret = get_return_array(self.args, rewards, final_v, horizons, filled)
        log_info['return'] = ret.mean().item()

        # update the value
        for _ in range(self.args.value_iters):
            ret_pre = self.pi_value.forward(states.reshape(bs * epi_len, -1))  # (bs*epi_len, 1)
            pi_v_loss = (torch.square(ret_pre.reshape(bs, epi_len, 1) - ret) * filled).sum() / filled.sum()
            self.pi_value_optim.zero_grad()
            pi_v_loss.backward()
            self.pi_value_optim.step()
        log_info['pi_v_loss'] = pi_v_loss.item()

        # get advantage function
        curr_v = self.pi_value.forward(states.reshape(bs * epi_len, -1))
        curr_v = (curr_v.reshape(bs, epi_len, 1) * filled).detach()
        next_v = self.pi_value.forward(next_states.reshape(bs * epi_len, -1))
        next_v = (next_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach()
        adv = get_advantage(self.args, rewards, curr_v, next_v, filled)  # (bs, epi_len, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.args.is_discrete:
            pi_func_a = acts.reshape(bs * epi_len)  # (bs*epi_len, )
        else:
            pi_func_a = acts.reshape(bs * epi_len, self.args.act_dim)  # (bs*epi_len, act_dim)

        _, log_a, _ = self.pi_func.forward(states.reshape(bs * epi_len, -1), a=pi_func_a)  # (bs * epi_len, )
        log_a = log_a.view(bs, epi_len, 1).detach()  # (bs, epi_len, 1)

        for pi_iter in range(self.args.pi_iters):
            _, log_a_new, _ = self.pi_func.forward(states.reshape(bs * epi_len, -1), a=pi_func_a)  # (bs * epi_len, )
            ratio = torch.exp(log_a_new.view(bs, epi_len, 1) - log_a)  # log_a_old
            clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            # pi_loss = - (torch.min(ratio * adv, clip_adv) * filled).sum() / filled.sum()  # TODO: try not with the filled
            approx_kl = (log_a - log_a_new.view(bs, epi_len, 1)).mean().item()
            if approx_kl > 1.5 * self.args.target_kl:
                break

            self.pi_func_optim.zero_grad()
            pi_loss.backward()
            self.pi_func_optim.step()

        log_info['pi_loss'] = pi_loss.item()
        log_info['pi_iters'] = pi_iter

        # log to tensorboard
        write_summary(self.writer, info=log_info, step=episode_id)
        print("Training info: ", log_info)





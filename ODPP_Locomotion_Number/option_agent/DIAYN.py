import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from learner import policy, value, decoder
from utils.summary_tools import write_summary
from option_agent.base_option_agent import Option_Agent


class DIAYN_Agent(Option_Agent):
    def __init__(self, args):
        super(DIAYN_Agent, self).__init__(args)

        # build the networks
        ## prior
        self.dummy_prior_func = torch.distributions.Categorical(logits=torch.ones(self.args.code_dim))
        ## policy
        if self.args.is_discrete:
            self.pi_func = policy.CategoricalPolicy(input_dim=(self.args.obs_dim + self.args.code_dim), hidden_dim=self.args.pi_hidden_dim,
                                                    action_dim=self.args.act_dim)
        else:
            self.pi_func = policy.GaussianPolicy(input_dim=(self.args.obs_dim + self.args.code_dim), hidden_dim=self.args.pi_hidden_dim,
                                                 action_dim=self.args.act_dim, output_activation=F.tanh, act_range=self.args.act_range)  # danger
        self.pi_value = value.ValueFuntion(input_dim=(self.args.obs_dim + self.args.code_dim), hidden_dim=self.args.pi_hidden_dim)

        self.pi_func_params = list(self.pi_func.parameters())
        self.pi_value_params = list(self.pi_value.parameters())
        self.pi_func_optim = Adam(params=self.pi_func_params, lr=self.args.lr)
        self.pi_value_optim = Adam(params=self.pi_value_params, lr=self.args.lr)
        ## decoder
        self.dec_func = decoder.Decoder(input_dim=self.args.obs_dim, hidden_dim=self.args.dec_hidden_dim,
                                        code_dim=self.args.code_dim)  # take s_(t+1) as input

        self.dec_func_params = list(self.dec_func.parameters())
        self.dec_func_optim = Adam(params=self.dec_func_params, lr=self.args.lr)

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

    def _get_return(self, rwd): # the rwd for the last state in next_states is 0, since we can't calculate its reward, and there is not a performance measure for the whole trajectory
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

    # DIAYN needs more training iterations since the the size of the training samples is much larger
    def train(self, episode_id, train_batch, traj_rwd):
        # set up the data
        options = (train_batch.get_item("option")).unsqueeze(1).repeat(1, self.args.traj_length, 1) # (bs, 1) -> (bs, traj_len, 1)
        option_onehots = (train_batch.get_item("option_onehot")).unsqueeze(1).repeat(1, self.args.traj_length, 1) # (bs, code_dim) -> (bs, traj_len, code_dim)
        filled = train_batch.get_item("filled") # (bs, traj_len, 1)
        states = train_batch.get_item("state") # (bs, traj_len, obs_dim)
        acts = train_batch.get_item("action") # (bs, traj_len, 1) or (bs, traj_len, act_dim)
        rewards = train_batch.get_item("reward") # (bs, traj_len, 1)
        dones = train_batch.get_item("done") # (bs, traj_len, 1)
        next_states = train_batch.get_item("next_state") # (bs, traj_len, obs_dim)
        log_info = {}
        # log_info['env_rwd'] = traj_rwd

        # update the decoder
        for _ in range(self.args.dec_iters):
            _, log_gt, _ = self.dec_func.forward(next_states.reshape(self.args.traj_num * self.args.traj_length, -1),
                                                 gt=options.reshape(self.args.traj_num * self.args.traj_length)) # (bs * traj_num, )
            dec_loss = - (log_gt.reshape(self.args.traj_num, self.args.traj_length, 1) * filled).sum() / filled.sum()
            # dec_loss = F.cross_entropy(self.dec_func.logits, options.reshape(self.args.traj_num * self.args.traj_length))
            self.dec_func_optim.zero_grad()
            dec_loss.backward()
            self.dec_func_optim.step()
        log_info['decoder_loss'] = dec_loss.item()

        # get the intrinsic reward
        _, int_rwd, _ = self.dec_func.forward(next_states.reshape(self.args.traj_num * self.args.traj_length, -1),
                                              gt=options.reshape(self.args.traj_num * self.args.traj_length)) # (bs * traj_len, )
        int_rwd = int_rwd.view(self.args.traj_num, self.args.traj_length, 1).detach() # (bs, traj_len, 1)
        opt_ent_rwd = self.dummy_prior_func.log_prob(options.to('cpu')) # (bs, traj_len, 1)
        opt_ent_rwd = opt_ent_rwd.to(int_rwd.device)

        pi_input = torch.cat([states.reshape(self.args.traj_num * self.args.traj_length, -1), option_onehots.reshape(self.args.traj_num * self.args.traj_length, -1)], dim=1)
        next_pi_input = torch.cat([next_states.reshape(self.args.traj_num * self.args.traj_length, -1), option_onehots.reshape(self.args.traj_num * self.args.traj_length, -1)], dim=1)
        if self.args.is_discrete:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length) # (bs*traj_len, )
        else:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length, self.args.act_dim) # (bs*traj_len, act_dim)

        _, act_ent_rwd, _ = self.pi_func.forward(pi_input, a=pi_func_a) # (bs * traj_len, ) TODO: use the entropy based on the dist
        act_ent_rwd = act_ent_rwd.view(self.args.traj_num, self.args.traj_length, 1).detach() # (bs, traj_len, 1)
        log_info['reconstruction'] = int_rwd.mean().item()
        log_info['pi_entropy'] = act_ent_rwd.mean().item()

        # TODO: ignore the act_ent_rwd, or add a weight term 'beta'; add a weight term between the intrinsic and extrinsic term
        int_rwd = (rewards + int_rwd - self.args.beta * (opt_ent_rwd + act_ent_rwd)) * filled
        log_info['objective'] = int_rwd.mean().item()
        # get the return with the discount factor
        ret = self._get_return(int_rwd) # (bs, traj_len, 1) high variance due to the long horizon


        # update the baseline
        for _ in range(self.args.value_iters):
            ret_pre = self.pi_value.forward(pi_input) # (bs*traj_len, 1)
            pi_v_loss = (torch.square(ret_pre.reshape(self.args.traj_num, self.args.traj_length, 1) - ret) * filled).sum() / filled.sum()
            # pi_v_loss = F.mse_loss(input=ret_pre, target=ret.reshape(self.args.traj_num * self.args.traj_length, 1))
            self.pi_value_optim.zero_grad()
            pi_v_loss.backward()
            self.pi_value_optim.step()
        log_info['pi_v_loss'] = pi_v_loss.item()

        # # update the policy network
        v_func = self.pi_value.forward(pi_input)  # (bs * traj_len, 1)
        v_func = (v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = self.pi_value.forward(next_pi_input)  # (bs * traj_len, 1)
        next_v_func = (next_v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = next_v_func * (1.0 - dones) # (bs * traj_len, 1) # danger

        adv = self._get_advantage(int_rwd, v_func, next_v_func, filled)  # (bs, traj_length, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        _, log_a, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs * traj_len, ) TODO: use the entropy based on the dist
        log_a = log_a.view(self.args.traj_num, self.args.traj_length, 1).detach()  # (bs, traj_len, 1)

        for pi_iter in range(self.args.pi_iters):
            _, log_a_new, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs*traj_len, )
            ratio = torch.exp(log_a_new.view(self.args.traj_num, self.args.traj_length, 1) - log_a) # log_a_old
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





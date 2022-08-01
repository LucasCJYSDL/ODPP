import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from learner import policy, value, rnn_decoder
from utils.summary_tools import write_summary
from option_agent.base_option_agent import Option_Agent

class VALOR_Agent(Option_Agent):
    def __init__(self, args):
        super(VALOR_Agent, self).__init__(args)
        # build the networks
        ## prior
        self.dummy_prior_func = torch.distributions.Categorical(logits=torch.ones(self.args.code_dim))
        ## policy
        if self.args.is_discrete:
            self.pi_func = policy.CategoricalPolicy(input_dim=(self.args.obs_dim + self.args.code_dim), hidden_dim=self.args.pi_hidden_dim,
                                                    action_dim=self.args.act_dim)
        else:
            self.pi_func = policy.GaussianPolicy(input_dim=(self.args.obs_dim + self.args.code_dim), hidden_dim=self.args.pi_hidden_dim,
                                                 action_dim=self.args.act_dim, output_activation=F.tanh, act_range=args.act_range)  # danger
        self.pi_value = value.ValueFuntion(input_dim=(self.args.obs_dim + self.args.code_dim), hidden_dim=self.args.pi_hidden_dim)

        self.pi_func_params = list(self.pi_func.parameters())
        self.pi_value_params = list(self.pi_value.parameters())
        self.pi_func_optim = Adam(params=self.pi_func_params, lr=self.args.lr)
        self.pi_value_optim = Adam(params=self.pi_value_params, lr=self.args.lr)
        ## decoder
        self.dec_func = rnn_decoder.RNN_Decoder(input_dim=self.args.obs_dim, hidden_dim=self.args.rnn_dec_hidden_dim, code_dim=self.args.code_dim) # take traj as input
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

    def _get_landmarks(self, horizons, init_s, next_states):
        horizon_array = horizons.detach().clone().cpu().numpy()
        init_s_array = init_s.detach().clone().cpu().numpy()
        next_state_array = next_states.detach().clone().cpu().numpy()

        landmark_array = []
        batch_size = horizons.shape[0]
        for i in range(batch_size):
            horizon = horizon_array[i][0]
            interval = (horizon + 1) // self.args.landmark_num
            landmarks = []
            for j in range(self.args.landmark_num - 1):
                left_idx = int((horizon - 1) - (j + 1) * interval)
                right_idx = int((horizon - 1) - j * interval)
                assert left_idx >= 0 and right_idx >= 0
                landmarks.insert(0, next_state_array[i][right_idx] - next_state_array[i][left_idx])
            landmarks.insert(0, next_state_array[i][left_idx] - init_s_array[i])
            landmark_array.append(landmarks)

        return torch.tensor(landmark_array, dtype=next_states.dtype, device=next_states.device)

    def _get_return_array(self, rewards, ret, horizons, filled):
        batch_size = rewards.shape[0]
        traj_len = rewards.shape[1]

        reward_array = rewards.detach().clone().cpu().numpy()
        ret_array = ret.detach().clone().cpu().numpy()
        horizon_array = horizons.detach().clone().cpu().numpy()

        for i in range(batch_size):
            horizon = horizon_array[i][0]
            reward_array[i][horizon - 1][0] += ret_array[i][0]  # update the return as part of the reward of the last step (states)

        return_array = []
        bootstrap = 0
        for i in range(traj_len - 1, -1, -1):
            bootstrap = reward_array[:, i] + self.args.discount * bootstrap
            return_array.insert(0, bootstrap.copy())
        return_array = np.transpose(return_array, (1, 0, 2))
        return_tensor = torch.tensor(return_array, dtype=rewards.dtype, device=rewards.device)
        reward_tensor = torch.tensor(reward_array, dtype=rewards.dtype, device=rewards.device)

        return return_tensor * filled, reward_tensor * filled  # filter the rewards that is not corresponding to the steps in states

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

    def train(self, episode_id, train_batch):
        # set up the data
        options = train_batch.get_item("option") # (bs, 1)
        option_onehots = train_batch.get_item("option_onehot") # (bs, code_dim)
        filled = train_batch.get_item("filled") # (bs, traj_len, 1)
        dones = train_batch.get_item("done") # (bs, traj_len, 1)
        horizons = train_batch.get_item("horizon") # (bs, 1)
        states = train_batch.get_item("state") # (bs, traj_len, obs_dim)
        acts = train_batch.get_item("action")
        rewards = train_batch.get_item("reward") # (bs, traj_len, 1)
        next_states = train_batch.get_item("next_state") # (bs, traj_len, obs_dim)
        init_s = states[:, 0] # (bs, obs_dim)
        landmarks = self._get_landmarks(horizons, init_s, next_states) # (bs, landmark_num, obs_dim)
        log_info = {}

        # update the decoder
        for _ in range(self.args.dec_iters):
            _, log_gt, _ = self.dec_func.forward(landmarks, gt=options.squeeze(-1)) # (bs, )
            # dec_loss = - log_gt.mean()
            dec_loss = F.cross_entropy(self.dec_func.logits, options.squeeze(-1))
            self.dec_func_optim.zero_grad()
            dec_loss.backward()
            self.dec_func_optim.step()
        log_info['decoder_loss'] = dec_loss.item()

        # update the value function
        _, ret, _ = self.dec_func.forward(landmarks, gt=options.squeeze(-1)) # (bs, )
        pi_input = torch.cat([states, option_onehots.unsqueeze(1).repeat(1, self.args.traj_length, 1)], dim=-1) \
            .reshape(self.args.traj_num * self.args.traj_length, -1)  # (bs * traj_len, obs_dim+code_dim)
        next_pi_input = torch.cat([next_states, option_onehots.unsqueeze(1).repeat(1, self.args.traj_length, 1)], dim=-1) \
            .reshape(self.args.traj_num * self.args.traj_length, -1)  # (bs * traj_len, obs_dim+code_dim)

        if self.args.is_discrete:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length) # (bs*traj_len, )
        else:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length, self.args.act_dim) # (bs*traj_len, act_dim)

        _, log_a, _ = self.pi_func.forward(pi_input, a=pi_func_a) # (bs*traj_len, )
        log_a = log_a.view(self.args.traj_num, self.args.traj_length, 1).detach()
        traj_ent = (log_a * filled).sum(dim=1) # (bs, 1) TODO: use the entropy based on the dist
        ret = (ret.unsqueeze(-1) - self.args.beta * 0.001 * traj_ent).detach() # (bs, 1), adopted as episodic reward
        log_info['objective'] = ret.mean().item()
        ret, rewards = self._get_return_array(rewards, ret, horizons, filled) # (bs, traj_length, 1)

        # TODO: exchange the order of the training of pi and value
        for _ in range(self.args.value_iters):
            ret_pre = self.pi_value.forward(pi_input) # (bs * traj_len, 1)
            pi_v_loss = (torch.square(ret_pre.reshape(self.args.traj_num, self.args.traj_length, 1) - ret) * filled).sum() / filled.sum()
            # pi_v_loss = F.mse_loss(input=ret_pre, target=ret.reshape(self.args.traj_num * self.args.traj_length, 1))
            self.pi_value_optim.zero_grad()
            pi_v_loss.backward()
            self.pi_value_optim.step()
        log_info['pi_v_loss'] = pi_v_loss.item()

        # # update the policy
        v_func = self.pi_value.forward(pi_input) # (bs * traj_len, 1)
        v_func = (v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = self.pi_value.forward(next_pi_input) # (bs * traj_len, 1)
        next_v_func = (next_v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = next_v_func * (1.0 - dones)  # (bs * traj_len, 1) # danger
        adv = self._get_advantage(rewards, v_func, next_v_func, filled) # (bs, traj_length, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for pi_iter in range(self.args.pi_iters):
            _, log_a_new, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs*traj_len, )
            ratio = torch.exp(log_a_new.view(self.args.traj_num, self.args.traj_length, 1) - log_a)
            clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            # pi_loss = -(torch.min(ratio * adv, clip_adv) * filled).sum() / filled.sum() # TODO: try not with the filled

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
        print("Training info: ", log_info)

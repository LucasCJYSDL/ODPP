import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from learner import policy, value, rnn_decoder, prior
from utils.summary_tools import write_summary
from spectral_DPP_agent.laprepr import LapReprLearner
from spectral_DPP_agent.dpp import get_kernel_matrix, get_posterior_prob, get_expected_sampling_length, map_inference
from option_agent.base_option_agent import Option_Agent

class ODPP_Agent(Option_Agent):
    def __init__(self, args):
        super(ODPP_Agent, self).__init__(args)
        # build the networks
        ## prior
        self.dummy_prior_func = torch.distributions.Categorical(logits=torch.ones(self.args.code_dim))
        self.prior_func = prior.Prior(input_dim=self.args.obs_dim, hidden_dim=self.args.prior_hidden_dim, code_dim=self.args.code_dim+1)
        self.prior_value = value.ValueFuntion(input_dim=self.args.obs_dim, hidden_dim=self.args.prior_hidden_dim)

        self.prior_func_params = list(self.prior_func.parameters())
        self.prior_value_params = list(self.prior_value.parameters())
        self.prior_func_optim = Adam(params=self.prior_func_params, lr=self.args.lr)
        self.prior_value_optim = Adam(params=self.prior_value_params, lr=self.args.lr)
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
        self.dec_func = rnn_decoder.RNN_Decoder(input_dim=self.args.obs_dim, hidden_dim=self.args.rnn_dec_hidden_dim,
                                                code_dim=self.args.code_dim)  # take traj as input
        self.dec_func_params = list(self.dec_func.parameters())
        self.dec_func_optim = Adam(params=self.dec_func_params, lr=self.args.lr)

    def eval_mode(self):
        self.prior_func.eval()
        self.pi_func.eval()

    def train_mode(self):
        self.prior_func.train()
        self.pi_func.train()

    def cuda(self):
        self.prior_func.cuda()
        self.prior_value.cuda()
        self.pi_func.cuda()
        self.pi_value.cuda()
        self.dec_func.cuda()

    def save_models(self, path):
        torch.save(self.prior_func.state_dict(), "{}/prior_func.th".format(path))
        torch.save(self.prior_value.state_dict(), "{}/prior_value.th".format(path))
        torch.save(self.pi_func.state_dict(), "{}/pi_func.th".format(path))
        torch.save(self.pi_value.state_dict(), "{}/pi_value.th".format(path))
        torch.save(self.dec_func.state_dict(), "{}/decoder_func.th".format(path))

    def load_models(self, path):
        self.prior_func.load_state_dict(torch.load("{}/prior_func.th".format(path), map_location=lambda storage, loc: storage))
        self.prior_value.load_state_dict(torch.load("{}/prior_value.th".format(path), map_location=lambda storage, loc: storage))
        self.pi_func.load_state_dict(torch.load("{}/pi_func.th".format(path), map_location=lambda storage, loc: storage))
        self.pi_value.load_state_dict(torch.load("{}/pi_value.th".format(path), map_location=lambda storage, loc: storage))
        self.dec_func.load_state_dict(torch.load("{}/decoder_func.th".format(path), map_location=lambda storage, loc: storage))

    def _get_landmark_idx(self, map_samples, horizon):
        assert len(map_samples) <= self.args.landmark_num
        left_num = self.args.landmark_num - len(map_samples)
        if left_num == 0:
            return np.sort(map_samples)
        # get uniform landmarks
        uni_landmarks = []
        interval = (horizon + 1) // self.args.landmark_num
        for j in range(self.args.landmark_num):
            uni_landmarks.insert(0, int(horizon - j * interval)) # index in trajectory rather than next_states
        # get a padding vector to denote which interval is occupied
        padding = [0 for _ in range(self.args.landmark_num + 1)]
        for idx in map_samples: # indexed from 0
            padding_idx = idx // interval
            padding[padding_idx] = 1

        for i in range(self.args.landmark_num, -1, -1):
            if padding[i] == 0:
                assert i >= 1
                map_samples.append(uni_landmarks[i-1])
                left_num -= 1
                if left_num == 0:
                    return np.sort(map_samples)

    def _get_landmarks(self, horizons, init_s, next_states, lap_repr: LapReprLearner, option_onehots):

        #traj_data = torch.cat([init_s, next_states], dim=1) # (bs, traj_len+1, obs_dim)
        #traj_input = torch.cat([traj_data, option_onehots.unsqueeze(1).repeat(1, self.args.traj_length + 1, 1)], dim=-1)
        #self.pi_func.forward(traj_input)
        #traj_feature = self.pi_func.mu.feature.cpu().numpy() # (bs, traj_len+1, hidden_dim)

        horizon_array = horizons.detach().clone().cpu().numpy()
        init_s_array = init_s.detach().clone().cpu().numpy()
        next_state_array = next_states.detach().clone().cpu().numpy()

        traj_feature_list = []
        traj_kernel_matrices = []
        posterior_probs = []
        map_lengths = []
        landmark_array = []
        batch_size = horizons.shape[0]
        for i in range(batch_size):
            horizon = horizon_array[i][0]
            # get the whole trajactory and kernel matrix
            trajectory = np.concatenate([init_s_array[i], next_state_array[i, :horizon]], axis=0) # (horizon+1, obs_dim)
            B = lap_repr.get_embedding_matrix(trajectory.copy()) # (horizon+1, D), numpy.ndarray
            #B = traj_feature[i][:(horizon+1)] # (horizon+1, D), numpy.ndarray
            #embedding_norm = np.linalg.norm(B, axis=1)
            #B = B / embedding_norm.reshape(B.shape[0], 1)

            L = get_kernel_matrix(B) # (horizon+1, horizon+1)
            traj_kernel_matrices.append(L.copy())
            # MAP inference based on DPP
            samples = map_inference(L, max_length=self.args.landmark_num, epsilon=self.args.dpp_epsilon)
            map_lengths.append(len(samples))
            posterior_probs.append(get_posterior_prob(L.copy(), np.array(samples)))
            landmark_idx = self._get_landmark_idx(samples, horizon)
            # TODO: use the sum of the whole traj
            traj_feature_list.append(np.sum(B[np.array(landmark_idx)], axis=0))

            assert len(landmark_idx) == self.args.landmark_num
            # collect landmarks based on the indexes, TODO: use 's_i' rather than 's_(i+1) - s_i'
            landmarks = []
            landmarks.append(trajectory[landmark_idx[0]] - trajectory[0])
            for j in range(self.args.landmark_num - 1):
                landmarks.append(trajectory[landmark_idx[j+1]] - trajectory[landmark_idx[j]])
            landmark_array.append(landmarks)

        return torch.tensor(landmark_array, dtype=next_states.dtype, device=next_states.device), traj_kernel_matrices, traj_feature_list, \
               torch.tensor(posterior_probs, dtype=next_states.dtype, device=next_states.device), map_lengths # (bs, landmark_num, obs_dim), (bs, )

    def _get_expected_length(self, L_list):
        batch_size = len(L_list)
        exp_len_list = []
        for i in range(batch_size):
            exp_len_list.append(get_expected_sampling_length(L_list[i]))

        return torch.tensor(np.array(exp_len_list), dtype=torch.float32, device=self.args.device)

    def _get_traj_feature(self, traj_fea_list):
        if not self.args.dpp_traj_feature:
            traj_feature = self.dec_func.inter_states.detach().clone() # (bs, landmark_num, 2 * hidden_dim), note 2 * hidden_dim should <= bs
            traj_feature = traj_feature.sum(dim=1).cpu().numpy() # (bs, 2 * hidden_dim)
        else:
            traj_feature = np.array(traj_fea_list) # (bs, D)

        return traj_feature

    def _get_traj_batch_dpp_measure(self, traj_feature): # (bs, D)
        batch_size = traj_feature.shape[0]
        g_tau_batch = np.zeros(batch_size, dtype=np.float32)
        h_tau_batch = np.zeros(batch_size, dtype=np.float32)
        # collect g
        g_batch_size = self.args.batch_traj
        assert batch_size % g_batch_size == 0
        g_slice_num = batch_size // g_batch_size
        for i in range(g_slice_num):
            temp_B = traj_feature[(i*g_batch_size):((i+1)*g_batch_size)] # (g_bs, D)
            temp_L = get_kernel_matrix(temp_B) # (g_bs, g_bs) danger, since D may be smaller than g_bs
            temp_exp_len = get_expected_sampling_length(temp_L)
            g_tau_batch[(i*g_batch_size):((i+1)*g_batch_size)] = temp_exp_len
        # collect h
        h_batch_size = self.args.batch_c * self.args.batch_traj
        assert batch_size % h_batch_size == 0
        h_slice_num = batch_size // h_batch_size
        for i in range(h_slice_num):
            temp_B = traj_feature[(i*h_batch_size):((i+1)*h_batch_size)] # (h_bs, D)
            temp_L = get_kernel_matrix(temp_B) # (h_bs, h_bs) danger, since D may be smaller than h_bs
            temp_exp_len = get_expected_sampling_length(temp_L)
            h_tau_batch[(i*h_batch_size):((i+1)*h_batch_size)] = temp_exp_len

        return torch.tensor(g_tau_batch.reshape((batch_size, 1)), dtype=torch.float32, device=self.args.device), \
               torch.tensor(h_tau_batch.reshape((batch_size, 1)), dtype=torch.float32, device=self.args.device)

    def _get_return_array(self, rewards, ret, horizons, filled):
        batch_size = rewards.shape[0]
        traj_len = rewards.shape[1]

        reward_array = rewards.detach().clone().cpu().numpy()
        ret_array = ret.detach().clone().cpu().numpy()
        horizon_array = horizons.detach().clone().cpu().numpy()

        for i in range(batch_size):
            horizon = horizon_array[i][0]
            reward_array[i][horizon-1][0] += ret_array[i][0] # update the return as part of the reward of the last step (states)

        return_array = []
        bootstrap = 0
        for i in range(traj_len-1, -1, -1):
            bootstrap = reward_array[:, i] + self.args.discount * bootstrap
            return_array.insert(0, bootstrap.copy())
        return_array = np.transpose(return_array, (1, 0, 2))
        return_tensor = torch.tensor(return_array, dtype=rewards.dtype, device=rewards.device)
        reward_tensor = torch.tensor(reward_array, dtype=rewards.dtype, device=rewards.device)

        return return_tensor * filled, reward_tensor * filled # filter the rewards that is not corresponding to the steps in states

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
        options = train_batch.get_item("option")  # (bs, 1)
        option_onehots = train_batch.get_item("option_onehot")  # (bs, code_dim)
        filled = train_batch.get_item("filled")  # (bs, traj_len, 1)
        dones = train_batch.get_item("done") # (bs, traj_len, 1)
        horizons = train_batch.get_item("horizon")  # (bs, 1)
        states = train_batch.get_item("state")  # (bs, traj_len, obs_dim)
        acts = train_batch.get_item("action")
        rewards = train_batch.get_item("reward")  # (bs, traj_len, 1)
        next_states = train_batch.get_item("next_state")  # (bs, traj_len, obs_dim)
        init_s = states[:, 0:1]  # (bs, 1, obs_dim)
        landmarks, L_list, traj_fea_list, posterior_probs, map_lengths = self._get_landmarks(horizons, init_s, next_states, lap_repr, option_onehots)  # (bs, landmark_num, obs_dim)
        log_info = {}
        log_info['map_length'] = np.mean(map_lengths)
        log_info['poster_prob'] = posterior_probs.mean().item()
        # log_info['env_rwd'] = traj_rwd

        # update the decoder
        for _ in range(self.args.dec_iters):
            _, log_gt, _ = self.dec_func.forward(landmarks, gt=options.squeeze(-1))  # (bs, )
            dec_loss = - (log_gt * posterior_probs).mean()
            # dec_loss = F.cross_entropy(self.dec_func.logits, options.squeeze(-1))
            self.dec_func_optim.zero_grad()
            dec_loss.backward()
            self.dec_func_optim.step()
        log_info['decoder_loss'] = dec_loss.item()

        # preparation
        _, dec_rwd, _ = self.dec_func.forward(landmarks, gt=options.squeeze(-1))  # (bs, )
        log_info['rec_rwd'] = dec_rwd.mean().item()
        dec_rwd = (dec_rwd * posterior_probs).unsqueeze(-1).detach() # (bs, 1)
        log_info['rec_rwd_posterior'] = dec_rwd.mean().item()
        f_tau = self._get_expected_length(L_list)  # (bs, )
        log_info['exp_len'] = f_tau.mean().item()
        traj_feature = self._get_traj_feature(traj_fea_list)
        g_tau_batch, h_tau_batch = self._get_traj_batch_dpp_measure(traj_feature) # (bs, 1), (bs, 1)
        log_info['g_tau'] = g_tau_batch.mean().item()
        log_info['h_tau'] = h_tau_batch.mean().item()

        pi_input = torch.cat([states, option_onehots.unsqueeze(1).repeat(1, self.args.traj_length, 1)], dim=-1) \
            .reshape(self.args.traj_num * self.args.traj_length, -1)  # (bs * traj_len, obs_dim+code_dim)
        next_pi_input = torch.cat([next_states, option_onehots.unsqueeze(1).repeat(1, self.args.traj_length, 1)], dim=-1) \
            .reshape(self.args.traj_num * self.args.traj_length, -1)  # (bs * traj_len, obs_dim+code_dim) # TODO: use samples of options as next_option_onehots
        if self.args.is_discrete:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length)  # (bs*traj_len, )
        else:
            pi_func_a = acts.reshape(self.args.traj_num * self.args.traj_length, self.args.act_dim)  # (bs*traj_len, act_dim)

        # update the policy network
        _, log_a, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs*traj_len, )
        traj_ent = (log_a.view(self.args.traj_num, self.args.traj_length, 1) * filled).sum(dim=1)
        pi_ret = self.args.dec_w * dec_rwd - self.args.beta * traj_ent.detach() + self.args.alpha_1 * f_tau.unsqueeze(-1).detach() \
                 - self.args.alpha_2 * g_tau_batch + self.args.alpha_3 * h_tau_batch # (bs, 1)
        log_info['pi_objective'] = pi_ret.mean().item()
        pi_ret, rewards = self._get_return_array(rewards, pi_ret, horizons, filled) # (bs, traj_len, 1)


        for _ in range(self.args.value_iters):
            ret_pre = self.pi_value.forward(pi_input)  # (bs * traj_len, 1)
            pi_v_loss = (torch.square(ret_pre.view(self.args.traj_num, self.args.traj_length, -1) - pi_ret) * filled).sum() / filled.sum()
            self.pi_value_optim.zero_grad()
            pi_v_loss.backward()
            self.pi_value_optim.step()
        log_info['pi_v_loss'] = pi_v_loss.item()

        v_func = self.pi_value.forward(pi_input)  # (bs * traj_len, 1)
        v_func = (v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = self.pi_value.forward(next_pi_input)  # (bs * traj_len, 1)
        next_v_func = (next_v_func.detach()).reshape(self.args.traj_num, self.args.traj_length, 1)
        next_v_func = next_v_func * (1.0 - dones)  # (bs, traj_length, 1)

        adv = self._get_advantage(rewards, v_func, next_v_func, filled)  # (bs, traj_length, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        _, log_a, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs*traj_len, )
        log_a = log_a.view(self.args.traj_num, self.args.traj_length, 1).detach()  # (bs, traj_len, 1)
        for pi_iter in range(self.args.pi_iters):
            _, log_a_new, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs*traj_len, )
            ratio = torch.exp(log_a_new.view(self.args.traj_num, self.args.traj_length, 1) - log_a)
            clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            # pi_loss = -(torch.min(ratio * adv, clip_adv) * filled).sum() / filled.sum()  # TODO: try not with the filled

            approx_kl = (log_a - log_a_new.view(self.args.traj_num, self.args.traj_length, 1)).mean().item()
            if approx_kl > 1.5 * self.args.target_kl:
                break

            self.pi_func_optim.zero_grad()
            pi_loss.backward()
            self.pi_func_optim.step()

        log_info['pi_loss'] = pi_loss.item()
        log_info['pi_iters'] = pi_iter

        # update the prior
        if episode_id >= self.args.keep_prior_iters:
            _, prior_ent, _ = self.prior_func.forward(init_s.squeeze(1), code_gt=options.squeeze(-1))  # (bs, )

            _, log_a, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs*traj_len, )
            traj_ent = (log_a.view(self.args.traj_num, self.args.traj_length, 1) * filled).sum(dim=1)  # (bs, 1) TODO: use the entropy based on the dist

            traj_ret = self.args.dec_w * dec_rwd - self.args.beta * traj_ent.detach() + self.args.alpha_1 * f_tau.unsqueeze(-1).detach() # (bs, 1)
            prior_ret = - prior_ent.unsqueeze(-1).detach() + traj_ret - self.args.alpha_2 * g_tau_batch + self.args.alpha_3 * h_tau_batch # (bs, 1)
            log_info['prior_return'] = prior_ret.mean().item()
            log_info['prior_ent'] = prior_ent.mean().item()

            ## update the baseline for the prior network
            for _ in range(self.args.value_iters):
                prior_ret_pre = self.prior_value.forward(init_s.squeeze(1))  # (bs, 1)
                prior_v_loss = F.mse_loss(input=prior_ret_pre, target=prior_ret)
                self.prior_value_optim.zero_grad()
                prior_v_loss.backward()
                self.prior_value_optim.step()
            log_info['prior_v_loss'] = prior_v_loss.item()

            ## update the prior network
            prior_base = self.prior_value.forward(init_s.squeeze(1))  # (bs, 1)
            prior_adv = (prior_ret - prior_base).detach()  # (bs, 1)
            if self.args.normalized:
                prior_adv = (prior_adv - prior_adv.mean()) / (prior_adv.std() + 1e-8)

            _, log_c, _ = self.prior_func.forward(init_s.squeeze(1), code_gt=options.squeeze(-1))  # (bs, )
            log_c = log_c.view(self.args.traj_num, 1).detach()

            for pi_iter in range(self.args.pi_iters):
                _, log_c_new, _ = self.prior_func.forward(init_s.squeeze(1), code_gt=options.squeeze(-1))  # (bs, )
                ratio = torch.exp(log_c_new.view(self.args.traj_num, 1) - log_c)
                clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * prior_adv
                prior_loss = -(torch.min(ratio * prior_adv, clip_adv)).mean()

                approx_kl = (log_c - log_c_new.view(self.args.traj_num, 1)).mean().item()
                if approx_kl > 1.5 * self.args.target_kl:
                    break

                self.prior_func_optim.zero_grad()
                prior_loss.backward()
                self.prior_func_optim.step()

            log_info['prior_loss'] = prior_loss.item()
            log_info['prior_pi_iters'] = pi_iter

        # log to tensorboard
        write_summary(self.writer, info=log_info, step=episode_id)
        write_summary(self.writer, info=traj_rwd, step=episode_id)
        print("Training info: ", log_info)














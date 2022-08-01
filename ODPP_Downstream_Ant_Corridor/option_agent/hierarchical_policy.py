import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from learner import prior, value, policy
from utils.summary_tools import write_summary


def get_final_state_value(args, target_v, horizons):
    target_v_array = target_v.cpu().numpy()
    horizon_array = horizons.cpu().numpy()
    bs = horizon_array.shape[0]
    final_v_array = []
    for i in range(bs):
        horizon = horizon_array[i][0]
        final_v_array.append(target_v_array[i][horizon - 1])

    return torch.tensor(np.array(final_v_array), dtype=torch.float32, device=args.device)  # (bs, 1)


def get_return_array(args, rewards, final_v, horizons, filled, durations=None):
    bs = rewards.shape[0]
    epi_len = rewards.shape[1]

    reward_array = rewards.cpu().numpy()
    final_v_array = final_v.cpu().numpy()
    horizon_array = horizons.cpu().numpy()
    if durations is not None:
        duration_array = durations.cpu().numpy()

    new_reward_array = np.zeros(shape=[bs, epi_len + 1, 1], dtype=np.float32)
    new_reward_array[:, :epi_len, :] = reward_array.copy()
    for i in range(bs):
        horizon = horizon_array[i][0]
        new_reward_array[i][horizon][0] = final_v_array[i][0]  # update the return as the reward of the last step (next_states)

    return_array = []
    bootstrap = 0
    for i in range(epi_len, -1, -1):
        if durations is None or i == epi_len:
            bootstrap = new_reward_array[:, i] + args.discount * bootstrap
        else:
            duration = np.power(args.discount, duration_array[:, i])  # (bs, 1)
            bootstrap = new_reward_array[:, i] + duration * bootstrap  # (bs, 1)

        return_array.insert(0, bootstrap.copy())
    return_array = np.transpose(return_array, (1, 0, 2))
    return_tensor = torch.tensor(return_array, dtype=rewards.dtype, device=rewards.device)

    return return_tensor[:, :-1] * filled  # filter the rewards that is not corresponding to the steps in states, (bs, epi_len, 1)


def get_advantage(args, rewards, v_func, next_v_func, filled, durations=None):
    epi_len = rewards.shape[1]
    reward_array = rewards.cpu().numpy()
    v_func_array = v_func.cpu().numpy()
    next_v_func_array = next_v_func.cpu().numpy()
    filled_array = filled.cpu().numpy()
    if durations is not None:
        duration_array = durations.cpu().numpy()

    if durations is None:
        delta_array = reward_array.copy() + args.discount * next_v_func_array.copy() - v_func_array.copy()
    else:
        delta_array = reward_array.copy() + np.power(args.discount, duration_array) * next_v_func_array.copy() - v_func_array.copy()  # (bs, epi_len, 1)

    delta_array = delta_array * filled_array  # filter the fake data

    adv_array = []
    bootstrap = 0
    for i in range(epi_len - 1, -1, -1):
        if durations is None:
            bootstrap = delta_array[:, i] + args.discount * args.lamda * bootstrap
        else:
            bootstrap = delta_array[:, i] + np.power((args.discount * args.lamda), duration_array[:, i]) * bootstrap  # (bs, 1)

        adv_array.insert(0, bootstrap.copy())
    adv_array = np.transpose(adv_array, (1, 0, 2))

    return torch.tensor(adv_array, dtype=rewards.dtype, device=rewards.device)


class HierPolicy(object):
    def __init__(self, args):
        torch.manual_seed(args.seed + 1000)
        torch.cuda.manual_seed_all(args.seed + 1000)
        self.args = args
        # the high_level policy is to select options given a state, like the prior
        self.prior_func = prior.Prior(input_dim=self.args.obs_dim, hidden_dim=self.args.prior_hidden_dim, code_dim=self.args.code_dim, is_high=True)
        self.prior_value = value.ValueFuntion(input_dim=self.args.obs_dim, hidden_dim=self.args.prior_hidden_dim)

        self.prior_func_params = list(self.prior_func.parameters())
        self.prior_value_params = list(self.prior_value.parameters())
        self.prior_func_optim = Adam(params=self.prior_func_params, lr=self.args.lr)
        self.prior_value_optim = Adam(params=self.prior_value_params, lr=self.args.lr)

        # primitive policy
        if self.args.is_discrete:
            self.pi_func = policy.CategoricalPolicy(input_dim=self.args.obs_dim, hidden_dim=self.args.pi_hidden_dim, action_dim=self.args.act_dim)
        else:
            self.pi_func = policy.GaussianPolicy(input_dim=self.args.obs_dim, hidden_dim=self.args.pi_hidden_dim,
                                                 action_dim=self.args.act_dim, output_activation=F.tanh, act_range=self.args.act_range)  # danger
        self.pi_value = value.ValueFuntion(input_dim=self.args.obs_dim, hidden_dim=self.args.pi_hidden_dim)

        # print("1: ", self.pi_func)
        # for parameters in self.pi_func.parameters():
        #     print("2: ", parameters)

        self.pi_func_params = list(self.pi_func.parameters())
        self.pi_value_params = list(self.pi_value.parameters())
        self.pi_func_optim = Adam(params=self.pi_func_params, lr=self.args.lr)
        self.pi_value_optim = Adam(params=self.pi_value_params, lr=self.args.lr)

        log_dir = os.path.join(self.args.log_dir, 'hierarchy')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

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

    def save_models(self, path, is_prim=False):
        if not is_prim:
            torch.save(self.prior_func.state_dict(), "{}/high_pi_func.th".format(path))
            torch.save(self.prior_value.state_dict(), "{}/high_value.th".format(path))
        else:
            torch.save(self.pi_func.state_dict(), "{}/prim_pi_func.th".format(path))
            torch.save(self.pi_value.state_dict(), "{}/prim_value.th".format(path))

    def load_models(self, path):
        self.prior_func.load_state_dict(torch.load("{}/high_pi_func.th".format(path), map_location=lambda storage, loc: storage))
        self.prior_value.load_state_dict(torch.load("{}/high_value.th".format(path), map_location=lambda storage, loc: storage))
        self.pi_func.load_state_dict(torch.load("{}/prim_pi_func.th".format(path), map_location=lambda storage, loc: storage))
        self.pi_value.load_state_dict(torch.load("{}/prim_value.th".format(path), map_location=lambda storage, loc: storage))

    def init_prior(self, low_agent):
        print("Loading Prior from the low-level agent!")
        # print("1: ", self.prior_func.state_dict())
        self.prior_func.load_state_dict(low_agent.prior_func.state_dict())
        # print("2: ", self.prior_func.state_dict())
        # self.prior_value.load_state_dict(low_agent.prior_value.state_dict())
        # self.prior_func_optim.load_state_dict(low_agent.prior_func_optim.state_dict())
        # self.prior_value_optim.load_state_dict(low_agent.prior_value_optim.state_dict())

    def sample_option(self, init_state):
        with torch.no_grad():
            init_state = torch.tensor([init_state], device=self.args.device, dtype=torch.float32)
            c, _, _ = self.prior_func.forward(init_state)
            c_onehot = F.one_hot(c, self.args.code_dim+1).float()

        return c, c_onehot # (1,), (1, code_dim+1)

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.tensor([state], device=self.args.device, dtype=torch.float32)
            act, _, _ = self.pi_func.forward(state)

        return act # (1, ) or (1, act_dim)

    def sample_action_batch(self, state):
        with torch.no_grad():
            state = torch.tensor(state, device=self.args.device, dtype=torch.float32)
            act, _, _ = self.pi_func.forward(state)

        return act # (env_num, ) or (env_num, act_dim)

    # train the primitive policy with PPO
    def prim_train(self, episode_id, train_batch, success_ratio):
        horizons = train_batch.get_item("horizon")  # (bs, 1)
        dones = train_batch.get_item("done") # (bs, epi_len, 1)
        filled = train_batch.get_item("filled")  # (bs, epi_len, 1)
        states = train_batch.get_item("state")  # (bs, epi_len, obs_dim)
        acts = train_batch.get_item("action")  # (bs, epi_len, 1) or (bs, epi_len, act_dim)
        rewards = train_batch.get_item("reward")  # (bs, epi_len, 1)
        next_states = train_batch.get_item("next_state")  # (bs, epi_len, obs_dim)
        bs = next_states.shape[0]
        epi_len = next_states.shape[1]
        log_info = {}
        log_info['prim_success_ratio'] = success_ratio

        # get target value
        target_v = self.pi_value.forward(next_states.reshape(bs * epi_len, -1)) # (bs * epi_len, 1)
        target_v = (target_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach() # (bs, epi_len, 1)
        final_v = get_final_state_value(self.args, target_v, horizons)

        # get return value
        log_info['prim_mean_rwd'] = ((rewards * filled).sum() / bs).item()
        ret = get_return_array(self.args, rewards, final_v, horizons, filled)

        # update the value function
        for _ in range(self.args.value_iters):
            ret_pre = self.pi_value.forward(states.reshape(bs * epi_len, -1)) # (bs * epi_len, 1)
            pi_v_loss = (torch.square(ret_pre.view(bs, epi_len, 1) - ret) * filled).sum() / filled.sum()
            self.pi_value_optim.zero_grad()
            pi_v_loss.backward()
            self.pi_value_optim.step()
        log_info['pi_v_loss'] = pi_v_loss.item()

        # get advantage function
        curr_v = self.pi_value.forward(states.reshape(bs * epi_len, -1))
        curr_v = (curr_v.reshape(bs, epi_len, 1) * filled).detach()
        next_v = self.pi_value.forward(next_states.reshape(bs * epi_len, -1))
        next_v = (next_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach()
        adv = get_advantage(self.args, rewards, curr_v, next_v, filled) # (bs, epi_len, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.args.is_discrete:
            pi_func_a = acts.reshape(bs * epi_len) # (bs * epi_len, )
        else:
            pi_func_a = acts.reshape(bs * epi_len, self.args.act_dim) # (bs * epi_len, act_dim)

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

    # train the high-level policy with PPO
    def high_train(self, episode_id, train_batch, final_rwd):
        horizons = train_batch.get_item("horizon")  # (bs, 1)
        dones = train_batch.get_item("high_done")  # (bs, epi_len, 1)
        filled = train_batch.get_item("high_filled")  # (bs, epi_len, 1)
        states = train_batch.get_item("high_state")  # (bs, epi_len, obs_dim)
        options = train_batch.get_item("high_option")  # (bs, epi_len, 1)
        rewards = train_batch.get_item("high_reward")  # (bs, epi_len, 1)
        durations = train_batch.get_item("option_duration") # (bs, epi_len, 1)
        next_states = train_batch.get_item("high_next_state")  # (bs, epi_len, obs_dim)
        bs = next_states.shape[0]
        epi_len = next_states.shape[1]
        log_info = {}
        log_info['final_rwd'] = final_rwd
        # get target value
        target_v = self.prior_value.forward(next_states.reshape(bs * epi_len, -1))  # (bs * epi_len, 1)
        target_v = (target_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach()  # (bs, epi_len, 1)
        final_v = get_final_state_value(self.args, target_v, horizons)

        # get return value
        log_info['high_mean_rwd'] = ((rewards * filled).sum() / bs).item()
        ret = get_return_array(self.args, rewards, final_v, horizons, filled, durations)
        # log_info['high_return'] = (ret * filled).sum() / filled.sum()

        # update the value function
        for _ in range(self.args.value_iters):
            ret_pre = self.prior_value.forward(states.reshape(bs * epi_len, -1))  # (bs * epi_len, 1)
            prior_v_loss = (torch.square(ret_pre.view(bs, epi_len, 1) - ret) * filled).sum() / filled.sum()
            self.prior_value_optim.zero_grad()
            prior_v_loss.backward()
            self.prior_value_optim.step()
        log_info['high_pi_v_loss'] = prior_v_loss.item()

        # get advantage function
        curr_v = self.prior_value.forward(states.reshape(bs * epi_len, -1))
        curr_v = (curr_v.reshape(bs, epi_len, 1) * filled).detach()
        next_v = self.prior_value.forward(next_states.reshape(bs * epi_len, -1))
        next_v = (next_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach()
        adv = get_advantage(self.args, rewards, curr_v, next_v, filled, durations)  # (bs, epi_len, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        _, log_c, _ = self.prior_func.forward(states.reshape(bs * epi_len, -1), options.reshape(bs * epi_len))  # (bs * epi_len, )
        log_c = log_c.view(bs, epi_len, 1).detach()  # (bs, epi_len, 1)

        for prior_iter in range(self.args.pi_iters): # not prior_iters
            _, log_c_new, _ = self.prior_func.forward(states.reshape(bs * epi_len, -1), options.reshape(bs * epi_len))  # (bs * epi_len, )
            ratio = torch.exp(log_c_new.view(bs, epi_len, 1) - log_c)  # log_a_old
            clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv
            prior_loss = - (torch.min(ratio * adv, clip_adv) * filled).sum() / filled.sum()  # TODO: try not with the filled, NO, since the replay buffer for the hierarchical training is too sparse
            approx_kl = (log_c - log_c_new.view(bs, epi_len, 1)).mean().item()
            if approx_kl > 1.5 * self.args.target_kl: # danger, negative value
                break

            self.prior_func_optim.zero_grad()
            prior_loss.backward()
            self.prior_func_optim.step()

        log_info['high_pi_loss'] = prior_loss.item()
        log_info['high_pi_iters'] = prior_iter

        # log to tensorboard
        write_summary(self.writer, info=log_info, step=episode_id)
        print("Training info: ", log_info)







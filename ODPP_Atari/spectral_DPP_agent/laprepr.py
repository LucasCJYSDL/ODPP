import os
import collections
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from configs import get_laprepr_args
from utils import torch_tools, timer_tools, summary_tools
from spectral_DPP_agent.spectral_buffer import EpisodicReplayBuffer
from learner.base_mlp import MLP


def l2_dist(x1, x2, generalized):
    if not generalized:
        return (x1 - x2).pow(2).sum(-1)
    d = x1.shape[1]
    weight = np.arange(d, 0, -1).astype(np.float32)
    weight = torch_tools.to_tensor(weight, x1.device)
    return ((x1 - x2).pow(2)) @ weight.T

def pos_loss(x1, x2, generalized=False):
    return l2_dist(x1, x2, generalized).mean()

# used in the original code
# def _rep_loss(inprods, n, k, c, reg):
#
#     norms = inprods[torch.arange(n), torch.arange(n)]
#     part1 = inprods.pow(2).sum() - norms.pow(2).sum()
#     part1 = part1 / ((n - 1) * n)
#     part2 = - 2 * c * norms.mean() / k
#     part3 = c * c / k
#     # regularization
#     # if reg > 0.0:
#     #     reg_part1 = norms.pow(2).mean()
#     #     reg_part2 = - 2 * c * norms.mean()
#     #     reg_part3 = c * c
#     #     reg_part = (reg_part1 + reg_part2 + reg_part3) / n
#     # else:
#     #     reg_part = 0.0
#     # return part1 + part2 + part3 + reg * reg_part
#     return part1 + part2 + part3

def _rep_loss(inprods, n, k, c, reg):

    norms = inprods[torch.arange(n), torch.arange(n)]
    part1 = (inprods.pow(2).sum() - norms.pow(2).sum()) / ((n - 1) * n)
    part2 = - 2 * c * norms.mean()
    part3 = c * c * k

    return part1 + part2 + part3

def neg_loss(x, c=1.0, reg=0.0, generalized=False): # derivation and modification
    """
    x: n * d.
    The formula shown in the paper
    """
    n = x.shape[0]
    d = x.shape[1]
    if not generalized:
        inprods = x @ x.T
        return _rep_loss(inprods, n, d, c, reg)

    tot_loss = 0.0
    # tot_loss = torch.tensor(0.0, device=x.device, requires_grad=True) # danger
    for k in range(1, d+1):
        inprods = x[:, :k] @ x[:, :k].T
        tot_loss += _rep_loss(inprods, n, k, c, reg)
    return tot_loss


class LapReprLearner:

    def __init__(self, common_args, env):

        self.args = get_laprepr_args(common_args)
        self.env = env
        # NN
        self._repr_fn = MLP(layers=[self.args.obs_dim]+[self.args.lap_n_units for _ in range(self.args.lap_n_layers)]+[self.args.d],
                            activation=torch.nn.functional.relu, init=False)
        self._repr_fn.to(device=self.args.device)

        # optimizer
        opt = getattr(optim, self.args.lap_opt_args_name)
        self._optimizer = opt(self._repr_fn.parameters(), lr=self.args.lap_opt_args_lr)
        # replay_buffer
        self._replay_buffer = EpisodicReplayBuffer(max_size=self.args.lap_replay_buffer_size)

        self._global_step = 0
        self._train_info = collections.OrderedDict()

        # # create ckpt save dir and log dir
        self.saver_dir = './spectral_DPP_agent' + self.args.model_dir
        print(self.saver_dir)
        if not os.path.exists(self.saver_dir):
            os.makedirs(self.saver_dir)
        self.log_dir = './spectral_DPP_agent' + self.args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

    def _collect_samples(self):
        # collect trajectories from random actions
        print('Start collecting samples......')
        timer = timer_tools.Timer()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 1000
        while total_n_steps < self.args.n_samples:
            cur_obs = self.env.reset() # TODO: to make sure that env.reset() will initialize in the whole state space
            cur_obs = cur_obs / 255.0
            epi_len = 0
            episode = []
            while True:
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = next_obs / 255.0
                # redundant info
                transition = {'s': cur_obs, 'a': action, 'r': reward, 'next_s': next_obs, 'done': done}
                cur_obs = next_obs
                epi_len += 1
                episode.append(transition)
                # log
                total_n_steps += 1
                if (total_n_steps + 1) % collect_batch == 0:
                    print('({}/{}) steps collected.'.format(total_n_steps + 1, self.args.n_samples))
                if (epi_len >= self.args.lap_episode_limit) or done: # TODO: do not depend on 'done'
                    break
            final_transition = {'s': cur_obs, 'a': self.env.action_space.sample(), 'r': 0.0, 'next_s': cur_obs, 'done': True}
            episode.append(final_transition) # to make sure the last state in the episodes can be sampled in the future process
            self._replay_buffer.add_steps(episode)
        time_cost = timer.time_cost()
        print('Data collection finished, time cost: {}s'.format(time_cost))

    def _collect_samples_with_options(self, option_agent, episode_idx):
        print('Start collecting hierarchical samples......')
        timer = timer_tools.Timer()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 1000

        while total_n_steps < self.args.n_samples:
            cur_obs = self.env.reset() # TODO: to make sure that env.reset() will initialize in the whole state space
            is_primitive = True
            c_onehot = None

            epi_len = 0
            episode = []
            option_duration = -1
            while True:
                if (option_duration >= self.args.traj_length - 1) or is_primitive:
                    option_duration = -1

                if option_duration == -1: # high-level policy
                    if np.random.random() < self.args.primitive_ratio:
                        is_primitive = True # choose the primitive action # TODO: setting args.primitive_ratio = 0
                    else:
                        # TODO: always uniformly sampling 'c'
                        c, c_onehot = option_agent.sample_option(episode_idx, cur_obs)
                        is_primitive = False

                option_duration += 1

                if is_primitive:
                    action = self.env.action_space.sample()
                else:
                    assert c_onehot is not None
                    action = option_agent.sample_action(cur_obs, c_onehot)
                    action = action.detach().clone().cpu().numpy()[0]

                next_obs, reward, done, _ = self.env.step(action)
                # redundant info
                transition = {'s': cur_obs, 'a': action, 'r': reward, 'next_s': next_obs, 'done': done}
                cur_obs = next_obs
                epi_len += 1
                episode.append(transition)
                # log
                total_n_steps += 1
                if (total_n_steps + 1) % collect_batch == 0:
                    print('({}/{}) steps collected.'.format(total_n_steps + 1, self.args.n_samples))
                if epi_len >= self.args.lap_episode_limit or done:
                    break
            final_transition = {'s': cur_obs, 'a': self.env.action_space.sample(), 'r': 0.0, 'next_s': cur_obs, 'done': True}
            episode.append(final_transition) # to make sure the last state in the episodes can be sampled in the future process
            self._replay_buffer.add_steps(episode)
        time_cost = timer.time_cost()
        print('Hierarchical data collection finished, time cost: {}s'.format(time_cost))

    def train(self, option_agent=None, episode_idx=None):
        #self._replay_buffer = EpisodicReplayBuffer(max_size=self.args.lap_replay_buffer_size)
        # TODO: clear the replay buffer collected previously
        if option_agent is None:
            self._collect_samples()
        else:
            self._collect_samples_with_options(option_agent, episode_idx)
        # learning begins
        timer = timer_tools.Timer()
        timer.set_step(0)
        for step in tqdm(range(self.args.lap_train_steps)):
            assert step == self._global_step % self.args.lap_train_steps
            self._train_step()
            # save
            if (step + 1) % self.args.lap_save_freq == 0:
                saver_path = os.path.join(self.saver_dir, 'model_{}.ckpt'.format(step+1))
                torch.save(self._repr_fn.state_dict(), saver_path)
            # print info
            if step == 0 or (step + 1) % self.args.lap_print_freq == 0:
                steps_per_sec = timer.steps_per_sec(step)
                print('Training steps per second: {:.4g}.'.format(steps_per_sec))
                summary_str = summary_tools.get_summary_str(step=self._global_step, info=self._train_info)
                print(summary_str)
        # save the final laprepr model
        saver_path = os.path.join(self.saver_dir, 'final_model.ckpt')
        torch.save(self._repr_fn.state_dict(), saver_path)
        # log the time cost
        time_cost = timer.time_cost()
        print('Training finished, time cost {:.4g}s.'.format(time_cost))
        print("Visualize the learned embeddings!!!")
        # self.visualize_embeddings()

    def _train_step(self):
        train_batch = self._get_train_batch()
        loss = self._build_loss(train_batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._global_step += 1

    def _get_train_batch(self): # how will the discount influence the performance?
        s1, s2 = self._replay_buffer.sample_steps(self.args.lap_batch_size, mode='pair', discount=self.args.lap_discount)
        s_neg, _ = self._replay_buffer.sample_steps(self.args.lap_batch_size, mode='single')
        s1, s2, s_neg = map(self._get_obs_batch, [s1, s2, s_neg])

        batch = {}
        batch['s1'] = self._tensor(s1)
        batch['s2'] = self._tensor(s2)
        batch['s_neg'] = self._tensor(s_neg)
        return batch

    def _build_loss(self, batch): # modification
        s1 = batch['s1']
        s2 = batch['s2']
        s_neg = batch['s_neg']
        s1_repr = self._repr_fn(s1)
        s2_repr = self._repr_fn(s2)
        s_neg_repr = self._repr_fn(s_neg)

        loss_positive = pos_loss(s1_repr, s2_repr, generalized=self.args.generalized)
        loss_negative = neg_loss(s_neg_repr, c=self.args.c_neg, reg=self.args.reg_neg, generalized=self.args.generalized)

        assert loss_positive.requires_grad and loss_negative.requires_grad # danger
        loss = loss_positive + self.args.w_neg * loss_negative
        info = self._train_info
        info['loss_pos'] = loss_positive.item()
        info['loss_neg'] = loss_negative.item()
        info['loss_total'] = loss.item()
        summary_tools.write_summary(self.writer, info=info, step=self._global_step)
        return loss

    def _get_obs_batch(self, steps):

        obs_batch = steps
        return np.stack(obs_batch, axis=0)

    def _tensor(self, x):
        return torch_tools.to_tensor(x, self.args.device)

    def get_embedding_matrix(self, obs_input, normalized=True):
        # obs_input: (batch_size, obs_dim)
        obs_input = self._tensor(obs_input) # (batch_size, obs_dim)
        with torch.no_grad():
            raw_embedding_segment = self._repr_fn(obs_input)

        if normalized:
            embedding = raw_embedding_segment.cpu().detach().clone().numpy()
            embedding_norm = np.linalg.norm(embedding, axis=1)
            embedding = embedding / embedding_norm.reshape(embedding.shape[0], 1)
            return embedding

        else:
            return raw_embedding_segment

    def visualize_embeddings(self):
        import matplotlib.pyplot as plt

        result_path = os.path.dirname(os.path.abspath(__file__)) + "/visualization"
        dir = os.path.abspath(os.path.join(result_path, self.args.unique_token, str(self._global_step)))
        if not os.path.exists(dir):
            os.makedirs(dir)

        sample_num = -1 # all the samples
        interval = self.args.ev_interval
        data_input = self._replay_buffer.get_all_steps(max_num=sample_num)
        obs_input = self._get_obs_batch(data_input)
        obs_input = self._tensor(obs_input)  # maybe too much for the gpu?
        # print("1: ", obs_input.shape)
        data_size = int(obs_input.shape[0])

        embeddings = []
        with torch.no_grad():  # danger
            cur_idx = 0
            while cur_idx < data_size:
                next_idx = min(cur_idx + interval, data_size)
                data_segment = obs_input[cur_idx:next_idx, :]
                raw_embedding_segment = self._repr_fn(data_segment)
                embeddings = embeddings + raw_embedding_segment.cpu().detach().clone().tolist()
                cur_idx = next_idx
        embeddings = np.array(embeddings)
        assert embeddings.shape[0] == data_size
        embeddings = np.around(embeddings, 6)

        for dim in tqdm(range(self.args.d)):
            axis_x = np.array(data_input)[:, 0]
            axis_y = np.array(data_input)[:, 1]
            value = embeddings[:, dim]
            plt.figure()
            plt.scatter(x=axis_x, y=axis_y, c=value, cmap="viridis", alpha=0.3)
            plt.colorbar()
            plt.savefig(dir + '/' + 'embedding_{}.png'.format(dim))








import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from configs import get_hierarchical_args
from option_agent.hierarchical_policy import HierPolicy
from utils.buffer import OneHot, EpisodeBatch, ReplayBuffer
from runner import Runner

class HierRunner(object):
    def __init__(self, args, env, high_env, low_runner: Runner):
        self.args = get_hierarchical_args(args)
        self.env = env
        self.high_env = high_env
        self.low_runner = low_runner
        self.agent = HierPolicy(self.args)

        if self.args.cuda:
            self.agent.cuda()
        # if len(self.args.load_dir) > 0:
        #     print("Loading model from {}.".format(self.args.load_dir))
        #     self.agent.load_models(path=self.args.load_dir)

        # high level buffer
        self.high_scheme = {"high_option": {"vshape": 1, "dtype": torch.long}, "high_state": {"vshape": args.obs_dim}, "high_reward": {"vshape": 1},
                            "high_done": {"vshape": 1}, "high_next_state": {"vshape": args.obs_dim}, "high_filled": {"vshape": 1},
                            "horizon": {"vshape": 1, "dtype": torch.long, "traj_const": True}, "option_duration": {"vshape": 1, "dtype": torch.int64}}
        self.high_preprocess = {"high_option": ("high_option_onehot", OneHot(out_dim=self.args.code_dim + 1))}
        self.high_buffer = ReplayBuffer(self.high_scheme, self.high_preprocess, self.args.high_buffer_size, self.args.hier_episode_limit)

        # low level buffer
        self.low_scheme = {"option": {"vshape": 1, "dtype": torch.long}, "state": {"vshape": args.obs_dim}, "reward": {"vshape": 1},
                       "done": {"vshape": 1}, "next_state": {"vshape": args.obs_dim}, "updated": {"vshape": 1},
                       "horizon": {"vshape": 1, "dtype": torch.long, "traj_const": True}, "filled": {"vshape": 1}, "init_state": {"vshape": args.obs_dim}}
        self.low_preprocess = {"option": ("option_onehot", OneHot(out_dim=self.args.code_dim))} # not including the last dim -- primitive action
        if self.args.is_discrete:
            self.low_scheme['action'] = {"vshape": 1, "dtype": torch.long}
            self.low_preprocess['action'] = ("action_onehot", OneHot(out_dim=self.args.act_dim))
        else:
            self.low_scheme['action'] = {"vshape": args.act_dim}
        self.low_buffer = ReplayBuffer(self.low_scheme, self.low_preprocess, self.args.low_buffer_size, self.args.hier_episode_limit)
        # buffer for training the primitive policy
        self.prim_buffer = ReplayBuffer(self.low_scheme, self.low_preprocess, self.args.prim_buffer_size, self.args.hier_episode_limit)

    def _prim_rollout(self):
        self.agent.eval_mode()

        assert self.args.prim_buffer_size % self.args.thread_num == 0
        rollout_num = self.args.prim_buffer_size // self.args.thread_num
        success_num = 0
        for rollout_idx in tqdm(range(rollout_num)):
            s = self.env.reset() # (env_num, obs_dim)
            epi_batch = EpisodeBatch(self.low_scheme, self.low_preprocess, self.args.thread_num, self.args.hier_episode_limit, device=self.args.device)
            horizons = [1 for _ in range(self.args.thread_num)]
            done_vec = [False for _ in range(self.args.thread_num)]

            for time_step in range(self.args.hier_episode_limit):
                act = self.agent.sample_action_batch(s) # (env_num, ) or (env_num, act_dim)
                next_s, env_rwd, env_done = self.env.step(act.detach().clone().cpu().numpy(), done_vec, s) # (env_num, obs_dim), (env_num, ), (env_num)

                new_done_vec, done_list, filled_list = [], [], []
                for env_id in range(self.args.thread_num):
                    if time_step < self.args.hier_episode_limit - 1:
                        env_rwd[env_id] = 0.0
                    if done_vec[env_id]:
                        filled_list.append(0.0)
                    else:
                        filled_list.append(1.0)
                    if env_done[env_id]:
                        done = True
                    else:
                        done = False
                        if horizons[env_id] < self.args.hier_episode_limit: # cutoff
                            horizons[env_id] += 1
                    new_done_vec.append(done)
                    done_list.append(float(done))
                done_vec = new_done_vec

                transition = {"state": s, "action": act, "reward": env_rwd, "done": done_list, "next_state": next_s, "filled": filled_list}
                epi_batch.update(transition, ts=time_step)

                if np.array(env_done).all():
                    break  # danger
                else:
                    s = next_s

            epi_batch.update(data={"horizon": horizons})

            for i in range(self.args.thread_num):
                temp_sum = torch.sum( epi_batch["done"][i]).cpu().numpy()
                if temp_sum > 0:
                    success_num += 1

            self.prim_buffer.insert_episode_batch(epi_batch)
        # print(self.prim_buffer["reward"])

        return float(success_num) / self.args.prim_buffer_size

    def _high_rollout(self, episode_idx):

        self.agent.eval_mode()
        self.low_runner.agent.eval_mode()

        epi_batch = EpisodeBatch(self.high_scheme, self.high_preprocess, self.args.high_buffer_size, self.args.hier_episode_limit, device=self.args.device)

        # update the primitive ratio
        primitive_ratio = (self.args.high_episode_num - episode_idx) / float(self.args.high_episode_num) * self.args.high_primitive_ratio
        success_num = 0
        final_rwd_list = []
        for traj_idx in tqdm(range(self.args.high_buffer_size)):
            s = self.high_env.reset()
            high_s = s.copy()
            is_primitive = True
            c_onehot = None
            high_r = 0.0
            high_epi_len = 0
            option_duration = -1
            for time_step in range(self.args.hier_episode_limit):
                if (option_duration >= self.args.traj_length - 1) or is_primitive:
                    if time_step > 0:
                        # print(high_r)
                        transition = {"high_state": [high_s], "high_option": c, "high_reward": [high_r], "high_done": [0.0],
                                      "high_next_state": [s], "high_filled": [1.0], "option_duration": [option_duration + 1]}
                        epi_batch.update(transition, bs=traj_idx, ts=high_epi_len)
                        high_epi_len += 1
                        high_s = s.copy()
                        high_r = 0.0
                    option_duration = -1

                if option_duration == -1: # use the high-level policy # inappropriate for MPI

                    if np.random.random() < primitive_ratio: # dangerous for the on-policy training
                        c = torch.tensor(self.args.code_dim, dtype=torch.int64, device=self.args.device)
                        c_onehot = F.one_hot(c, self.args.code_dim + 1).float()
                        is_primitive = True # choose the primitive action, noise
                    else:
                        c, c_onehot = self.agent.sample_option(s)
                        if c.cpu().numpy() > self.args.code_dim - 1:
                            is_primitive = True
                        else:
                            is_primitive = False

                option_duration += 1

                if is_primitive:
                    act = self.agent.sample_action(s)
                else:
                    assert c.cpu().numpy() <= self.args.code_dim - 1
                    act = self.low_runner.agent.sample_action(s, option=c_onehot[:, :-1])

                next_s, reward, done, _ = self.high_env.step(act.detach().clone().cpu().numpy()[0])
                # print("1: ", reward)
                if time_step < self.args.hier_episode_limit - 1:
                    reward = 0.0
                # print("reward is: ", reward)
                # if episode_idx % 10 == 0:
                #     self.high_env.render()
                high_r += (self.args.discount) ** option_duration * reward

                if done:
                    success_num += 1
                    break
                else:
                    s = next_s
            final_rwd_list.append(reward)
            # print("2: ", reward, high_r)
            transition = {"high_state": [high_s], "high_option": c, "high_reward": [high_r], "high_done": [float(done)],
                          "high_next_state": [next_s], "high_filled": [1.0], "option_duration": [option_duration + 1]}
            epi_batch.update(transition, bs=traj_idx, ts=high_epi_len)
            epi_batch.update(data={"horizon": [high_epi_len + 1]}, bs=traj_idx)

        self.high_buffer.insert_episode_batch(epi_batch)
        # print(self.high_buffer["high_reward"])
        return np.mean(final_rwd_list)

    def _low_rollout(self, episode_idx):
        self.low_runner.agent.eval_mode()
        assert self.args.low_buffer_size % self.args.thread_num == 0
        rollout_num = self.args.low_buffer_size // self.args.thread_num

        for rollout_idx in tqdm(range(rollout_num)):
            s = self.env.reset()
            epi_batch = EpisodeBatch(self.low_scheme, self.low_preprocess, self.args.thread_num, self.args.hier_episode_limit, device=self.args.device)
            horizons = [1 for _ in range(self.args.thread_num)]
            done_vec = [False for _ in range(self.args.thread_num)]
            option_duration = [-1 for _ in range(self.args.thread_num)]
            is_update = [1.0 for _ in range(self.args.thread_num)]
            c_list = torch.zeros((self.args.thread_num, ), dtype=torch.int64, device=self.args.device)
            c_onehot_list = F.one_hot(c_list, self.args.code_dim).float()
            init_s_list = s.copy()

            for time_step in range(self.args.hier_episode_limit):
                for env_idx in range(self.args.thread_num):
                    if option_duration[env_idx] >= self.args.traj_length - 1:
                        option_duration[env_idx] = -1

                for env_idx in range(self.args.thread_num):
                    if option_duration[env_idx] == -1: # time to select new options based on the prior of the option agent
                        is_update[env_idx] = 1.0
                        c, c_onehot = self.low_runner.agent.sample_option(self.args.episode_num + episode_idx, s[env_idx])
                        init_s = s[env_idx].copy()
                        c_list[env_idx] = c[0]
                        c_onehot_list[env_idx] = c_onehot[0]
                        init_s_list[env_idx] = init_s

                for env_idx in range(self.args.thread_num):
                    option_duration[env_idx] += 1

                act = self.low_runner.agent.sample_action_batch(s, option=c_onehot_list)
                next_s, reward, env_done = self.env.step(act.detach().clone().cpu().numpy(), done_vec, s)

                new_done_vec, done_list, filled_list = [], [], []
                for env_id in range(self.args.thread_num):
                    if done_vec[env_id]:
                        filled_list.append(0.0)
                    else:
                        filled_list.append(1.0)

                    if env_done[env_id]:
                        done = True
                    else:
                        done = False
                        if horizons[env_id] < self.args.hier_episode_limit:  # cutoff
                            horizons[env_id] += 1
                    new_done_vec.append(done)
                    done_list.append(float(done))
                done_vec = new_done_vec

                transition = {"state": s, "action": act, "reward": reward, "done": done_list, "next_state": next_s, "filled": filled_list,
                              "updated": is_update, "option": c_list, "init_state": init_s_list}

                epi_batch.update(transition, ts=time_step)
                is_update = [0.0 for _ in range(self.args.thread_num)]

                if np.array(env_done).all():
                    break
                else:
                    s = next_s

            epi_batch.update(data={"horizon": horizons})
            self.low_buffer.insert_episode_batch(epi_batch)

    def train(self):
        # decouple the training of the primitive policy, low-level policy and high-level policy
        # first, we need to train the primitive policy
        for idx in range(self.args.prim_episode_num):
            print("########################### Primitive Policy Training Episode {} ###########################".format(idx))
            print("Collecting trajectories with primitive policy: ")
            prim_success_ratio = self._prim_rollout()
            print("Success ratio of this episode: ", prim_success_ratio)
            print("Start training the primitive policy: ")
            self.agent.train_mode()

            if self.prim_buffer.can_sample(self.args.prim_buffer_size):
                prim_train_batch = self.prim_buffer.sample(self.args.prim_buffer_size)
                if prim_train_batch.device != self.args.device:
                    prim_train_batch.to(self.args.device)
                self.agent.prim_train(idx, prim_train_batch, prim_success_ratio)
            # save model
            if idx % self.args.ckpt_interval == 0:
                save_path = os.path.join(self.args.model_dir, str(idx))
                os.makedirs(save_path, exist_ok=True)
                print("Saving models to {}.".format(save_path))
                self.agent.save_models(save_path, is_prim=True)

        # # then we need to train the option (low-level) policy and prior based on the entrinsic rwd (or maybe both intrinsic and extrinsic)
        # for idx in range(self.args.low_episode_num):
        #     print("########################### Low-level Training Episode {} ###########################".format(idx))
        #     print("Collecting trajectories with Low-level policy: ")
        #
        #     self._low_rollout(idx)
        #     print("Start training the low-level policy: ")
        #     self.low_runner.agent.train_mode()
        #
        #     if self.low_buffer.can_sample(self.args.low_buffer_size):
        #         low_train_batch = self.low_buffer.sample(self.args.low_buffer_size)
        #         if low_train_batch.device != self.args.device:
        #             low_train_batch.to(self.args.device)
        #         self.low_runner.agent.train_with_env_rwd(idx+self.args.episode_num, low_train_batch)
        #
        #     # save model
        #     if idx % self.args.ckpt_interval == 0:
        #         save_path = os.path.join(self.args.model_dir, str(idx+self.args.episode_num)) # we have trained the option policy for 'args.episode_num' iterations
        #         os.makedirs(save_path, exist_ok=True)
        #         print("Saving models to {}.".format(save_path))
        #         self.low_runner.agent.save_models(save_path)

        # last we need to train the high-level policy
        ## initialize the high level policy with the trained prior, only for VIC and ODPP
        # if self.args.agent_id == 'ODPP' or self.args.agent_id == 'VIC':
        #     # TODO: with or without initialization
        #     self.agent.init_prior(self.low_runner.agent)

        for idx in range(self.args.high_episode_num):
            print("########################### High-level Training Episode {} ###########################".format(idx))

            print("Collecting trajectories with high-level policy: ")
            final_reward = self._high_rollout(idx)
            print("Success ratio of this episode: ", final_reward)
            print("Start training the high-level policy: ")
            self.agent.train_mode()
            self.low_runner.agent.train_mode()

            if self.high_buffer.can_sample(self.args.high_buffer_size):
                high_train_batch = self.high_buffer.sample(self.args.high_buffer_size)
                if high_train_batch.device != self.args.device:
                    high_train_batch.to(self.args.device)
                self.agent.high_train(idx, high_train_batch, final_reward)
            # save model
            if idx % self.args.ckpt_interval == 0:
                save_path = os.path.join(self.args.model_dir, str(idx))
                os.makedirs(save_path, exist_ok=True)
                print("Saving models to {}.".format(save_path))
                self.agent.save_models(save_path)

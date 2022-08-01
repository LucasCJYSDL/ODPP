import os
import torch
from tqdm import tqdm
import numpy as np
from configs import get_rl_args
from option_agent import REGISTRY as agent_REGISTRY
from utils.buffer import OneHot, EpisodeBatch, ReplayBuffer
from spectral_DPP_agent.laprepr import LapReprLearner
from visualization.draw_trajectory import draw_traj

class Runner(object):
    def __init__(self, args, env, eval_env, lap_repr: LapReprLearner):
        self.args = get_rl_args(args)
        self.env = env # vec_env
        self.eval_env = eval_env
        self.lap_repr = lap_repr
        self.agent = agent_REGISTRY[self.args.agent_id](self.args)
        if self.args.cuda:
            self.agent.cuda()
        if len(self.args.load_dir) > 0: ###
            print("Loading model from {}.".format(self.args.load_dir))
            self.agent.load_models(path=self.args.load_dir)

        self.scheme = {"option": {"vshape": 1, "dtype": torch.long, "traj_const": True}, "state": {"vshape": args.obs_dim}, "reward": {"vshape": 1},
                       "done": {"vshape": 1}, "next_state": {"vshape": args.obs_dim},
                       "horizon": {"vshape": 1, "dtype": torch.long, "traj_const": True}, "filled": {"vshape": 1}}
        self.preprocess = {"option": ("option_onehot", OneHot(out_dim=self.args.code_dim))}
        if self.args.is_discrete:
            self.scheme['action'] = {"vshape": 1, "dtype": torch.long}
            self.preprocess['action'] = ("action_onehot", OneHot(out_dim=self.args.act_dim))
        else:
            self.scheme['action'] = {"vshape": args.act_dim}

        self.buffer = ReplayBuffer(self.scheme, self.preprocess, self.args.buffer_size, self.args.traj_length)

    def _traj_rollout(self, episode_batch, c_onehot, s):
        # collect a trajectory and update the buffer accordingly
        horizons = [1 for _ in range(self.args.thread_num)]
        done_vec = [False for _ in range(self.args.thread_num)]
        traj_rwd = {'tot_rwd': 0.0, 'fwd_rwd': 0.0, 'sde_rwd': 0.0, 'ctr_rwd': 0.0, 'sur_rwd': 0.0}

        for time_step in range(self.args.traj_length):
            act = self.agent.sample_action_batch(s, c_onehot) # (env_num, ) or (env_num, act_dim)
            # print("act: ", act)
            a = act.detach().clone().cpu().numpy()
            # if np.square(a).sum() >= 1e4:
            #     print("danger: ", a)
            #     a = self.env.action_space.sample()
            next_s, env_rwd, env_done, fwd_r, sde_r, ctr_r, sur_r = self.env.step(a, done_vec, s) # (env_num, obs_dim), (env_num, ), (env_num)
            traj_rwd['tot_rwd'] += np.sum(env_rwd)
            traj_rwd['fwd_rwd'] += np.sum(fwd_r)
            traj_rwd['sde_rwd'] += np.sum(sde_r)
            traj_rwd['ctr_rwd'] += np.sum(ctr_r)
            traj_rwd['sur_rwd'] += np.sum(sur_r)
            # print(traj_rwd)
            new_done_vec, done_list, rwd_list, filled_list = [], [], [], []
            for env_id in range(self.args.thread_num):
                rwd_list.append(0.0) # no extrinsic rwd
                if done_vec[env_id]:
                    filled_list.append(0.0)
                else:
                    filled_list.append(1.0)
                if env_done[env_id] or (time_step == self.args.traj_length - 1):
                    done = True
                else:
                    done = False
                    horizons[env_id] += 1

                new_done_vec.append(done)
                done_list.append(float(done))
            done_vec = new_done_vec

            transition = {"state": s, "action": act, "reward": rwd_list, "done": done_list, "next_state": next_s, "filled": filled_list}

            episode_batch.update(transition, ts=time_step)
            if np.array(env_done).all():  # rarely happen
                print("Pure Luck!!!", time_step)
                break  # danger
            else:
                s = next_s

        episode_batch.update(data={"horizon": horizons})
        # print(traj_rwd)
        # for k in traj_rwd.keys():
        #     traj_rwd[k] = traj_rwd[k] / float(time_step + 1)

        return traj_rwd

    def _rollout(self, episode_id):
        # collect a certain number of trajectories of the fixed length
        # TODO: use net.eval() or not
        self.agent.eval_mode()
        assert self.args.traj_num % self.args.thread_num == 0
        rollout_num = self.args.traj_num // self.args.thread_num
        traj_rwd = {'tot_rwd': 0.0, 'fwd_rwd': 0.0, 'sde_rwd': 0.0, 'ctr_rwd': 0.0, 'sur_rwd': 0.0}
        for rollout_idx in tqdm(range(rollout_num)):
            episode_batch = EpisodeBatch(self.scheme, self.preprocess, self.args.thread_num, self.args.traj_length, device=self.args.device)
            s = self.env.reset() # (env_num, obs_dim) TODO: to make sure that env.reset() will initialize in the whole state space
            c, c_onehot = self.agent.sample_option_batch(episode_id, s) # (env_num, code_dim)
            episode_batch.update(data={"option": c})
            temp_traj_rwd = self._traj_rollout(episode_batch, c_onehot, s)
            # traj_rwd += temp_traj_rwd
            for k in traj_rwd.keys():
                traj_rwd[k] += temp_traj_rwd[k]

            self.buffer.insert_episode_batch(episode_batch)

        for k in traj_rwd.keys():
            traj_rwd[k] = traj_rwd[k] / float(self.args.traj_num)

        return traj_rwd

    def _rollout_batch(self, episode_id):
        self.agent.eval_mode()

        traj_idx = 0
        traj_rwd = {'tot_rwd': 0.0, 'fwd_rwd': 0.0, 'sde_rwd': 0.0, 'ctr_rwd': 0.0, 'sur_rwd': 0.0}
        while traj_idx < self.args.traj_num:
            init_s = self.env.reset()  # (env_num, obs_dim) TODO: to make sure that env.reset() will initialize in the whole state space
            init_s = init_s[0] # (obs_dim, )
            for c_idx in range(self.args.batch_c):
                c, c_onehot = self.agent.sample_option(episode_id, init_s) # (1, ), (1, code_dim)
                s = np.array([init_s.copy() for _ in range(self.args.thread_num)])  # (env_num, obs_dim)
                c = c.repeat((self.args.thread_num)) # (env_num)
                c_onehot = c_onehot.repeat((self.args.thread_num, 1)) # (env_num, code_dim)

                assert self.args.batch_traj % self.args.thread_num == 0
                for t_idx in range(self.args.batch_traj//self.args.thread_num):
                    episode_batch = EpisodeBatch(self.scheme, self.preprocess, self.args.thread_num, self.args.traj_length, device=self.args.device)
                    episode_batch.update(data={"option": c})
                    self.env.reset()
                    self.env.set_init_state(s) # danger TODO
                    temp_traj_rwd = self._traj_rollout(episode_batch, c_onehot, s)
                    # traj_rwd += temp_traj_rwd
                    for k in traj_rwd.keys():
                        traj_rwd[k] += temp_traj_rwd[k]
                    self.buffer.insert_episode_batch(episode_batch)
                    traj_idx += self.args.thread_num
            print('({}/{}) trajectories collected.'.format(traj_idx, self.args.traj_num))

        for k in traj_rwd.keys():
            traj_rwd[k] = traj_rwd[k] / float(self.args.traj_num)

        return traj_rwd

    # def _draw_option_trajectories(self, epi_idx, landmark_only=False):
    #     traj_list = {}
    #     raw_traj_list = {}
    #     interval = self.args.traj_length // self.args.landmark_num
    #     for c_id in tqdm(range(self.args.code_dim)):
    #         c = torch.tensor([c_id], dtype=torch.int64, device=self.args.device)
    #         c_onehot = torch.nn.functional.one_hot(c, self.args.code_dim).float()
    #
    #         traj_list[c_id] = []
    #         raw_traj_list[c_id] = []
    #
    #         for traj_id in range(self.args.visual_traj_num):
    #             traj_x, traj_y, raw_traj_x, raw_traj_y = [], [], [], []
    #             s = self.eval_env.reset()
    #             xy = self.eval_env.get_xy()
    #             traj_x.append(xy[0])
    #             traj_y.append(xy[1])
    #             raw_traj_x.append(xy[0])
    #             raw_traj_y.append(xy[1])
    #
    #             for step_id in range(self.args.traj_length):
    #                 act = self.agent.sample_action(s, c_onehot)  # (1, ) or (1, act_dim)
    #                 # print("act: ", act)
    #                 next_s, env_rwd, env_done, _ = self.eval_env.step(act.detach().clone().cpu().numpy()[0])
    #                 xy = self.eval_env.get_xy()
    #                 raw_traj_x.append(xy[0])
    #                 raw_traj_y.append(xy[1])
    #
    #                 if (step_id + 1) % interval == 0:
    #                     traj_x.append(xy[0])
    #                     traj_y.append(xy[1])
    #
    #                 s = next_s
    #
    #             traj_list[c_id].append({'x': traj_x, 'y': traj_y})
    #             raw_traj_list[c_id].append({'x': raw_traj_x, 'y': raw_traj_y})
    #
    #     draw_traj(self.args.env_id, self.args.code_dim, traj_list, self.args.unique_token, epi_idx)
    #     draw_traj(self.args.env_id, self.args.code_dim, raw_traj_list, self.args.unique_token, epi_idx, is_raw=True)

    def _draw_option_trajectories(self, epi_idx, landmark_only=False):
        # only for ODPP and VIC on Corridor
        start_list = [np.array([0, 0]), np.array([-4, 12]), np.array([-12, -4]), np.array([4, -12]), np.array([12, 4]),
                      np.array([-20, 8]), np.array([8, 20]), np.array([20, -8]), np.array([-8, -20])]
        traj_list = {}
        raw_traj_list = {}
        interval = self.args.traj_length // self.args.landmark_num

        for s_id in range(len(start_list)):
            traj_list[s_id] = []
            raw_traj_list[s_id] = []

            for traj_id in range(3):
                traj_x, traj_y, raw_traj_x, raw_traj_y = [], [], [], []
                s = self.eval_env.reset()
                s = self.eval_env.set_init_xy(start_list[s_id])
                xy = self.eval_env.get_xy()
                traj_x.append(xy[0])
                traj_y.append(xy[1])
                raw_traj_x.append(xy[0])
                raw_traj_y.append(xy[1])
                c, c_onehot = self.agent.sample_option(epi_idx, s)

                for step_id in range(self.args.traj_length):
                    act = self.agent.sample_action(s, c_onehot)  # (1, ) or (1, act_dim)
                    # print("act: ", act)
                    next_s, env_rwd, env_done, _ = self.eval_env.step(act.detach().clone().cpu().numpy()[0])
                    xy = self.eval_env.get_xy()
                    raw_traj_x.append(xy[0])
                    raw_traj_y.append(xy[1])

                    if (step_id + 1) % interval == 0:
                        traj_x.append(xy[0])
                        traj_y.append(xy[1])

                    s = next_s

                traj_list[s_id].append({'x': traj_x, 'y': traj_y})
                raw_traj_list[s_id].append({'x': raw_traj_x, 'y': raw_traj_y})

        draw_traj(self.args.env_id, len(start_list), traj_list, self.args.unique_token, epi_idx)
        draw_traj(self.args.env_id, len(start_list), raw_traj_list, self.args.unique_token, epi_idx, is_raw=True)


    def train(self):
        for idx in range(self.args.episode_num):
            print("########################### Training Episode {} ###########################".format(idx))
            if (self.args.agent_id == 'ODPP' or self.args.agent_id == 'DCO') and (idx % self.args.lap_train_interval == 0):
                print("Time to train the Laplacian embeddings!")
                if idx == 0:
                    self.lap_repr.train()
                else:
                    self.lap_repr.train(option_agent=self.agent, episode_idx=idx)

            print("Collecting trajectories: ")
            if self.args.agent_id == 'ODPP':
                traj_rwd = self._rollout_batch(idx)
            else:
                traj_rwd = self._rollout(idx)
            print("Trajectory reward: ", traj_rwd)

            # start training
            print("Start training: ")
            self.agent.train_mode()
            if self.buffer.can_sample(self.args.traj_num):
                train_batch = self.buffer.sample(self.args.traj_num)
                if train_batch.device != self.args.device:
                    train_batch.to(self.args.device)
                if self.args.agent_id == 'ODPP' or self.args.agent_id == 'DCO':
                    self.agent.train(idx, train_batch, self.lap_repr, traj_rwd=traj_rwd)
                else:
                    self.agent.train(idx, train_batch, traj_rwd=traj_rwd)
            # save model
            if idx % self.args.ckpt_interval == 0:
                save_path = os.path.join(self.args.model_dir, str(idx))
                os.makedirs(save_path, exist_ok=True)
                print("Saving models to {}.".format(save_path))
                self.agent.save_models(save_path)
                # if self.args.visualize_traj:
                #     print("Start to visualize option trajectories.")
                #     self._draw_option_trajectories(idx, landmark_only=True)


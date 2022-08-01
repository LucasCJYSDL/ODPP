import os
import gym
import torch
import datetime
import random
import numpy as np
from configs import get_common_args
from runner import Runner
from hierarchical_runner import HierRunner
from spectral_DPP_agent.laprepr import LapReprLearner
from visualization.draw_option_dist import draw_option_ori_dist, draw_option_choice_dist
from utils.env_wrapper import EnvWrapper
import robo_env

def main():
    # prepare
    args = get_common_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    unique_token = f"{args.env_id}_seed{args.seed}_alg{args.agent_id}_{datetime.datetime.now()}"
    args.model_dir = args.model_dir + '/' + unique_token
    args.log_dir = args.log_dir + '/' + unique_token
    args.unique_token = unique_token

    if torch.cuda.is_available() and args.cuda:
        args.device = torch.device('cuda')
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        args.device = torch.device('cpu')
    print('device: {}.'.format(args.device))

    # set up the environment
    # TODO: add output activation on the continuous policy based on the (min, max) of the action space
    env = gym.make(args.env_id)
    if isinstance(env.action_space, gym.spaces.Box):
        args.is_discrete = False
        args.act_dim = env.action_space.shape[0]
        args.act_range = min(env.action_space.high)
        assert args.act_range > 0
        print("action range: ", args.act_range)
    else:
        assert isinstance(env.action_space, gym.spaces.Discrete), env.action_space
        args.is_discrete = True
        args.act_dim = env.action_space.n
    args.obs_dim = env.observation_space.shape[0]

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # learn the laplacian representation
    lap_repr = LapReprLearner(args, env)
    # key part
    vec_env = EnvWrapper(args.env_id, args.seed, args.thread_num)

    runner = Runner(args, vec_env, env, lap_repr)
    runner.train()
    # draw_option_ori_dist(args, env, runner)
    # draw_option_choice_dist(args, env, runner)

    # print("Start to train the hierarchical policy!")
    # hier_runner = HierRunner(args, vec_env, env, runner)
    # hier_runner.train()

if __name__ == '__main__':
    main()

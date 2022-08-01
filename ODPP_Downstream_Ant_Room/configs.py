import argparse

def get_common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_id', type=str, default='Ant4Rooms-v0', help='RL environment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--agent_id', type=str, default='DIAYN', help='algorithm for option discovery') # 'VIC' or 'VALOR' or 'DIAYN' or 'ODPP' or 'DCO' for now
    parser.add_argument('--base_alg', type=str, default='PPO', help='algorithm for optimizing the NNs')
    parser.add_argument('--log_dir', type=str, default='./log', help='where to save the log files')
    parser.add_argument('--model_dir', type=str, default='./ckpt', help='where to save the ckpt files')
    parser.add_argument('--load_dir', type=str, default='./pre_ckpt/DIAYN', help='where to load the pre_trained ckpt files')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--thread_num', type=int, default=10, help='number of threads to run in parallel')
    parser.add_argument('--render', type=bool, default=False, help='whether to render the env')
    parser.add_argument('--visualize_traj', type=bool, default=True, help='whether to visualize the options')

    args = parser.parse_args()
    return args

def get_rl_args(args):
    # rollout parameters
    args.episode_num = 15000  # 10000 the total number of training episodes
    args.lap_train_interval = 15000
    args.ckpt_interval = 50
    args.traj_num = 200
    args.visual_traj_num = 1
    args.traj_length = 50 ## ??
    args.landmark_num = 10 # TODO: fine-tuning
    args.buffer_size = 200 # for 'reinforce', the buffer_size should be equal to traj_num; for 'SAC', it should be larger than that
    args.batch_c = 10  # for the same s_0 how many c to collect
    args.batch_traj = 10  # for the same (s_0, c) how many trajectories to collect

    # network structure ## ??
    args.nn_feature_dim = 50
    args.code_dim = 10 # number of skills to learn
    args.prior_hidden_dim = 64
    args.pi_hidden_dim = 64
    args.rnn_dec_hidden_dim = 64
    args.dec_hidden_dim = 180

    # objective terms
    args.normalized = True
    args.dec_w = 1.0 # 1.0
    args.beta = 1e-3 # 1e-3
    args.lr = 1e-3
    args.dec_iters = 10 # TODO: fine-tuning
    args.value_iters = 10 # TODO: fine-tuning
    args.keep_prior_iters = 15000 # 1000

    # PPO related
    args.discount = 0.99 # should be low when using 'reinforce' to reduce the variance, TODO: set it as 1 to be in line with the original objective
    args.lamda = 0.97 # used for calculating GAE advantage TODO: set it as 1 to be in line with the original objective
    args.clip_ratio = 0.2
    args.pi_iters = 3 # TODO: fine-tuning
    args.target_kl = 0.01 # TODO: fine-tuning

    # DPP related
    args.dpp_epsilon = 1.0  # 1.0 original data: 1E-10 TODO: fine-tuning
    args.dpp_only = False  # only keep the DPP-based term or not
    args.alpha_1 = 0.0001  # 0.0001 Please refer to the paper for definition # TODO: fine-tuning
    args.alpha_2 = 0.0  # 0.01 TODO: fine-tuning
    args.alpha_3 = 0.001 # 0.01 TODO: fine-tuning
    args.dpp_traj_feature = True  # whether to use DPP to define the trajectory feature

    return args

def get_laprepr_args(args):

    args.d = 30 # the smallest d eigenvectors, should be larger or equal to the skill number, since it's upper bound the rank
    args.n_samples = 500000 # 70000 the total number of samples for training: TODO: fine-tuning
    args.lap_episode_limit = 25000 # TODO: fine-tuning
    args.primitive_ratio = 0.5 # TODO: fine-tuning
    args.w_neg = 1.0
    args.c_neg = 1.0
    args.reg_neg = 0.0
    args.generalized = True # generalized spectral drawing or not
    args.normalized_feature = True

    args.lap_n_layers = 3
    args.lap_n_units = 256

    args.lap_batch_size = 128
    args.lap_discount = 0.9 # important hyperparameters # 0.9
    args.lap_replay_buffer_size = 10000 # in fact 10000 * epi_length; original parameter: 100000,
    args.lap_opt_args_name = 'Adam'
    args.lap_opt_args_lr = 0.001
    args.lap_train_steps = 30000 # 30000, TODO: fine-tuning
    args.lap_print_freq = 5000
    args.lap_save_freq = 10000

    args.ev_n_samples = 70000
    args.ev_interval = 10000

    return args

def get_hierarchical_args(args):
    # rollout
    args.prim_episode_num = 10 # 10000
    args.high_episode_num = 10000
    args.low_episode_num = 0 # TODO: 0

    args.high_buffer_size = 10
    args.low_buffer_size = 10
    args.prim_buffer_size = 10

    args.hier_episode_limit = 300
    args.high_primitive_ratio = 0.0

    return args


import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from runner import Runner

def get_option_orientation(args, env, runner: Runner):
    env.set_sample_inits(False)

    ori_list = {}
    traj_num = 100
    traj_len = 50
    for c_id in range(args.code_dim):
        c = torch.tensor([c_id], dtype=torch.int64, device=args.device)
        c_onehot = torch.nn.functional.one_hot(c, args.code_dim).float()
        ori_list[c_id] = []

        for t_id in range(traj_num):
            s = env.reset()
            for s_id in range(traj_len):
                act = runner.agent.sample_action(s, c_onehot)  # (1, ) or (1, act_dim)
                # print("act: ", act)
                next_s, env_rwd, env_done, info = env.step(act.detach().clone().cpu().numpy()[0])
                ori = info['ori'] # (-pi, pi)
                if act.detach().clone().cpu().numpy()[0][0] < 0:
                    ori = ori + np.pi # (0, 2*pi)
                    if ori > np.pi:
                        ori -= 2*np.pi
                ori_list[c_id].append(ori/np.pi*180)

                s = next_s
    env.set_sample_inits(True)

    return ori_list

def draw_option_ori_dist(args, env, runner: Runner):
    sns.set_theme(style="darkgrid")
    ori_list = get_option_orientation(args, env, runner)
    # sns.set(style="white", palette="muted", color_codes=True)

    # Set up the matplotlib figure
    sns.set(font_scale=1.5)
    # f, axes = plt.subplots(2, 5, figsize=(20, 5), sharey=True)
    f, axes = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)
    sns.despine(left=True)

    # for c_id in range(args.code_dim):
    #     row = c_id // 5
    #     col = c_id % 5
    #     sns.distplot(ori_list[c_id], color="m", ax=axes[row, col])
    #     axes[row, col].set_title('Option #{}'.format(c_id))

    sns.distplot(ori_list[3], color="m", ax=axes[0])
    axes[0].set_title('Option #{}'.format(3))

    sns.distplot(ori_list[5], color="m", ax=axes[1])
    axes[1].set_title('Option #{}'.format(5))

    plt.setp(axes, xticks=[-180, -90, 0, 90, 180])
    plt.tight_layout()
    f.text(0.5, 0, 'Orientation', ha='center')
    # f.text(0.04, 0.5, 'Density', va='center', rotation='vertical')
    plt.show()

def draw_option_choice_dist(args, env, runner: Runner):
    # start_list = [np.array([-4, 20]), np.array([-20, -4]), np.array([4, -20]), np.array([20, 4])]
    start_list = [np.array([-4, 20]), np.array([4, -20])]

    traj_list = {}
    traj_num = 100

    for s_id in range(len(start_list)):
        traj_list[s_id] = []

        for traj_id in range(traj_num):
            s = env.reset()
            s = env.set_init_xy(start_list[s_id])
            # print(s)
            c, c_onehot, c_dist = runner.agent.sample_option(0, s)
            traj_list[s_id].append(c_dist.cpu().numpy()[0])

        traj_list[s_id] = np.mean(traj_list[s_id], axis=0)
        # print(traj_list[s_id])

    # draw histgrams
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.5)
    # f, axes = plt.subplots(2, 2, figsize=(8, 5), sharey=True)
    f, axes = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)
    sns.despine(left=True)

    x = np.arange(10)

    # for s_id in range(len(start_list)):
    #     row = s_id // 2
    #     col = s_id % 2
    #     sns.barplot(x=x, y=traj_list[s_id], color="cornflowerblue", ax=axes[row, col])
    #     axes[row, col].set_title('Location #{}'.format(s_id+1))

    sns.barplot(x=x, y=traj_list[0], color="cornflowerblue", ax=axes[0])
    axes[0].set_title('Location #{}'.format(1))

    sns.barplot(x=x, y=traj_list[1], color="cornflowerblue", ax=axes[1])
    axes[1].set_title('Location #{}'.format(3))

    plt.tight_layout()
    f.text(0.5, 0.01, 'Option Choice', ha='center')
    # f.text(0.04, 0.5, 'Density', va='center', rotation='vertical')
    plt.show()




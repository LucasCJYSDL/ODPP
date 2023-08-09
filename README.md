# ODPP

Codebase for my paper: ODPP: A Unified Algorithm Framework for Unsupervised Option Discovery based on Determinantal Point Process

Language: Python

The following parts are included:
- Benchmarks built with Mujoco, including Point/Ant Maze and Ant Locomotion.
- An implementation of the option discovery algorithm proposed in our paper, including PPO, spectral learning, and DPP-related algorithms as the components.
- Implementations of the SOTA unsupervised option discovery algorithms as baselines: VIC, VALOR, DIAYN, and DCO (Deep Covering Option Discovery).

## How to config the environments:
- python 3.6
- pytorch 1.6
- tensorboard 2.5
- mujoco_py >= 1.5
- atari_py 0.2.6
- gym  0.19.0
- matplotlib
- tqdm
- seaborn
- ...

## Experiments on the ablation study
- You need to first enter the folder 'ODPP_Ablation'.

- To run the code with algorithm XXX (i.e., VIC, VALOR, DIAYN, ODPP or DCO):

```bash
python main.py --agent_id='XXX'
```

- For the hyperparameters, please refer to 'ODPP_Ablation/configs.py'.

- To run ODPP without certain DPP terms, you can simply do that by setting 'args.alpha_x' in 'ODPP_Ablation/configs.py' as 0, where 'x' can be 1, 2 or 3 with the same definition as in the paper.

- To run ODPP using trajectory features defined with hidden layer output, you can change 'args.dpp_traj_feature' in 'ODPP_Ablation/configs.py' to 'False'.

## Learning options for the 3D locomotion tasks
- You need to first enter the folder 'ODPP_Locomotion'.

- To run the code with algorithm XXX (i.e., VIC, VALOR, DIAYN, ODPP or DCO) and random seed Y for which we simply choose 0, 1, or 2:

```bash
python main.py --agent_id='XXX' --seed=Y
```

## Experiments on the number of options to learn at a time for the 3D locomotion tasks
- You need to first enter the folder 'ODPP_Locomotion_Number'.

- To run ODPP without a certain number of options to learn at the same time, you can simply do that by setting 'args.code_dim' in 'ODPP_Locomotion_Number/configs.py' as 10, 20, 30, 40, 50 or 60, and then use:
```bash
python main.py
```

## Experiments on applying the learned options to the downstream tasks
- To run the experiments on a certain downtream task, you need to first enter the corresponding folder and the instructions to run the codes are the same.

- In this section, we take the Mujoco Room goal-achieving task with the Point agent as an example:

- You need to first enter the folder 'ODPP_Downstream_Point_Room'.

- To run the code with algorithm XXX (i.e., VIC, VALOR, DIAYN, ODPP or DCO) and random seed Y for which we simply choose 0, 1, or 2:

```bash
python main.py --agent_id='XXX' --seed=Y --load_dir='./pre_ckpt/XXX'
```

- To run the code with PPO, you only need to change the 'args.prim_episode_num' in 'ODPP_Downstream_Point_Room/configs.py' as 25000, then the first 25000 training steps are trained with PPO.

- For the goal-achieving tasks with the Point agent, you can change the goal in the environment by modifying 'ODPP_Downstream_Point_Room/robo_env/robo_maze/maze_task.py' or 'ODPP_Downstream_Point_Corridor/robo_env/robo_maze/maze_task.py'. In the former one, you need to change the goal in Line 108 as any one of the four goals listed in Line 107. In the latter one, In the former one, you need to change the goal in Line 169 as any one of the four goals listed in Line 168.

## Experiments on the Prior Network
- You need to first enter the folder 'ODPP_Prior'.

- You can run the code simply by:
```bash
python main.py
```

## Experiments on the relationship with Laplacian-based option discovery
- You need to first enter the folder 'ODPP_Relation'.

- You can run the code simply by:
```bash
python main.py
```

## How to run experiments on OpenAI Gym tasks
- You need to first enter the folder 'ODPP_Atari'.
- To run the code with algorithm XXX (i.e., VIC, VALOR, DIAYN, ODPP, DCO, DADS, APS) and random seed Y on task ZZZ (i.e., AirRaid-ram-v0, CartPole-v1, or Riverraid-ram-v0):

```bash
python main.py --agent_id='XXX' --seed=Y --env_id='ZZZ'
```

- For the simpler task 'CartPole-v1', we learn options with a longer horizon, i.e., 100. Hence, you need to change the value of 'args.traj_length' on Line 29 of 'configs.py' to 100. While, for the other tasks, the value of 'args.traj_length' should be 50.
- The tensorboard files containing the training information can be found in 'option_agent/log', within which the term 'tot_rwd' records the change of the trajectory reward in the training process.
- The introduction of the Atari environments that we have evaluated on can be found in [Atari Wiki](https://github.com/openai/gym/wiki/Table-of-environments).


## How to get the quantitative ablation study results

- First, you need to generate checkpoints for different algorithms in the Mujoco Maze tasks (Point4Rooms-v0 or PointCorridor-v0) as introduced in the first section above.
- Alternatively, you can directly use the provided checkpoints in 'ODPP_Quan/pre_ckpt' and 'ODPP_Quan/pre_ckpt_corr' which are for Point4Rooms-v0 and PointCorridor-v0, respectively. In these folders, 'ODPP_MAP' refers to 'ODPP (L<sup>IB</sup>)', 'ODPP_MAP_f' refers to  'ODPP (L<sup>IB</sup>, L<sup>DPP</sup><sub>1</sub>)', and 'ODPP_MAP_f_g_h' refers to our algorithm.
- To test the checkpoints, you need to enter the folder 'ODPP_Quan'.
- To run the code with checkpoint XXX on task ZZZ (Point4Rooms-v0 or PointCorridor-v0):
```bash
python main.py --env_id='ZZZ' --load_dir='XXX'
```
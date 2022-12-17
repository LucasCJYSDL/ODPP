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
- gym <= 2.0
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


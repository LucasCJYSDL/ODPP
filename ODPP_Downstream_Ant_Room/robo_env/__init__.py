"""
Mujoco Maze
----------

A maze environment using mujoco that supports custom tasks and robots.
"""

import gym

from robo_env.robo_maze.ant import AntEnv
from robo_env.robo_maze.point import PointEnv
from robo_env.robo_maze.swimmer import SwimmerEnv

from robo_env.robo_control.ant import AntControlEnv
from robo_env.robo_control.half_cheetah import HalfCheetahControlEnv
from robo_env.robo_control.humanoid import HumanoidControlEnv

from robo_env.robo_maze.maze_task import TaskRegistry

gym.envs.register(id="AntControl-v0",
                  entry_point="robo_env.robo_control.ant:AntControlEnv",
                  max_episode_steps=5000,
                  kwargs=dict(
                       task="forward",
                       expose_all_qpos=True))

gym.envs.register(id="HalfCheetahControl-v0",
                  entry_point="robo_env.robo_control.half_cheetah:HalfCheetahControlEnv",
                  max_episode_steps=5000,
                  kwargs=dict(expose_all_qpos=True))

gym.envs.register(id="HumanoidControl-v0",
                  entry_point="robo_env.robo_control.humanoid:HumanoidControlEnv",
                  max_episode_steps=5000,
                  kwargs=dict(expose_all_qpos=True))

gym.envs.register(id="HumanoidControl-v1",
                  entry_point="robo_env.robo_control.humanoid:HumanoidControlEnv",
                  max_episode_steps=5000,
                  kwargs=dict(expose_all_qpos=True, task='goal'))

gym.envs.register(id="HumanoidControl-v2",
                  entry_point="robo_env.robo_control.humanoid:HumanoidControlEnv",
                  max_episode_steps=5000,
                  kwargs=dict(expose_all_qpos=True, task='follow_goals'))


for maze_id in TaskRegistry.keys():
    for i, task_cls in enumerate(TaskRegistry.tasks(maze_id)):
        point_scale = task_cls.MAZE_SIZE_SCALING.point
        if point_scale is not None:
            # Point
            gym.envs.register(
                id=f"Point{maze_id}-v{i}",
                entry_point="robo_env.robo_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=PointEnv,
                    maze_task=task_cls,
                    maze_size_scaling=point_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=5000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )

        ant_scale = task_cls.MAZE_SIZE_SCALING.ant
        if ant_scale is not None:
            # Ant
            gym.envs.register(
                id=f"Ant{maze_id}-v{i}",
                entry_point="robo_env.robo_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=AntEnv,
                    maze_task=task_cls,
                    maze_size_scaling=ant_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING
                ),
                max_episode_steps=100000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )

        swimmer_scale = task_cls.MAZE_SIZE_SCALING.swimmer
        if swimmer_scale is not None:
            # Swimmer
            gym.envs.register(
                id=f"Swimmer{maze_id}-v{i}",
                entry_point="robo_env.robo_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=SwimmerEnv,
                    maze_task=task_cls,
                    maze_size_scaling=swimmer_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=5000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )

__version__ = "0.2.0"

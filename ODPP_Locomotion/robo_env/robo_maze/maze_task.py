"""Maze tasks that are defined by their map, termination condition, and goals.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np

from robo_env.robo_maze.maze_env_utils import MazeCell


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


class MazeGoal:
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 1.0,
        custom_size: Optional[float] = None,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb
        self.threshold = threshold
        self.custom_size = custom_size

    def neighbor(self, obs: np.ndarray) -> float: # so the first 2 dimensions have to be (x, y)
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]


class MazeTask(ABC):
    REWARD_THRESHOLD: float
    PENALTY: Optional[float] = None
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=4.0, swimmer=4.0)
    INNER_REWARD_SCALING: float = 0.01

    def __init__(self, scale: float) -> None:
        self.goals = []
        self.scale = scale

    def sample_goals(self) -> bool:
        return False

    def termination(self, obs: np.ndarray) -> bool:
        for goal in self.goals:
            if goal.neighbor(obs):
                return True
        return False

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass


class DistRewardMixIn:
    REWARD_THRESHOLD: float = -1000.0
    goals: List[MazeGoal]
    scale: float

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs) / self.scale


class GoalReward4Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=4.0, swimmer=4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([8.0 * scale, -8.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        # return [
        #     [B, B, B, B, B, B, B, B, B],
        #     [B, E, E, E, B, E, E, E, B],
        #     [B, E, E, E, E, E, E, E, B],
        #     [B, E, E, E, B, E, E, E, B],
        #     [B, B, E, B, B, B, E, B, B],
        #     [B, E, E, E, B, E, E, E, B],
        #     [B, E, E, E, E, E, E, E, B],
        #     [B, R, R, R, B, E, E, E, B],
        #     [B, B, B, B, B, B, B, B, B],
        # ]

        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, B, B, E, B, B, E, B, B, B, B, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, E, B, E, B, B, E, B, B, E, E, E, B],
            [B, E, B, E, B, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, B, E, B, E, B],
            [B, E, E, E, B, E, E, R, E, E, B, E, E, E, B],
            [B, E, B, E, B, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, B, E, B, E, B],
            [B, E, E, E, B, B, E, B, B, E, B, E, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, B, B, B, B, E, B, B, E, B, B, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]


class DistReward4Rooms(GoalReward4Rooms, DistRewardMixIn):
    pass



# mainly to test the prior
class GoalRewardCorridor(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=4.0, swimmer=4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([8.0 * scale, -8.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, B, B, E, E, E, E, E, B],
            [B, E, B, B, E, B, B, E, B, B, B, E, B],
            [B, E, B, B, E, B, B, E, B, B, B, E, B],
            [B, E, B, B, B, B, B, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B, B, B, B, B],
            [B, B, B, B, B, E, R, E, B, B, B, B, B],
            [B, B, B, B, B, E, E, E, E, E, E, E, B],
            [B, E, E, E, B, E, B, B, B, B, B, E, B],
            [B, E, B, B, B, E, B, B, E, B, B, E, B],
            [B, E, B, B, B, E, B, B, E, B, B, E, B],
            [B, E, E, E, E, E, B, B, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]


class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "4Rooms": [GoalReward4Rooms, DistReward4Rooms],
        "Corridor": [GoalRewardCorridor]
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]

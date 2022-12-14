# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from gym import utils
import numpy as np
import random
from gym.envs.mujoco import mujoco_env


def mass_center(sim):
    mass = np.expand_dims(sim.model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


# pylint: disable=missing-docstring
class HumanoidControlEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 expose_all_qpos=False,
                 model_path='humanoid.xml',
                 task=None,
                 goal=None):

        self._task = task
        self._goal = goal
        if self._task == "follow_goals":
            self._goal_list = [
                np.array([3.0, -0.5]),
                np.array([6.0, 8.0]),
                np.array([12.0, 12.0]),
            ]
            self._goal = self._goal_list[0]
            print("Following a trajectory of goals:", self._goal_list)

        self._expose_all_qpos = expose_all_qpos
        xml_path = os.path.dirname(os.path.abspath(__file__)) + "/assets"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))
        print("model path: ", model_path)
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        if self._expose_all_qpos:
            return np.concatenate([
                data.qpos.flat, data.qvel.flat,
                # data.cinert.flat, data.cvel.flat,
                # data.qfrc_actuator.flat, data.cfrc_ext.flat
            ])
        return np.concatenate([
            data.qpos.flat[2:], data.qvel.flat, data.cinert.flat, data.cvel.flat,
            data.qfrc_actuator.flat, data.cfrc_ext.flat
        ])

    def step(self, a):
        pos_before = mass_center(self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.sim.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        if self._task == "follow_goals":
            xposafter = self.sim.data.qpos.flat[0]
            yposafter = self.sim.data.qpos.flat[1]
            reward = -np.linalg.norm(np.array([xposafter, yposafter]).T - self._goal)
            # update goal
            if np.abs(reward) < 0.5:
                self._goal = self._goal_list[0]
                self._goal_list = self._goal_list[1:]
                print("Goal Updated:", self._goal)

        elif self._task == "goal":
            xposafter = self.sim.data.qpos.flat[0]
            yposafter = self.sim.data.qpos.flat[1]
            reward = -np.linalg.norm(np.array([xposafter, yposafter]).T - self._goal)

        qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0)) # too strict TODO
        done = False
        return self._get_obs(), reward, done, dict(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=-quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + np.random.uniform(low=-c, high=c, size=self.sim.model.nq),
            self.init_qvel + np.random.uniform(low=-c, high=c, size=self.sim.model.nv))

        if self._task == "follow_goals":
            self._goal = self._goal_list[0]
            self._goal_list = self._goal_list[1:]
            print("Current goal:", self._goal)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2.0

    # just to make sure
    def seed(self, seed_idx=None):
        super().seed(seed_idx)
        self.action_space.np_random.seed(seed_idx)
        random.seed(seed_idx)
        np.random.seed(seed_idx)

    def set_init_state(self, state: np.ndarray):
        # assert len(state) == 18  # self._expose_all_qpos == True
        qpos = state[:self.sim.model.nq]
        qvel = state[self.sim.model.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()

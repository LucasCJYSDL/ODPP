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
import math
from gym import utils
import numpy as np
import random
from gym.envs.mujoco import mujoco_env


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


# pylint: disable=missing-docstring
class AntControlEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 task="motion",
                 goal=None,
                 expose_all_qpos=False,
                 expose_body_coms=None,
                 expose_body_comvels=None,
                 expose_foot_sensors=False,
                 use_alt_path=False,
                 model_path="ant.xml"):
        self._task = task
        self._goal = goal
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._expose_foot_sensors = expose_foot_sensors
        self._body_com_indices = {}
        self._body_comvel_indices = {}

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py

        xml_path = os.path.dirname(os.path.abspath(__file__)) + "/assets"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))
        # print("model path: ", model_path)
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.sim.data.qpos.flat[1]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.sim.data.qpos.flat[1]

        forward_reward = (xposafter - xposbefore) / self.dt
        sideward_reward = (yposafter - yposbefore) / self.dt

        ctrl_cost = .5 * np.square(a).sum()
        survive_reward = 1.0
        if self._task == "forward":
            reward = forward_reward - ctrl_cost + survive_reward
        elif self._task == "backward":
            reward = -forward_reward - ctrl_cost + survive_reward
        elif self._task == "left":
            reward = sideward_reward - ctrl_cost + survive_reward
        elif self._task == "right":
            reward = -sideward_reward - ctrl_cost + survive_reward
        elif self._task == "goal":
            reward = -np.linalg.norm(np.array([xposafter, yposafter]) - self._goal)
        elif self._task == "motion":
            reward = np.sum(np.abs(np.array([forward_reward, sideward_reward]))) - ctrl_cost + survive_reward
            # print("Here: ", reward, forward_reward, sideward_reward, ctrl_cost, survive_reward)
        else:
            raise NotImplementedError

        state = self.state_vector()
        notdone = np.isfinite(state).all() and ctrl_cost < 1e4
        done = not notdone
        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_forward=np.abs(forward_reward),
            reward_sideward=np.abs(sideward_reward),
            reward_ctrl=-ctrl_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        # No crfc observation
        if self._expose_all_qpos:
            obs = np.concatenate([
                self.sim.data.qpos.flat[:15],
                self.sim.data.qvel.flat[:14],
            ])
        else:
            obs = np.concatenate([
                self.sim.data.qpos.flat[2:15],
                self.sim.data.qvel.flat[:14],
            ])

        # if self._expose_body_coms is not None:
        #     for name in self._expose_body_coms:
        #         com = self.get_body_com(name)
        #         if name not in self._body_com_indices:
        #             indices = range(len(obs), len(obs) + len(com))
        #             self._body_com_indices[name] = indices
        #         obs = np.concatenate([obs, com])
        #
        # if self._expose_body_comvels is not None:
        #     for name in self._expose_body_comvels:
        #         comvel = self.get_body_comvel(name)
        #         if name not in self._body_comvel_indices:
        #             indices = range(len(obs), len(obs) + len(comvel))
        #             self._body_comvel_indices[name] = indices
        #         obs = np.concatenate([obs, comvel])

        if self._expose_foot_sensors:
            obs = np.concatenate([obs, self.sim.data.sensordata])
        return obs

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(size=self.sim.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + np.random.randn(self.sim.model.nv) * .1

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2.5

    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.sim.data.qpos[3:7]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    # just to make sure
    def seed(self, seed_idx=None):
        super().seed(seed_idx)
        self.action_space.np_random.seed(seed_idx)
        random.seed(seed_idx)
        np.random.seed(seed_idx)

    def set_init_state(self, state: np.ndarray):
        assert len(state) == 29 # self._expose_all_qpos == True
        qpos = state[:15]
        qvel = state[15:]
        new_qpos = self.sim.data.qpos.copy()
        new_qpos[:15] = qpos
        new_qvel = self.sim.data.qvel.copy()
        new_qvel[:14] = qvel
        self.set_state(new_qpos, new_qvel)
        return self._get_obs()

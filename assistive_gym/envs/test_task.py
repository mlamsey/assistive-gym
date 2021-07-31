import os
from gym import spaces
import numpy as np
import pybullet as p
from .env import AssistiveEnv

class TestEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(TestEnv, self).__init__(robot=robot, human=human, task='test', obs_robot_len=21, obs_human_len=(19 if human else 0))

    def step(self, action):
        # Compute action (do nothing)
        n_dof = len(action)
        # action = np.zeros(n_dof).ravel()
        # Execute action
        self.take_step(action)

        # Get observation
        obs = self._get_obs()
        return obs

    def _get_obs(self, agent=None):
        observation = np.zeros(20).ravel() # can be anything
        reward = 0
        done = False
        info = 'placeholder'
        return observation, reward, done, info

    def reset(self):
        super(TestEnv, self).reset()
        self.build_assistive_env()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs()

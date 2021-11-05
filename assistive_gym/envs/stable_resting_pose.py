import numpy as np
import pybullet as p
from .env import AssistiveEnv

from .agents.human import Human
from .agents.human import head_joints


class BasePoseEnv(AssistiveEnv):
    def __init__(self, human):
        super(BasePoseEnv, self).__init__(robot=None, human = human, task='pose_analysis', )

    def step(self, action):
        n_dof = len(action)
        action = np.zeros(n_dof).ravel()
        self.take_step(action)
        return self._get_obs()

    def _get_obs(self, agent=None):
        observation = None
        reward = 0
        done = False
        info = 'n/a'
        return observation, reward, done, info

    def reset(self):
        super(BasePoseEnv, self).reset()
        self.build_assistive_env()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)


class StableRestingPoseEnv(BasePoseEnv):
    def __init__(self):
        super(BasePoseEnv, self).__init__(human=Human(controllable_joint_indices=head_joints, controllable=False))


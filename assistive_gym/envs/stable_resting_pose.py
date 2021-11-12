import numpy as np
import pybullet as p
from .env import AssistiveEnv

import math

from .agents.human import Human
from .agents.human import head_joints


def configure_human(human):
    human.impairment = None
    # human.set_all_joints_stiffness(0.02)
    human.set_whole_body_frictions(lateral_friction=5., spinning_friction=1., rolling_friction=1.)

    joint_pos = default_sitting_pose(human)
    # joint_pos = []
    human.setup_joints(joint_pos, use_static_joints=True, reactive_force=None)

    start_pos = [0, 0, 0.425]
    # start_orient = [math.sqrt(2)/2., 0., 0., -math.sqrt(2)/2.]
    start_orient = [0.5, 0, 0, -0.8660254]
    human.set_base_pos_orient(start_pos, start_orient)
    human.set_on_ground()

    joint_i = [pose[0] for pose in joint_pos]
    joint_th = [pose[1] for pose in joint_pos]
    joint_gains = [0.5] * len(joint_i)
    forces = [10.] * len(joint_i)
    human.control(joint_i, joint_th, joint_gains, forces)

# def set_joint_stiffnesses(human):
#     human.set_joint_stiffness(human.j_)


def default_sitting_pose(human):
    # Arms
    joint_pos = [(human.j_right_shoulder_x, 30.),
                 (human.j_left_shoulder_x, -30.),
                 (human.j_right_shoulder_y, 60.),
                 (human.j_left_shoulder_y, 60.),
                 (human.j_right_elbow, -110.),
                 (human.j_left_elbow, -110.)]

    # Legs
    joint_pos += [(human.j_right_knee, 45.),
                  (human.j_left_knee, 45.),
                  (human.j_right_hip_x, -45.),
                  (human.j_left_hip_x, -45.)]

    # Torso
    joint_pos += [(human.j_waist_x, 0.)]
    return joint_pos


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
        self.build_assistive_env(fixed_human_base=False)

        # Human init
        configure_human(self.human)

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)


class StableRestingPoseEnv(BasePoseEnv):
    def __init__(self):
        human = Human(controllable_joint_indices=head_joints, controllable=True)
        super(BasePoseEnv, self).__init__(human=human)


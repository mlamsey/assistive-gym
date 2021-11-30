import numpy as np
import pybullet as p
from .env import AssistiveEnv

import math
import os

from .agents.human import Human
from .agents.human import right_arm_joints, left_arm_joints, torso_joints, head_joints
controllable_joints = right_arm_joints + left_arm_joints + torso_joints + head_joints


def configure_human(human):
    human.impairment = None
    # human.set_all_joints_stiffness(0.02)
    human.set_whole_body_frictions(lateral_friction=50., spinning_friction=10., rolling_friction=10.)

    joint_pos = default_sitting_pose(human)
    human.setup_joints(joint_pos, use_static_joints=False, reactive_force=None)

    start_pos = [0, 0.05, 0.875]
    start_orient = [0, 0, 0, 1]
    human.set_base_pos_orient(start_pos, start_orient)
    # human.set_on_ground()

    joint_i = [pose[0] for pose in joint_pos]
    joint_th = [pose[1] for pose in joint_pos]
    joint_gains = [10.] * len(joint_i)
    # forces = [50.] * len(joint_i)
    forces = [1.] * len(joint_i)

    # tweak joint control
    for i in range(len(joint_gains)):
        if i not in controllable_joints:
            joint_gains[i] = 0.
            forces[i] = 0.

    human.control(joint_i, joint_th, joint_gains, forces)

# def set_joint_stiffnesses(human):
#     human.set_joint_stiffness(human.j_)


def default_sitting_pose(human):
    # Arms
    joint_pos = [(human.j_right_shoulder_x, 30.),
                 (human.j_left_shoulder_x, -30.),
                 (human.j_right_shoulder_y, 0.),
                 (human.j_left_shoulder_y, 0.),
                 (human.j_right_elbow, -90.),
                 (human.j_left_elbow, -90.)]

    # Legs
    joint_pos += [(human.j_right_knee, 90.),
                  (human.j_left_knee, 90.),
                  (human.j_right_hip_x, -90.),
                  (human.j_left_hip_x, -90.)]

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
        observation = self._get_obs()
        self.steps += 1
        reward = 0
        done = False if self.steps < self.max_steps else True
        info = {"n/a": 'n/a'}  # must be a dict
        return observation, reward, done, info

    def _get_obs(self, agent=None):
        # observation = self.human.get_joint_angles()
        observation = self.human.get_pos_orient(self.human.head)
        return observation

    def reset(self):
        super(BasePoseEnv, self).reset()
        self.build_assistive_env(fixed_human_base=False)
        plane_path = os.path.join(self.directory, "primitives", "plane_chair.urdf")
        plane_chair = p.loadURDF(plane_path, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)

        # Human init
        configure_human(self.human)

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        return self._get_obs()


class StableRestingPoseEnv(BasePoseEnv):
    def __init__(self):
        human = Human(controllable_joint_indices=controllable_joints, controllable=True)
        super(BasePoseEnv, self).__init__(human=human)
        self.steps = 0
        self.max_steps = 100


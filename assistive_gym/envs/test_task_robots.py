from .test_task import TestEnv

# Robot Import
from .agents.stretch import Stretch
from .agents.pr2 import PR2

# Human Import
from .agents.human import Human
from .agents import human

# Robot Configuration
robot_arm = 'left'

# Human Configuration
# human_controllable_joint_indices = human.right_arm_joints

class TestPR2Env(TestEnv):
    def __init__(self):
        super(TestPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class TestStretchEnv(TestEnv):
    def __init__(self):
        super(TestStretchEnv, self).__init__(robot=Stretch('wheel_' + robot_arm), human=None)
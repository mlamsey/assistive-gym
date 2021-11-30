import gym, sys, argparse
import numpy as np
from .learn import make_env

from .envs.agents import human as human

import matplotlib.pyplot as plt
import math

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def get_action(env, coop):
    n_dof = env.action_space.shape
    return np.zeros(n_dof).ravel()

def viewer(env_name):
    # config
    total_episodes = 1
    n_episodes = 0

    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)

    joint_history = []
    while n_episodes < total_episodes:
        simulation_finished = False
        env.render()
        observation = env.reset()
        action = get_action(env, coop)

        # Output observation and action sizes
        if coop:
            print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))
        else:
            print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        while not simulation_finished:
            observation, reward, simulation_finished, info = env.step(get_action(env, coop))
            joint_history.append(observation)
            if coop:
                # IDK what this does
                done = done['__all__']

        # waist_index = human.torso_joints[3]
        # waist_history = [obs[waist_index] for obs in joint_history]
        head_pos_history = [j[0] for j in joint_history]

        n_episodes += 1
        plt.plot(head_pos_history)
        # plt.ylim(-math.pi/2, math.pi/2)
        plt.ylim(-1., 1.5)
        plt.legend(["Head X", "Head Y", "Head Z"])
        plt.grid(color="lightgray")
        plt.ylabel("Position (m)")
        plt.xlabel("Simulation Step")
        plt.show()


if __name__ == "__main__":
    default_task = 'TestStretch-v1'
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default=default_task,
                        help='Environment to test (default: (' + default_task + ')')
    args = parser.parse_args()

    viewer(args.env)
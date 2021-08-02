import gym, sys, argparse
import numpy as np
from .learn import make_env

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def get_action(env, coop):
    n_dof = env.action_space.shape
    return np.zeros(n_dof).ravel()

def viewer(env_name):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)

    while True:
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
            observation, reward, done, info = env.step(get_action(env, coop))
            if coop:
                # IDK what this does
                done = done['__all__']

if __name__ == "__main__":
    default_task = 'TestStretch-v1'
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default=default_task,
                        help='Environment to test (default: (' + default_task + ')')
    args = parser.parse_args()

    viewer(args.env)
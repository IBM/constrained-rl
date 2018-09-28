# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import tensorflow as tf
import numpy as np
from .constraint_network import ConstraintNetwork
from ceres.networks import NetworkSaverMLP

class ConstraintNetworkMLP(ConstraintNetwork, NetworkSaverMLP):
    '''
    Constraint network with MLP and save/restore functions
    '''

    def __init__(self, observation_space, action_space, config):
        NetworkSaverMLP.__init__(self, network_id='cnet')
        ConstraintNetwork.__init__(self, observation_space, action_space, config)

    def build_model(self):
        return NetworkSaverMLP.build_model(self, self.observation, self.n_outputs,
                                           self.config.mlp_hidden_layers,
                                           self.initializer,
                                           self.activation_common)


def play_cnet():
    '''
    Load a trained constrained network and print constraint predictions from random states
    '''
    from ceres.tools import ExtraArgs
    from ceres.envs import ConstrainedEnv
    import gym
    extra_args = ExtraArgs(ignore_max_timesteps=True, ignore_max_iterations=True, ignore_max_episodes=True)
    assert len(extra_args.env_id) > 0, 'Required argument --env_id'
    env = gym.make(extra_args.env_id)
    assert isinstance(env.unwrapped, ConstrainedEnv), 'The chosen environment {0} does not support constraints'.format(extra_args.env_id)
    assert len(extra_args.trained_cnet) > 0, 'Required argument --trained_cnet'
    cnet_config = ConstraintConfig.from_backup(extra_args.trained_cnet)
    cnet = ConstraintNetworkMLP(env.observation_space, env.action_space, cnet_config)

    n_obs = env.observation_space.shape[0]
    def random_state():
        pass

    cmd_str = '[r/Return]: random state, [q]: quit, otherwise input comma-separated state of length {0}\n'.format(n_obs)
    with tf.Session() as sess:
        def predict_constraints(state):
            observation = [state]
            ineq_mat, ineq_vec = sess.run([cnet.ineq_mat, cnet.ineq_vec], feed_dict={cnet.observation: observation})
            ineq_mat = ineq_mat[0]
            ineq_vec = ineq_vec[0]
            return ineq_mat, ineq_vec

        def predict_and_print_constraints(state):
            print('Input state: {0}'.format(state))
            ineq_mat, ineq_vec = predict_constraints(state)
            env.unwrapped.print_ineq(ineq_mat=ineq_mat, ineq_vec=ineq_vec)

        cnet.restore_model(extra_args.trained_cnet, session=sess)
        while True:
            cmd = input(cmd_str)
            if cmd == 'q':
                break
            elif (cmd == 'r') or (cmd == ''):
                state = np.random.rand(n_obs)
                predict_and_print_constraints(state)
            else:
                try:
                    state = list(map(float, cmd.split(',')))
                    assert len(state) == n_obs, 'input state {0} is of length {1}, expected {2}'.format(state, len(state), n_obs)
                    predict_and_print_constraints(state)
                except Exception as e:
                    print('Invalid command \'{0}\': {1}'.format(cmd, str(e)))


if __name__ == '__main__':
    play_cnet()


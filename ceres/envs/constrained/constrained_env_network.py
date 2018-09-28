# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np
import tensorflow as tf
from ceres.constraints import ConstraintNetworkMLP, ConstraintConfig
from .constrained_env import ConstrainedEnv

class ConstrainedEnvNetwork(ConstrainedEnv):
    '''
    Environment with constraints predicted by a constraint network.
    Make sure to call init_constraint_prediction before running it.
    '''

    def __init__(self, *args, **kwargs):
        super(ConstrainedEnvNetwork, self).__init__(*args, **kwargs)
        self.ineq_mat_params, self.ineq_vec_params = None, None
        self.is_initialized_constraint_prediction = False

    def init_constraint_prediction(self, cnet, session=None):
        '''
        Initialize constraint network, either from a ConstraintNetworkMLP object or a path to a trained network backup
        '''
        if session is None:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.cnet_session = tf.Session(config=tf_config)
        else:
            self.cnet_session = session

        if type(cnet) == str: # path to constraint network backup
            cnet_config = ConstraintConfig.from_backup(cnet)
            self.cnet = ConstraintNetworkMLP(self.observation_space, self.action_space, cnet_config)
            self.cnet.restore_model(cnet, session=self.cnet_session)
        else:
            assert isinstance(cnet, ConstraintNetworkMLP)
            self.cnet = cnet
        self.is_initialized_constraint_prediction = True

    def update_ineq_matrices(self, state):
        '''
        Predict ineq matrices by passing an input state through the constraint network and compute auxiliary variables
        '''
        assert self.is_initialized_constraint_prediction, 'Constraint prediction is not initialized: call init_constraint_prediction'
        feed_dict = {self.cnet.observation: np.expand_dims(state, axis=0)}
        cnet_outputs = [self.cnet.ineq_mat, self.cnet.ineq_vec, self.cnet.ineq_mat_params, self.cnet.ineq_vec_params, self.cnet.interior_point]
        ineq_outputs = self.cnet_session.run(cnet_outputs, feed_dict=feed_dict)
        self.ineq_mat, self.ineq_vec, self.ineq_mat_params, self.ineq_vec_params, self.ineq_interior_point = [_v[0] for _v in ineq_outputs]
        self.ineq_interior_point_flat = np.squeeze(self.ineq_interior_point)
        # Check constraint prediction validity
        if not (np.all(np.isfinite(self.ineq_mat)) and np.all(np.isfinite(self.ineq_vec))):
            error_str = 'Invalid inequality matrices: make sure you are using a recent version of Tensorflow' # In some versions, tf.cos and tf.sin can output infinity for large inputs
            print('Inequality parameters')
            print(self.ineq_mat_params)
            print(self.ineq_vec_params)
            print('Processed constraints')
            self.print_ineq()
            raise ValueError(error_str)

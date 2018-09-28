# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from .constrained_env import ConstrainedEnv

class ConstrainedEnvFixed(ConstrainedEnv):
    '''
    A base class for constrained environments with fixed constraints.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ineq_matrices()

    def update_ineq_matrices(self, state):
        '''
        This purposely does nothing since ineq matrices only need to be built once
        '''
        pass

    def init_ineq_matrices(self):
        '''
        Define fixed ineq matrices here 
        '''
        raise NotImplementedError('Implement init_ineq_matrices in subclass {0}'.format(type(self)))


# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from baselines.ppo1.mlp_policy import MlpPolicy
from ceres.networks import NetworkSaverMLP

class MlpPolicySaver(MlpPolicy, NetworkSaverMLP):

    '''
    Baselines MlpPolicy with save / restore functions
    '''

    def __init__(self, name, *args, session=None, **kwargs):
        MlpPolicy.__init__(self, name, *args, **kwargs)
        NetworkSaverMLP.__init__(self, network_id=name)

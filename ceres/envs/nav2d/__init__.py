# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from gym.envs.registration import register
from .nav2d_ceres import *

# We list all used Nav2d environments here and call them by '<environment class name>-v0'
gwenv_list = []
gwenv_list.append('Nav2dPosCeres')
gwenv_list.append('Nav2dPosFixedMazeCeres')
gwenv_list.append('Nav2dPosRandomHolesCeres')
gwenv_list.append('Nav2dForceCeres')
gwenv_list.append('Nav2dForceFixedMazeCeres')
gwenv_list.append('Nav2dForceRandomHolesCeres')
gwenv_list.append('Nav2dPosFixedMazeCeres5N')
gwenv_list.append('Nav2dPosRandomHolesCeres5N')
gwenv_list.append('Nav2dForceRandomHolesCeres5N')

for gwenv in gwenv_list:
    env = locals()[gwenv]
    register(
        id='{0}-v0'.format(gwenv),
        entry_point='ceres.envs.nav2d:{0}'.format(gwenv),
        max_episode_steps=env.max_episode_steps,
        reward_threshold=100.0,
    )

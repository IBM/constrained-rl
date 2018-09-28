# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import gym
import numpy as np

class SnapshotInfo(object):
    '''
    A simple class to store snapshot metadata
    '''
    __slots__ = ['i_trajectory', 'i_state', 'action_level', 'action_weight']
    def __init__(self, i_trajectory=None, i_state=None, action_level=None, action_weight=None):
        self.i_trajectory = i_trajectory
        self.i_state = i_state
        self.action_level = action_level
        self.action_weight = action_weight


class ResetterEnv(gym.Env):
    '''
    A base class to uniformize trajectory snapshotting and restoring.
    Some functions must be implemented either by the base RL environment (e.g., Nav2d)
    or by a task-specific resetter environment (e.g., ResetterEnvCeres)
    '''
    max_reference_steps_per_episode = -1

    def __init__(self):
        super(ResetterEnv, self).__init__()
        self.init_reference()

    def init_reference(self):
        '''
        Define the type of reference snapshots to restore the environment to
        '''
        self._init_reference_parameters()
        self._init_reference_trajectories()

    def get_random_reference_index(self):
        '''
        Pick a random trajectory, then a random state within that trajectory
        '''
        i_traj  = np.random.randint(0, len(self.reference_trajectories))
        i_state = np.random.randint(0, len(self.reference_trajectories[i_traj]))
        return i_traj, i_state

    def get_random_reference_snapshot(self):
        '''
        Get a random snapshot with the associated metadata
        '''
        i_traj, i_state = self.get_random_reference_index()
        snapshot = self.reference_trajectories[i_traj][i_state].snapshot
        assert snapshot is not None
        snapshot_info = SnapshotInfo(i_trajectory=i_traj, i_state=i_state)
        return snapshot, snapshot_info

    def reset_random(self):
        raise NotImplementedError('Implement this in environment class {0}'.format(type(self)))

    def reset_and_restore(self, snapshot):
        raise NotImplementedError('Implement this in environment class {0}'.format(type(self)))

    def _init_reference_parameters(self):
        raise NotImplementedError('Implement this in resetter class {0}'.format(type(self)))

    def _init_reference_trajectories(self):
        raise NotImplementedError('Implement this in resetter class {0}'.format(type(self)))


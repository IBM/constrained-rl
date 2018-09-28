# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np
from .resetter_env import ResetterEnv

class ResetterEnvCeres(ResetterEnv):
    '''
    Resetter base environment implementing functions required by the CERES logic,
    e.g., reset from trajectory midpoints, remove identified demonstrations, etc.
    '''
    max_reference_trajectories = -1 # Set this to a non negative value in a child class

    def _init_reference_parameters(self):
        '''
        The maximum number of trajectories to keep is application-specific
        '''
        assert self.max_reference_trajectories >= 0, 'max_reference_trajectories must be non negative: set it in class {0}'.format(type(self))

    def _init_reference_trajectories(self):
        '''
        Setup empty reference trajectory list
        '''
        self.reference_trajectories = []
        self.reset_count_per_trajectory = [] # these values are incremented outside the environment
        self.filter_reset_per_trajectory = [] # check for difference before incrementing

    def add_reference_trajectory(self, trajectory):
        '''
        Add a new reference trajectory and generate new metadata
        '''
        if len(self.reference_trajectories) < self.max_reference_trajectories:
            self.reference_trajectories.append(trajectory)
            self.reset_count_per_trajectory.append(0)
            self.filter_reset_per_trajectory.append(None)
        else:
            pass # Skip if full, add other behaviors in the future

    def get_random_reference_index(self):
        '''
        Return trajectory midpoints
        '''
        assert len(self.reference_trajectories) > 0, 'No active trajectory'
        i_traj = np.random.randint(0, len(self.reference_trajectories))
        i_state = self.get_reference_trajectory_midpoint(i_traj)
        return i_traj, i_state

    def get_reference_trajectory(self, i_traj):
        return self.reference_trajectories[i_traj]

    def get_reference_trajectory_midpoint(self, i_traj):
        i_state = self.reference_trajectories[i_traj].get_midpoint()
        return i_state

    def remove_empty_trajectories(self):
        '''
        Remove trajectories that have no active state
        '''
        i_traj_active = []
        for (i, traj) in enumerate(self.reference_trajectories):
            if traj.length_active > 0:
                i_traj_active.append(i)
        n_remove = len(self.reference_trajectories) - len(i_traj_active)
        self.reference_trajectories = [self.reference_trajectories[i] for i in i_traj_active]
        self.reset_count_per_trajectory = [self.reset_count_per_trajectory[i] for i in i_traj_active]
        self.filter_reset_per_trajectory = [self.filter_reset_per_trajectory[i] for i in i_traj_active]
        return n_remove

    def check_remove_traj(self, traj):
        '''
        Check if the trajectory can be removed based on the number of active snapshots
        '''
        do_remove_traj = traj.length_active == 0
        if traj.length_active == 1: # remove also if the only demonstration left is already classified
            demonstration = traj.get_demonstration(traj.active_demonstrations[0])
            do_remove_traj = demonstration.test_is_classified()
            if do_remove_traj:
                #print('Final demonstration is already classified as {0}'.format(demonstration.action_indicator))
                pass
            else:
                traj.do_reset_after_last_active = True
        return do_remove_traj

    def update_reference_trajectory(self, i_traj, is_resized, remove_if_emptied=False):
        '''
        Reset trajectory metadata and remove if applicable
        '''
        traj = self.reference_trajectories[i_traj]
        if is_resized:
            self.reset_count_per_trajectory[i_traj] = 0
            self.filter_reset_per_trajectory[i_traj] = None
        if remove_if_emptied:
            if self.check_remove_traj(traj):
                self.reference_trajectories.pop(i_traj)
                self.reset_count_per_trajectory.pop(i_traj)
                self.filter_reset_per_trajectory.pop(i_traj)

    def get_reference_trajectory_active_demonstrations_from(self, i_traj, begin, remove_demonstrations=False, return_copy=True, remove_if_emptied=False):
        '''
        Get a sub-trajectory starting from a given active demonstration and update metadata
        '''
        traj = self.reference_trajectories[i_traj]
        subtraj, is_resized = traj.get_active_demonstrations_from(begin, remove_demonstrations=remove_demonstrations, return_copy=return_copy)
        self.update_reference_trajectory(i_traj, is_resized, remove_if_emptied=remove_if_emptied)
        return subtraj

    def get_reference_trajectory_active_demonstrations_to(self, i_traj, end, remove_demonstrations=False, return_copy=True, remove_if_emptied=False):
        '''
        Get a sub-trajectory up to a given active demonstration and update metadata
        '''
        traj = self.reference_trajectories[i_traj]
        subtraj, is_resized = traj.get_active_demonstrations_to(end, remove_demonstrations=remove_demonstrations, return_copy=return_copy)
        self.update_reference_trajectory(i_traj, is_resized, remove_if_emptied=remove_if_emptied)
        return subtraj

    def get_reference_trajectory_demonstration(self, i_traj, i_state, return_copy=True):
        '''
        Get a chosen demonstration within a chosen trajectory, or copy thereof for separate processing
        '''
        traj = self.reference_trajectories[i_traj]
        demonstration = traj.get_demonstration(i_state, return_copy=return_copy)
        return demonstration

    def increment_trajectory_reset_count(self, i_traj, increment=1, increment_reset_count_on_change=None):
        '''
        Increment the number of times a trajectory has been reset too, unless a reset criterion is set
        '''
        if increment_reset_count_on_change is not None:
            if increment_reset_count_on_change == self.filter_reset_per_trajectory[i_traj]:
                return
            else:
                self.filter_reset_per_trajectory[i_traj] = increment_reset_count_on_change
        self.reset_count_per_trajectory[i_traj] += increment

    def get_trajectory_reset_count(self, i_traj):
        return self.reset_count_per_trajectory[i_traj]


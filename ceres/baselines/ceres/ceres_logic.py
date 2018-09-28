# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

class CeresLogic(object):
    '''
    Identify demonstrations as positive or negative
    based on trajectory success, failure or possible recovery.
    '''

    def __init__(self, env, constraint_demonstration_buffer, extra_args):
        self.env = env
        self.constraint_demonstration_buffer = constraint_demonstration_buffer
        self.extra_args = extra_args
 
    def process_trajectory(self,
                           sampled_demonstration_traj, traj_failure, traj_success,
                           uncertain_demonstration_trajectories,
                           recovery_info, is_direct_policy,
                           remove_reference_trajectory_if_emptied=False,
                           increment_reset_count_on_change=None):
        '''
        Main function: sorts newly sampled demonstrations into positive or negative
        and stores unidentified demonstrations for further transfer between direct and recovery agents.
        '''
        n_new_demonstrations = 0
        if traj_success:
            n_new_demonstrations += self.do_success_common(sampled_demonstration_traj, verbose=False)
            if is_direct_policy:
                pass # Do nothing special
            else:
                n_new_demonstrations += self.do_success_or_survive_recovery(recovery_info, remove_reference_trajectory_if_emptied=remove_reference_trajectory_if_emptied, verbose=False)
        else:
            # We count how many steps are such that we are still alive after max_recovery_steps actions
            n_survivable_states = sampled_demonstration_traj.length_active - self.env.unwrapped.max_recovery_steps
            if not traj_failure:
                n_survivable_states += 1
            # Good demonstrations are those leading to such survivable states
            n_positive_demonstrations = n_survivable_states - 1
            # Clip w.r.t. zero only after computing number of positive demonstrations, in case the trajectory is shorter than the target number of recovery steps
            n_survivable_states = max(0, n_survivable_states)
            n_positive_demonstrations = max(0, n_positive_demonstrations)
            n_new_demonstrations += self.do_add_survivable_as_positive_common(sampled_demonstration_traj, n_positive_demonstrations, verbose=False)
            if traj_failure:
                n_new_demonstrations += self.do_failure_common(sampled_demonstration_traj, verbose=False)
            if is_direct_policy:
                n_new_demonstrations += self.do_uncertain_direct(uncertain_demonstration_trajectories, sampled_demonstration_traj, n_survivable_states, verbose=False)
            else:
                if n_survivable_states > 0: # Agent survived for max_recovery_steps
                    n_new_demonstrations += self.do_success_or_survive_recovery(recovery_info, remove_reference_trajectory_if_emptied=remove_reference_trajectory_if_emptied, verbose=False)
                else:
                    n_new_demonstrations += self.do_failed_recovery(recovery_info, remove_reference_trajectory_if_emptied=remove_reference_trajectory_if_emptied, increment_reset_count_on_change=increment_reset_count_on_change, verbose=False)
        return n_new_demonstrations
    
    
    def get_info(self, demonstration_traj):
        n_negative_demonstrations = len(self.constraint_demonstration_buffer.negative_demonstrations)
        n_positive_demonstrations = len(self.constraint_demonstration_buffer.positive_demonstrations)
        active_demonstrations = [_e for _e in demonstration_traj.active_demonstrations]
        return n_negative_demonstrations, n_positive_demonstrations, active_demonstrations
    
    def build_str_before_after(self, negative_before, negative_after, positive_before, positive_after, active_before, active_after):
        str_negative = '    negative {0} -> {1}: {2}'.format(negative_before, negative_after, negative_after-negative_before)
        str_positive = '    positive {0} -> {1}: {2}'.format(positive_before, positive_after, positive_after-positive_before)
        n_active_before = active_before[1] - active_before[0]
        n_active_after = active_after[1] - active_after[0]
        str_active = '    active {0} {1} -> {2} {3}: {4}'.format(n_active_before, active_before, n_active_after, active_after, n_active_after-n_active_before)
        str_before_after = '\n'.join([str_negative, str_positive, str_active])
        return str_before_after
    
    def do_success_common(self, sampled_demonstration_traj, verbose=False):
        '''
        Add all elements of sampled trajectory as positive
        '''
        n_new_demonstrations = 0
        if verbose:
            negative_before, positive_before, active_before = self.get_info(sampled_demonstration_traj)
        active_demonstrations = sampled_demonstration_traj.active_demonstrations
        for _i in range(active_demonstrations[0], active_demonstrations[1]):
            _e = sampled_demonstration_traj[_i]
            _e.action_indicator = 1.
            self.constraint_demonstration_buffer.add_demonstration(_e)
            n_new_demonstrations += 1
        if verbose:
            negative_after, positive_after, active_after = self.get_info(sampled_demonstration_traj)
            print('Success common: \n{0}'.format(self.build_str_before_after(negative_before, negative_after, positive_before, positive_after, active_before, active_after)))
        return n_new_demonstrations
    
    def do_failure_common(self, sampled_demonstration_traj, verbose=False):
        '''
        Add last demonstration as negative demonstration
        Do not remove from sampled trajectory since we also want to try recovering from the last state
        '''
        if verbose:
            negative_before, positive_before, active_before = self.get_info(sampled_demonstration_traj)
        last_sampled_demonstration = sampled_demonstration_traj.get_demonstration(sampled_demonstration_traj.active_demonstrations[1]-1, return_copy=False)
        last_sampled_demonstration.action_indicator = 0.
        self.constraint_demonstration_buffer.add_demonstration(last_sampled_demonstration)
        n_new_demonstrations = 1
        if verbose:
            negative_after, positive_after, active_after = self.get_info(sampled_demonstration_traj)
            print('Failure common: \n{0}'.format(self.build_str_before_after(negative_before, negative_after, positive_before, positive_after, active_before, active_after)))
        return n_new_demonstrations
    
    def do_add_survivable_as_positive_common(self, sampled_demonstration_traj, n_positive_demonstrations, verbose=False):
        '''
        Add as positive demonstrations such that the agent is still alive after max_recovery_steps
        '''
        n_new_demonstrations = 0
        if n_positive_demonstrations == 0:
            return n_new_demonstrations # do nothing
        if verbose:
            negative_before, positive_before, active_before = self.get_info(sampled_demonstration_traj)
        sampled_demonstrations, is_resized = sampled_demonstration_traj.get_active_demonstrations_to(n_positive_demonstrations, remove_demonstrations=True, return_copy=False)
        for _e in sampled_demonstrations:
            _e.action_indicator = 1.
            self.constraint_demonstration_buffer.add_demonstration(_e)
            n_new_demonstrations += 1
        if verbose:
            negative_after, positive_after, active_after = self.get_info(sampled_demonstration_traj)
            print('Add survivable as positive common: \n{0}'.format(self.build_str_before_after(negative_before, negative_after, positive_before, positive_after, active_before, active_after)))
        return n_new_demonstrations
    
    def do_success_or_survive_recovery(self, recovery_info, remove_reference_trajectory_if_emptied=False, verbose=False):
        '''
        Set all demonstrations leading to the starting snapshot as positive
        '''
        n_new_demonstrations = 0
        if recovery_info is None: # this is None when the environment is reset to a random state
            return n_new_demonstrations
        if verbose:
            reference_demonstration_traj = self.env.unwrapped.get_reference_trajectory(recovery_info.i_trajectory)
            i_midpoint_before = recovery_info.i_state
            assert reference_demonstration_traj.get_midpoint() == i_midpoint_before
            negative_before, positive_before, active_before = self.get_info(reference_demonstration_traj)
        reference_demonstrations_before_midpoint = self.env.unwrapped.get_reference_trajectory_active_demonstrations_to(recovery_info.i_trajectory, recovery_info.i_state, remove_demonstrations=True, return_copy=False, remove_if_emptied=remove_reference_trajectory_if_emptied)
        for _e in reference_demonstrations_before_midpoint:
            _e.action_indicator = 1.
            self.constraint_demonstration_buffer.add_demonstration(_e)
            n_new_demonstrations += 1
        if verbose:
            negative_after, positive_after, active_after = self.get_info(reference_demonstration_traj)
            if reference_demonstration_traj.length_active > 0:
                i_midpoint_after = reference_demonstration_traj.get_midpoint()
            else:
                i_midpoint_after = '(empty)'
            str_midpoint = '    midpoint {0} -> {1}'.format(i_midpoint_before, i_midpoint_after)
            print('Do success or survive recovery: \n{0}\n{1}'.format(self.build_str_before_after(negative_before, negative_after, positive_before, positive_after, active_before, active_after), str_midpoint))
        return n_new_demonstrations
    
    def do_failed_recovery(self, recovery_info, remove_reference_trajectory_if_emptied=False, increment_reset_count_on_change=None, verbose=True):
        '''
        Count how many times recovery was attempted from a given snapshot.
        Store as negative if number of failures exceeds max_recovery_attempts
        '''
        n_new_demonstrations = 0
        if recovery_info is None:
            return n_new_demonstrations
        # Only try for a given number of attempts
        self.env.unwrapped.increment_trajectory_reset_count(recovery_info.i_trajectory, increment_reset_count_on_change=increment_reset_count_on_change)
        if self.env.unwrapped.get_trajectory_reset_count(recovery_info.i_trajectory) >= self.extra_args.max_recovery_attempts:
            if verbose:
                reference_demonstration_traj = self.env.unwrapped.get_reference_trajectory(recovery_info.i_trajectory)
                i_midpoint_before = recovery_info.i_state
                assert reference_demonstration_traj.get_midpoint() == i_midpoint_before
                negative_before, positive_before, active_before = self.get_info(reference_demonstration_traj)
            # Add action leading to starting snapshot as negative but do not remove it from active
            if recovery_info.i_state > 0:
                demonstration = self.env.unwrapped.get_reference_trajectory_demonstration(recovery_info.i_trajectory, recovery_info.i_state-1, return_copy=False)
                demonstration.action_indicator = 0.
                self.constraint_demonstration_buffer.add_demonstration(demonstration)
                n_new_demonstrations += 1
            # Add all actions from starting snapshot as negative and remove
            reference_demonstrations_from_midpoint = self.env.unwrapped.get_reference_trajectory_active_demonstrations_from(recovery_info.i_trajectory, recovery_info.i_state, remove_demonstrations=True, return_copy=False, remove_if_emptied=remove_reference_trajectory_if_emptied)
            for _e in reference_demonstrations_from_midpoint:
                _e.action_indicator = 0.
                self.constraint_demonstration_buffer.add_demonstration(_e)
                n_new_demonstrations += 1
            if verbose:
                negative_after, positive_after, active_after = self.get_info(reference_demonstration_traj)
                if reference_demonstration_traj.length_active > 0:
                    i_midpoint_after = reference_demonstration_traj.get_midpoint()
                else:
                    i_midpoint_after = '(empty)'
                str_midpoint = '    midpoint {0} -> {1}'.format(i_midpoint_before, i_midpoint_after)
                print('Do failed recovery: \n{0}\n{1}'.format(self.build_str_before_after(negative_before, negative_after, positive_before, positive_after, active_before, active_after), str_midpoint))
        return n_new_demonstrations
    
    def do_uncertain_direct(self, uncertain_demonstration_trajectories, sampled_demonstration_traj, n_survivable_states, verbose=False):
        '''
        Store unidentified demonstrations as uncertain
        '''
        n_new_demonstrations = 0
        # All survivable states have been added before, add the rest as uncertain
        if verbose:
            negative_before, positive_before, active_before = self.get_info(sampled_demonstration_traj)
            n_uncertain_trajectories_before = len(uncertain_demonstration_trajectories)
        uncertain_demonstration_trajectories.append(sampled_demonstration_traj)
        if verbose:
            negative_after, positive_after, active_after = self.get_info(sampled_demonstration_traj)
            n_uncertain_trajectories_after = len(uncertain_demonstration_trajectories)
            str_uncertain_trajectories = '    uncertain trajectories {0} -> {1}: {2}'.format(n_uncertain_trajectories_before, n_uncertain_trajectories_after, n_uncertain_trajectories_after-n_uncertain_trajectories_before)
            print('Do uncertain direct: \n{0}\n{1}'.format(self.build_str_before_after(negative_before, negative_after, positive_before, positive_after, active_before, active_after), str_uncertain_trajectories))
        return n_new_demonstrations
   

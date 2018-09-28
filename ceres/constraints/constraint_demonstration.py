# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from collections import deque
import numpy as np
import os

class ConstraintDemonstration(object):
    '''
    Demonstrations used for constraint network training.
    Main attributes are state, action, and action indicator (+1 for positive, 0 for negative).
    For recovery training, also store environment snapshot.
    Action weight and level can also be stored, e.g., for prioritizing some actions over others if all cannot be properly separated (not implemented).
    '''

    __slots__  = ('snapshot', 'state', 'action', 'action_indicator', 'action_weight', 'action_level', 'is_terminal')

    def __init__(self, demonstration=None, snapshot=None, state=None, action=None, action_indicator=None, action_weight=None, action_level=None, is_terminal=None):
        if demonstration is None:
            if is_terminal is None:
                is_terminal = False
            self.is_terminal = is_terminal
            if self.is_terminal:
                assert action is None, 'A terminal demonstration should not have associated actions'
            self.is_terminal = is_terminal
            self.snapshot = snapshot
            self.state = state
            self.action = action
            self.action_indicator = action_indicator
            self.action_weight = action_weight
            self.action_level = action_level
        else:
            assert isinstance(demonstration, ConstraintDemonstration)
            for _v in (snapshot, state, action, action_indicator, action_weight, is_terminal):
                assert _v is None
            for _k in self.__slots__:
                setattr(self, _k, getattr(demonstration, _k))

    def __repr__(self):
        repr_str_list = []
        for _k in sorted(self.__slots__):
            repr_str_list.append('{0}: {1}'.format(_k, str(getattr(self, _k))))
        repr_str = '{{{0}}}'.format(', '.join(repr_str_list))
        return repr_str

    def test_is_classified(self):
        return self.action_indicator is not None

    def to_dict(self):
        d = {_k: getattr(self, _k) for _k in self.__slots__}
        return d


class ConstraintDemonstrationTrajectory(object):
    '''
    A sequence of demonstrations with functions for getting the trajectory's mid-point, sub-trajectories,
    and to disable recovery from demonstrations that are already sorted.
    '''

    def __init__(self, demonstrations=None, states=None, actions=None, action_indicators=None, action_weights=None):
        if demonstrations is None:
            demonstrations = []
            assert states is not None, 'Provide either demonstration list or individual content'
            for state, action, action_indicator, action_weight in zip(states, actions, action_indicators, action_weights):
                demonstration = ConstraintDemonstration(state=state, action=action, action_indicator=action_indicator, action_weight=action_weight)
                demonstrations.append(demonstration)
        else:
            for _e in demonstrations:
                assert isinstance(_e, ConstraintDemonstration)
        self.demonstrations = demonstrations 
        self.length_all = len(self.demonstrations)
        self.length_active = self.length_all
        if self.demonstrations[-1].is_terminal:
            self.length_active -= 1
        self.active_demonstrations = [0, self.length_active]
        self.do_reset_after_last_active = False

    def __len__(self):
        raise ValueError('Ambiguous length: use attributes .length_all or .length_active')

    def __getitem__(self, i):
        return self.demonstrations[i]

    def get_midpoint(self):
        '''
        Return the mid-point (rounded down) demonstration within the active trajectory.
        '''
        assert self.length_active > 0, 'Cannot get midpoint from empty demonstrations'
        i_state = self.active_demonstrations[0] + int(self.length_active / 2)
        if (self.length_active == 1) and self.do_reset_after_last_active:
            i_state += 1
            assert i_state < self.length_all
            demonstration = self.demonstrations[i_state]
            assert demonstration.is_terminal or demonstration.test_is_classified()
        return i_state

    def get_active_demonstrations_from(self, begin, remove_demonstrations=False, return_copy=True):
        '''
        Get a sub-trajectory starting from a given demonstration to the last active demonstration
        '''
        end = self.active_demonstrations[1]
        demonstrations = self.demonstrations[begin:end]
        if end > begin:
            if remove_demonstrations: # end earlier
                self.active_demonstrations[1] = begin
                self.length_active = self.active_demonstrations[1] - self.active_demonstrations[0]
            if return_copy:
                demonstrations = [ConstraintDemonstration(_e) for _e in demonstrations]
        is_resized = remove_demonstrations and (len(demonstrations) > 0)
        return demonstrations, is_resized

    def get_active_demonstrations_to(self, end, remove_demonstrations=False, return_copy=True):
        '''
        Get a sub-trajectory starting from the first active demonstration up to a given demonstration
        '''
        begin = self.active_demonstrations[0]
        demonstrations = self.demonstrations[begin:end]
        if end > begin:
            if remove_demonstrations: # start later
                self.active_demonstrations[0] = end
                self.length_active = self.active_demonstrations[1] - self.active_demonstrations[0]
            if return_copy:
                demonstrations = [ConstraintDemonstration(_e) for _e in demonstrations]
        is_resized = remove_demonstrations and (len(demonstrations) > 0)
        return demonstrations, is_resized

    def get_demonstration(self, i_state, return_copy=True):
        '''
        Get a chosen demonstration within the trajectory, or copy thereof for separate processing
        '''
        demonstration = self.demonstrations[i_state]
        if return_copy:
            demonstration = ConstraintDemonstration(demonstration)
        return demonstration

class ConstraintDemonstrationBuffer(object):
    '''
    Store positive and negative demonstrations, with functions for iterating over all and write / restore.
    '''

    buffer_basename = 'constraint_demonstration_buffer.npz'

    def __init__(self, max_size=None, positive_demonstrations=None, negative_demonstrations=None, even_split=True):
        self.even_split = bool(even_split)
        if max_size == None:
            self.max_size = None # in case we get numpy None
            self.max_size_per_buffer = None
        else:
            self.max_size = int(max_size)
            self.max_size_per_buffer = int(self.max_size/2) if self.even_split else self.max_size
        if positive_demonstrations is None:
            self.positive_demonstrations = deque(maxlen=self.max_size_per_buffer)
        else:
            self.positive_demonstrations = deque(positive_demonstrations, maxlen=self.max_size_per_buffer)
        if negative_demonstrations is None:
            self.negative_demonstrations = deque(maxlen=self.max_size_per_buffer)
        else:
            self.negative_demonstrations  = deque(negative_demonstrations, maxlen=self.max_size_per_buffer)

    def __len__(self):
        return len(self.positive_demonstrations) + len(self.negative_demonstrations)

    def add_demonstration(self, demonstration):
        '''
        Store a positive or negative demonstration into the appropriate buffer and remove old demonstrations if using a shared max size.
        '''
        assert isinstance(demonstration, ConstraintDemonstration)
        buffer_insert = self.positive_demonstrations if demonstration.action_indicator else self.negative_demonstrations
        buffer_insert.append(demonstration)
        if not self.even_split and max_size is not None: # check the total size
            if len(self.positive_demonstrations) + len(self.negative_demonstrations) > self.max_size:
                buffer_insert.popleft()

    def get_random_sample(self, action_indicator=None):
        '''
        Get a random demonstration from positivive or negative if specified, otherwise random.
        '''
        if action_indicator is None:
            action_indicator = np.random.rand() < 0.5
        buffer_sample = self.positive_demonstrations if action_indicator else self.negative_demonstrations
        assert len(buffer_sample) > 0, 'Cannot get sample from empty buffer'
        i_sample = np.random.randint(0, len(buffer_sample))
        demonstration = buffer_sample[i_sample]
        return demonstration

    def get_random_batch(self, batch_size, balance_positive_negative=False):
        '''
        Get a random batch of demonstrations, optionally balanced between positive and negative.
        '''
        if len(self.positive_demonstrations) == 0:
            if len(self.negative_demonstrations) == 0:
                raise ValueError('Cannot get batch from empty buffer')
            else:
                action_indicators = [0.] * batch_size
        else:
            if len(self.negative_demonstrations) == 0:
                action_indicators = [1.] * batch_size
            else:
                action_indicators = [np.random.rand() for _ in range(batch_size)]
                if balance_positive_negative: # same probability to get positive or negative
                    action_indicators = [_i < 0.5 for _i in action_indicators]
                else:
                    threshold_positive = (float(len(self.positive_demonstrations)) / (len(self.positive_demonstrations) + len(self.negative_demonstrations)))
                    action_indicators = [_i < threshold_positive for _i in action_indicators]
        batch = [self.get_random_sample(action_indicator=_i) for _i in action_indicators]
        return batch

    def iterate_epoch(self, batch_size):
        '''
        Return random batches, without balancing between positive and negative demonstrations.
        '''
        n_all = len(self.positive_demonstrations) + len(self.negative_demonstrations)
        i_shuffle = np.arange(n_all)
        np.random.shuffle(i_shuffle)
        n_batches = int(np.ceil(n_all/batch_size))
        for i_batch in range(n_batches):
            i_demonstrations_batch = i_shuffle[(i_batch * batch_size):((i_batch + 1) * batch_size)]
            demonstrations = [self.get_demonstration_from_global_index(_i) for _i in i_demonstrations_batch]
            yield demonstrations

    def get_demonstration_from_global_index(self, i):
        '''
        Return a demonstration from an index between 0 and n_positive + n_negative.
        '''
        if i < len(self.positive_demonstrations):
            return self.positive_demonstrations[i]
        else:
            return self.negative_demonstrations[i - len(self.positive_demonstrations)]

    def get_shuffle_indices(self, n_el, n_max):
        '''
        Build random indices ranging from 0 to n_el until a maximum size n_max is reached, repeating indices minimally.
        '''
        assert n_el > 0, 'Empty buffer'
        if n_el == n_max:
            i_shuffle = np.arange(n_el)
            np.random.shuffle(i_shuffle)
        else:
            n_shuffle = int(np.ceil(n_max / n_el))
            i_shuffle_list = []
            for _ in range(n_shuffle):
                i_shuffle_loc = np.arange(n_el)
                np.random.shuffle(i_shuffle_loc)
                i_shuffle_list.append(i_shuffle_loc)
            i_shuffle = np.concatenate(i_shuffle_list) # gather as single list
            i_shuffle = i_shuffle[:n_max] # resize to match size
        return i_shuffle

    def iterate_epoch_balanced(self, batch_size, n_max=0):
        '''
        Return batches such that every demonstration of the largest buffer appears exactly once,
        with demonstrations of the other buffer re-appearing a minimum number of times to balance batches
        '''
        n_positive = len(self.positive_demonstrations)
        n_negative = len(self.negative_demonstrations)
        if n_max == 0:
            n_max = max(n_positive, n_negative)

        i_shuffle_positive = self.get_shuffle_indices(n_positive, n_max)
        i_shuffle_negative  = self.get_shuffle_indices(n_negative, n_max)

        n_all = 2*n_max
        i_positive = 0
        i_negative = 0
        n_batches = int(np.ceil(n_all/batch_size))
        for i_batch in range(n_batches):
            demonstrations = batch_size * [None]
            for i_demonstration in range(batch_size):
                is_out_positive = i_positive >= len(i_shuffle_positive)
                is_out_negative = i_negative >= len(i_shuffle_negative)
                if is_out_positive and is_out_negative: # both buffers have been completely iterated through
                    demonstrations = demonstrations[:i_demonstration]
                    break
                elif (not is_out_positive) and (not is_out_negative): # both buffers still have demonstrations
                    do_add_positive = np.random.rand() < 0.5
                else: # only one buffer has not been completely iterated through
                    do_add_positive = not is_out_positive
                if do_add_positive:
                    demonstrations[i_demonstration] = self.positive_demonstrations[i_shuffle_positive[i_positive]]
                    i_positive += 1
                else:
                    demonstrations[i_demonstration] = self.negative_demonstrations[i_shuffle_negative[i_negative]]
                    i_negative += 1
            yield demonstrations

    def check_path_backup(self, path_backup):
        '''
        Build path to backup file if a directory is provided, otherwise check the extension of the provided file path.
        '''
        if os.path.isdir(path_backup):
            path_backup = os.path.join(path_backup, self.buffer_basename)
        elif not path_backup[-4:] == '.npz':
            path_backup = os.path.join(os.path.dirname(path_backup), self.buffer_basename)
        return path_backup

    def init_saver(self, path_backup):
        '''
        Build backup path
        '''
        self.path_backup = self.check_path_backup(path_backup)

    def write(self, path_backup=None, verbose=True):
        '''
        Save buffer to disk
        '''
        if path_backup is None:
            path_backup = self.path_backup
        buffer_as_dict = {'max_size': self.max_size,
                          'positive_demonstrations': [_e.to_dict() for _e in self.positive_demonstrations],
                          'negative_demonstrations': [_e.to_dict() for _e in self.negative_demonstrations],
                          'even_split': self.even_split}
        np.savez(path_backup, **buffer_as_dict)
        if verbose:
            print('Wrote buffer backup: {0} ({1} positive, {2} negative)'.format(path_backup, len(self.positive_demonstrations), len(self.negative_demonstrations)))

    @classmethod
    def from_backup(cls, path_backup, verbose=False):
        '''
        Build buffer from path
        '''
        demonstration_buffer = cls()
        demonstration_buffer.restore_buffer(path_backup, keep_size=False, keep_newest=True, verbose=verbose)
        return demonstration_buffer
    
    def restore_buffer(self, path_backup, keep_size=True, keep_newest=True, verbose=False):
        '''
        Restore buffer from path
        '''
        if keep_size:
            max_size = self.max_size
        path_backup = self.check_path_backup(path_backup)
        buffer_as_dict = np.load(path_backup)
        positive_demonstrations = [ConstraintDemonstration(**_d) for _d in buffer_as_dict['positive_demonstrations']]
        negative_demonstrations  = [ConstraintDemonstration(**_d) for _d in buffer_as_dict['negative_demonstrations']]
        self.__init__(max_size=buffer_as_dict['max_size'],
                      positive_demonstrations=positive_demonstrations,
                      negative_demonstrations=negative_demonstrations,
                      even_split=buffer_as_dict['even_split'])
        if keep_size:
            self.resize(max_size, keep_newest=keep_newest)
        if verbose:
            print('Load buffer backup: {0} ({1} positive, {2} negative)'.format(path_backup, len(self.positive_demonstrations), len(self.negative_demonstrations)))


    def resize(self, new_max_size, keep_newest=True):
        '''
        Resize buffer and remove excess demonstrations, new or old.
        '''
        if self.max_size == new_max_size:
            return
        self.max_size = new_max_size
        if self.even_split:
            self.max_size_per_buffer = int(self.max_size / 2)
            keep_positive = min(len(self.positive_demonstrations), self.max_size_per_buffer)
            keep_negative  = min(len(self.negative_demonstrations),  self.max_size_per_buffer)
        else:
            self.max_size_per_buffer = self.max_size
            total_excess = len(self.positive_demonstrations) + len(self.negative_demonstrations) - self.max_size
            ratio_excess = total_excess / self.max_size
            keep_positive = int(len(self.positive_demonstrations)/ratio_excess)
            keep_negative = int(len(self.negative_demonstrations)/ratio_excess)
        if keep_newest:
            last_positive  = len(self.positive_demonstrations)
            first_positive = last_positive - keep_positive
            last_negative  = len(self.negative_demonstrations)
            first_negative = last_negative - keep_negative
        else:
            first_positive = 0
            first_negative  = 0
            last_positive = first_positive + keep_positive
            last_negative  = first_negative  + keep_negative
        self.positive_demonstrations = deque([self.positive_demonstrations[_i] for _i in range(first_positive, last_positive)], maxlen=self.max_size)
        self.negative_demonstrations = deque([self.negative_demonstrations[_i] for _i in range(first_negative, last_negative)], maxlen=self.max_size)

    def empty(self):
        '''
        Empty buffer
        '''
        self.positive_demonstrations = deque(maxlen=self.max_size)
        self.negative_demonstrations = deque(maxlen=self.max_size)

def play_buffer():
    '''
    Load and replay demonstrations
    '''
    from ceres.tools import ExtraArgs
    from ceres.envs import ResetterEnv
    import gym
    extra_args = ExtraArgs()
    assert len(extra_args.constraint_demonstration_buffer) > 0, 'Required argument --constraint_demonstration_buffer'
    constraint_demonstration_buffer = ConstraintDemonstrationBuffer.from_backup(extra_args.constraint_demonstration_buffer)
    env = gym.make(extra_args.env_id)
    env = env.unwrapped
    assert isinstance(env, ResetterEnv)
    env.reset()

    demonstration = constraint_demonstration_buffer.positive_demonstrations[0]
    env.reset_and_restore(snapshot=demonstration.snapshot)
    if extra_args.render:
        env.render(mode='human')

    restore_positive = True
    i_positive = 0
    i_negative = 0
    valid_commands_str = '[Enter]: play next in current buffer, [g]: switch to positive buffer, [b]: switch to negative buffer, [q] to quit'
    while True:
        if restore_positive:
            if len(constraint_demonstration_buffer.positive_demonstrations) == 0:
                print('No positive example, switch to negative')
                restore_positive = False
                assert len(constraint_demonstration_buffer.negative_demonstrations) > 0
        else:
            if len(constraint_demonstration_buffer.negative_demonstrations) == 0:
                print('No negative example, switch to positive')
                restore_positive = True
                assert len(constraint_demonstration_buffer.positive_demonstrations) > 0
        current_buffer = constraint_demonstration_buffer.positive_demonstrations if restore_positive else constraint_demonstration_buffer.negative_demonstrations
        current_index = i_positive if restore_positive else i_negative
        if current_index > len(current_buffer):
            current_index = 0
        buffer_info_str = 'Current buffer: {0} ({1}/{2})'.format(('positive' if restore_positive else 'negative'), current_index, len(current_buffer)-1)
        input_str = '{0}\n  Action? {1}\n'.format(buffer_info_str, valid_commands_str)
        command = input(input_str)
        if command == '':
            demonstration = current_buffer[current_index]
            env.reset_and_restore(snapshot=demonstration.snapshot)
            if extra_args.render:
                env.render(mode='human')
            env.step(demonstration.action)
            print('  Restored state {0}, played action {1}'.format(demonstration.state, demonstration.action))
            if extra_args.render:
                env.render(mode='human')
            if restore_positive:
                i_positive += 1
            else:
                i_negative += 1
        elif command == 'b':
            restore_positive = False
        elif command == 'g':
            restore_positive = True
        elif command == 'q':
            return
        else:
            print('Invalid command {0}. Possible commands: {1}'.format(command, valid_commands_str))

if __name__ == '__main__':
    play_buffer()

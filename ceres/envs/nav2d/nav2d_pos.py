# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import gym
import numpy as np
from collections import deque

class Nav2dPos(gym.Env):
    '''
    Control an agent to navigate to a point by position commands
    '''

    max_episode_steps = 100

    is_state_target_rel_pos = False

    min_x = -1.
    min_y = -1.
    max_x =  1.
    max_y =  1.
    max_step_x = 0.10
    max_step_y = 0.10
    world_width  = max_x - min_x
    world_height = max_y - min_y

    stats_max_length = 100

    max_normalized_obs = 2.
    max_normalized_act = 2.
    alpha_xy_epsilon = 1e-5

    reward_scaling_factor = 1.
    do_randomize_agent_pos = True
    do_randomize_target_pos = True
    use_reward_near_target = True
    reward_near_target     = 10.
    use_reward_out_of_range = True
    reward_out_of_range     = -10.

    agent_radius  = 0.025
    target_radius = 0.025
    near_target_threshold = 2.*target_radius
    reward_name_near_target = 'near_target'

    use_reward_alive = True
    reward_alive     = -0.01
    use_reward_distance    = True
    reward_distance_factor = 0.01

    do_normalize_observations = True
    do_normalize_actions      = True
    do_clip_command = True
    max_step_norm = min(max_step_x, max_step_y)

    info_key_failure = 'failure'
    info_key_success = 'success'

    def init_sampling_range(self):
        self.agent_pos_x_range_sampling = [self.min_x + self.agent_radius, self.max_x - self.agent_radius]
        self.agent_pos_y_range_sampling = [self.min_y + self.agent_radius, self.max_y - self.agent_radius]
        self.target_pos_x_range_sampling = [self.min_x, self.max_x]
        self.target_pos_y_range_sampling = [self.min_y, self.max_y]

    def init_misc(self):
        self.metadata['render.modes'].append('human')
        self.reward_seq = []
        self.ep_reward_history = deque(maxlen=self.stats_max_length)
        self.ep_reward_avg = 0.

    def __init__(self):
        self.init_sampling_range()
        self.init_action_space()
        self.init_observation_space()
        self.init_misc()
        self.is_init_rendering = False
        super().__init__()

    def init_rendering(self):
        pass

    def seed(self, seed):
        pass

    def render(self, mode='human', close=False):
        pass

    def calc_new_pos_from_step(self, action):
        dx = action[self.action_index['agent_move_x']]
        dy = action[self.action_index['agent_move_y']]
        new_agent_pos_x = self.agent_pos_x + dx
        new_agent_pos_y = self.agent_pos_y + dy
        return new_agent_pos_x, new_agent_pos_y

    def do_action(self):
        self.agent_pos_x, self.agent_pos_y = self.calc_new_pos_from_step(self.command_play)

    def add_action(self, act_name, act_min, act_max):
        assert not (act_name in self.action_names), 'Action {0} already added'.format(act_name)
        self.action_names.append(act_name)
        self.action_index[act_name] = len(self.action_space_denormalized_low)
        self.action_space_denormalized_low.append(act_min)
        self.action_space_denormalized_high.append(act_max)
        self.action_space_normalized_low.append(-self.max_normalized_act)
        self.action_space_normalized_high.append(self.max_normalized_act)

    def setup_actions(self):
        self.add_action('agent_move_x', -self.max_step_x, self.max_step_x)
        self.add_action('agent_move_y', -self.max_step_y, self.max_step_y)

    def init_action_space(self):
        self.action_space_denormalized_low  = []
        self.action_space_denormalized_high = []
        self.action_space_normalized_low  = []
        self.action_space_normalized_high = []
        self.action_names = []
        self.action_index = {}

        self.setup_actions()

        self.action_space_denormalized_low  = np.array(self.action_space_denormalized_low)
        self.action_space_denormalized_high = np.array(self.action_space_denormalized_high)
        self.action_space_normalized_low  = np.array(self.action_space_normalized_low)
        self.action_space_normalized_high = np.array(self.action_space_normalized_high)
        self.action_space = gym.spaces.Box(low=self.action_space_normalized_low, high=self.action_space_normalized_high, dtype=np.float32)
        self.n_act = self.action_space.shape[0]

        self.init_action_normalization()

    def build_action(self, action_dict):
        action = np.zeros(self.n_act)
        for action_field in ['agent_move_x', 'agent_move_y']:
            assert action_field in action_dict, 'Missing necessary field {0}'.format(action_field)
            action[self.action_index[action_field]] = action_dict[action_field]
        return action


    def add_observation(self, obs_name, obs_min, obs_max):
        assert not (obs_name in self.observation_names), 'Observation {0} already added'.format(obs_name)
        self.observation_index[obs_name] = len(self.observation_space_low)
        self.observation_names.append(obs_name)
        self.observation_space_low.append(obs_min)
        self.observation_space_high.append(obs_max)

    def init_observation_space(self):
        # The observation space consists of the agent and target positions
        self.observation_space_low  = []
        self.observation_space_high = []
        self.observation_names = []
        self.observation_index = {}

        self.setup_observations()

        self.observation_space_low = np.array(self.observation_space_low)
        self.observation_space_high = np.array(self.observation_space_high)
        self.observation_space = gym.spaces.Box(low=self.observation_space_low, high=self.observation_space_high, dtype=np.float32)
        self.n_obs = self.observation_space.shape[0]
        assert self.n_obs > 0

        self.init_observation_normalization()

    def setup_observations(self):
        # Agent position
        self.agent_pos_x_range = [self.min_x, self.max_x]
        self.agent_pos_y_range = [self.min_y, self.max_y]
        self.add_observation('agent_pos_x', self.agent_pos_x_range[0], self.agent_pos_x_range[1])
        self.add_observation('agent_pos_y', self.agent_pos_y_range[0], self.agent_pos_y_range[1])

        # Target position
        self.target_pos_x_range = [self.min_x, self.max_x]
        self.target_pos_y_range = [self.min_y, self.max_y]
        self.max_rel_pos_x = self.target_pos_x_range[1] - self.agent_pos_x_range[0]
        self.max_rel_pos_y = self.target_pos_y_range[1] - self.agent_pos_y_range[0]
        assert (self.max_rel_pos_x > 0) and (self.max_rel_pos_y > 0)
        if self.is_state_target_rel_pos:
            self.add_observation('target_rel_pos_x', -self.max_rel_pos_x, self.max_rel_pos_x)
            self.add_observation('target_rel_pos_y', -self.max_rel_pos_y, self.max_rel_pos_y)
        else:
            self.add_observation('target_pos_x', self.target_pos_x_range[0], self.target_pos_x_range[1])
            self.add_observation('target_pos_y', self.target_pos_y_range[0], self.target_pos_y_range[1])

        self.setup_observations_task_specific()

    def setup_observations_task_specific(self):
        # Implement this in child classes
        pass


    def create_normalizer_denormalizer(self, pm, mid, max_normalized):
        def normalizer(x):
            return (x - mid)/pm*max_normalized
        def denormalizer(x):
            return x/max_normalized*pm + mid
        return normalizer, denormalizer

    def create_normalizing_denormalizing_matrices(self, pm_list, mid_list, max_normalized_list):
        normalize_scale_mat = np.diag([max_normalized/pm for pm, max_normalized in zip(pm_list, max_normalized_list)])
        normalize_shift_vec = np.array([[-mid*max_normalized/pm] for mid, pm, max_normalized in zip(mid_list, pm_list, max_normalized_list)])
        denormalize_scale_mat = np.diag([pm / max_normalized for pm, max_normalized in zip(pm_list, max_normalized_list)])
        denormalize_shift_vec = np.array([[mid] for mid in mid_list])
        return normalize_scale_mat, normalize_shift_vec, denormalize_scale_mat, denormalize_shift_vec


    def init_action_normalization(self):
        self.normalize_act_by_index   = []
        self.denormalize_act_by_index = []
        self.normalize_act_by_name    = {}
        self.denormalize_act_by_name  = {}
        pm_list, mid_list, max_normalized_list = [], [], []
        for i_act, (act_name, act_min, act_max) in enumerate(zip(self.action_names, self.action_space_denormalized_low, self.action_space_denormalized_high)):
            mid = 0.5*(act_min + act_max)
            pm  = act_max - mid
            assert pm > 0, 'Action bounds are ill-defined'
            pm_list.append(pm)
            mid_list.append(mid)
            max_normalized_list.append(self.max_normalized_act)
            normalize_act, denormalize_act = self.create_normalizer_denormalizer(pm, mid, self.max_normalized_act)
            self.normalize_act_by_index.append(normalize_act)
            self.normalize_act_by_name[act_name] = normalize_act
            self.denormalize_act_by_index.append(denormalize_act)
            self.denormalize_act_by_name[act_name] = denormalize_act
        self.normalize_act_scale, self.normalize_act_shift, self.denormalize_act_scale, self.denormalize_act_shift = \
                self.create_normalizing_denormalizing_matrices(pm_list, mid_list, max_normalized_list)

    def normalize_act_vector(self, x):
        return np.array([f(v) for (f, v) in zip(self.normalize_act_by_index, x)])
    def normalize_obs_vector(self, x):
        return np.array([f(v) for (f, v) in zip(self.normalize_obs_by_index, x)])
    def denormalize_act_vector(self, x):
        return np.array([f(v) for (f, v) in zip(self.denormalize_act_by_index, x)])
    def denormalize_obs_vector(self, x):
        return np.array([f(v) for (f, v) in zip(self.denormalize_obs_by_index, x)])

    def init_observation_normalization(self):
        self.normalize_obs_by_index   = []
        self.denormalize_obs_by_index = []
        self.normalize_obs_by_name    = {}
        self.denormalize_obs_by_name  = {}
        pm_list, mid_list, max_normalized_list = [], [], []
        for i_obs, (obs_name, obs_min, obs_max) in enumerate(zip(self.observation_names, self.observation_space_low, self.observation_space_high)):
            mid = 0.5*(obs_min + obs_max)
            pm  = obs_max - mid
            assert pm > 0, 'Action bounds are ill-defined'
            pm_list.append(pm)
            mid_list.append(mid)
            max_normalized_list.append(self.max_normalized_obs)
            normalize_obs, denormalize_obs = self.create_normalizer_denormalizer(pm, mid, self.max_normalized_obs)
            self.normalize_obs_by_index.append(normalize_obs)
            self.normalize_obs_by_name[obs_name] = normalize_obs
            self.denormalize_obs_by_index.append(denormalize_obs)
            self.denormalize_obs_by_name[obs_name] = denormalize_obs
        self.normalize_obs_scale, self.normalize_obs_shift, self.denormalize_obs_scale, self.denormalize_obs_shift = \
                self.create_normalizing_denormalizing_matrices(pm_list, mid_list, max_normalized_list)

    def zero_action(self):
        return np.zeros(self.n_act)


    def update_env(self):
        pass

    def fill_state(self, state):
        state[self.observation_index['agent_pos_x']] = self.agent_pos_x
        state[self.observation_index['agent_pos_y']] = self.agent_pos_y
        if self.is_state_target_rel_pos:
            state[self.observation_index['target_rel_pos_x']] = self.target_pos_x - self.agent_pos_x
            state[self.observation_index['target_rel_pos_y']] = self.target_pos_y - self.agent_pos_y
        else:
            state[self.observation_index['target_pos_x']] = self.target_pos_x
            state[self.observation_index['target_pos_y']] = self.target_pos_y
        self.fill_state_task_specific(state)

    def fill_state_task_specific(self, state):
        # Implement this in child classes
        pass

    def calc_state(self):
        state = np.zeros(self.n_obs)
        self.fill_state(state)
        self.current_state = state
        state = self.normalize_observation(state)
        self.output_state = state
        self.calc_state_task_specific()
        return state

    def calc_snapshot(self):
        snapshot_state = self.calc_state()
        snapshot_task_specific = self.calc_snapshot_task_specific()
        if len(snapshot_task_specific) > 0:
            snapshot = np.concatenate([snapshot_state, snapshot_task_specific])
        else:
            snapshot = snapshot_state
        return snapshot

    def calc_state_task_specific(self):
        pass

    def calc_snapshot_task_specific(self):
        return []

    def reset_agent(self):
        self.reset_agent_pos()

    def reset_agent_pos(self):
        if self.do_randomize_agent_pos:
            self.agent_pos_x = np.random.rand()*(self.agent_pos_x_range_sampling[1] - self.agent_pos_x_range_sampling[0]) + self.agent_pos_x_range_sampling[0]
            self.agent_pos_y = np.random.rand()*(self.agent_pos_y_range_sampling[1] - self.agent_pos_y_range_sampling[0]) + self.agent_pos_y_range_sampling[0]
        else:
            self.agent_pos_x = -0.5
            self.agent_pos_y = 0

    def reset_target(self):
        self.reset_target_pos()

    def reset_target_pos(self):
        if self.do_randomize_target_pos:
            self.target_pos_x = np.random.rand()*(self.target_pos_x_range_sampling[1] - self.target_pos_x_range_sampling[0]) + self.target_pos_x_range_sampling[0]
            self.target_pos_y = np.random.rand()*(self.target_pos_y_range_sampling[1] - self.target_pos_y_range_sampling[0]) + self.target_pos_y_range_sampling[0]
        else:
            self.target_pos_x = 0.5
            self.target_pos_y = 0.

    def reset(self):
        return self.reset_random()

    def reset_random(self):
        return self.reset_and_restore(snapshot=None)

    def restore_snapshot(self, snapshot):
        snapshot_state = snapshot[:self.n_obs]
        self.restore_snapshot_state(snapshot_state)
        snapshot_task_specific = snapshot[self.n_obs:]
        self.restore_snapshot_task_specific(snapshot_task_specific)

    def restore_snapshot_state(self, snapshot_state):
        state = snapshot_state # in this environment, they're the same
        denormalized_state = self.denormalize_observation(state)
        self.restore_agent(denormalized_state)
        self.restore_target(denormalized_state)

    def restore_snapshot_task_specific(self, snapshot_task_specific):
        pass

    def restore_agent(self, denormalized_state):
        self.restore_agent_pos(denormalized_state)

    def restore_target(self, denormalized_state):
        self.restore_target_pos(denormalized_state)

    def reset_state(self):
        self.reset_agent()
        self.reset_target()

    def reset_and_restore(self, snapshot=None):
        ### Here, snapshot = state
        if len(self.reward_seq) > 0:
            self.ep_reward_history.append(np.sum(self.reward_seq))
            self.ep_reward_avg = np.average(self.ep_reward_history)
        self.i_step = 0
        self.done_base = False
        self.agent_pos_seq = []
        self.action_seq = []
        self.state_seq  = []
        self.reward_seq = []
        if snapshot is None:
            self.reset_state()
        else:
            self.restore_snapshot(snapshot)
        self.store_agent_pos()
        self.reset_task_specific()
        self.update_env()
        state = self.calc_state()
        self.state_seq.append(self.output_state)
        return state

    def restore_agent_pos(self, denormalized_state):
        self.agent_pos_x = denormalized_state[self.observation_index['agent_pos_x']]
        self.agent_pos_y = denormalized_state[self.observation_index['agent_pos_y']]

    def restore_target_pos(self, denormalized_state):
        if self.is_state_target_rel_pos:
            target_rel_pos_x = denormalized_state[self.observation_index['target_rel_pos_x']]
            target_rel_pos_y = denormalized_state[self.observation_index['target_rel_pos_y']]
            self.target_pos_x = self.agent_pos_x + target_rel_pos_x
            self.target_pos_y = self.agent_pos_y + target_rel_pos_y
        else:
            self.target_pos_x = denormalized_state[self.observation_index['target_pos_x']]
            self.target_pos_y = denormalized_state[self.observation_index['target_pos_y']]

    def reset_task_specific(self):
        pass

    def store_agent_pos(self):
        self.agent_pos_seq.append((self.agent_pos_x, self.agent_pos_y))

    def denormalize_observation(self, a):
        if self.do_normalize_observations:
            return self.denormalize_obs_vector(a)
        else:
            return a

    def normalize_observation(self, a):
        if self.do_normalize_observations:
            return self.normalize_obs_vector(a)
        else:
            return a

    def denormalize_action(self, a):
        if self.do_normalize_actions:
            return self.denormalize_act_vector(a)
        else:
            return a

    def normalize_action(self, a):
        if self.do_normalize_actions:
            return self.normalize_act_vector(a)
        else:
            return a

    def clip_vector_by_norm(self, vec, max_norm):
        vec_norm = np.linalg.norm(vec)
        if vec_norm <= max_norm:
            return vec
        else:
            return vec / vec_norm * max_norm

    def clip_command(self, a):
        if self.do_clip_command:
            return self.clip_vector_by_norm(a, self.max_step_norm)
        else:
            return a

    def clip_action(self, a):
        a_clipped = np.clip(a, self.action_space_normalized_low, self.action_space_normalized_high)
        return a_clipped

    def calc_command_from_action(self, action_raw):
        action = self.clip_action(action_raw)
        command = self.denormalize_action(action)
        command = self.clip_command(command)
        action = self.normalize_action(command)
        return command

    def step(self, a):
        self.action_raw = a
        self.command_play = self.calc_command_from_action(self.action_raw)

        self.do_action()
        self.update_env()
        state = self.calc_state()

        reward_dict, done_dict = self.calc_rewards()
        reward = sum([_v for _k, _v in reward_dict.items()])
        done   = any([_v for _k, _v in done_dict.items()])

        success = done_dict[self.reward_name_near_target]
        failure = done and (not success)
        self.done_base = done

        self.store_agent_pos()
        self._step_task_specific()

        self.i_step += 1

        self.action_seq.append(self.action_raw)
        self.state_seq.append(self.output_state)
        self.reward_seq.append(reward)

        info = {'reward': reward_dict,
                'done': done_dict,
                self.info_key_success: success,
                self.info_key_failure: failure}

        return state, reward, done, info

    def print_state(self, state):
        for (obs_name, obs_val) in zip(self.observation_names, state):
            print('  {0}: {1}'.format(obs_name, obs_val))

    def _step_task_specific(self):
        pass

    def add_reward_done(self, reward_dict, done_dict, name, reward, done):
        reward_dict[name] = reward
        reward_dict[name] *= self.reward_scaling_factor
        done_dict[name] = done

    def calc_rewards(self):
        reward_dict = {}
        done_dict   = {}
        if self.use_reward_distance:
            self.add_reward_done(reward_dict, done_dict, *self.calc_reward_distance())
        if self.use_reward_near_target:
            self.add_reward_done(reward_dict, done_dict, *self.calc_reward_near_target())
        if self.use_reward_out_of_range:
            self.add_reward_done(reward_dict, done_dict, *self.calc_reward_out_of_range())
        if self.use_reward_alive:
            self.add_reward_done(reward_dict, done_dict, *self.calc_reward_alive())
        return reward_dict, done_dict

    def calc_reward_alive(self):
        reward_name = 'alive'
        return reward_name, self.reward_alive, False
    
    def calc_distance_to_target(self, agent_pos_x=None, agent_pos_y=None):
        if agent_pos_x is None:
            agent_pos_x = self.agent_pos_x
        if agent_pos_y is None:
            agent_pos_y = self.agent_pos_y
        rel_pos_x = self.target_pos_x - self.agent_pos_x
        rel_pos_y = self.target_pos_y - self.agent_pos_y
        distance = np.sqrt((rel_pos_x ** 2) + (rel_pos_y ** 2))
        return distance

    def calc_reward_distance(self):
        reward_name = 'distance'
        distance = self.calc_distance_to_target()
        reward_distance = -float(distance)
        reward_distance *= self.reward_distance_factor
        done = False
        return reward_name, reward_distance, done

    def test_out_of_range(self, agent_pos_x=None, agent_pos_y=None, min_distance=None):
        if agent_pos_x is None:
            agent_pos_x = self.agent_pos_x
        if agent_pos_y is None:
            agent_pos_y = self.agent_pos_y
        if min_distance is None:
            min_distance = self.agent_radius
        agent_in_range_x = self.agent_pos_x_range[0] + min_distance <= agent_pos_x <= self.agent_pos_x_range[1] - min_distance
        agent_in_range_y = self.agent_pos_y_range[0] + min_distance <= agent_pos_y <= self.agent_pos_y_range[1] - min_distance
        agent_in_range = agent_in_range_x and agent_in_range_y
        out_of_range = not agent_in_range
        return out_of_range

    def calc_reward_out_of_range(self):
        reward_name = 'out_of_range'
        done = self.test_out_of_range()
        if done:
            reward = self.reward_out_of_range
        else:
            reward = 0.
        return reward_name, reward, done

    def test_near_target(self, agent_pos_x=None, agent_pos_y=None):
        if agent_pos_x is None:
            agent_pos_x = self.agent_pos_x
        if agent_pos_y is None:
            agent_pos_y = self.agent_pos_y
        distance = self.calc_distance_to_target(agent_pos_x, agent_pos_y)
        is_near_target = distance < self.near_target_threshold
        return is_near_target

    def calc_reward_near_target(self):
        reward_name = self.reward_name_near_target
        done = self.test_near_target()
        if done:
            reward_near_target = self.reward_near_target
        else:
            reward_near_target = 0.
        return reward_name, reward_near_target, done

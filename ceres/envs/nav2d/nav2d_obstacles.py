# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np
from .nav2d_pos import Nav2dPos
from .obstacles import ObstacleSquare, ObstacleCircle

class Nav2dObstacles(Nav2dPos):
    '''
    Control an agent to navigate to a point by position commands
    while avoiding fixed and/or random obstacles
    '''
    obstacles = []

    fixed_obstacles = []
    random_obstacles = []
    n_random_square_obstacles = 0
    n_random_circle_obstacles = 0
    random_square_obstacle_dim_range = [0.1, 0.1]
    random_circle_obstacle_dim_range = [0.1, 0.1]

    use_reward_collision = True
    reward_collision     = -10.
    max_reset_attempts = 1e3

    is_state_closest_obstacle = False
    max_dist_closest_obstacle = 0.
    is_state_obstacle_params = False
    state_lidar_angles = []
    lidar_segments = []

    do_order_closest_obstacle_by_distance = True # if False, order by obstacle index

    min_x_random_obstacles = -1.
    max_x_random_obstacles =  1.
    min_y_random_obstacles = -1.
    max_y_random_obstacles =  1.

    def reset_agent_pos(self):
        if self.do_randomize_agent_pos:
            do_re_randomize  = True
            n_randomize_agent = 0
            while do_re_randomize:
                n_randomize_agent += 1
                assert n_randomize_agent <= self.max_reset_attempts
                self.agent_pos_x = np.random.rand()*(self.agent_pos_x_range_sampling[1] - self.agent_pos_x_range_sampling[0]) + self.agent_pos_x_range_sampling[0]
                self.agent_pos_y = np.random.rand()*(self.agent_pos_y_range_sampling[1] - self.agent_pos_y_range_sampling[0]) + self.agent_pos_y_range_sampling[0]
                is_collision_agent = self.test_any_collision(self.agent_pos_x, self.agent_pos_y)
                is_out_of_range_agent = self.test_out_of_range()
                do_re_randomize = is_collision_agent or is_out_of_range_agent
        else:
            self.agent_pos_x = -0.75
            self.agent_pos_y = -0.75

    def reset_target_pos(self):
        if self.do_randomize_target_pos:
            is_collision_target = True
            n_randomize_target = 0
            while is_collision_target:
                n_randomize_target += 1
                assert n_randomize_target <= self.max_reset_attempts
                self.target_pos_x = np.random.rand()*(self.target_pos_x_range_sampling[1] - self.target_pos_x_range_sampling[0]) + self.target_pos_x_range_sampling[0]
                self.target_pos_y = np.random.rand()*(self.target_pos_y_range_sampling[1] - self.target_pos_y_range_sampling[0]) + self.target_pos_y_range_sampling[0]
                is_collision_target = self.test_any_collision(self.target_pos_x, self.target_pos_y)
        else:
            self.target_pos_x = 0.75
            self.target_pos_y = 0.75


    def calc_rewards(self):
        reward_dict, done_dict = super().calc_rewards()
        if self.use_reward_collision:
            self.add_reward_done(reward_dict, done_dict, *self.calc_reward_collision())
        return reward_dict, done_dict

    def calc_reward_collision(self):
        reward_name = 'collision'
        done_collision = self.test_any_collision(self.agent_pos_x, self.agent_pos_y)
        if done_collision:
            reward = self.reward_collision
        else:
            reward = 0.
        return reward_name, reward, done_collision

    def test_any_collision(self, x, y):
        is_collision = any([o.test_collision(x, y, min_distance=self.agent_radius) for o in self.obstacles])
        return is_collision

    def setup_observations_task_specific(self):
        super().setup_observations_task_specific()
        if self.is_state_closest_obstacle:
            for i_obstacle in range(len(self.fixed_obstacles) + self.n_random_square_obstacles + self.n_random_circle_obstacles):
                obstacle_x_str = 'closest_{0}_rel_x'.format(i_obstacle)
                obstacle_y_str = 'closest_{0}_rel_y'.format(i_obstacle)
                self.add_observation(obstacle_x_str, -self.max_rel_pos_x, self.max_rel_pos_x)
                self.add_observation(obstacle_y_str, -self.max_rel_pos_y, self.max_rel_pos_y)
        if self.is_state_obstacle_params:
            for i_obstacle in range(len(self.fixed_obstacles) + self.n_random_square_obstacles + self.n_random_circle_obstacles):
                if i_obstacle < len(self.fixed_obstacles):
                    obstacle_parameters = self.fixed_obstacles[i_obstacle].required_parameters
                elif i_obstacle < len(self.fixed_obstacles) + self.n_random_square_obstacles:
                    obstacle_parameters = ObstacleSquare.required_parameters
                else:
                    obstacle_parameters = ObstacleCircle.required_parameters
                for _k in obstacle_parameters:
                    # We use the X range as limits but in reality, obstacles can extend beyond
                    state_str = 'obstacle_{0}_{1}'.format(i_obstacle, _k)
                    state_min = self.min_x_random_obstacles
                    state_max = self.max_x_random_obstacles
                    self.add_observation(state_str, state_min, state_max)
        if len(self.state_lidar_angles) > 0:
            for i_angle, lidar_angle in enumerate(self.state_lidar_angles):
                state_str = 'lidar_{0}'.format(i_angle)
                state_min = self.min_x
                state_max = self.max_x
                self.add_observation(state_str, state_min, state_max)


    def fill_state_task_specific(self, state):
        super().fill_state_task_specific(state)
        if self.is_state_closest_obstacle:
            if self.do_order_closest_obstacle_by_distance:
                dist_list = []
                proj_list = []
            for i_obstacle, obstacle in enumerate(self.obstacles):
                obstacle_x_str = 'closest_{0}_rel_x'.format(i_obstacle)
                obstacle_y_str = 'closest_{0}_rel_y'.format(i_obstacle)
                proj_x, proj_y, is_strictly_inside = obstacle.project(self.agent_pos_x, self.agent_pos_y)
                rel_x = proj_x - self.agent_pos_x
                rel_y = proj_y - self.agent_pos_y
                dist = np.sqrt(rel_x**2 + rel_y**2)
                if (self.max_dist_closest_obstacle == 0.) or (dist < self.max_dist_closest_obstacle):
                    # Only include obstacles within a certain range
                    if self.do_order_closest_obstacle_by_distance:
                        dist_list.append(dist)
                        proj_list.append((rel_x, rel_y))
                    else:
                        state[self.observation_index[obstacle_x_str]] = rel_x
                        state[self.observation_index[obstacle_y_str]] = rel_y
                else:
                    if not self.do_order_closest_obstacle_by_distance:
                        state[self.observation_index[obstacle_x_str]] = 0.
                        state[self.observation_index[obstacle_y_str]] = 0.
            if self.do_order_closest_obstacle_by_distance:
                i_dist_sorted = np.argsort(dist_list)
                proj_list_sorted = [proj_list[_i] for _i in i_dist_sorted]
                dist_list_sorted = [dist_list[_i] for _i in i_dist_sorted]
                for i_obstacle, obstacle in enumerate(self.obstacles):
                    obstacle_x_str = 'closest_{0}_rel_x'.format(i_obstacle)
                    obstacle_y_str = 'closest_{0}_rel_y'.format(i_obstacle)
                    if i_obstacle < len(proj_list_sorted):
                        state[self.observation_index[obstacle_x_str]] = proj_list_sorted[i_obstacle][0]
                        state[self.observation_index[obstacle_y_str]] = proj_list_sorted[i_obstacle][1]
                    else:
                        state[self.observation_index[obstacle_x_str]] = 0.
                        state[self.observation_index[obstacle_y_str]] = 0.
        if self.is_state_obstacle_params:
            for i_obstacle, obstacle in enumerate(self.obstacles):
                params = obstacle.to_array()
                for (_k, _v) in zip(obstacle.required_parameters, params):
                    # We use the X range as limits but in reality, obstacles can extend beyond
                    state_str = 'obstacle_{0}_{1}'.format(i_obstacle, _k)
                    state[self.observation_index[state_str]] = _v
        if len(self.state_lidar_angles) > 0:
            p1 = np.array([self.agent_pos_x, self.agent_pos_y])
            is_collision_agent = self.test_any_collision(self.agent_pos_x, self.agent_pos_y)
            is_out_of_range_agent = self.test_out_of_range()
            if is_collision_agent or is_out_of_range_agent:
                self.lidar_segments = [(p1, p1, 0.)]*len(self.state_lidar_angles)
            else:
                self.lidar_segments = [None] * len(self.state_lidar_angles)
                for i_angle, lidar_angle in enumerate(self.state_lidar_angles):
                    p2, unit_vec, dist = self.calc_intersection_with_border(p1, lidar_angle)
                    if dist == 0:
                        p2 = p1
                    else:
                        for i_obstacle, o in enumerate(self.obstacles):
                            is_intersect, p_candidate = o.intersection_with_line(p1, p2)
                            if is_intersect:
                                candidate_dist = np.linalg.norm(p_candidate - p1)
                                if candidate_dist < dist:
                                    p2 = p_candidate
                                    dist = candidate_dist
                    self.lidar_segments[i_angle] = (p1 + self.agent_radius * unit_vec, p2, dist)
            for i_angle, segment in enumerate(self.lidar_segments):
                state_str = 'lidar_{0}'.format(i_angle)
                state[self.observation_index[state_str]] = segment[2]

    def calc_intersection_with_border(self, p1, angle):
        unit_vec = np.array([np.cos(angle), np.sin(angle)])
        p2_candidates = []

        if unit_vec[0] != 0.:
            if unit_vec[0] > 0.:
                dx = self.max_x - self.agent_pos_x
            else:
                dx = self.min_x - self.agent_pos_x
            dy = dx * np.tan(angle)
            if self.min_y <= (p1[1] + dy) <= self.max_y:
                p2_candidates.append(p1 + np.array([dx, dy]))
        if unit_vec[1] != 0.:
            if unit_vec[1] > 0.:
                dy = self.max_x - self.agent_pos_y
            else:
                dy = self.min_x - self.agent_pos_y
            dx = dy * np.tan(np.pi/2. - angle)
            if self.min_x <= (p1[0] + dx) <= self.max_x:
                p2_candidates.append(p1 + np.array([dx, dy]))
        if len(p2_candidates) == 1:
            p2 = p2_candidates[0]
        elif len(p2_candidates) == 2:
            p2 = p2_candidates[np.argmin([np.linalg.norm(_p) for _p in p2_candidates])]
        else: # no intersection with border, return point
            p2 = p1
        dist = np.linalg.norm(p2 - p1)
        return p2, unit_vec, dist

    def reset_state(self):
        self.reset_random_obstacles()
        self.obstacles = self.fixed_obstacles + self.random_obstacles
        super().reset_state()

    def reset_random_obstacles(self):
        self.random_obstacles = []
        if self.n_random_square_obstacles > 0:
            obstacle_centers_x = np.random.rand(self.n_random_square_obstacles)
            obstacle_centers_x *= (self.max_x_random_obstacles - self.min_x_random_obstacles)
            obstacle_centers_x += self.min_x_random_obstacles
            obstacle_centers_y = np.random.rand(self.n_random_square_obstacles)
            obstacle_centers_y *= (self.max_y_random_obstacles - self.min_y_random_obstacles)
            obstacle_centers_y += self.min_y_random_obstacles
            obstacle_centers = np.array([[x, y] for x, y in zip(obstacle_centers_x, obstacle_centers_y)])
            obstacle_dims = np.random.rand(self.n_random_square_obstacles, 2)
            obstacle_dims *= (self.random_square_obstacle_dim_range[1] - self.random_square_obstacle_dim_range[0])
            obstacle_dims += self.random_square_obstacle_dim_range[0]
            obstacle_shift_top_left = np.array([[-width_height[0]/2., width_height[1]/2.] for width_height in obstacle_dims])
            obstacle_shift_bottom_right = np.array([[width_height[0]/2., -width_height[1]/2.] for width_height in obstacle_dims])
            obstacle_top_left = obstacle_centers + obstacle_shift_top_left
            obstacle_bottom_right = obstacle_centers + obstacle_shift_bottom_right
            for top_left, bottom_right in zip(obstacle_top_left, obstacle_bottom_right):
                obstacle = ObstacleSquare(top_left_x=top_left[0], top_left_y=top_left[1], bottom_right_x=bottom_right[0], bottom_right_y=bottom_right[1])
                self.random_obstacles.append(obstacle)
        if self.n_random_circle_obstacles > 0:
            obstacle_centers_x = np.random.rand(self.n_random_circle_obstacles)
            obstacle_centers_x *= (self.max_x_random_obstacles - self.min_x_random_obstacles)
            obstacle_centers_x += self.min_x_random_obstacles
            obstacle_centers_y = np.random.rand(self.n_random_circle_obstacles)
            obstacle_centers_y *= (self.max_y_random_obstacles - self.min_y_random_obstacles)
            obstacle_centers_y += self.min_y_random_obstacles
            obstacle_centers = np.array([[x, y] for x, y in zip(obstacle_centers_x, obstacle_centers_y)])
            obstacle_dims = np.random.rand(self.n_random_circle_obstacles)
            obstacle_dims *= (self.random_circle_obstacle_dim_range[1] - self.random_circle_obstacle_dim_range[0])
            obstacle_dims += self.random_circle_obstacle_dim_range[0]
            for obstacle_center, obstacle_radius in zip(obstacle_centers, obstacle_dims):
                obstacle = ObstacleCircle(center_x=obstacle_center[0], center_y=obstacle_center[1], radius=obstacle_radius)
                self.random_obstacles.append(obstacle)

    def calc_snapshot_task_specific(self):
        obstacle_params = [o.to_array() for o in self.random_obstacles]
        if len(obstacle_params) > 0:
            obstacle_params = np.concatenate(obstacle_params)
        else:
            obstacle_params = np.array(obstacle_params)
        return obstacle_params

    def restore_snapshot_task_specific(self, snapshot_task_specific):
        obstacle_params = snapshot_task_specific
        i_start = 0
        for i_obstacle in range(len(self.random_obstacles)):
            current_obstacle = self.random_obstacles[i_obstacle]
            i_end = i_start + len(current_obstacle.required_parameters)
            params = obstacle_params[i_start:i_end]
            i_start = i_end
            new_obstacle = current_obstacle.FromArray(params)
            self.random_obstacles[i_obstacle] = new_obstacle


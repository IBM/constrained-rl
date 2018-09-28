# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np
from .nav2d_pos import Nav2dPos

class Nav2dForce(Nav2dPos):
    '''
    Control an agent to navigate to a point by force commands
    '''

    max_vel = 0.10
    max_acc = 0.05
    delta_time = 1. # one action per frame
    do_clip_vel = True
    do_randomize_agent_vel = True
    agent_mass = 1.

    def setup_actions(self):
        self.add_action('agent_set_acc_x', -self.max_acc, self.max_acc)
        self.add_action('agent_set_acc_y', -self.max_acc, self.max_acc)

    def setup_observations_task_specific(self):
        super().setup_observations_task_specific()
        # Agent velocity
        self.agent_vel_x_range = [-self.max_vel, self.max_vel]
        self.agent_vel_y_range = [-self.max_vel, self.max_vel]
        self.add_observation('agent_vel_x', self.agent_vel_x_range[0], self.agent_vel_x_range[1])
        self.add_observation('agent_vel_y', self.agent_vel_y_range[0], self.agent_vel_y_range[1])

    def do_action(self):
        self.agent_acc_x, self.agent_acc_y = self.agent_mass * self.command_play
        self.agent_vel_x += self.agent_acc_x*self.delta_time
        self.agent_vel_y += self.agent_acc_y*self.delta_time
        if self.do_clip_vel:
            self.agent_vel_x, self.agent_vel_y = self.clip_vector_by_norm([self.agent_vel_x, self.agent_vel_y], self.max_vel)
        self.agent_pos_x += self.agent_vel_x*self.delta_time
        self.agent_pos_y += self.agent_vel_y*self.delta_time


    def clip_command(self, a):
        if self.do_clip_command:
            return self.clip_vector_by_norm(a, self.max_acc)
        else:
            return a

    def fill_state_task_specific(self, state):
        super().fill_state_task_specific(state)
        state[self.observation_index['agent_vel_x']] = self.agent_vel_x
        state[self.observation_index['agent_vel_y']] = self.agent_vel_y

    def reset_agent(self):
        super().reset_agent()
        self.reset_agent_vel()

    def reset_agent_vel(self):
        if self.do_randomize_agent_vel:
            agent_vel_norm = np.random.rand()*self.max_vel
            agent_vel_angle = np.random.rand()*2.*np.pi
            self.agent_vel_x = agent_vel_norm*np.cos(agent_vel_angle)
            self.agent_vel_y = agent_vel_norm*np.sin(agent_vel_angle)
        else:
            self.agent_vel_x = 0
            self.agent_vel_y = 0


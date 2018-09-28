# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

class PlotConfig(object):
    '''
    Plot parameters, optionally built from ExtraArgs objects
    '''

    def __init__(self, n_average=201, timesteps_per_iteration=1024):
        self.n_average = n_average
        self.color_rewards_ind = 'b'
        self.color_rewards_avg = 'r'
        self.color_rewards_std = 'b'
        self.label_y = 'Reward'
        self.set_timesteps_per_iteration(timesteps_per_iteration)

    @classmethod
    def from_extra_args(cls, extra_args):
        plot_config = cls(n_average=extra_args.plot_average,
                          timesteps_per_iteration=(extra_args.n_direct * extra_args.timesteps_per_actorbatch))
        return plot_config

    def set_timesteps_per_iteration(self, timesteps_per_iteration):
        self.timesteps_per_iteration = timesteps_per_iteration
        if self.timesteps_per_iteration == 1:
            self.label_x_iterations = 'Timesteps'
        else:
            self.label_x_iterations = 'Iterations [{0} timesteps]'.format(self.timesteps_per_iteration)


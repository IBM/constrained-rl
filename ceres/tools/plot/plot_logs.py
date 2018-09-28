# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import matplotlib.pyplot as plt
import numpy as np
import os

class PlotLogs(object):
    '''
    Base class for plotting reward plots, provided input logs
    '''

    default_seed_value = 'N/A'
    plot_as_rows = True

    def __init__(self, plot_config):
        self.plot_config = plot_config
        self.plots = []
        self.n_plots = 0
        super().__init__()

    def add_plot(self, title='', paths='', label='', color='k', skip_error=False):
        if not type(paths) == list:
            assert type(paths) == str
            paths = [paths]
        for i_path, path in enumerate(paths):
            assert os.path.exists(path), 'Experiment path does not exist: {0}'.format(path)
            if path[-1] == '/':
                paths[i_path] = path[:-1]
        try:
            path_sessions, suffix_sessions = self.load_paths(paths)
            plot_info = {
                'title': title,
                'label': label,
                'color': color,
                'path_sessions': path_sessions,
                'suffix_sessions': suffix_sessions,
            }
            self.plots.append(plot_info)
            print('Found logs for {0}: {1}'.format(' '.join(title.split('\n')), plot_info['path_sessions']))
            success = True
            self.n_plots += 1
        except Exception as e:
            print('Could not find logs for {0}'.format(' '.join(title.split('\n'))))
            success = False
            if skip_error:
                print('Skip')
            else:
                raise(e)
        return success


    def load_paths(self, paths=None):
        raise NotImplementedError('Implement this into your child class')

    def calc_plots(self):
        raise NotImplementedError('Implement this in your own child class')

    def plot(self, show=True):
        self.calc_plots()
        plt.figure()
        n_cols = self.n_plots
        max_elements = min([len(plot_info['t']) for plot_info in self.plots])
        reward_min = min([min(plot_info['rewards']) for plot_info in self.plots])
        reward_max = max([max(plot_info['rewards']) for plot_info in self.plots])
        y_lim = [reward_min - 0.10*(reward_max - reward_min),
                 reward_max + 0.10*(reward_max - reward_min)]
        y_lim = np.array(y_lim)
        x_vec_average_list = []
        y_vec_average_list = []
        for i_plot, plot_info in enumerate(self.plots):
            t_averaged = plot_info['t']
            rew_averaged = plot_info['rewards']
            if self.plot_as_rows:
                ax = plt.subplot(n_cols, 1, i_plot+1)
            else:
                ax = plt.subplot(1, n_cols, i_plot+1)
            x_vec = np.array(t_averaged)/self.plot_config.timesteps_per_iteration
            label_x = self.plot_config.label_x_iterations
            y_vec = np.array(rew_averaged)
            # Individual rewards
            x_plot = x_vec
            y_plot = y_vec
            plt.plot(x_plot, y_plot, self.plot_config.color_rewards_ind, alpha=0.5, label='Episode reward')
            # Compute moving average
            x_vec_select = x_vec
            y_vec_average = self.calc_moving_average(y_vec, n=self.plot_config.n_average)
            x_vec_average_list.append(x_vec_select)
            y_vec_average_list.append(y_vec_average)
            # Standard deviation
            y_vec_std = self.calc_moving_std(y_vec, n=self.plot_config.n_average)
            y_vec_average_minus_std = y_vec_average - y_vec_std
            y_vec_average_plus_std = y_vec_average + y_vec_std
            x_plot = x_vec_select
            y_plot_minus = y_vec_average_minus_std
            y_plot_plus = y_vec_average_plus_std
            plt.fill_between(x_plot, y_plot_minus, y_plot_plus, facecolor=self.plot_config.color_rewards_std, alpha=1., label='Standard deviation')
            # Average rewards
            x_plot = x_vec_select
            y_plot = y_vec_average
            plt.plot(x_plot, y_plot, self.plot_config.color_rewards_avg, alpha=1., label='Average reward')
            if self.plot_as_rows:
                plt.ylabel(self.plot_config.label_y)
                if i_plot == len(self.plots)-1:
                    plt.xlabel(label_x)
                else:
                    pass
            else:
                plt.xlabel(label_x)
                if i_plot == 0:
                    plt.ylabel(self.plot_config.label_y)
                else:
                    pass
            title_loc = plot_info['title']
            plt.title(title_loc)
            plt.ylim(y_lim)
        # Legends
        bottom_legend_artists = []
        bottom_legend_labels = []
        # Individual rewards
        reward_ind_artist = plt.Line2D((0, 1), (0, 0), alpha=0.5, color=self.plot_config.color_rewards_ind)
        bottom_legend_artists.append(reward_ind_artist)
        bottom_legend_labels.append('Episode reward')
        # Average rewards
        reward_avg_artist = plt.Line2D((0, 1), (0, 0), color=self.plot_config.color_rewards_avg)
        bottom_legend_artists.append(reward_avg_artist)
        bottom_legend_labels.append('Reward average')
        # Standard deviations
        reward_std_artist = plt.Line2D((0, 1), (0, 0), color=self.plot_config.color_rewards_std)
        bottom_legend_artists.append(reward_std_artist)
        bottom_legend_labels.append('Reward std. dev.')
        ax.legend(bottom_legend_artists,
                  bottom_legend_labels,
                  loc='lower center',
                  fancybox=True,
                  ncol=3)

        # Plot average rewards in the same graph
        plt.figure()
        for i_plot, plot_info in enumerate(self.plots):
            x_vec = x_vec_average_list[i_plot]
            y_vec = y_vec_average_list[i_plot]
            plt.plot(x_vec, y_vec, label=plot_info['label'], color=plot_info['color'])
        plt.xlabel(label_x)
        plt.ylabel(self.plot_config.label_y)
        plt.title('Average rewards')
        plt.legend()

        if show:
            plt.show()

    def calc_moving_average(self, a, n=3, fill=True) :
        assert n % 2 == 1, 'Number of samples to average must be odd'
        assert len(a) >= n, 'Not enough samples to average: {0} vs {1}'.format(len(a), n)
        if fill:
            assert len(a) > 1.5*n, 'Not enough samples to fill'
        if n == 1:
            return a
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        res = ret[n - 1:] / n
        if fill:
            add_el = len(a) - len(res)
            add_left = int((n-1)/2)
            add_right = add_left
            el_left = []
            for _i in range(add_left):
                subvec = a[_i:_i + n]
                assert len(subvec) > 0
                el_left.append(np.mean(subvec))
            el_right = []
            for _i in range(len(a)-add_right, len(a)):
                subvec = a[-n+_i:_i]
                assert len(subvec) > 0
                el_right.append(np.mean(subvec))
            res = el_left + list(res) + el_right
            res = np.array(res)
            assert len(res) == len(a)
        return res


    def calc_moving_std(self, a, n=3, fill=True) :
        assert n % 2 == 1, 'Number of samples to average must be odd'
        n_total = len(a)
        res = np.zeros(n_total)
        if n == 1:
            return res
        n_side = int((n-1)/2)
        assert n_side > 0
        for i_center in range(n_total):
            i_left  = max(0, i_center - n_side)
            i_right = min(n_total-1, i_center + n_side)
            res[i_center] = np.std(a[i_left:(i_right+1)])
        return res






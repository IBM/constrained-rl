# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import matplotlib.pyplot as plt
import numpy as np
import os
import ast
from ceres.tools.plot import PlotConfig, PlotLogs

class PlotLogsBaselines(PlotLogs):
    '''
    Load (multiple) baselines logs and plots rewards together with useful statistics.
    Supports logs distributed across multiple processes and seeds.
    '''

    key_ep_reward = 'r'
    key_ep_length = 'l'
    key_ep_time   = 't'
    keys_ep = [key_ep_time, key_ep_length, key_ep_reward]

    def load_paths(self, paths=None):
        '''
        Find log paths by looking for directories of the form "worker_*"
        '''
        assert type(paths) == list
        suffix_sessions = {}
        target_dirname_0 = {}
        target_worker_dir_default = 'worker_0'
        for _i, _d in enumerate(paths):
            _d_basename = os.path.basename(_d)
            if 'worker' in _d_basename:
                target_worker_dir = '_'.join(_d_basename.split('_')[:2])
            else:
                target_worker_dir = target_worker_dir_default
            if _d_basename[:len(target_worker_dir)] == target_worker_dir:
                target_dirname_0[_i] = _d_basename
                suffix_sessions[_i] = _d_basename[len(target_worker_dir):]
            else:
                target_dirname_0[_i] = target_worker_dir
                suffix_sessions[_i] = ''

        path_sessions = {}
        for i_path, path_loc in enumerate(paths):
            path_sessions[i_path] = path_loc
        # For each directory in path_sessions, look for most recent worker_0 directory
        for _i, _d in path_sessions.items():
            target_dirname = target_dirname_0[_i]
            if os.path.basename(_d) == target_dirname:
                continue
            subdirs = [subdir for subdir in os.listdir(_d) if os.path.isdir(os.path.join(_d, subdir))]
            assert len(subdirs) > 0, 'Could not find any subdirectory in {0}'.format(_d)
            if target_dirname in subdirs:
                path_sessions[_i] = os.path.join(_d, target_dirname)
            else: # Take the most recent directory
                subdirs.sort()
                found_target_dir = False
                for subdir in reversed(subdirs):
                    path_xp = os.path.join(_d, subdir, target_dirname)
                    if os.path.exists(path_xp):
                        path_sessions[_i] = path_xp
                        found_target_dir = True
                        break
                assert found_target_dir, 'Could not find {0} directory in {1}'.format(target_dirname, path_xp)
        # Finally, return folder that contains worker_0
        for _i, _d in path_sessions.items():
            assert os.path.basename(_d) == target_dirname_0[_i]
            path_sessions[_i] = os.path.join(_d, os.pardir)
        return path_sessions, suffix_sessions

    def calc_plots(self):
        '''
        Build plots from session paths
        '''
        for i_plot, plot_info in enumerate(self.plots):
            path_sessions = plot_info['path_sessions']
            suffix_sessions = plot_info['suffix_sessions']
            plot_info['t'], plot_info['rewards'] = self.calc_plot_sessions(path_sessions, suffix_sessions)

    def calc_plot_sessions(self, path_sessions, suffix_sessions):
        '''
        Calculate reward plots across multiple sessions (e.g., seeds)
        '''
        t_sessions = []
        rewards_sessions = []
        lengths_sessions = []
        for i_session, (seed, path_session) in enumerate(path_sessions.items()):
            t_session, rewards_session, lengths_session = self.calc_plot_workers(path_session, suffix_session=suffix_sessions[seed])
            t_sessions.append(t_session)
            rewards_sessions.append(rewards_session)
            lengths_sessions.append(lengths_session)
            if i_session == 0:
                len_min = len(t_session)
            else:
                len_min = min(len_min, len(t_session))
        assert len_min > 0
        n_steps_sessions = [np.cumsum(x) for x in lengths_sessions]
        t_averaged = list(n_steps_sessions[0])
        rewards_averaged = list(rewards_sessions[0])
        for t_worker, rewards_worker in zip(n_steps_sessions[1:], rewards_sessions[1:]):
            t_averaged, [rewards_averaged] = self.combine_logs_xy(t_averaged, [rewards_averaged], t_worker, [rewards_worker])
        return t_averaged, rewards_averaged

    def parse_worker_monitor_csv(self, path_monitor, n_workers=1):
        '''
        Parse baselines monitor files
        '''
        with open(path_monitor, 'r') as f:
            lines = f.read().splitlines()
        header = lines[0]
        labels = lines[1]
        labels = labels.split(',')
        i_key = {_k: labels.index(_k) for _k in self.keys_ep}
        lines = lines[2:]
        res = {_k: [] for _k in self.keys_ep}
        n_ep = len(lines)
        for i_episode in range(n_ep):
            line = lines[i_episode]
            ep_info = line.split(',')
            res[self.key_ep_reward].append(float(ep_info[i_key[self.key_ep_reward]]))
            res[self.key_ep_time].append(float(ep_info[i_key[self.key_ep_time]]))
            res[self.key_ep_length].append(int(ep_info[i_key[self.key_ep_length]]))
        continue_index = 0
        continue_path = path_monitor + '.continue{0}'.format(continue_index)
        while os.path.exists(continue_path):
            raise NotImplementedError()
        t = res[self.key_ep_time]
        rewards = res[self.key_ep_reward]
        lengths = res[self.key_ep_length]
        return t, rewards, lengths

    def combine_logs_xy(self, x1, y1_list, x2, y2_list):
        '''
        Combine (x1, y1) with (x2, y2) sorting on increasing elements of (x1, x2)
        '''
        n_x1 = len(x1)
        n_x2 = len(x2)
        x = np.zeros(n_x1 + n_x2)
        y_list = [np.zeros(n_x1 + n_x2) for _ in y1_list]
        i_x1 = 0
        i_x2 = 0
        i_x = 0
        while (i_x1 < n_x1) and (i_x2 < n_x2):
            if x1[i_x1] < x2[i_x2]:
                x[i_x] = x1[i_x1]
                for (y, y1) in zip(y_list, y1_list):
                    y[i_x] = y1[i_x1]
                i_x1 += 1
            else:
                x[i_x] = x2[i_x2]
                for (y, y2) in zip(y_list, y2_list):
                    y[i_x] = y2[i_x2]
                i_x2 += 1
            i_x += 1
        for _i, _x in enumerate(x1[i_x1:]):
            x[i_x + _i] = _x
            for (y, y1) in zip(y_list, y1_list):
                y[i_x + _i] = y1[i_x1 + _i]
        for _i, (_x, _y) in enumerate(zip(x2[i_x2:], y2[i_x2:])):
            x[i_x + _i] = _x
            for (y, y2) in zip(y_list, y2_list):
                y[i_x + _i] = y2[i_x2 + _i]
        return x, y_list

    def combine_logs_xyz(self, x1, y1, z1, x2, y2, z2):
        '''
        Combine (x1, y1, z1) with (x2, y2, z2) sorting on increasing elements of (x1, x2)
        '''
        x, [y, z] = self.combine_logs_xy(x1, [y1, z1], x2, [y2, z2])
        return x, y, z

    def combine_workers(self, t_workers, rewards_workers, lengths_workers):
        '''
        Combine individual worker reward sequences into a single reward sequence
        '''
        t = t_workers[0]
        rewards = rewards_workers[0]
        lengths = lengths_workers[0]
        for t_worker, rewards_worker, lengths_worker in zip(t_workers[1:], rewards_workers[1:], lengths_workers[1:]):
            t, rewards, lengths = self.combine_logs_xyz(t, rewards, lengths, t_worker, rewards_worker, lengths_worker)
        return t, rewards, lengths

    def calc_plot_workers(self, path_session, suffix_session=''):
        '''
        Parse logs across workers and build a reward sequence
        '''
        # Load rewards across workers
        path_worker_logs = []
        print('Processing session {0}'.format(path_session))
        for _d in os.listdir(path_session):
            if 'worker' in _d:
                worker_str_suffix = _d[-len(suffix_session):]
                if len(suffix_session) > 0:
                    if worker_str_suffix != suffix_session:
                        print('  (ignore path {0}: does not contain suffix {1})'.format(_d, suffix_session))
                        continue
                worker_monitor_dir = os.path.join(path_session, _d)
                worker_monitor_path = os.path.join(worker_monitor_dir, 'monitor.csv')
                assert os.path.isfile(worker_monitor_path), 'Could not find logs at path: {0}'.format(worker_monitor_path)
                path_worker_logs.append(worker_monitor_path)
        n_workers = len(path_worker_logs)
        t_workers = []
        rewards_workers = []
        lengths_workers = []
        for path_monitor_json in path_worker_logs:
            print('  {0}'.format(path_monitor_json))
            t_worker, rewards_worker, lengths_worker = self.parse_worker_monitor_csv(path_monitor_json, n_workers=n_workers)
            t_workers.append(t_worker)
            rewards_workers.append(rewards_worker)
            lengths_workers.append(lengths_worker)
        t, rewards, lengths = self.combine_workers(t_workers, rewards_workers, lengths_workers)
        return t, rewards, lengths


def main():
    from ceres.tools.io import ExtraArgs
    extra_args = ExtraArgs()
    plot_config = PlotConfig.from_extra_args(extra_args)
    plotter = PlotLogsBaselines(plot_config)

    assert len(extra_args.plot_path) > 0
    color_list = ['g', 'b', 'r', 'k']
    path_info_dict = {}
    path_list = []
    title_list = []
    for i_path, path_info in enumerate(extra_args.plot_path):
        if '=' in path_info:
            title, path_loc = path_info.split('=')
        else:
            path_loc = path_info
            title = '{0}:{1}'.format(i_path, os.path.basename(path_loc))
        if not (title in title_list):
            title_list.append(title)
            path_info_dict[title] = []
        path_info_dict[title].append(path_loc)
    if len(color_list) < len(title_list):
        for _ in range(len(title_list) - len(color_list)):
            color_random = np.random.rand(3)
            color_list.append(color_random)
    for i_plot, title in enumerate(title_list):
        plotter.add_plot(title=title,
                                paths=path_info_dict[title],
                                label=title,
                                color=color_list[i_plot],
                                skip_error=False)

    plotter.plot(show=True)



if __name__ == '__main__':
    main()

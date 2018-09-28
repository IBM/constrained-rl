# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as pticf
import time
import os

class NetworkSaver(object):
    '''
    A simple class implementing save and restore functions for neural networks in Tensorflow
    '''

    model_basename = 'model'

    def __init__(self, network_id):
        self.network_id = network_id
        self.tf_var_prefix = '{0}/'.format(self.network_id)

    def get_var_name_mapping(self, backup_network_id=None):
        '''
        Return a dict associating this network's variable names to trainable tensors.
        Optional argument backup_network_id allows loading weights that were saved under a different name from network_id
        '''
        var_name_mapping = {}
        for v in tf.trainable_variables():
            if self.tf_var_prefix == v.name[:len(self.tf_var_prefix)]:
                v_name_train = v.name
                if backup_network_id is not None:
                    v_name_train = backup_network_id.join(v_name_train.split(self.network_id))
                v_name_train = v_name_train.split(':')[0]
                var_name_mapping[v_name_train] = v
        return var_name_mapping

    def restore_model(self, path_restore, session=None, backup_network_id=None, verbose=True):
        '''
        Restore trained weights
        '''
        if session is None:
            assert hasattr(self, 'session'), 'Either pass session as argument or set during saver initialization'
            session = self.session
        var_name_mapping = self.get_var_name_mapping(backup_network_id=backup_network_id)
        saver = tf.train.Saver(var_name_mapping)
        path_model = self.get_latest_model(path_restore, model_basename=self.model_basename)
        try:
            saver.restore(session, path_model)
            print('Restored network: {0}'.format(path_model))
        except Exception as e:
            print('Could not restore {0} from checkpoint: {1}'.format(type(self), path_model))
            print('This is the content of the checkpoint file:')
            pticf(file_name=path_model, tensor_name='', all_tensors=False)
            raise e

    def init_saver(self, path_backup_dir, session=None, max_to_keep=1):
        '''
        Build backup path for future use
        '''
        if session is not None:
            self.session = session
        os.makedirs(path_backup_dir, exist_ok=True)
        self.path_model = os.path.join(path_backup_dir, self.model_basename)
        var_name_mapping = self.get_var_name_mapping()
        var_to_save = [_e for _k, _e in var_name_mapping.items()]
        self.saver = tf.train.Saver(var_to_save, max_to_keep=max_to_keep)

    def save_model(self, global_step=None, verbose=True, path_model=None, session=None):
        '''
        Save model from given session to given path if specified,
        else take these from previous init_saver call
        '''
        if path_model is None:
            assert hasattr(self, 'path_model'), 'Specify path_model or set it at initialization'
            path_model = self.path_model
        if session is None:
            assert hasattr(self, 'path_model'), 'Specify session or set it at initialization'
            session = self.session
        if global_step is None:
            self.saver.save(session, self.path_model)
        else:
            self.saver.save(session, self.path_model, global_step=global_step)
        if verbose:
            print('Save network: {0}'.format(path_model))

    @classmethod
    def get_latest_model(cls, path_model, model_basename='model'):
        '''
        Check for files of the form <model_basename>-<iter> and return the most recent
        '''
        model_index_extension = '.index'
    
        if os.path.isdir(path_model):
            path_model = os.path.join(path_model, model_basename)
        path_model_full = path_model + model_index_extension
    
        if not os.path.isfile(path_model_full):
            # Check for files of the form model-1000.index
            path_model_dirname = os.path.dirname(path_model_full)
            model_basename = os.path.basename(path_model)
            files_in_dir = os.listdir(path_model_dirname)
            path_model_candidates = []
            model_iter_numbers = []
            for _f in files_in_dir:
                if _f[:len(model_basename)] != model_basename:
                    continue
                if _f[-len(model_index_extension):] != model_index_extension:
                    continue
                _f_base = _f[:-len(model_index_extension)]
                _f_base_split = _f_base.split('-')
                assert (_f_base_split[0] == model_basename) and (len(_f_base_split) == 2), 'Invalid file {0}, expected {1}-<iter>{2}'.format(_f, model_basename, model_index_extension)
                i_iter = int(_f_base_split[1])
                model_iter_numbers.append(i_iter)
            assert len(model_iter_numbers) > 0, 'Cannot find any model candidate in directory {0}'.format(path_model_dirname)
            model_basename = '{0}-{1}'.format(model_basename, max(model_iter_numbers))
            path_model = os.path.join(path_model_dirname, model_basename)
            path_model_full = path_model + model_index_extension
            assert os.path.isfile(path_model_full), 'Model backup file does not exist: {0}'.format(path_model_full)
        return path_model

    def build_model(self, *args, **kwargs):
        raise NotImplementedError('Implement build_model in child classes')

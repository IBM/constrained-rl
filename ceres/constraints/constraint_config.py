# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import os
from ceres.tools.io.h5_helper import save_dict_as_h5, load_dict_from_h5

class ConstraintConfig(object):
    '''
    Constraint network configuration with save and restore functions
    '''

    valid_param = ['mlp_hidden_layers',
                   'n_ineq',
                   'loss_weights',
                   'spherical_coordinates',
                   'normalize_ineq_mat',
                   'predict_interior_point',
                   'interior_point_margin_min',
                   'interior_point_margin_max',
                   'interior_point_max']
    cnet_config_filename = 'cnet_config.h5'

    def __init__(self, **kwargs):
        self.set_default()
        self.set(**kwargs)

    def set_default(self):
        self.spherical_coordinates = False
        self.normalize_ineq_mat = False
        self.predict_interior_point = False
        self.interior_point_margin_min = 0.
        self.interior_point_margin_max = 0.
        self.interior_point_max = 0.
        self.loss_weights = {}

    def set(self, **kwargs):
        for key, value in kwargs.items():
            assert key in self.valid_param, 'Invalid parameter type {0}'.format(key)
            setattr(self, key, value)

    def save(self, path_save):
        d = self.__dict__
        assert os.path.isdir(path_save), 'Config save function only takes a directory as input'
        path_save = os.path.join(path_save, self.cnet_config_filename)
        save_dict_as_h5(d, path_save, verbose=True)

    @classmethod
    def from_backup(cls, path_save):
        if os.path.isdir(path_save):
            path_cnet_dir = path_save
        else:
            path_cnet_dir = os.path.dirname(path_save)
        path_cnet_config = os.path.join(path_cnet_dir, cls.cnet_config_filename)
        d = load_dict_from_h5(path_cnet_config, verbose=False)
        cnet_config = cls(**d)
        return cnet_config

    @classmethod
    def from_extra_args(cls, args):
        cnet_config = cls(mlp_hidden_layers=args.cnet_hidden_layers,
                          n_ineq=args.cnet_n_ineq,
                          loss_weights=args.cnet_loss_weights,
                          spherical_coordinates=args.cnet_spherical_coordinates,
                          normalize_ineq_mat=args.cnet_normalize_ineq_mat,
                          predict_interior_point=args.cnet_predict_interior_point,
                          interior_point_margin_min=args.cnet_interior_point_margin_min,
                          interior_point_margin_max=args.cnet_interior_point_margin_max,
                          interior_point_max=args.cnet_interior_point_max,
        )
        return cnet_config



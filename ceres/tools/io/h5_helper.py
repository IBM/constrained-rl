# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import os
import h5py

'''
Helper functions for writing and loading data in the HDF5 format
'''

def save_dict_as_h5(d, path_save, confirm_overwrite=True, verbose=False):
    assert type(d) == dict, 'Invalid dictionary argument: {0}'.format(d)
    assert type(path_save) == str, 'Invalid path argument: {0}'.format(path_save)
    assert os.path.isdir(os.path.dirname(path_save)), 'Directory for save path does not exist: {0}'.format(path_save)
    if os.path.isfile(path_save):
        if confirm_overwrite:
            if input('File exists: {0}\nOverwrite?[y/N]\n'.format(path_save)).lower() != 'y':
                print('Cancel write')
                return False
        os.remove(path_save)
    # Check that no nested dictionary
    with h5py.File(path_save, 'w') as h5f:
        write_dict(h5f, d)
    if verbose:
        print('Wrote backup: {0}'.format(path_save))
    return True

def write_dict(h5f, d):
    for _k, _v in d.items():
        if type(_v) == dict:
            grp = h5f.create_group(_k)
            write_dict(grp, _v)
        else:
            h5f.create_dataset(_k, data=_v)

def load_dict_from_h5(path_save, verbose=False):
    d = {}
    with h5py.File(path_save, 'r') as h5f:
        read_dict(h5f, d)
    if verbose:
        print('Loaded {0} from backup: {0}'.format(','.join(d.keys())))
    return d

def read_dict(h5f, d):
    for _k, _v in h5f.items():
        if isinstance(_v, h5py.Dataset):
            d[_k] = _v[()]
        else:
            assert isinstance(_v, h5py.Group)
            d[_k] = {}
            read_dict(_v, d[_k])


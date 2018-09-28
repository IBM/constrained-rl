# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import tensorflow as tf
from .network_saver import NetworkSaver

class NetworkSaverMLP(NetworkSaver):
    '''
    A simple multilayer perceptron with save / restore functions
    '''

    def build_model(self, observation, n_outputs,
                    hidden_layers,
                    kernel_initializer,
                    activation_hidden):
        var_names_begin = [_v.name for _v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        self.hidden_layers = hidden_layers
        assert len(self.hidden_layers) > 0
        self.dense_layers = []
        last_layer = observation
        self.layer_names = []
        for i_layer, layer_size in enumerate(self.hidden_layers):
            layer_name = '{0}dense_{1}'.format(self.tf_var_prefix, i_layer)
            self.layer_names.append(layer_name)
            dense_layer = tf.layers.dense(inputs=last_layer, units=layer_size, activation=None, kernel_initializer=kernel_initializer, name=layer_name)
            dense_layer = activation_hidden(dense_layer)
            self.dense_layers.append(dense_layer)
            last_layer = dense_layer
        layer_name = '{0}dense_{1}'.format(self.tf_var_prefix, 'output')
        self.layer_names.append(layer_name)
        output_layer = tf.layers.dense(inputs=last_layer, units=n_outputs, kernel_initializer=kernel_initializer, name=layer_name)
        self.dense_layers.append(output_layer)
        var_names_end = [_v.name for _v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        self.network_var_names = [_v for _v in var_names_end if not (_v in var_names_begin)]
        return output_layer


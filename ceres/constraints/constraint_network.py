# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import tensorflow as tf
import numpy as np
from .constraint_loss import ConstraintLoss
from .constraint_config import ConstraintConfig
from ceres.tools.math.spherical_coordinates import SphericalCoordinates

class ConstraintNetwork(object):
    '''
    Learn and predict state-dependent constraints on actions
    '''

    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_space_pm = np.array([0.5*(high - low) for (low, high) in zip(self.action_space.low, self.action_space.high)])
        self.action_space_mid = np.array([0.5*(high + low) for (low, high) in zip(self.action_space.low, self.action_space.high)])
        self.n_obs = self.observation_space.shape[0]
        self.n_act = self.action_space.shape[0]
        self.config = config
        self.n_ineq = self.config.n_ineq
        self.init_default()

        # Constraints are of the form G x <= h, with G of size n_ineq x n_act and h of size n_ineq x 1
        self.init_prediction_size()

        # Build model
        self.observation = tf.placeholder(dtype=tf.float32, shape=(None, self.n_obs), name='observation')
        self.batch_size = tf.shape(self.observation)[0]
        self.batch_size_float = tf.cast(self.batch_size, dtype=tf.float32)
        self.ones_batch = tf.ones([self.batch_size], dtype=tf.float32)
        self.zeros_batch = tf.zeros([self.batch_size], dtype=tf.float32)

        self.input_layer = self.observation
        self.output_layer = self.build_model()
        self.init_regularization()

        # Transform output layer into constraint matrices G, h
        self.ineq_mat_params, self.ineq_vec_params = self.split_predictions(self.output_layer)
        self.ineq_mat, self.ineq_vec = self.build_ineq(self.ineq_mat_params, self.ineq_vec_params)

        ##################################
        # Individual actions with labels #
        ##################################

        # Reference action and action indicator for training
        self.action = tf.placeholder(dtype=tf.float32, shape=(None, self.n_act), name='action')
        self.action_indicator = tf.placeholder(dtype=tf.float32, shape=(None), name='action_indicator')
        self.action_tensor = tf.expand_dims(self.action, -1)

        # Count positive and negative demonstrations
        self.is_positive = self.action_indicator
        self.is_negative = 1. - self.is_positive
        self.n_positive = tf.cast(tf.reduce_sum(self.is_positive), dtype=tf.int32)
        self.n_negative = tf.cast(tf.reduce_sum(self.is_negative), dtype=tf.int32)

        # Build constraint margins and scores
        self.ineq_diff, self.ineq_satisfaction_margin, self.ineq_violation_margin = self.build_ineq_margins(self.ineq_mat, self.ineq_vec, self.action_tensor)
        self.n_positive_satisfied, self.n_positive_violated, self.n_negative_satisfied, self.n_negative_violated = self.calc_constraint_score()

        # Construct training losses
        self.loss, self.losses = self.init_training_loss()

    def init_prediction_size(self):
        '''
        Initialize the number of output variables, depending on: number of constraints; spherical coordinates; interior point prediction
        '''
        if self.config.spherical_coordinates:
            assert self.n_act >= 2
            self.n_ineq_mat_params = self.n_ineq * (self.n_act - 1)
        else:
            self.n_ineq_mat_params = self.n_ineq * self.n_act
        self.n_ineq_vec_params = self.n_ineq
        if self.config.predict_interior_point:
            # Add an action x0 to the constraint predictions
            self.n_ineq_vec_params += self.n_act
        self.n_outputs = self.n_ineq_mat_params + self.n_ineq_vec_params

    def init_training_loss(self):
        '''
        Defer the construction of constraint loss terms to a ConstraintLoss object
        '''
        constraint_loss = ConstraintLoss(self)
        total_loss = constraint_loss.total_loss
        losses = constraint_loss.losses
        return total_loss, losses

    def calc_constraint_score(self):
        '''
        Count the number of positive / negative demonstrations that satisfy / violate the constraint
        '''
        positive_loss = tf.reduce_max(self.ineq_violation_margin, axis=1)
        positive_loss = tf.multiply(self.action_indicator, positive_loss)
        n_positive_violated = tf.where(positive_loss > 0., self.ones_batch, self.zeros_batch)
        n_positive_violated = tf.cast(tf.reduce_sum(n_positive_violated), dtype=tf.int32)
        n_positive_satisfied = self.n_positive - n_positive_violated
        negative_loss = tf.reduce_min(self.ineq_satisfaction_margin, axis=1)
        negative_loss = tf.multiply(self.is_negative, negative_loss)
        n_negative_satisfied = tf.where(negative_loss > 0., self.ones_batch, self.zeros_batch)
        n_negative_satisfied = tf.cast(tf.reduce_sum(n_negative_satisfied), dtype=tf.int32)
        n_negative_violated = self.n_negative - n_negative_satisfied
        return n_positive_satisfied, n_positive_violated, \
               n_negative_satisfied, n_negative_violated

    def init_default(self):
        '''
        Default initialization and nonlinearity parameters
        '''
        self.initializer = tf.random_normal_initializer(mean=0., stddev=0.1)
        self.activation_common = tf.nn.relu

    def split_predictions(self, output_layer):
        '''
        Split the output layer into inequality matrix and vector parameters
        '''
        # First n_ineq * self.n_act are G, remaining n_ineq are h
        ineq_mat_params = tf.slice(output_layer, [0, 0], [-1, self.n_ineq_mat_params])
        ineq_vec_params = tf.slice(output_layer, [0, self.n_ineq_mat_params], [-1, self.n_ineq_vec_params])
        return ineq_mat_params, ineq_vec_params

    def build_ineq(self, ineq_mat_params, ineq_vec_params):
        '''
        Transform ineq predicted parameters into solver-compatible matrices
        '''
        ineq_mat = self.build_ineq_mat(ineq_mat_params)
        ineq_vec = self.build_ineq_vec(ineq_vec_params, ineq_mat)
        return ineq_mat, ineq_vec

    def build_ineq_mat(self, ineq_mat_params):
        '''
        Reshape predicted parameters and optionally ensure ineq matrix normalization
        '''
        ineq_mat_epsilon = 1.e-7
        assert not (self.config.spherical_coordinates and self.config.normalize_ineq_mat), 'Cannot have simultaneously spherical coordinates and ineq mat normalization'
        if self.config.spherical_coordinates:
            ineq_mat = tf.reshape(ineq_mat_params, [-1, self.n_ineq, self.n_act-1])
            self.sc = SphericalCoordinates(self.n_act, input_angles=ineq_mat)
            ineq_mat = self.sc.output_unit_vec
        else:
            ineq_mat = tf.reshape(ineq_mat_params, [-1, self.n_ineq, self.n_act])
            if self.config.normalize_ineq_mat:
                # Make each line of G of unit norm
                norm_row = tf.norm(ineq_mat, ord='euclidean', axis=-1, keep_dims=True)
                norm_row += ineq_mat_epsilon
                norm_mat = tf.tile(norm_row, [1, 1, self.n_act])
                ineq_mat = tf.divide(ineq_mat, norm_mat)
        return ineq_mat

    def build_ineq_vec(self, ineq_vec_params, ineq_mat):
        '''
        Reshape predicted parameters and optionally build an interior point that satisfies all constraints
        '''
        ineq_vec = ineq_vec_params
        if self.config.predict_interior_point:
            # In this case, we split h_pred into x0 and h+
            # with x0 an action and h+ non negative.
            # Then: h = g*x0 + h+
            interior_point = tf.slice(ineq_vec, [0, 0], [-1, self.n_act])
            action_space_mid_tensor = tf.convert_to_tensor(self.action_space_mid, dtype=ineq_vec.dtype)
            action_space_mid_tensor = tf.expand_dims(action_space_mid_tensor, axis=0)
            action_space_mid_tensor = tf.tile(action_space_mid_tensor, [tf.shape(ineq_vec)[0], 1])
            action_space_pm_tensor = tf.convert_to_tensor(self.action_space_pm, dtype=ineq_vec.dtype)
            action_space_pm_tensor = tf.expand_dims(action_space_pm_tensor, axis=0)
            action_space_pm_tensor = tf.tile(action_space_pm_tensor, [tf.shape(ineq_vec)[0], 1])
            if self.config.interior_point_max >= 0.:
                # force the interior point to be in a given domain within the action space
                interior_point_low = action_space_mid_tensor - self.config.interior_point_max*action_space_pm_tensor
                interior_point_high = action_space_mid_tensor + self.config.interior_point_max*action_space_pm_tensor
                interior_point = tf.clip_by_value(interior_point, interior_point_low, interior_point_high)
            interior_point = tf.expand_dims(interior_point, axis=-1)
            ineq_vec_plus = tf.slice(ineq_vec, [0, self.n_act], [-1, -1])
            ineq_vec_plus = tf.nn.relu(ineq_vec_plus)
            zeros_like_ineq_vec_plus = tf.zeros_like(ineq_vec_plus)
            if self.config.interior_point_margin_min != 0.:
                # each row of h is a scalar, so we use the min action range as basis for the margin
                ineq_vec_plus_min_val = self.config.interior_point_margin_min * np.min(self.action_space_pm)
                ineq_vec_plus += ineq_vec_plus_min_val
            if self.config.interior_point_margin_max > 0.:
                # each row of h is a scalar, so we use the min action range as basis for the margin
                ineq_vec_plus_max_val = self.config.interior_point_margin_max * np.min(self.action_space_pm)
                ineq_vec_plus = tf.clip_by_value(ineq_vec_plus, 0., ineq_vec_plus_max_val)
            ineq_vec_interior_point = tf.matmul(ineq_mat, interior_point)
            ineq_vec_interior_point = tf.squeeze(ineq_vec_interior_point, axis=-1)
            ineq_vec = ineq_vec_interior_point + ineq_vec_plus
        else:
            interior_point = tf.zeros([1, self.n_act, 1], dtype=ineq_vec.dtype)
            interior_point = tf.tile(interior_point, [tf.shape(ineq_vec)[0], 1, 1])
        # Reshape into matrices
        self.interior_point = interior_point
        ineq_vec = tf.expand_dims(ineq_vec, axis=-1)
        return ineq_vec

    def build_ineq_margins(self, ineq_mat, ineq_vec, action_tensor, do_squeeze=True):
        '''
        Constraints are satisfied if Gx <= h, hence satisfaction margin = max(0, h - Gx)
        Constraints are violated if Gx > h, hence violation margin = max(0, Gx - h)
        '''
        ineq_diff = ineq_vec - tf.matmul(ineq_mat, action_tensor)
        ineq_satisfaction_margin = tf.nn.relu(ineq_diff)
        ineq_violation_margin = tf.nn.relu(-ineq_diff)
        if do_squeeze:
            ineq_diff = tf.squeeze(ineq_diff, axis=-1)
            ineq_satisfaction_margin = tf.squeeze(ineq_satisfaction_margin, axis=-1)
            ineq_violation_margin = tf.squeeze(ineq_violation_margin, axis=-1)
        return ineq_diff, ineq_satisfaction_margin, ineq_violation_margin

    def init_regularization(self):
        '''
        Find model weights for regularization
        '''
        gr = tf.get_default_graph()
        self.model_weights = {name: gr.get_tensor_by_name('{0}/kernel:0'.format(name)) for name in self.layer_names}

    def build_model(self):
        raise NotImplementedError('Implement build_model within child classes')


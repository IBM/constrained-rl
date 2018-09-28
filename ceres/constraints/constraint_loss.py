# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import tensorflow as tf

class ConstraintLoss(object):
    '''
    Define constraint network loss terms
    '''

    no_normalization = ['_loss_l2'] # add loss functions that do not require division by batch size

    def __init__(self, network):
        '''
        Loss terms are defined from constraint network variables
        '''
        self.network = network
        self.init_losses()
        self.init_total_loss()

    def init_losses(self):
        '''
        Initialize individual losses from a dictionary of loss names and weights,
        e.g., loss_weights = {'l2': 0.0001, 'positive_violation_max': 1.0, 'negative_satisfaction_min': 1.0}
        will call loss functions '_loss_l2', '_loss_positive_violation_max' and '_loss_negative_satisfaction_min'
        '''
        self.losses = {}
        for loss_name, loss_weight in self.network.config.loss_weights.items():
            if loss_weight != 0.:
                loss_func_name = '_loss_{0}'.format(loss_name)
                assert hasattr(self, loss_func_name), 'Undefined loss function {0}'.format(loss_func_name)
                loss = getattr(self, loss_func_name)()
                if not loss_func_name in self.no_normalization:
                    loss = loss / self.network.batch_size_float
                self.losses[loss_name] = loss_weight * loss

    def init_total_loss(self):
        '''
        Sum up individual loss terms if available, otherwise zero
        '''
        loss_list = [v for k, v in self.losses.items()]
        if len(loss_list) == 0:
            print('Warning: no CNet loss defined')
            self.total_loss = 0.
        else:
            self.total_loss = tf.add_n(loss_list)

    def _loss_l2(self):
        '''
        L2 norm of the neural network weights
        '''
        assert len(self.network.model_weights) > 0
        loss = tf.add_n([tf.nn.l2_loss(w) for _k, w in self.network.model_weights.items()])
        return loss

    def _loss_positive_violation_max(self, order=1):
        '''
        Maximum violation margin for positive demonstrations, supports squaring
        '''
        loss = self.network.ineq_violation_margin
        loss = tf.reduce_max(loss, axis=1)
        if order == 2:
            loss = tf.square(loss)
        loss = tf.multiply(self.network.is_positive, loss)
        loss = tf.reduce_sum(loss)
        return loss

    def _loss_pvm(self, order=1):
        '''
        Shortname for positive violation max
        '''
        pvm_loss = self._loss_positive_violation_max(order=order)
        return pvm_loss

    def _loss_pvm_1d(self):
        '''
        Positive violation max, 1st order
        '''
        return self._loss_pvm(order=1)

    def _loss_pvm_2d(self):
        '''
        Positive violation max, squared
        '''
        return self._loss_pvm(order=2)

    def _loss_positive_violation_norm(self, order=1):
        '''
        Since we're seeking to zero all violation margins, we can minimize the total norm (L1 or L2)
        '''
        loss = self.network.ineq_violation_margin
        if order == 2:
            loss = tf.square(loss)
        else:
            assert order == 1, 'Only order 1 and 2 supported'
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.multiply(self.network.is_positive, loss)
        loss = tf.reduce_sum(loss)
        return loss

    def _loss_pvn(self, order=1):
        '''
        Shortname for positive violation norm
        '''
        pvn_loss = self._loss_positive_violation_norm(order=order)
        return pvn_loss

    def _loss_pvn_1d(self):
        '''
        Positive violation norm, L1 norm
        '''
        return self._loss_pvn(order=1)

    def _loss_pvn_2d(self):
        '''
        Positive violation norm, L2 norm
        '''
        return self._loss_pvn(order=2)

    def _loss_negative_satisfaction_min(self, order=1):
        '''
        Minimum satisfaction margin for negative demonstrations, supports squaring
        '''
        loss = self.network.ineq_satisfaction_margin
        loss = tf.reduce_min(loss, axis=1)
        if order == 2:
            loss = tf.square(loss)
        loss = tf.multiply(self.network.is_negative, loss)
        loss = tf.reduce_sum(loss)
        return loss

    def _loss_nsm(self, order=1):
        '''
        Shortname for negative satisfaction min
        '''
        nsm_loss = self._loss_negative_satisfaction_min(order=order)
        return nsm_loss

    def _loss_nsm_1d(self):
        '''
        Negative satisfaction min, 1st order
        '''
        return self._loss_nsm(order=1)

    def _loss_nsm_2d(self):
        '''
        Negative satisfaction min, squared
        '''
        return self._loss_nsm(order=2)

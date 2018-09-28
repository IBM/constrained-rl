# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from baselines.common import fmt_row, zipsame
import baselines.common.tf_util as U
import numpy as np
from mpi4py import MPI
from ceres.baselines.common import mpi_select
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments

class ConstraintTrainer(object):
    '''
    Everything to train a constraint network
    '''

    def __init__(self, extra_args, logger,
                 cnet, constraint_demonstration_buffer,
                 mpi_comm, mpi_rank, is_direct_policy,
                 cnet_recovery_id_in_direct_exchange_ids,
                 cnet_exchange_ids, n_exchange_processes,
                 adam_epsilon=1.e-5):
        self.extra_args = extra_args
        self.logger = logger
        self.cnet = cnet
        self.constraint_demonstration_buffer = constraint_demonstration_buffer
        self.mpi_comm = mpi_comm
        self.mpi_rank = mpi_rank
        self.is_direct_policy = is_direct_policy
        self.cnet_recovery_id_in_direct_exchange_ids = cnet_recovery_id_in_direct_exchange_ids
        self.cnet_exchange_ids = cnet_exchange_ids
        self.n_exchange_processes = n_exchange_processes

        self.cnet_comparison_loss_indices, self.cnet_loss_and_accuracy_names, \
                self.cnet_lossandgrad, self.cnet_adam, self.cnet_compute_losses_and_scores, \
                self.cnet_compute_scores = self.build_cnet_training_vars(adam_epsilon)
        self.init_interruption()

    def check_interruption(self, do_train_cnet, activation_probability):
        '''
        Disable constraint training if accuracy exceeds a given threshold for a given number of iterations,
        reenable it if accuracy is lower than another given threshold for another given number of iterations.
        '''
        if do_train_cnet:
            if activation_probability >= self.cnet_disable_threshold:
                self.cnet_switch_iters_so_far += 1
                if self.cnet_switch_iters_so_far == self.cnet_disable_iters:
                    do_train_cnet = False
                    self.cnet_switch_iters_so_far = 0
                    self.logger.log('Disable CNet training: achieved accuracy {0} for {1} iterations'.format(self.cnet_disable_threshold, self.cnet_disable_iters))
        else:
            if activation_probability <= self.cnet_enable_threshold:
                self.cnet_switch_iters_so_far += 1
                if self.cnet_switch_iters_so_far == self.cnet_enable_iters:
                    do_train_cnet = True
                    self.cnet_switch_iters_so_far = 0
                    self.logger.log('Re-enable CNet training: accuracy lower than {0} for {1} iterations'.format(self.cnet_enable_threshold, self.cnet_enable_iters))
        return do_train_cnet

    def init_interruption(self):
        '''
        Setup interruption criterion of the form <interruption_type>:<disable_threshold>:<disable_iters>:<enable_threshold>:<enable_iters>
        '''
        self.do_use_prior_accuracy_as_activation_probability = 'prior' in self.extra_args.adaptive_constraint_activation
        self.do_check_interrupt_cnet_training = len(self.extra_args.interrupt_constraint_training) > 0
        if self.do_check_interrupt_cnet_training:
            self.cnet_interruption_type, self.cnet_disable_threshold, self.cnet_disable_iters, self.cnet_enable_threshold, self.cnet_enable_iters = self.extra_args.interrupt_constraint_training.split(':')
            self.cnet_disable_threshold = float(self.cnet_disable_threshold)
            self.cnet_disable_iters = int(self.cnet_disable_iters)
            self.cnet_enable_threshold = float(self.cnet_enable_threshold)
            assert 0. <= self.cnet_disable_threshold <= 1., 'Disable threshold should be between 0 and 1 but got {0}'.format(self.cnet_disable_threshold)
            assert 0. <= self.cnet_enable_threshold <= 1., 'Enable threshold should be between 0 and 1 but got {0}'.format(self.cnet_enable_threshold)
            self.cnet_enable_iters = int(self.cnet_enable_iters)
            self.cnet_switch_iters_so_far = 0
        self.do_evaluate_cnet_losses_before = self.do_use_prior_accuracy_as_activation_probability or self.do_check_interrupt_cnet_training


    def init(self):
        '''
        Call this after initializing the constraint network, before training.
        '''
        self.cnet_adam.sync()
        self.cnet_newlosses_dummy, self.cnet_losses_and_scores_dummy, self.cnet_g_dummy, \
                self.cnet_scores_dummy = self.build_dummy_cnet_vars()

    def test_early_stop_classification_accuracy(self, i_epoch, positive_accuracy, negative_accuracy):
        '''
        Check constraint accuracy for positive and negative demonstrations
        '''
        do_stop_positive = positive_accuracy >= self.extra_args.early_stop_positive
        do_stop_negative = negative_accuracy   >= self.extra_args.early_stop_negative
        return do_stop_positive and do_stop_negative

    def calc_classification_accuracy(self, n_positive_satisfied, n_positive_violated, n_negative_satisfied, n_negative_violated):
        '''
        Calculate accuracy from demonstration numbers
        '''
        n_positive = n_positive_satisfied + n_positive_violated
        n_negative = n_negative_satisfied + n_negative_violated
        if n_positive > 0:
            test_loss_positive = 1. * n_positive_satisfied / n_positive
        else:
            test_loss_positive = 1.
        if n_negative > 0:
            test_loss_negative = 1. * n_negative_violated / n_negative
        else:
            test_loss_negative = 1.
        test_loss_mean = 0.5 * (test_loss_positive + test_loss_negative)
        test_classification_accuracy = [test_loss_positive, test_loss_negative, test_loss_mean]
        return test_classification_accuracy


    def test_improvement_classification_accuracy(self, mean_cnet_losses, best_cnet_losses,
                                                 mean_classification_accuracy, best_classification_accuracy):
        '''
        Check if training loss has decreased or classification accuracy has increased
        '''
        if self.extra_args.cnet_improvement_metric == 'mean_accuracy':
            ### Improvement = higher mean accuracy
            improvement = mean_classification_accuracy[2] > best_classification_accuracy[2]
        elif self.extra_args.cnet_improvement_metric == 'min_accuracy':
            ### Improvement = higher min accuracy
            improvement = min(mean_classification_accuracy[:2]) > min(best_classification_accuracy[:2])
        elif self.extra_args.cnet_improvement_metric == 'total_loss':
            ### Improvement = lower total loss
            improvement = sum(mean_cnet_losses) < sum(best_cnet_losses)
        elif self.extra_args.cnet_improvement_metric == 'mean_loss':
            ### Improvement = lower mean loss
            improvement = np.average([mean_cnet_losses[_i] for _i in self.cnet_comparison_loss_indices]) < np.average([best_cnet_losses[_i] for _i in self.cnet_comparison_loss_indices])
        elif self.extra_args.cnet_improvement_metric == 'max_loss':
            ### Improvement = lower max loss
            improvement = np.max([mean_cnet_losses[_i] for _i in self.cnet_comparison_loss_indices]) < np.max([best_cnet_losses[_i] for _i in self.cnet_comparison_loss_indices])
        else:
            raise ValueError('Invalid argument cnet_improvement_metric={0}'.format(self.extra_args.cnet_improvement_metric))
        return improvement

    def build_dummy_cnet_vars(self):
        '''
        Compute dummy update parameters
        '''
        batch_cnet_observations_dummy = np.zeros([1, self.cnet.observation_space.shape[0]])
        batch_cnet_actions_dummy = np.zeros([1, self.cnet.action_space.shape[0]])
        batch_cnet_action_indicators_dummy = np.zeros([1])
        *cnet_newlosses_dummy, cnet_g_dummy = self.cnet_lossandgrad(batch_cnet_observations_dummy, batch_cnet_actions_dummy, batch_cnet_action_indicators_dummy)
        cnet_scores_dummy = [1., 1., 1.]
        cnet_losses_and_scores_dummy = cnet_newlosses_dummy + cnet_scores_dummy
        cnet_g_dummy *= 0.
        return cnet_newlosses_dummy, cnet_losses_and_scores_dummy, cnet_g_dummy, cnet_scores_dummy

    def build_cnet_training_vars(self, adam_epsilon):
        '''
        Build training var optimizers from CNet inputs and losses
        '''
        cnet_lossandgrad_inputs = [self.cnet.observation, self.cnet.action, self.cnet.action_indicator]
        cnet_loss_names = sorted(self.cnet.losses.keys())
        cnet_comparison_loss_indices = [_k for (_k, _v) in enumerate(cnet_loss_names) if _v != 'l2']
        cnet_loss_and_accuracy_names = cnet_loss_names + ['positive satisfied', 'negative violated', 'any correct']
        cnet_losses = [self.cnet.losses[_k] for _k in cnet_loss_names]
        cnet_total_loss = self.cnet.loss
        cnet_var_list = [_v for (_k, _v) in self.cnet.get_var_name_mapping().items()]
        cnet_total_loss_grad = U.flatgrad(cnet_total_loss, cnet_var_list)
        cnet_lossandgrad_outputs = cnet_losses + [cnet_total_loss_grad]
        cnet_lossandgrad = U.function(cnet_lossandgrad_inputs, cnet_lossandgrad_outputs)
        cnet_adam = MpiAdam(cnet_var_list, epsilon=adam_epsilon)
        cnet_scores = [self.cnet.n_positive_satisfied, self.cnet.n_positive_violated, self.cnet.n_negative_satisfied, self.cnet.n_negative_violated]
        cnet_losses_and_scores = cnet_losses + cnet_scores
        cnet_compute_losses_and_scores = U.function(cnet_lossandgrad_inputs, cnet_losses_and_scores)
        cnet_compute_scores = U.function(cnet_lossandgrad_inputs, cnet_scores)
        return cnet_comparison_loss_indices, cnet_loss_and_accuracy_names, cnet_lossandgrad, cnet_adam, cnet_compute_losses_and_scores, cnet_compute_scores

    def evaluate_cnet_losses(self, do_log=True):
        '''
        Compute losses alone, without gradients
        '''
        losses = []
        if self.do_dummy_cnet_update:
            batch_generator = self.n_batches_this_epoch * [None]
        else:
            batch_generator = self.constraint_demonstration_buffer.iterate_epoch(self.extra_args.cnet_batch_size) # Iterate through all examples exactly once
        for i_batch, batch_demonstrations in enumerate(batch_generator):
            if self.do_dummy_cnet_update:
                test_losses = self.cnet_losses_and_scores_dummy
            else:
                batch_cnet_observations = np.stack([_e.state for _e in batch_demonstrations])
                batch_cnet_actions = np.stack([_e.action for _e in batch_demonstrations])
                batch_cnet_action_indicators = np.array([_e.action_indicator for _e in batch_demonstrations])
                test_losses_all = self.cnet_compute_losses_and_scores(batch_cnet_observations, batch_cnet_actions, batch_cnet_action_indicators)
                *test_losses_base, n_positive_satisfied, n_positive_violated, n_negative_satisfied, n_negative_violated = test_losses_all
                test_classification_accuracy = self.calc_classification_accuracy(n_positive_satisfied, n_positive_violated, n_negative_satisfied, n_negative_violated)
                test_losses = test_losses_base + test_classification_accuracy
            losses.append(test_losses)
            if i_batch > self.n_batches_this_epoch:
                break # allow early stop if max number of batches per epoch is set
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        mean_classification_accuracy_test = meanlosses[-3:]
        if len(self.extra_args.adaptive_constraint_activation) > 0:
            if 'average' in self.extra_args.adaptive_constraint_activation:
                activation_probability = mean_classification_accuracy_test[2]
            elif 'positive' in self.extra_args.adaptive_constraint_activation:
                activation_probability = mean_classification_accuracy_test[0]
            elif 'negative' in self.extra_args.adaptive_constraint_activation:
                activation_probability = mean_classification_accuracy_test[1]
            elif 'min' in self.extra_args.adaptive_constraint_activation:
                activation_probability = min(mean_classification_accuracy_test[:2])
            else:
                raise ValueError('Invalid argument adaptive_constraint_activation={0}'.format(self.extra_args.adaptive_constraint_activation))
        else:
            activation_probability = 1.
        if do_log:
            self.logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, self.cnet_loss_and_accuracy_names):
                self.logger.record_tabular('loss_'+name, lossval)
        return meanlosses, activation_probability

    def prepare_constraint_update(self, do_train_cnet, iters_so_far):
        '''
        Check the number of new demonstrations and if training can be skipped this iteration
        '''
        self.iters_so_far = iters_so_far
        self.cnet_cur_lrmult_epoch = 1.0

        # Update CNet
        n_positive_demonstrations_local = len(self.constraint_demonstration_buffer.positive_demonstrations)
        n_negative_demonstrations_local = len(self.constraint_demonstration_buffer.negative_demonstrations)
        n_demonstrations_buffer = n_positive_demonstrations_local + n_negative_demonstrations_local

        self.do_dummy_cnet_update = (n_positive_demonstrations_local == 0) or (n_negative_demonstrations_local == 0)

        # Since all environments can have different buffer sizes, we have to agree on a number of iterations
        n_positive_negative_demonstrations_local = np.array([n_positive_demonstrations_local, n_negative_demonstrations_local])
        n_positive_negative_demonstrations_global = np.array([0, 0])
        self.mpi_comm.Allreduce(n_positive_negative_demonstrations_local, n_positive_negative_demonstrations_global, op=MPI.SUM)
        self.n_positive_demonstrations_global, self.n_negative_demonstrations_global = n_positive_negative_demonstrations_global
        self.n_total_demonstrations_global = self.n_positive_demonstrations_global + self.n_negative_demonstrations_global

        self.logger.log('Optimizing CNet over {0} demonstrations: {1} positive, {2} negative'.format(n_demonstrations_buffer, self.n_positive_demonstrations_global, self.n_negative_demonstrations_global))

        n_max_demonstrations_local_buffer  = np.array([max(n_positive_demonstrations_local, n_negative_demonstrations_local)])
        n_max_demonstrations_global_buffer = np.array([0])
        self.mpi_comm.Allreduce(n_max_demonstrations_local_buffer, n_max_demonstrations_global_buffer, op=MPI.MAX)
        self.n_max_demonstrations_global = n_max_demonstrations_global_buffer[0]
        n_demonstrations_per_epoch = 2*self.n_max_demonstrations_global
        self.n_batches_this_epoch = int(np.ceil(n_demonstrations_per_epoch / self.extra_args.cnet_batch_size))
        n_batches_this_epoch_str = 'CNet batches per epoch: {0}'.format(self.n_batches_this_epoch)
        if (self.extra_args.cnet_training_batches > 0) and (self.n_batches_this_epoch > self.extra_args.cnet_training_batches):
            self.n_batches_this_epoch = self.extra_args.cnet_training_batches
            n_batches_this_epoch_str += ', capped at {0}'.format(self.extra_args.cnet_training_batches)
        self.logger.log(n_batches_this_epoch_str)

        if self.do_evaluate_cnet_losses_before:
            # Compute accuracy before
            cnet_test_losses_before, activation_probability_before = self.evaluate_cnet_losses(do_log=False)
            self.logger.log('Constraint accuracy pre-training: {0:.1f}'.format(activation_probability_before*100.))
        else:
            activation_probability_before = 1.
        if self.do_check_interrupt_cnet_training:
            if self.cnet_interruption_type == 'prior_accuracy':
                do_train_cnet = self.check_interruption(do_train_cnet, activation_probability_before)
            else:
                assert self.cnet_interruption_type == 'accuracy'
        return do_train_cnet, activation_probability_before


    def update_constraint_network(self, do_train_cnet):
        '''
        Update the constraint network and control learning rate decay / training early stop
        depending on training loss and accuracy
        '''
        if not do_train_cnet:
            activation_probability_after = None # unused
            self.logger.log('Skip training CNet this iteration')
            return do_train_cnet, activation_probability_after
        best_cnet_losses = None
        best_classification_accuracy = [0., 0., 0.]
        i_epoch_best = 0
        if self.extra_args.cnet_decay_epochs > 0:
            n_epochs_without_improvement = 0
        for i_epoch in range(self.extra_args.cnet_training_epochs):
            if (i_epoch % 10) == 0:
                epoch_str = 'Iter {0}, epoch {1}/{2}'.format(self.iters_so_far, i_epoch, self.extra_args.cnet_training_epochs)
                epoch_str += ' ({0}: positive {1:.1f} %, negative {2:.1f} %, avg {3:.1f} % out of {4} / {5} / {6}'.format(i_epoch_best, 100.*best_classification_accuracy[0], 100.*best_classification_accuracy[1], 100.*best_classification_accuracy[2], self.n_positive_demonstrations_global, self.n_negative_demonstrations_global, self.n_total_demonstrations_global)
                self.logger.log(epoch_str)
                self.logger.log(fmt_row(13, self.cnet_loss_and_accuracy_names))

            losses = [] # list of tuples, each of which gives the loss for a minibatch
            training_classification_accuracy = []
            #for batch in d.iterate_once(optim_batchsize):
            if self.do_dummy_cnet_update:
                batch_generator = self.n_batches_this_epoch * [None]
            else:
                batch_generator = self.constraint_demonstration_buffer.iterate_epoch_balanced(self.extra_args.cnet_batch_size, n_max=self.n_max_demonstrations_global)
            for i_batch, batch_demonstrations in enumerate(batch_generator):
                if self.do_dummy_cnet_update:
                    newlosses, g = self.cnet_newlosses_dummy, self.cnet_g_dummy
                    new_training_classification_accuracy = self.cnet_scores_dummy
                else:
                    batch_cnet_observations = np.stack([_e.state for _e in batch_demonstrations])
                    batch_cnet_actions = np.stack([_e.action for _e in batch_demonstrations])
                    batch_cnet_action_indicators = np.array([_e.action_indicator for _e in batch_demonstrations])
                    *newlosses, g = self.cnet_lossandgrad(batch_cnet_observations, batch_cnet_actions, batch_cnet_action_indicators)
                    training_classification_scores = self.cnet_compute_scores(batch_cnet_observations, batch_cnet_actions, batch_cnet_action_indicators)
                    new_training_classification_accuracy = self.calc_classification_accuracy(*training_classification_scores)
                self.cnet_adam.update(g, self.extra_args.cnet_learning_rate * self.cnet_cur_lrmult_epoch)
                losses.append(newlosses)
                training_classification_accuracy.append(new_training_classification_accuracy)
                if i_batch >= (self.n_batches_this_epoch - 1):
                    break
            mean_cnet_losses,_,_ = mpi_moments(losses, axis=0)
            mean_classification_accuracy,_,_ = mpi_moments(training_classification_accuracy, axis=0)
            mean_losses_and_accuracy = np.concatenate([mean_cnet_losses, mean_classification_accuracy])
            self.logger.log(fmt_row(13, mean_losses_and_accuracy))
            if self.test_early_stop_classification_accuracy(i_epoch, mean_classification_accuracy[0], mean_classification_accuracy[1]):
                break
            if (i_epoch == 0) or self.test_improvement_classification_accuracy(mean_cnet_losses, best_cnet_losses,
                                                                               mean_classification_accuracy, best_classification_accuracy):
                if i_epoch == 0:
                    mean_classification_accuracy_first_epoch = mean_classification_accuracy
                i_epoch_best = i_epoch
                best_cnet_losses = mean_cnet_losses
                best_classification_accuracy = mean_classification_accuracy
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1
            if (self.extra_args.cnet_decay_epochs > 0):
                if (n_epochs_without_improvement >= self.extra_args.cnet_decay_epochs):
                    if self.cnet_cur_lrmult_epoch > self.extra_args.cnet_decay_max:
                        self.cnet_cur_lrmult_epoch *= 0.5
                        i_epoch_best = i_epoch
                        best_cnet_losses = mean_cnet_losses
                        best_classification_accuracy = mean_classification_accuracy
                        n_epochs_without_improvement = 0
                        if self.cnet_cur_lrmult_epoch >= self.extra_args.cnet_decay_max:
                            self.logger.log('Halve CNet learning rate multiplier to {0}'.format(self.cnet_cur_lrmult_epoch))
                        else:
                            self.cnet_cur_lrmult_epoch = self.extra_args.cnet_decay_max
                            self.logger.log('Keep CNet learning rate multiplier to {0}'.format(self.cnet_cur_lrmult_epoch))
                    else:
                        self.logger.log('No improvement at max decay {0} for {1} epochs'.format(self.cnet_cur_lrmult_epoch, self.extra_args.cnet_decay_epochs))
                        break

        self.logger.log('Evaluating CNet losses...')
        cnet_test_losses_after, activation_probability_after = self.evaluate_cnet_losses(do_log=True)

        if self.do_check_interrupt_cnet_training:
            if self.cnet_interruption_type == 'accuracy':
                do_train_cnet = self.check_interruption(do_train_cnet, activation_probability_after)
            else:
                assert self.cnet_interruption_type == 'prior_accuracy'

        return do_train_cnet, activation_probability_after

    def synchronize_recovery_trajectories(self, env, seg, n_reference_trajectories_before_sampling):
        '''
        Send uncertain demonstrations from direct to recovery agents.
        '''
        uncertain_demonstration_trajectories = seg['uncertain_demonstration_trajectories']

        # Synchronize recovery trajectories
        if self.is_direct_policy:
            # Send new reference trajectories to recovery agents
            n_trajs_total = len(uncertain_demonstration_trajectories)
            n_trajs_per_process = [int(n_trajs_total / self.n_exchange_processes) for _ in range(self.n_exchange_processes)]
            n_trajs_per_process[0] += int(n_trajs_total % self.n_exchange_processes)
            _i_begin = 0
            exchange_trajs_all = []
            for _i_process in range(self.n_exchange_processes):
                _i_end = _i_begin + n_trajs_per_process[_i_process]
                exchange_trajs_process = uncertain_demonstration_trajectories[_i_begin:_i_end]
                exchange_trajs_all.append(exchange_trajs_process)
                _i_begin = _i_end
            exchange_trajs_all = mpi_select.Bcast_select(self.mpi_comm, self.mpi_rank, self.mpi_rank, self.cnet_exchange_ids[self.mpi_rank], exchange_trajs_all, tag=self.mpi_rank)
            print(' **** Direct {0}: {1} reference trajectories sent to recovery {2}'.format(self.mpi_rank, len(uncertain_demonstration_trajectories), self.cnet_exchange_ids[self.mpi_rank]))
        else:
            # Receive new reference trajectories from direct agents
            n_removed_reference_trajectories = n_reference_trajectories_before_sampling - len(env.unwrapped.reference_trajectories)
            new_reference_trajectories = []
            for _i_process, mpi_rank_direct in enumerate(self.cnet_exchange_ids[self.mpi_rank]):
                exchange_trajs_all = [[] for _ in range(self.n_exchange_processes)]
                exchange_trajs_all = mpi_select.Bcast_select(self.mpi_comm, self.mpi_rank, mpi_rank_direct, self.cnet_exchange_ids[mpi_rank_direct], exchange_trajs_all, tag=mpi_rank_direct)
                exchange_trajs_process = exchange_trajs_all[self.cnet_recovery_id_in_direct_exchange_ids[self.mpi_rank][mpi_rank_direct]]
                new_reference_trajectories += exchange_trajs_process

            # Remove empty reference trajectories after iterating through sampled episodes to avoid indexing issues
            for traj in new_reference_trajectories:
                env.unwrapped.add_reference_trajectory(traj)
            print(' **** Recovery {0}: {1} reference trajectories ({2} removed, {3} added)'.format(self.mpi_rank, len(env.unwrapped.reference_trajectories), n_removed_reference_trajectories, len(new_reference_trajectories)))


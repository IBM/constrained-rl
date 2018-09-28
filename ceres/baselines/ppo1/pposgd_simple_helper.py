'''
These are components of OpenAI's baselines.ppo1.pposgd_simple.learn
cut into individual functions for re-use in CERES
OpenAI Baselines is licensed under the MIT License, see LICENSE
'''

from baselines.ppo1.pposgd_simple import add_vtarg_and_adv
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from collections import deque
from mpi4py import MPI

def calc_end_training(max_timesteps, timesteps_so_far,
                      max_episodes, episodes_so_far,
                      max_iters, iters_so_far,
                      max_seconds, tstart):
    if max_timesteps and timesteps_so_far >= max_timesteps:
        return True
    elif max_episodes and episodes_so_far >= max_episodes:
        return True
    elif max_iters and iters_so_far >= max_iters:
        return True
    elif max_seconds and time.time() - tstart >= max_seconds:
        return True
    else:
        return False

def build_policy_training_vars(pi, oldpi, clip_param, entcoeff, adam_epsilon):
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    return loss_names, var_list, lossandgrad, adam, assign_old_eq_new, compute_losses

def adjust_policy_learning_rate(schedule,
                                max_timesteps, timesteps_so_far,
                                max_episodes, episodes_so_far,
                                max_iters, iters_so_far):
    if schedule == 'constant':
        cur_lrmult = 1.0
    elif schedule == 'linear':
        if max_timesteps > 0:
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        elif max_episodes > 0:
            cur_lrmult =  max(1.0 - float(episodes_so_far) / max_episodes, 0)
        elif max_iters > 0:
            cur_lrmult =  max(1.0 - float(iters_so_far) / max_iters, 0)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return cur_lrmult

def update_policy(pi, seg, gamma, lam,
                  logger, optim_epochs, optim_batchsize, optim_stepsize, cur_lrmult,
                  loss_names, lossandgrad, adam, assign_old_eq_new, compute_losses,
                  mpi_moments_fn):
        
    add_vtarg_and_adv(seg, gamma, lam)

    # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
    ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
    vpredbefore = seg["vpred"] # predicted value function before udpate
    atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
    d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
    optim_batchsize = optim_batchsize or ob.shape[0]

    if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

    assign_old_eq_new() # set old parameter values to new parameter values
    logger.log("Optimizing...")
    logger.log(fmt_row(13, loss_names))
    # Here we do a bunch of optimization epochs over the data
    for _ in range(optim_epochs):
        losses = [] # list of tuples, each of which gives the loss for a minibatch
        for batch in d.iterate_once(optim_batchsize):
            *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            adam.update(g, optim_stepsize * cur_lrmult)
            losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(losses, axis=0)))

    logger.log("Evaluating losses...")
    losses = []
    for batch in d.iterate_once(optim_batchsize):
        newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
        losses.append(newlosses)
    meanlosses,_,_ = mpi_moments_fn(losses)
    logger.log(fmt_row(13, meanlosses))
    for (lossval, name) in zipsame(meanlosses, loss_names):
        logger.record_tabular("loss_"+name, lossval)
    return vpredbefore, tdlamret, optim_batchsize

def log_iter_info(lenbuffer, rewbuffer, tstart,
                  vpredbefore, tdlamret, seg,
                  episodes_so_far, timesteps_so_far,
                  do_dump_tabular, allgather_fn):
    logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
    lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
    listoflrpairs = allgather_fn(lrlocal) # list of tuples
    lens, rews = map(flatten_lists, zip(*listoflrpairs))
    lenbuffer.extend(lens)
    rewbuffer.extend(rews)
    logger.record_tabular("EpLenMean", np.mean(lenbuffer))
    logger.record_tabular("EpRewMean", np.mean(rewbuffer))
    logger.record_tabular("EpThisIter", len(lens))
    episodes_so_far += len(lens)
    timesteps_so_far += sum(lens)
    logger.record_tabular("EpisodesSoFar", episodes_so_far)
    logger.record_tabular("TimestepsSoFar", timesteps_so_far)
    logger.record_tabular("TimeElapsed", time.time() - tstart)
    if do_dump_tabular:
        logger.dump_tabular()
    return episodes_so_far, timesteps_so_far

def build_counters():
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    return iters_so_far, episodes_so_far, timesteps_so_far, tstart, lenbuffer, rewbuffer

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

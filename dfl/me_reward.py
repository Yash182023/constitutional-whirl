#!/usr/bin/env python

from armmanworld import ArmmanWorld
import maxentedited as M
from dfl.whittle import newWhittleIndex
import pandas as pd
import time
import argparse
from irl_maxent import plot as P
from dfl.policy import getProbs,get_trajectory_probability
from irl_maxent import trajectory as T
from irl_maxent import solver as S
import matplotlib.pyplot as plt
from dfl.synthetic import generateDataset
from irl_maxent import optimizer as O
import tensorflow as tf
import itertools

import numpy as np
import matplotlib.pyplot as plt


def setup_mdp(size,n_states=2,H=10,k=1,seed=0,transitionseed=0):
    """
    Set-up our MDP/GridWorld
    """
    # create our world
    world = ArmmanWorld(size,k,H,transitionseed)

    # set up the reward function

    # reward = tf.constant(np.random.rand(size,2),dtype=float)
    np.random.seed(seed)
    w = np.array([[np.random.uniform(low=0, high=0.5)],[np.random.uniform(low=0.5, high=1)],[np.random.uniform(low=0, high=0.5)],[np.random.uniform(low=0.5, high=1)]])
    reward_numpy = np.array([[w[0][0],w[1][0]],[w[2][0],w[3][0]]])
    reward = tf.constant(reward_numpy,dtype=float)
    # print(reward_numpy[0,0]+reward_numpy[1,0], reward_numpy[0,0]+reward_numpy[1,1], reward_numpy[0,1]+reward_numpy[1,0], reward_numpy[0,1]+reward_numpy[1,1])
    # set up terminal states
    terminal = [24]

    return world, reward, terminal

def get_features(world):
    features = np.zeros((world.n_states, 2* world.size))
    for i in range(world.n_states):
        raw_bin = bin(i)[2:]
        binary = (world.size - len(raw_bin)) * '0' + raw_bin
        for j in range(len(binary)):
            features[i, 2 * j + int(binary[j])] = 1
    return features

def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = get_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.Sga(lr=O.linear_decay(lr0=0.01))

    # actually do some inverse reinforcement learning
    reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = get_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward

def create_dataset(world,reward,num_trajectories,seed=0):
    discount = 0.99
    return generateDataset(world.size, 2, 1, num_trajectories, world.H, world.k, discount, run_whittle=True,
                              env='general', H=world.H, seed=seed,
                              Rewards=reward)

def create_trajectory(world,dataset):
    trajectories = []
    actions = list(itertools.combinations(np.arange(world.size), world.k))
    actions_full = np.zeros((len(actions),world.size))
    for i in range(len(actions)):
        for j in range(len(actions[i])):
            actions_full[i,actions[i][j]] = 1
    for _, probability, raw_R_data, traj, _, state_record, action_record, reward_record in dataset:
        for i in range(len(state_record)):
            trajectory = []
            for j in range(world.H-1):
                state = (state_record[i][0][j]).dot(1 << np.arange(state_record[i][0][j].shape[-1] - 1, -1, -1))
                next_state = (state_record[i][0][j+1]).dot(1 << np.arange(state_record[i][0][j+1].shape[-1] - 1, -1, -1))
                action = np.where(actions_full==action_record[i,0,j])[0][0]
                trajectory.append(tuple((int(state),int(action),int(next_state))))
            trajectories.append(T.Trajectory(trajectory))
    return trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARMMAN decision-focused learning')
    parser.add_argument('--seed', default=0, type=int, help='rewards seed')
    parser.add_argument('--transitionseed', default=0, type=int, help='transitionseed')
    parser.add_argument('--trajseed', default=0, type=int, help='trajseed')
    parser.add_argument('--ntrajs', default=2, type=int, help='trajseed')
    parser.add_argument('--n', default=2, type=int, help='trajseed')
    parser.add_argument('--k', default=1, type=int, help='trajseed')
    args = parser.parse_args()
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    num_states = 2
    size = args.n
    k=args.k
    H=10
    epsilon = 0.1
    world, reward, terminal = setup_mdp(size, n_states=num_states, H=H, k=k, seed=args.seed, transitionseed=args.transitionseed)
    dataset = create_dataset(world, reward, args.ntrajs, seed=args.trajseed)
    trajectories = create_trajectory(world, dataset)
    t0 = time.time()
    reward_maxent = maxent(world, terminal, trajectories)

    _, _, _, _, _, state_record, action_record, _ = dataset[0]
    w_optimal = tf.reshape(newWhittleIndex(tf.constant(world.small_p_transition,dtype=float), reward), (size, num_states))
    w_learn = tf.reshape(newWhittleIndex(tf.constant(world.small_p_transition, dtype=float), tf.constant(np.reshape(reward_maxent,(size,num_states)),dtype=float)),(size,num_states))
    b_probs, b_w_selected = getProbs(state_record[:, 0, :, :].reshape(-1, size), policy=3, ts=None, w=w_optimal, k=k,
                                             epsilon=epsilon)
    beh_probs_raw = tf.reshape(
        b_probs, (args.ntrajs, H, size))

    t_probs, t_w_selected = getProbs(state_record[:, 0, :, :].reshape(-1, size), policy=3, ts=None, w=w_learn, k=k,
                                        epsilon=epsilon)
    target_probs_raw = tf.reshape(t_probs, (args.ntrajs, H, size))
    policy_difference = np.sum(tf.norm(target_probs_raw - beh_probs_raw, ord=1).numpy()) / (size * num_states * args.ntrajs)
    print('WHITTLE DIFFERENCE: {w}'.format(w=policy_difference))
    t1 = time.time()
    print("TIME: "+ str(t1-t0))
    pd.DataFrame(np.array([t1-t0])).to_excel('medata/me_ntraj={nt}_seed={s}_transitionseed={ts}_trajseed={tr}_time={t}.xlsx'.format(nt = args.ntrajs, s=args.seed,ts=args.transitionseed, tr=args.trajseed, t=time.time()))

import tensorflow as tf
import numpy as np
import random
import itertools
import time
import sys
sys.path.insert(0, '../')

from model import SyntheticANN
from trajectory import getSimulatedTrajectories
from dflutils import generateRandomTMatrix
from whittle import whittleIndex, newWhittleIndex
from environments import POMDP2MDP
from ope import opeSimulator
from content_reward_simulate import create_full_transition_matrix,create_transitions_biased_full, create_content_transitions,create_transitions_random,create_transitions, create_full_transitions

def getRData(n_benefs, n_states):
    R = np.arange(n_states)  # sorted(np.random.uniform(size=n_states))
    return (R - np.min(R)) / np.ptp(R)  # * 2 - 1 # normalize rewards to between [-1,1]
    # return np.repeat(R.reshape(1, -1), n_benefs,
    #                        axis=0)  # using the same rewards across all arms (for simplicity)


def dummyProbabilities(n_benefs, n_states):
    P = np.zeros((n_benefs, n_states, 2, n_states))
    P[0, 0, 0, 0] = 1
    P[0, 0, 0, 1] = 0

    P[1, 0, 0, 0] = 1
    P[1, 0, 0, 1] = 0

    P[0, 0, 1, 0] = 0
    P[0, 0, 1, 1] = 1

    P[1, 0, 1, 0] = 0
    P[1, 0, 1, 1] = 1

    P[0, 1, 0, 0] = 1
    P[0, 1, 0, 1] = 0

    P[1, 1, 0, 0] = 1
    P[1, 1, 0, 1] = 0

    P[0, 1, 1, 0] = 0
    P[0, 1, 1, 1] = 1

    P[1, 1, 1, 0] = 0
    P[1, 1, 1, 1] = 1
    return P

def genProbabilities(n_benefs,seed=0):
    np.random.seed(seed)
    P = create_transitions_random(n_benefs)
    return create_full_transition_matrix(P)


def genProbabilitiesBigMDP(n_benefs, n_big_states,n_actions,k,seed=0, P = None):
    if P is None:
        full_p = genProbabilities(n_benefs,seed)
    else:
        full_p = P
    big_mdp_transition = np.ones((n_big_states,n_big_states,n_actions))
    actions = np.array(list(itertools.combinations(np.arange(n_benefs),k)))
    for i in range(n_big_states):
        for j in range(n_big_states):
            for k in range(len(actions)):
                curr_state = bin(i)[2:]
                curr_state = (n_benefs-len(curr_state)) * '0' + curr_state
                next_state = bin(j)[2:]
                next_state = (n_benefs - len(next_state)) * '0' + next_state
                for arm in range(len(curr_state)):
                    if arm in actions[k]:
                        big_mdp_transition[i, j, k] *= full_p[arm, int(curr_state[arm]), 1, int(next_state[arm])]
                    else:
                        big_mdp_transition[i, j, k] *= full_p[arm, int(curr_state[arm]), 0, int(next_state[arm])]
    return big_mdp_transition

def genProbabilitiesFourState(n_benefs, n_states,seed=1):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    P = create_transitions_random(n_benefs)
    return create_full_transitions(P)


def generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma, env='general', H=10, run_whittle=False, seed=None, Rewards=None, w_computed=None,P=None):
    # n_benefs: number of beneficiaries in a cohort
    # n_states: number of states
    # n_instances: number of cohorts in the whole dataset
    # n_trials: number of trajectories we obtained from each cohort. In the real-world dataset, n_trials=1
    # L: number of time steps
    # K: budget
    # gamma: discount factor

    # Set up random seed
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # Generate T_data => ANN => features
    # Return a tensorflow dataset that includes (features, traj, T_data) as an instance.
    dataset = []

    # Randomly initializing a neural network
    model = SyntheticANN()

    # Prepare for parallelization
    state_matrix = np.zeros((n_instances, L, n_benefs))
    action_matrix = np.zeros((n_instances, L, n_benefs))

    # Generating synthetic data
    for i in range(n_instances):
        # Generate rewards from uniform distribution
        # if Rewards == None:
        #     R = np.arange(n_states) # sorted(np.random.uniform(size=n_states))
        #     R = (R - np.min(R)) / np.ptp(R) # * 2 - 1 # normalize rewards to between [-1,1]
        #     raw_R_data = np.repeat(R.reshape(1, -1), n_benefs,
        #                            axis=0)  # using the same rewards across all arms (for simplicity)
        #     raw_R_data = tf.constant(raw_R_data, dtype=tf.float32)
        # else:
        #     raw_R_data = Rewards
        raw_R_data = Rewards

        # Generate transition probabilities
        if P is not None:
            raw_T_data = P
        else:
            raw_T_data = genProbabilities(n_benefs,seed=seed)
        
        # Generate features using the transition probabilities
        feature = model(tf.constant(raw_T_data.reshape(-1,2*n_states*n_states), dtype=tf.float32))
        # print(tf.norm(feature, axis=1))
        noise_level = 0.0
        feature = feature + tf.random.normal(shape=(n_benefs, 16,)) * noise_level

        # Generate environment parameters
        if env=='general':
            T_data, R_data = raw_T_data, raw_R_data
        elif env=='POMDP':
            T_data, R_data = POMDP2MDP(tf.constant(raw_T_data, dtype=tf.float32), raw_R_data, H)
            T_data, R_data = T_data.numpy(), R_data.numpy()
            # print('raw R', raw_R_data)
            # print('new R', R_data)

        # Different choices of Whittle indices
        if w_computed != None:
            w = w_computed
        elif run_whittle:
            # w = whittleIndex(tf.constant(T_data, dtype=tf.float32)).numpy() # Old Whittle index computation. This only works for n_states=2.
            w = newWhittleIndex(tf.constant(T_data, dtype=tf.float32),R_data).numpy() # New Whittle index computation. It should work for multiple states.
            w = np.reshape(w,(n_benefs,T_data.shape[1]))
        else:
            w = np.zeros((n_benefs, T_data.shape[1])) # All zeros. This is to disable Whittle index policy to speed up simulation.
        
        assert w.shape == (n_benefs, T_data.shape[1])

        # start_time = time.time()
        traj, simulated_rewards, state_record, action_record, reward_record = getSimulatedTrajectories(
                                                                n_benefs=n_benefs, T=L, K=K, n_trials=n_trials, gamma=gamma,
                                                                T_data=T_data, R_data=R_data.numpy(),
                                                                w=w, replace=False, policies=[3], fast=True
                                                                )
        # print('slow version', time.time() - start_time)

        # # The fast version is only implemented for policy_id = 3
        # start_time = time.time()
        # traj, simulated_rewards, state_record, action_record, reward_record = getSimulatedTrajectories(
        #                                                         n_benefs=n_benefs, T=L, K=K, n_trials=n_trials, gamma=gamma,
        #                                                         seed=sim_seed, T_data=T_data, R_data=R_data,
        #                                                         w=w, replace=False, policies=[3], fast=True
        #                                                         )
        # print('fast version', time.time() - start_time)


        # Initialize simulation-based ope
        # This part takes the longest preprocessing time.
        # start_time = time.time()
        # OPE_sim_n_trials = 100
        # ope_simulator = opeSimulator(traj, n_benefs, L, n_states, OPE_sim_n_trials, gamma, beh_policy_name='random', T_data=T_data, R_data=R_data, env=env, H=H)
        # print('Initializing simulator time', time.time() - start_time)

        # print('real T data:', T_data[0])
        # print('empirical T data:', ope_simulator.emp_T_data[0])

        instance = (feature, raw_T_data, raw_R_data, traj, simulated_rewards, state_record, action_record, reward_record)
        # print('average simulated rewards (random, rr, whittle, soft-whittle):', np.mean(simulated_rewards, axis=0))
        dataset.append(instance)

    return dataset

def loadSyntheticData(n_benefs=100, n_states=2, n_trials=10, L=10, K=10, gamma=0.99):
    T_data = tf.constant(generateRandomTMatrix(n_benefs, n_states=n_states), dtype=tf.float32)
    w = whittleIndex(T_data)
    cluster_ids = np.arange(len(T_data)) # one cluster per person

    return T_data.numpy(), w.numpy(), None, cluster_ids

if __name__ == '__main__':
    # Testing data generation
    n_benefs = 50
    n_instances = 20
    n_trials = 100
    L = 10
    K = 3
    n_states = 3
    gamma = 0.99
    env = 'POMDP'
    H = 10
    print(genProbabilitiesBigMDP(2,2,2,1))
    # T_data = generateRandomTMatrix(n_benefs, n_states=n_states)
    # dataset = generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma=gamma, env=env, H=H, run_whittle=False)

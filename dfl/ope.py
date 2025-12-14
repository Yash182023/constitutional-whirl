import tensorflow as tf
# from fast_soft_sort.tf_ops import soft_rank, soft_sort
import numpy as np
import tqdm

# from dfl.policy import getActionProb, getActionProbNaive
# from dfl.config import dim_dict

from policy import getProbs,get_trajectory_probability
from config import policy_map
from trajectory import getSimulatedTrajectories
from trajectory import getEmpTransitionMatrix



# def opeIS(traj, w, n_benefs, T, K, n_trials, gamma, target_policy_name, beh_policy_name):
#     compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
#     gamma_series = np.array([gamma**t for t in range(T-1)]) # Kai edited: it was **t-1** instead of **t**.
# 
#     beh_probs    = np.zeros((n_trials, T, n_benefs))
#     target_probs = np.zeros((n_trials, T, n_benefs))
# 
#     v = []
#     # w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing
#     for benef in tqdm.tqdm(range(n_benefs), desc='OPE'):
#         v_i = 0
#         for trial in range(n_trials):
#             imp_weight = 1
#             v_i_tau = 0
#             for ts in range(T-1):
#                 a_i_t = traj[trial, # trial index
#                                 0, # policy index
#                                 ts, # time index
#                                 dim_dict['action'], # tuple dimension
#                                 benef # benef index
#                                 ].astype(int)
# 
#                 s_t = traj[trial, # trial index
#                                 0, # policy index
#                                 ts, # time index
#                                 dim_dict['state'], # tuple dimension
#                                 : # benef index
#                                 ].astype(int)
#                 pi_tar = getActionProb(s_t, a_i_t,
#                                            policy=compare['target'],
#                                            benef=benef, ts=ts,
#                                            w=w, k=K, N=n_benefs)
#                 pi_beh = getActionProb(s_t, a_i_t,
#                                            policy=compare['beh'],
#                                            benef=benef, ts=ts,
#                                            w=w, k=K, N=n_benefs)
#                 imp_weight*= pi_tar/pi_beh
#                 # if imp_weight>1:
#                 #     print('weight: ', imp_weight)
#                 v_i_t_tau = gamma_series[ts] * traj[trial, # trial index
#                                                 0, # policy index
#                                                 ts, # time index
#                                                 dim_dict['reward'], # tuple dimension
#                                                 benef # benef index
#                                                 ] * imp_weight
#                 v_i_tau += v_i_t_tau
# 
#                 beh_probs[trial, ts, benef]    = pi_beh
#                 target_probs[trial, ts, benef] = pi_tar
# 
#             v_i += v_i_tau
#         v.append(v_i/n_trials)
#     ope = np.sum(v)
#     # print(f'OPE: {ope}')
#     return ope

def eval_policy_fast(state_record, action_record, w, w_opt, H, K, gamma, target_policy_name, beh_policy_name, transition_probabilities, epsilon=0.1, reward_estimate=None,rewards_true = None):
    ntr, _, L, N = state_record.shape
    _, n_states = tf.shape(w)
    action_record_beh = action_record[:, 0, :, :]

    # Batch topk to get probabilities
    b_probs, b_w_selected = getProbs(state_record[:, 0, :, :].reshape(-1, N), policy=3, ts=None, w=w_opt, k=K,
                 epsilon=epsilon)
    beh_probs_raw = tf.reshape(
        b_probs, (ntr, L, N))

    t_probs, t_w_selected = getProbs(state_record[:, 0, :, :].reshape(-1, N), policy=3, ts=None, w=w, k=K, epsilon=epsilon)
    target_probs_raw = tf.reshape(t_probs,(ntr, L, N))
    probs_norm = tf.norm(target_probs_raw-beh_probs_raw)/(N*L*ntr)
    probs_trajectory_given_policy = get_trajectory_probability(state_record[:, 0, :, :].reshape(-1, N), action_record_beh, w,K,epsilon,ntr, transition_probabilities)

    return -1*probs_trajectory_given_policy,probs_norm

def eval_policy_simple(state_record, action_record, w, w_opt, H, K, gamma, target_policy_name, beh_policy_name, transition_probabilities, epsilon=0.1, reward_estimate=None,rewards_true = None):
    ntr, _, L, N = state_record.shape
    _, n_states = tf.shape(w)
    # rewards_true = np.zeros((1,2))
    # rewards_true[:,1] += 1
    # rewards_true = tf.Variable(rewards_true,dtype=float)
    # return tf.norm(rewards - reward_estimate, ord='euclidean'),None

    # compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    # gamma_series = np.array([gamma**t for t in range(H-1)])
    # v = []
    # w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing

    # state_record_beh = np.concatenate([np.tile(np.arange(N), (ntr, L, 1)).reshape(ntr, L, N, 1), state_record[:,compare['beh'],:,:].reshape(ntr, L, N, 1)], axis=-1).astype(int)
    action_record_beh = action_record[:, 0, :, :]

    # Get the corresponding Whittle indices
    # whittle_indices = tf.gather_nd(w, state_record_beh)

    # Batch topk to get probabilities
    # b_probs, b_w_selected = getProbs(state_record[:, 0, :, :].reshape(-1, N), policy=3, ts=None, w=w_opt, k=K,
    #              epsilon=epsilon)
    # beh_probs_raw = tf.reshape(
    #     b_probs, (ntr, L, N))

    t_probs, t_w_selected = getProbs(state_record[:, 0, :, :].reshape(-1, N), policy=3, ts=None, w=w, k=K, epsilon=epsilon)
    # target_probs_raw = tf.reshape(t_probs,(ntr, L, N))
    # top_k_distance = np.abs(np.sum(b_w_selected - t_w_selected))

    # beh_probs = beh_probs_raw * action_record_beh + (1 - beh_probs_raw) * (1 - action_record_beh)
    # target_probs = target_probs_raw * action_record_beh + (1 - target_probs_raw) * (1 - action_record_beh)
    # probs_norm = tf.norm(target_probs_raw-beh_probs_raw)/(N*L*ntr)
    probs_trajectory_given_policy = get_trajectory_probability(state_record[:, 0, :, :].reshape(-1, N), action_record_beh, w,K,epsilon,ntr, transition_probabilities)
    # w_norm = tf.norm(w - w_opt, ord='euclidean')
    # w_flattened_opt = tf.argsort(tf.reshape(w_opt,n_states*N))
    # w_flattened_train =tf.argsort(tf.reshape(w,n_states*N))
    # w_order_norm = tf.math.count_nonzero(w_flattened_train != w_flattened_opt)
    # if np.min(reward_estimate) < 0:
    #     reward_estimate = reward_estimate - np.min(reward_estimate)
    # reward_norm = tf.norm(rewards_true - reward_estimate, ord='euclidean')
    return -1*probs_trajectory_given_policy,None, None, None, None #top_k_distance/np.sum(b_w_selected)

def eval_policy(state_record, action_record, w, w_opt, H, K, gamma, target_policy_name, beh_policy_name, transition_probabilities, epsilon=0.1, reward_estimate=None,rewards_true = None,traj_k=0):
    ntr, _, L, N = state_record.shape
    _, n_states = tf.shape(w)
    if traj_k == 0:
        traj_k = K
    # rewards_true = np.zeros((1,2))
    # rewards_true[:,1] += 1
    # rewards_true = tf.Variable(rewards_true,dtype=float)
    # return tf.norm(rewards - reward_estimate, ord='euclidean'),None

    # compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    # gamma_series = np.array([gamma**t for t in range(H-1)])
    # v = []
    # w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing

    # state_record_beh = np.concatenate([np.tile(np.arange(N), (ntr, L, 1)).reshape(ntr, L, N, 1), state_record[:,compare['beh'],:,:].reshape(ntr, L, N, 1)], axis=-1).astype(int)
    action_record_beh = action_record[:, 0, :, :]

    # Get the corresponding Whittle indices
    # whittle_indices = tf.gather_nd(w, state_record_beh)

    # Batch topk to get probabilities
    # b_probs, _ = getProbs(state_record[:, 0, :, :].reshape(-1, N), policy=3, ts=None, w=w_opt, k=traj_k,
    #              epsilon=epsilon)
    # beh_probs_raw = tf.reshape(
    #     b_probs, (ntr, L, N))

    # t_probs, _ = getProbs(state_record[:, 0, :, :].reshape(-1, N), policy=3, ts=None, w=w, k=traj_k, epsilon=epsilon)
    # target_probs_raw = tf.reshape(t_probs,(ntr, L, N))
    # top_k_distance = np.abs(np.sum(b_w_selected - t_w_selected))

    # beh_probs = beh_probs_raw * action_record_beh + (1 - beh_probs_raw) * (1 - action_record_beh)
    # target_probs = target_probs_raw * action_record_beh + (1 - target_probs_raw) * (1 - action_record_beh)
    # probs_norm = tf.norm(target_probs_raw-beh_probs_raw)
    probs_trajectory_given_policy = get_trajectory_probability(state_record[:, 0, :, :].reshape(-1, N), action_record_beh, w,K,epsilon,ntr, transition_probabilities)
    # w_norm = tf.norm(w - w_opt, ord='euclidean')
    # w_flattened_opt = tf.argsort(tf.reshape(w_opt,n_states*N))
    # w_flattened_train =tf.argsort(tf.reshape(w,n_states*N))
    # w_order_norm = tf.math.count_nonzero(w_flattened_train != w_flattened_opt)
    # if np.min(reward_estimate) < 0:
    #     reward_estimate = reward_estimate - np.min(reward_estimate)
    # reward_norm = tf.norm(rewards_true - reward_estimate, ord='euclidean')
#     print(probs_trajectory_given_policy)
    return -1*probs_trajectory_given_policy,None, None, None, None #top_k_distance/np.sum(b_w_selected)

#This is the parallelized implementation of the same OPE. Ideally these two should match but the parallelized version is faster.
def opeIS_parallel(state_record, action_record, reward_record, w, w_opt, T, K, n_trials, gamma, target_policy_name, beh_policy_name, single_trajectory=False, epsilon=0.1, reward_estimate=None):
    ntr, n_states, L, N = state_record.shape
    rewards_true = np.zeros((1,2))
    rewards_true[:,1] += 1
    rewards_true = tf.Variable(rewards_true,dtype=float)
    # return tf.norm(rewards - reward_estimate, ord='euclidean'),None

    compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    gamma_series = np.array([gamma**t for t in range(T-1)])

    ntr, _, L, N = state_record.shape

    v = []
    # w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing

    # state_record_beh = np.concatenate([np.tile(np.arange(N), (ntr, L, 1)).reshape(ntr, L, N, 1), state_record[:,compare['beh'],:,:].reshape(ntr, L, N, 1)], axis=-1).astype(int)
    action_record_beh = action_record[:,0,:,:]

    # Get the corresponding Whittle indices
    # whittle_indices = tf.gather_nd(w, state_record_beh)

    # Batch topk to get probabilities
    beh_probs_raw    = tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, N), policy=compare['beh'],    ts=None, w=w_opt, k=K, epsilon=epsilon), (ntr, L, N))
    target_probs_raw = tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, N), policy=compare['target'], ts=None, w=w, k=K, epsilon=epsilon), (ntr, L, N))


    # Use action to select the corresponding probabilities
    beh_probs    = beh_probs_raw * action_record_beh + (1 - beh_probs_raw) * (1 - action_record_beh)
    target_probs = target_probs_raw * action_record_beh + (1 - target_probs_raw) * (1 - action_record_beh)

    # Importance sampling weights
    IS_weights = target_probs / beh_probs # [ntr, L, N]

    # OPE
    total_probs = np.ones((ntr, N))
    ope = 0
    ess = 0
    for t in range(T-1):
        if not single_trajectory:
            rewards = reward_record[:, 0, t, :]
            total_probs = IS_weights[:,t,:] * total_probs # shape: [n_trials, n_benefs]
            IS_sum = tf.reduce_sum(total_probs, axis=0) + 0.0001 # shape: [1, n_benefs], add a small constant to avoid nan issue
            IS_square_sum = tf.reduce_sum(total_probs**2, axis=0) #/ n_benefs # shape: [1, n_benefs]
            ope += rewards * total_probs * gamma_series[t] / IS_sum
            ess += IS_sum ** 2 / IS_square_sum # shape: [1, n_benefs]
        else:
            rewards = reward_record[:, 0, t, :]
            IS_sum = tf.reduce_mean(IS_weights[:,:T-1,:], axis=1) # Average IS of one trajectory with shape: [n_trials=1, n_benefs]
            IS_square_sum = tf.reduce_mean(IS_weights[:,:T-1,:]**2, axis=1) # shape: [n_trails=1, n_benefs]
            ope += rewards * IS_weights[:,t,:] / IS_sum * gamma_series[t]
            ess += IS_sum ** 2 / IS_square_sum

    ope = tf.reduce_sum(ope)
    w_norm = tf.norm(w-w_opt, ord='euclidean')
    if np.min(reward_estimate)<0:
        reward_estimate = reward_estimate - np.min (reward_estimate)
    return w_norm, tf.norm(rewards_true - reward_estimate, ord='euclidean')


def eval_reward(state_record, action_record, reward_record, w, w_opt, T, K, n_trials, gamma, target_policy_name, beh_policy_name, single_trajectory=False, epsilon=0.1):
    compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    gamma_series = np.array([gamma**t for t in range(T-1)])

    ntr, _, L, N = state_record.shape

    v = []
    # w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing

    # state_record_beh = np.concatenate([np.tile(np.arange(N), (ntr, L, 1)).reshape(ntr, L, N, 1), state_record[:,compare['beh'],:,:].reshape(ntr, L, N, 1)], axis=-1).astype(int)
    action_record_beh = action_record[:,0,:,:]

    # Get the corresponding Whittle indices
    # whittle_indices = tf.gather_nd(w, state_record_beh)

    # Batch topk to get probabilities
    opt_probs_raw    = tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, N), policy=3,    ts=None, w=w_opt, k=K, epsilon=epsilon), (ntr, L, N))
    train_probs_raw = tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, N), policy=3, ts=None, w=w, k=K, epsilon=epsilon), (ntr, L, N))

    # Use action to select the corresponding probabilities
    opt_probs    = opt_probs_raw * action_record_beh + (1 - opt_probs_raw) * (1 - action_record_beh)
    training_probs = train_probs_raw * action_record_beh + (1 - train_probs_raw) * (1 - action_record_beh)

    # Importance sampling weights
    # IS_weights = target_probs / beh_probs # [ntr, L, N]

    # OPE
    # total_probs = np.ones((ntr, N))
    # ope = 0
    # ess = 0
    # for t in range(T-1):
    #     if not single_trajectory:
    #         rewards = reward_record[:, 0, t, :]
    #         total_probs = IS_weights[:,t,:] * total_probs # shape: [n_trials, n_benefs]
    #         IS_sum = tf.reduce_sum(total_probs, axis=0) + 0.0001 # shape: [1, n_benefs], add a small constant to avoid nan issue
    #         IS_square_sum = tf.reduce_sum(total_probs**2, axis=0) #/ n_benefs # shape: [1, n_benefs]
    #         ope += rewards * total_probs * gamma_series[t] / IS_sum
    #         ess += IS_sum ** 2 / IS_square_sum # shape: [1, n_benefs]
    #     else:
    #         rewards = reward_record[:, 0, t, :]
    #         IS_sum = tf.reduce_mean(IS_weights[:,:T-1,:], axis=1) # Average IS of one trajectory with shape: [n_trials=1, n_benefs]
    #         IS_square_sum = tf.reduce_mean(IS_weights[:,:T-1,:]**2, axis=1) # shape: [n_trails=1, n_benefs]
    #         ope += rewards * IS_weights[:,t,:] / IS_sum * gamma_series[t]
    #         ess += IS_sum ** 2 / IS_square_sum
    #
    # ope = tf.reduce_sum(ope)
    # return ope, ess

# def opeISNaive(traj, w, n_benefs, T, K, n_trials, gamma, target_policy_name, beh_policy_name):
#     compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
#     gamma_series = np.array([gamma**(t-1) for t in range(T-1)])
# 
#     v = []
#     for trial in range(n_trials):
#         imp_weight = 1
#         v_tau = 0
#         for ts in range(T-1):
#             a_t = traj[trial, # trial index
#                             0, # policy index
#                             ts, # time index
#                             dim_dict['action'], # tuple dimension
#                             : # benef index
#                             ].astype(int)
#             # a_t_encoded = encode_vector(a_t, N_ACTIONS)
# 
#             s_t = traj[trial, # trial index
#                             0, # policy index
#                             ts, # time index
#                             dim_dict['state'], # tuple dimension
#                             : # benef index
#                             ].astype(int)
#             # s_t_encoded = encode_vector(s_t, N_STATES)
# 
#             pi_tar = getActionProbNaive(s_t, a_t, policy=compare['target'],
#                                         w=w, k=K, N=n_benefs)
#             pi_beh = getActionProbNaive(s_t, a_t, policy=compare['beh'],
#                                         w=w, k=K, N=n_benefs)
# 
#             imp_weight*= pi_tar/pi_beh
#             # if imp_weight>1:
#             #     print('weight: ', imp_weight)
#             v_t_tau = gamma_series[ts] * traj[trial, # trial index
#                                             0, # policy index
#                                             ts, # time index
#                                             dim_dict['reward'], # tuple dimension
#                                             : # benef index
#                                             ].sum() * imp_weight
#             v_tau += v_t_tau
#   
#     v.append(v_tau)
#     ope = np.mean(v)
#     print(f'OPE Naive: {ope}')
#     return ope

# Simulation-based OPE (differentiable and parallelizable)
class opeSimulator(object):
    def __init__(self, beh_traj, n_benefs, T, m, OPE_sim_n_trials, gamma, beh_policy_name, T_data, R_data, env='general', H=None, use_informed_prior=False):
        self.n_benefs = n_benefs
        self.T = T
        self.m = m
        self.H = H
        self.OPE_sim_n_trials = OPE_sim_n_trials
        self.gamma = gamma

        policy_id = policy_map[beh_policy_name]
        # self.emp_T_data, self.emp_R_data = getEmpTransitionMatrix(traj=beh_traj, policy_id=policy_id, n_benefs=n_benefs, m=m, env=env, H=H, use_informed_prior=use_informed_prior)
        if T_data is not None:
            self.emp_T_data = T_data # Directly using the real T_data
            self.emp_R_data = R_data # Reward list is explicitly given in the MDP version
        else:
            self.emp_T_data, self.emp_R_data = getEmpTransitionMatrix(traj=beh_traj, policy_id=policy_id, n_benefs=n_benefs, m=m, env=env, H=H, use_informed_prior=use_informed_prior)
    
    def __call__(self, w, K, epsilon=0.1):
        self.K = K
        self.epsilon = epsilon
        compute = tf.custom_gradient(lambda x: self._compute(x))
        return compute(w)

    def _compute(self, w_raw):
        w = tf.stop_gradient(w_raw)
        # Fast soft Whittle simulation
        traj, simulated_rewards, state_record, action_record, reward_record = getSimulatedTrajectories(
                                                    n_benefs=self.n_benefs, T=self.T, K=self.K, n_trials=self.OPE_sim_n_trials, gamma=self.gamma, epsilon=self.epsilon, 
                                                    T_data=self.emp_T_data, R_data=self.emp_R_data,
                                                    w=w.numpy(), policies=[3], fast=True
                                                    )
        
        average_reward = tf.reduce_mean(tf.convert_to_tensor(simulated_rewards, dtype=tf.float32))

        def gradient_function(dsoln):
            gamma_list = np.reshape(self.gamma ** np.arange(self.T), (1,1,self.T,1))
            discounted_reward_record = reward_record * gamma_list
            cumulative_rewards = tf.math.cumsum(tf.convert_to_tensor(discounted_reward_record, dtype=tf.float32), axis=2, reverse=True)
            with tf.GradientTape() as tmp_tape:
                tmp_tape.watch(w)
                probs_raw = tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, self.n_benefs), policy=3, ts=None, w=w, k=self.K, epsilon=self.epsilon), (self.OPE_sim_n_trials, self.T, self.n_benefs))
                selected_probs = probs_raw * action_record[:,0,:,:] + (1 - probs_raw) * (1 - action_record[:,0,:,:]) # [ntr, 1, self.T, n_benefs]

                # tf.reshape(getProbs(state_record[:,0,:,:].reshape(-1, self.n_benefs), policy=3, ts=None, w=w, k=self.K), (-1, 1, self.T, self.n_benefs))
                selected_logprobs = tf.math.log(selected_probs)
                
                # total_reward = tf.reduce_mean(tf.reduce_sum(cumulative_rewards[:,0,-1,:] * selected_logprobs[:,-1,:], axis=(-1)))
                total_reward = tf.reduce_mean(tf.reduce_sum(cumulative_rewards[:,0,:,:] * selected_logprobs[:,:,:], axis=(-1)))

            dtotal_dw = tmp_tape.gradient(total_reward, w)
            del tmp_tape

            return dtotal_dw * dsoln

        return tf.stop_gradient(average_reward), gradient_function


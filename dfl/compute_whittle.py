"""
standard Whittle index computation based on binary search

POSSIBLE OPTIMIZATIONS TO HELP SPEED
- keep track of top k WIs so far. then in future during binary search, if we go below that WI just quit immediately
"""

import sys
import numpy as np

import heapq  # priority queue

whittle_threshold = 1e-4
value_iteration_threshold = 1e-2


def get_content_state_reward(content_state,reward_function):
    if reward_function == 1:
        return [0, .8, .9, 1][content_state]
    if reward_function == 2:
        return [0, 1, 1, 1][content_state]
    else:
        return [0, .33, .66, 1][content_state]


def arm_value_iteration(transitions, state, lamb_val, discount, threshold=value_iteration_threshold,
                        content_state=None, content_transitions = None,reward_function=1, reward_estimate = [0,1]):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """
    assert discount < 1

    if content_state is not None:
        n_states, _, n_actions = content_transitions.shape
    else:
        n_states, n_actions = transitions.shape
    value_func = np.random.rand(n_states)
    difference = np.ones((n_states))
    iters = 0

    # lambda-adjusted reward function
    def reward_simple(s, a):
        if reward_estimate is None:
            return s - a * lamb_val
        return reward_estimate[s] - a * lamb_val

    def reward_content_optimized(c_s, a):
        return get_content_state_reward(c_s, reward_function) - a * lamb_val
        # return s - a * lamb_val


    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_states, n_actions))

        for s in range(n_states):
            Q_val_s0 = 0
            Q_val_s1 = 0
            for a in range(n_actions):
                # transitioning to state = 0
                if content_state is None:
                    reward = reward_simple(s,a)
                    Q_func[s, a] += reward
                    # transitioning to state = 0
                    Q_func[s, a] += (1 - transitions[s, a]) * (discount * value_func[0])

                    # transitioning to state = 1
                    Q_func[s, a] += transitions[s, a] * (discount * value_func[1])
                else:
                    reward = reward_content_optimized(s, a)
                    # transitioning to 0
                    Q_func[s,a] += reward
                    Q_func[s, a] += (content_transitions[s, 0, a]) * ( discount * value_func[0])
                    # transitioning to 1
                    Q_func[s, a] += content_transitions[s, 1, a] * (discount * value_func[1])
                    # transitioning to 2
                    Q_func[s, a] += (content_transitions[s, 2, a]) * (discount * value_func[2])
                    # transitioning to 3
                    Q_func[s, a] += content_transitions[s,3, a] * (discount * value_func[3])
            value_func[s] = np.max(Q_func[s, :])

        difference = np.abs(orig_value_func - value_func)

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
    return np.argmax(Q_func[state, :])


def get_init_bounds(transitions):
    lb = -1
    ub = 1
    return lb, ub


def arm_compute_whittle(transitions, state, discount, subsidy_break, eps=whittle_threshold, content_state=None, content_transitions = None,reward=1, reward_estimate=None):
    """
    compute whittle index for a single arm using binary search

    subsidy_break = the min value at which we stop iterating

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    lb, ub = get_init_bounds(transitions)  # return lower and upper bounds on WI
    top_WI = []
    while abs(ub - lb) > eps:
        lamb_val = (lb + ub) / 2
        # print('lamb', lamb_val, lb, ub)

        # we've already filled our knapsack with higher-valued WIs
        if ub < subsidy_break:
            # print('breaking early!', subsidy_break, lb, ub)
            return -10

        action = arm_value_iteration(transitions, state, lamb_val, discount, content_state=content_state, content_transitions = content_transitions, reward_function=reward, reward_estimate=reward_estimate)
        if action == 0:
            # optimal action is passive: subsidy is too high
            ub = lamb_val
        elif action == 1:
            # optimal action is active: subsidy is too low
            lb = lamb_val
        else:
            raise Error(f'action not binary: {action}')
    subsidy = (ub + lb) / 2
    return subsidy


def arm_compute_whittle_optimized(transitions, state, budget, discount, subsidy_break, eps=whittle_threshold, content_state=None, content_transitions = None, baseline=True, reward=1, reward_estimate=None):
    """
    compute whittle index for a single arm using binary search

    subsidy_break = the min value at which we stop iterating

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    n = len(state)
    lb, ub = get_init_bounds(transitions)  # return lower and upper bounds on WI
    top_WI = set()
    while len(top_WI) != budget:
        lamb_val = (lb + ub) / 2
        # we've already filled our knapsack with higher-valued WIs
        temp_top_WI=set()
        for i in range(n):
            if i not in top_WI:
                action = arm_value_iteration(transitions[i], state[i], lamb_val, discount, content_state=None if baseline else content_state[i],
                                         content_transitions= None if baseline else content_transitions[i], reward_function=reward, reward_estimate=reward_estimate)
                if action == 1:
                    # optimal action is active: subsibdy is too low
                    temp_top_WI.add(i)
                elif action != 0:
                    raise Error(f'action not binary: {action}')
        if len(temp_top_WI) + len(top_WI) <= budget:
            top_WI = top_WI.union(temp_top_WI)
            ub = lamb_val
        elif abs(ub-lb) < eps:
            top_WI = top_WI.union(list(temp_top_WI)[:budget-len(top_WI)])
            lb = lamb_val
        else:
            lb = lamb_val
    return top_WI
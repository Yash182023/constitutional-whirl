import numpy as np
# create different types of listeners
from compute_whittle import arm_compute_whittle,arm_compute_whittle_optimized


def create_transitions(n):
    transitions = np.zeros((n, 2, 2))
    for i in range(n):
        transitions[i, 0, 0] = 0.3
        transitions[i, 0, 1] = 0.6
        transitions[i, 1, 0] = 0.8
        transitions[i, 1, 1] = 0.9
    return transitions


def create_transitions_random(n):
    transitions = np.zeros((n, 2, 2))
    for i in range(n):
        transitions[i, 0, 0] = np.random.uniform(0,0.5)
        transitions[i, 0, 1] = np.random.uniform(.5,1)
        transitions[i, 1, 0] = np.random.uniform(0,.7)
        transitions[i, 1, 1] = np.random.uniform(.8,.999)
    return transitions

def create_transitions_n_history(n_history,n_benefs,seed=0):
    np.random.seed(seed)
    transitions = np.zeros((n_benefs, 2, 2))
    for i in range(n_benefs):
        transitions[i, 0, 0] = np.random.uniform(0, 0.5)
        transitions[i, 0, 1] = np.random.uniform(.5, 1)
        transitions[i, 1, 0] = np.random.uniform(0, .7)
        transitions[i, 1, 1] = np.random.uniform(.8, .999)
    transitions_full = np.zeros((n_benefs, 2**n_history, 2, 2**n_history))
    for i in range(n_benefs):
        for j in range(2**n_history):
            bin_val = bin(j)[2:]
            curr_state = (n_history-len(bin_val))*'0' + bin_val
            next_state_0 = int(curr_state[1:] + '0', 2)
            next_state_1 = int(curr_state[1:] + '1', 2)

            # action = 0
            trans = transitions[i,int(bin(j)[-1]), 0]
            transitions_full[i, j, 0, next_state_0] = 1 - trans
            transitions_full[i, j, 0, next_state_1] = trans

            # action = 1
            trans = transitions[i,int(bin(j)[-1]), 1]
            transitions_full[i, j, 1, next_state_0] = 1 - trans
            transitions_full[i, j, 1, next_state_1] = trans

    return transitions_full


def create_full_transition_matrix(T):
    n, n_states, n_actions = np.shape(T)
    full_transitions = np.zeros((n, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = T
    full_transitions[:, :, :, 0] = 1-T
    return full_transitions


def create_transitions_biased(n):
    transitions = np.zeros((n, 2, 2))
    for i in range(n):
        if i%2==0:
            transitions[i, 0, 0] = 0.8
            transitions[i, 0, 1] = 0.9
            transitions[i, 1, 0] = 0.2
            transitions[i, 1, 1] = 0.9
        else:
            transitions[i, 0, 0] = 0.4
            transitions[i, 0, 1] = 0.7
            transitions[i, 1, 0] = 0.5
            transitions[i, 1, 1] = 0.8
    return transitions

def create_transitions_biased_full(n):
    transitions = np.zeros((n, 2, 2, 2))
    for i in range(n):
        if i%2==0:
            transitions[i, 0, 0, 1] = 0.8
            transitions[i, 0, 0, 0] = 1 - transitions[i, 0, 0, 1]
            transitions[i, 0, 1, 1] = 0.9
            transitions[i, 0, 1, 0] = 1 - transitions[i, 0, 1, 1]
            transitions[i, 1, 0, 1] = 0.2
            transitions[i, 1, 0, 0] = 1 - transitions[i, 1, 0, 1]
            transitions[i, 1, 1, 1] = 0.9
            transitions[i, 1, 1, 0] = 1 - transitions[i, 1, 1, 1]
        else:
            transitions[i, 0, 0, 1] = 0.4
            transitions[i, 0, 0, 0] = 1 - transitions[i, 0, 0, 1]
            transitions[i, 0, 1, 1] = 0.7
            transitions[i, 0, 1, 0] = 1 - transitions[i, 0, 1, 1]
            transitions[i, 1, 0, 1] = 0.5
            transitions[i, 1, 0, 0] = 1 - transitions[i, 1, 0, 1]
            transitions[i, 1, 1, 1] = 0.8
            transitions[i, 1, 1, 0] = 1 - transitions[i, 1, 1, 1]
    return transitions


def create_content_transitions(transitions):
    n, n_states, n_actions = np.shape(transitions)
    content_transitions = np.zeros((n, 4, 4, 2))
    for i in range(n):
        for a in range(n_actions):
            content_transitions[i, 0, 0, a] = 1 - transitions[i, 0, a]
            content_transitions[i, 1, 0, a] = 1 - transitions[i, 0, a]
            content_transitions[i, 2, 1, a] = 1 - transitions[i, 1, a]
            content_transitions[i, 3, 1, a] = 1 - transitions[i, 1, a]
            content_transitions[i, 0, 2, a] = transitions[i, 0, a]
            content_transitions[i, 1, 2, a] = transitions[i, 0, a]
            content_transitions[i, 2, 3, a] = transitions[i, 1, a]
            content_transitions[i, 3, 3, a] = transitions[i, 1, a]
    return content_transitions

def create_full_transitions(transitions):
    n, n_states, n_actions = np.shape(transitions)
    content_transitions = np.zeros((n, 4, 2, 4))
    for i in range(n):
        for a in range(n_actions):
            content_transitions[i, 0, a, 0] = 1 - transitions[i, 0, a]
            content_transitions[i, 1, a, 0] = 1 - transitions[i, 0, a]
            content_transitions[i, 2, a, 1] = 1 - transitions[i, 1, a]
            content_transitions[i, 3, a, 1] = 1 - transitions[i, 1, a]
            content_transitions[i, 0, a, 2] = transitions[i, 0, a]
            content_transitions[i, 1, a, 2] = transitions[i, 0, a]
            content_transitions[i, 2, a, 3] = transitions[i, 1, a]
            content_transitions[i, 3, a, 3] = transitions[i, 1, a]
    return content_transitions


def step(state, arms_to_pull, transitions):
    n = len(transitions)
    new_state = np.zeros(n, dtype=int)
    for i in range(n):
        if i in arms_to_pull:
            new_state[i] = int(1 if np.random.random() < transitions[i][state[i]][1] else 0)
        else:
            new_state[i] = int(1 if np.random.random() < transitions[i][state[i]][0] else 0)
    return new_state


def get_whittle_indices(transitions, state, discount, subsidy_break, reward_estimate):
    n=len(state)
    whittle_indices = np.zeros(n)
    for i in range(n):
        whittle_indices[i] = arm_compute_whittle(transitions[i], state[i], discount, subsidy_break, reward_estimate=reward_estimate)
    return whittle_indices


def get_whittle_indices_with_content_state(transitions, state, discount, subsidy_break, content_state, content_transitions, reward_estimate):
    n=len(state)
    whittle_indices = np.zeros(n)
    for i in range(n):
        whittle_indices[i] = arm_compute_whittle(transitions[i], state[i], discount, subsidy_break,
                                                 content_state=content_state[i], content_transitions=content_transitions[i], reward_estimate=reward_estimate)
    return whittle_indices


def step_content_state(content_state, state):
    # content state for values 4 state = [00, 10, 01, 11]
    n = len(state)
    new_content_state = np.zeros(n, dtype=int)
    for i in range(n):
        if content_state[i] == 0:
            if state[i] == 1:
                new_content_state[i] = 2
        if content_state[i] == 1:
            new_content_state[i] = max(min(content_state[i] + (2*state[i]-1), 3), 0)
        if content_state[i] == 2:
            new_content_state[i] = max(min(content_state[i] + (2 * state[i] - 1), 3), 0)
        if content_state[i] == 3:
            if state[i] == 0:
                new_content_state[i] = 1
            else:
                new_content_state[i] = 1
    return new_content_state

def get_top_actions(transitions, state, k, discount, subsidy_break=0, content_state=None,
                                                content_transitions=None,baseline=False, optimized_whittle=False, reward = 1, reward_estimate=[0,1]):
    if optimized_whittle:
        return list(arm_compute_whittle_optimized(transitions, state, k, discount, subsidy_break, content_state=content_state,
                                                content_transitions=content_transitions,baseline=baseline, reward=reward, reward_estimate=reward_estimate))
    else:
        if baseline:
            whittle_indices = get_whittle_indices(transitions, state, discount, subsidy_break, reward_estimate=reward_estimate)
        else:
            whittle_indices = get_whittle_indices_with_content_state(transitions, state, discount, subsidy_break, content_state=content_state, content_transitions=content_transitions, reward=reward, reward_estimate=reward_estimate)
        return np.argsort(whittle_indices)[-1 * k:]

def reward_whittle_comparison(n,k,t, discount=0.9, subsidy_break=0,verbose=False, num_examples=1):
    transitions = create_transitions_biased(n)
    content_transitions = create_content_transitions(transitions)
    total_engagement = np.zeros(t)
    total_reward = np.zeros(t)
    r1_total_engagement = np.zeros(t)
    r1_total_reward = np.zeros(t)
    r2_total_engagement = np.zeros(t)
    r2_total_reward = np.zeros(t)
    r3_total_engagement = np.zeros(t)
    r3_total_reward = np.zeros(t)
    print("TRANSITION PROBABILITIES")
    print("==================")
    print("Active from state 1: " + str(transitions[:, 1, 1]))
    print("Active from state 0: " + str(transitions[:, 0, 1]))
    print("Passive from state 1: " + str(transitions[:, 1, 0]))
    print("Passive from state 0: " + str(transitions[:, 0, 0]))
    for i in range(num_examples):
        state = np.zeros(n, dtype=int)
        state[:int(n/2)] = 1
        content_state = np.zeros(n, dtype=int)
        content_state[:int(n/2)] = 2
        # simple whittle index
        for timestep in range(t):
            top_actions = get_top_actions(transitions, state, k, discount, subsidy_break, baseline=True,
                                          optimized_whittle=True)
            total_engagement[timestep] += np.sum(content_state > 0)
            total_reward[timestep] += np.sum(state)
            state = step(state, top_actions, transitions)
            content_state = step_content_state(content_state, state)
        # content optimized whittle # 1
        state = np.zeros(n, dtype=int)
        state[:int(n/2)] = 1
        content_state = np.zeros(n, dtype=int)
        content_state[:int(n / 2)] = 2
        for timestep in range(t):
            top_actions = get_top_actions(transitions, state, k, discount, subsidy_break, content_state=content_state,
                                          content_transitions=content_transitions, baseline=False,
                                          optimized_whittle=True,reward=1)
            r1_total_engagement[timestep] += np.sum(content_state > 0)
            r1_total_reward[timestep] += np.sum(state)
            state = step(state, top_actions, transitions)
            content_state = step_content_state(content_state, state)
        # content optimized whittle # 2
        state = np.zeros(n, dtype=int)
        state[:int(n/2)] = 1
        content_state = np.zeros(n, dtype=int)
        content_state[:int(n / 2)] = 2
        for timestep in range(t):
            top_actions = get_top_actions(transitions, state, k, discount, subsidy_break,
                                          content_state=content_state,
                                          content_transitions=content_transitions, baseline=False,
                                          optimized_whittle=True,reward=2)
            r2_total_engagement[timestep] += np.sum(content_state > 0)
            r2_total_reward[timestep] += np.sum(state)
            state = step(state, top_actions, transitions)
            content_state = step_content_state(content_state, state)
        # content optimized whittle # 3
        state = np.zeros(n, dtype=int)
        state[:int(n/2)] = 1
        content_state = np.zeros(n, dtype=int)
        content_state[:int(n / 2)] = 2
        for timestep in range(t):
            top_actions = get_top_actions(transitions, state, k, discount, subsidy_break,
                                          content_state=content_state,
                                          content_transitions=content_transitions, baseline=False,
                                          optimized_whittle=True,reward=3)
            r3_total_engagement[timestep] += np.sum(content_state > 0)
            r3_total_reward[timestep] += np.sum(state)
            state = step(state, top_actions, transitions)
            content_state = step_content_state(content_state, state)
    return total_engagement / float(num_examples), total_reward / float(num_examples), r1_total_engagement / float(
        num_examples), r1_total_reward / float(num_examples), r2_total_engagement / float(
        num_examples), r2_total_reward / float(num_examples), r3_total_engagement / float(
        num_examples), r3_total_reward / float(num_examples)


def whittle_comparison(n, k, t, discount=0.9, subsidy_break=0,verbose=False, num_examples=1):
    transitions = create_transitions_biased(n)
    content_transitions = create_content_transitions(transitions)
    total_engagement = np.zeros(t)
    total_reward = np.zeros(t)
    content_total_engagement = np.zeros(t)
    content_total_reward = np.zeros(t)
    print("TRANSITION PROBABILITIES")
    print("==================")
    print("Active from state 1: " + str(transitions[:, 1, 1]))
    print("Active from state 0: " + str(transitions[:, 0, 1]))
    print("Passive from state 1: " + str(transitions[:, 1, 0]))
    print("Passive from state 0: " + str(transitions[:, 0, 0]))
    for i in range(num_examples):
        state = np.zeros(n, dtype=int)
        # state[:int(n/2)] = 1
        content_state = np.zeros(n, dtype=int)
        # content_state[:int(n/2)] = 2
        # simple whittle index
        print("SIMPLE WHITTLE")
        print("==================")
        for timestep in range(t):
            top_actions = get_top_actions(transitions, state, k, discount, subsidy_break, baseline=True, optimized_whittle=True)
            total_engagement[timestep] += np.sum(content_state>0)
            total_reward[timestep] += np.sum(state)
            state = step(state, top_actions, transitions)
            content_state = step_content_state(content_state, state)
            if verbose:
                print(state, content_state, top_actions)
        # content optimized whittle
        state = np.zeros(n, dtype=int)
        # state[:int(n/2)] = 1
        content_state = np.zeros(n, dtype=int)
        # content_state[:int(n / 2)] = 2
        print("CONTENT OPTIMIZED WHITTLE")
        print("==================")
        for timestep in range(t):
            top_actions = get_top_actions(transitions, state, k, discount, subsidy_break, content_state=content_state, content_transitions=content_transitions,baseline=False, optimized_whittle=True)
            content_total_engagement[timestep] += np.sum(content_state > 0)
            content_total_reward[timestep] += np.sum(state)
            state = step(state, top_actions, transitions)
            content_state = step_content_state(content_state, state)
            if verbose:
                print(state, content_state, top_actions)
    return total_engagement/float(num_examples), total_reward/float(num_examples), content_total_engagement/float(num_examples), content_total_reward/float(num_examples)


def create_passive_states(n, t):
    listener_type = 0
    passive_states = np.zeros(n, t)
    for i in range(n):
        listener_type = listener_type % 5
        passive_states[i] = generate_listener_probabilities(t, listener_type)
        listener_type += 1
    return passive_states


def generate_listener_probabilities(t, listener_type):
    # low listener
    passive_states = np.zeros(t)
    pattern = np.random.randint(6) + 1
    if listener_type == 0:
        low_prob = 0.1
        high_prob = 0.3
    # low medium listener
    if listener_type == 1:
        low_prob = 0.3
        high_prob = 0.5
    # medium high listener
    if listener_type == 2:
        low_prob = 0.5
        high_prob = 0.7
    # high listener
    if listener_type == 3:
        low_prob = 0.7
        high_prob = 0.9
    # random listener
    if listener_type == 4:
        low_prob = np.random.random()
        high_prob = low_prob
    for i in range(t):
        if i % 6 <= pattern:
            passive_states[i] = 1 if np.random.random() <= low_prob else 1
        else:
            passive_states[i] = 1 if np.random.random() <= high_prob else 1
    return passive_states

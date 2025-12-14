import os
import shutil
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enhanced Data Generator with Multiple Beta Profiles

def generate_real_world_proxy_transitions(n_arms, n_states=2, seed=None,
                                         beta_a=2, beta_b=2, profile_name="Standard"):
    """
    Args:
        beta_a, beta_b: Beta distribution parameters for grit
        profile_name: Descriptive name for this parameter set
    """
    if seed is not None:
        np.random.seed(seed)

    transitions = np.zeros((n_arms, 2, 2, 2))
    metadata = {
        'grit': np.zeros(n_arms),
        'self_recovery': np.zeros(n_arms),
        'intervention_lift': np.zeros(n_arms),
        'profile': profile_name,
        'beta_params': (beta_a, beta_b)
    }

    for i in range(n_arms):
        grit = np.random.beta(a=beta_a, b=beta_b)
        metadata['grit'][i] = grit

        p_stay_active = grit * 0.9
        p_self_recovery = np.random.beta(a=1, b=5) * 0.2
        metadata['self_recovery'][i] = p_self_recovery

        # Passive transitions
        transitions[i, 0, 0, 1] = p_self_recovery
        transitions[i, 0, 0, 0] = 1 - p_self_recovery
        transitions[i, 0, 1, 1] = p_stay_active
        transitions[i, 0, 1, 0] = 1 - p_stay_active

        # Active transitions
        lift = 0.4 * (1 - grit)
        metadata['intervention_lift'][i] = lift

        p_active_recovery = np.clip(p_self_recovery + lift, 0, 0.95)
        p_active_retention = np.clip(p_stay_active + (lift/2), 0, 0.99)

        transitions[i, 1, 0, 1] = p_active_recovery
        transitions[i, 1, 0, 0] = 1 - p_active_recovery
        transitions[i, 1, 1, 1] = p_active_retention
        transitions[i, 1, 1, 0] = 1 - p_active_retention

    return transitions.astype(np.float32), metadata


def validate_data_quality(transitions, metadata):
    """Comprehensive validation with monotonicity checks."""
    n = len(transitions)
    report = {
        'n_beneficiaries': n,
        'profile': metadata['profile'],
        'beta_params': metadata['beta_params'],
        'grit_mean': np.mean(metadata['grit']),
        'grit_std': np.std(metadata['grit']),
        'bimodality_coefficient': None,
        'monotonicity_violations': 0,
        'intervention_efficacy': 0
    }

    grit = metadata['grit']
    m3 = stats.skew(grit)
    m4 = stats.kurtosis(grit, fisher=False)
    bc = (m3**2 + 1) / m4 if m4 != 0 else 0
    report['bimodality_coefficient'] = bc

    for i in range(n):
        P = transitions[i]
        if P[1, 0, 1] < P[0, 0, 1] - 1e-6 or P[1, 1, 1] < P[0, 1, 1] - 1e-6:
            report['monotonicity_violations'] += 1
        else:
            report['intervention_efficacy'] += 1

    return report

# CONSTRAINT ENFORCEMENT METHODS


class ConstraintMethod:
    """Base class for constraint enforcement methods."""

    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon
        self.name = "Base"

    def project(self, rewards):
        raise NotImplementedError

    def check_violations(self, rewards):
        """Counting number of constraint violations."""
        violations = np.sum(rewards[:, 1] < rewards[:, 0] - self.epsilon)
        return violations


class ProjectedGradient(ConstraintMethod):
    """Projected Gradient Descent - projects to feasible region after each update."""

    def __init__(self, epsilon=1e-3):
        super().__init__(epsilon)
        self.name = "Projected_Gradient"

    def project(self, rewards):
        """Projects rewards to satisfy R(1) >= R(0)."""
        r_sick = rewards[:, 0]
        r_healthy = rewards[:, 1]

        violations = r_healthy < r_sick
        r_healthy[violations] = r_sick[violations] + self.epsilon

        return np.stack([r_sick, r_healthy], axis=1)


class LagrangianMultiplier(ConstraintMethod):
    """Lagrangian method - adds penalty term to objective."""

    def __init__(self, lambda_init=1.0, lambda_lr=0.01, epsilon=1e-3):
        super().__init__(epsilon)
        self.name = "Lagrangian"
        self.lambda_penalty = lambda_init
        self.lambda_lr = lambda_lr

    def get_penalty(self, rewards):
        """Compute penalty for constraint violations."""
        violations = np.maximum(0, rewards[:, 0] - rewards[:, 1] + self.epsilon)
        penalty = self.lambda_penalty * np.sum(violations)
        return penalty

    def update_lambda(self, rewards):
        """Update Lagrange multiplier based on constraint satisfaction."""
        violation_magnitude = np.mean(np.maximum(0, rewards[:, 0] - rewards[:, 1]))
        self.lambda_penalty += self.lambda_lr * violation_magnitude
        self.lambda_penalty = np.clip(self.lambda_penalty, 0.1, 100.0)

    def project(self, rewards):
        """Soft projection using current lambda."""
        r_sick = rewards[:, 0]
        r_healthy = rewards[:, 1]

        # soft constraint
        violations = r_healthy < r_sick
        adjustment = self.lambda_penalty * (r_sick[violations] - r_healthy[violations])
        r_healthy[violations] += adjustment * 0.1  # Gradual adjustment

        return np.stack([r_sick, r_healthy], axis=1)


class LogBarrier(ConstraintMethod):
    """Log-barrier method - adds logarithmic penalty near constraint boundary."""

    def __init__(self, mu=0.1, epsilon=1e-3):
        super().__init__(epsilon)
        self.name = "Log_Barrier"
        self.mu = mu  # Barrier parameter

    def get_barrier(self, rewards):
        """Compute log-barrier penalty."""
        slack = rewards[:, 1] - rewards[:, 0] + self.epsilon
        slack = np.maximum(slack, 1e-8)  # Avoid log(0)
        barrier = -self.mu * np.sum(np.log(slack))
        return barrier

    def project(self, rewards):
        """Apply barrier-based adjustment."""
        r_sick = rewards[:, 0]
        r_healthy = rewards[:, 1]

        slack = r_healthy - r_sick + self.epsilon
        violations = slack <= 0

        # For severe violations
        if np.any(violations):
            r_healthy[violations] = r_sick[violations] + 2 * self.epsilon

        return np.stack([r_sick, r_healthy], axis=1)


class Unconstrained(ConstraintMethod):
    """Baseline: No constraints applied."""

    def __init__(self):
        super().__init__()
        self.name = "Unconstrained"

    def project(self, rewards):
        """No projection - return rewards as-is."""
        return rewards

# ABLATION - Test Without Aggregate Command

def train_baseline_comparison(transitions, risk_scores, n_arms=500, n_folds=5,
                              test_aggregate=True, constraint_method=None):
    """
    Comparing violations with and without aggregate command.

    """
    from dfl.whittle import newWhittleIndex
    from dfl.trajectory import getSimulatedTrajectories
    from dfl.ope import eval_policy

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    results_with_command = []
    results_without_command = []

    print("\n" + "="*80)
    print(f"ABLATION STUDY: {'WITH' if test_aggregate else 'WITHOUT'} Aggregate Command")
    print("="*80)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_arms))):
        n_train = len(train_idx)
        train_transitions = transitions[train_idx]
        train_risk_scores = risk_scores[train_idx]

        # Generate baseline trajectories
        reward_numpy = np.zeros((n_train, 2))
        reward_numpy[:, 1] = 1
        reward_opt = tf.constant(reward_numpy, dtype=tf.float32)

        w_opt = newWhittleIndex(
            tf.constant(train_transitions, dtype=tf.float32),
            reward_opt
        ).numpy()
        w_opt = np.reshape(w_opt, (n_train, 2))

        _, _, state_traj, action_traj, _ = getSimulatedTrajectories(
            n_benefs=n_train, T=11, K=int(60 * n_train / n_arms),
            n_trials=1, gamma=0.99, T_data=train_transitions,
            R_data=reward_numpy, w=w_opt, replace=False,
            policies=[3], fast=True
        )

        # Test WITHOUT aggregate command
        violations_without = test_irl_violations(
            train_transitions, state_traj, action_traj,
            w_opt, constraint_method, epochs=50
        )
        results_without_command.append(violations_without)

        # Test WITH aggregate command
        if test_aggregate:
            edited_action_traj = apply_aggregate_command(
                action_traj, train_risk_scores, K=int(60 * n_train / n_arms)
            )
            violations_with = test_irl_violations(
                train_transitions, state_traj, edited_action_traj,
                w_opt, constraint_method, epochs=50
            )
            results_with_command.append(violations_with)

        print(f"Fold {fold_idx+1}: Without={violations_without:.1f}%, " +
              (f"With={violations_with:.1f}%" if test_aggregate else ""))

    return {
        'without_command_mean': np.mean(results_without_command),
        'without_command_std': np.std(results_without_command),
        'with_command_mean': np.mean(results_with_command) if test_aggregate else None,
        'with_command_std': np.std(results_with_command) if test_aggregate else None,
        'all_without': results_without_command,
        'all_with': results_with_command if test_aggregate else None
    }


def test_irl_violations(transitions, state_traj, action_traj, w_opt,
                       constraint_method, epochs=50):
    from dfl.whittle import newWhittleIndex
    from dfl.ope import eval_policy

    n, n_states = transitions.shape[0], 2
    rewards = np.zeros((n, n_states))
    reward_param = tf.Variable(rewards, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            w = newWhittleIndex(
                tf.constant(transitions, dtype=tf.float32), reward_param
            )
            w = tf.reshape(w, (n, n_states))

            performance, _, _, _, _ = eval_policy(
                state_record=state_traj, action_record=action_traj,
                w=w, w_opt=w_opt, H=10, K=int(60 * n / 500),
                gamma=0.99, target_policy_name='soft-whittle',
                beh_policy_name='soft-whittle',
                transition_probabilities=transitions, epsilon=0.1,
                reward_estimate=reward_param, rewards_true=None
            )

        grads = tape.gradient(performance, reward_param)
        optimizer.apply_gradients(zip([grads], [reward_param]))

        if constraint_method:
            projected = constraint_method.project(reward_param.numpy())
            reward_param.assign(projected)

    final_rewards = reward_param.numpy()
    violations = np.sum(final_rewards[:, 1] < final_rewards[:, 0])
    return (violations / n) * 100

# Ground Truth Correlation Analysis

def analyze_reward_quality(results, metadata):
    print("\n" + "="*80)
    print("GROUND TRUTH CORRELATION ANALYSIS")
    print("="*80)

    correlations = {}

    for fold_idx, fold in enumerate(results['folds']):
        train_idx = fold['train_idx']
        true_grit = metadata['grit'][train_idx]

        for method_name, method_data in fold['methods'].items():
            learned_rewards = method_data['learned_rewards']
            r_healthy = learned_rewards[:, 1]  # Reward for adherence state

            # Correlation between learned R(healthy) and true grit
            corr, p_value = stats.pearsonr(r_healthy, true_grit)

            if method_name not in correlations:
                correlations[method_name] = []
            correlations[method_name].append(corr)

    # Print results
    print("\nCorrelation: Learned R(Healthy) vs True Grit")
    print("(Higher correlation = better reward learning)\n")

    summary = []
    for method_name, corrs in correlations.items():
        mean_corr = np.mean(corrs)
        std_corr = np.std(corrs)
        summary.append({
            'Method': method_name,
            'Correlation (mean ± std)': f"{mean_corr:.3f} ± {std_corr:.3f}",
            'Quality': '✓ Good' if mean_corr > 0.3 else '⚠ Weak'
        })

    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    print("\nInterpretation:")
    print("  > 0.5: Strong correlation - rewards capture adherence well")
    print("  0.3-0.5: Moderate - rewards partially capture adherence")
    print("  < 0.3: Weak - rewards may not be meaningful")

    return correlations

# Beta Parameter Sensitivity Analysis

def beta_sensitivity_experiment(n_arms=500, n_folds=3, epochs=30):
    beta_profiles = [
        (2, 2, "Standard (U-shaped)"),
        (0.5, 0.5, "Extreme Bimodal"),
        (5, 2, "Right-skewed (High Grit)"),
        (2, 5, "Left-skewed (Low Grit)"),
        (1, 1, "Uniform")
    ]

    print("\n" + "="*80)
    print("BETA PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)

    sensitivity_results = []

    for beta_a, beta_b, profile_name in beta_profiles:
        print(f"\nTesting: {profile_name} (Beta({beta_a}, {beta_b}))")

        transitions, metadata = generate_real_world_proxy_transitions(
            n_arms, seed=SEED, beta_a=beta_a, beta_b=beta_b,
            profile_name=profile_name
        )

        validation = validate_data_quality(transitions, metadata)
        print(f"  Bimodality Coefficient: {validation['bimodality_coefficient']:.3f}")
        print(f"  Grit Mean: {validation['grit_mean']:.3f} ± {validation['grit_std']:.3f}")

        risk_scores = np.random.RandomState(SEED).randint(0, 4, size=n_arms)

        baseline_violations = []
        constrained_violations = []

        # Quick 3-fold test per profile
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

        for fold_idx, (train_idx, _) in enumerate(kf.split(np.arange(n_arms))):
            # Test baseline
            v_baseline = quick_violation_test(
                transitions[train_idx], risk_scores[train_idx],
                use_constraint=False, epochs=epochs
            )
            baseline_violations.append(v_baseline)

            # Test with PGD
            v_constrained = quick_violation_test(
                transitions[train_idx], risk_scores[train_idx],
                use_constraint=True, epochs=epochs
            )
            constrained_violations.append(v_constrained)

        sensitivity_results.append({
            'Profile': profile_name,
            'Beta': f"({beta_a}, {beta_b})",
            'Baseline Violations (%)': f"{np.mean(baseline_violations):.1f} ± {np.std(baseline_violations):.1f}",
            'PGD Violations (%)': f"{np.mean(constrained_violations):.1f} ± {np.std(constrained_violations):.1f}",
            'Grit Mean': f"{validation['grit_mean']:.3f}"
        })

    df = pd.DataFrame(sensitivity_results)
    print("\n" + "="*80)
    print("SENSITIVITY RESULTS")
    print("="*80)
    print("\n" + df.to_string(index=False))

    return sensitivity_results


def quick_violation_test(transitions, risk_scores, use_constraint, epochs=30):
    from dfl.whittle import newWhittleIndex
    from dfl.trajectory import getSimulatedTrajectories

    n = len(transitions)
    reward_numpy = np.zeros((n, 2))
    reward_numpy[:, 1] = 1

    w_opt = newWhittleIndex(
        tf.constant(transitions, dtype=tf.float32),
        tf.constant(reward_numpy, dtype=tf.float32)
    ).numpy()
    w_opt = np.reshape(w_opt, (n, 2))

    _, _, state_traj, action_traj, _ = getSimulatedTrajectories(
        n_benefs=n, T=11, K=int(60 * n / 500), n_trials=1, gamma=0.99,
        T_data=transitions, R_data=reward_numpy, w=w_opt,
        replace=False, policies=[3], fast=True
    )

    edited_action = apply_aggregate_command(action_traj, risk_scores, K=int(60 * n / 500))

    constraint = ProjectedGradient() if use_constraint else None
    violations_pct = test_irl_violations(
        transitions, state_traj, edited_action, w_opt, constraint, epochs=epochs
    )

    return violations_pct

# K-FOLD TRAINING FRAMEWORK

def train_whirl_with_constraint(
    transition_probs,
    state_traj,
    action_traj,
    w_opt,
    constraint_method,
    epochs=50,
    learning_rate=0.01,
    L=10,
    K=60,
    gamma=0.99,
    verbose=False
):
    """
    Args:
        transition_probs: Transition probability matrix
        state_traj: State trajectories
        action_traj: Action trajectories
        w_opt: Optimal Whittle indices
        constraint_method: ConstraintMethod instance
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        L: Time horizon
        K: Budget constraint
        gamma: Discount factor
        verbose: Print training progress

    Returns:
        learned_rewards: Final learned rewards
        training_history: Dict with training metrics
    """
    from dfl.whittle import newWhittleIndex
    from dfl.ope import eval_policy

    n, n_states = transition_probs.shape[0], 2

    # Initialize rewards
    rewards = np.zeros((n, n_states))
    reward_param = tf.Variable(rewards, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Training history
    history = {
        'performance': [],
        'violations': [],
        'reward_norm': []
    }

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Compute Whittle indices
            w = newWhittleIndex(
                tf.constant(transition_probs, dtype=tf.float32),
                reward_param
            )
            w = tf.reshape(w, (n, n_states))

            # Evaluate policy
            performance, _, _, _, _ = eval_policy(
                state_record=state_traj,
                action_record=action_traj,
                w=w,
                w_opt=w_opt,
                H=L,
                K=K,
                gamma=gamma,
                target_policy_name='soft-whittle',
                beh_policy_name='soft-whittle',
                transition_probabilities=transition_probs,
                epsilon=0.1,
                reward_estimate=reward_param,
                rewards_true=None
            )

            total_loss = performance
            if hasattr(constraint_method, 'get_penalty'):
                penalty = constraint_method.get_penalty(reward_param.numpy())
                total_loss += penalty
            elif hasattr(constraint_method, 'get_barrier'):
                barrier = constraint_method.get_barrier(reward_param.numpy())
                total_loss += barrier

        # Apply gradients
        grads = tape.gradient(total_loss, reward_param)
        optimizer.apply_gradients(zip([grads], [reward_param]))

        # Apply constraint projection
        current_rewards = reward_param.numpy()
        projected_rewards = constraint_method.project(current_rewards)
        reward_param.assign(projected_rewards)

        # Update lambda for Lagrangian method
        if isinstance(constraint_method, LagrangianMultiplier):
            constraint_method.update_lambda(projected_rewards)

        # Log metrics
        violations = constraint_method.check_violations(projected_rewards)
        history['performance'].append(float(performance.numpy()))
        history['violations'].append(violations)
        history['reward_norm'].append(np.linalg.norm(projected_rewards))

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch}: Perf={performance.numpy():.4f}, "
                  f"Violations={violations}, Norm={history['reward_norm'][-1]:.4f}")

    return reward_param.numpy(), history


def k_fold_experiment(
    n_arms=500,
    n_folds=5,
    constraint_methods=None,
    epochs=50,
    verbose=True
):
    """
    Args:
        n_arms: Number of beneficiaries
        n_folds: Number of folds for cross-validation
        constraint_methods: List of ConstraintMethod instances
        epochs: Training epochs per fold
        verbose: Print progress

    Returns:
        results: Dict containing results for all methods and folds
    """
    from dfl.whittle import newWhittleIndex
    from dfl.trajectory import getSimulatedTrajectories

    if constraint_methods is None:
        constraint_methods = [
            Unconstrained(),
            ProjectedGradient(),
            LagrangianMultiplier(),
            LogBarrier()
        ]

    print(f"\n{'='*80}")
    print(f"STARTING {n_folds}-FOLD CROSS-VALIDATION EXPERIMENT")
    print(f"Arms: {n_arms}, Methods: {[m.name for m in constraint_methods]}")
    print(f"{'='*80}\n")

    # Generate full dataset
    transitions, metadata = generate_real_world_proxy_transitions(n_arms, seed=SEED)

    # Validate data quality
    validation_report = validate_data_quality(transitions, metadata)
    print("Data Quality Report:")
    for key, val in validation_report.items():
        print(f"  {key}: {val}")
    print()

    # risk scores
    risk_scores = np.random.RandomState(SEED).randint(0, 4, size=n_arms)

    # K-fold splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Results storage
    results = {
        'transitions': transitions,
        'metadata': metadata,
        'risk_scores': risk_scores,
        'validation_report': validation_report,
        'folds': []
    }

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_arms))):
        print(f"\n--- FOLD {fold_idx + 1}/{n_folds} ---")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        fold_results = {
            'train_idx': train_idx,
            'test_idx': test_idx,
            'methods': {}
        }

        n_train = len(train_idx)
        train_transitions = transitions[train_idx]

        reward_numpy = np.zeros((n_train, 2))
        reward_numpy[:, 1] = 1
        reward_opt = tf.constant(reward_numpy, dtype=tf.float32)

        w_opt = newWhittleIndex(
            tf.constant(train_transitions, dtype=tf.float32),
            reward_opt
        ).numpy()
        w_opt = np.reshape(w_opt, (n_train, 2))

        _, _, state_traj, action_traj, _ = getSimulatedTrajectories(
            n_benefs=n_train,
            T=11,
            K=int(60 * n_train / n_arms),
            n_trials=1,
            gamma=0.99,
            T_data=train_transitions,
            R_data=reward_numpy,
            w=w_opt,
            replace=False,
            policies=[3],
            fast=True
        )

        train_risk_scores = risk_scores[train_idx]
        edited_action_traj = apply_aggregate_command(
            action_traj, train_risk_scores, K=int(60 * n_train / n_arms)
        )

        for method in constraint_methods:
            print(f"\n  Training with {method.name}...")

            learned_rewards, history = train_whirl_with_constraint(
                transition_probs=train_transitions,
                state_traj=state_traj,
                action_traj=edited_action_traj,
                w_opt=w_opt,
                constraint_method=method,
                epochs=epochs,
                learning_rate=0.01,
                L=10,
                K=int(60 * n_train / n_arms),
                gamma=0.99,
                verbose=verbose
            )

            test_violations = method.check_violations(learned_rewards)

            fold_results['methods'][method.name] = {
                'learned_rewards': learned_rewards,
                'training_history': history,
                'test_violations': test_violations,
                'final_performance': history['performance'][-1]
            }

            print(f"    Final violations: {test_violations}/{n_train}")
            print(f"    Final performance: {history['performance'][-1]:.4f}")

        results['folds'].append(fold_results)

    return results


def apply_aggregate_command(action_traj, risk_scores, K=60, L=10):
    """
    Args:
        action_traj: Original action trajectories
        risk_scores: Risk score for each beneficiary
        K: Budget constraint
        L: Time horizon

    Returns:
        edited_action_traj: Modified action trajectories
    """
    n = len(risk_scores)
    edited_action_traj = np.copy(action_traj)

    for h in range(L):
        # Finding beneficiaries to swap
        called_low_risk = set()
        uncalled_high_risk = set()

        for arm in range(n):
            if edited_action_traj[0, 0, h, arm] == 1 and risk_scores[arm] in {0, 1}:
                called_low_risk.add(arm)
            if edited_action_traj[0, 0, h, arm] == 0 and risk_scores[arm] in {2, 3}:
                uncalled_high_risk.add(arm)

        # Swaping with probability proportional to availability
        if len(uncalled_high_risk) > 0 and len(called_low_risk) > 0:
            prob_switch = min(len(called_low_risk) / len(uncalled_high_risk), 1.0)

            arms = np.arange(n)
            np.random.shuffle(arms)

            for arm in arms:
                if (arm in uncalled_high_risk and
                    len(called_low_risk) > 0 and
                    np.random.uniform() < prob_switch):

                    edited_action_traj[0, 0, h, arm] = 1
                    target = called_low_risk.pop()
                    edited_action_traj[0, 0, h, target] = 0

    return edited_action_traj

# STATISTICAL ANALYSIS & VISUALIZATION

def analyze_results(results):
    """
    Args:
        results: Dict from k_fold_experiment

    Returns:
        analysis: Dict with statistical metrics
    """
    n_folds = len(results['folds'])
    method_names = list(results['folds'][0]['methods'].keys())

    analysis = {
        'summary_stats': {},
        'statistical_tests': {}
    }

    # Aggregate metrics across folds
    for method_name in method_names:
        violations = []
        performances = []

        for fold in results['folds']:
            method_data = fold['methods'][method_name]
            violations.append(method_data['test_violations'])
            performances.append(method_data['final_performance'])

        violations = np.array(violations)
        performances = np.array(performances)

        analysis['summary_stats'][method_name] = {
            'violations_mean': np.mean(violations),
            'violations_std': np.std(violations),
            'violations_sem': stats.sem(violations),
            'performance_mean': np.mean(performances),
            'performance_std': np.std(performances),
            'performance_sem': stats.sem(performances),
            'violations_all': violations,
            'performances_all': performances
        }

    if 'Unconstrained' in method_names:
        baseline_violations = analysis['summary_stats']['Unconstrained']['violations_all']
        baseline_performance = analysis['summary_stats']['Unconstrained']['performances_all']

        for method_name in method_names:
            if method_name == 'Unconstrained':
                continue

            method_violations = analysis['summary_stats'][method_name]['violations_all']
            method_performance = analysis['summary_stats'][method_name]['performances_all']

            # Wilcoxon signed-rank test for violations
            violation_stat, violation_p = stats.wilcoxon(
                baseline_violations, method_violations, alternative='greater'
            )

            # Wilcoxon signed-rank test for performance
            perf_stat, perf_p = stats.wilcoxon(
                baseline_performance, method_performance, alternative='two-sided'
            )

            analysis['statistical_tests'][method_name] = {
                'violation_reduction_p_value': violation_p,
                'violation_statistic': violation_stat,
                'performance_difference_p_value': perf_p,
                'performance_statistic': perf_stat
            }

    return analysis


def plot_comprehensive_results(results, analysis, save_path=None):
    """
    Args:
        results: Dict from k_fold_experiment
        analysis: Dict from analyze_results
        save_path: Optional path to save figure
    """
    method_names = list(analysis['summary_stats'].keys())
    n_methods = len(method_names)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Color palette
    colors = sns.color_palette("husl", n_methods)
    method_colors = {name: colors[i] for i, name in enumerate(method_names)}

    # ---- Row 1: Violations and Performance ----

    # Plot 1: Violation rates across folds
    ax1 = fig.add_subplot(gs[0, 0:2])
    violation_data = []
    for method_name in method_names:
        violations = analysis['summary_stats'][method_name]['violations_all']
        for fold_idx, v in enumerate(violations):
            violation_data.append({
                'Method': method_name,
                'Fold': fold_idx + 1,
                'Violations': v
            })

    df_violations = pd.DataFrame(violation_data)
    sns.boxplot(data=df_violations, x='Method', y='Violations', ax=ax1, palette=method_colors)
    ax1.set_title('Constraint Violations Across Folds', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Violations')
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Performance comparison
    ax2 = fig.add_subplot(gs[0, 2:4])
    performance_data = []
    for method_name in method_names:
        perfs = analysis['summary_stats'][method_name]['performances_all']
        for fold_idx, p in enumerate(perfs):
            performance_data.append({
                'Method': method_name,
                'Fold': fold_idx + 1,
                'Performance': p
            })

    df_performance = pd.DataFrame(performance_data)
    sns.violinplot(data=df_performance, x='Method', y='Performance', ax=ax2, palette=method_colors)
    ax2.set_title('Policy Performance Across Folds', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Performance Score')
    ax2.tick_params(axis='x', rotation=45)

    # ---- Row 2: Training Dynamics ----

    # Plot 3: Training curves (violations over epochs)
    ax3 = fig.add_subplot(gs[1, 0:2])
    for method_name in method_names:
        fold_0_history = results['folds'][0]['methods'][method_name]['training_history']
        ax3.plot(fold_0_history['violations'],
                label=method_name,
                color=method_colors[method_name],
                linewidth=2)
    ax3.set_title('Violation Trajectory During Training (Fold 1)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Number of Violations')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Training curves (performance over epochs)
    ax4 = fig.add_subplot(gs[1, 2:4])
    for method_name in method_names:
        fold_0_history = results['folds'][0]['methods'][method_name]['training_history']
        ax4.plot(fold_0_history['performance'],
                label=method_name,
                color=method_colors[method_name],
                linewidth=2)
    ax4.set_title('Performance Trajectory During Training (Fold 1)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Performance')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # ---- Row 3: Reward Analysis and Statistical Tests ----

    # Plot 5: Learned reward distributions (Fold 1, Projected Gradient)
    ax5 = fig.add_subplot(gs[2, 0:2])
    if 'Projected_Gradient' in method_names:
        pg_rewards = results['folds'][0]['methods']['Projected_Gradient']['learned_rewards']
        ax5.scatter(pg_rewards[:, 0], pg_rewards[:, 1],
                   alpha=0.6, s=20, c='blue', label='Learned Rewards')

        # Add constraint line (y = x)
        lims = [min(pg_rewards.min(), -0.5), max(pg_rewards.max(), 1.5)]
        ax5.plot(lims, lims, 'r--', linewidth=2, label='Constraint Boundary (R₁=R₀)')

        ax5.set_xlabel('R(State=0) - Dropout')
        ax5.set_ylabel('R(State=1) - Adherence')
        ax5.set_title('Reward Space: Projected Gradient (Fold 1)', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)

    # Plot 6: Statistical significance heatmap
    ax6 = fig.add_subplot(gs[2, 2:4])
    if 'statistical_tests' in analysis and len(analysis['statistical_tests']) > 0:
        test_methods = list(analysis['statistical_tests'].keys())
        p_values_violations = [analysis['statistical_tests'][m]['violation_reduction_p_value']
                              for m in test_methods]
        p_values_performance = [analysis['statistical_tests'][m]['performance_difference_p_value']
                               for m in test_methods]

        heatmap_data = np.array([p_values_violations, p_values_performance]).T
        sns.heatmap(heatmap_data,
                   annot=True,
                   fmt='.4f',
                   xticklabels=['Violation\nReduction', 'Performance\nDifference'],
                   yticklabels=test_methods,
                   cmap='RdYlGn_r',
                   vmin=0,
                   vmax=0.1,
                   ax=ax6,
                   cbar_kws={'label': 'p-value'})
        ax6.set_title('Statistical Tests vs Unconstrained (p-values)',
                     fontsize=14, fontweight='bold')

    plt.suptitle('Constitutional WHIRL: Comprehensive Experimental Results',
                fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    return fig


def print_summary_table(analysis):
    """
    Args:
        analysis: Dict from analyze_results
    """
    print("\n" + "="*80)
    print("SUMMARY TABLE: K-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)

    # Create summary dataframe
    summary_data = []
    for method_name, stats_dict in analysis['summary_stats'].items():
        summary_data.append({
            'Method': method_name,
            'Violations (Mean ± SEM)': f"{stats_dict['violations_mean']:.2f} ± {stats_dict['violations_sem']:.2f}",
            'Violations (%)': f"{(stats_dict['violations_mean']/500)*100:.1f}%",
            'Performance (Mean ± SEM)': f"{stats_dict['performance_mean']:.4f} ± {stats_dict['performance_sem']:.4f}",
            'Violation Reduction': 'Baseline' if method_name == 'Unconstrained' else
                                   f"{((analysis['summary_stats']['Unconstrained']['violations_mean'] - stats_dict['violations_mean'])/analysis['summary_stats']['Unconstrained']['violations_mean']*100):.1f}%"
        })

    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))

    # Statistical significance table
    if 'statistical_tests' in analysis and len(analysis['statistical_tests']) > 0:
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS (vs Unconstrained Baseline)")
        print("="*80)

        sig_data = []
        for method_name, test_results in analysis['statistical_tests'].items():
            violation_p = test_results['violation_reduction_p_value']
            perf_p = test_results['performance_difference_p_value']

            sig_data.append({
                'Method': method_name,
                'Violation Reduction': 'p < 0.001 ***' if violation_p < 0.001 else
                                      f'p < 0.01 **' if violation_p < 0.01 else
                                      f'p < 0.05 *' if violation_p < 0.05 else
                                      f'p = {violation_p:.4f} (n.s.)',
                'Performance Difference': 'p < 0.001 ***' if perf_p < 0.001 else
                                         f'p < 0.01 **' if perf_p < 0.01 else
                                         f'p < 0.05 *' if perf_p < 0.05 else
                                         f'p = {perf_p:.4f} (n.s.)',
                'Interpretation': '✓ Significantly safer' if violation_p < 0.05 else '✗ Not significant'
            })

        df_sig = pd.DataFrame(sig_data)
        print("\n" + df_sig.to_string(index=False))
        print("\nNote: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")

    print("\n" + "="*80)


def generate_latex_table(analysis, caption="K-Fold Cross-Validation Results"):
    """
    Args:
        analysis: Dict from analyze_results
        caption: Table caption

    Returns:
        latex_str: LaTeX table code
    """
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += "\\label{tab:results}\n"
    latex_str += "\\begin{tabular}{lccc}\n"
    latex_str += "\\toprule\n"
    latex_str += "Method & Violations (\\%) & Performance & Significance \\\\\n"
    latex_str += "\\midrule\n"

    for method_name, stats_dict in analysis['summary_stats'].items():
        violations_pct = (stats_dict['violations_mean']/500)*100
        perf_mean = stats_dict['performance_mean']
        perf_sem = stats_dict['performance_sem']

        # Significance marker
        sig_marker = ""
        if method_name in analysis.get('statistical_tests', {}):
            p_val = analysis['statistical_tests'][method_name]['violation_reduction_p_value']
            if p_val < 0.001:
                sig_marker = "$^{***}$"
            elif p_val < 0.01:
                sig_marker = "$^{**}$"
            elif p_val < 0.05:
                sig_marker = "$^{*}$"

        latex_str += f"{method_name.replace('_', ' ')} & {violations_pct:.1f}\\% & "
        latex_str += f"{perf_mean:.4f} $\\pm$ {perf_sem:.4f} & {sig_marker} \\\\\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    return latex_str


def plot_reward_landscape_comparison(results, fold_idx=0):
    """
    Args:
        results: Dict from k_fold_experiment
        fold_idx: Which fold to visualize
    """
    fold = results['folds'][fold_idx]
    method_names = list(fold['methods'].keys())
    n_methods = len(method_names)

    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for idx, method_name in enumerate(method_names):
        ax = axes[idx]
        rewards = fold['methods'][method_name]['learned_rewards']

        # Scatter plot
        ax.scatter(rewards[:, 0], rewards[:, 1],
                  alpha=0.5, s=15, c='steelblue', edgecolors='none')

        # Constraint boundary
        lims = [min(rewards.min(), -0.5), max(rewards.max(), 1.5)]
        ax.plot(lims, lims, 'r--', linewidth=2, label='R₁ = R₀', alpha=0.7)

        # Shade feasible region
        ax.fill_between(lims, lims, lims[1], alpha=0.1, color='green', label='Feasible')
        ax.fill_between(lims, lims[0], lims, alpha=0.1, color='red', label='Infeasible')

        # Count violations
        violations = np.sum(rewards[:, 1] < rewards[:, 0])

        ax.set_xlabel('R(Dropout)', fontsize=11)
        ax.set_ylabel('R(Adherence)', fontsize=11)
        ax.set_title(f'{method_name.replace("_", " ")}\n({violations} violations)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle(f'Learned Reward Landscapes (Fold {fold_idx+1})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig


def export_results_to_csv(results, analysis, filename='constitutional_whirl_results.csv'):
    """
    Args:
        results: Dict from k_fold_experiment
        analysis: Dict from analyze_results
        filename: Output filename
    """
    rows = []

    for fold_idx, fold in enumerate(results['folds']):
        for method_name, method_data in fold['methods'].items():
            rows.append({
                'fold': fold_idx + 1,
                'method': method_name,
                'violations': method_data['test_violations'],
                'performance': method_data['final_performance'],
                'violations_pct': (method_data['test_violations'] / len(fold['train_idx'])) * 100
            })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults exported to {filename}")

    return df

# MAIN EXECUTION WITH ALL ABLATIONS

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONSTITUTIONAL WHIRL: COMPLETE EXPERIMENTAL FRAMEWORK")
    print("="*80)

    # EXPERIMENT 1: The Core Question - Aggregate Command Impact
    print("\n\n### EXPERIMENT 1: AGGREGATE COMMAND ABLATION ###")
    print("Question: Is the violation rate caused by IRL or the aggregate command?\n")

    transitions, metadata = generate_real_world_proxy_transitions(500, seed=SEED)
    risk_scores = np.random.RandomState(SEED).randint(0, 4, size=500)

    ablation_results = train_baseline_comparison(
        transitions, risk_scores, n_arms=500, n_folds=5,
        test_aggregate=True, constraint_method=None
    )

    print("\n" + "="*80)
    print("ABLATION RESULTS: WITH vs WITHOUT Aggregate Command")
    print("="*80)
    print(f"WITHOUT Command: {ablation_results['without_command_mean']:.1f}% ± {ablation_results['without_command_std']:.1f}%")
    print(f"WITH Command:    {ablation_results['with_command_mean']:.1f}% ± {ablation_results['with_command_std']:.1f}%")

    # Statistical test
    if ablation_results['all_with']:
        stat, p_val = stats.wilcoxon(
            ablation_results['all_without'],
            ablation_results['all_with'],
            alternative='less'
        )
        print(f"\nWilcoxon test: p = {p_val:.4f}")
        if p_val < 0.05:
            print("✓ Aggregate command SIGNIFICANTLY increases violations")
        else:
            print("✗ No significant difference - problem is inherent to IRL")

    # EXPERIMENT 2: Beta Parameter Sensitivity
    print("\n\n### EXPERIMENT 2: BETA PARAMETER SENSITIVITY ###")
    print("Question: Are results robust across different population distributions?\n")

    sensitivity_results = beta_sensitivity_experiment(
        n_arms=500, n_folds=3, epochs=30
    )

    # EXPERIMENT 3: Full K-Fold with All Constraint Methods
    print("\n\n### EXPERIMENT 3: FULL K-FOLD EVALUATION ###")
    print("Question: Which constraint method works best?\n")

    constraint_methods = [
        Unconstrained(),
        ProjectedGradient(epsilon=0.001),
        LagrangianMultiplier(lambda_init=1.0, lambda_lr=0.01),
        LogBarrier(mu=0.1)
    ]

    full_results = k_fold_experiment(
        n_arms=500, n_folds=5, constraint_methods=constraint_methods,
        epochs=50, verbose=True
    )

    analysis = analyze_results(full_results)

    # EXPERIMENT 4: Ground Truth Correlation
    print("\n\n### EXPERIMENT 4: REWARD QUALITY ANALYSIS ###")
    print("Question: Do learned rewards correlate with true adherence propensity?\n")

    correlations = analyze_reward_quality(full_results, metadata)

    # All Outputs
    print("\n\n### GENERATING OUTPUTS ###")

    print_summary_table(analysis)
    plot_comprehensive_results(full_results, analysis,
                               save_path='constitutional_whirl_full.png')
    plot_reward_landscape_comparison(full_results, fold_idx=0)
    export_results_to_csv(full_results, analysis)

    latex = generate_latex_table(analysis)
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print(latex)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n### Key Findings:")
    print(f"1. Baseline IRL violation rate: {ablation_results['with_command_mean']:.1f}%")
    print(f"   - WITHOUT aggregate command: {ablation_results['without_command_mean']:.1f}%")
    print(f"   - Aggregate command {'significantly increases' if p_val < 0.05 else 'does not affect'} violations")

    print(f"\n2. Projected Gradient method:")
    pgd_stats = analysis['summary_stats']['Projected_Gradient']
    print(f"   - Reduces violations to {pgd_stats['violations_mean']:.1f}%")
    print(f"   - Performance: {pgd_stats['performance_mean']:.2f} ± {pgd_stats['performance_sem']:.2f}")

    print(f"\n3. Reward quality:")
    pgd_corr = np.mean(correlations.get('Projected_Gradient', [0]))
    print(f"   - Correlation with true grit: {pgd_corr:.3f}")
    print(f"   - Rewards {'are' if pgd_corr > 0.3 else 'may not be'} meaningful")

    print("\n### Limitations:")
    print("   - Synthetic data (Beta distributions)")
    print("   - N=5 folds limits statistical power")
    print("   - Aggregate command may introduce artifacts")

    print("\n### Next Steps:")
    print("   - Validate on real maternal health data")
    print("   - Test with N=10+ folds or multiple seeds")
    print("   - Extend to multi-state (n_states > 2)")
    print("   - Compare with Safe RL baselines (CPO, RCPO)")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
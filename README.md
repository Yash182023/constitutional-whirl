# Constitutional WHIRL: Safety Constraints for Adherence Bandits

This repository contains the code and experimental results for a research study on **Safe Inverse Reinforcement Learning (IRL)** in public health. This work extends the **WHIRL framework** (Jain et al., NeurIPS 2024).

## The Problem
Unconstrained IRL algorithms applied to heterogeneous population data often learn **inverted reward functions** ($R_{sick} > R_{healthy}$) to minimize trajectory error. In our experiments on proxy data, this "unethical" reward structure occurred in **40.9%** of cases.

## The Solution
This project implements a **Constitutional WHIRL** framework that enforces monotonicity constraints ($R_{healthy} \ge R_{sick}$) during the policy gradient update loop. We compared three methods:
1.  **Projected Gradient Descent (PGD)**
2.  **Lagrangian Relaxation**
3.  **Log-Barrier Optimization**

## Key Results (5-Fold Cross-Validation)
| Method | Safety Violations | Policy Performance |
|--------|-------------------|--------------------|
| Unconstrained (Baseline) | 40.9% | 906 ± 12 |
| **Projected Gradient (Ours)** | **0.0%** | **953 ± 10** |
| Lagrangian | 40.8% | 942 ± 11 |

**Key Finding:** Projected Gradient constraints eliminated safety violations (0%) without compromising intervention efficacy.


## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full experimental suite (Ablation + 5-Fold CV):
   ```bash
   python run_experiments.py
   ```

## Repository Structure
*   `run_experiments.py`: The main script containing the K-Fold experimental framework, data generation, and constraint classes.
*   `dfl/`: Core logic for Whittle Index computation and Policy Evaluation (based on original WHIRL codebase).
*   `results/`: Generated plots and CSV logs from the experiments.

## References
*   **Original Paper:** Jain, G., et al. (2024). *WHIRL: Whittle Index Reinforcement Learning*. NeurIPS.
*   **Foundational Work:** Killian, T., et al. (2023). *Adherence Bandits*. AAAI.

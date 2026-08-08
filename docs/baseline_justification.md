# Baseline Justification

## Included In The P302 Raw-Data Suite

`Rule-based heuristic` exists to answer whether RL is better than a simple, explainable EMS. It is not claimed to be optimal; it is the engineering common-sense reference point.

`SAC` exists because SAC is a standard off-policy continuous-control baseline and is closest to the current actor-critic implementation. It answers how dangerous and useful a pure RL policy is without a learned or external safety shield.

`SAC + reward safety penalty` exists because the most direct criticism is that unsafe behavior could be solved by writing the violation into the reward. This baseline uses the same SAC stack without projection, but penalizes attempted safety-bound violations during training.

`SAC + SafetyNet during training` exists as a shielded RL baseline. It tests whether projection alone is enough, without giving the method OCC or the full curriculum/adaptive machinery.

`SAC + SafetyNet + OCC` exists as an ablation of the proposed method. It isolates the effect of the opportunity-cost critic that penalizes raw actions which need correction.

`OURS: SAC + SafetyNet + OCC + curriculum/adaptive settings` is the main method. It must be reported in both raw-policy and deployment-aware layers so the claim is not reduced to "SafetyNet fixed everything."

## Included As Literature Or StressM Baselines

`PPO-Lagrangian` is a constrained RL / CMDP baseline. It is meaningful for the safe-RL claim, especially in StressM experiments where the older PPO-Lagrangian implementation is already available.

`PPO + SafetyNet` checks whether an on-policy learner with the same external shield can reach similar behavior. If it is weaker, the explanation is data efficiency and continuous microgrid control, not that PPO is an invalid method.

## Excluded From The Main P302 Table

`TD3` is optional. It is a reasonable continuous off-policy baseline, but SAC already covers the stronger maximum-entropy actor-critic family used in this repository.

`DDPG` is excluded because it is older and less stable; after SAC/TD3/PPO-style comparisons, its information gain is low.

`DQN` is excluded because P302 power dispatch is continuous. Discretizing actions would introduce a separate action-grid design that can dominate the result and make the comparison less fair.

## Fairness Rule

All P302 baselines use the same dataset, SoC bounds, battery limits, action semantics, episode length, observation space, and rollout dates. The only allowed differences are the learning algorithm and the safety mechanism being tested.

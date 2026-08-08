# Planned Safety Experiments And Evaluation Settings

This document records the additional experiments and evaluation settings discussed on 2026-06-18. The purpose is to make the next thesis/simulation work explicit before changing code or retraining models.

Related rule record:

- [Greedy Heuristic Baselines](greedy_heuristic_baselines.md)
- [Human-Knowledge Heuristic Rule](human_knowledge_heuristic_rule.md)

## Motivation

The current results compare profit and violation counts, but several issues remain:

- A conservative heuristic can look strong because the economic opportunity is small.
- RL can achieve higher reward while producing unsafe or infeasible actions.
- Current SoC violation accounting clips SoC back to the operational range, which hides how long a policy remains outside the safe operating region.
- The thesis claim should not be "RL always beats heuristic." A more defensible claim is that CORAL improves deployability by reducing unsafe attempts, realized violations, and safety-layer dependence while approaching a strong expert baseline.

## Evaluation Philosophy

Do not evaluate RL correctness only by cumulative reward or profit.

A deployable controller should be evaluated by:

- economic performance;
- operational safety;
- infeasible action attempts;
- reliance on runtime safety layers;
- recovery behavior after entering unsafe states;
- degradation under deployment-oriented stress tests.

## Confirmed Thesis Evaluation Layout

The thesis should keep method design, result analysis, and reproducibility details separated.

- Chapter 3 defines the experiment logic: baseline definitions, ablation hypotheses, fairness constraints, and metric definitions.
- Chapter 4 reports the actual scenarios, held-out data, metric tables, convergence or behavior analysis, and interpretation.
- Appendix B records concrete hyperparameters, greedy thresholds, deployment command settings, and flow-rate assumptions.

For all future baseline / ablation runs, keep the following result layers separate:

- `raw actor policy`: the unshielded model output before SafetyNet / CORAL projection.
- `shielded deployed policy`: the action after safety correction and command feasibility checks.
- `training`: data and metrics observed during learning.
- `evaluation`: validation rollouts used for model selection or sanity checks.
- `held-out`: final comparison intervals not used for training or tuning.

Main result tables should not mix training, evaluation, and held-out numbers. If a method lacks a metric from the same held-out protocol, leave the thesis table cell as TODO rather than copying a number from another run.

Current Chapter 4 priority:

- Use the main baseline / ablation table as the central quantitative evidence.
- Report total cost or net return, realized violations, raw unsafe attempts, safety interventions, and SoH or capacity-related indicators only when the output is available and consistently defined.
- Treat flow/no-flow comparison as preliminary unless both settings are evaluated under the same protocol.
- Keep old single-day or early multi-day behavior plots as pre-deployment behavior checks / historical comparison only, not as the main proof of thesis readiness.

## Boundary Definitions

Use two different SoC boundary concepts.

## Physical Hard Boundary

```text
0% <= SoC <= 100%
```

The battery state cannot physically go below 0% or above 100%. Simulation may clamp only to this physical range.

## Operational Safety Boundary

```text
20% <= SoC <= 80%
```

This is the safe operating window used for deployment and thesis evaluation. Leaving this range is an operational violation even if the physical state remains possible.

## Proposed SoC Accounting Change

For evaluation, avoid immediately clipping SoC back to the 20%-80% operational range.

Instead:

1. Apply the action and update SoC continuously.
2. Physically clamp only to 0%-100%.
3. Count whether the resulting SoC is outside the operational range.
4. Keep counting each step that remains outside the safe range.

This makes the model responsible not only for crossing the boundary, but also for how long it remains unsafe.

## New Or Refined Safety Metrics

## 1. Boundary Crossing Violation

Counts the moment a policy pushes the system from inside the operational range to outside it.

Example:

```text
79% -> 81%
```

This counts as a crossing violation.

## 2. Unsafe-State Duration

Counts every step in which SoC remains outside the operational range.

Example:

```text
81.0%, 80.8%, 80.5%
```

Each step counts toward unsafe duration.

This is important because a high-SoC violation can persist when PV is active and discharge is not allowed.

## 3. Recovery Behavior

Tracks whether and how quickly a controller returns to the operational safe range.

Useful metrics:

- `recovery_steps`: number of steps needed to return to 20%-80%.
- `max_soc_overshoot`: maximum amount above 80%.
- `max_soc_undershoot`: maximum amount below 20%.
- `unsafe_area`: sum of distance outside the safe range over time.

Example:

```text
unsafe_area += max(0, soc - 0.80) + max(0, 0.20 - soc)
```

## 4. Invalid Discharge Condition

Counts discharge attempts that are inconsistent with platform supply logic.

Discharge is invalid if the policy requests battery discharge while:

- PV is actively supporting the load;
- no effective load is present;
- load exceeds the battery's standalone discharge capability;
- SoC or voltage/health lock makes discharge infeasible.

This is the most platform-specific violation and should be emphasized more than voltage/frequency violations that are not modeled in this EMS layer.

## 5. Unsafe Attempt / Safety Intervention

Keep the distinction among:

- `attempted violation`: raw action would become unsafe;
- `realized violation`: system state actually enters unsafe operational region;
- `safety intervention`: SafetyNet/CORAL modifies an action;
- `projection magnitude`: how much the safety layer changed the action.

This allows CORAL to show value even when realized violations are low.

## Experiment A: Greedy Heuristic Baselines

Implement three compact rule-based baselines instead of one complete expert controller:

- safety-first greedy;
- profit-first greedy;
- balanced safety-profit greedy.

All three should keep shared hardware guards:

- no discharge while PV support is active;
- no battery partial-assist behavior under `solo_only` semantics;
- no discharge when load exceeds standalone battery capability;
- one-step SoC prediction for charge/discharge room;
- voltage cutoff and SoC reserve checks.

Specification:

- [Greedy Heuristic Baselines](greedy_heuristic_baselines.md)

Expected interpretation:

- If profit-first earns more but causes more unsafe attempts or blocked actions, that is a useful safety-profit trade-off result.
- If safety-first is very safe but inactive, that is also useful context.
- RL should be judged by whether it approaches the useful economic behavior while reducing manual rule dependence and unsafe attempts.
- A high `pv_support_ratio` should not be treated as unlimited PV surplus. If no reliable surplus estimate is available, PV-based charging should remain conservative.

## Experiment B: No 20%-80% Clip Evaluation

Create an evaluation mode where:

- operational safe range remains 20%-80%;
- physical SoC range is 0%-100%;
- SoC is not clipped back to 20%-80% after each violation;
- unsafe duration and recovery metrics are reported.

This should be used for evaluation first. Training can still keep protective mechanisms if needed.

Key comparison:

- raw SAC may leave the safe range and remain unsafe;
- heuristic should avoid crossings if one-step prediction is correct;
- SafetyNet/CORAL should reduce boundary crossing and unsafe duration.

## Experiment C: Penalty Escalation Protocol

Instead of fixing one arbitrary violation penalty, evaluate models under progressively stricter safety penalties.

Suggested levels:

```text
Level 0: no safety penalty
Level 1: low penalty
Level 2: medium penalty
Level 3: high penalty
...
```

For each method, report:

- minimum penalty level required to reach near-zero realized violations;
- profit at the first safe level;
- profit drop from Level 0 to the safe level;
- remaining unsafe attempts;
- SafetyNet/CORAL intervention count;
- unsafe-state duration.

Purpose:

> Compare how much economic performance each method must sacrifice to become deployment-feasible.

This is more defensible than claiming that one fixed reward weight defines the correct safety-profit trade-off.

## Experiment D: Deployment-Oriented Stress Tests

Stress tests should not assume RL will always beat heuristic. They should measure how each controller degrades.

Candidate stressors:

- different initial SoC values;
- consecutive low-PV days;
- high-PV days with high SoC risk;
- small measured load versus old load-scale assumptions;
- load changes;
- SoC observation noise or offset;
- delayed SoC observation;
- PV support ratio threshold perturbation;
- voltage cutoff proxy activation.

Report:

- profit drop;
- violation increase;
- unsafe duration;
- invalid discharge attempts;
- intervention count;
- whether the controller becomes overly inactive.

## Experiment E: Export / Sell-Back Simulation

Keep this as a simulation-only extension unless the platform supports real sell-back.

Current thesis priority decision from 2026-06-22:

- Do not start the next thesis evidence push from export / sell-back.
- First lengthen validation and evaluation windows in `thesis_sim` and check whether safety, profit, safety/profit/balanced greedy baselines, and RL/CORAL results separate more clearly over longer horizons.
- Only if longer validation still cannot produce a clear and defensible safety-profit comparison should export / sell-back be introduced as a later simulation feature.

Current preliminary result:

- export-week training is runnable;
- CORAL improves over SAC+SafetyNet in profit under one tested setting;
- it still does not clearly beat the rule heuristic;
- flow-free behavior can create many low-flow events.

Recommendation:

- Do not use current export results as final thesis evidence.
- Keep export as optional appendix/future experiment unless additional tuning produces a clear and defensible story.

## Experiment F: Flow-Control Feasibility

Flow control should be treated as a hardware-aware extension, not the main profit claim.

Deployment command semantics:

- Standby is not zero-flow anymore.
- When battery power is zero, command should keep the battery ID and set standby flow to 50%.
- Expected standby command:

```text
01,0,50,
```

- Do not use `00,0,50,` unless the vendor confirms that `PP=00` still controls pump flow.
- Active charge/discharge should use at least 60% flow.

Rationale:

- The liquid-flow battery voltage is not reliable when electrolyte flow is stopped.
- With 0% flow, the measured voltage can drift down toward around 2.3 V even when this does not represent the true operating voltage.
- Maintaining 50% standby flow is a deployment measurement requirement, not an RL economic action.

Report flow-specific indicators separately:

- standby flow command should be 50% for voltage observation;
- active flow should be at least 60%;
- flow-power mismatch count;
- pump energy;
- flow-limited action count.

Do not overclaim flow-control economic superiority unless a stable comparison supports it.

## Recommended Priority

Highest priority:

1. Use safety-first, profit-first, and balanced safety-profit greedy as the main heuristic baseline family.
2. Lengthen `thesis_sim` validation/evaluation windows before changing the market model.
3. Implement no-20/80-clip evaluation mode.
4. Add unsafe duration and recovery metrics.
5. Re-run the main baseline table with the new safety accounting and longer-horizon validation.

Second priority:

6. Penalty escalation protocol.
7. Deployment stress tests.

Optional:

8. Export/sell-back simulation, only if longer validation still cannot separate the story.
9. Expanded flow-control comparison.

## Thesis Framing

Suggested claim:

> This thesis does not validate RL solely through accumulated reward. Instead, it evaluates whether learned policies remain deployable under platform-specific feasibility constraints, including SoC operating boundaries, invalid discharge conditions, unsafe action attempts, safety-layer interventions, and recovery behavior after unsafe states.

Suggested heuristic framing:

> The rule-based baseline is intentionally strong and knowledge-based. It encodes one-step SoC prediction and platform-specific feasibility rules. Therefore, matching or approaching its performance while reducing manual rule dependence and unsafe attempts is meaningful evidence of RL deployability.


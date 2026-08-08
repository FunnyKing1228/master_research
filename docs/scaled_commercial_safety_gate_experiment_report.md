# Scaled Commercial Microgrid Strict-Safety Experiment Report

## 1. Experiment Purpose

This experiment was designed to move beyond the original P302-scale setting, where absolute power and energy values are very small and profit differences between controllers are easily compressed. The goal is not to claim that the physical P302 platform itself is a commercial microgrid. Instead, the experiment defines a physically motivated commercial-scale design-space scenario using the same measured PV/load temporal structure.

The central research question is:

> Under a strict battery safety gate, can learned EMS controllers preserve economic benefit compared with expert/greedy baselines?

The final interpretation is deliberately safety-first:

1. Raw learned policies are useful for diagnosing profit-safety trade-offs.
2. Strict safety metrics are required before profit comparison is meaningful.
3. A common deployment safety layer should be treated as part of the evaluation environment, not as a method-specific advantage.
4. Common-margin fairness and per-method certified-margin fairness answer different questions and should both be reported.
5. Under that shared safety layer, learned controllers satisfy zero strict SoC violation and outperform zero-violation greedy baselines.

## 2. Source Dataset and Scale-Up Method

### 2.1 Source Dataset

The scaled commercial dataset is generated from:

```text
data/processed/thesis_coral_stage1_training_clean_windows.csv
```

The output dataset and metadata are:

```text
data/processed/thesis_scaled_commercial_60kw_clean_windows.csv
data/processed/thesis_scaled_commercial_60kw_clean_windows_meta.json
```

The source contains measured clean-window PV/load time series from the earlier thesis experiment. The important design choice is that the temporal pattern and PV/load ratio are preserved.

### 2.2 Scaling Rule

The target peak load is set to `60 kW`.

Let:

- `L_source_peak` be the original peak load.
- `L_target_peak = 60 kW`.
- `s = L_target_peak / L_source_peak`.

From the metadata:

```text
L_source_peak = 0.009423 kW
s = 6367.398917542184
```

Both `Consumption` and `Solar` are multiplied by the same scale factor:

```text
Consumption_scaled = Consumption_source × s
Solar_scaled       = Solar_source × s
```

Because PV and load are scaled by the same factor, every timestamp preserves the measured PV/load ratio:

```text
Solar_scaled / Consumption_scaled = Solar_source / Consumption_source
```

This means the experiment does not invent a new solar profile. It preserves the measured ratio and timing while moving the magnitude into a commercial-scale range.

### 2.3 Dataset Statistics

| Quantity | Source P302 clean-window | Scaled commercial |
|---|---:|---:|
| Rows | 1556 | 1556 |
| Peak load | 0.009423 kW | 60.000 kW |
| Peak PV | 0.012612 kW | 80.308 kW |
| Mean load | 0.005551 kW | 35.349 kW |
| Mean PV | 0.002179 kW | 13.875 kW |
| Peak PV/load ratio | 1.338 | 1.338 |
| Energy PV/load ratio | 0.393 | 0.393 |

The scaled PV peak of `80.308 kW` is not independently chosen. It is the result of preserving the PV/load ratio after setting peak load to `60 kW`.

### 2.4 PV Boolean / PV Sufficiency

If a `PV_bool` column exists, it is recomputed after scaling:

```text
PV_bool = 1 if Solar / Consumption >= 0.8 else 0
```

This matches the experiment's PV sufficiency threshold. The thesis wording should avoid binary source attribution such as "fully solar supplied" unless directly measurable. Preferred wording is:

- "PV support increases"
- "grid demand decreases"
- "PV support ratio is high"

## 3. Commercial Microgrid Component Specification

The main scenario is a small-commercial 60 kW peak-load microgrid.

### 3.1 Load and PV

| Component | Specification | Rationale |
|---|---:|---|
| Peak load | 60 kW | Small-commercial / light-commercial scale. |
| Mean load | 35.349 kW | Result of scaled measured clean-window data. |
| Peak PV | 80.308 kW | Result of preserving measured PV/load ratio. |
| Mean PV | 13.875 kW | Result of scaled measured clean-window data. |
| PV/load energy ratio | 0.393 | Preserved from source data. |

### 3.2 Battery Energy Storage System

| Parameter | Value |
|---|---:|
| Battery energy capacity | 240 kWh |
| Battery charge power | 60 kW |
| Battery discharge power | 60 kW |
| Duration at rated power | 4 h |
| Battery efficiency | 0.90 |
| True SoC safety bounds | 20-80% |
| SoC clipping after violation | disabled |

The `240 kWh / 60 kW` sizing corresponds to a 4-hour battery, which is common in commercial and industrial storage contexts for TOU arbitrage, peak shaving, and short-duration resilience. The true safety bounds were corrected to `20-80%`. This is important: strict violation is evaluated against `20-80%`, not against the internal SafetyNet projection range.

### 3.3 Flow Battery / Pump-Related Parameters

The current main results use the no-flow action setting, but the commercial scenario retains flow-related parameters for later flow-control experiments.

| Parameter | Value |
|---|---:|
| Pump max power | 4.8 kW |
| Pump max power fraction | 8% of 60 kW battery power |
| Pump power curve | `P_pump(Q) = P_max × Q^3` |
| `flow_R_base_ohm` | 10.0 |
| `flow_k_R` | 0.5 |
| `flow_V_OCV_charge` | 8.5 |
| `flow_V_OCV_discharge` | 5.6 |
| `flow_min_active_fraction` | 0.15 |
| `flow_power_min_fraction` | 0.15 |

The pump max power follows the lower end of a common flow-battery auxiliary load range (`8-15%` of rated system power). In this no-flow experiment, the flow action is disabled:

```text
use_flow_rate_action: false
```

Flow-control should therefore be treated as a later extension, not as part of the main thesis result unless it is rerun under the same strict safety protocol.

### 3.4 Hardware-Aligned Discharge Rule

The simulation avoids treating the battery as a third partial-assist source.

The main discharge configuration is:

```text
discharge_auto: true
discharge_mode: solo_only
enforce_solo_discharge_load_limit: true
```

This means:

- PV and grid may co-support load.
- Battery discharge is not treated as parallel partial support with PV/grid.
- If battery discharge is valid, it must effectively be capable of serving the load according to the simulation's solo-discharge logic.

This is aligned with prior project guardrails and avoids overclaiming binary PV-vs-grid switching.

## 4. Environment and Safety Definitions

### 4.1 Episode and Rollout Setup

| Parameter | Value |
|---|---:|
| Time step | 0.25 h |
| Steps per day | 96 |
| Training episodes | typically 1000 |
| Fair rollout days | 16 |
| Evaluation window | same rollout days for all compared policies |

The fair rollout comparison uses the same 16 scaled commercial rollout days for all policies. This is important because comparing training episode averages, held-out episodes, and heuristic rollouts directly can be misleading.

### 4.2 Strict SoC Accounting

Earlier versions of the simulator counted boundary hits and clipped SoC back to the allowed range. This was not strict enough for deployment safety because a controller could push SoC outside the boundary without remaining outside in the simulated state.

The strict version disables post-step clipping:

```text
clip_soc_to_bounds: false
```

If the controller pushes SoC beyond the true `20-80%` bounds, SoC remains out of bounds until later actions bring it back. Every out-of-bound step contributes to strict violation metrics.

### 4.3 Safety Metrics

The main safety metrics are:

| Metric | Meaning |
|---|---|
| `strict_soc_violation_steps` | Number of time steps where SoC is outside 20-80%; summary CSV stores per-day mean, while report tables use total `x/1536` steps. |
| `strict_soc_violation_hours` | Time duration outside 20-80%. |
| `strict_soc_violation_kwh` | Integrated out-of-bound energy magnitude. |
| `strict_soc_violation_max_kwh` | Worst single-step out-of-bound excess. |
| `violations_realized` | Legacy boundary-hit / event-style count. |
| `violations_attempted` | Raw policy action would cross bounds before projection. |

For thesis safety claims, the strict metrics should be primary. Legacy realized violations can be retained as a secondary diagnostic.

The strict safety gate is:

```text
strict_soc_violation_steps = 0
strict_soc_violation_hours = 0
strict_soc_violation_kwh = 0
```

Profit ranking is meaningful only after applying this gate.

## 5. Controllers and Baselines

### 5.1 Heuristic Baselines

The heuristic baselines include:

- Safety-first greedy
- Balanced greedy
- Profit-first greedy

These are hand-coded policies and serve as expert-style reference controllers. In the 20/80 strict evaluation, they satisfy zero strict violation.

### 5.2 Learned Baselines

The learned controllers include:

- SAC
- SAC + reward safety penalty
- SAC + SafetyNet projection
- SAC + SafetyNet + OCC
- CORAL
- PPO
- PPO + SafetyNet

### 5.3 OCC Interpretation

OCC means Opportunity Cost Critic. In this experiment its role is to make the learned controller aware of boundary position and boundary risk. It is not merely an added penalty; it provides an auxiliary learning signal that helps the policy internalize the future opportunity cost of consuming SoC safety margin.

A good thesis interpretation is:

> OCC helps the controller understand that approaching the SoC boundary reduces future operational flexibility. Even when it does not eliminate all raw violations, it tends to reduce violation magnitude and stabilize profit.

### 5.4 SafetyNet and Common Deployment Safety Layer

SafetyNet projection modifies the action before deployment to keep the controller away from unsafe SoC regions. A key fairness concern is that this should not be applied only to the proposed method.

Therefore, the final fair comparison treats SafetyNet margin as an environment-level deployment safety layer:

```text
true SoC bounds: 20-80%
common deployment margin: 0.04
internal projection bounds: 24-76%
```

All learned controllers are evaluated under the same deployment safety layer. Method names do not repeat "common SafetyNet" in the final main table because it is part of the shared environment.

## 6. Three-Layer Evaluation Protocol

The final experiment is best understood as three layers.

### 6.0 Why Perform a Margin Sweep

The `soc_margin` sweep is not arbitrary hyperparameter tuning and does not change the true safety boundary. The true SoC safety bounds remain `20-80%`, and strict violations are always evaluated against `20-80%`. The margin only tightens the internal projection bounds used by the deployment safety layer:

```text
true bounds: 20-80%
soc_margin = 0.04
projection bounds: 24-76%
```

In other words, the margin controls the conservativeness of the safety filter/shield. The question is:

> How much conservative deployment buffer is required for each policy to pass the zero strict-violation gate?

This is aligned with several safe RL evaluation ideas:

- Safety layers and action projection modify candidate policy actions before execution.
- Runtime shielding monitors and corrects actions that would violate safety specifications.
- Constrained/safe RL studies often analyze cost limits, constraint thresholds, penalty multipliers, or safety-filter conservativeness to reveal reward-safety trade-offs.

Therefore, the margin sweep is best described as a certifiability / robustness sensitivity analysis. Its purpose is to estimate the smallest tested conservative buffer under which a policy can be considered strict-safe.

The relationship does not have to be monotonic. If the margin is too large, the internal feasible set becomes narrow, the projection layer may over-intervene, and the closed-loop trajectory can shift away from the states seen during training. In the flow-rate setting, pump losses and flow-dependent power limits further interact with projected battery actions. This explains why an overly conservative margin can reduce recovery capability and cause violations to reappear.

### 6.1 Layer 1: Raw Policy Diagnostics

Question:

> Before adding a common deployment safety layer, how safe is each learned controller by itself?

This layer is diagnostic only. It should not be used as the headline profit comparison because learned policies and heuristics do not share the same safety assumptions.

Raw 20/80 strict results:

| Method | Net profit | Violation steps | Strict hours | Strict kWh | Gate |
|---|---:|---:|---:|---:|---|
| SAC | -1534.420 | 938/1536 | 14.656 | 189.585 | Fail |
| SAC + reward safety penalty | -1599.707 | 655/1536 | 10.234 | 119.039 | Fail |
| SAC+SN | -1544.922 | 227/1536 | 3.547 | 58.617 | Fail |
| SAC+SN+OCC | -1532.215 | 476/1536 | 7.438 | 2.436 | Fail |
| CORAL | -1520.490 | 345/1536 | 5.391 | 3.013 | Fail |
| PPO | -1616.295 | 716/1536 | 11.188 | 195.548 | Fail |
| PPO+SN | -1595.227 | 304/1536 | 4.750 | 2.102 | Fail |
| Safety-first greedy | -1726.835 | 0/1536 | 0.000 | 0.000 | Pass |
| Balanced greedy | -1766.000 | 0/1536 | 0.000 | 0.000 | Pass |
| Profit-first greedy | -1829.203 | 0/1536 | 0.000 | 0.000 | Pass |

Interpretation:

- Raw learned policies often obtain better profit than greedy baselines.
- However, most fail the strict 20/80 safety gate.
- SAC+SN+OCC and CORAL have much smaller strict kWh than raw SAC/PPO, suggesting better boundary-aware behavior.
- This layer demonstrates why profit-only ranking is misleading.

### 6.2 Layer 2: Margin Sensitivity / Certifiability

Question:

> How much shared deployment safety margin is required for each learned method to satisfy strict safety?

Fine margin sweep:

| Common margin | Projection bounds | CORAL violation steps | CORAL net | SAC+SN+OCC violation steps | SAC+SN+OCC net | PPO+SN violation steps | PPO+SN net | All learned safe? |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.030 | 23-77% | 1/1536 | -1569.518 | 1/1536 | -1551.743 | 1/1536 | -1610.821 | No |
| 0.035 | 23.5-76.5% | 0/1536 | -1573.949 | 1/1536 | -1575.335 | 1/1536 | -1626.169 | No |
| 0.036 | 23.6-76.4% | 0/1536 | -1570.445 | 1/1536 | -1573.615 | 1/1536 | -1626.462 | No |
| 0.038 | 23.8-76.2% | 1/1536 | -1567.028 | 1/1536 | -1563.752 | 1/1536 | -1647.172 | No |
| 0.040 | 24-76% | 0/1536 | -1571.089 | 0/1536 | -1567.566 | 0/1536 | -1612.649 | Yes |
| 0.045 | 24.5-75.5% | 0/1536 | -1574.616 | 0/1536 | -1577.345 | 0/1536 | -1619.409 | Yes |

Interpretation:

- A common margin of `0.02` was not enough for all learned baselines.
- A common margin of `0.04` was the smallest tested value that made all learned baselines satisfy zero strict violation.
- CORAL reached zero strict violation at `0.035` and `0.036`, while SAC+SN+OCC still had a small residual violation.
- PPO+SN also reaches zero strict violation once the common margin reaches `0.04`, and remains an important safe learned baseline because it preserves better profit than greedy baselines under the common safety layer.
- Therefore, CORAL's strongest safety story is not "highest profit under every setting" but "easier to certify under smaller safety margins."

Suggested thesis wording:

> Although CORAL is not always the highest-profit learned controller under a common deployment margin, the margin sweep indicates that CORAL can satisfy the strict safety gate with a smaller conservative buffer, suggesting improved certifiability under limited safety margin.

### 6.3 Layer 3: Main Fair Comparison Under Common Safety Layer

Question:

> Once all learned controllers are evaluated in the same safe deployment environment, which controller preserves the best net profit?

This layer defines fairness as a shared deployment environment: the same true SoC bounds, the same projection margin, and the same rollout days for every learned controller. It is the right answer to the deployment question: if the operator allows one common safety layer, which controller has the best economics inside that environment?

The final common deployment setting is:

```text
true SoC bounds: 20-80%
common safety margin: 0.04
internal projection bounds: 24-76%
```

Main fair comparison:

| Method | Net profit | Violation steps | Strict hours | Strict kWh | Gate |
|---|---:|---:|---:|---:|---|
| SAC+SN+OCC | -1567.566 | 0/1536 | 0.000 | 0.000 | Pass |
| CORAL | -1571.089 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC+SN | -1576.740 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC | -1578.759 | 0/1536 | 0.000 | 0.000 | Pass |
| PPO+SN | -1612.649 | 0/1536 | 0.000 | 0.000 | Pass |
| PPO | -1638.157 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC + reward safety penalty | -1639.182 | 0/1536 | 0.000 | 0.000 | Pass |
| Safety-first greedy | -1726.835 | 0/1536 | 0.000 | 0.000 | Pass |
| Balanced greedy | -1766.000 | 0/1536 | 0.000 | 0.000 | Pass |
| Profit-first greedy | -1829.203 | 0/1536 | 0.000 | 0.000 | Pass |

Interpretation:

- Under the common safety layer, all learned controllers pass the strict safety gate.
- Learned controllers outperform all zero-violation greedy baselines.
- SAC+SN+OCC gives the best profit in the common `0.04` margin environment.
- CORAL is very close and has a favorable certifiability result at smaller margins.
- The learned-controller differences are not large enough to claim overwhelming dominance of one method.

### 6.4 Layer 3b: Per-Method Minimum Certified Margin

Question:

> If each learned controller is allowed to use the smallest tested margin that makes it strict-safe, which controller performs best after its own certification?

This is a different fairness definition. It is not the same deployment environment; it is method-specific certification. It is useful because it separates two claims:

- Common-margin fairness: all methods run inside the same deployment safety layer.
- Per-method certification fairness: each method uses only the conservative buffer it needs to pass the strict gate.

Based on the tested margin grid:

| Method | Smallest tested 0-violation margin | Projection bounds | Net profit at that margin | Violation steps |
|---|---:|---|---:|---:|
| CORAL | 0.035 | 23.5-76.5% | -1573.949 | 0/1536 |
| SAC | 0.038 | 23.8-76.2% | -1578.059 | 0/1536 |
| SAC+SN | 0.038 | 23.8-76.2% | -1587.711 | 0/1536 |
| SAC+SN+OCC | 0.040 | 24-76% | -1567.566 | 0/1536 |
| PPO | 0.040 | 24-76% | -1638.157 | 0/1536 |
| PPO+SN | 0.040 | 24-76% | -1612.649 | 0/1536 |
| SAC + reward safety penalty | 0.040 | 24-76% | -1639.182 | 0/1536 |

Interpretation:

- CORAL requires a smaller tested margin, supporting the claim that it is easier to safety-certify.
- SAC+SN+OCC still has the best profit once it clears the gate at `0.04`.
- Because one row in the sweep is mildly non-monotonic (`0.038`), this should be described as the "smallest tested 0-violation margin," not a mathematically exact minimum.
- The thesis can report both Layer 3 and Layer 3b: the former is deployment fairness, while the latter is certification fairness.

### 6.5 Layer 4: Flow-Rate Control Extension

Flow-rate control should not be mixed into the no-flow main table because it changes the action space, pump auxiliary losses, effective power limits, and training difficulty. It is better presented as an actuator extension.

The flow-control methods have now been retrained under the same 20/80 strict protocol. The full flow-specific report is:

```text
docs/scaled_commercial_flow_rate_strict_retrain_report_zh.md
```

The main result is:

| Flow result | Conclusion |
|---|---|
| Layer 1 raw diagnostics | SAC+SN, SAC+SN+OCC, and CORAL are already strict-safe in raw flow rollout; SAC, SAC penalty, PPO, PPO+SN, and flow heuristics still fail. |
| Layer 2 margin sensitivity | The SAC/CORAL family remains strict-safe for common margins from `0.02` to `0.12`; PPO/PPO+SN still fail even when tested up to `0.20`. |
| Layer 3 common safety layer | No all-method common margin was found. For the SAC/CORAL family only, `0.02` is the smallest tested common margin that certifies all family members. |

The right thesis treatment is:

- Keep the no-flow strict safety gate as the main result.
- Present flow-control as an actuator-space extension and stress test.
- Emphasize that SafetyNet/OCC/CORAL remain certifiable in the harder flow action space.
- Do not claim that all learned methods can be jointly certified under flow-rate control.

## 7. How to Present the Results in the Thesis

### 7.1 Main Claim

Recommended main claim:

> In the scaled commercial scenario, raw learned controllers can achieve higher economic return but often violate strict SoC safety constraints. After introducing a common deployment safety layer and applying a strict zero-violation gate, learned controllers remain more profitable than greedy baselines. Boundary-aware variants such as SAC+SN+OCC and CORAL provide the strongest learned-controller results, with CORAL showing favorable certifiability at smaller safety margins.

### 7.2 What Not to Claim

Avoid claiming:

- "CORAL universally beats all methods."
- "RL is always better than greedy."
- "The system is fully supplied by solar."
- "SafetyNet margin changes the true safety boundary."

The true safety boundary remains `20-80%`; the margin only changes the internal projection bounds used by the deployment safety layer.

### 7.3 Recommended Chapter Placement

Possible thesis organization:

1. **Scenario construction**
   - Explain why P302-scale magnitudes are too small for clear commercial EMS profit comparison.
   - Define scaled commercial scenario.
   - Explain PV/load ratio preservation.

2. **Strict safety metric**
   - Explain why clipping SoC hides persistent violations.
   - Define strict violation steps, hours, kWh, and max kWh.
   - Define safety gate.

3. **Raw diagnostics**
   - Show learned policies can be profitable but unsafe.
   - Discuss OCC reducing violation magnitude.

4. **Margin sensitivity**
   - Show required safety margin.
   - Discuss CORAL certifiability.

5. **Common safety-layer fair comparison**
   - Treat SafetyNet margin as part of deployment environment.
   - Compare all controllers under the same safety layer.
   - Emphasize learned controllers outperform greedy after safety gate.

## 8. Reproducibility Notes

### 8.1 Build Dataset

```powershell
py thesis_sim\code\build_scaled_commercial_dataset.py --target-peak-load-kw 60
```

### 8.2 Main Configs

```text
thesis_sim/configs/thesis_scaled_commercial_60kw_noflow.yaml
thesis_sim/configs/thesis_scaled_commercial_60kw_flow.yaml
```

The current main no-flow config uses:

```text
soc_min: 0.20
soc_max: 0.80
clip_soc_to_bounds: false
```

### 8.3 Fair Rollout Outputs

Raw 20/80 strict baseline:

```text
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_strict_20_80/fair_rollout_summary.csv
```

Common-margin sweep:

```text
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_common_margin_0030/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_common_margin_0035/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_common_margin_0036/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_common_margin_0038/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_common_margin_004/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_common_margin_0045/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/fair_rollout_strict_20_80/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/fair_rollout_common_margin_004/fair_rollout_summary.csv
```

Decision notes:

```text
thesis_sim/outputs/scaled_commercial_60kw_noflow_tuning/decision_notes.md
```

Canvas summary:

```text
C:\Users\Administrator\.cursor\projects\c-Users-Administrator-Downloads-HaoYuResearch\canvases\strict-safety-gate-results.canvas.tsx
```

## 9. Limitations and Caveats

1. This is a scaled design-space experiment, not direct commercial hardware validation.
2. PV/load timing and ratios are preserved from measured data, but the commercial magnitude is hypothetical.
3. The no-flow result should not be overgeneralized to flow-control action spaces. The flow-control strict retrain completes the same three-layer protocol, but no all-method common safety layer was found, so flow-control should be framed as an actuator-space extension and stress test.
4. Common SafetyNet margin is a deployment safety layer. Per-method margins should be presented separately as certification analysis.
5. Profit differences among learned controllers under the common safety layer are modest. The stronger thesis contribution is the strict safety-gated evaluation protocol, certifiability analysis, and the finding that learned controllers preserve economic value after satisfying strict safety.

## 9.1 Future Extension: Grid Export / Sell-Back Scenario

The current experiments use:

```text
allow_grid_export: false
allow_grid_trading: false
```

Therefore, the profit conclusion applies to a no-export behind-the-meter EMS setting. A future sell-back scenario can test whether safety-certified controllers remain useful when PV surplus can be exported to the grid.

This should be treated as a separate scenario because:

- The reward model changes: profit can come from export revenue, not only grid-cost reduction.
- The action interpretation changes: PV self-consumption, battery charging, battery discharge to load, grid export, and curtailment must be separated.
- The strict 20-80% SoC safety gate should remain, but the economic objective changes.
- Current no-export checkpoints should not be used to claim sell-back performance without retraining or at least a separate rollout protocol.

## 10. Short Thesis-Ready Summary

This experiment constructs a scaled 60 kW small-commercial microgrid scenario by multiplying the measured P302 clean-window PV and load time series by the same factor, preserving the PV/load ratio while moving the system to commercial-scale magnitudes. The battery is specified as a 60 kW / 240 kWh system with true SoC safety bounds of 20-80%. To avoid undercounting unsafe behavior, strict SoC accounting disables post-step clipping and measures out-of-bound duration and magnitude.

The results show that raw learned policies can produce high profit but frequently violate strict SoC safety. A common deployment SafetyNet margin is therefore introduced as an environment-level safety layer and applied to all learned controllers. With the smallest tested common margin that makes all learned baselines safe (`soc_margin = 0.04`, projection range 24-76%), learned controllers achieve zero strict violation and outperform all zero-violation greedy baselines. SAC+SN+OCC gives the best common-margin profit, while CORAL remains close. Under per-method certified-margin analysis, CORAL reaches zero violation with a smaller tested margin, suggesting better safety certifiability. Flow-control is best presented as an actuator-space stress test: SAC/CORAL-family methods can be certified after retraining, but no all-method common safety layer was found.

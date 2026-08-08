# Seminar Baseline Protocol

This file freezes the P302 baseline experiment protocol used for the seminar comparison.
Do not change these values during a comparison run unless the whole run is restarted and renamed.

## Frozen Dataset And Episodes

- Dataset: `data/processed/training_v20_0511_strict_clean_full_days_raw_only.csv`
- Episode length: `96` steps
- Control interval: `15 min`
- Evaluation unit: one complete day
- Rationale: the raw-only dataset contains clean full days that are not guaranteed to be consecutive, so one-day episodes avoid crossing data gaps while preserving the PV, load, and TOU daily cycle.

## Frozen Hardware And Safety Bounds

- Battery capacity: `0.0112 kWh`
- Charge power limit: `0.0085 kW`
- Discharge power limit: `0.0056 kW`
- One-way efficiency: `0.95`
- Safe SoC range: `0.20-0.80`
- Voltage cutoff SoC: `0.20`
- Deployment discharge semantics: `discharge_auto: true`, `discharge_mode: solo_only`
- Action dead zone: `0.00005 kW`

## Frozen Observation And Reward Surface

- Deployment observation style: enabled
- PV boolean observation: enabled
- PV support ratio observation: enabled
- Price observation: enabled
- Reward version: `v16sp`
- TOU reward scale: `800`
- Guided teacher: disabled for all baselines

## Comparison Layers

Layer 1, raw policy comparison, reports how often the uncorrected policy attempts unsafe actions:

- `violations_attempted`
- low-SoC discharge attempts
- high-SoC charge attempts
- raw action magnitude near safety bounds

Layer 2, deployment-aware comparison, applies the same hard safety projection to all deployable policies and reports:

- `safety_projected_meaningful`
- `projection_delta_mean_w`
- `projection_delta_max_w`
- `net_profit`
- `pv_to_battery_wh`
- `useful_discharge_wh`
- effective `situation_code = 1` events where available

## Implementable P302 Baselines

The current P302 deployment repository has a SAC training stack. PPO-Lagrangian and PPO+SafetyNet exist in the older StressM project and are valid literature baselines, but they are not wired to the P302 clean raw-data environment yet. For this repository's P302 run, the executable baselines are:

- `rule_heuristic`
- `sac_raw`
- `sac_penalty`
- `sac_train_safetynet`
- `sac_sn_occ`
- `ours_full`

PPO baselines should be reported as StressM robustness evidence or explicitly marked as pending P302 porting, not mixed into the P302 raw-data table as if they used the same environment.

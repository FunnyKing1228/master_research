# Greedy Heuristic Baselines

This note replaces the earlier plan to use one complete human-knowledge rule as the main rule-based baseline. The thesis baseline should instead compare three simple, transparent greedy controllers:

- Safety-first greedy.
- Profit-first greedy.
- Balanced safety-profit greedy.

The goal is not to claim that any rule has perfect knowledge. The goal is to provide interpretable non-learning references that expose the safety-profit trade-off more clearly than a single complex expert controller.

Related appendix:

- [Human-Knowledge Heuristic Rule](human_knowledge_heuristic_rule.md)

## Shared Scope

All three greedy policies run at each 15-minute EMS decision step and output one battery power command:

```text
power_kw > 0: charge
power_kw = 0: standby
power_kw < 0: discharge
```

If flow control is enabled, flow is derived from the selected power command. Standby should use the platform standby flow convention; active charge/discharge should use at least the active-flow minimum.

These policies are designed for baseline comparison only. They do not change the simulation core, reward function, PV interpretation, or hardware feasibility semantics.

## Input Fields

Each greedy policy may use:

- `soc`: current battery SoC.
- `load_kw`: measured or deployment-style effective load.
- `pv_kw`: measured PV support / PV availability proxy.
- `pv_support_ratio = pv_kw / max(load_kw, eps)`, clipped only for observation/reporting.
- `price`: TOU price at the current step.
- `battery_charge_power_kw`: maximum battery charge command.
- `battery_discharge_power_kw`: maximum standalone discharge command.
- `battery_capacity_kwh`.
- `battery_efficiency`.
- `time_step`: normally `0.25` h.
- `soc_min`, `soc_max`: operational safety range, normally `0.20` to `0.80`.
- `voltage_cutoff_soc`, if configured.
- `pv_obs_boolean_threshold_kw`, used only as a conservative PV-active discharge block.
- `pv_surplus_threshold_kw`, if an explicit PV-surplus proxy is available.

The policies should not infer binary source selection from `grid == 0`, bus/grid voltage comparisons, or an instantaneous source label. PV and grid may support the load at the same time.

## Shared Hard Guards

These guards are common to all three policies and are applied before any profit-oriented action:

- Do not discharge when `pv_kw > pv_obs_boolean_threshold_kw`.
- Do not discharge when `load_kw` is effectively zero.
- Do not discharge when `load_kw > battery_discharge_power_kw` under `solo_only` semantics.
- Do not discharge when `soc <= voltage_cutoff_soc`.
- Do not discharge when one-step SoC prediction would fall below the policy's discharge target.
- Do not charge when one-step SoC prediction would exceed the policy's charge target.
- If the action magnitude is below the action deadband, use standby.

The PV-active discharge block is intentionally conservative. It avoids treating the battery as a third partial-assist source alongside PV/grid. This does not mean PV is assumed to fully supply the load; it only means battery discharge is not considered valid while PV support is active.

## One-Step Power Limits

Charging room:

```text
charge_room_kw =
    max(0, (soc_charge_target - soc) * battery_capacity_kwh
              / (time_step * battery_efficiency))
```

Discharging room:

```text
discharge_room_kw =
    max(0, (soc - soc_discharge_target) * battery_capacity_kwh
              * battery_efficiency / time_step)
```

Candidate power commands are clipped by these rooms before execution.

## Policy A: Safety-First Greedy

### Intent

Safety-first greedy prioritizes avoiding invalid discharge, voltage cutoff, and SoC-boundary pressure. It accepts lower profit and more standby behavior.

### Decision Priority

1. Apply shared hard guards.
2. If SoC is at or below voltage cutoff, allow only small off-peak recovery charge; otherwise standby.
3. If PV support is strong and SoC has conservative room, charge gently.
4. If off-peak and SoC is low, charge gently for recovery.
5. If peak price, PV inactive, load can be served by battery alone, and SoC has large reserve, discharge exactly enough to serve the load.
6. Otherwise standby.

### Charging Conditions

- PV-support charge:
  - `pv_support_ratio >= 1.05`.
  - `soc < 0.72`.
  - Use explicit `pv_surplus_kw = max(0, pv_kw - load_kw)` if positive.
  - If surplus is uncertain, use only a conservative fraction such as `0.30 * charge_limit`.
- Off-peak recovery:
  - `price <= offpeak_threshold`.
  - `soc < 0.35`.
  - Use a small command such as `0.25 * charge_limit`.

### Discharging Conditions

- `price >= peak_threshold`.
- PV is inactive.
- `load_kw <= battery_discharge_power_kw`.
- `soc > 0.34`.
- One-step SoC remains above `0.26`.
- Command must be large enough to serve the load under solo-only semantics:

```text
power_kw = -min(load_kw, discharge_limit_kw, discharge_room_kw)
```

If the clipped command cannot cover the load, standby.

### Standby Conditions

Standby is selected whenever any hard guard blocks the candidate action or no rule above applies.

## Policy B: Profit-First Greedy

### Intent

Profit-first greedy tries to exploit TOU price differences and reduce grid demand more aggressively while still respecting hardware hard guards.

### Decision Priority

1. Apply shared hard guards.
2. Charge from PV support when PV support is high enough and SoC has room.
3. Charge during off-peak periods up to a higher SoC target for later peak use.
4. Discharge during mid/peak price periods when PV is inactive and the battery can serve the load alone.
5. Otherwise standby.

### Charging Conditions

- PV-support charge:
  - `pv_support_ratio >= 0.95`.
  - `soc < 0.78`.
  - Use explicit PV surplus if positive.
  - If only support ratio is available, cap the command, for example `0.70 * charge_limit`.
- Off-peak arbitrage/recovery charge:
  - `price <= offpeak_threshold`.
  - `soc < 0.74`.
  - Command around `0.60 * charge_limit`, clipped by one-step room.

This rule should be described as grid-demand and TOU optimization, not as proof of exclusive PV supply.

### Discharging Conditions

- `price >= mid_or_peak_threshold`, e.g. shoulder/peak price.
- PV is inactive.
- `load_kw <= battery_discharge_power_kw`.
- `soc > 0.24`.
- One-step SoC remains above `0.22`.
- Command must be able to serve the load by itself:

```text
power_kw = -min(load_kw, discharge_limit_kw, discharge_room_kw)
```

If the command cannot cover the load after clipping, standby.

### Standby Conditions

Standby is selected when PV is active during an otherwise profitable discharge window, when load exceeds standalone battery capability, or when SoC room is insufficient.

## Policy C: Balanced Safety-Profit Greedy

### Intent

Balanced greedy starts from the profit-first idea but adds margins that reduce unsafe attempts and safety-layer dependence. It is the preferred compact rule baseline when only one heuristic row can be shown.

### Decision Priority

1. Apply shared hard guards.
2. Charge from PV support only with a moderate support threshold and conservative power cap.
3. Charge off-peak only to a moderate reserve target.
4. Discharge only during peak periods, not shoulder periods.
5. Require extra SoC reserve before discharge.
6. Otherwise standby.

### Charging Conditions

- PV-support charge:
  - `pv_support_ratio >= 1.00`.
  - `soc < 0.74`.
  - Use explicit PV surplus if positive.
  - If surplus is uncertain, cap at about `0.45 * charge_limit`.
- Off-peak reserve charge:
  - `price <= offpeak_threshold`.
  - `soc < 0.55`.
  - Cap at about `0.35 * charge_limit`.

### Discharging Conditions

- `price >= peak_threshold`.
- PV is inactive.
- `load_kw <= battery_discharge_power_kw`.
- `soc > 0.30`.
- One-step SoC remains above `0.26`.
- Command must be able to serve the load alone.

### Standby Conditions

Balanced greedy should standby more often than profit-first in ambiguous conditions, especially when PV support is active, SoC margin is thin, or the load is near the standalone discharge limit.

## Expected Evaluation Metrics

Report the three greedy policies with the same metrics used for RL and safety-layer comparison:

- Net profit / grid cost savings.
- Grid import and grid demand reduction.
- PV-to-battery energy, using conservative "PV support" wording.
- Useful discharge energy.
- Realized SoC violations.
- Attempted invalid discharge or blocked invalid discharge counts.
- Safety projection / intervention count if a safety layer is applied.
- Projection magnitude.
- Unsafe-state duration and recovery metrics if no-20/80-clipping evaluation is enabled.
- Flow-power mismatch and pump energy if flow control is part of the experiment.

## Thesis Framing

Suggested method text:

> We compare the learned controller against three transparent greedy EMS baselines. The safety-first rule prioritizes feasibility and SoC reserve, the profit-first rule greedily exploits TOU/grid-demand opportunities subject to hardware guards, and the balanced rule adds conservative margins to reduce unsafe attempts. These baselines use PV support and grid demand conservatively; they do not assume binary PV-versus-grid source selection.

Suggested interpretation:

> A higher-profit greedy rule is not automatically more deployable if it causes more invalid attempts or relies heavily on projection. Conversely, a conservative rule may be safe but economically inactive. The comparison is therefore reported as a safety-profit trade-off rather than a single reward ranking.

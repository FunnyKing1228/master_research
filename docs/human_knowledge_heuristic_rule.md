# Human-Knowledge Heuristic Rule

Status update: this complete expert-style rule is no longer the recommended main thesis baseline. Based on the latest advisor direction, the main baseline should be the simpler three-policy greedy comparison described in:

- [Greedy Heuristic Baselines](greedy_heuristic_baselines.md)

This document is kept as an appendix/design record for the earlier "complete human knowledge" controller idea. It can still explain why hand-written rules can be strong, but it should not be presented as the primary baseline design unless the thesis explicitly needs a complex expert-controller appendix.

This note records the intended strong rule-based controller for thesis and baseline discussion. The goal is not to create a weak baseline that RL can easily beat. Instead, this heuristic represents an expert-engineered controller that uses explicit platform knowledge.

## Positioning

The heuristic should be described as a strong knowledge-based baseline:

- It uses manually encoded microgrid operation knowledge.
- It has access to explicit one-step SoC prediction formulas.
- It directly checks platform feasibility rules such as invalid discharge.
- It is transparent and conservative, but may require manual redesign when platform assumptions change.

The intended comparison is therefore:

> expert-written if-else control versus learned policy with runtime safety assurance.

RL does not need to unconditionally beat this controller in every metric. The key question is whether RL can approach expert-designed economic behavior while reducing unsafe attempts and dependence on manually enumerated rules.

## Inputs

At each 15-minute decision step:

- `soc`: current SoC.
- `load_kw`: measured load power.
- `pv_kw`: PV support or available PV proxy.
- `pv_support_ratio`: PV support relative to load. This is not the same as PV surplus.
- `price`: TOU price.
- `hour`, `weekday`: time features.
- `battery_charge_limit_kw`: maximum charge power.
- `battery_discharge_limit_kw`: maximum discharge power.
- `soc_min`, `soc_max`: operational safety range, e.g. 0.20 to 0.80.
- `dt_h`: decision interval, normally 0.25 h.
- `battery_capacity_kwh`.
- `battery_efficiency`.
- flow-control limits if flow action is enabled.

## One-Step SoC Prediction

The heuristic must decide not only whether to charge/discharge, but also how much power can be used without leaving the safe operating region after one step.

For charging:

```text
soc_next = soc + charge_kw * dt_h * eta / capacity_kwh
```

Maximum safe charging power:

```text
soc_room_charge_kw =
    (soc_max_target - soc) * capacity_kwh / (dt_h * eta)
```

For discharging:

```text
soc_next = soc - discharge_kw * dt_h / (eta * capacity_kwh)
```

Maximum safe discharging power:

```text
soc_room_discharge_kw =
    (soc - soc_min_target) * capacity_kwh * eta / dt_h
```

Use buffer targets instead of exact boundaries when possible, for example:

```text
soc_min_target = 0.22
soc_max_target = 0.78
```

This keeps the heuristic away from the hard operational boundary.

## Rule Priority

Rules are applied in priority order.

## 1. Hard Safety And Data Validity

Standby if any of the following holds:

- Missing or stale load/PV/battery data.
- SoC is already below a low lock threshold and no safe recovery action is available.
- Voltage cutoff or health lock is active.
- Previous window indicated firmware override or false discharge.
- No effective load is present and the candidate action is discharge.

Standby output:

```text
power_kw = 0
flow_pct = 50
```

## 2. Invalid Discharge Prevention

Discharge is allowed only if all feasibility conditions are true:

- PV is not actively supporting the load.
- Effective load exists.
- `load_kw <= battery_discharge_limit_kw`.
- `soc > soc_min_target`.
- One-step discharge will not move SoC below `soc_min_target`.

If any condition fails, the heuristic must not discharge.

This is the main platform-specific feasibility rule. In the target platform, PV and grid may support the load together, but battery discharge should not be treated as an arbitrary third partial-assist source.

## 3. PV-Support Charging

If PV support is strong and SoC has room to charge:

```text
if pv_support_ratio >= pv_sufficient_threshold
   and soc < soc_max_target:
```

Then choose:

```text
charge_kw = min(
    battery_charge_limit_kw,
    pv_surplus_or_safe_charge_proxy_kw,
    soc_room_charge_kw
)
```

Important caveat:

`pv_support_ratio` is an observation feature, not a direct measurement of surplus PV power. A ratio near or above 1 only indicates that PV appears able to support the current load. It does not imply that enough surplus PV exists to charge the battery at an arbitrary rate.

For example:

```text
load_kw = 0.4 W
pv_support_ratio = 1.0
```

This may mean PV is just enough to cover the load. If the controller starts charging aggressively, the extra charging power may come from the grid or create an infeasible operating assumption.

Therefore, the heuristic should not interpret:

```text
pv_support_ratio >= 1.0
```

as:

```text
unlimited PV surplus is available
```

If a reliable PV surplus estimate is available, use it explicitly:

```text
pv_surplus_kw = max(0, pv_available_kw - load_kw)
charge_kw = min(
    pv_surplus_kw,
    battery_charge_limit_kw,
    soc_room_charge_kw
)
```

If PV surplus is uncertain and only `pv_support_ratio` is available, use a conservative fraction of the charge limit rather than assuming full surplus:

```text
charge_kw = min(
    0.2 to 0.5 * battery_charge_limit_kw,
    soc_room_charge_kw
)
```

An even more conservative option is gradual charging:

```text
charge_fraction = clip((pv_support_ratio - 0.8) / 0.7, 0, 1)
charge_kw = min(
    charge_fraction * 0.3 * battery_charge_limit_kw,
    soc_room_charge_kw
)
```

This means high PV support allows charging, but does not automatically justify maximum charging power.

## 4. Off-Peak Recovery Charging

If SoC is low and electricity price is off-peak:

```text
if price <= offpeak_threshold
   and soc < recovery_soc_target:
```

Then allow small grid-assisted recovery charging:

```text
charge_kw = min(
    recovery_charge_fraction * battery_charge_limit_kw,
    soc_room_charge_kw
)
```

This is not an arbitrage claim. It is a safety-margin recovery rule.

## 5. Peak Discharge

If price is high and discharge is feasible:

```text
if price >= peak_threshold
   and all invalid-discharge checks pass:
```

Then choose:

```text
discharge_kw = min(
    load_kw,
    battery_discharge_limit_kw,
    soc_room_discharge_kw
)
```

Output action is negative:

```text
power_kw = -discharge_kw
```

No partial assist is allowed under `solo_only` semantics.

## 6. Otherwise Standby

If none of the above rules applies:

```text
power_kw = 0
flow_pct = 50
```

## Flow Rule

Flow should be derived from power, not randomly selected.

```text
if abs(power_kw) <= deadband:
    flow_pct = 50
else:
    required_fraction = abs(power_kw) / corresponding_power_limit_kw
    flow_pct = 100 * max(0.60, required_fraction)
```

Active charging/discharging uses at least 60% flow. Standby uses 50% flow to keep voltage measurement reliable.

## Why This Heuristic Can Be Very Strong

This heuristic can directly calculate the next-step SoC and avoid many violations by construction. That is precisely why it should be called a strong expert baseline, not a simple rule.

If it outperforms RL in some profit metrics, the interpretation should be:

- the heuristic encodes substantial human knowledge;
- the economic opportunity in the environment may be limited;
- RL should be evaluated not only by profit but also by unsafe attempts, safety-layer reliance, and behavior under deployment-oriented stress tests.

## Relationship To Planned Experiments

This heuristic is linked to the planned safety-first evaluation protocol in:

- [Planned Safety Experiments And Evaluation Settings](planned_safety_experiments.md)


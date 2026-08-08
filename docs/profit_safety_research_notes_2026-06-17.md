# Profit And Safety Research Notes (2026-06-17)

## 1. Can safety violations include more than SoC?

Yes. SoC is only the most obvious battery-state constraint. For P302, the more defensible framing is:

- Energy-state violations: SoC/SoE below the discharge floor or above the charge ceiling.
- Power-limit violations: commanded charge/discharge exceeds battery or hardware power limits.
- Action-projection violations: raw policy action requires SafetyNet/guard correction by more than the dead zone.
- PV/load feasibility violations: battery-solo discharge is requested when PV is active, load is zero, or the discharge power cannot cover the load under `solo_only` semantics.
- Voltage cutoff violations: discharge is attempted while voltage/SoC cutoff lock is active.
- Flow-power violations: requested flow rate cannot support the requested power, so the flow-power guard clips the action.
- Current/thermal health violations: over-current, abnormal charge-like current during commanded discharge, repeated low-voltage samples, or SoH health lock activation.
- Grid/relay mode feasibility violations: command mode does not support the intended energy path, e.g. standby mode leaves load unserved in the real firmware.
- Data-quality violations: stale `Data.txt`, missing measured load, impossible negative/zero sensor combinations, or schema mismatch.

For thesis/reporting, the safest split is:

- realized violations: physical or simulated state actually crosses a bound;
- attempted violations: raw action would cross a bound before shielding;
- intervention events: safety layer changes the action to prevent the violation.

This lets CORAL show value even when realized violations are near zero: the point is not only fewer failures, but fewer unsafe attempts and less reliance on last-moment projection.

## 2. What does current `profit` actually mean?

The current environment does not model real energy selling. In `core/microgrid_env.py`, the deployment economic signal is based on:

- `baseline_grid_kw = max(0, load - PV)`
- `grid_kw = remaining grid demand after action + pump cost`
- `grid_cost = grid_kw * dt * price`
- `baseline_cost = baseline_grid_kw * dt * price`
- `grid_savings = baseline_cost - grid_cost`
- `total_revenue += max(0, grid_savings)`
- `total_cost += grid_cost`
- `net_profit = total_revenue - total_cost`

So `net_profit` is not a clean accounting profit. It mixes avoided-grid-cost "revenue" with remaining grid cost. A clearer name would be `net_grid_value` or `relative_grid_savings_minus_cost`.

This matters because the current metric can make differences look small or counterintuitive, especially when load is tiny and sell-back is disabled.

## 3. Why can the heuristic look competitive or higher?

The v22 flow baseline summary suggests the heuristic is not winning through clever discharge. It mostly avoids action:

- heuristic rollout has `situation_code = 4` for 1231 of 1248 steps;
- only 17 steps are `situation_code = 3`;
- `useful_discharge_wh = 0`;
- `pump_power_wh` is almost zero;
- attempted violations and projections are zero.

This means the heuristic is essentially a conservative no-op / light-charge policy. In a small-load, no-sell-back setting, doing little can look strong because:

- the available economic upside is tiny;
- each active flow/action introduces pump cost or grid cost;
- RL policies still explore or produce actions that are later blocked/clipped;
- SafetyNet/CORAL variants may sacrifice profit for lower realized risk;
- the v22 flow comparison still uses `deployment_group_power_kw: 0.0023` in configs, while current hardware is closer to `0.0001`.

Therefore, if heuristic profit is close to RL, that does not automatically mean heuristic is better. It may mean the current task has too little economic opportunity and the metric rewards inactivity.

## 4. Should simulation add a selling/export mechanism?

It is a reasonable next experiment, but it must be framed as a simulation extension, not as the current deployed hardware claim.

Why it helps:

- Adds a real arbitrage channel: charge/store when PV or price is favorable, discharge/export when price is high.
- Increases the action space where learning can outperform simple threshold rules.
- Makes price observation more meaningful than in pure load-serving mode.
- Lets CORAL show value under higher-risk opportunities: more profit chances but also more unsafe attempts to manage.

Recommended conservative design:

- Add `allow_grid_export` or `sell_back_enabled`.
- Add `feed_in_tariff_ratio`, e.g. sell price = `0.3-0.8 * buy_price`, not equal full retail price by default.
- Compute exported energy only after load is served and battery/grid constraints are respected.
- Penalize or disallow export from grid-charged energy if that would be unrealistic.
- Report two tables:
  - deployment-aligned no-export table;
  - simulation-only export/arbitrage stress table.

This avoids overclaiming while giving a stronger demonstration of RL value.

## 5. Immediate analysis plan

Short-term, before changing training:

- Rename or explain `net_profit` as deployment-aware economic value, not literal accounting profit.
- Add safety metrics beyond SoC to the report table: attempted violations, meaningful projection count, PV-active discharge blocks, load-over-discharge blocks, voltage cutoff blocks, flow-power-limited count, pump energy, and invalid mode/data events.
- Show an "opportunity-limited" explanation: with no sell-back and very small load, the best policy may be close to standby.
- Compare energy behavior, not only profit: PV-to-battery Wh, useful discharge Wh, pump Wh, situation-code counts, and projection delta.

Medium-term:

- Rerun baseline suite with `deployment_group_power_kw: 0.0001` to match actual hardware.
- Add a selling/export simulation scenario as a separate ablation.
- Add a stronger heuristic baseline and a "do nothing" baseline, so rule-based performance is interpretable.

## 6. Suggested answer to the committee

"目前 SoC 是核心安全限制，但不是唯一限制。我們可以把安全分成狀態違規、動作可行性違規與部署介入事件。除了 SoC，上限功率、PV 存在時放電、負載不足以支撐 battery-solo、電壓 cutoff、flow-rate 對可用功率的限制，以及 SafetyNet 投影量都可以作為違規或風險指標。這樣可以更清楚呈現 CORAL 的價值：不只是最後有沒有越界，也包含原始策略是否經常嘗試不安全動作，以及安全層需要介入多少。"

"至於 profit，目前 P302 硬體與模擬設定偏向負載供應，沒有真正賣電，所以套利空間有限。在負載很小時，保守 heuristic 幾乎不動作反而可能看起來接近最佳。後續可以加入一個明確標成 simulation-only 的賣電/併網回售情境，讓價格訊號有更大的操作空間，再比較 CORAL 與 rule-based heuristic 在高機會、高風險條件下的差異。"

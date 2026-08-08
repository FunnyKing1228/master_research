from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from livestock_microgrid_env import MicrogridEnv  # noqa: E402


class CarbonFocusedMicrogridEnv(MicrogridEnv):
    """Carbon-oriented variant of the livestock farm microgrid environment.

    The original environment is mostly economic: it rewards lower electricity
    cost and PV usage. This variant is intentionally more carbon-oriented:

    - penalize every kWh imported from the grid using a carbon factor
    - keep a smaller electricity-cost term so the policy still sees TOU signals
    - reward PV utilization
    - keep strong safety penalties from the parent environment

    This makes the agent prefer PV charging and grid import reduction over pure
    electricity price arbitrage.
    """

    def __init__(
        self,
        *args,
        carbon_factor_kg_per_kwh: float = 0.495,
        carbon_weight: float = 7.5,
        cost_weight: float = 0.25,
        pv_reward_weight: float = 0.08,
        cycling_cost_weight: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.carbon_factor_kg_per_kwh = carbon_factor_kg_per_kwh
        self.carbon_weight = carbon_weight
        self.cost_weight = cost_weight
        self.pv_reward_weight = pv_reward_weight
        self.cycling_cost_weight = cycling_cost_weight

    def _calculate_reward(
        self,
        grid_import_kw: float,
        price: float,
        pv_used_kw: float,
        battery_power_kw: float,
        safety_penalty: float,
    ) -> float:
        grid_energy_kwh = grid_import_kw * self.dt_hours
        carbon_kg = grid_energy_kwh * self.carbon_factor_kg_per_kwh
        grid_cost_twd = grid_energy_kwh * price
        pv_used_kwh = pv_used_kw * self.dt_hours
        battery_throughput_kwh = abs(battery_power_kw) * self.dt_hours

        carbon_penalty = self.carbon_weight * carbon_kg
        cost_penalty = self.cost_weight * grid_cost_twd
        pv_bonus = self.pv_reward_weight * pv_used_kwh
        cycling_cost = self.cycling_cost_weight * battery_throughput_kwh

        return -carbon_penalty - cost_penalty + pv_bonus - cycling_cost - safety_penalty

    def step(self, action: np.ndarray):
        """Apply a carbon-oriented action shield before the parent dynamics.

        Charging from the grid can reduce electricity cost under TOU pricing but
        usually increases emissions. For the carbon-focused version, positive
        battery action is clipped to PV surplus only. This makes the environment
        explicitly prefer PV charging instead of price arbitrage.
        """

        row = self.profile.iloc[self.step_idx]
        command = float(np.clip(action[0], -1.0, 1.0))
        if command > 0.0:
            pv_surplus_kw = max(float(row["weather_adjusted_power_kw"]) - float(row["load_kw"]), 0.0)
            max_pv_charge_action = pv_surplus_kw / self.battery.max_power_kw
            command = min(command, max_pv_charge_action)
        return super().step(np.array([command], dtype=np.float32))

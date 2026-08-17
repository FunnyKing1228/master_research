"""Provisional off-grid LFP environment for newHW.

This module is isolated from the P302 environment by design.  It models only
PV, one battery, one fixed load, curtailment, and unmet load.  It does not
model a grid, TOU price, pump, or flow action.

TODO(newHW): topology, BMS limits, SoC calibration, power limits, and the final
reward objective require human/hardware confirmation.  See
docs/handover/newHW_pending_data.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class NewHWMicrogridEnvironment(gym.Env):
    """Minimal 1D off-grid battery dispatch environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        dataset_csv_path: str,
        episode_length: int,
        time_step: float,
        battery_capacity_kwh: float,
        battery_charge_power_kw: float,
        battery_discharge_power_kw: float,
        battery_efficiency: float,
        soc_min: float,
        soc_max: float,
        initial_soc: float,
        soc_init_mode: str = "fixed",
        reward: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        path = Path(dataset_csv_path)
        if not path.exists():
            raise FileNotFoundError(f"newHW dataset not found: {path}")
        self.episode_data = pd.read_csv(path, parse_dates=["timestamp"])
        required = {"timestamp", "Solar", "Consumption"}
        missing = sorted(required.difference(self.episode_data.columns))
        if missing:
            raise ValueError(f"newHW dataset missing columns: {missing}")
        if len(self.episode_data) < 1:
            raise ValueError("newHW dataset is empty")

        self.pv_data = self.episode_data["Solar"].astype(float).to_numpy()
        self.load_data = self.episode_data["Consumption"].astype(float).to_numpy()
        self.price_data = np.zeros(len(self.episode_data), dtype=float)
        self.pv_bool_data = (self.pv_data > 0.001).astype(float)

        self.episode_length = min(int(episode_length), len(self.episode_data))
        self.time_step = float(time_step)
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.battery_charge_power_kw = float(battery_charge_power_kw)
        self.battery_discharge_power_kw = float(battery_discharge_power_kw)
        self.battery_power_kw = max(
            self.battery_charge_power_kw, self.battery_discharge_power_kw
        )
        self.battery_efficiency = float(battery_efficiency)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.soc_min_eff = self.soc_min
        self.soc_max_eff = self.soc_max
        self.initial_soc = float(initial_soc)
        self._soc_init_mode = str(soc_init_mode)

        # Compatibility attributes used by the shared SAC/SafetyNet utilities.
        self.use_flow_rate_action = False
        self.flow_idle_fraction = 0.0
        self.hard_guard = True
        self.clip_soc_to_bounds = True
        self.ramp_limit_kw = None
        self.safetynet_ramp_kw = None
        self.reward_scaling = 1.0
        self.fixed_start_idx: Optional[int] = 0
        self.soc_physical_floor = 0.0
        self.action_dead_zone_kw = 0.0

        reward = reward or {}
        # TODO(newHW): these weights are a provisional smoke objective only.
        self.reward_weights = {
            "served_load": float(reward.get("served_load", 1.0)),
            "unmet_load": float(reward.get("unmet_load", 12.0)),
            "low_soc_reserve": float(reward.get("low_soc_reserve", 2.0)),
            "battery_throughput": float(reward.get("battery_throughput", 0.05)),
            "pv_curtailment": float(reward.get("pv_curtailment", 0.05)),
        }
        self.reserve_soc = float(reward.get("reserve_soc", 0.20))

        # Observation: SoC, load kW, PV kW, continuous PV/load support ratio,
        # sin(hour), cos(hour).  No grid or price observation exists.
        low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, np.inf, np.inf, 5.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.current_soc = self.initial_soc
        self.current_step = 0
        self.start_idx = 0
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.soc_violations = 0
        self.action_violations = 0
        self.strict_soc_violation_steps = 0
        self.strict_soc_violation_duration_h = 0.0
        self.strict_soc_violation_kwh = 0.0
        self.strict_soc_violation_max_kwh = 0.0
        self.total_unmet_load_kwh = 0.0
        self.total_served_load_kwh = 0.0
        self.total_pv_curtailed_kwh = 0.0
        self.total_battery_throughput_kwh = 0.0

    def _dataset_index(self) -> int:
        return min(self.start_idx + self.current_step, len(self.episode_data) - 1)

    def _get_obs_hour_dow(self, step: Optional[int] = None) -> Tuple[int, int]:
        offset = self.current_step if step is None else int(step)
        idx = min(self.start_idx + offset, len(self.episode_data) - 1)
        timestamp = pd.Timestamp(self.episode_data.iloc[idx]["timestamp"])
        return int(timestamp.hour), int(timestamp.dayofweek)

    def _observation(self) -> np.ndarray:
        idx = self._dataset_index()
        load_kw = float(self.load_data[idx])
        pv_kw = float(self.pv_data[idx])
        ratio = float(np.clip(pv_kw / max(load_kw, 1e-9), 0.0, 5.0))
        hour, _ = self._get_obs_hour_dow()
        angle = 2.0 * np.pi * hour / 24.0
        return np.array(
            [
                self.current_soc,
                load_kw,
                pv_kw,
                ratio,
                np.sin(angle),
                np.cos(angle),
            ],
            dtype=np.float32,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        max_start = max(0, len(self.episode_data) - self.episode_length)
        if self.fixed_start_idx is not None:
            self.start_idx = int(np.clip(self.fixed_start_idx, 0, max_start))
        else:
            self.start_idx = int(self.np_random.integers(0, max_start + 1))
        if self._soc_init_mode == "full_random":
            self.current_soc = float(self.np_random.uniform(self.soc_min, self.soc_max))
        else:
            self.current_soc = float(np.clip(self.initial_soc, self.soc_min, self.soc_max))
        self._reset_counters()
        return self._observation(), {
            "soc": self.current_soc,
            "start_idx": self.start_idx,
            "status": "PROVISIONAL_IN_SAMPLE_ONLY",
        }

    def predict_soc_raw(self, current_soc: float, action_kw: float) -> float:
        action_kw = float(action_kw)
        if action_kw >= 0.0:
            delta_kwh = action_kw * self.time_step * self.battery_efficiency
        else:
            delta_kwh = action_kw * self.time_step / max(self.battery_efficiency, 1e-9)
        return float(current_soc + delta_kwh / self.battery_capacity_kwh)

    def step(self, action: Any):
        requested_kw = float(np.asarray(action, dtype=float).reshape(-1)[0])
        if requested_kw < -self.battery_discharge_power_kw - 1e-9:
            self.action_violations += 1
        if requested_kw > self.battery_charge_power_kw + 1e-9:
            self.action_violations += 1
        requested_kw = float(
            np.clip(
                requested_kw,
                -self.battery_discharge_power_kw,
                self.battery_charge_power_kw,
            )
        )

        idx = self._dataset_index()
        load_kw = max(0.0, float(self.load_data[idx]))
        pv_kw = max(0.0, float(self.pv_data[idx]))
        pv_to_load_kw = min(pv_kw, load_kw)
        residual_load_kw = max(0.0, load_kw - pv_to_load_kw)
        pv_surplus_kw = max(0.0, pv_kw - pv_to_load_kw)

        charge_kw = 0.0
        discharge_kw = 0.0
        if requested_kw > 0.0:
            soc_charge_limit_kw = max(
                0.0,
                (self.soc_max - self.current_soc)
                * self.battery_capacity_kwh
                / (self.time_step * max(self.battery_efficiency, 1e-9)),
            )
            charge_kw = min(requested_kw, pv_surplus_kw, soc_charge_limit_kw)
        elif requested_kw < 0.0:
            soc_discharge_limit_kw = max(
                0.0,
                (self.current_soc - self.soc_min)
                * self.battery_capacity_kwh
                * self.battery_efficiency
                / self.time_step,
            )
            discharge_kw = min(
                -requested_kw,
                residual_load_kw,
                self.battery_discharge_power_kw,
                soc_discharge_limit_kw,
            )

        applied_action_kw = charge_kw - discharge_kw
        next_soc_raw = self.predict_soc_raw(self.current_soc, applied_action_kw)
        if next_soc_raw < self.soc_min - 1e-9 or next_soc_raw > self.soc_max + 1e-9:
            self.soc_violations += 1
        self.current_soc = float(np.clip(next_soc_raw, self.soc_min, self.soc_max))

        served_load_kw = pv_to_load_kw + discharge_kw
        unmet_load_kw = max(0.0, load_kw - served_load_kw)
        pv_curtailed_kw = max(0.0, pv_surplus_kw - charge_kw)
        throughput_kw = charge_kw + discharge_kw
        self.total_served_load_kwh += served_load_kw * self.time_step
        self.total_unmet_load_kwh += unmet_load_kw * self.time_step
        self.total_pv_curtailed_kwh += pv_curtailed_kw * self.time_step
        self.total_battery_throughput_kwh += throughput_kw * self.time_step

        load_denom = max(load_kw, 1e-9)
        served_fraction = float(np.clip(served_load_kw / load_denom, 0.0, 1.0))
        unmet_fraction = float(np.clip(unmet_load_kw / load_denom, 0.0, 1.0))
        low_soc_deficit = max(0.0, self.reserve_soc - self.current_soc) / max(
            self.reserve_soc, 1e-9
        )
        throughput_fraction = throughput_kw / max(self.battery_power_kw, 1e-9)
        curtailment_fraction = pv_curtailed_kw / max(pv_kw, 1e-9) if pv_kw > 0 else 0.0
        reward = (
            self.reward_weights["served_load"] * served_fraction
            - self.reward_weights["unmet_load"] * unmet_fraction
            - self.reward_weights["low_soc_reserve"] * low_soc_deficit
            - self.reward_weights["battery_throughput"] * throughput_fraction
            - self.reward_weights["pv_curtailment"] * curtailment_fraction
        )

        # newHW has no grid.  Code 1 means battery is actively serving the
        # off-grid residual load; all non-discharge states use code 4.
        situation_code = 1 if discharge_kw > 1e-9 else 4
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = self.start_idx + self.current_step >= len(self.episode_data)
        obs = self._observation()
        info = {
            "status": "PROVISIONAL_IN_SAMPLE_ONLY",
            "current_soc": self.current_soc,
            "load": load_kw,
            "pv": pv_kw,
            "grid_kw": 0.0,
            "pv_to_load": pv_to_load_kw,
            "pv_to_battery": charge_kw,
            "useful_discharge": discharge_kw,
            "served_load_kw": served_load_kw,
            "unmet_load_kw": unmet_load_kw,
            "pv_curtailed_kw": pv_curtailed_kw,
            "requested_action_kw": requested_kw,
            "applied_action_kw": applied_action_kw,
            "situation_code": situation_code,
            "soc_violations": self.soc_violations,
            "action_violations": self.action_violations,
            "strict_soc_violation_steps": self.strict_soc_violation_steps,
            "strict_soc_violation_duration_h": self.strict_soc_violation_duration_h,
            "strict_soc_violation_kwh": self.strict_soc_violation_kwh,
            "strict_soc_violation_max_kwh": self.strict_soc_violation_max_kwh,
            "total_unmet_load_kwh": self.total_unmet_load_kwh,
            "total_served_load_kwh": self.total_served_load_kwh,
            "total_pv_curtailed_kwh": self.total_pv_curtailed_kwh,
            "total_battery_throughput_kwh": self.total_battery_throughput_kwh,
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "flow_action": 0.0,
            "flow_power_limited": 0,
            "flow_too_low_active": 0,
            "flow_power_mismatch": 0,
            "pump_power_kw": 0.0,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info


def create_microgrid_env_newHW(
    config: Dict[str, Any],
) -> NewHWMicrogridEnvironment:
    env = config["env"]
    if bool(env.get("allow_grid_trading", False)):
        raise ValueError("newHW is off-grid; allow_grid_trading must be false")
    if bool(env.get("use_flow_rate_action", False)):
        raise ValueError("newHW LFP has no pump; use_flow_rate_action must be false")
    return NewHWMicrogridEnvironment(
        dataset_csv_path=env["dataset_csv_path"],
        episode_length=env["episode_length"],
        time_step=env.get("time_step", 0.25),
        battery_capacity_kwh=env["battery_capacity_kwh"],
        battery_charge_power_kw=env["battery_charge_power_kw"],
        battery_discharge_power_kw=env["battery_discharge_power_kw"],
        battery_efficiency=env.get("battery_efficiency", 0.95),
        soc_min=env["soc_min"],
        soc_max=env["soc_max"],
        initial_soc=env["initial_soc"],
        soc_init_mode=env.get("soc_init_mode", "fixed"),
        reward=config.get("reward_newHW", {}),
    )

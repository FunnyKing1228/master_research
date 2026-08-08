from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = ROOT / "core"
BASELINE_DIR = ROOT / "data" / "scripts" / "baselines"
DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "p302" / "config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "heuristic_pv_scenarios"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from train_sac_microgrid import create_environment, get_power_limits  # type: ignore  # noqa: E402
from run_seminar_baseline_suite import (  # type: ignore  # noqa: E402
    current_episode_inputs,
    format_env_action,
    heuristic_action_kw,
)


SCENARIOS: Tuple[Tuple[str, str, float], ...] = (
    ("low", "Low PV support", 0.5),
    ("mid", "Mid PV support", 1.0),
    ("high", "High PV support", 1.5),
)

DEFAULT_START_HOUR_TIMES: Tuple[str, ...] = (
    "2026-05-01 00:00:00",
    "2026-05-01 06:00:00",
    "2026-05-01 12:00:00",
    "2026-05-01 18:00:00",
)
SITUATIONAL_SUFFIX = "_situational_headroom"
POLICY_SUFFIXES = {
    "legacy": "",
    "situational": SITUATIONAL_SUFFIX,
    "safety": "_safety",
    "profit": "_profit",
    "balanced": "_balanced",
}


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    dataset_path = ROOT / config["env"]["dataset_csv_path"]
    time_col = config["env"].get("dataset_time_column", "timestamp")
    df = pd.read_csv(dataset_path, parse_dates=[time_col])
    return df.sort_values(time_col).reset_index(drop=True)


def timestamp_to_index(df: pd.DataFrame, timestamp: str, time_col: str) -> int:
    target = pd.Timestamp(timestamp)
    matches = df.index[df[time_col] == target].tolist()
    if not matches:
        raise ValueError(f"Start timestamp not found in dataset: {timestamp}")
    return int(matches[0])


def refresh_scaled_pv_state(env: Any, multiplier: float) -> None:
    base_pv = np.asarray(env.episode_data["pv"], dtype=float)
    env.episode_data["pv"] = np.maximum(base_pv * float(multiplier), 0.0)
    if hasattr(env, "_refresh_episode_pv_bool"):
        env._refresh_episode_pv_bool()


def future_pv_support_stats(env: Any, step: int, window_steps: int) -> Tuple[float, float]:
    episode_data = getattr(env, "episode_data", {}) or {}
    load_arr = np.asarray(episode_data.get("load", []), dtype=float)
    pv_arr = np.asarray(episode_data.get("pv", []), dtype=float)
    start = int(step) + 1
    end = min(len(load_arr), len(pv_arr), start + int(window_steps))
    if start >= end:
        return 0.0, 0.0
    ratios = np.divide(
        pv_arr[start:end],
        np.clip(load_arr[start:end], 1e-9, None),
        out=np.zeros(end - start, dtype=float),
        where=np.ones(end - start, dtype=bool),
    )
    ratios = np.clip(ratios, 0.0, float(getattr(env, "pv_support_ratio_max", 1.5)))
    return float(np.max(ratios)), float(np.mean(ratios))


def explain_heuristic_gate(
    state: np.ndarray,
    env: Any,
    price: float,
    load_kw: float,
    pv_kw: float,
    future_pv_support_max: float,
    future_pv_support_mean: float,
    heuristic_policy: str,
) -> Dict[str, Any]:
    soc = float(state[0])
    charge_limit_kw, discharge_limit_kw = get_power_limits(env)
    pv_support_ratio = pv_kw / max(load_kw, 1e-9)
    pv_active = pv_kw > float(getattr(env, "pv_obs_boolean_threshold_kw", 0.001))
    is_peak = price >= 7.0
    is_mid_or_peak = price >= 4.69
    is_offpeak = price <= 2.2
    policy = {
        "legacy": "balanced",
        "situational": "balanced",
    }.get(heuristic_policy, heuristic_policy)
    can_solo_discharge = (not pv_active) and load_kw <= abs(discharge_limit_kw)
    discharge_price_gate = is_mid_or_peak if policy == "profit" else is_peak
    peak_soc_floor = 0.24 if policy == "profit" else 0.34 if policy == "safety" else 0.30
    soc_min_target = 0.22 if policy == "profit" else 0.26
    if can_solo_discharge:
        dt_h = float(getattr(env, "time_step", 0.25))
        capacity_kwh = float(getattr(env, "battery_capacity_kwh", 0.0))
        eta = float(getattr(env, "battery_efficiency", 0.95))
        peak_soc_next = soc - load_kw * dt_h / max(eta * capacity_kwh, 1e-9)
    else:
        peak_soc_next = soc
    headroom_target_soc = 0.34
    headroom_min_soc = 0.35
    current_pv_low = pv_support_ratio <= 0.20
    future_pv_rises = (
        future_pv_support_max >= 0.80
        and future_pv_support_mean >= 0.25
        and future_pv_support_max >= pv_support_ratio + 0.50
    )
    if can_solo_discharge:
        headroom_safe_kw = max(0.0, (soc - headroom_target_soc) * capacity_kwh * eta / max(dt_h, 1e-9))
    else:
        headroom_safe_kw = 0.0
    can_headroom_discharge = (
        heuristic_policy == "situational"
        and (not is_peak)
        and price <= 4.69
        and current_pv_low
        and future_pv_rises
        and can_solo_discharge
        and soc > headroom_min_soc
        and headroom_safe_kw > max(float(getattr(env, "action_dead_zone_kw", 0.0)), 1e-9)
    )

    if pv_active and discharge_price_gate:
        reason = "pv_active_blocks_discharge"
    elif soc <= float(getattr(env, "voltage_cutoff_soc", 0.20)):
        reason = "voltage_cutoff_recovery_charge" if is_offpeak else "voltage_cutoff_standby"
    elif soc <= 0.22:
        reason = "low_soc_offpeak_recovery_charge" if is_offpeak else "low_soc_reserve_no_discharge"
    elif pv_support_ratio >= (1.05 if policy == "safety" else 0.95 if policy == "profit" else 1.00) and soc < (0.72 if policy == "safety" else 0.78 if policy == "profit" else 0.74):
        reason = "pv_support_charge_window"
    elif (
        discharge_price_gate
        and soc > peak_soc_floor
        and can_solo_discharge
        and peak_soc_next >= soc_min_target
    ):
        reason = "peak_discharge_allowed"
    elif discharge_price_gate and soc > peak_soc_floor and can_solo_discharge:
        reason = "peak_reserve_check_no_discharge"
    elif can_headroom_discharge:
        reason = "pre_solar_headroom_discharge"
    elif not is_peak:
        reason = "not_peak_price"
    elif soc <= 0.30:
        reason = "soc_reserve_for_peak_discharge"
    elif not can_solo_discharge:
        reason = "load_over_discharge_limit"
    else:
        reason = "standby_no_rule_matched"

    return {
        "heuristic_gate_reason": reason,
        "heuristic_is_peak": int(is_peak),
        "heuristic_is_mid_or_peak": int(is_mid_or_peak),
        "heuristic_is_offpeak": int(is_offpeak),
        "heuristic_can_solo_discharge": int(can_solo_discharge),
        "heuristic_pv_active_for_decision": int(pv_active),
        "heuristic_soc_for_decision": soc,
        "heuristic_pv_support_ratio_for_decision": float(np.clip(pv_support_ratio, 0.0, 1.5)),
        "future_pv_support_max": float(future_pv_support_max),
        "future_pv_support_mean": float(future_pv_support_mean),
        "heuristic_policy": heuristic_policy,
        "charge_limit_w": float(charge_limit_kw) * 1000.0,
        "discharge_limit_w": float(abs(discharge_limit_kw)) * 1000.0,
        "headroom_target_soc": headroom_target_soc,
    }


def rollout_scenario(
    base_config: Dict[str, Any],
    dataset_df: pd.DataFrame,
    scenario_key: str,
    scenario_label: str,
    pv_multiplier: float,
    start_timestamp: str,
    reset_seed: int,
    heuristic_policy: str = "legacy",
    future_window_steps: int = 32,
) -> pd.DataFrame:
    config = copy.deepcopy(base_config)
    max_steps = int(config.get("training", {}).get("max_steps", config["env"].get("episode_length", 96)))
    config["env"]["episode_length"] = max_steps
    config["training"]["max_steps"] = max_steps

    np.random.seed(reset_seed)
    env = create_environment(config)
    time_col = config["env"].get("dataset_time_column", "timestamp")
    start_idx = timestamp_to_index(dataset_df, start_timestamp, time_col)
    env.fixed_start_idx = start_idx
    state, reset_info = env.reset(seed=reset_seed)
    start_soc = float(reset_info.get("soc", env.current_soc))
    refresh_scaled_pv_state(env, pv_multiplier)
    state = env._get_state() if hasattr(env, "_get_state") else state

    prev_soc_violations = 0
    rows: List[Dict[str, Any]] = []
    for step in range(max_steps):
        timestamp = dataset_df.loc[start_idx + step, time_col]
        info_now = current_episode_inputs(env)
        future_max, future_mean = future_pv_support_stats(env, step, future_window_steps)
        gate_info = explain_heuristic_gate(
            state,
            env,
            price=info_now["price"],
            load_kw=info_now["load"],
            pv_kw=info_now["pv"],
            future_pv_support_max=future_max,
            future_pv_support_mean=future_mean,
            heuristic_policy=heuristic_policy,
        )
        action_kw = heuristic_action_kw(
            state,
            env,
            price=info_now["price"],
            load_kw=info_now["load"],
            pv_kw=info_now["pv"],
            future_pv_support_max=future_max,
            future_pv_support_mean=future_mean,
            enable_headroom_discharge=heuristic_policy == "situational",
            policy=heuristic_policy,
        )
        action = format_env_action(action_kw, env)
        next_state, reward, terminated, truncated, step_info = env.step(action)

        current_violations = int(step_info.get("soc_violations", 0))
        realized = max(0, current_violations - prev_soc_violations)
        prev_soc_violations = current_violations

        load_kw = float(step_info.get("load", info_now["load"]))
        pv_kw = float(step_info.get("pv", info_now["pv"]))
        pv_support_ratio = float(np.clip(pv_kw / max(load_kw, 1e-9), 0.0, 1.5))
        grid_import_kw = float(step_info.get("grid_import_kw", step_info.get("grid_kw", 0.0)))
        grid_export_kw = float(step_info.get("grid_export_kw", 0.0))
        blocked_flags = [
            "voltage_blocked",
            "blocked_by_pv",
            "blocked_by_load",
            "guard_block_low_soc_discharge",
            "guard_block_pv_active_discharge",
            "guard_block_voltage_cutoff",
            "guard_block_load_over_discharge_limit",
        ]
        invalid_discharge_blocked = int(any(bool(step_info.get(flag, 0)) for flag in blocked_flags))

        rows.append(
            {
                "scenario": scenario_key,
                "scenario_label": scenario_label,
                "pv_multiplier": float(pv_multiplier),
                "timestamp": timestamp,
                "step": step,
                "hour": step * float(getattr(env, "time_step", 0.25)),
                "price": float(step_info.get("price", info_now["price"])),
                "load_w": load_kw * 1000.0,
                "pv_support_w": pv_kw * 1000.0,
                "pv_support_ratio": pv_support_ratio,
                "grid_import_w": grid_import_kw * 1000.0,
                "grid_export_w": grid_export_kw * 1000.0,
                "grid_draw_w": float(step_info.get("grid_kw", grid_import_kw)) * 1000.0,
                "baseline_grid_w": float(step_info.get("baseline_grid_kw", 0.0)) * 1000.0,
                "action_raw_w": float(action_kw) * 1000.0,
                "action_applied_w": float(step_info.get("applied_action_kw", action_kw)) * 1000.0,
                "soc": float(step_info.get("current_soc", next_state[0])),
                "reward": float(reward),
                "total_revenue": float(step_info.get("total_revenue", 0.0)),
                "total_cost": float(step_info.get("total_cost", 0.0)),
                "net_profit": float(step_info.get("total_revenue", 0.0)) - float(step_info.get("total_cost", 0.0)),
                "pv_to_load_w": float(step_info.get("pv_to_load", 0.0)) * 1000.0,
                "pv_to_battery_w": float(step_info.get("pv_to_battery", 0.0)) * 1000.0,
                "useful_discharge_w": float(step_info.get("useful_discharge", 0.0)) * 1000.0,
                "violations_realized": int(realized),
                "invalid_discharge_blocked": invalid_discharge_blocked,
                "situation_code": int(step_info.get("situation_code", 4)),
                "flow_action": float(step_info.get("flow_action", 0.0)),
                "flow_power_limited": int(step_info.get("flow_power_limited", 0)),
                "flow_too_low_active": int(step_info.get("flow_too_low_active", 0)),
                "flow_power_mismatch": int(step_info.get("flow_power_mismatch", 0)),
                "pump_power_w": float(step_info.get("pump_power_kw", 0.0)) * 1000.0,
                "start_soc": start_soc,
                **gate_info,
            }
        )

        state = next_state
        if terminated or truncated:
            break

    return pd.DataFrame.from_records(rows)


def summarize_rollout(df: pd.DataFrame, dt_h: float) -> Dict[str, Any]:
    last = df.iloc[-1]
    return {
        "scenario": str(last["scenario"]),
        "scenario_label": str(last["scenario_label"]),
        "pv_multiplier": float(last["pv_multiplier"]),
        "steps": int(len(df)),
        "start_time": str(df["timestamp"].iloc[0]),
        "start_soc": float(df["start_soc"].iloc[0]),
        "end_soc": float(df["soc"].iloc[-1]),
        "soc_min": float(df["soc"].min()),
        "soc_max": float(df["soc"].max()),
        "total_revenue_twd": float(last["total_revenue"]),
        "total_cost_twd": float(last["total_cost"]),
        "net_profit_twd": float(last["net_profit"]),
        "grid_import_kwh": float(df["grid_import_w"].clip(lower=0.0).sum() * dt_h / 1000.0),
        "grid_export_kwh": float(df["grid_export_w"].clip(lower=0.0).sum() * dt_h / 1000.0),
        "baseline_grid_kwh": float(df["baseline_grid_w"].clip(lower=0.0).sum() * dt_h / 1000.0),
        "pv_support_kwh": float(df["pv_support_w"].clip(lower=0.0).sum() * dt_h / 1000.0),
        "pv_to_load_kwh": float(df["pv_to_load_w"].clip(lower=0.0).sum() * dt_h / 1000.0),
        "pv_to_battery_kwh": float(df["pv_to_battery_w"].clip(lower=0.0).sum() * dt_h / 1000.0),
        "useful_discharge_kwh": float(df["useful_discharge_w"].clip(lower=0.0).sum() * dt_h / 1000.0),
        "avg_pv_support_ratio": float(df["pv_support_ratio"].mean()),
        "max_pv_support_ratio": float(df["pv_support_ratio"].max()),
        "violations_realized": int(df["violations_realized"].sum()),
        "invalid_discharge_blocked": int(df["invalid_discharge_blocked"].sum()),
        "flow_power_limited": int(df["flow_power_limited"].sum()),
        "flow_too_low_active": int(df["flow_too_low_active"].sum()),
        "flow_power_mismatch": int(df["flow_power_mismatch"].sum()),
        "pump_energy_wh": float(df["pump_power_w"].clip(lower=0.0).sum() * dt_h),
    }


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_timeseries(rollouts: Dict[str, pd.DataFrame], out_path: Path) -> None:
    colors = {"low": "#4e79a7", "mid": "#59a14f", "high": "#f28e2b"}
    fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True, constrained_layout=True)

    for key, df in rollouts.items():
        color = colors.get(key, None)
        label = str(df["scenario_label"].iloc[0])
        x = pd.to_datetime(df["timestamp"])
        axes[0].plot(x, df["load_w"], color="#222222", linewidth=1.8, alpha=0.22)
        axes[0].plot(x, df["pv_support_w"], color=color, linewidth=2.0, label=f"{label}: PV support")
        axes[1].plot(x, df["pv_support_ratio"], color=color, linewidth=2.0, label=label)
        axes[2].step(x, df["action_applied_w"], where="mid", color=color, linewidth=1.7, label=label)
        axes[3].plot(x, df["soc"], color=color, linewidth=2.0, label=label)
        axes[4].step(x, df["grid_import_w"], where="mid", color=color, linewidth=1.7, label=f"{label}: import")
        if df["grid_export_w"].abs().sum() > 1e-9:
            axes[4].step(x, -df["grid_export_w"], where="mid", color=color, linewidth=1.2, linestyle="--", alpha=0.8)

    x0 = pd.to_datetime(next(iter(rollouts.values()))["timestamp"])
    load0 = next(iter(rollouts.values()))["load_w"]
    axes[0].plot(x0, load0, color="#222222", linewidth=2.2, linestyle="--", label="Load")
    axes[0].set_ylabel("Power (W)")
    axes[0].set_title("Heuristic rollout under scaled PV support scenarios")
    axes[0].legend(loc="upper left", ncols=2, frameon=True, framealpha=0.9)

    axes[1].axhline(0.8, color="#777777", linewidth=1.0, linestyle=":", label="PV sufficiency threshold")
    axes[1].set_ylabel("PV support\nratio")
    axes[1].set_ylim(0.0, 1.55)
    axes[1].legend(loc="upper left", ncols=2, frameon=True, framealpha=0.9)

    axes[2].axhline(0.0, color="#333333", linewidth=0.8)
    axes[2].set_ylabel("Battery\npower (W)")
    axes[2].legend(loc="upper left", ncols=3, frameon=True, framealpha=0.9)

    axes[3].axhline(0.20, color="#d62728", linewidth=1.0, linestyle="--", label="SoC bounds")
    axes[3].axhline(0.80, color="#d62728", linewidth=1.0, linestyle="--")
    axes[3].set_ylabel("SoC")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].legend(loc="upper left", ncols=3, frameon=True, framealpha=0.9)

    axes[4].axhline(0.0, color="#333333", linewidth=0.8)
    axes[4].set_ylabel("Grid import\n(W)")
    axes[4].legend(loc="upper left", ncols=3, frameon=True, framealpha=0.9)
    axes[4].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 25, 3)))
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[4].set_xlabel("Time of day")

    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = metrics_df["scenario_label"].tolist()
    colors = ["#4e79a7", "#59a14f", "#f28e2b"]

    axes[0, 0].bar(labels, metrics_df["net_profit_twd"], color=colors)
    axes[0, 0].set_ylabel("TWD")
    axes[0, 0].set_title("Net profit / grid cost savings")

    axes[0, 1].bar(labels, metrics_df["grid_import_kwh"], color=colors)
    axes[0, 1].set_ylabel("kWh")
    axes[0, 1].set_title("Grid import")

    axes[1, 0].bar(labels, metrics_df["pv_to_battery_kwh"], color=colors)
    axes[1, 0].set_ylabel("kWh")
    axes[1, 0].set_title("PV to battery")

    axes[1, 1].bar(labels, metrics_df["invalid_discharge_blocked"], color=colors)
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("Blocked invalid discharge attempts")

    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=15)

    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_start_hour_validation(rollouts: Dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    palette = ["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1"]

    for idx, (key, df) in enumerate(rollouts.items()):
        color = palette[idx % len(palette)]
        label = str(df["scenario_label"].iloc[0])
        x = df["hour"].astype(float)
        axes[0].plot(x, df["soc"], color=color, linewidth=2.0, label=label)
        axes[1].step(x, df["action_applied_w"], where="mid", color=color, linewidth=1.6, label=label)
        axes[2].step(x, df["grid_import_w"], where="mid", color=color, linewidth=1.6, label=label)
        axes[3].plot(x, df["net_profit"], color=color, linewidth=1.8, label=label)

    axes[0].axhline(0.20, color="#d62728", linewidth=1.0, linestyle="--", label="SoC bounds")
    axes[0].axhline(0.80, color="#d62728", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("SoC")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Battery\npower (W)")

    axes[2].set_ylabel("Grid import\n(W)")
    axes[3].set_ylabel("Net profit\n(TWD)")
    axes[3].set_xlabel("Hours since rollout start")

    for ax in axes:
        ax.legend(loc="upper left", ncols=2, frameon=True, framealpha=0.9)

    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_start_hour_validation(args: argparse.Namespace, config: Dict[str, Any], dataset_df: pd.DataFrame) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_h = float(config["env"].get("time_step", 0.25))
    starts = tuple(args.start_hour_validation)
    multiplier = float(args.start_hour_pv_multiplier)
    suffix = POLICY_SUFFIXES.get(args.heuristic_policy, f"_{args.heuristic_policy}")

    rollouts: Dict[str, pd.DataFrame] = {}
    metrics: List[Dict[str, Any]] = []
    all_rows: List[pd.DataFrame] = []
    for start in starts:
        stamp = pd.Timestamp(start)
        key = f"start_{stamp.hour:02d}"
        label = f"Start {stamp.strftime('%H:%M')}"
        df = rollout_scenario(
            config,
            dataset_df,
            scenario_key=key,
            scenario_label=label,
            pv_multiplier=multiplier,
            start_timestamp=start,
            reset_seed=args.seed,
            heuristic_policy=args.heuristic_policy,
            future_window_steps=args.future_window_steps,
        )
        df["rollout_start"] = start
        rollouts[key] = df
        all_rows.append(df)
        row = summarize_rollout(df, dt_h=dt_h)
        row["rollout_start"] = start
        metrics.append(row)

    rollout_df = pd.concat(all_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)
    rollout_df.to_csv(out_dir / f"start_hour_rollouts_mid_pv{suffix}.csv", index=False)
    metrics_df.to_csv(out_dir / f"start_hour_metrics_mid_pv{suffix}.csv", index=False)
    with open(out_dir / f"start_hour_metrics_mid_pv{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_start_hour_validation(rollouts, out_dir / f"heuristic_start_hour_validation_mid_pv{suffix}.png")
    plot_metrics(metrics_df, out_dir / f"heuristic_start_hour_metrics_mid_pv{suffix}.png")

    print(f"Saved start-hour validation outputs to: {out_dir}")
    print(metrics_df.to_string(index=False))


def parse_scenarios(values: Iterable[str]) -> Tuple[Tuple[str, str, float], ...]:
    parsed: List[Tuple[str, str, float]] = []
    for value in values:
        key, multiplier = value.split(":", 1)
        key = key.strip().lower()
        parsed.append((key, f"{key.title()} PV support", float(multiplier)))
    return tuple(parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule heuristic under scaled low/mid/high PV scenarios.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", default="2026-05-01 00:00:00")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario as key:pv_multiplier. Repeatable. Default: low:0.5 mid:1.0 high:1.5",
    )
    parser.add_argument(
        "--start-hour-validation",
        nargs="*",
        default=None,
        metavar="TIMESTAMP",
        help="Run 24h rollouts from the given timestamps instead of the low/mid/high PV suite.",
    )
    parser.add_argument(
        "--start-hour-pv-multiplier",
        type=float,
        default=1.0,
        help="PV multiplier used for --start-hour-validation.",
    )
    parser.add_argument(
        "--heuristic-policy",
        choices=["legacy", "situational", "safety", "profit", "balanced"],
        default="balanced",
        help="Use one of the greedy heuristic policies. legacy/situational are retained as balanced aliases.",
    )
    parser.add_argument(
        "--future-window-steps",
        type=int,
        default=32,
        help="Lookahead steps used by situational headroom heuristic.",
    )
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()
    config = load_config(args.config)
    dataset_df = load_dataset(config)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.start_hour_validation is not None:
        if len(args.start_hour_validation) == 0:
            args.start_hour_validation = list(DEFAULT_START_HOUR_TIMES)
        run_start_hour_validation(args, config, dataset_df)
        return

    scenarios = parse_scenarios(args.scenario) if args.scenario else SCENARIOS
    dt_h = float(config["env"].get("time_step", 0.25))
    suffix = POLICY_SUFFIXES.get(args.heuristic_policy, f"_{args.heuristic_policy}")

    rollouts: Dict[str, pd.DataFrame] = {}
    metrics: List[Dict[str, Any]] = []
    for key, label, multiplier in scenarios:
        df = rollout_scenario(
            config,
            dataset_df,
            scenario_key=key,
            scenario_label=label,
            pv_multiplier=multiplier,
            start_timestamp=args.start,
            reset_seed=args.seed,
            heuristic_policy=args.heuristic_policy,
            future_window_steps=args.future_window_steps,
        )
        rollouts[key] = df
        df.to_csv(out_dir / f"rollout_{key}{suffix}.csv", index=False)
        metrics.append(summarize_rollout(df, dt_h=dt_h))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / f"metrics_summary{suffix}.csv", index=False)
    with open(out_dir / f"metrics_summary{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_timeseries(rollouts, out_dir / f"heuristic_pv_scenarios_timeseries{suffix}.png")
    plot_metrics(metrics_df, out_dir / f"heuristic_pv_scenarios_metrics{suffix}.png")

    print(f"Saved outputs to: {out_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

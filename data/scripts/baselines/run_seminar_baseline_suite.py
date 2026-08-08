"""Run and summarize the seminar P302 baseline suite.

The script intentionally keeps all shared environment settings frozen from
``configs/experiments/p302/config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml`` and only
changes the algorithm/safety mechanism fields for each baseline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "core"
DEFAULT_BASE_CONFIG = ROOT / "configs" / "experiments" / "p302" / "config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml"
BASE_CONFIG = DEFAULT_BASE_CONFIG
CONFIG_DIR = ROOT / "configs" / "baselines" / "research"
OUTPUT_DIR = ROOT / "experiments" / "seminar_baseline_results"
TRAIN_SCRIPT = ROOT / "core" / "train_sac_microgrid.py"

if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from train_sac_microgrid import create_environment, get_power_limits  # noqa: E402


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    label: str
    kind: str
    phase: str
    variant: Optional[str] = None
    beta_occ: float = 0.0
    safetynet_warmup_episodes: int = 0
    attempted_violation_penalty: float = 0.0
    safety_projection_penalty: float = 0.0
    notes: str = ""


BASELINES: List[BaselineSpec] = [
    BaselineSpec(
        key="heuristic_safety",
        label="Safety-first greedy",
        kind="heuristic",
        phase="main",
        variant="safety",
        notes="Conservative non-learning EMS reference.",
    ),
    BaselineSpec(
        key="heuristic_profit",
        label="Profit-first greedy",
        kind="heuristic",
        phase="main",
        variant="profit",
        notes="Greedy TOU/grid-demand heuristic with hard hardware guards.",
    ),
    BaselineSpec(
        key="heuristic_balanced",
        label="Balanced greedy",
        kind="heuristic",
        phase="main",
        variant="balanced",
        notes="Profit-oriented greedy policy with additional safety margins.",
    ),
    BaselineSpec(
        key="sac_raw",
        label="SAC",
        kind="train",
        phase="main",
        variant="sac",
        notes="Pure SAC without projection or extra attempted-violation shaping.",
    ),
    BaselineSpec(
        key="sac_penalty",
        label="SAC + reward safety penalty",
        kind="train",
        phase="main",
        variant="sac_penalty",
        attempted_violation_penalty=0.25,
        notes="Reward-only safety shaping, no SafetyNet projection.",
    ),
    BaselineSpec(
        key="sac_train_safetynet",
        label="SAC + SafetyNet projection",
        kind="train",
        phase="main",
        variant="sac_sn",
        safety_projection_penalty=0.025,
        attempted_violation_penalty=0.10,
        notes="Shielded RL baseline without OCC.",
    ),
    BaselineSpec(
        key="sac_sn_occ",
        label="SAC + SafetyNet + OCC",
        kind="train",
        phase="ablation",
        variant="sac_sn",
        beta_occ=0.8,
        safety_projection_penalty=0.025,
        attempted_violation_penalty=0.10,
        notes="Ablation with OCC but no long curriculum warmup.",
    ),
    BaselineSpec(
        key="ours_full",
        label="CORAL",
        kind="train",
        phase="main",
        variant="sac_sn",
        beta_occ=0.8,
        safetynet_warmup_episodes=250,
        safety_projection_penalty=0.025,
        attempted_violation_penalty=0.10,
        notes="CORAL full P302 method settings.",
    ),
]


def load_base_config() -> Dict[str, Any]:
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scale_warmup(spec: BaselineSpec, episodes: int) -> int:
    if spec.safetynet_warmup_episodes <= 0:
        return 0
    return min(spec.safetynet_warmup_episodes, max(0, episodes // 4))


def build_config(spec: BaselineSpec, episodes: int, smoke: bool, seed: int = 42) -> Dict[str, Any]:
    cfg = copy.deepcopy(load_base_config())
    cfg["random_seed"] = int(seed)
    cfg["training"]["total_episodes"] = int(episodes)
    cfg["training"]["max_steps"] = int(
        cfg.get("training", {}).get(
            "max_steps",
            cfg.get("env", {}).get("episode_length", 96),
        )
    )
    cfg["training"]["eval_every"] = max(1, min(20, int(episodes)))
    cfg["training"]["eval_episodes"] = 1 if smoke else int(cfg["training"].get("eval_episodes", 3))
    cfg["training"]["save_every"] = max(1, min(50, int(episodes)))
    cfg["training"]["variant"] = spec.variant or "sac"
    cfg["training"]["safetynet_warmup_episodes"] = scale_warmup(spec, episodes)
    cfg["sac"]["beta_occ"] = float(spec.beta_occ)
    cfg["reward"]["attempted_violation_penalty"] = float(spec.attempted_violation_penalty)
    cfg["reward"]["safety_projection_penalty"] = float(spec.safety_projection_penalty)
    cfg["guided_teacher"]["enabled"] = False
    cfg["guided_teacher"]["demo_episodes"] = 0
    cfg["logging"]["plot_results"] = False
    cfg["logging"]["save_models"] = True
    cfg["logging"]["save_metrics"] = True
    cfg["logging"]["csv_per_episode"] = True

    if smoke:
        cfg["sac"]["batch_size"] = 32
        cfg["sac"]["warmup_steps"] = 32
        cfg["sac"]["buffer_size"] = 5000
        cfg["sac"]["update_every"] = 16
        cfg["logging"]["log_interval"] = 1
    return cfg


def run_label(stage: str, tag: str = "") -> str:
    clean_tag = str(tag).strip().replace(" ", "_")
    return stage if not clean_tag else f"{stage}_{clean_tag}"


def write_configs(episodes: int, smoke: bool, seed: int = 42, tag: str = "") -> List[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for spec in BASELINES:
        if spec.kind != "train":
            continue
        cfg = build_config(spec, episodes=episodes, smoke=smoke, seed=seed)
        suffix = run_label("smoke" if smoke else "main", tag)
        path = CONFIG_DIR / f"{suffix}_{spec.key}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        paths.append(path)
    return paths


def experiment_name(stage: str, spec: BaselineSpec, tag: str = "") -> str:
    return f"seminar_{run_label(stage, tag)}_{spec.key}"


def has_completed_training(exp_name: str) -> bool:
    log_path = ROOT / "experiments" / exp_name / "logs" / "episode_log.csv"
    model_path = ROOT / "experiments" / exp_name / "models" / "best_sac_model.pth"
    return log_path.exists() and model_path.exists()


def run_train(spec: BaselineSpec, stage: str, episodes: int, smoke: bool, force: bool = False, tag: str = "") -> None:
    if spec.kind != "train":
        return
    cfg_path = CONFIG_DIR / f"{run_label('smoke' if smoke else 'main', tag)}_{spec.key}.yaml"
    exp_name = experiment_name(stage, spec, tag=tag)
    if has_completed_training(exp_name) and not force:
        print(f"[skip] {exp_name} already has logs and model")
        return
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(cfg_path),
        "--episodes",
        str(episodes),
        "--name",
        exp_name,
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_training_stage(
    stage: str,
    episodes: int,
    include_ablation: bool,
    force: bool = False,
    seed: int = 42,
    tag: str = "",
) -> None:
    smoke = stage == "smoke"
    write_configs(episodes=episodes, smoke=smoke, seed=seed, tag=tag)
    wanted = [b for b in BASELINES if b.kind == "train" and (include_ablation or b.phase == "main")]
    for spec in wanted:
        run_train(spec, stage=stage, episodes=episodes, smoke=smoke, force=force, tag=tag)
    write_heuristic_result(stage=run_label(stage, tag), include_ablation=include_ablation)


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    dataset_path = ROOT / cfg["env"]["dataset_csv_path"]
    df = pd.read_csv(dataset_path, parse_dates=[cfg["env"].get("dataset_time_column", "timestamp")])
    return df.sort_values(cfg["env"].get("dataset_time_column", "timestamp")).reset_index(drop=True)


def heuristic_action_kw(
    state: np.ndarray,
    env: Any,
    price: float,
    load_kw: float,
    pv_kw: float,
    future_pv_support_max: Optional[float] = None,
    future_pv_support_mean: Optional[float] = None,
    enable_headroom_discharge: bool = False,
    policy: str = "balanced",
) -> float:
    del future_pv_support_max, future_pv_support_mean, enable_headroom_discharge
    policy = {
        "legacy": "balanced",
        "situational": "balanced",
        "rule": "balanced",
    }.get(str(policy).lower(), str(policy).lower())
    if policy not in {"safety", "profit", "balanced"}:
        raise ValueError(f"Unknown heuristic policy: {policy}")

    soc = float(state[0])
    charge_limit_kw, discharge_limit_kw = get_power_limits(env)
    charge_limit_kw = max(0.0, float(charge_limit_kw))
    discharge_limit_kw = abs(float(discharge_limit_kw))
    pv_ratio = pv_kw / max(load_kw, 1e-9)
    pv_active = pv_kw > float(getattr(env, "pv_obs_boolean_threshold_kw", 0.001))
    pv_surplus_kw = max(0.0, pv_kw - load_kw)
    is_peak = price >= 7.0
    is_offpeak = price <= 2.2
    is_mid_or_peak = price >= 4.69

    dt_h = float(getattr(env, "time_step", 0.25))
    capacity_kwh = float(getattr(env, "battery_capacity_kwh", 0.0))
    eta = float(getattr(env, "battery_efficiency", 0.95))
    action_deadband_kw = max(float(getattr(env, "action_dead_zone_kw", 0.0)), 1e-9)

    if policy == "safety":
        soc_min_target = 0.26
        soc_max_target = 0.72
        peak_soc_floor = 0.34
        pv_charge_ratio = 1.05
        pv_charge_fraction = 0.30
        offpeak_recovery_target = 0.35
        offpeak_charge_fraction = 0.25
        discharge_price_gate = is_peak
    elif policy == "profit":
        soc_min_target = 0.22
        soc_max_target = 0.78
        peak_soc_floor = 0.24
        pv_charge_ratio = 0.95
        pv_charge_fraction = 0.70
        offpeak_recovery_target = 0.74
        offpeak_charge_fraction = 0.60
        discharge_price_gate = is_mid_or_peak
    else:
        soc_min_target = 0.26
        soc_max_target = 0.74
        peak_soc_floor = 0.30
        pv_charge_ratio = 1.00
        pv_charge_fraction = 0.45
        offpeak_recovery_target = 0.55
        offpeak_charge_fraction = 0.35
        discharge_price_gate = is_peak

    def safe_charge_room_kw(target_soc: float) -> float:
        return max(0.0, (target_soc - soc) * capacity_kwh / max(dt_h * eta, 1e-9))

    def safe_discharge_room_kw(target_soc: float) -> float:
        return max(0.0, (soc - target_soc) * capacity_kwh * eta / max(dt_h, 1e-9))

    if soc <= float(getattr(env, "voltage_cutoff_soc", 0.20)):
        if is_offpeak and soc < offpeak_recovery_target:
            charge_kw = min(charge_limit_kw * offpeak_charge_fraction, safe_charge_room_kw(offpeak_recovery_target))
            return charge_kw if charge_kw > action_deadband_kw else 0.0
        return 0.0

    if soc >= soc_max_target:
        charge_allowed_kw = 0.0
    else:
        charge_allowed_kw = safe_charge_room_kw(soc_max_target)

    if pv_ratio >= pv_charge_ratio and charge_allowed_kw > action_deadband_kw:
        if pv_surplus_kw > float(getattr(env, "pv_surplus_threshold_kw", 0.0002)):
            charge_kw = min(charge_limit_kw, pv_surplus_kw, charge_allowed_kw)
        else:
            charge_kw = min(charge_limit_kw * pv_charge_fraction, charge_allowed_kw)
        if charge_kw > action_deadband_kw:
            return charge_kw

    if is_offpeak and soc < offpeak_recovery_target and charge_allowed_kw > action_deadband_kw:
        charge_kw = min(charge_limit_kw * offpeak_charge_fraction, safe_charge_room_kw(offpeak_recovery_target))
        if charge_kw > action_deadband_kw:
            return charge_kw

    has_effective_load = load_kw > action_deadband_kw
    can_solo_discharge = has_effective_load and load_kw <= discharge_limit_kw + 1e-9
    if (
        discharge_price_gate
        and (not pv_active)
        and can_solo_discharge
        and soc > peak_soc_floor
    ):
        discharge_room_kw = safe_discharge_room_kw(soc_min_target)
        discharge_kw = min(discharge_limit_kw, load_kw, discharge_room_kw)
        # In solo-only mode the command must be able to cover the load by itself.
        if discharge_kw >= load_kw * 0.99 and discharge_kw > action_deadband_kw:
            return -discharge_kw
    return 0.0


def heuristic_flow_fraction(action_kw: float, env: Any) -> float:
    if not getattr(env, "use_flow_rate_action", False):
        return 0.0
    if abs(action_kw) <= max(float(getattr(env, "action_dead_zone_kw", 0.0)), 1e-9):
        return float(getattr(env, "flow_idle_fraction", 0.0))

    if action_kw > 0:
        p_limit_kw = float(getattr(env, "battery_charge_power_kw", 0.0))
    else:
        p_limit_kw = float(getattr(env, "battery_discharge_power_kw", 0.0))
    required_flow = abs(float(action_kw)) / max(p_limit_kw, 1e-9)
    min_active = float(getattr(env, "flow_min_active_fraction", 0.0))
    return float(np.clip(max(min_active, required_flow), 0.0, 1.0))


def format_env_action(action_kw: float, env: Any) -> List[float]:
    if getattr(env, "use_flow_rate_action", False):
        return [float(action_kw), heuristic_flow_fraction(action_kw, env)]
    return [float(action_kw)]


def current_episode_inputs(env: Any) -> Dict[str, float]:
    """Read the current episode's aligned price/load/PV values."""
    step = int(getattr(env, "current_step", 0))
    episode_data = getattr(env, "episode_data", {}) or {}

    def read_value(key: str, fallback_attr: str, default: float = 0.0) -> float:
        series = episode_data.get(key)
        if series is not None and step < len(series):
            return float(series[step])
        fallback = getattr(env, fallback_attr, None)
        if fallback is not None and step < len(fallback):
            return float(fallback[step])
        return float(default)

    return {
        "price": read_value("price", "price_data", 0.0),
        "load": read_value("load", "load_data", 0.0),
        "pv": read_value("pv", "pv_data", 0.0),
    }


def rollout_heuristic(policy: str = "balanced") -> pd.DataFrame:
    cfg = load_base_config()
    max_steps = int(
        cfg.get("training", {}).get(
            "max_steps",
            cfg.get("env", {}).get("episode_length", 96),
        )
    )
    cfg["training"]["max_steps"] = max_steps
    env = create_environment(cfg)
    dataset = load_dataset(cfg)
    n_episodes = max(1, len(dataset) // max_steps)
    rows: List[Dict[str, Any]] = []

    for episode_idx in range(n_episodes):
        start_idx = episode_idx * max_steps
        env.fixed_start_idx = start_idx
        state, _ = env.reset(seed=42 + episode_idx)
        prev_soc_violations = 0
        timestamp = dataset.loc[start_idx, cfg["env"].get("dataset_time_column", "timestamp")]
        for step in range(max_steps):
            info_now = current_episode_inputs(env)
            action_kw = heuristic_action_kw(
                state,
                env,
                price=info_now["price"],
                load_kw=info_now["load"],
                pv_kw=info_now["pv"],
                policy=policy,
            )
            action = format_env_action(action_kw, env)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            current_violations = int(step_info.get("soc_violations", 0))
            realized = max(0, current_violations - prev_soc_violations)
            prev_soc_violations = current_violations
            rows.append(
                {
                    "episode": int(episode_idx),
                    "start_time": str(timestamp),
                    "step": step,
                    "reward": float(reward),
                    "soc": float(step_info.get("current_soc", next_state[0])),
                    "action_safe_w": float(step_info.get("applied_action_kw", action_kw)) * 1000.0,
                    "action_raw_w": float(action_kw) * 1000.0,
                    "violations_realized": realized,
                    "violations_attempted": 0,
                    "safety_projected_meaningful": 0,
                    "projection_delta_mean_w": 0.0,
                    "projection_delta_max_w": 0.0,
                    "revenue": float(step_info.get("total_revenue", 0.0)),
                    "cost": float(step_info.get("total_cost", 0.0)),
                    "net_profit": float(step_info.get("total_revenue", 0.0)) - float(step_info.get("total_cost", 0.0)),
                    "pv_to_battery_wh": float(step_info.get("pv_to_battery", 0.0)) * 250.0,
                    "useful_discharge_wh": float(step_info.get("useful_discharge", 0.0)) * 250.0,
                    "situation_code": int(step_info.get("situation_code", 4)),
                    "flow_action": float(step_info.get("flow_action", 0.0)),
                    "flow_power_limited": int(step_info.get("flow_power_limited", 0)),
                    "flow_too_low_active": int(step_info.get("flow_too_low_active", 0)),
                    "flow_power_mismatch": int(step_info.get("flow_power_mismatch", 0)),
                    "pump_power_wh": float(step_info.get("pump_power_kw", 0.0)) * 250.0,
                    "heuristic_policy": policy,
                }
            )
            state = next_state
            if terminated or truncated:
                break
    return pd.DataFrame(rows)


def heuristic_specs(include_ablation: bool = False) -> List[BaselineSpec]:
    return [b for b in BASELINES if b.kind == "heuristic" and (include_ablation or b.phase == "main")]


def write_heuristic_result(stage: str, include_ablation: bool = False) -> None:
    out_dir = OUTPUT_DIR / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    # Regenerate deterministic heuristic rollouts so bug fixes do not leave
    # stale baseline CSVs in place.
    for spec in heuristic_specs(include_ablation=include_ablation):
        policy = spec.variant or "balanced"
        path = out_dir / f"rollout_{spec.key}.csv"
        df = rollout_heuristic(policy=policy)
        df.to_csv(path, index=False)


def summarize_training_log(exp_name: str, label: str, last_n: int = 20) -> Optional[Dict[str, Any]]:
    log_path = ROOT / "experiments" / exp_name / "logs" / "episode_log.csv"
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    tail = df.tail(min(last_n, len(df)))
    def mean_or_nan(col: str) -> float:
        return float(tail[col].mean()) if col in tail.columns else math.nan

    return {
        "baseline": label,
        "experiment": exp_name,
        "episodes": int(len(df)),
        "source": "training_log_tail",
        "violations_realized": float(tail["violations_realized"].mean()),
        "violations_attempted": float(tail["violations_attempted"].mean()),
        "safety_projected_meaningful": float(tail["safety_projected_meaningful"].mean()),
        "projection_delta_mean_w": float(tail["projection_delta_mean_w"].mean()),
        "projection_delta_max_w": float(tail["projection_delta_max_w"].max()),
        "net_profit": float(tail["net_profit"].mean()),
        "pv_to_battery_wh": math.nan,
        "useful_discharge_wh": math.nan,
        "situation_1_count": math.nan,
        "flow_action_mean": mean_or_nan("flow_action_mean"),
        "flow_active_mean": mean_or_nan("flow_active_mean"),
        "flow_power_limited_count": mean_or_nan("flow_power_limited_count"),
        "flow_too_low_active_count": mean_or_nan("flow_too_low_active_count"),
        "flow_power_mismatch_count": mean_or_nan("flow_power_mismatch_count"),
        "pump_power_wh": mean_or_nan("pump_power_wh"),
    }


def summarize_heuristic(spec: BaselineSpec, stage: str, tag: str = "") -> Optional[Dict[str, Any]]:
    stage_dir = run_label(stage, tag)
    path = OUTPUT_DIR / stage_dir / f"rollout_{spec.key}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    episode_col = "episode" if "episode" in df.columns else "day"
    daily = df.groupby(episode_col, as_index=False).agg(
        violations_realized=("violations_realized", "sum"),
        violations_attempted=("violations_attempted", "sum"),
        safety_projected_meaningful=("safety_projected_meaningful", "sum"),
        projection_delta_mean_w=("projection_delta_mean_w", "mean"),
        projection_delta_max_w=("projection_delta_max_w", "max"),
        net_profit=("net_profit", "last"),
        pv_to_battery_wh=("pv_to_battery_wh", "sum"),
        useful_discharge_wh=("useful_discharge_wh", "sum"),
        situation_1_count=("situation_code", lambda s: int((s == 1).sum())),
        flow_action_mean=("flow_action", "mean"),
        flow_power_limited_count=("flow_power_limited", "sum"),
        flow_too_low_active_count=("flow_too_low_active", "sum"),
        flow_power_mismatch_count=("flow_power_mismatch", "sum"),
        pump_power_wh=("pump_power_wh", "sum"),
    )
    return {
        "baseline": spec.label,
        "experiment": f"seminar_{stage_dir}_{spec.key}",
        "episodes": int(len(daily)),
        "source": "heuristic_rollout_days",
        **{col: float(daily[col].mean()) for col in daily.columns if col != episode_col},
    }


def collect_summary(stage: str, include_ablation: bool, tag: str = "") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in heuristic_specs(include_ablation=include_ablation):
        h = summarize_heuristic(spec, stage, tag=tag)
        if h:
            rows.append(h)
    wanted = [b for b in BASELINES if b.kind == "train" and (include_ablation or b.phase == "main")]
    for spec in wanted:
        row = summarize_training_log(experiment_name(stage, spec, tag=tag), spec.label)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "baseline",
        "violations_attempted",
        "safety_projected_meaningful",
        "projection_delta_mean_w",
        "projection_delta_max_w",
        "net_profit",
        "pv_to_battery_wh",
        "useful_discharge_wh",
        "situation_1_count",
        "flow_action_mean",
        "flow_active_mean",
        "flow_power_limited_count",
        "flow_too_low_active_count",
        "flow_power_mismatch_count",
        "pump_power_wh",
    ]
    existing = [c for c in cols if c in df.columns]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Seminar Baseline Summary\n\n")
        f.write("| " + " | ".join(existing) + " |\n")
        f.write("| " + " | ".join(["---"] * len(existing)) + " |\n")
        for _, row in df[existing].iterrows():
            values = []
            for col in existing:
                value = row[col]
                if isinstance(value, (float, np.floating)) and not pd.isna(value):
                    values.append(f"{float(value):.4f}")
                elif pd.isna(value):
                    values.append("")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")
        f.write("\n")


def plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = df["baseline"].tolist()

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, df["violations_attempted"].fillna(0), width, label="attempted violations")
    ax.bar(x + width / 2, df["safety_projected_meaningful"].fillna(0), width, label="meaningful projections")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("events / episode")
    ax.set_title("Raw Policy Safety And SafetyNet Reliance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "raw_policy_safety.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax.bar(x, df["net_profit"].fillna(0))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("TWD / episode")
    ax.set_title("Deployment-Aware Economic Metric")
    fig.tight_layout()
    fig.savefig(out_dir / "deployment_profit.png", dpi=180)
    plt.close(fig)

    energy_cols = [c for c in ["pv_to_battery_wh", "useful_discharge_wh"] if c in df.columns]
    if energy_cols and df[energy_cols].notna().any().any():
        fig, ax = plt.subplots(figsize=(11, 5))
        bottom = np.zeros(len(df))
        x = np.arange(len(labels))
        for col in energy_cols:
            values = df[col].fillna(0).to_numpy()
            ax.bar(x, values, bottom=bottom, label=col)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Wh / episode")
        ax.set_title("PV Storage And Useful Discharge")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "deployment_energy_use.png", dpi=180)
        plt.close(fig)

    if "flow_active_mean" in df.columns and df["flow_active_mean"].notna().any():
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(labels))
        ax.bar(x, df["flow_active_mean"].fillna(0))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("active flow fraction")
        ax.set_title("Active Flow Fraction By Baseline")
        fig.tight_layout()
        fig.savefig(out_dir / "active_flow_fraction.png", dpi=180)
        plt.close(fig)


def generate_assets(stage: str, include_ablation: bool, tag: str = "") -> None:
    out_dir = OUTPUT_DIR / run_label(stage, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_heuristic_result(stage=run_label(stage, tag), include_ablation=include_ablation)
    df = collect_summary(stage, include_ablation=include_ablation, tag=tag)
    csv_path = out_dir / "baseline_summary.csv"
    md_path = out_dir / "baseline_summary.md"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_markdown_table(df, md_path)
    plot_summary(df, out_dir)
    print(f"[assets] wrote {csv_path}")
    print(f"[assets] wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seminar baseline suite.")
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--run-main", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--generate-assets", action="store_true")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--smoke-episodes", type=int, default=3)
    parser.add_argument("--stage", choices=["smoke", "main", "ablation"], default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="", help="Optional suffix for configs, experiments, and result folders")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base YAML used for all generated baseline variants",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated summary assets and heuristic rollouts",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    global BASE_CONFIG, OUTPUT_DIR
    args = parse_args()
    BASE_CONFIG = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    OUTPUT_DIR = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.write_configs:
        write_configs(episodes=args.episodes, smoke=False, seed=args.seed, tag=args.tag)
        write_configs(episodes=args.smoke_episodes, smoke=True, seed=args.seed, tag=args.tag)
        print(f"[configs] wrote templates to {CONFIG_DIR}")
    if args.run_smoke:
        run_training_stage("smoke", episodes=args.smoke_episodes, include_ablation=False, force=args.force, seed=args.seed, tag=args.tag)
        generate_assets("smoke", include_ablation=False, tag=args.tag)
    if args.run_main:
        run_training_stage("main", episodes=args.episodes, include_ablation=False, force=args.force, seed=args.seed, tag=args.tag)
        generate_assets("main", include_ablation=False, tag=args.tag)
    if args.run_ablation:
        run_training_stage("ablation", episodes=args.episodes, include_ablation=True, force=args.force, seed=args.seed, tag=args.tag)
        generate_assets("ablation", include_ablation=True, tag=args.tag)
    if args.generate_assets:
        generate_assets(args.stage, include_ablation=args.stage == "ablation", tag=args.tag)


if __name__ == "__main__":
    main()


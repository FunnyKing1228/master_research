"""Auditable in-sample rollout for provisional newHW checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import train_sac_microgrid as shared  # noqa: E402
from analyze_energy_bound_newHW import calculate_energy_bounds  # noqa: E402
from microgrid_env_newHW import create_microgrid_env_newHW  # noqa: E402
from safety_net import clear_residual_buffer, project as safety_project  # noqa: E402


def rollout(config: dict, model_path: Path) -> tuple[pd.DataFrame, dict]:
    env = create_microgrid_env_newHW(config)
    env.fixed_start_idx = 0
    state, _ = env.reset(seed=int(config.get("random_seed", 42)))
    device = "cpu"
    agent = shared.create_agent(
        config,
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        device=device,
    )
    agent.load(str(model_path))
    clear_residual_buffer()

    records: list[dict] = []
    previous_safe_kw = 0.0
    projected_count = 0
    attempted_count = 0
    for step in range(env.episode_length):
        idx = env.start_idx + env.current_step
        timestamp = pd.Timestamp(env.episode_data.iloc[idx]["timestamp"])
        soc_before = float(state[0])
        action_norm = agent.select_action(state, evaluate=True)
        raw_kw = shared.norm_to_power_kw(float(action_norm[0]), env)
        predicted_raw_soc = env.predict_soc_raw(soc_before, raw_kw)
        attempted = int(
            predicted_raw_soc < env.soc_min or predicted_raw_soc > env.soc_max
        )
        safe_kw, projected, projection_delta_kw = safety_project(
            state=state,
            action=np.array([raw_kw], dtype=np.float32),
            prev_action=previous_safe_kw,
            pmax=env.battery_power_kw,
            pmin=env.battery_discharge_power_kw,
            pmax_positive=env.battery_charge_power_kw,
            ramp_kw=None,
            soc_bounds=(env.soc_min, env.soc_max),
            env=env,
        )
        next_state, reward, terminated, truncated, info = env.step([safe_kw])
        projected_count += int(projected)
        attempted_count += attempted
        records.append(
            {
                "timestamp": timestamp,
                "soc_before": soc_before,
                "soc_after": float(info["current_soc"]),
                "load_w": float(info["load"]) * 1000.0,
                "pv_w": float(info["pv"]) * 1000.0,
                "raw_action_w": raw_kw * 1000.0,
                "safe_action_w": safe_kw * 1000.0,
                "applied_action_w": float(info["applied_action_kw"]) * 1000.0,
                "served_load_w": float(info["served_load_kw"]) * 1000.0,
                "unmet_load_w": float(info["unmet_load_kw"]) * 1000.0,
                "pv_curtailed_w": float(info["pv_curtailed_kw"]) * 1000.0,
                "attempted_soc_violation": attempted,
                "safety_projected": int(projected),
                "projection_delta_w": projection_delta_kw * 1000.0,
                "situation_code": int(info["situation_code"]),
                "reward": float(reward),
            }
        )
        state = next_state
        previous_safe_kw = safe_kw
        if terminated or truncated:
            break

    frame = pd.DataFrame(records)
    bound_summary, oracle_trace = calculate_energy_bounds(env.episode_data, config)
    for column in oracle_trace.columns:
        if column != "timestamp":
            frame[column] = oracle_trace[column].to_numpy()[: len(frame)]
    dt_h = float(env.time_step)
    load_kwh = float(frame["load_w"].sum() * dt_h / 1000.0)
    unmet_kwh = float(frame["unmet_load_w"].sum() * dt_h / 1000.0)
    served_fraction = float(1.0 - unmet_kwh / max(load_kwh, 1e-9))
    oracle_fraction = float(bound_summary["chronological_oracle_served_fraction"])
    cyclic_oracle_fraction = float(
        bound_summary["terminal_soc_neutral_oracle_served_fraction"]
    )
    summary = {
        "status": "IN_SAMPLE_ROLLOUT_NOT_GENERALIZATION_VALIDATION",
        "todo_marker": "TODO(newHW)",
        "model": str(model_path),
        "steps": int(len(frame)),
        "hours": float(len(frame) * dt_h),
        "load_energy_kwh": load_kwh,
        "unmet_load_kwh": unmet_kwh,
        "served_energy_fraction": served_fraction,
        "physical_bound_under_current_assumptions": bound_summary,
        "agent_fraction_of_chronological_oracle": served_fraction
        / max(oracle_fraction, 1e-9),
        "agent_served_energy_shortfall_vs_oracle_kwh": float(
            bound_summary["chronological_oracle_served_kwh"]
            - load_kwh * served_fraction
        ),
        "agent_fraction_of_terminal_soc_neutral_oracle": served_fraction
        / max(cyclic_oracle_fraction, 1e-9),
        "agent_served_energy_shortfall_vs_terminal_soc_neutral_oracle_kwh": float(
            bound_summary["terminal_soc_neutral_oracle_served_kwh"]
            - load_kwh * served_fraction
        ),
        "loss_of_load_step_fraction": float(
            (frame["unmet_load_w"] > 1e-6).mean()
        ),
        "loss_of_load_step_fraction_excess_vs_oracle": float(
            (frame["unmet_load_w"] > 1e-6).mean()
            - bound_summary["chronological_oracle_loss_of_load_step_fraction"]
        ),
        "rollout_initial_soc_assumption": float(config["env"]["initial_soc"]),
        "reward_definition": {
            "status": config["reward_newHW"]["objective_status"],
            "formula": (
                "served_load*served_fraction - unmet_load*unmet_fraction "
                "- low_soc_reserve*reserve_deficit - "
                "battery_throughput*throughput_fraction - "
                "pv_curtailment*curtailment_fraction"
            ),
            "weights": config["reward_newHW"],
        },
        "soc_min": float(frame["soc_after"].min()),
        "soc_max": float(frame["soc_after"].max()),
        "soc_end": float(frame["soc_after"].iloc[-1]),
        "attempted_soc_violation_steps": int(attempted_count),
        "safetynet_projection_steps": int(projected_count),
        "safetynet_projection_fraction": float(projected_count / max(len(frame), 1)),
        "environment_soc_violations": int(env.soc_violations),
        "situation_1_steps": int((frame["situation_code"] == 1).sum()),
        "situation_4_steps": int((frame["situation_code"] == 4).sum()),
        "situation_2_or_3_steps": int(frame["situation_code"].isin([2, 3]).sum()),
        "caveats": [
            "The same 47-hour data was used for training and rollout.",
            "No independent validation/test period exists.",
            "3-day and 5-day continuous rollout cannot be run.",
            "Reward, BMS limits, SoC limits, power limits and initial SoC are provisional.",
            "The 4.8 W standby state is not added separately to the fixed 28.2 W load.",
        ],
    }
    return frame, summary


def plot_rollout(frame: pd.DataFrame, summary: dict, output: Path) -> None:
    timestamps = pd.to_datetime(frame["timestamp"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "newHW IN-SAMPLE rollout — NOT A VALIDATION\n"
        f"agent={summary['served_energy_fraction']:.1%}, "
        f"finite/cyclic bounds="
        f"{summary['physical_bound_under_current_assumptions']['chronological_oracle_served_fraction']:.1%}/"
        f"{summary['physical_bound_under_current_assumptions']['terminal_soc_neutral_oracle_served_fraction']:.1%}, "
        f"loss-of-load steps={summary['loss_of_load_step_fraction']:.1%}, "
        f"SafetyNet projection={summary['safetynet_projection_fraction']:.1%}"
    )
    axes[0].plot(timestamps, frame["pv_w"], label="PV")
    axes[0].plot(timestamps, frame["load_w"], label="Load (28.2 W inferred)")
    axes[0].plot(timestamps, frame["served_load_w"], label="Served load")
    axes[0].plot(
        timestamps,
        frame["oracle_served_load_w"],
        linestyle="--",
        label="Oracle served load",
    )
    axes[0].fill_between(
        timestamps,
        0,
        frame["unmet_load_w"],
        alpha=0.35,
        color="#d62728",
        label="Unmet load",
    )
    axes[0].set_ylabel("Power (W)")
    axes[0].legend(ncols=4)

    axes[1].plot(timestamps, frame["soc_after"], label="Simulated SoC")
    axes[1].axhline(0.10, linestyle="--", color="#d62728", label="Provisional min")
    axes[1].axhline(0.90, linestyle="--", color="#d62728", label="Provisional max")
    axes[1].set_ylabel("SoC")
    axes[1].legend()

    axes[2].plot(timestamps, frame["raw_action_w"], label="Raw policy")
    axes[2].plot(timestamps, frame["safe_action_w"], label="SafetyNet")
    axes[2].plot(timestamps, frame["applied_action_w"], label="Applied")
    axes[2].set_ylabel("Battery action (W)")
    axes[2].legend(ncols=3)

    dt_h = 0.25
    axes[3].plot(
        timestamps,
        (frame["unmet_load_w"] * dt_h / 1000.0).cumsum(),
        label="Agent cumulative unmet load",
    )
    axes[3].plot(
        timestamps,
        (frame["oracle_unmet_load_w"] * dt_h / 1000.0).cumsum(),
        linestyle="--",
        label="Finite-window oracle unmet",
    )
    axes[3].plot(
        timestamps,
        (frame["cyclic_oracle_unmet_load_w"] * dt_h / 1000.0).cumsum(),
        linestyle=":",
        label="Terminal-SoC-neutral unmet",
    )
    axes[3].plot(
        timestamps,
        (frame["pv_curtailed_w"] * dt_h / 1000.0).cumsum(),
        label="Cumulative PV curtailment",
    )
    axes[3].set_ylabel("Energy (kWh)")
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auditable newHW in-sample rollout")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if not args.experiment.startswith("newHW_"):
        raise ValueError("Only newHW_* experiments are accepted")

    experiment_dir = ROOT / "experiments" / args.experiment
    config_path = experiment_dir / "configs" / "experiment_config.yaml"
    model_path = experiment_dir / "models" / args.model
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "config" in config and isinstance(config["config"], dict):
        config = config["config"]
    frame, summary = rollout(config, model_path)

    output_dir = experiment_dir / "results" / "in_sample_rollout_newHW"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem
    frame.to_csv(output_dir / f"{stem}_audit_newHW.csv", index=False)
    (output_dir / f"{stem}_summary_newHW.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_rollout(frame, summary, output_dir / f"{stem}_rollout_newHW.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

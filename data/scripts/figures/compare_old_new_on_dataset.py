from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "data" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_thesis_behavior_figures import load_config, load_dataset, rollout_episode  # type: ignore


MODEL_SPECS = [
    {
        "label": "old version",
        "experiment": "v16sp_guided_teacher_v10_0505_toufix_occfix",
        "model": "best_sac_model.pth",
    },
    {
        "label": "new",
        "experiment": "v16sp_no_teacher_v12_0505_s2block_deploysafe",
        "model": "final_sac_model.pth",
    },
]


def load_experiment_config(experiment: str) -> Dict[str, Any]:
    config_path = ROOT / "experiments" / experiment / "configs" / "experiment_config.yaml"
    config = load_config(str(config_path))
    if "config" in config and isinstance(config["config"], dict):
        config = config["config"]
    return config


def rollout_model(spec: Dict[str, str], date: str, dataset_path: str | None) -> pd.DataFrame:
    config = load_experiment_config(spec["experiment"])
    if dataset_path:
        config["env"]["dataset_csv_path"] = dataset_path
    dataset_df = load_dataset(config["env"]["dataset_csv_path"])
    model_path = ROOT / "experiments" / spec["experiment"] / "models" / spec["model"]
    df = rollout_episode(
        config,
        dataset_df,
        model_path=model_path,
        start_date=f"{date} 00:00:00",
        days=1,
        reset_seed=42,
    )
    df["model"] = spec["label"]
    df["projection_delta_w"] = (df["action_safe_w"] - df["action_raw_w"]).abs()
    return df


def summarize_rollout(df: pd.DataFrame) -> Dict[str, float | str]:
    dt_hr = 0.25
    baseline_grid_w = (df["load_w"] - df["pv_bus_w"]).clip(lower=0.0)
    grid_cost = ((df["grid_draw_w"] / 1000.0) * dt_hr * df["price"]).sum()
    baseline_cost = ((baseline_grid_w / 1000.0) * dt_hr * df["price"]).sum()
    return {
        "model": str(df["model"].iloc[0]),
        "soc_start": float(df["soc_start"].iloc[0]),
        "soc_end": float(df["soc"].iloc[-1]),
        "soc_min": float(df["soc"].min()),
        "soc_max": float(df["soc"].max()),
        "charge_steps": int((df["action_applied_w"] > 1e-6).sum()),
        "discharge_steps": int((df["action_applied_w"] < -1e-6).sum()),
        "pv_to_battery_wh": float(df["pv_to_battery_w"].clip(lower=0.0).sum() * dt_hr),
        "useful_discharge_wh": float(df["useful_discharge_w"].clip(lower=0.0).sum() * dt_hr),
        "grid_cost_twd": float(grid_cost),
        "baseline_cost_twd": float(baseline_cost),
        "grid_savings_twd": float(baseline_cost - grid_cost),
        "reward_sum": float(df["reward"].sum()),
        "projection_events": int((df["projection_delta_w"] > 0.05).sum()),
        "projection_delta_wh": float(df["projection_delta_w"].sum() * dt_hr),
        "projection_delta_mean_w": float(df["projection_delta_w"].mean()),
        "projection_delta_max_w": float(df["projection_delta_w"].max()),
    }


def summarize_training_log(spec: Dict[str, str], last_n: int = 100) -> Dict[str, float | str]:
    log_path = ROOT / "experiments" / spec["experiment"] / "logs" / "episode_log.csv"
    df = pd.read_csv(log_path)
    tail = df.tail(min(last_n, len(df)))
    fields = [
        "ep_reward",
        "violations_attempted",
        "violations_realized",
        "safety_projected_meaningful",
        "projection_delta_mean_w",
        "projection_delta_max_w",
        "revenue",
        "cost",
        "net_profit",
    ]
    out: Dict[str, float | str] = {"model": spec["label"], "last_n": len(tail)}
    for field in fields:
        if field in tail.columns:
            out[field] = float(tail[field].mean())
    return out


def plot_rollout_comparison(rollouts: List[pd.DataFrame], output_path: Path, title: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    colors = {"old version": "#d62728", "new": "#1f77b4"}
    fig, axes = plt.subplots(5, 1, figsize=(13.5, 13.2), sharex=True)
    ax_pv, ax_soc, ax_action, ax_correction, ax_grid = axes

    reference = rollouts[0].copy()
    timestamps = pd.to_datetime(reference["timestamp"])
    ax_pv.plot(timestamps, reference["load_w"], color="#333333", linewidth=1.8, label="load")
    ax_pv.plot(timestamps, reference["pv_bus_w"], color="#2ca02c", linewidth=2.0, label="observed bus PV")
    ax_pv.fill_between(timestamps, reference["pv_bus_w"], color="#2ca02c", alpha=0.16)
    ax_pv.set_ylabel("Power (W)")
    ax_pv.legend(loc="upper right")
    ax_pv.grid(True, linestyle="--", alpha=0.25)

    for df in rollouts:
        ts = pd.to_datetime(df["timestamp"])
        label = str(df["model"].iloc[0])
        color = colors[label]
        ax_soc.plot(ts, df["soc"], color=color, linewidth=2.0, label=label)
        ax_action.plot(
            ts,
            df["action_raw_w"],
            color=color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.75,
            label=f"{label} raw policy",
        )
        ax_action.plot(
            ts,
            df["action_safe_w"],
            color=color,
            linewidth=2.0,
            label=f"{label} after SafetyNet",
        )
        ax_correction.plot(ts, df["projection_delta_w"], color=color, linewidth=2.0, label=label)
        ax_correction.fill_between(ts, df["projection_delta_w"], color=color, alpha=0.18)
        ax_grid.plot(ts, df["grid_draw_w"], color=color, linewidth=1.9, label=label)

    ax_soc.axhline(0.2, color="#aa0000", linestyle="--", linewidth=1.0)
    ax_soc.axhline(0.8, color="#aa0000", linestyle="--", linewidth=1.0)
    ax_soc.set_ylabel("SoC")
    ax_soc.legend(loc="upper right")
    ax_soc.grid(True, linestyle="--", alpha=0.25)

    ax_action.axhline(0.0, color="#333333", linewidth=0.9)
    ax_action.set_ylabel("Battery Action (W)")
    ax_action.legend(loc="upper right", ncol=2)
    ax_action.grid(True, linestyle="--", alpha=0.25)

    ax_correction.axhline(0.05, color="#aa0000", linestyle="--", linewidth=1.0, label="meaningful threshold")
    ax_correction.set_ylabel("SafetyNet\nCorrection (W)")
    ax_correction.legend(loc="upper right")
    ax_correction.grid(True, linestyle="--", alpha=0.25)

    ax_grid.set_ylabel("Grid Draw (W)")
    ax_grid.set_xlabel("Time")
    ax_grid.legend(loc="upper right")
    ax_grid.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(title, y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def plot_safety_summary(
    training_summary: pd.DataFrame,
    rollout_summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 12,
        }
    )
    colors = {"old version": "#d62728", "new": "#1f77b4"}
    labels = ["old version", "new"]
    bar_colors = [colors[label] for label in labels]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    ax_violation, ax_projected, ax_delta, ax_soc = axes.ravel()

    def ordered_values(df: pd.DataFrame, column: str) -> List[float]:
        return [float(df.loc[df["model"] == label, column].iloc[0]) for label in labels]

    def annotate_bars(ax: plt.Axes, values: List[float], suffix: str = "") -> None:
        ymax = max(values) if values else 1.0
        for idx, value in enumerate(values):
            ax.text(idx, value + ymax * 0.055, f"{value:.2f}{suffix}", ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0.0, ymax * 1.22 if ymax > 0 else 1.0)

    attempted = ordered_values(training_summary, "violations_attempted")
    ax_violation.bar(labels, attempted, color=bar_colors, alpha=0.88)
    ax_violation.set_title("Unsafe Raw Actions")
    ax_violation.set_ylabel("Attempted violations / episode")
    annotate_bars(ax_violation, attempted)
    ax_violation.grid(True, axis="y", linestyle="--", alpha=0.25)

    projected = ordered_values(training_summary, "safety_projected_meaningful")
    ax_projected.bar(labels, projected, color=bar_colors, alpha=0.88)
    ax_projected.set_title("SafetyNet Reliance")
    ax_projected.set_ylabel("Meaningful corrections / episode")
    annotate_bars(ax_projected, projected)
    ax_projected.grid(True, axis="y", linestyle="--", alpha=0.25)

    delta = ordered_values(training_summary, "projection_delta_mean_w")
    ax_delta.bar(labels, delta, color=bar_colors, alpha=0.88)
    ax_delta.set_title("Correction Magnitude")
    ax_delta.set_ylabel("Mean |safe - raw| (W)")
    annotate_bars(ax_delta, delta)
    ax_delta.grid(True, axis="y", linestyle="--", alpha=0.25)

    soc_min = ordered_values(rollout_summary, "soc_min")
    soc_max = ordered_values(rollout_summary, "soc_max")
    soc_x = [-0.18, 0.18]
    for idx, label in enumerate(labels):
        x_pos = soc_x[idx]
        ax_soc.vlines(x_pos, soc_min[idx], soc_max[idx], color=colors[label], linewidth=8, alpha=0.85)
        ax_soc.scatter([x_pos, x_pos], [soc_min[idx], soc_max[idx]], color=colors[label], s=70, zorder=3)
        ax_soc.text(x_pos, soc_max[idx] + 0.025, f"{soc_min[idx]:.2f}-{soc_max[idx]:.2f}", ha="center")
    ax_soc.axhspan(0.2, 0.8, color="#2ca02c", alpha=0.12, label="target range")
    ax_soc.axhline(0.2, color="#aa0000", linestyle="--", linewidth=1.0)
    ax_soc.axhline(0.8, color="#aa0000", linestyle="--", linewidth=1.0)
    ax_soc.set_xticks(soc_x, labels)
    ax_soc.set_xlim(-0.55, 0.55)
    ax_soc.set_ylim(0.0, 1.0)
    ax_soc.set_title("20%-80% SoC Target")
    ax_soc.set_ylabel("Rollout SoC range")
    ax_soc.legend(loc="upper right")
    ax_soc.grid(True, axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(title, y=0.985)
    fig.text(
        0.5,
        0.015,
        "Goal: keep SoC inside the 20%-80% operating range while reducing unsafe raw actions before SafetyNet correction.",
        ha="center",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.95), h_pad=2.4, w_pad=1.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved safety summary: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old/new model on the same project dataset day.")
    parser.add_argument("--date", default="2026-04-29")
    parser.add_argument("--dataset", default="data/processed/training_v17_0504_curated.csv")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "raw" / "figures" / "old_new_dataset_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollouts = [rollout_model(spec, args.date, args.dataset) for spec in MODEL_SPECS]
    rollout_summary = pd.DataFrame([summarize_rollout(df) for df in rollouts])
    training_summary = pd.DataFrame([summarize_training_log(spec) for spec in MODEL_SPECS])

    for df in rollouts:
        df.to_csv(output_dir / f"{df['model'].iloc[0].replace(' ', '_')}_rollout_{args.date}.csv", index=False)
    rollout_summary.to_csv(output_dir / f"rollout_summary_{args.date}.csv", index=False)
    training_summary.to_csv(output_dir / f"training_summary_{args.date}.csv", index=False)

    plot_rollout_comparison(
        rollouts,
        output_dir / f"old_new_dataset_rollout_{args.date}.png",
        title=f"Old vs New Model on Project Dataset ({args.date})",
    )
    plot_safety_summary(
        training_summary,
        rollout_summary,
        output_dir / f"old_new_safety_summary_{args.date}.png",
        title="Old vs New Model: Safety-Oriented Retraining",
    )
    print("\nTraining summary")
    print(training_summary.to_string(index=False))
    print("\nRollout summary")
    print(rollout_summary.to_string(index=False))


if __name__ == "__main__":
    main()


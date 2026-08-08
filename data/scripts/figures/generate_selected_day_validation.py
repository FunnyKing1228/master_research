from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from generate_thesis_behavior_figures import (
    ROOT,
    load_config,
    load_dataset,
    plot_overlay,
    plot_single_day,
    rollout_episode,
    setup_style,
)


def summarize_episode(df: pd.DataFrame) -> Dict[str, float]:
    action_w = df["action_applied_w"].astype(float)
    summary = {
        "steps": float(len(df)),
        "start_soc": float(df.attrs.get("start_soc", df["soc"].iloc[0])),
        "end_soc": float(df["soc"].iloc[-1]),
        "soc_min": float(df["soc"].min()),
        "soc_max": float(df["soc"].max()),
        "charge_steps": float((action_w > 1e-6).sum()),
        "discharge_steps": float((action_w < -1e-6).sum()),
        "idle_steps": float((action_w.abs() <= 1e-6).sum()),
        "pv_charge_wh": float(df["pv_to_battery_w"].clip(lower=0.0).sum() * 0.25),
        "useful_discharge_wh": float(df["useful_discharge_w"].clip(lower=0.0).sum() * 0.25),
        "avg_pv_ratio": float(df["pv_support_ratio"].mean()),
        "max_pv_ratio": float(df["pv_support_ratio"].max()),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate selected-day validation plots.")
    parser.add_argument("--experiment", required=True, help="Experiment folder name under experiments/")
    parser.add_argument("--model", default="best_sac_model.pth", help="Model filename under experiment models/")
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["2026-04-09 00:00:00", "2026-04-10 00:00:00"],
        help="Start dates to validate",
    )
    parser.add_argument(
        "--dataset-override",
        default=None,
        help="Optional dataset CSV path override for evaluation",
    )
    parser.add_argument("--output-subdir", default="selected_day_validation", help="Subdirectory under results/")
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()

    experiment_dir = ROOT / "experiments" / args.experiment
    config_path = experiment_dir / "configs" / "experiment_config.yaml"
    model_path = experiment_dir / "models" / args.model
    out_dir = experiment_dir / "results" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(config_path))
    if "config" in config and isinstance(config["config"], dict):
        config = config["config"]
    if args.dataset_override:
        config["env"]["dataset_csv_path"] = str(Path(args.dataset_override))
    dataset_df = load_dataset(config["env"]["dataset_csv_path"])

    episodes: Dict[str, pd.DataFrame] = {}
    for start_date in args.dates:
        episode_df = rollout_episode(
            config,
            dataset_df,
            model_path=model_path,
            start_date=start_date,
            days=1,
            reset_seed=42,
        )
        episodes[start_date] = episode_df
        stamp = str(pd.Timestamp(start_date).date())
        plot_single_day(episode_df, out_dir / f"{stamp}_single_day_behavior")
        summary = summarize_episode(episode_df)
        print(f"=== {stamp} ===")
        for key, value in summary.items():
            print(f"{key}={value:.6f}")

    if len(episodes) >= 2:
        plot_overlay(
            episodes,
            out_dir / "selected_dates_overlay",
            title="Selected sunny-day validation",
        )

    print(f"Saved selected-day validation to: {out_dir}")


if __name__ == "__main__":
    main()

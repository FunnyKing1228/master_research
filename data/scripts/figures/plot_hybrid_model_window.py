from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from generate_thesis_behavior_figures import (
    ROOT,
    load_config,
    load_dataset,
    plot_signed_battery_power,
    rollout_episode,
    save_figure,
    setup_style,
    style_time_axis,
)


def anonymize_day_labels(ax: plt.Axes, timestamps: pd.Series) -> None:
    start_day = timestamps.iloc[0].normalize()
    day_starts = pd.date_range(start=start_day, end=timestamps.iloc[-1].normalize(), freq="D")
    ax.set_xticks(day_starts)

    labels: list[str] = []
    for idx, _ in enumerate(day_starts):
        if idx == 0:
            labels.append("Day A")
        else:
            labels.append(f"Day A+{idx}")
    ax.set_xticklabels(labels)
    ax.set_xlabel("Anonymized day")


def plot_crossday_anonymized(df: pd.DataFrame, out_stem: Path, title: str) -> None:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.1, 1.0, 1.0]},
    )

    axes[0].step(df["timestamp"], df["price"], where="mid", color="#2f2f2f", linewidth=2.0)
    axes[0].set_ylabel("Price")
    axes[0].set_title(title, pad=10)

    axes[1].plot(df["timestamp"], df["soc"], color="#1f6f5f", linewidth=2.1)
    axes[1].axhline(0.1, color="#d62728", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.9, color="#d62728", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("SoC")
    axes[1].set_ylim(0.0, 1.0)

    plot_signed_battery_power(axes[2], df["timestamp"], df["action_applied_w"])
    axes[2].axhline(0.0, color="#444444", linewidth=0.8)
    axes[2].axhline(8.5, color="#7fb3d5", linestyle=":", linewidth=1.0)
    axes[2].axhline(-5.6, color="#f1948a", linestyle=":", linewidth=1.0)
    axes[2].set_ylabel("Battery\npower (W)")
    axes[2].legend(loc="upper left", ncols=2, frameon=True, framealpha=0.9)

    share_df = build_load_share_frame(df)
    axes[3].plot(df["timestamp"], share_df["grid_load_ratio"], color="#2c7be5", linewidth=1.8, label="Grid share")
    axes[3].plot(df["timestamp"], share_df["pv_load_ratio"], color="#f39c12", linewidth=1.8, label="PV share")
    axes[3].plot(
        df["timestamp"],
        share_df["battery_load_ratio"],
        color="#c0392b",
        linewidth=1.6,
        linestyle="--",
        label="Battery useful share",
    )
    axes[3].set_ylabel("Load share")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].legend(loc="upper right", frameon=True, framealpha=0.9)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    anonymize_day_labels(axes[-1], df["timestamp"])
    plt.subplots_adjust(hspace=0.08, top=0.93)
    save_figure(fig, out_stem)


def plot_crossday_single_day_style(
    df: pd.DataFrame,
    out_stem: Path,
    title: str,
    anonymize_days: bool = False,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.1, 0.8]},
    )

    timestamps = df["timestamp"]
    pv_supply_w = df["pv_to_load_w"].clip(lower=0.0)
    battery_supply_w = df["useful_discharge_w"].clip(lower=0.0)
    grid_supply_w = (df["load_w"] - pv_supply_w - battery_supply_w).clip(lower=0.0)

    axes[0].stackplot(
        timestamps,
        pv_supply_w,
        battery_supply_w,
        grid_supply_w,
        colors=["#f6c445", "#e15759", "#4e79a7"],
        alpha=0.88,
        labels=["PV supply", "Battery discharge", "Grid supply"],
    )
    axes[0].plot(
        timestamps,
        df["load_w"],
        color="#2f2f2f",
        linewidth=2.0,
        linestyle="--",
        label="Total load",
    )
    axes[0].plot(
        timestamps,
        df["pv_bus_w"],
        color="#ffb000",
        linewidth=2.4,
        linestyle="-",
        label="Total PV Available",
        zorder=5,
    )
    axes[0].set_ylabel("Power Supply (W)")
    axes[0].set_ylim(0.0, max(float(df["load_w"].max()), float(df["pv_bus_w"].max())) * 1.08)
    axes[0].set_title(title, pad=10)
    axes[0].legend(loc="upper left", ncols=2, frameon=True, framealpha=0.9)

    axes[1].plot(timestamps, df["soc"], color="#1f6f5f", linewidth=2.2, label="SoC")
    axes[1].axhline(0.1, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.85, label="SoC lower bound")
    axes[1].axhline(0.9, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.85, label="SoC upper bound")
    axes[1].set_ylabel("SoC")
    axes[1].set_ylim(0.0, 1.0)

    ax_soc_action = axes[1].twinx()
    charge_w = df["action_applied_w"].clip(lower=0.0)
    discharge_w = df["action_applied_w"].clip(upper=0.0)
    bar_width_days = 10 / (24 * 60)
    ax_soc_action.bar(
        timestamps,
        charge_w,
        width=bar_width_days,
        color="#17becf",
        alpha=0.75,
        label="Battery charge",
        align="center",
    )
    ax_soc_action.bar(
        timestamps,
        discharge_w,
        width=bar_width_days,
        color="#d62728",
        alpha=0.75,
        label="Battery discharge",
        align="center",
    )
    ax_soc_action.axhline(8.5, color="#17becf", linestyle=":", linewidth=1.0, alpha=0.45, label="Charge limit")
    ax_soc_action.axhline(-5.6, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.45, label="Discharge limit")
    ax_soc_action.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    ax_soc_action.set_ylabel("Battery Power (W)")

    lines_l, labels_l = axes[1].get_legend_handles_labels()
    lines_r, labels_r = ax_soc_action.get_legend_handles_labels()
    axes[1].legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", ncols=2, frameon=True, framealpha=0.9)

    axes[2].step(
        timestamps,
        df["price"],
        where="mid",
        color="#2f2f2f",
        linewidth=2.0,
        label="Price",
    )
    axes[2].set_ylabel("Price (TWD/kWh)")
    axes[2].legend(loc="upper left", frameon=True, framealpha=0.9)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    if anonymize_days:
        anonymize_day_labels(axes[-1], timestamps)
    else:
        style_time_axis(axes[-1], days=int(df.attrs.get("days", 1)))

    plt.subplots_adjust(hspace=0.1, top=0.93)
    save_figure(fig, out_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot anonymized hybrid model rollout window.")
    parser.add_argument("--experiment", required=True, help="Experiment name under experiments/")
    parser.add_argument("--model", default="best_sac_model.pth", help="Model file name")
    parser.add_argument("--dataset", default="data/processed/training_v16_hybrid50.csv", help="Dataset path")
    parser.add_argument("--start-date", default="2026-05-04 00:00:00", help="Window start timestamp")
    parser.add_argument("--days", type=int, default=5, help="Number of days")
    parser.add_argument("--output-subdir", default="hybrid_window_validation", help="Results subdirectory")
    parser.add_argument(
        "--style",
        choices=["anonymized", "single_day_thesis"],
        default="anonymized",
        help="Figure layout style",
    )
    parser.add_argument(
        "--anonymize-days",
        action="store_true",
        help="Replace x-axis dates with Day A / Day A+1 labels",
    )
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
    config["env"]["dataset_csv_path"] = str((ROOT / args.dataset).resolve()) if not Path(args.dataset).is_absolute() else args.dataset
    dataset_df = load_dataset(config["env"]["dataset_csv_path"])

    rollout_df = rollout_episode(
        config,
        dataset_df,
        model_path=model_path,
        start_date=args.start_date,
        days=int(args.days),
        reset_seed=42,
    )

    if args.style == "single_day_thesis":
        suffix = "single_day_style_anonymized" if args.anonymize_days else "single_day_style"
        out_stem = out_dir / f"hybrid_{int(args.days)}day_window_{suffix}"
        title = f"Continuous {int(args.days)}-day validation in thesis single-day style"
        plot_crossday_single_day_style(rollout_df, out_stem, title=title, anonymize_days=args.anonymize_days)
    else:
        out_stem = out_dir / f"hybrid_{int(args.days)}day_window_anonymized"
        title = f"Hybrid {int(args.days)}-day rollout (anonymized days)"
        plot_crossday_anonymized(rollout_df, out_stem, title=title)

    print(out_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()

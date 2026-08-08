from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = ROOT / "data" / "processed" / "training_v16.csv"
OUT_DIR = ROOT / "experiments" / "dataset_validation"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    df["load_kw"] = df["Consumption"].astype(float)
    df["pv_kw"] = df["Solar"].astype(float)
    df["pv_ratio"] = df["pv_kw"] / df["load_kw"].clip(lower=1e-9)
    return df


def plot_daily_max_ratio(df: pd.DataFrame) -> Path:
    daily = (
        df.groupby("date")
        .agg(
            max_ratio=("pv_ratio", "max"),
            steps_ge_090=("pv_ratio", lambda s: int((s >= 0.9).sum())),
            steps_ge_100=("pv_ratio", lambda s: int((s >= 1.0).sum())),
        )
        .reset_index()
    )

    colors = [
        "#d62728" if ratio >= 1.0 else "#ff7f0e" if ratio >= 0.9 else "#4e79a7"
        for ratio in daily["max_ratio"]
    ]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(daily["date"].astype(str), daily["max_ratio"], color=colors)
    ax.axhline(0.9, color="#ff7f0e", linestyle="--", linewidth=1.0, label="0.9 threshold")
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, label="1.0 surplus")
    ax.set_title("Dataset validation: strongest solar support by day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily max PV/load ratio")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    out_path = OUT_DIR / "pv_ratio_daily_max.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sunny_days_detail(df: pd.DataFrame) -> Path:
    target_dates = {"2026-04-09", "2026-04-10"}
    sunny = df[df["date"].astype(str).isin(target_dates)].copy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (date_key, group) in zip(axes, sunny.groupby("date")):
        ax.plot(group["timestamp"], group["load_kw"], color="#4e79a7", linewidth=2, label="Load (kW)")
        ax.plot(group["timestamp"], group["pv_kw"], color="#f28e2b", linewidth=2, label="PV (kW)")

        ax_ratio = ax.twinx()
        ax_ratio.plot(
            group["timestamp"],
            group["pv_ratio"],
            color="#59a14f",
            linewidth=2,
            linestyle="--",
            label="PV/load ratio",
        )
        ax_ratio.axhline(0.9, color="#ff7f0e", linestyle=":")
        ax_ratio.axhline(1.0, color="#d62728", linestyle=":")

        ax.set_title(f"{date_key} solar coverage validation")
        ax.set_ylabel("Power (kW)")
        ax_ratio.set_ylabel("PV/load")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_ratio.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    fig.tight_layout()
    out_path = OUT_DIR / "pv_ratio_sunny_days_detail.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize_windows(df: pd.DataFrame, episode_length: int = 288) -> None:
    n_rows = len(df)
    max_start = n_rows - episode_length
    hits_090 = np.array(
        [int((df.iloc[s : s + episode_length]["pv_ratio"] >= 0.9).sum()) for s in range(max_start + 1)]
    )
    hits_100 = np.array(
        [int((df.iloc[s : s + episode_length]["pv_ratio"] >= 1.0).sum()) for s in range(max_start + 1)]
    )

    print("=== Dataset Summary ===")
    print(f"rows={n_rows}")
    print(f"days={df['date'].nunique()}")
    print(f"steps_ge_090={(df['pv_ratio'] >= 0.9).sum()}")
    print(f"steps_ge_100={(df['pv_ratio'] >= 1.0).sum()}")
    print(f"fraction_ge_090={(df['pv_ratio'] >= 0.9).mean():.6f}")
    print(f"fraction_ge_100={(df['pv_ratio'] >= 1.0).mean():.6f}")
    print()
    print("=== 3-Day Window Exposure ===")
    print(f"window_count={len(hits_090)}")
    print(f"episodes_with_any_ge_090={(hits_090 > 0).sum()}")
    print(f"episodes_with_any_ge_100={(hits_100 > 0).sum()}")
    print(f"fraction_with_any_ge_090={(hits_090 > 0).mean():.6f}")
    print(f"fraction_with_any_ge_100={(hits_100 > 0).mean():.6f}")
    print(f"mean_ge_090_per_window={hits_090.mean():.6f}")
    print(f"max_ge_090_per_window={hits_090.max()}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    df = load_dataset()
    daily_path = plot_daily_max_ratio(df)
    sunny_path = plot_sunny_days_detail(df)
    summarize_windows(df, episode_length=288)
    print(daily_path)
    print(sunny_path)


if __name__ == "__main__":
    main()

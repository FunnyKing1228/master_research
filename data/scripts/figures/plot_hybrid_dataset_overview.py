from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "data" / "processed" / "training_v16_hybrid50.csv"
DEFAULT_OUT = ROOT / "experiments" / "dataset_validation" / "hybrid50_5day_overview.png"


def add_boolean_spans(ax: plt.Axes, timestamps: pd.Series, bool_values: pd.Series) -> None:
    values = bool_values.fillna(0.0).to_numpy(dtype=float) > 0.5
    start_idx = None
    label_used = False
    for idx, is_true in enumerate(values):
        if is_true and start_idx is None:
            start_idx = idx
        is_last = idx == len(values) - 1
        if start_idx is not None and ((not is_true) or is_last):
            end_idx = idx if not is_true else idx + 1
            start_ts = timestamps.iloc[start_idx]
            if end_idx < len(timestamps):
                end_ts = timestamps.iloc[end_idx]
            else:
                delta = timestamps.iloc[-1] - timestamps.iloc[-2] if len(timestamps) > 1 else pd.Timedelta(minutes=15)
                end_ts = timestamps.iloc[-1] + delta
            ax.axvspan(
                start_ts,
                end_ts,
                color="#f6c445",
                alpha=0.18,
                lw=0,
                label=("PV sufficient (ratio >= 0.8)" if not label_used else None),
            )
            label_used = True
            start_idx = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 5-day hybrid dataset overview.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Input dataset CSV")
    parser.add_argument("--start-date", default="2026-05-04 00:00:00", help="Start timestamp")
    parser.add_argument("--days", type=int, default=5, help="Number of days to plot")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["Solar"] = df["Solar"].astype(float)
    df["Consumption"] = df["Consumption"].astype(float)
    df["PV_bool"] = df["PV_bool"].astype(float)
    df["pv_ratio"] = df["Solar"] / df["Consumption"].clip(lower=1e-9)

    start_ts = pd.Timestamp(args.start_date)
    end_ts = start_ts + pd.Timedelta(days=int(args.days))
    sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    if sub.empty:
        raise ValueError(f"No data found in range {start_ts} to {end_ts}.")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 0.9, 0.8]},
    )

    axes[0].plot(sub["timestamp"], sub["Consumption"] * 1000.0, color="#2f2f2f", linewidth=2.0, label="Load")
    axes[0].plot(sub["timestamp"], sub["Solar"] * 1000.0, color="#ffb000", linewidth=2.2, label="PV available")
    axes[0].set_ylabel("Power (W)")
    axes[0].set_title("Hybrid training dataset overview (5-day window)")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.9)

    add_boolean_spans(axes[1], sub["timestamp"], sub["PV_bool"])
    axes[1].plot(sub["timestamp"], sub["pv_ratio"], color="#8c4f00", linewidth=2.0, label="PV/load ratio")
    axes[1].axhline(0.8, color="#d98c00", linestyle="--", linewidth=1.0, label="Sufficient threshold = 0.8")
    axes[1].set_ylabel("PV / Load")
    axes[1].legend(loc="upper left", frameon=True, framealpha=0.9)

    axes[2].step(sub["timestamp"], sub["price"], where="mid", color="#2f2f2f", linewidth=2.0, label="Price")
    axes[2].set_ylabel("Price\n(TWD/kWh)")
    axes[2].legend(loc="upper left", frameon=True, framealpha=0.9)

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 25, 6)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].set_xlabel("Time")

    plt.subplots_adjust(hspace=0.08, top=0.93)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()

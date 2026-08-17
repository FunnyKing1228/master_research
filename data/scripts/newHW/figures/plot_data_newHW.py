"""Plot provisional newHW processed data with explicit quality warnings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot provisional newHW data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        "newHW data preparation — PROVISIONAL / IN-SAMPLE ONLY\n"
        "TODO(newHW): fixed 28.2 W inferred load; SoC anchor and BMS behavior unconfirmed",
        fontsize=12,
    )

    axes[0].plot(df["timestamp"], df["Solar"] * 1000.0, label="PV (measured)", color="#f2a900")
    axes[0].plot(
        df["timestamp"],
        df["Consumption"] * 1000.0,
        label="Load (fixed inferred baseline)",
        color="#333333",
    )
    axes[0].set_ylabel("Power (W)")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        df["timestamp"],
        df["soc_reconstructed_unclipped"],
        label="Integrated relative SoC (unclipped)",
        color="#9467bd",
    )
    axes[1].plot(
        df["timestamp"],
        df["soc_reconstructed"],
        label="Clipped display SoC",
        color="#1f77b4",
    )
    axes[1].axhspan(0.0, 1.0, color="#2ca02c", alpha=0.08)
    axes[1].set_ylabel("Provisional SoC")
    axes[1].legend(loc="lower left")

    axes[2].plot(
        df["timestamp"],
        df["battery_voltage_v"],
        label="Battery voltage (masked/interpolated mean)",
        color="#d62728",
    )
    axes[2].plot(
        df["timestamp"],
        df["battery_voltage_valid_fraction"] * 30.0,
        label="Valid-voltage fraction × 30",
        color="#7f7f7f",
        alpha=0.7,
    )
    axes[2].set_ylabel("Voltage / quality")
    axes[2].legend(loc="lower left")

    axes[3].bar(
        df["timestamp"],
        df["max_source_gap_seconds"],
        width=0.008,
        label="Max source gap in 15-min bin",
        color="#8c564b",
    )
    axes[3].axhline(60.0, color="#d62728", linestyle="--", label="60 s warning")
    axes[3].set_ylabel("Gap (s)")
    axes[3].set_xlabel("Timestamp")
    axes[3].legend(loc="upper right")

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

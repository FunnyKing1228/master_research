from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = RAW_DIR / "figures"

START = pd.Timestamp("2026-04-27 00:00")
END = pd.Timestamp("2026-04-28 12:00")
OUTPUT = FIG_DIR / "deployment_successful_closed_loop_cycle_2026-04-27_to_2026-04-28.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a clean deployment charge/discharge cycle figure.")
    parser.add_argument("--start", default=str(START), help="Start timestamp, e.g. '2026-04-27 00:00'.")
    parser.add_argument("--end", default=str(END), help="End timestamp, e.g. '2026-04-28 12:00'.")
    parser.add_argument("--output", default=str(OUTPUT), help="Output PNG path.")
    return parser.parse_args()


def deployment_files_for_window(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    return [RAW_DIR / f"deployment_v2_{date:%Y-%m-%d}.csv" for date in dates]


def load_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    files = deployment_files_for_window(start, end)
    df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = ["load_kw", "pv_kw", "batt_p_mean_mW", "soc"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    df = df.sort_values("timestamp").dropna(subset=["timestamp"])

    df["load_w"] = df["load_kw"] * 1000.0
    df["pv_w"] = df["pv_kw"] * 1000.0
    df["battery_power_w"] = df["batt_p_mean_mW"] / 1000.0
    df["soc_pct"] = df["soc"] * 100.0

    # A light rolling median keeps one-minute sensor jitter readable on slides.
    plot_cols = ["load_w", "pv_w", "battery_power_w", "soc_pct"]
    df[plot_cols] = df[plot_cols].rolling(window=5, min_periods=1, center=True).median()
    return df


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    output = Path(args.output)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = load_window(start, end)

    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, (ax_power, ax_soc) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.60},
    )
    fig.patch.set_facecolor("white")

    ax_power.plot(df["timestamp"], df["load_w"], color="black", linestyle="--", lw=2.0, label="Load Demand")
    ax_power.plot(df["timestamp"], df["pv_w"], color="#f2b705", linestyle="-", lw=2.4, label="PV Supply")
    charge_w = df["battery_power_w"].clip(lower=0)
    discharge_w = df["battery_power_w"].clip(upper=0)
    ax_power.fill_between(
        df["timestamp"],
        0,
        charge_w,
        where=charge_w > 0,
        color="#2ca25f",
        alpha=0.45,
        linewidth=0,
        label="Actual Battery Charge",
    )
    ax_power.fill_between(
        df["timestamp"],
        0,
        discharge_w,
        where=discharge_w < 0,
        color="#de2d26",
        alpha=0.45,
        linewidth=0,
        label="Actual Battery Discharge",
    )
    ax_power.axhline(0, color="black", lw=1.0)
    ax_power.set_ylabel("Power (W)")
    ax_power.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=False)

    ax_soc.plot(df["timestamp"], df["soc_pct"], color="#3b2f8f", linestyle="-", lw=2.6, label="SoC")
    ax_soc.axhline(10, color="#b24a4a", linestyle="--", lw=1.5)
    ax_soc.axhline(90, color="#b24a4a", linestyle="--", lw=1.5)
    ax_soc.set_ylabel("SoC (%)")
    ax_soc.set_ylim(0, 100)
    ax_soc.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False)

    for ax in (ax_power, ax_soc):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", color="0.90", linewidth=0.8)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_soc.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax_soc.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax_soc.set_xlim(start, end)

    fig.suptitle("Milestone: Achieving Full Closed-Loop Control in Field Deployment", fontsize=16, y=0.99)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.80, hspace=0.62)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()


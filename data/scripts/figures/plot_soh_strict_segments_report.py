"""
Create a report figure for strict 25-75% SoH deployment segments.

The figure is intended for presentation/report use: it shows the two usable
deployment charge segments, their estimated SoH, and the corresponding effective
capacity converted from the nominal 2000 mAh SLFB capacity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SEG_DIR = ROOT / "data" / "soh_segments_strict_25_75"
CYCLE_DIR = SEG_DIR / "cycles"
OUT_DIR = ROOT / "data" / "raw" / "figures" / "soh_strict_segments"
NOMINAL_CAPACITY_MAH = 2000.0
NOMINAL_CAPACITY_WH = 11.2


def _load_data() -> pd.DataFrame:
    summary = pd.read_csv(SEG_DIR / "soh_segment_summary.csv")
    pred = pd.read_csv(SEG_DIR / "soh_predictions.csv")
    df = summary.merge(pred, on="file", how="left")
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values("start").reset_index(drop=True)
    df["soh_pct"] = df["soh"] * 100.0
    df["effective_capacity_mah"] = df["soh"] * NOMINAL_CAPACITY_MAH
    df["effective_capacity_wh"] = df["soh"] * NOMINAL_CAPACITY_WH
    return df


def _cycle_trace(row: pd.Series) -> pd.DataFrame:
    path = CYCLE_DIR / str(row["file"])
    trace = pd.read_csv(path)
    trace["time_min"] = trace["time"] / 60.0
    return trace


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_data()

    fig = plt.figure(figsize=(15.0, 9.5))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.07,
        right=0.97,
        top=0.88,
        bottom=0.09,
        wspace=0.32,
        hspace=0.48,
    )
    ax_v = fig.add_subplot(gs[0, 0])
    ax_i = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_note = fig.add_subplot(gs[1, 1])

    colors = ["#1f77b4", "#ff7f0e"]
    labels = []
    for idx, row in df.iterrows():
        trace = _cycle_trace(row)
        label = (
            f"{row['start']:%m/%d %H:%M} "
            f"SoH={row['soh_pct']:.1f}%"
        )
        labels.append(f"{row['start']:%m/%d}")
        ax_v.plot(
            trace["time_min"],
            trace["voltage"],
            color=colors[idx],
            linewidth=2.0,
            label=label,
        )
        ax_i.plot(
            trace["time_min"],
            trace["current"] * 1000.0,
            color=colors[idx],
            linewidth=1.8,
            label=label,
        )

    ax_v.set_title("Voltage Curve (Natural 25-75% Charge Segments)", pad=12)
    ax_v.set_xlabel("Elapsed Time (min)")
    ax_v.set_ylabel("Battery Voltage (V)")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="lower left", fontsize=9, framealpha=0.9)

    ax_i.set_title("Measured Charge Current", pad=12)
    ax_i.set_xlabel("Elapsed Time (min)")
    ax_i.set_ylabel("Current (mA)")
    ax_i.grid(True, alpha=0.3)

    bars = ax_bar.bar(
        labels,
        df["effective_capacity_mah"],
        color=colors,
        alpha=0.85,
    )
    ax_bar.axhline(
        NOMINAL_CAPACITY_MAH,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax_bar.set_ylim(0, NOMINAL_CAPACITY_MAH * 1.30)
    ax_bar.set_title("Estimated Effective Capacity from SoH", pad=28)
    ax_bar.set_ylabel("Capacity (mAh)")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.text(
        0.98,
        0.96,
        "Nominal capacity = 2000 mAh",
        transform=ax_bar.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.4"},
    )
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + NOMINAL_CAPACITY_MAH * 0.055,
            f"{row['soh_pct']:.1f}%\n"
            f"{row['effective_capacity_mah']:.0f} mAh\n"
            f"{row['effective_capacity_wh']:.2f} Wh",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax_note.axis("off")
    lines = ["Strict segment selection", ""]
    for _, row in df.iterrows():
        lines.append(
            f"{row['start']:%Y-%m-%d %H:%M} - {row['end']:%H:%M}"
        )
        lines.append(
            f"  SoC: {row['soc_min']*100:.1f}-{row['soc_max']*100:.1f}%"
        )
        lines.append(
            f"  V: {row['voltage_min']:.2f}-{row['voltage_max']:.2f} V, "
            f"Imean: {row['current_ma_mean']:.0f} mA"
        )
        lines.append(
            f"  SoH: {row['soh_pct']:.1f}% -> "
            f"{row['effective_capacity_mah']:.0f} mAh / "
            f"{row['effective_capacity_wh']:.2f} Wh"
        )
        lines.append("")
    lines.append(
        "Note: model estimates from partial deployment charge segments."
    )
    lines.append(
        "They are not laboratory capacity-test ground truth."
    )
    ax_note.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax_note.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(
        "Offline SoH Estimate from Natural Deployment Charge Segments",
        fontsize=17,
        fontweight="bold",
        y=0.96,
    )
    out_path = OUT_DIR / "soh_strict_25_75_segments_report.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


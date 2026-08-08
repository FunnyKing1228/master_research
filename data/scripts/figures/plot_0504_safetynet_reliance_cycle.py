from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "figures" / "report_0504"
OUTPUT = OUT_DIR / "challenge3_safetynet_reliance_cycle_2026-04-27_to_2026-04-29_1330.png"

START = pd.Timestamp("2026-04-27 00:00")
END = pd.Timestamp("2026-04-29 13:30")


def deployment_files_for_window(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    return [RAW_DIR / f"deployment_v2_{date:%Y-%m-%d}.csv" for date in dates]


def raw_files_for_window(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    return [RAW_DIR / f"raw_data_v2_{date:%Y-%m-%d}.csv" for date in dates]


def load_deployment_window() -> pd.DataFrame:
    files = deployment_files_for_window(START, END)
    df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = [
        "load_kw",
        "pv_kw",
        "batt_p_mean_mW",
        "soc",
        "action_raw_kw",
        "action_power_kw",
        "voltage_cutoff_active",
        "voltage_cutoff_day_locked",
        "guard_block_voltage_cutoff",
        "guard_block_low_soc_discharge",
        "guard_block_pv_active_discharge",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["timestamp"] >= START) & (df["timestamp"] <= END)].copy()
    df = df.sort_values("timestamp").dropna(subset=["timestamp"])

    df["load_w"] = df["load_kw"] * 1000.0
    df["pv_w"] = df["pv_kw"] * 1000.0
    df["battery_power_w"] = df["batt_p_mean_mW"] / 1000.0
    df["action_raw_w"] = df["action_raw_kw"] * 1000.0
    df["action_safe_w"] = df["action_power_kw"] * 1000.0
    df["soc_pct"] = df["soc"] * 100.0

    plot_cols = [
        "load_w",
        "pv_w",
        "battery_power_w",
        "action_raw_w",
        "action_safe_w",
        "soc_pct",
    ]
    df[plot_cols] = df[plot_cols].rolling(window=5, min_periods=1, center=True).median()
    return df


def load_raw_voltage_window() -> pd.DataFrame:
    files = raw_files_for_window(START, END)
    raw = pd.concat((pd.read_csv(path, usecols=["timestamp", "voltage_v"]) for path in files), ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw["voltage_v"] = pd.to_numeric(raw["voltage_v"], errors="coerce")
    raw = raw[(raw["timestamp"] >= START) & (raw["timestamp"] <= END)].copy()
    raw.loc[raw["voltage_v"] <= 0.5, "voltage_v"] = pd.NA
    return raw.sort_values("timestamp").dropna(subset=["timestamp", "voltage_v"])


def mask_to_spans(timestamps: pd.Series, mask: pd.Series, max_gap_seconds: int = 180) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    active = pd.DataFrame({"timestamp": timestamps, "mask": mask.fillna(False).astype(bool)})
    active = active[active["mask"]].copy()
    if active.empty:
        return []

    groups = active["timestamp"].diff().dt.total_seconds().fillna(0).gt(max_gap_seconds).cumsum()
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, group in active.groupby(groups):
        start = group["timestamp"].iloc[0]
        end = group["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
        spans.append((start, end))
    return spans


def count_rising_edges(mask: pd.Series) -> int:
    active = mask.fillna(False).astype(bool)
    return int((active & ~active.shift(fill_value=False)).sum())


def add_discharge_forbidden_spans(ax: plt.Axes, spans: list[tuple[pd.Timestamp, pd.Timestamp]], label: str | None = None) -> None:
    for idx, (start, end) in enumerate(spans):
        ax.axvspan(
            start,
            end,
            color="#de2d26",
            alpha=0.10,
            linewidth=0,
            label=label if idx == 0 else None,
        )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color="0.90", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_deployment_window()
    raw_voltage = load_raw_voltage_window()

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

    fig, (ax_power, ax_control, ax_voltage, ax_soc) = plt.subplots(
        4,
        1,
        figsize=(12, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.85, 0.75], "hspace": 0.48},
    )
    fig.patch.set_facecolor("white")

    cutoff_mask = (
        (df["voltage_cutoff_active"] > 0)
        | (df["voltage_cutoff_day_locked"] > 0)
        | (df["guard_block_voltage_cutoff"] > 0)
    )
    cutoff_spans = mask_to_spans(df["timestamp"], cutoff_mask)

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
        alpha=0.42,
        linewidth=0,
        label="Actual Battery Charge",
    )
    ax_power.fill_between(
        df["timestamp"],
        0,
        discharge_w,
        where=discharge_w < 0,
        color="#de2d26",
        alpha=0.42,
        linewidth=0,
        label="Actual Battery Discharge",
    )
    ax_power.axhline(0, color="black", lw=1.0)
    ax_power.set_ylabel("Power (W)")
    ax_power.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=False)

    ax_control.plot(
        df["timestamp"],
        df["action_raw_w"],
        color="#7f7f7f",
        linestyle="--",
        lw=2.2,
        label="Raw Model Action",
    )
    ax_control.plot(
        df["timestamp"],
        df["action_safe_w"],
        color="#1f77b4",
        linestyle="-",
        lw=2.4,
        label="Final Action after SafetyNet",
    )
    ax_control.fill_between(
        df["timestamp"],
        df["action_raw_w"],
        df["action_safe_w"],
        color="#9ecae1",
        alpha=0.28,
        linewidth=0,
        label="SafetyNet Correction",
    )
    ax_control.axhline(0, color="black", lw=1.0)
    ax_control.set_ylabel("Control (W)")
    ax_control.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False)

    add_discharge_forbidden_spans(ax_voltage, cutoff_spans, label="Cutoff Protect Active")
    ax_voltage.plot(
        raw_voltage["timestamp"],
        raw_voltage["voltage_v"],
        color="#252525",
        lw=0.9,
        alpha=0.80,
        label="Raw Battery Voltage",
    )
    low_voltage = raw_voltage[(raw_voltage["voltage_v"] > 0.5) & (raw_voltage["voltage_v"] < 4.2)]
    ax_voltage.scatter(
        low_voltage["timestamp"],
        low_voltage["voltage_v"],
        color="#de2d26",
        s=14,
        alpha=0.9,
        label="Raw < 4.2V",
        zorder=3,
    )
    ax_voltage.axhline(4.2, color="#de2d26", linestyle="--", lw=1.5, label="Cutoff Threshold (4.2V)")
    ax_voltage.axhline(5.0, color="#f28e2b", linestyle=":", lw=1.5, label="Recover Threshold (5.0V)")
    ax_voltage.set_ylabel("Battery V")
    ax_voltage.set_ylim(0, 13)
    ax_voltage.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=False)

    ax_soc.plot(df["timestamp"], df["soc_pct"], color="#3b2f8f", linestyle="-", lw=2.6, label="SoC")
    ax_soc.axhline(10, color="#b24a4a", linestyle="--", lw=1.5)
    ax_soc.axhline(90, color="#b24a4a", linestyle="--", lw=1.5)
    ax_soc.set_ylabel("SoC (%)")
    ax_soc.set_ylim(0, 100)
    ax_soc.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False)

    for ax in (ax_power, ax_control, ax_voltage, ax_soc):
        style_axis(ax)

    ax_soc.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax_soc.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax_soc.set_xlim(START, END)

    fig.suptitle("Challenge III: Model Reliance on SafetyNet Corrections", fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.84, hspace=0.62)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")

    correction_w = (df["action_raw_w"] - df["action_safe_w"]).abs()
    raw_low_voltage_events = count_rising_edges((raw_voltage["voltage_v"] > 0.5) & (raw_voltage["voltage_v"] < 4.2))
    cutoff_active_events = count_rising_edges(df["voltage_cutoff_active"] > 0)
    print(f"Saved figure: {OUTPUT}")
    print(f"Median correction W: {correction_w.median():.2f}")
    print(f"95th percentile correction W: {correction_w.quantile(0.95):.2f}")
    print(f"Raw low-voltage events (<4.2V, excluding 0V glitches): {raw_low_voltage_events}")
    print(f"Voltage cutoff active events: {cutoff_active_events}")
    print(f"Cutoff protect spans: {len(cutoff_spans)}")


if __name__ == "__main__":
    main()


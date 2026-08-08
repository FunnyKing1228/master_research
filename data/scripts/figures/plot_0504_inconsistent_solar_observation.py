from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "figures" / "report_0504"
OUTPUT = OUT_DIR / "inconsistent_observations_pv_demand_censoring_2026-04-27.png"

START = pd.Timestamp("2026-04-27 08:00")
END = pd.Timestamp("2026-04-27 16:30")


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DIR / "raw_data_v2_2026-04-27.csv")
    dep = pd.read_csv(RAW_DIR / "deployment_v2_2026-04-27.csv")

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    dep["timestamp"] = pd.to_datetime(dep["timestamp"], errors="coerce")

    for col in ["solar_p_mw", "mppt_p_mw", "bus_p_mw"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in ["pv_kw", "load_kw", "batt_p_mean_mW"]:
        dep[col] = pd.to_numeric(dep[col], errors="coerce")

    raw = raw[(raw["timestamp"] >= START) & (raw["timestamp"] <= END)].copy()
    dep = dep[(dep["timestamp"] >= START) & (dep["timestamp"] <= END)].copy()

    raw_1min = (
        raw.set_index("timestamp")[["solar_p_mw", "mppt_p_mw", "bus_p_mw"]]
        .resample("1min")
        .median()
        .reset_index()
    )
    dep = dep.sort_values("timestamp")
    raw_1min = raw_1min.sort_values("timestamp")

    df = pd.merge_asof(dep, raw_1min, on="timestamp", direction="nearest", tolerance=pd.Timedelta("45s"))
    df["panel_raw_w"] = df["solar_p_mw"] / 1000.0
    df["controller_observed_w"] = df["pv_kw"] * 1000.0
    df["load_w"] = df["load_kw"] * 1000.0
    df["battery_charge_w"] = (df["batt_p_mean_mW"] / 1000.0).clip(lower=0)
    df["power_demand_w"] = df["load_w"] + df["battery_charge_w"]

    smooth_cols = [
        "controller_observed_w",
        "load_w",
        "battery_charge_w",
        "power_demand_w",
    ]
    df[smooth_cols] = df[smooth_cols].rolling(window=7, min_periods=1, center=True).median()
    tolerance_w = 0.6
    df["demand_capped"] = df["controller_observed_w"] >= (df["power_demand_w"] - tolerance_w)
    df["pv_limited"] = df["controller_observed_w"] < (df["power_demand_w"] - tolerance_w)
    return df.dropna(subset=["panel_raw_w", "controller_observed_w"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

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

    fig, ax = plt.subplots(figsize=(12, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        df["timestamp"],
        df["power_demand_w"],
        color="black",
        lw=3.0,
        linestyle="-",
        label="Power Demand (Load + Battery Charging)",
    )
    ax.plot(
        df["timestamp"],
        df["controller_observed_w"],
        color="#1f77b4",
        lw=2.8,
        label="Controller-Observed PV",
    )
    ax.fill_between(
        df["timestamp"],
        df["controller_observed_w"],
        df["power_demand_w"],
        where=df["pv_limited"],
        color="#4e79a7",
        alpha=0.16,
        linewidth=0,
        label="PV-Limited: observed PV is likely real available PV",
    )
    ax.scatter(
        df.loc[df["demand_capped"], "timestamp"],
        df.loc[df["demand_capped"], "controller_observed_w"],
        s=18,
        color="#f28e2b",
        alpha=0.85,
        label="Demand-Capped: available PV may be higher",
    )

    ax.set_title("Inconsistent Observations: PV Measurement Is Censored by Demand", pad=26)
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time")
    ax.set_xlim(START, END)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color="0.90", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=2, frameon=False)

    caption = (
        "If observed PV is below demand, the system is PV-limited and the measurement is closer to true available PV. "
        "If observed PV reaches demand, the measurement is demand-capped: available PV may be higher, but the controller cannot observe how much higher."
    )
    fig.text(0.08, 0.035, caption, ha="left", va="bottom", fontsize=11, color="0.25")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.74)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved figure: {OUTPUT}")
    print(f"Demand-capped samples: {int(df['demand_capped'].sum())}")
    print(f"Controller-observed max W: {df['controller_observed_w'].max():.2f}")
    print(f"Power demand max W: {df['power_demand_w'].max():.2f}")


if __name__ == "__main__":
    main()


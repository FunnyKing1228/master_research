"""Re-audit Scenario 1 windows by load bus / grid / current phases.

The previous audit lumped all positive-current samples together as "firmware
override". This script splits them into three physically distinct phases:

  A  Isolated discharge attempt: V<5.5, grid<1W, |I_raw|<200mA
     ??load bus appears disconnected (load_p_mw collapses to <0.05W),
       battery hovers near its no-load voltage even though we commanded sit=1.

  B  Firmware CV charging:       V>6.0, I_raw>+200mA, grid>5W
     ??grid is actively charging the battery; our sit=1 command was overridden.

  C  Parallel grid+battery:      V in 5..7.5, grid>1W, load>0.5W, |I|<200mA
     ??grid covers load while battery floats. (Empirically: zero samples.)

Outputs go to data/raw/figures/deployment_0515_0518_scenario1_phases/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "figures" / "deployment_0515_0518_scenario1_phases"
DAYS = pd.date_range("2026-05-15", "2026-05-18", freq="D")


PHASE_A_VOLTAGE_MAX = 5.5
PHASE_A_CURRENT_MAX_MA = 200.0
PHASE_A_GRID_MAX_W = 1.0

PHASE_B_VOLTAGE_MIN = 6.0
PHASE_B_CURRENT_MIN_MA = 200.0
PHASE_B_GRID_MIN_W = 5.0

PHASE_C_VOLTAGE_RANGE = (5.0, 7.5)
PHASE_C_GRID_MIN_W = 1.0
PHASE_C_LOAD_MIN_W = 0.5
PHASE_C_CURRENT_MAX_MA = 200.0


def load_raw() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in DAYS:
        path = RAW_DIR / f"raw_data_v2_{day:%Y-%m-%d}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No raw_data_v2 files for 2026-05-15..18")
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    for col in [
        "voltage_v",
        "current_ma",
        "current_raw_ma",
        "grid_p_mw",
        "load_p_mw",
        "bus_p_mw",
        "situation_code",
        "soc_calc",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["grid_w"] = df["grid_p_mw"] / 1000.0
    df["load_w"] = df["load_p_mw"] / 1000.0
    df["pv_w"] = df["bus_p_mw"] / 1000.0
    return df


def classify(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    sit1 = df["situation_code"] == 1
    phase_a = (
        sit1
        & (df["voltage_v"] < PHASE_A_VOLTAGE_MAX)
        & (df["grid_w"] < PHASE_A_GRID_MAX_W)
        & (df["current_raw_ma"].abs() < PHASE_A_CURRENT_MAX_MA)
    )
    phase_b = (
        sit1
        & (df["voltage_v"] > PHASE_B_VOLTAGE_MIN)
        & (df["current_raw_ma"] > PHASE_B_CURRENT_MIN_MA)
        & (df["grid_w"] > PHASE_B_GRID_MIN_W)
    )
    phase_c = (
        sit1
        & df["voltage_v"].between(*PHASE_C_VOLTAGE_RANGE)
        & (df["grid_w"] > PHASE_C_GRID_MIN_W)
        & (df["load_w"] > PHASE_C_LOAD_MIN_W)
        & (df["current_raw_ma"].abs() < PHASE_C_CURRENT_MAX_MA)
    )

    df["sit1_phase_a"] = phase_a.astype(int)
    df["sit1_phase_b"] = phase_b.astype(int)
    df["sit1_phase_c"] = phase_c.astype(int)
    df["sit1_other"] = (sit1 & ~(phase_a | phase_b | phase_c)).astype(int)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, g in df.groupby(df["timestamp"].dt.date):
        sit1 = g[g["situation_code"] == 1]
        if sit1.empty:
            continue
        rows.append(
            {
                "date": str(date),
                "sit1_samples": len(sit1),
                "phase_A_samples": int(sit1["sit1_phase_a"].sum()),
                "phase_B_samples": int(sit1["sit1_phase_b"].sum()),
                "phase_C_samples": int(sit1["sit1_phase_c"].sum()),
                "other_samples": int(sit1["sit1_other"].sum()),
                "phase_A_pct": float(sit1["sit1_phase_a"].mean() * 100.0),
                "phase_B_pct": float(sit1["sit1_phase_b"].mean() * 100.0),
                "load_w_mean_in_A": float(sit1.loc[sit1["sit1_phase_a"] == 1, "load_w"].mean()),
                "load_w_mean_in_B": float(sit1.loc[sit1["sit1_phase_b"] == 1, "load_w"].mean()),
                "voltage_v_mean_in_A": float(sit1.loc[sit1["sit1_phase_a"] == 1, "voltage_v"].mean()),
                "voltage_v_mean_in_B": float(sit1.loc[sit1["sit1_phase_b"] == 1, "voltage_v"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_overview(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160}
    )
    plot_df = df.set_index("timestamp").resample("5min").median(numeric_only=True).reset_index()
    fig, axes = plt.subplots(5, 1, figsize=(15, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.0, 1.0, 1.0, 0.5]})

    ax = axes[0]
    ax.plot(plot_df["timestamp"], plot_df["load_w"], color="#4C566A", lw=1.2, label="Load (measured)")
    ax.plot(plot_df["timestamp"], plot_df["pv_w"], color="#EBCB8B", lw=1.2, label="PV bus")
    ax.plot(plot_df["timestamp"], plot_df["grid_w"], color="#BF616A", lw=1.0, label="Grid")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper left", ncol=3, frameon=False)

    ax = axes[1]
    ax.plot(plot_df["timestamp"], plot_df["voltage_v"], color="#B48EAD", lw=1.2)
    ax.axhline(4.2, color="#BF616A", ls="--", lw=0.9, label="Cutoff 4.2V")
    ax.axhline(5.6, color="#A3BE8C", ls="--", lw=0.9, label="Discharge 5.6V")
    ax.axhline(8.5, color="#5E81AC", ls="--", lw=0.9, label="Charge 8.5V")
    ax.set_ylabel("Battery V")
    ax.legend(loc="upper left", ncol=3, frameon=False)

    ax = axes[2]
    ax.plot(plot_df["timestamp"], plot_df["current_raw_ma"], color="#2E3440", lw=1.0)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel("Firmware\ncurrent (mA)")

    ax = axes[3]
    ax.plot(plot_df["timestamp"], plot_df["soc_calc"] * 100.0, color="#5E81AC", lw=1.2)
    ax.set_ylim(-2, 102)
    ax.set_ylabel("SoC (%)")

    ax = axes[4]
    sit1 = df["situation_code"] == 1
    phase_a_pts = df.loc[sit1 & (df["sit1_phase_a"] == 1), "timestamp"]
    phase_b_pts = df.loc[sit1 & (df["sit1_phase_b"] == 1), "timestamp"]
    other_pts = df.loc[sit1 & (df["sit1_other"] == 1), "timestamp"]
    ax.scatter(phase_a_pts, np.full(len(phase_a_pts), 1), s=4, color="#BF616A",
               label=f"A: isolated drop  ({len(phase_a_pts)})")
    ax.scatter(phase_b_pts, np.full(len(phase_b_pts), 2), s=4, color="#5E81AC",
               label=f"B: firmware CV charge  ({len(phase_b_pts)})")
    ax.scatter(other_pts, np.full(len(other_pts), 3), s=4, color="0.6",
               label=f"other  ({len(other_pts)})")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["A", "B", "other"])
    ax.set_ylim(0.5, 3.5)
    ax.set_ylabel("Phase")
    ax.legend(loc="upper left", ncol=3, frameon=False)

    for ax in axes:
        ax.grid(True, axis="y", color="0.9", lw=0.6)

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=8))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    axes[-1].set_xlim(pd.Timestamp("2026-05-15 00:00"), pd.Timestamp("2026-05-18 23:59"))

    fig.suptitle(
        "Scenario 1 phases 2026-05-15..18: A=isolated drop (load bus off), B=firmware CV charge",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_zoom_0515_evening(df: pd.DataFrame, out_path: Path) -> None:
    start = pd.Timestamp("2026-05-15 17:30")
    end = pd.Timestamp("2026-05-15 22:30")
    z = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    z = z.set_index("timestamp").resample("1min").median(numeric_only=True).reset_index()

    fig, axes = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True)

    ax = axes[0]
    ax.plot(z["timestamp"], z["load_w"], color="#4C566A", lw=1.2, label="Load (measured)")
    ax.plot(z["timestamp"], z["grid_w"], color="#BF616A", lw=1.1, label="Grid")
    ax.plot(z["timestamp"], z["pv_w"], color="#EBCB8B", lw=1.0, label="PV")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper left", ncol=3, frameon=False)

    ax = axes[1]
    ax.plot(z["timestamp"], z["voltage_v"], color="#B48EAD", lw=1.2)
    ax.axhline(4.2, color="#BF616A", ls="--", lw=0.9)
    ax.axhline(5.6, color="#A3BE8C", ls="--", lw=0.9)
    ax.set_ylabel("Battery V")

    ax = axes[2]
    ax.plot(z["timestamp"], z["current_raw_ma"], color="#2E3440", lw=1.0)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel("Current (mA)")

    ax = axes[3]
    ax.step(z["timestamp"], z["situation_code"], where="post", color="#434C5E", lw=1.1)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.5, 4.5)
    ax.set_ylabel("Mode")

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlim(start, end)

    fig.suptitle("2026-05-15 evening: load bus collapses to ~0W during sit=1", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    flagged = classify(raw)
    flagged.to_csv(OUT_DIR / "raw_0515_0518_phases.csv", index=False, encoding="utf-8-sig")
    summary = summarize(flagged)
    summary.to_csv(OUT_DIR / "scenario1_phases_summary.csv", index=False, encoding="utf-8-sig")

    plot_overview(flagged, OUT_DIR / "scenario1_phases_overview.png")
    plot_zoom_0515_evening(flagged, OUT_DIR / "scenario1_phases_zoom_0515_evening.png")

    print(f"Wrote outputs to: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


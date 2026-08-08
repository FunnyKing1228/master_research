"""Audit why grid power rises during Scenario 1 windows on 2026-05-15..18.

Findings are documented in:
  data/raw/figures/deployment_0515_0518_scenario1_audit/

For each 10-second raw sample inside Scenario 1, we flag whether the firmware
behaviour matches a real battery discharge (negative current, ~5.6V) or a grid
charging event (positive current, voltage closer to 8.5V).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "figures" / "deployment_0515_0518_scenario1_audit"
DAYS = pd.date_range("2026-05-15", "2026-05-18", freq="D")

DISCHARGE_V_MAX = 6.0   # SLFB discharge nominal 5.6V; anything above is suspicious
CHARGE_CURRENT_MIN_MA = 20.0


def _read_deployment(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()
    header, data = rows[0], rows[1:]
    if (
        "guard_block_discharge_intent_threshold" not in header
        and any(len(r) == len(header) + 1 for r in data)
    ):
        idx = header.index("guard_block_load_over_discharge_limit") + 1
        header = header[:idx] + ["guard_block_discharge_intent_threshold"] + header[idx:]
    fixed: list[list[str]] = []
    for r in data:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        else:
            r = r[: len(header)]
        fixed.append(r)
    return pd.DataFrame(fixed, columns=header)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frames: list[pd.DataFrame] = []
    dep_frames: list[pd.DataFrame] = []
    for day in DAYS:
        rp = RAW_DIR / f"raw_data_v2_{day:%Y-%m-%d}.csv"
        dp = RAW_DIR / f"deployment_v2_{day:%Y-%m-%d}.csv"
        if rp.exists():
            df = pd.read_csv(rp)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            raw_frames.append(df)
        if dp.exists():
            df = _read_deployment(dp)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            dep_frames.append(df)

    raw = pd.concat(raw_frames, ignore_index=True).sort_values("timestamp")
    dep = pd.concat(dep_frames, ignore_index=True).sort_values("timestamp")
    numeric_raw = [
        "voltage_v",
        "current_ma",
        "current_raw_ma",
        "grid_p_mw",
        "load_p_mw",
        "bus_p_mw",
        "mppt_p_mw",
        "soc_calc",
        "situation_code",
    ]
    for c in numeric_raw:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    numeric_dep = [
        "load_kw",
        "pv_kw",
        "action_power_kw",
        "power_mw_cmd",
        "flow_pct_cmd",
        "situation_code",
        "soc",
        "batt_v_mean",
        "batt_i_mean_ma",
        "load_p_mean_mW",
        "bus_p_mean_mW",
    ]
    for c in numeric_dep:
        if c in dep.columns:
            dep[c] = pd.to_numeric(dep[c], errors="coerce")
    return raw, dep


def classify(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["grid_w"] = df["grid_p_mw"] / 1000.0
    df["load_w"] = df["load_p_mw"] / 1000.0
    df["pv_w"] = df["bus_p_mw"] / 1000.0
    df["batt_power_w_measured"] = df["voltage_v"] * df["current_raw_ma"] / 1000.0
    sit1 = df["situation_code"] == 1
    looks_like_charge = (
        sit1
        & (df["current_raw_ma"] > CHARGE_CURRENT_MIN_MA)
        & (df["voltage_v"] > BATTERY_CUTOFF_V)
    )
    looks_like_real_discharge = (
        sit1
        & (df["current_raw_ma"] < -CHARGE_CURRENT_MIN_MA)
        & (df["voltage_v"] < DISCHARGE_V_MAX)
    )
    df["sit1_apparent_charge"] = looks_like_charge.astype(int)
    df["sit1_real_discharge"] = looks_like_real_discharge.astype(int)
    df["sit1_total"] = sit1.astype(int)
    return df


BATTERY_CUTOFF_V = 4.2


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    sit1 = df[df["situation_code"] == 1]
    by_day = sit1.groupby(sit1["timestamp"].dt.date).agg(
        sit1_samples=("timestamp", "size"),
        sit1_apparent_charge=("sit1_apparent_charge", "sum"),
        sit1_real_discharge=("sit1_real_discharge", "sum"),
        voltage_mean=("voltage_v", "mean"),
        raw_current_mean=("current_raw_ma", "mean"),
        proc_current_mean=("current_ma", "mean"),
        grid_w_mean=("grid_w", "mean"),
        load_w_mean=("load_w", "mean"),
        pv_w_mean=("pv_w", "mean"),
        batt_w_measured_mean=("batt_power_w_measured", "mean"),
    )
    by_day["sit1_apparent_charge_pct"] = (
        by_day["sit1_apparent_charge"] / by_day["sit1_samples"] * 100.0
    )
    by_day["sit1_real_discharge_pct"] = (
        by_day["sit1_real_discharge"] / by_day["sit1_samples"] * 100.0
    )
    return by_day.reset_index().rename(columns={"timestamp": "date"})


def plot_audit(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160}
    )
    plot_df = df.set_index("timestamp").resample("2min").median(numeric_only=True).reset_index()

    fig, axes = plt.subplots(4, 1, figsize=(14, 9.5), sharex=True)

    ax = axes[0]
    ax.plot(plot_df["timestamp"], plot_df["grid_w"], color="#BF616A", lw=1.2, label="Grid power")
    ax.plot(plot_df["timestamp"], plot_df["load_w"], color="#4C566A", lw=1.0, label="Load power")
    ax.plot(plot_df["timestamp"], plot_df["pv_w"], color="#EBCB8B", lw=1.0, alpha=0.9, label="PV bus")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper left", ncol=3, frameon=False)

    ax = axes[1]
    ax.plot(plot_df["timestamp"], plot_df["voltage_v"], color="#B48EAD", lw=1.1, label="Battery voltage")
    ax.axhline(4.2, color="#BF616A", ls="--", lw=0.8, label="Cutoff 4.2V")
    ax.axhline(5.6, color="#A3BE8C", ls="--", lw=0.8, label="Discharge 5.6V")
    ax.axhline(8.5, color="#5E81AC", ls="--", lw=0.8, label="Charge 8.5V")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper left", ncol=4, frameon=False)

    ax = axes[2]
    ax.plot(plot_df["timestamp"], plot_df["current_raw_ma"], color="#2E3440", lw=1.0, label="Firmware raw current")
    ax.plot(plot_df["timestamp"], plot_df["current_ma"], color="#D08770", lw=0.8, alpha=0.85, label="SoC accounting current")
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper left", ncol=2, frameon=False)

    ax = axes[3]
    sit = plot_df["situation_code"]
    ax.step(plot_df["timestamp"], sit, where="post", color="#4C566A", lw=1.1, label="Situation code")
    sit1_mask = df["situation_code"] == 1
    ax.scatter(
        df.loc[sit1_mask & df["sit1_apparent_charge"].astype(bool), "timestamp"],
        np.full(int((sit1_mask & df["sit1_apparent_charge"].astype(bool)).sum()), 1.5),
        s=4,
        color="#BF616A",
        label="Sit1 but firmware looks like CHARGE",
    )
    ax.scatter(
        df.loc[sit1_mask & df["sit1_real_discharge"].astype(bool), "timestamp"],
        np.full(int((sit1_mask & df["sit1_real_discharge"].astype(bool)).sum()), 0.5),
        s=4,
        color="#A3BE8C",
        label="Sit1 with real discharge",
    )
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.2, 4.5)
    ax.set_ylabel("Mode")
    ax.legend(loc="upper left", ncol=3, frameon=False)

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=8))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    axes[-1].set_xlim(pd.Timestamp("2026-05-15 00:00"), pd.Timestamp("2026-05-18 23:59"))

    fig.suptitle(
        "Scenario 1 audit 2026-05-15..18: firmware mostly looks like CHARGE, not DISCHARGE",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_zoom(df: pd.DataFrame, out_path: Path) -> None:
    start = pd.Timestamp("2026-05-15 18:00")
    end = pd.Timestamp("2026-05-15 22:30")
    z = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    z = z.set_index("timestamp").resample("1min").median(numeric_only=True).reset_index()

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    ax = axes[0]
    ax.plot(z["timestamp"], z["grid_w"], color="#BF616A", lw=1.2, label="Grid")
    ax.plot(z["timestamp"], z["load_w"], color="#4C566A", lw=1.0, label="Load")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper left", ncol=2, frameon=False)

    ax = axes[1]
    ax.plot(z["timestamp"], z["voltage_v"], color="#B48EAD", lw=1.2)
    ax.axhline(4.2, color="#BF616A", ls="--", lw=0.8)
    ax.axhline(5.6, color="#A3BE8C", ls="--", lw=0.8)
    ax.axhline(8.5, color="#5E81AC", ls="--", lw=0.8)
    ax.set_ylabel("Battery V")

    ax = axes[2]
    ax.plot(z["timestamp"], z["current_raw_ma"], color="#2E3440", lw=1.1, label="Firmware raw current")
    ax.plot(z["timestamp"], z["current_ma"], color="#D08770", lw=0.8, label="SoC accounting current")
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper left", ncol=2, frameon=False)

    ax = axes[3]
    ax.step(z["timestamp"], z["situation_code"], where="post", color="#4C566A", lw=1.2)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.2, 4.5)
    ax.set_ylabel("Mode")

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlim(start, end)

    fig.suptitle("2026-05-15 evening: Scenario 1 commanded but firmware charges from grid", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, _dep = load_all()
    raw_flagged = classify(raw)

    summary = summarize(raw_flagged)
    summary.to_csv(OUT_DIR / "scenario1_audit_summary.csv", index=False, encoding="utf-8-sig")
    raw_flagged.to_csv(OUT_DIR / "raw_0515_0518_scenario1_flagged.csv", index=False, encoding="utf-8-sig")

    plot_audit(raw_flagged, OUT_DIR / "scenario1_audit_overview.png")
    plot_zoom(raw_flagged, OUT_DIR / "scenario1_audit_zoom_0515_evening.png")

    print(f"Wrote audit outputs to: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


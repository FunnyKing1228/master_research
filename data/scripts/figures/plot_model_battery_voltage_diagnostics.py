from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "figures" / "model_battery_voltage_diagnostics"


def _load_kind(kind: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    frames = []
    prefix = f"{kind}_v2_"
    for path in sorted(RAW_DIR.glob(f"{prefix}2026-*.csv")):
        date = path.stem.replace(prefix, "")
        if date < start_date:
            continue
        if end_date and date > end_date:
            continue
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df["date"] = date
        for col in df.columns:
            if col not in ("timestamp", "date", "session_id", "experiment_name", "model_file", "current_mode", "load_source"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def _sensor_signature(df: pd.DataFrame) -> pd.Series:
    cols = [
        "voltage_v",
        "charge_voltage_v",
        "current_raw_ma",
        "mppt_p_mw",
        "bus_p_mw",
        "load_p_mw",
        "grid_p_mw",
        "speed_percent",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].round(3).astype(str).agg("|".join, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-06-03")
    parser.add_argument("--end-date", default="2026-06-08")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    raw = _load_kind("raw_data", args.start_date, args.end_date)
    dep = _load_kind("deployment", args.start_date, args.end_date)
    if raw.empty:
        raise SystemExit("No raw_data_v2 files found.")

    sig = _sensor_signature(raw)
    raw["sensor_duplicate"] = sig.eq(sig.shift()).astype(int)
    raw["stuck_260_331"] = (
        np.isclose(raw.get("voltage_v", np.nan), 2.60)
        & np.isclose(raw.get("charge_voltage_v", np.nan), 3.31)
        & np.isclose(raw.get("current_raw_ma", np.nan), 58.0)
    ).astype(int)
    raw["below_cutoff"] = (raw.get("voltage_v", np.nan) < 4.2).astype(int)
    raw["firmware_soc_zero"] = (raw.get("soc_percent", np.nan) <= 0.0).astype(int)

    raw_5 = raw.set_index("timestamp").resample("5min").mean(numeric_only=True).reset_index()
    dep_15 = dep.copy()

    fig = plt.figure(figsize=(19, 14.5))
    gs = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.05, 1.25, 1.05, 1.0, 1.1],
        hspace=0.58,
    )
    ax_pv = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_pv)
    ax_i = fig.add_subplot(gs[2], sharex=ax_pv)
    ax_soc = fig.add_subplot(gs[3], sharex=ax_pv)
    ax_act = fig.add_subplot(gs[4], sharex=ax_pv)

    if "mppt_p_mw" in raw_5:
        ax_pv.plot(raw_5["timestamp"], raw_5["mppt_p_mw"] / 1000.0, color="#ffbf00", lw=1.8, label="MPPT power (W)")
    if "load_p_mw" in raw_5:
        ax_pv.plot(raw_5["timestamp"], raw_5["load_p_mw"] / 1000.0, color="#8c564b", lw=1.6, label="Load power (W)")
    if "bus_p_mw" in raw_5 and "load_p_mw" in raw_5:
        ratio = raw_5["bus_p_mw"] / raw_5["load_p_mw"].replace(0, np.nan)
        ax_pvb = ax_pv.twinx()
        ax_pvb.plot(raw_5["timestamp"], ratio, color="#2ca02c", lw=1.5, alpha=0.85, label="PV/load ratio")
        ax_pvb.axhline(0.8, color="#2ca02c", ls=":", lw=1.2, alpha=0.8, label="ratio=0.8")
        ax_pvb.set_ylabel("PV/load ratio")
        ax_pvb.set_ylim(0, max(1.2, float(np.nanpercentile(ratio, 98)) * 1.15 if ratio.notna().any() else 1.2))
        lines, labels = ax_pv.get_legend_handles_labels()
        lines2, labels2 = ax_pvb.get_legend_handles_labels()
        ax_pv.legend(lines + lines2, labels + labels2, loc="upper right", ncol=4)
    else:
        ax_pv.legend(loc="upper right", ncol=2)
    ax_pv.set_ylabel("Power (W)")
    ax_pv.grid(alpha=0.25)

    ax_v.plot(raw_5["timestamp"], raw_5["voltage_v"], color="#1f77b4", lw=1.5, label="Battery voltage")
    if "charge_voltage_v" in raw_5:
        ax_v.plot(raw_5["timestamp"], raw_5["charge_voltage_v"], color="#17becf", lw=1.0, alpha=0.7, label="Charge voltage")
    ax_v.axhline(4.2, color="red", ls="--", lw=1.0, label="4.2V cutoff")
    ax_v.axhline(2.60, color="purple", ls=":", lw=1.0, label="2.60V stuck level")
    ax_v.set_ylabel("Voltage (V)")
    ax_v.legend(loc="upper right", ncol=4)
    ax_v.grid(alpha=0.25)

    ax_i.plot(raw_5["timestamp"], raw_5["current_raw_ma"], color="#ff7f0e", lw=1.1, label="Raw current")
    ax_i.plot(raw_5["timestamp"], raw_5["current_ma"], color="#d62728", lw=1.0, alpha=0.7, label="Signed/control current")
    ax_i.axhline(0, color="black", lw=0.8)
    ax_i.set_ylabel("Current (mA)")
    ax_i.legend(loc="upper right")
    ax_i.grid(alpha=0.25)

    ax_soc.plot(raw_5["timestamp"], raw_5["soc_calc"] * 100.0, color="#2ca02c", lw=1.4, label="Calculated SoC")
    ax_soc.plot(raw_5["timestamp"], raw_5["soc_percent"], color="#7f7f7f", lw=1.0, alpha=0.75, label="Firmware SoC")
    ax_soc.set_ylabel("SoC (%)")
    ax_soc.legend(loc="upper right")
    ax_soc.grid(alpha=0.25)

    if not dep_15.empty:
        ax_act.step(dep_15["timestamp"], dep_15["action_raw_kw"] * 1000.0, where="post", color="#9467bd", lw=1.8, label="Model raw action (W)")
        ax_act.step(dep_15["timestamp"], dep_15["action_power_kw"] * 1000.0, where="post", color="#2ca02c", lw=2.0, label="Executed action (W)")
        ax_act.step(dep_15["timestamp"], dep_15["power_mw_cmd"] / 1000.0, where="post", color="#1f77b4", lw=1.8, ls="--", alpha=0.9, label="Command power magnitude (W)")
    ax_act.set_ylabel("Model / command (W)")
    ax_act.legend(loc="upper right", ncol=3)
    ax_act.grid(alpha=0.25)

    # Highlight periods where the frozen signature dominates in 5-minute bins.
    stuck_periods = raw_5[raw_5["stuck_260_331"] > 0.8]
    for ax in (ax_pv, ax_v, ax_i, ax_soc, ax_act):
        for ts in stuck_periods["timestamp"]:
            ax.axvspan(ts, ts + pd.Timedelta(minutes=5), color="crimson", alpha=0.06)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))

    summary = {
        "start": str(raw["timestamp"].min()),
        "end": str(raw["timestamp"].max()),
        "raw_samples": int(len(raw)),
        "deployment_steps": int(len(dep_15)),
        "below_cutoff_ratio": float(raw["below_cutoff"].mean()),
        "firmware_soc_zero_ratio": float(raw["firmware_soc_zero"].mean()),
        "stuck_260_331_ratio": float(raw["stuck_260_331"].mean()),
        "duplicate_ratio": float(raw["sensor_duplicate"].mean()),
    }
    if not dep_15.empty:
        for col in ["guard_block_voltage_cutoff", "voltage_cutoff_active", "cutoff_soc_fallback_applied"]:
            if col in dep_15:
                summary[f"{col}_steps"] = int(pd.to_numeric(dep_15[col], errors="coerce").fillna(0).sum())
        if "action_raw_kw" in dep_15:
            summary["model_discharge_intent_steps"] = int((dep_15["action_raw_kw"] < -1e-6).sum())
        if "action_power_kw" in dep_15:
            summary["executed_discharge_steps"] = int((dep_15["action_power_kw"] < -1e-6).sum())

    summary_path = OUT_DIR / "model_battery_voltage_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    fig.suptitle(
        "Model vs Battery Diagnostics: voltage anomaly is dominated by repeated low-voltage sensor state",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.935, bottom=0.07, hspace=0.58)
    fig_path = OUT_DIR / "model_battery_voltage_diagnostics.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"FIG={fig_path}")
    print(f"SUMMARY={summary_path}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()


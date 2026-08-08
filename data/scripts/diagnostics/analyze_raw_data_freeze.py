from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "figures" / "raw_data_freeze_analysis"


SIGNATURE_COLS = [
    "voltage_v",
    "charge_voltage_v",
    "current_raw_ma",
    "solar_p_mw",
    "mppt_p_mw",
    "bus_p_mw",
    "load_p_mw",
    "grid_p_mw",
    "speed_percent",
    "situation_code",
]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="ignore")
    df["source_file"] = path.name
    return df


def _signature(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in SIGNATURE_COLS if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    rounded = df[cols].apply(pd.to_numeric, errors="coerce").round(3)
    return rounded.astype(str).agg("|".join, axis=1)


def _run_lengths(sig: pd.Series, ts: pd.Series) -> pd.DataFrame:
    if sig.empty:
        return pd.DataFrame()
    group = sig.ne(sig.shift()).cumsum()
    rows = []
    for _, idx in sig.groupby(group).groups.items():
        block = ts.loc[idx]
        rows.append(
            {
                "start": block.iloc[0],
                "end": block.iloc[-1],
                "n": len(block),
                "duration_min": (block.iloc[-1] - block.iloc[0]).total_seconds() / 60.0,
                "signature": sig.loc[idx[0]],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--freeze-minutes", type=float, default=30.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(RAW_DIR.glob("raw_data_v2_2026-*.csv"))
    frames = []
    daily_rows = []
    run_frames = []

    for path in paths:
        date = path.stem.replace("raw_data_v2_", "")
        if args.start_date and date < args.start_date:
            continue
        if args.end_date and date > args.end_date:
            continue
        df = _read_csv(path)
        if df.empty:
            continue
        sig = _signature(df)
        runs = _run_lengths(sig, df["timestamp"])
        if not runs.empty:
            runs["date"] = date
            run_frames.append(runs)
        same_prev = sig.eq(sig.shift()).fillna(False)
        intervals = df["timestamp"].diff().dt.total_seconds().dropna()
        daily_rows.append(
            {
                "date": date,
                "samples": len(df),
                "start": df["timestamp"].iloc[0],
                "end": df["timestamp"].iloc[-1],
                "coverage_hours": (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600.0,
                "median_gap_s": intervals.median() if not intervals.empty else np.nan,
                "max_gap_s": intervals.max() if not intervals.empty else np.nan,
                "unique_sensor_states": sig.nunique(),
                "duplicate_ratio": float(same_prev.mean()),
                "max_duplicate_run_samples": int(runs["n"].max()) if not runs.empty else 0,
                "max_duplicate_run_min": float(runs["duration_min"].max()) if not runs.empty else 0.0,
                "voltage_min": pd.to_numeric(df.get("voltage_v"), errors="coerce").min(),
                "voltage_max": pd.to_numeric(df.get("voltage_v"), errors="coerce").max(),
                "soc_start": pd.to_numeric(df.get("soc_calc"), errors="coerce").iloc[0],
                "soc_end": pd.to_numeric(df.get("soc_calc"), errors="coerce").iloc[-1],
            }
        )
        frames.append(df)

    if not frames:
        raise SystemExit("No raw_data_v2 files found in the requested range.")

    raw = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    daily = pd.DataFrame(daily_rows)
    runs_all = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    freeze_runs = runs_all[runs_all["duration_min"] >= args.freeze_minutes].copy()

    daily_path = OUT_DIR / "raw_data_daily_summary.csv"
    runs_path = OUT_DIR / "raw_data_freeze_runs.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    freeze_runs.to_csv(runs_path, index=False, encoding="utf-8-sig")

    first_freeze = freeze_runs.sort_values("start").iloc[0] if not freeze_runs.empty else None

    # Downsample for readable plotting.
    plot = raw.set_index("timestamp").resample("5min").mean(numeric_only=True).reset_index()
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.2, 1.2, 1.0], hspace=0.24)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3])

    if "voltage_v" in plot:
        ax0.plot(plot["timestamp"], plot["voltage_v"], label="Battery voltage (V)", color="#1f77b4", lw=1.5)
    ax0.axhline(4.2, color="red", ls="--", lw=1.0, label="4.2V cutoff")
    ax0.set_ylabel("Voltage (V)")
    ax0.legend(loc="upper right")
    ax0.grid(alpha=0.25)

    if "soc_calc" in plot:
        ax1.plot(plot["timestamp"], plot["soc_calc"] * 100.0, label="Calculated SoC (%)", color="#2ca02c", lw=1.5)
    if "current_raw_ma" in plot:
        ax1b = ax1.twinx()
        ax1b.plot(plot["timestamp"], plot["current_raw_ma"], label="Raw current (mA)", color="#ff7f0e", lw=1.0, alpha=0.65)
        ax1b.set_ylabel("Current (mA)")
    ax1.set_ylabel("SoC (%)")
    ax1.grid(alpha=0.25)

    for col, label, color in [
        ("mppt_p_mw", "MPPT power", "#9467bd"),
        ("load_p_mw", "Load power", "#8c564b"),
        ("grid_p_mw", "Grid power", "#7f7f7f"),
    ]:
        if col in plot:
            ax2.plot(plot["timestamp"], plot[col] / 1000.0, label=label, lw=1.1, color=color)
    ax2.set_ylabel("Power (W)")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.25)

    if not freeze_runs.empty:
        for ax in (ax0, ax1, ax2):
            for _, row in freeze_runs.iterrows():
                ax.axvspan(row["start"], row["end"], color="crimson", alpha=0.10)

    dates = pd.to_datetime(daily["date"])
    ax3.bar(dates, daily["duplicate_ratio"] * 100.0, color="#d62728", alpha=0.75, label="Duplicate sensor ratio")
    ax3b = ax3.twinx()
    ax3b.plot(dates, daily["unique_sensor_states"], marker="o", color="#1f77b4", label="Unique sensor states")
    ax3.set_ylabel("Duplicate ratio (%)")
    ax3b.set_ylabel("Unique states")
    ax3.set_ylim(0, 105)
    ax3.grid(alpha=0.25)

    for ax in (ax0, ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    title = "Raw Data Freeze / Crash Analysis"
    if first_freeze is not None:
        title += f"  | first long freeze: {first_freeze['start']:%Y-%m-%d %H:%M}"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig_path = OUT_DIR / "raw_data_freeze_analysis.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"FIG={fig_path}")
    print(f"DAILY={daily_path}")
    print(f"RUNS={runs_path}")
    if first_freeze is not None:
        print(
            "FIRST_FREEZE="
            f"{first_freeze['start']} to {first_freeze['end']} "
            f"({first_freeze['duration_min']:.1f} min, n={int(first_freeze['n'])})"
        )
    print("LATEST_DAILY_SUMMARY")
    print(daily.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()


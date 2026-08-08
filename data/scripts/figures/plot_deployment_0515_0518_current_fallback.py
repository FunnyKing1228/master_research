"""Plot 2026-05-15..18 deployment logs with battery current and cutoff fallback.

The original logs were collected before configurable cutoff SoC fallback was
re-enabled. This script reconstructs what the control SoC would have looked
like if voltage cutoff had reset the tracker to 20% once per cutoff event.
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
OUT_DIR = RAW_DIR / "figures" / "deployment_0515_0518_current_fallback"
DAYS = pd.date_range("2026-05-15", "2026-05-18", freq="D")

CUTOFF_V = 4.2
RECOVER_V = 5.0
COOLDOWN = pd.Timedelta(minutes=5)
FALLBACK_SOC = 0.20
BATTERY_CAPACITY_MAH = 2000.0
BATTERY_EFFICIENCY = 0.95


def _load_daily(prefix: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in DAYS:
        path = RAW_DIR / f"{prefix}_{day:%Y-%m-%d}.csv"
        if not path.exists():
            continue
        df = _read_csv_flexible(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {prefix}_*.csv files found for {DAYS[0]:%Y-%m-%d}..{DAYS[-1]:%Y-%m-%d}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    for col in out.columns:
        if col not in {"timestamp", "source_file", "session_id", "experiment_name", "model_file", "current_mode", "load_source", "battery_id"}:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().any():
                out[col] = converted
    return out.reset_index(drop=True)


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read logs that may contain rows from two adjacent schema versions."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data_rows = rows[1:]

    if (
        path.name.startswith("deployment_v2")
        and "guard_block_discharge_intent_threshold" not in header
        and any(len(r) == len(header) + 1 for r in data_rows)
    ):
        insert_at = header.index("guard_block_load_over_discharge_limit") + 1
        header = (
            header[:insert_at]
            + ["guard_block_discharge_intent_threshold"]
            + header[insert_at:]
        )

    normalized: list[list[str]] = []
    for row in data_rows:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        normalized.append(row)

    return pd.DataFrame(normalized, columns=header)


def simulate_cutoff_fallback(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay SoC from processed current, then reset control SoC to 20% on cutoff."""
    df = raw.copy()
    df["soc_calc"] = pd.to_numeric(df["soc_calc"], errors="coerce").ffill()
    df["voltage_v"] = pd.to_numeric(df["voltage_v"], errors="coerce").fillna(0.0)
    df["current_ma"] = pd.to_numeric(df["current_ma"], errors="coerce").fillna(0.0)

    replay_no_fallback: list[float] = []
    sim_soc: list[float] = []
    cutoff_active: list[int] = []
    fallback_applied: list[int] = []
    events: list[dict[str, object]] = []

    active = False
    trigger_time: pd.Timestamp | None = None
    prev_ts: pd.Timestamp | None = None
    prev_logged_soc = float(df["soc_calc"].iloc[0])
    sim_base = float(df["soc_calc"].iloc[0])
    sim = sim_base

    for i, row in df.iterrows():
        ts = row["timestamp"]
        logged_soc = float(row["soc_calc"])
        voltage = float(row["voltage_v"])
        current_ma = float(row["current_ma"])

        # Deployment logs can span process restarts with a new initial SoC.
        # Treat large logged-SoC jumps as session resets, not physical current.
        if i > 0 and abs(logged_soc - prev_logged_soc) > 0.05:
            sim_base = logged_soc
            sim = logged_soc
            active = False
            trigger_time = None
            prev_ts = ts

        if prev_ts is not None:
            dt_sec = max(0.0, min((ts - prev_ts).total_seconds(), 3600.0))
            dt_h = dt_sec / 3600.0
            if current_ma > 0:
                delta_soc = current_ma * dt_h * BATTERY_EFFICIENCY / BATTERY_CAPACITY_MAH
            elif current_ma < 0:
                delta_soc = current_ma * dt_h / BATTERY_EFFICIENCY / BATTERY_CAPACITY_MAH
            else:
                delta_soc = 0.0
            sim_base = float(np.clip(sim_base + delta_soc, 0.0, 1.0))
            sim = float(np.clip(sim + delta_soc, 0.0, 1.0))

        applied = 0
        if voltage > 0:
            if voltage < CUTOFF_V and not active:
                before = sim
                sim = FALLBACK_SOC
                active = True
                trigger_time = ts
                applied = 1
                events.append(
                    {
                        "timestamp": ts,
                        "voltage_v": voltage,
                        "soc_original_before": before,
                        "soc_fallback_after": sim,
                        "correction_pct_point": (before - sim) * 100.0,
                    }
                )
            elif active and trigger_time is not None and voltage >= RECOVER_V and ts - trigger_time >= COOLDOWN:
                active = False
                trigger_time = None

        replay_no_fallback.append(sim_base)
        sim_soc.append(sim)
        cutoff_active.append(1 if active else 0)
        fallback_applied.append(applied)
        prev_ts = ts
        prev_logged_soc = logged_soc

    df["soc_current_replay"] = replay_no_fallback
    df["soc_fallback20"] = sim_soc
    df["cutoff_active_replay"] = cutoff_active
    df["fallback20_applied"] = fallback_applied
    events_df = pd.DataFrame(events)
    return df, events_df


def summarize(raw_sim: pd.DataFrame, dep: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day, g in raw_sim.groupby(raw_sim["timestamp"].dt.date):
        g = g.sort_values("timestamp")
        dt_h = g["timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0, upper=3600.0) / 3600.0
        current = pd.to_numeric(g["current_ma"], errors="coerce").fillna(0.0)
        voltage = pd.to_numeric(g["voltage_v"], errors="coerce").fillna(0.0)
        batt_power_w = voltage * current / 1000.0
        low_v = (voltage > 0) & (voltage < CUTOFF_V)
        day_events = events[events["timestamp"].dt.date == day] if not events.empty else events

        dep_day = dep[dep["timestamp"].dt.date == day] if not dep.empty else dep
        guard_cutoff = int(pd.to_numeric(dep_day.get("guard_block_voltage_cutoff", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        guard_load = int(pd.to_numeric(dep_day.get("guard_block_load_over_discharge_limit", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        guard_intent = int(pd.to_numeric(dep_day.get("guard_block_discharge_intent_threshold", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

        rows.append(
            {
                "date": str(day),
                "samples": len(g),
                "voltage_min_v": float(voltage[voltage > 0].min()) if (voltage > 0).any() else np.nan,
                "low_voltage_minutes": float((dt_h[low_v] * 60.0).sum()),
                "fallback_events": len(day_events),
                "max_soc_correction_pct_point": float(day_events["correction_pct_point"].max()) if len(day_events) else 0.0,
                "soc_original_start_pct": float(g["soc_calc"].iloc[0] * 100.0),
                "soc_original_end_pct": float(g["soc_calc"].iloc[-1] * 100.0),
                "soc_current_replay_end_pct": float(g["soc_current_replay"].iloc[-1] * 100.0),
                "soc_fallback20_end_pct": float(g["soc_fallback20"].iloc[-1] * 100.0),
                "charge_ah_from_current": float((current.clip(lower=0.0) * dt_h).sum() / 1000.0),
                "discharge_ah_from_current": float((-current.clip(upper=0.0) * dt_h).sum() / 1000.0),
                "battery_energy_wh_signed": float((batt_power_w * dt_h).sum()),
                "guard_cutoff_steps": guard_cutoff,
                "guard_load_over_limit_steps": guard_load,
                "guard_small_intent_steps": guard_intent,
            }
        )
    return pd.DataFrame(rows)


def plot_overview(raw_sim: pd.DataFrame, dep: pd.DataFrame, events: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(5, 1, figsize=(15, 11), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0, 0.9, 1.0, 0.55]})

    raw_plot = raw_sim.set_index("timestamp").resample("5min").median(numeric_only=True).reset_index()
    dep_plot = dep.copy()

    ax = axes[0]
    ax.plot(dep_plot["timestamp"], dep_plot["load_kw"] * 1000.0, color="#4C566A", lw=1.5, label="Load (15-min)")
    ax.plot(dep_plot["timestamp"], dep_plot["pv_kw"] * 1000.0, color="#EBCB8B", lw=1.5, label="PV bus (15-min)")
    batt_w = raw_plot["voltage_v"] * raw_plot["current_ma"] / 1000.0
    ax.fill_between(raw_plot["timestamp"], 0, batt_w, where=batt_w >= 0, color="#5E81AC", alpha=0.35, step="mid", label="Battery charge (+)")
    ax.fill_between(raw_plot["timestamp"], 0, batt_w, where=batt_w < 0, color="#BF616A", alpha=0.35, step="mid", label="Battery discharge (-)")
    ax.axhline(0, color="0.25", lw=0.8)
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper left", ncol=4, frameon=False)

    ax = axes[1]
    ax.plot(raw_plot["timestamp"], raw_plot["soc_calc"] * 100.0, color="#5E81AC", lw=1.0, alpha=0.75, label="Logged SoC")
    ax.plot(raw_plot["timestamp"], raw_plot["soc_current_replay"] * 100.0, color="#4C566A", lw=1.2, label="Current replay")
    ax.plot(raw_plot["timestamp"], raw_plot["soc_fallback20"] * 100.0, color="#D08770", lw=1.6, label="Replay: cutoff fallback to 20%")
    ax.axhspan(10, 90, color="#A3BE8C", alpha=0.10, label="10-90% band")
    ax.axhline(20, color="#BF616A", ls="--", lw=0.9, label="20% fallback / lower guard")
    ax.set_ylim(-2, 102)
    ax.set_ylabel("SoC (%)")
    ax.legend(loc="upper right", ncol=4, frameon=False)

    ax = axes[2]
    ax.plot(raw_plot["timestamp"], raw_plot["voltage_v"], color="#B48EAD", lw=1.2, label="Battery voltage")
    ax.axhline(CUTOFF_V, color="#BF616A", ls="--", lw=1.0, label="Cutoff 4.2 V")
    ax.axhline(RECOVER_V, color="#A3BE8C", ls="--", lw=1.0, label="Recover 5.0 V")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", ncol=3, frameon=False)

    ax = axes[3]
    ax.plot(raw_plot["timestamp"], raw_plot["current_ma"], color="#2E3440", lw=1.0, label="Processed current (SoC input)")
    if "current_raw_ma" in raw_plot.columns:
        ax.plot(raw_plot["timestamp"], raw_plot["current_raw_ma"], color="#88C0D0", lw=0.8, alpha=0.75, label="Raw firmware current")
    ax.axhline(0, color="0.25", lw=0.8)
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper right", ncol=2, frameon=False)

    ax = axes[4]
    sit = pd.to_numeric(dep_plot["situation_code"], errors="coerce")
    ax.step(dep_plot["timestamp"], sit, where="post", color="#434C5E", lw=1.2, label="Situation code")
    for guard_col, color, label in [
        ("guard_block_voltage_cutoff", "#BF616A", "Guard: voltage cutoff"),
        ("guard_block_load_over_discharge_limit", "#D08770", "Guard: load > discharge limit"),
        ("guard_block_discharge_intent_threshold", "#EBCB8B", "Guard: small discharge intent"),
    ]:
        if guard_col in dep_plot.columns:
            mask = pd.to_numeric(dep_plot[guard_col], errors="coerce").fillna(0) > 0
            ax.scatter(dep_plot.loc[mask, "timestamp"], np.full(mask.sum(), 4.5), s=18, color=color, label=label)
    if not events.empty:
        for ts in events["timestamp"]:
            for a in axes:
                a.axvline(ts, color="#BF616A", alpha=0.18, lw=1.0)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.6, 4.8)
    ax.set_ylabel("Mode")
    ax.legend(loc="upper left", ncol=4, frameon=False)

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    axes[-1].set_xlim(pd.Timestamp("2026-05-15 00:00"), pd.Timestamp("2026-05-18 23:59"))
    fig.suptitle("Deployment 2026-05-15 to 2026-05-18: Current, Voltage Cutoff, and 20% SoC Fallback Replay", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_fallback_zoom(raw_sim: pd.DataFrame, events: pd.DataFrame, out_path: Path) -> None:
    if events.empty:
        return

    first = events["timestamp"].min()
    start = first - pd.Timedelta(hours=6)
    end = first + pd.Timedelta(hours=18)
    z = raw_sim[(raw_sim["timestamp"] >= start) & (raw_sim["timestamp"] <= end)].copy()
    z = z.set_index("timestamp").resample("2min").median(numeric_only=True).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(z["timestamp"], z["soc_calc"] * 100.0, label="Logged SoC", color="#5E81AC", alpha=0.75)
    axes[0].plot(z["timestamp"], z["soc_current_replay"] * 100.0, label="Current replay", color="#4C566A")
    axes[0].plot(z["timestamp"], z["soc_fallback20"] * 100.0, label="Fallback replay", color="#D08770")
    axes[0].axhline(20, color="#BF616A", ls="--", lw=0.9)
    axes[0].set_ylabel("SoC (%)")
    axes[0].legend(frameon=False)

    axes[1].plot(z["timestamp"], z["voltage_v"], color="#B48EAD")
    axes[1].axhline(CUTOFF_V, color="#BF616A", ls="--", lw=0.9)
    axes[1].axhline(RECOVER_V, color="#A3BE8C", ls="--", lw=0.9)
    axes[1].set_ylabel("Voltage (V)")

    axes[2].plot(z["timestamp"], z["current_ma"], color="#2E3440")
    axes[2].axhline(0, color="0.25", lw=0.8)
    axes[2].set_ylabel("Current (mA)")

    for ts in events["timestamp"]:
        if start <= ts <= end:
            for ax in axes:
                ax.axvline(ts, color="#BF616A", alpha=0.25, lw=1.0)

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.suptitle("First Cutoff Event Zoom: What 20% Fallback Would Change", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _load_daily("raw_data_v2")
    dep = _load_daily("deployment_v2")

    raw_sim, events = simulate_cutoff_fallback(raw)
    summary = summarize(raw_sim, dep, events)

    raw_sim.to_csv(OUT_DIR / "raw_0515_0518_with_fallback20_replay.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "cutoff_fallback20_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "deployment_0515_0518_current_fallback_summary.csv", index=False, encoding="utf-8-sig")

    plot_overview(raw_sim, dep, events, OUT_DIR / "deployment_0515_0518_current_fallback_overview.png")
    plot_fallback_zoom(raw_sim, events, OUT_DIR / "deployment_0515_0518_fallback20_first_event_zoom.png")

    print(f"Wrote outputs to: {OUT_DIR}")
    print(summary.to_string(index=False))
    if not events.empty:
        print("\nFallback replay events:")
        print(events.to_string(index=False))


if __name__ == "__main__":
    main()


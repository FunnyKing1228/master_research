from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs" / "deployment_diagnostics_20260617_20260622"
DATES = pd.date_range("2026-06-17", "2026-06-22", freq="D").strftime("%Y-%m-%d").tolist()

TEXT_COLUMNS = {
    "battery_id",
    "session_id",
    "experiment_name",
    "model_file",
    "current_mode",
    "load_source",
    "soh_health_lock_reason",
    "soh_last_record_time",
    "soh_record_reason",
    "soh_model_path",
    "soh_last_prediction_time",
    "soh_prediction_status",
    "soh_prediction_method",
}


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read logs that may contain repeated headers or partial trailing rows."""
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return pd.DataFrame()

    headers: dict[int, list[str]] = {len(rows[0]): rows[0]}
    records: list[dict[str, str]] = []

    for row in rows[1:]:
        if not row:
            continue
        if row[0] == "timestamp":
            headers[len(row)] = row
            continue

        header = headers.get(len(row))
        if header is None:
            candidates = sorted(headers.values(), key=len)
            header = next((h for h in candidates if len(h) >= len(row)), candidates[-1])
            header = header[: len(row)]
        records.append(dict(zip(header, row)))

    return pd.DataFrame(records)


def _read_daily(prefix: str) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    for date in DATES:
        path = RAW_DIR / f"{prefix}_v2_{date}.csv"
        if not path.exists():
            continue

        df = _read_csv_flexible(path)
        if df.empty or "timestamp" not in df:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

        for col in df.columns:
            if col not in TEXT_COLUMNS and col != "timestamp" and col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        frames.append(df)
        sources.append(str(path.relative_to(ROOT)))

    if not frames:
        return pd.DataFrame(), sources
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("timestamp"), sources


def _series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _resample_numeric(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    numeric = df.set_index("timestamp").select_dtypes(include=[np.number])
    return numeric.resample(rule).mean().reset_index()


def _setup_time_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))


def _save_overview(dep: pd.DataFrame, raw: pd.DataFrame) -> Path:
    raw_5 = _resample_numeric(raw, "5min")

    fig, axes = plt.subplots(7, 1, figsize=(18, 17), sharex=True)

    ax = axes[0]
    if "action_raw_kw" in dep:
        ax.step(dep["timestamp"], _series(dep, "action_raw_kw") * 1000.0, where="post", lw=1.2, label="Model raw action")
    ax.step(dep["timestamp"], _series(dep, "action_power_kw") * 1000.0, where="post", lw=1.5, label="Final action")
    ax.step(dep["timestamp"], _series(dep, "power_mw_cmd") / 1000.0, where="post", lw=1.0, ls="--", label="Command magnitude")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Command (W)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[1]
    if "batt_p_mean_mW" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_p_mean_mW") / 1000.0, lw=1.2, label="Deployment batt power")
    if not raw_5.empty:
        raw_batt_w = _series(raw_5, "voltage_v") * _series(raw_5, "current_ma") / 1000.0
        ax.plot(raw_5["timestamp"], raw_batt_w, lw=1.2, alpha=0.8, label="Raw batt V*I")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Batt P (W)")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    ax = axes[2]
    if "batt_i_mean_ma" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_i_mean_ma"), lw=1.2, label="Deployment batt current")
    if not raw_5.empty and "current_ma" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "current_ma"), lw=1.1, alpha=0.85, label="Raw signed current")
    if not raw_5.empty and "current_raw_ma" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "current_raw_ma"), lw=0.9, alpha=0.65, label="Raw abs/current_raw")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[3]
    if "batt_v_mean" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_v_mean"), lw=1.2, label="Deployment batt voltage")
    if not raw_5.empty and "voltage_v" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "voltage_v"), lw=1.2, alpha=0.85, label="Raw batt voltage")
    ax.axhline(6.0, color="#d62728", lw=1.0, ls="--", alpha=0.8, label="6 V diagnostic ref.")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[4]
    if "soc" in dep:
        ax.plot(dep["timestamp"], _series(dep, "soc") * 100.0, lw=1.4, label="Deployment SoC")
    if not raw_5.empty and "soc_calc" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "soc_calc") * 100.0, lw=1.1, alpha=0.8, label="Raw control SoC")
    if not raw_5.empty and "soc_coulomb" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "soc_coulomb") * 100.0, lw=1.0, ls="--", alpha=0.75, label="Raw coulomb SoC")
    ax.axhline(20, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("SoC (%)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[5]
    for col, label in [("load_kw", "Deployment load"), ("pv_kw", "Deployment PV support")]:
        if col in dep:
            ax.plot(dep["timestamp"], _series(dep, col) * 1000.0, lw=1.2, label=label)
    if not raw_5.empty:
        for col, label in [("load_p_mw", "Raw load"), ("grid_p_mw", "Raw grid demand"), ("mppt_p_mw", "Raw MPPT/PV")]:
            if col in raw_5:
                ax.plot(raw_5["timestamp"], _series(raw_5, col) / 1000.0, lw=1.0, alpha=0.8, label=label)
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper right", ncol=5)
    _setup_time_axis(ax)

    ax = axes[6]
    if "flow_pct_cmd" in dep:
        ax.step(dep["timestamp"], _series(dep, "flow_pct_cmd"), where="post", lw=1.2, label="Command flow")
    if "action_flow_pct" in dep:
        ax.step(dep["timestamp"], _series(dep, "action_flow_pct"), where="post", lw=1.0, alpha=0.75, label="Model flow")
    if not raw_5.empty and "speed_percent" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "speed_percent"), lw=1.0, alpha=0.75, label="Raw speed")
    ax2 = ax.twinx()
    if "situation_code" in dep:
        ax2.step(dep["timestamp"], _series(dep, "situation_code"), where="post", lw=0.9, color="#7f7f7f", label="Deployment situation")
    if not raw_5.empty and "situation_code" in raw_5:
        ax2.step(raw_5["timestamp"], _series(raw_5, "situation_code"), where="post", lw=0.8, color="#bcbd22", alpha=0.75, label="Raw situation")
    ax.set_ylabel("Flow (%)")
    ax2.set_ylabel("Situation")
    ax.set_xlabel("Time")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=4)
    _setup_time_axis(ax)

    fig.suptitle("Deployment Diagnostics Overview: 2026-06-17 to 2026-06-22", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "overview_timeseries_20260617_20260622.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _save_intent_vs_actual(dep: pd.DataFrame, raw: pd.DataFrame) -> Path:
    raw_1 = _resample_numeric(raw, "1min")
    model_intent = _series(dep, "action_raw_kw").lt(-1e-6)
    command_discharge = _series(dep, "action_power_kw").lt(-1e-6)
    cutoff_cols = [
        "guard_block_voltage_cutoff",
        "voltage_cutoff_active",
        "voltage_cutoff_day_locked",
        "guard_block_low_soc_discharge",
        "guard_block_health_lock_discharge",
        "guard_block_firmware_override_discharge",
        "guard_block_isolated_load_bus_discharge",
        "guard_block_load_over_discharge_limit",
    ]

    fig, axes = plt.subplots(5, 1, figsize=(18, 13), sharex=True)

    ax = axes[0]
    ax.step(dep["timestamp"], _series(dep, "action_raw_kw") * 1000.0, where="post", lw=1.2, label="Model raw action")
    ax.step(dep["timestamp"], _series(dep, "action_power_kw") * 1000.0, where="post", lw=1.4, label="Final command/action")
    ax.scatter(dep.loc[model_intent, "timestamp"], np.zeros(model_intent.sum()), s=12, color="#d62728", label="Negative model intent")
    ax.scatter(dep.loc[command_discharge, "timestamp"], np.full(command_discharge.sum(), -0.25), s=16, color="#1f77b4", label="Negative command")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Action (W)")
    ax.legend(loc="upper right", ncol=4)
    _setup_time_axis(ax)

    ax = axes[1]
    if "batt_i_mean_ma" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_i_mean_ma"), lw=1.2, label="Deployment batt current")
    if not raw_1.empty and "current_ma" in raw_1:
        ax.plot(raw_1["timestamp"], _series(raw_1, "current_ma"), lw=1.0, alpha=0.85, label="Raw signed current")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-5, color="#d62728", lw=0.9, ls="--", alpha=0.7, label="-5 mA discharge ref.")
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[2]
    if "batt_p_mean_mW" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_p_mean_mW") / 1000.0, lw=1.2, label="Deployment batt power")
    if not raw_1.empty:
        raw_power = _series(raw_1, "voltage_v") * _series(raw_1, "current_ma") / 1000.0
        ax.plot(raw_1["timestamp"], raw_power, lw=1.0, alpha=0.85, label="Raw V*I")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    ax = axes[3]
    if "batt_v_mean" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_v_mean"), lw=1.1, label="Deployment voltage")
    if not raw_1.empty and "voltage_v" in raw_1:
        ax.plot(raw_1["timestamp"], _series(raw_1, "voltage_v"), lw=1.0, alpha=0.85, label="Raw voltage")
    ax.axhline(6.0, color="#d62728", lw=1.0, ls="--", alpha=0.8, label="6 V diagnostic ref.")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[4]
    plotted = 0
    for col in cutoff_cols:
        if col not in dep:
            continue
        active = _series(dep, col).fillna(0).gt(0).to_numpy()
        plotted += 1
        ax.fill_between(dep["timestamp"], plotted - 0.35, plotted + 0.35, where=active, step="post", alpha=0.45, label=col)
    if plotted:
        ax.set_yticks(range(1, plotted + 1))
        ax.set_yticklabels([c for c in cutoff_cols if c in dep])
    ax.set_ylim(0.4, max(1.6, plotted + 0.6))
    ax.set_ylabel("Blocks / cutoff")
    ax.set_xlabel("Time")
    if plotted:
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    _setup_time_axis(ax)

    fig.suptitle("Discharge Intent vs Actual Battery Response", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "discharge_intent_vs_actual_20260617_20260622.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _save_voltage_recovery(dep: pd.DataFrame, raw: pd.DataFrame) -> Path:
    raw_30s = _resample_numeric(raw, "30s")

    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

    ax = axes[0]
    if not raw_30s.empty and "voltage_v" in raw_30s:
        ax.plot(raw_30s["timestamp"], _series(raw_30s, "voltage_v"), lw=1.0, label="Raw voltage")
    if "batt_v_mean" in dep:
        ax.scatter(dep["timestamp"], _series(dep, "batt_v_mean"), s=10, alpha=0.75, label="Decision-window voltage")
    ax.axhline(6.0, color="#d62728", lw=1.0, ls="--", alpha=0.8, label="6 V diagnostic ref.")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[1]
    if not raw_30s.empty and "speed_percent" in raw_30s:
        ax.plot(raw_30s["timestamp"], _series(raw_30s, "speed_percent"), lw=1.0, label="Raw speed")
        flow_50 = _series(raw_30s, "speed_percent").ge(45)
        ax.fill_between(raw_30s["timestamp"], 0, 100, where=flow_50.to_numpy(), color="#9467bd", alpha=0.12, label="Raw speed >=45%")
    if "flow_pct_cmd" in dep:
        ax.step(dep["timestamp"], _series(dep, "flow_pct_cmd"), where="post", lw=1.2, label="Command flow")
    ax.set_ylabel("Flow (%)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[2]
    if not raw_30s.empty and "current_ma" in raw_30s:
        ax.plot(raw_30s["timestamp"], _series(raw_30s, "current_ma"), lw=1.0, label="Raw current")
    if "batt_i_mean_ma" in dep:
        ax.scatter(dep["timestamp"], _series(dep, "batt_i_mean_ma"), s=10, alpha=0.7, label="Decision-window current")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    ax = axes[3]
    if not raw_30s.empty and "situation_code" in raw_30s:
        ax.step(raw_30s["timestamp"], _series(raw_30s, "situation_code"), where="post", lw=1.0, label="Raw situation")
    if "situation_code" in dep:
        ax.scatter(dep["timestamp"], _series(dep, "situation_code"), s=10, alpha=0.75, label="Decision situation")
    ax.set_ylabel("Situation")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    fig.suptitle("Voltage Recovery / Pre-Measure Evidence", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "voltage_recovery_premeasure_20260617_20260622.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _daily_metrics(dep: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    raw = raw.copy()
    if not raw.empty:
        raw["date"] = raw["timestamp"].dt.strftime("%Y-%m-%d")
        raw["raw_batt_power_w"] = _series(raw, "voltage_v") * _series(raw, "current_ma") / 1000.0

    dep = dep.copy()
    if not dep.empty:
        dep["date"] = dep["timestamp"].dt.strftime("%Y-%m-%d")

    for date in DATES:
        d = dep.loc[dep["date"] == date].copy() if not dep.empty else pd.DataFrame()
        r = raw.loc[raw["date"] == date].copy() if not raw.empty else pd.DataFrame()

        model_intent = _series(d, "action_raw_kw").lt(-1e-6) if not d.empty else pd.Series(dtype=bool)
        command_discharge = _series(d, "action_power_kw").lt(-1e-6) if not d.empty else pd.Series(dtype=bool)
        final_charge = _series(d, "action_power_kw").gt(1e-6) if not d.empty else pd.Series(dtype=bool)
        raw_discharge = _series(r, "current_ma").lt(-5) if not r.empty else pd.Series(dtype=bool)
        dep_discharge = _series(d, "batt_i_mean_ma").lt(-5) if not d.empty else pd.Series(dtype=bool)
        low_raw_voltage = _series(r, "voltage_v").lt(6.0) if not r.empty else pd.Series(dtype=bool)
        low_dep_voltage = _series(d, "batt_v_mean").lt(6.0) if not d.empty else pd.Series(dtype=bool)

        raw_dt_h = r["timestamp"].diff().dt.total_seconds().median() / 3600.0 if len(r) > 1 else np.nan
        dep_dt_h = d["timestamp"].diff().dt.total_seconds().median() / 3600.0 if len(d) > 1 else np.nan
        if not np.isfinite(raw_dt_h) or raw_dt_h <= 0:
            raw_dt_h = 10.0 / 3600.0
        if not np.isfinite(dep_dt_h) or dep_dt_h <= 0:
            dep_dt_h = 15.0 / 60.0

        raw_discharge_wh = float((-r.loc[_series(r, "raw_batt_power_w").lt(0), "raw_batt_power_w"].sum()) * raw_dt_h) if not r.empty else 0.0
        dep_command_discharge_wh = float((-_series(d.loc[command_discharge], "action_power_kw").sum()) * 1000.0 * dep_dt_h) if command_discharge.any() else 0.0
        dep_actual_discharge_wh = float((-_series(d.loc[_series(d, "batt_p_mean_mW").lt(0)], "batt_p_mean_mW").sum() / 1000.0) * dep_dt_h) if not d.empty and "batt_p_mean_mW" in d else 0.0

        block_cols = [
            "guard_block_voltage_cutoff",
            "voltage_cutoff_active",
            "voltage_cutoff_day_locked",
            "guard_block_low_soc_discharge",
            "guard_block_health_lock_discharge",
            "guard_block_firmware_override_discharge",
            "guard_block_isolated_load_bus_discharge",
            "guard_block_load_over_discharge_limit",
        ]
        block_count = int(sum(_series(d, col).fillna(0).gt(0).sum() for col in block_cols if col in d)) if not d.empty else 0

        rows.append(
            {
                "date": date,
                "deployment_rows": int(len(d)),
                "raw_rows": int(len(r)),
                "model_negative_action_count": int(model_intent.sum()),
                "command_negative_action_count": int(command_discharge.sum()),
                "final_charge_action_count": int(final_charge.sum()),
                "actual_discharge_count_raw_current_lt_minus5ma": int(raw_discharge.sum()),
                "actual_discharge_count_dep_current_lt_minus5ma": int(dep_discharge.sum()),
                "command_discharge_wh": dep_command_discharge_wh,
                "actual_discharge_wh_raw_vi_negative": raw_discharge_wh,
                "actual_discharge_wh_dep_batt_p_negative": dep_actual_discharge_wh,
                "low_voltage_count_raw_lt6v": int(low_raw_voltage.sum()),
                "low_voltage_count_dep_lt6v": int(low_dep_voltage.sum()),
                "min_raw_voltage_v": float(_series(r, "voltage_v").min()) if not r.empty else np.nan,
                "median_raw_voltage_v": float(_series(r, "voltage_v").median()) if not r.empty else np.nan,
                "min_dep_voltage_v": float(_series(d, "batt_v_mean").min()) if not d.empty else np.nan,
                "median_dep_voltage_v": float(_series(d, "batt_v_mean").median()) if not d.empty else np.nan,
                "flow_ge45_raw_count": int(_series(r, "speed_percent").ge(45).sum()) if not r.empty and "speed_percent" in r else 0,
                "flow_ge45_command_count": int(_series(d, "flow_pct_cmd").ge(45).sum()) if not d.empty and "flow_pct_cmd" in d else 0,
                "guard_or_cutoff_block_count": block_count,
                "standby_or_rest_like_count": int(_series(d, "action_power_kw").abs().le(1e-9).sum()) if not d.empty else 0,
            }
        )

    return pd.DataFrame(rows)


def _save_daily_summary(metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    x = np.arange(len(metrics))
    labels = metrics["date"].str.replace("2026-", "", regex=False)

    ax = axes[0]
    ax.bar(x - 0.2, metrics["model_negative_action_count"], width=0.4, label="Model negative action")
    ax.bar(x + 0.2, metrics["command_negative_action_count"], width=0.4, label="Final negative command")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.bar(x - 0.25, metrics["command_discharge_wh"], width=0.25, label="Command discharge Wh")
    ax.bar(x, metrics["actual_discharge_wh_raw_vi_negative"], width=0.25, label="Actual raw V*I discharge Wh")
    ax.bar(x + 0.25, metrics["actual_discharge_wh_dep_batt_p_negative"], width=0.25, label="Actual dep batt_p discharge Wh")
    ax.set_ylabel("Wh")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    ax.bar(x - 0.2, metrics["low_voltage_count_raw_lt6v"], width=0.4, label="Raw voltage <6 V")
    ax.bar(x + 0.2, metrics["low_voltage_count_dep_lt6v"], width=0.4, label="Decision voltage <6 V")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[3]
    ax.bar(x - 0.2, metrics["guard_or_cutoff_block_count"], width=0.4, label="Guard/cutoff flags")
    ax.bar(x + 0.2, metrics["standby_or_rest_like_count"], width=0.4, label="Zero final action rows")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Daily Discharge Diagnostic Summary", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "daily_summary_20260617_20260622.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _event_windows(dep: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if dep.empty:
        return pd.DataFrame(rows)

    for _, row in dep.iterrows():
        action_raw = pd.to_numeric(row.get("action_raw_kw"), errors="coerce")
        action_final = pd.to_numeric(row.get("action_power_kw"), errors="coerce")
        if not (pd.notna(action_raw) and action_raw < -1e-6) and not (pd.notna(action_final) and action_final < -1e-6):
            continue

        t0 = row["timestamp"]
        window = raw[(raw["timestamp"] >= t0 - pd.Timedelta(minutes=2)) & (raw["timestamp"] <= t0 + pd.Timedelta(minutes=5))]
        raw_current_min = float(_series(window, "current_ma").min()) if not window.empty else np.nan
        raw_voltage_min = float(_series(window, "voltage_v").min()) if not window.empty else np.nan
        raw_voltage_median = float(_series(window, "voltage_v").median()) if not window.empty else np.nan
        rows.append(
            {
                "timestamp": t0.isoformat(),
                "date": t0.strftime("%Y-%m-%d"),
                "action_raw_w": float(action_raw * 1000.0) if pd.notna(action_raw) else np.nan,
                "action_final_w": float(action_final * 1000.0) if pd.notna(action_final) else np.nan,
                "power_mw_cmd": float(pd.to_numeric(row.get("power_mw_cmd"), errors="coerce")),
                "flow_pct_cmd": float(pd.to_numeric(row.get("flow_pct_cmd"), errors="coerce")),
                "dep_batt_v_mean": float(pd.to_numeric(row.get("batt_v_mean"), errors="coerce")),
                "dep_batt_i_mean_ma": float(pd.to_numeric(row.get("batt_i_mean_ma"), errors="coerce")),
                "dep_batt_p_mean_w": float(pd.to_numeric(row.get("batt_p_mean_mW"), errors="coerce") / 1000.0),
                "raw_window_current_min_ma": raw_current_min,
                "raw_window_voltage_min_v": raw_voltage_min,
                "raw_window_voltage_median_v": raw_voltage_median,
                "guard_block_voltage_cutoff": int(pd.to_numeric(row.get("guard_block_voltage_cutoff"), errors="coerce") or 0),
                "voltage_cutoff_active": int(pd.to_numeric(row.get("voltage_cutoff_active"), errors="coerce") or 0),
                "voltage_cutoff_day_locked": int(pd.to_numeric(row.get("voltage_cutoff_day_locked"), errors="coerce") or 0),
                "guard_block_low_soc_discharge": int(pd.to_numeric(row.get("guard_block_low_soc_discharge"), errors="coerce") or 0),
                "guard_block_health_lock_discharge": int(pd.to_numeric(row.get("guard_block_health_lock_discharge"), errors="coerce") or 0),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dep, dep_sources = _read_daily("deployment")
    raw, raw_sources = _read_daily("raw_data")
    if dep.empty and raw.empty:
        raise SystemExit("No deployment/raw files found for requested date range.")

    metrics = _daily_metrics(dep, raw)
    events = _event_windows(dep, raw)

    outputs = []
    if not dep.empty or not raw.empty:
        outputs.append(_save_overview(dep, raw))
    if not dep.empty:
        outputs.append(_save_intent_vs_actual(dep, raw))
    if not dep.empty or not raw.empty:
        outputs.append(_save_voltage_recovery(dep, raw))
    outputs.append(_save_daily_summary(metrics))

    metrics_path = OUT_DIR / "daily_metrics_20260617_20260622.csv"
    events_path = OUT_DIR / "discharge_intent_windows_20260617_20260622.csv"
    summary_path = OUT_DIR / "diagnostic_summary_20260617_20260622.json"
    metrics.to_csv(metrics_path, index=False)
    events.to_csv(events_path, index=False)

    usable_columns = {
        "deployment": sorted(dep.columns.tolist()) if not dep.empty else [],
        "raw_data": sorted(raw.columns.tolist()) if not raw.empty else [],
    }
    summary = {
        "date_range": [DATES[0], DATES[-1]],
        "deployment_sources": dep_sources,
        "raw_sources": raw_sources,
        "deployment_time_range": [
            dep["timestamp"].min().isoformat() if not dep.empty else None,
            dep["timestamp"].max().isoformat() if not dep.empty else None,
        ],
        "raw_time_range": [
            raw["timestamp"].min().isoformat() if not raw.empty else None,
            raw["timestamp"].max().isoformat() if not raw.empty else None,
        ],
        "row_counts": {"deployment": int(len(dep)), "raw": int(len(raw))},
        "usable_columns": usable_columns,
        "outputs": [str(path.relative_to(ROOT)) for path in outputs] + [str(metrics_path.relative_to(ROOT)), str(events_path.relative_to(ROOT))],
        "headline_metrics": {
            "model_negative_action_count": int(metrics["model_negative_action_count"].sum()),
            "command_negative_action_count": int(metrics["command_negative_action_count"].sum()),
            "actual_discharge_count_raw_current_lt_minus5ma": int(metrics["actual_discharge_count_raw_current_lt_minus5ma"].sum()),
            "actual_discharge_count_dep_current_lt_minus5ma": int(metrics["actual_discharge_count_dep_current_lt_minus5ma"].sum()),
            "low_voltage_count_raw_lt6v": int(metrics["low_voltage_count_raw_lt6v"].sum()),
            "low_voltage_count_dep_lt6v": int(metrics["low_voltage_count_dep_lt6v"].sum()),
            "guard_or_cutoff_block_count": int(metrics["guard_or_cutoff_block_count"].sum()),
            "command_discharge_wh": float(metrics["command_discharge_wh"].sum()),
            "actual_discharge_wh_raw_vi_negative": float(metrics["actual_discharge_wh_raw_vi_negative"].sum()),
            "actual_discharge_wh_dep_batt_p_negative": float(metrics["actual_discharge_wh_dep_batt_p_negative"].sum()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {summary_path.relative_to(ROOT)}")
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Wrote {metrics_path.relative_to(ROOT)}")
    print(f"Wrote {events_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

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
OUT_DIR = ROOT / "outputs" / "deployment_diagnostics_20260622_20260623"
DATES = ["2026-06-22", "2026-06-23"]
CUTOFF_V = 4.2

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

GUARD_COLUMNS = [
    "guard_force_charge_low_soc",
    "guard_block_low_soc_discharge",
    "guard_block_pv_active_discharge",
    "guard_block_voltage_cutoff",
    "warn_load_over_discharge_limit",
    "guard_block_load_over_discharge_limit",
    "guard_block_invalid_discharge",
    "guard_block_discharge_intent_threshold",
    "guard_block_firmware_override_discharge",
    "guard_block_isolated_load_bus_discharge",
    "guard_block_health_lock_discharge",
    "voltage_cutoff_active",
    "voltage_cutoff_day_locked",
    "cutoff_soc_fallback_applied",
]


def _read_csv_flexible(path: Path) -> pd.DataFrame:
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
            if col not in TEXT_COLUMNS and col not in {"timestamp", "date"}:
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


def _setup_time_axis(ax: plt.Axes, interval_hours: int = 6) -> None:
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))


def _flag_band(ax: plt.Axes, df: pd.DataFrame, col: str, y: float, label: str, color: str) -> bool:
    if col not in df:
        return False
    active = _series(df, col).fillna(0).gt(0).to_numpy()
    if not active.any():
        return False
    ax.fill_between(df["timestamp"], y - 0.35, y + 0.35, where=active, step="post", color=color, alpha=0.45, label=label)
    return True


def _save_overview(dep: pd.DataFrame, raw: pd.DataFrame) -> Path:
    raw_5 = _resample_numeric(raw, "5min")
    fig, axes = plt.subplots(8, 1, figsize=(18, 20), sharex=True)

    ax = axes[0]
    ax.step(dep["timestamp"], _series(dep, "action_raw_kw") * 1000.0, where="post", lw=1.2, label="Model raw action")
    ax.step(dep["timestamp"], _series(dep, "action_power_kw") * 1000.0, where="post", lw=1.5, label="Final action")
    ax.step(dep["timestamp"], _series(dep, "power_mw_cmd") / 1000.0, where="post", lw=1.0, ls="--", label="Physical command magnitude")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Action (W)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[1]
    if "batt_p_mean_mW" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_p_mean_mW") / 1000.0, lw=1.2, label="Decision batt power")
    if not raw_5.empty:
        raw_power = _series(raw_5, "voltage_v") * _series(raw_5, "current_ma") / 1000.0
        ax.plot(raw_5["timestamp"], raw_power, lw=1.0, alpha=0.85, label="Raw V*I")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Batt P (W)")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    ax = axes[2]
    if "batt_i_mean_ma" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_i_mean_ma"), lw=1.2, label="Decision current")
    if not raw_5.empty and "current_ma" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "current_ma"), lw=1.0, alpha=0.85, label="Raw signed current")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-5, color="#d62728", lw=0.9, ls="--", alpha=0.7, label="-5 mA ref.")
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[3]
    if "batt_v_mean" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_v_mean"), lw=1.2, label="Decision voltage")
    if not raw_5.empty and "voltage_v" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "voltage_v"), lw=1.0, alpha=0.85, label="Raw voltage")
    ax.axhline(CUTOFF_V, color="#d62728", lw=1.0, ls="--", label=f"{CUTOFF_V:.1f} V cutoff")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[4]
    if "soc" in dep:
        ax.plot(dep["timestamp"], _series(dep, "soc") * 100.0, lw=1.3, label="Deployment SoC")
    if not raw_5.empty and "soc_calc" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "soc_calc") * 100.0, lw=1.0, alpha=0.8, label="Raw control SoC")
    if not raw_5.empty and "soc_coulomb" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "soc_coulomb") * 100.0, lw=1.0, ls="--", alpha=0.8, label="Raw coulomb SoC")
    ax.set_ylabel("SoC (%)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[5]
    for col, label in [("load_kw", "Decision load"), ("pv_kw", "Decision PV support")]:
        if col in dep:
            ax.plot(dep["timestamp"], _series(dep, col) * 1000.0, lw=1.1, label=label)
    if not raw_5.empty:
        for col, label in [("load_p_mw", "Raw load"), ("grid_p_mw", "Raw grid demand"), ("mppt_p_mw", "Raw MPPT/PV")]:
            if col in raw_5:
                ax.plot(raw_5["timestamp"], _series(raw_5, col) / 1000.0, lw=0.9, alpha=0.8, label=label)
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper right", ncol=5)
    _setup_time_axis(ax)

    ax = axes[6]
    if "pv_support_ratio" in dep:
        ax.plot(dep["timestamp"], _series(dep, "pv_support_ratio"), lw=1.2, label="PV support ratio")
    if "pv_active" in dep:
        ax.step(dep["timestamp"], _series(dep, "pv_active"), where="post", lw=1.0, label="PV active/block state")
    if "price" in dep:
        ax2 = ax.twinx()
        ax2.step(dep["timestamp"], _series(dep, "price"), where="post", color="#7f7f7f", lw=0.9, alpha=0.8, label="Price")
        ax2.set_ylabel("Price")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=3)
    else:
        ax.legend(loc="upper right", ncol=2)
    ax.set_ylabel("PV / price")
    _setup_time_axis(ax)

    ax = axes[7]
    y = 0
    labels = []
    colors = plt.cm.tab20.colors
    for idx, col in enumerate(GUARD_COLUMNS):
        if col not in dep:
            continue
        if _series(dep, col).fillna(0).gt(0).any():
            y += 1
            if _flag_band(ax, dep, col, y, col, colors[idx % len(colors)]):
                labels.append(col)
    if labels:
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels, fontsize=8)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_ylim(0.4, max(1.6, len(labels) + 0.6))
    ax.set_ylabel("Flags")
    ax.set_xlabel("Time")
    _setup_time_axis(ax)

    fig.suptitle("Deployment Diagnostics Overview: 2026-06-22 to 2026-06-23", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "overview_timeseries_20260622_20260623.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _save_intent_guard_breakdown(dep: pd.DataFrame, raw: pd.DataFrame) -> Path:
    raw_1 = _resample_numeric(raw, "1min")
    neg_raw = _series(dep, "action_raw_kw").lt(-1e-6)
    neg_final = _series(dep, "action_power_kw").lt(-1e-6)

    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    ax = axes[0]
    ax.step(dep["timestamp"], _series(dep, "action_raw_kw") * 1000.0, where="post", lw=1.3, label="Model raw action")
    ax.step(dep["timestamp"], _series(dep, "action_power_kw") * 1000.0, where="post", lw=1.5, label="Final action")
    ax.scatter(dep.loc[neg_raw, "timestamp"], np.zeros(int(neg_raw.sum())), s=16, color="#d62728", label="Raw negative intent")
    ax.scatter(dep.loc[neg_final, "timestamp"], np.full(int(neg_final.sum()), -0.2), s=20, color="#1f77b4", label="Final negative command")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Action (W)")
    ax.legend(loc="upper right", ncol=4)
    _setup_time_axis(ax)

    ax = axes[1]
    if "batt_i_mean_ma" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_i_mean_ma"), lw=1.2, label="Decision current")
    if not raw_1.empty and "current_ma" in raw_1:
        ax.plot(raw_1["timestamp"], _series(raw_1, "current_ma"), lw=0.9, alpha=0.85, label="Raw signed current")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-5, color="#d62728", lw=0.9, ls="--", alpha=0.7, label="-5 mA ref.")
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[2]
    if "batt_v_mean" in dep:
        ax.plot(dep["timestamp"], _series(dep, "batt_v_mean"), lw=1.1, label="Decision voltage")
    if not raw_1.empty and "voltage_v" in raw_1:
        ax.plot(raw_1["timestamp"], _series(raw_1, "voltage_v"), lw=0.9, alpha=0.85, label="Raw voltage")
    ax.axhline(CUTOFF_V, color="#d62728", lw=1.0, ls="--", label=f"{CUTOFF_V:.1f} V cutoff")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[3]
    y = 0
    labels = []
    colors = plt.cm.tab20.colors
    neg_dep = dep.loc[neg_raw].copy()
    for idx, col in enumerate(GUARD_COLUMNS):
        if col not in dep:
            continue
        active_on_intent = _series(neg_dep, col).fillna(0).gt(0).any()
        if not active_on_intent:
            continue
        y += 1
        active = _series(dep, col).fillna(0).gt(0).to_numpy()
        ax.fill_between(dep["timestamp"], y - 0.35, y + 0.35, where=active, step="post", color=colors[idx % len(colors)], alpha=0.45, label=col)
        labels.append(col)
    if labels:
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels, fontsize=8)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_ylim(0.4, max(1.6, len(labels) + 0.6))
    ax.set_ylabel("Intent flags")
    _setup_time_axis(ax)

    ax = axes[4]
    if "flow_pct_cmd" in dep:
        ax.step(dep["timestamp"], _series(dep, "flow_pct_cmd"), where="post", lw=1.2, label="Command flow")
    if not raw_1.empty and "speed_percent" in raw_1:
        ax.plot(raw_1["timestamp"], _series(raw_1, "speed_percent"), lw=0.9, alpha=0.8, label="Raw speed")
    if "situation_code" in dep:
        ax2 = ax.twinx()
        ax2.step(dep["timestamp"], _series(dep, "situation_code"), where="post", lw=0.9, color="#7f7f7f", label="Situation")
        ax2.set_ylabel("Situation")
        lines, labels2 = ax.get_legend_handles_labels()
        lines3, labels3 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines3, labels2 + labels3, loc="upper right", ncol=3)
    else:
        ax.legend(loc="upper right")
    ax.set_ylabel("Flow (%)")
    ax.set_xlabel("Time")
    _setup_time_axis(ax)

    fig.suptitle("Discharge Intent vs Guard / Actual Response", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "discharge_intent_guard_breakdown_20260622_20260623.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _save_hourly_summary_plot(hourly: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    ax = axes[0]
    ax.bar(hourly["hour"], hourly["model_negative_action_count"], width=0.03, label="Raw negative intent")
    ax.bar(hourly["hour"], hourly["command_negative_action_count"], width=0.02, label="Final negative command")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for col in [
        "neg_intent_guard_block_pv_active_discharge_count",
        "neg_intent_guard_block_voltage_cutoff_count",
        "neg_intent_voltage_cutoff_active_count",
        "neg_intent_guard_block_low_soc_discharge_count",
        "neg_intent_warn_load_over_discharge_limit_count",
    ]:
        if col in hourly and hourly[col].sum() > 0:
            ax.step(hourly["hour"], hourly[col], where="mid", lw=1.2, label=col)
    ax.set_ylabel("Intent flags")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.step(hourly["hour"], hourly["min_dep_batt_v"], where="mid", lw=1.2, label="Min decision voltage")
    ax.step(hourly["hour"], hourly["min_raw_voltage_v"], where="mid", lw=1.2, label="Min raw voltage")
    ax.axhline(CUTOFF_V, color="#d62728", lw=1.0, ls="--", label=f"{CUTOFF_V:.1f} V cutoff")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[3]
    ax.step(hourly["hour"], hourly["raw_current_lt_minus5ma_count"], where="mid", lw=1.2, label="Raw current < -5 mA")
    ax.step(hourly["hour"], hourly["dep_current_lt_minus5ma_count"], where="mid", lw=1.2, label="Decision current < -5 mA")
    ax.set_ylabel("Actual discharge count")
    ax.set_xlabel("Hour")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    _setup_time_axis(ax, interval_hours=3)

    fig.suptitle("Hourly Discharge Diagnostic Summary", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "hourly_summary_20260622_20260623.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _event_windows(dep: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in dep.iterrows():
        action_raw = pd.to_numeric(row.get("action_raw_kw"), errors="coerce")
        action_final = pd.to_numeric(row.get("action_power_kw"), errors="coerce")
        if not (pd.notna(action_raw) and action_raw < -1e-6) and not (pd.notna(action_final) and action_final < -1e-6):
            continue
        t0 = row["timestamp"]
        window = raw[(raw["timestamp"] >= t0 - pd.Timedelta(minutes=2)) & (raw["timestamp"] <= t0 + pd.Timedelta(minutes=5))]
        out = {
            "timestamp": t0.isoformat(),
            "date": t0.strftime("%Y-%m-%d"),
            "action_raw_w": float(action_raw * 1000.0) if pd.notna(action_raw) else np.nan,
            "action_final_w": float(action_final * 1000.0) if pd.notna(action_final) else np.nan,
            "power_mw_cmd": float(pd.to_numeric(row.get("power_mw_cmd"), errors="coerce")),
            "flow_pct_cmd": float(pd.to_numeric(row.get("flow_pct_cmd"), errors="coerce")),
            "situation_code": float(pd.to_numeric(row.get("situation_code"), errors="coerce")),
            "dep_batt_v_mean": float(pd.to_numeric(row.get("batt_v_mean"), errors="coerce")),
            "dep_batt_i_mean_ma": float(pd.to_numeric(row.get("batt_i_mean_ma"), errors="coerce")),
            "dep_batt_p_mean_w": float(pd.to_numeric(row.get("batt_p_mean_mW"), errors="coerce") / 1000.0),
            "raw_window_current_min_ma": float(_series(window, "current_ma").min()) if not window.empty else np.nan,
            "raw_window_current_median_ma": float(_series(window, "current_ma").median()) if not window.empty else np.nan,
            "raw_window_voltage_min_v": float(_series(window, "voltage_v").min()) if not window.empty else np.nan,
            "raw_window_voltage_median_v": float(_series(window, "voltage_v").median()) if not window.empty else np.nan,
            "pv_support_ratio": float(pd.to_numeric(row.get("pv_support_ratio"), errors="coerce")),
            "pv_active": float(pd.to_numeric(row.get("pv_active"), errors="coerce")),
            "soc": float(pd.to_numeric(row.get("soc"), errors="coerce")),
            "load_w": float(pd.to_numeric(row.get("load_kw"), errors="coerce") * 1000.0),
            "pv_w": float(pd.to_numeric(row.get("pv_kw"), errors="coerce") * 1000.0),
        }
        for col in GUARD_COLUMNS:
            if col in dep:
                out[col] = int(pd.to_numeric(row.get(col), errors="coerce") or 0)
        rows.append(out)
    return pd.DataFrame(rows)


def _summaries(dep: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dep = dep.copy()
    raw = raw.copy()
    raw["raw_batt_power_w"] = _series(raw, "voltage_v") * _series(raw, "current_ma") / 1000.0

    daily_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []

    def summarize_window(label: object, d: pd.DataFrame, r: pd.DataFrame) -> dict[str, object]:
        neg_raw = _series(d, "action_raw_kw").lt(-1e-6)
        neg_final = _series(d, "action_power_kw").lt(-1e-6)
        zero_final = _series(d, "action_power_kw").abs().le(1e-9)
        neg_intent_zeroed = neg_raw & ~neg_final
        raw_discharge = _series(r, "current_ma").lt(-5)
        dep_discharge = _series(d, "batt_i_mean_ma").lt(-5)

        row: dict[str, object] = {
            "window": str(label),
            "deployment_rows": int(len(d)),
            "raw_rows": int(len(r)),
            "model_negative_action_count": int(neg_raw.sum()),
            "command_negative_action_count": int(neg_final.sum()),
            "negative_intent_zeroed_count": int(neg_intent_zeroed.sum()),
            "zero_final_action_count": int(zero_final.sum()),
            "min_action_raw_w": float((_series(d, "action_raw_kw") * 1000.0).min()) if len(d) else np.nan,
            "min_action_raw_time": d.loc[(_series(d, "action_raw_kw") * 1000.0).idxmin(), "timestamp"].isoformat() if len(d) and _series(d, "action_raw_kw").notna().any() else None,
            "min_action_final_w": float((_series(d, "action_power_kw") * 1000.0).min()) if len(d) else np.nan,
            "raw_current_lt_minus5ma_count": int(raw_discharge.sum()),
            "dep_current_lt_minus5ma_count": int(dep_discharge.sum()),
            "min_raw_current_ma": float(_series(r, "current_ma").min()) if len(r) else np.nan,
            "min_dep_current_ma": float(_series(d, "batt_i_mean_ma").min()) if len(d) else np.nan,
            "raw_negative_power_count": int(_series(r, "raw_batt_power_w").lt(0).sum()) if len(r) else 0,
            "dep_negative_power_count": int(_series(d, "batt_p_mean_mW").lt(0).sum()) if len(d) else 0,
            "min_raw_voltage_v": float(_series(r, "voltage_v").min()) if len(r) else np.nan,
            "median_raw_voltage_v": float(_series(r, "voltage_v").median()) if len(r) else np.nan,
            "min_dep_batt_v": float(_series(d, "batt_v_mean").min()) if len(d) else np.nan,
            "median_dep_batt_v": float(_series(d, "batt_v_mean").median()) if len(d) else np.nan,
            "dep_batt_v_below_cutoff_count": int(_series(d, "batt_v_mean").lt(CUTOFF_V).sum()) if len(d) else 0,
            "raw_voltage_below_cutoff_count": int(_series(r, "voltage_v").lt(CUTOFF_V).sum()) if len(r) else 0,
            "pv_active_count": int(_series(d, "pv_active").fillna(0).gt(0).sum()) if len(d) else 0,
            "neg_intent_pv_active_count": int((_series(d.loc[neg_raw], "pv_active").fillna(0).gt(0)).sum()) if len(d) else 0,
            "median_pv_support_ratio": float(_series(d, "pv_support_ratio").median()) if len(d) else np.nan,
            "neg_intent_median_pv_support_ratio": float(_series(d.loc[neg_raw], "pv_support_ratio").median()) if neg_raw.any() else np.nan,
            "soc_below_20pct_count": int(_series(d, "soc").lt(0.2).sum()) if len(d) else 0,
            "neg_intent_soc_below_20pct_count": int(_series(d.loc[neg_raw], "soc").lt(0.2).sum()) if len(d) else 0,
            "premeasure_like_flow_ge45_zero_power_count": int((_series(d, "flow_pct_cmd").ge(45) & _series(d, "action_power_kw").abs().le(1e-9)).sum()) if len(d) else 0,
            "mode3_count": int(_series(d, "situation_code").eq(3).sum()) if len(d) else 0,
        }
        for col in GUARD_COLUMNS:
            if col in d:
                row[f"{col}_count"] = int(_series(d, col).fillna(0).gt(0).sum())
                row[f"neg_intent_{col}_count"] = int(_series(d.loc[neg_raw], col).fillna(0).gt(0).sum()) if neg_raw.any() else 0
                row[f"zeroed_intent_{col}_count"] = int(_series(d.loc[neg_intent_zeroed], col).fillna(0).gt(0).sum()) if neg_intent_zeroed.any() else 0
        return row

    for date in DATES:
        d = dep.loc[dep["date"] == date]
        r = raw.loc[raw["date"] == date]
        daily_rows.append(summarize_window(date, d, r))

    if not dep.empty:
        start = min(dep["timestamp"].min(), raw["timestamp"].min() if not raw.empty else dep["timestamp"].min()).floor("h")
        end = max(dep["timestamp"].max(), raw["timestamp"].max() if not raw.empty else dep["timestamp"].max()).ceil("h")
        for hour in pd.date_range(start, end, freq="h", inclusive="left"):
            h_end = hour + pd.Timedelta(hours=1)
            d = dep[(dep["timestamp"] >= hour) & (dep["timestamp"] < h_end)]
            r = raw[(raw["timestamp"] >= hour) & (raw["timestamp"] < h_end)]
            row = summarize_window(hour.isoformat(), d, r)
            row["hour"] = hour
            hourly_rows.append(row)

    total = summarize_window("total", dep, raw)
    software_evidence = {
        "has_warn_load_over_discharge_limit": "warn_load_over_discharge_limit" in dep.columns,
        "has_old_guard_block_load_over_discharge_limit": "guard_block_load_over_discharge_limit" in dep.columns,
        "warn_load_over_discharge_limit_total": int(_series(dep, "warn_load_over_discharge_limit").fillna(0).gt(0).sum()) if "warn_load_over_discharge_limit" in dep else 0,
        "old_guard_block_load_over_discharge_limit_total": int(_series(dep, "guard_block_load_over_discharge_limit").fillna(0).gt(0).sum()) if "guard_block_load_over_discharge_limit" in dep else 0,
    }
    total["software_evidence"] = software_evidence
    total["columns"] = {
        "deployment": sorted(dep.columns.tolist()),
        "raw_data": sorted(raw.columns.tolist()),
    }
    return pd.DataFrame(daily_rows), pd.DataFrame(hourly_rows), total


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dep, dep_sources = _read_daily("deployment")
    raw, raw_sources = _read_daily("raw_data")
    if dep.empty and raw.empty:
        raise SystemExit("No deployment/raw files found for 2026-06-22 to 2026-06-23.")
    if dep.empty:
        raise SystemExit("Deployment logs are required for guard/action diagnostics.")

    daily, hourly, total = _summaries(dep, raw)
    events = _event_windows(dep, raw)

    outputs = [
        _save_overview(dep, raw),
        _save_intent_guard_breakdown(dep, raw),
        _save_hourly_summary_plot(hourly),
    ]

    daily_path = OUT_DIR / "daily_summary_20260622_20260623.csv"
    hourly_path = OUT_DIR / "hourly_summary_20260622_20260623.csv"
    events_path = OUT_DIR / "discharge_intent_events_20260622_20260623.csv"
    summary_path = OUT_DIR / "diagnostic_summary_20260622_20260623.json"

    daily.to_csv(daily_path, index=False)
    hourly.to_csv(hourly_path, index=False)
    events.to_csv(events_path, index=False)

    summary = {
        "date_range": [DATES[0], DATES[-1]],
        "deployment_sources": dep_sources,
        "raw_sources": raw_sources,
        "deployment_time_range": [dep["timestamp"].min().isoformat(), dep["timestamp"].max().isoformat()],
        "raw_time_range": [raw["timestamp"].min().isoformat() if not raw.empty else None, raw["timestamp"].max().isoformat() if not raw.empty else None],
        "row_counts": {"deployment": int(len(dep)), "raw": int(len(raw))},
        "headline_metrics": total,
        "outputs": [str(path.relative_to(ROOT)) for path in outputs]
        + [str(daily_path.relative_to(ROOT)), str(hourly_path.relative_to(ROOT)), str(events_path.relative_to(ROOT))],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"Wrote {summary_path.relative_to(ROOT)}")
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"Wrote {daily_path.relative_to(ROOT)}")
    print(f"Wrote {hourly_path.relative_to(ROOT)}")
    print(f"Wrote {events_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_deployment_diagnostics_20260617_20260622 as base


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "outputs" / "deployment_diagnostics_20260617_20260622"
START = pd.Timestamp("2026-06-18")
END = pd.Timestamp("2026-06-20")
CUTOFF_V = 4.2


def _series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _setup_time_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))


def _resample_raw(raw: pd.DataFrame) -> pd.DataFrame:
    numeric = raw.set_index("timestamp").select_dtypes(include=[np.number])
    return numeric.resample("5min").mean().reset_index()


def _flag_band(ax: plt.Axes, dep: pd.DataFrame, col: str, y: float, label: str, color: str) -> None:
    if col not in dep:
        return
    active = _series(dep, col).fillna(0).gt(0).to_numpy()
    ax.fill_between(
        dep["timestamp"],
        y - 0.35,
        y + 0.35,
        where=active,
        step="post",
        color=color,
        alpha=0.45,
        label=label,
    )


def _save_zoom(dep: pd.DataFrame, raw: pd.DataFrame) -> Path:
    raw_5 = _resample_raw(raw)
    fig, axes = plt.subplots(6, 1, figsize=(18, 15), sharex=True)

    ax = axes[0]
    ax.plot(raw_5["timestamp"], _series(raw_5, "voltage_v"), lw=1.2, label="Raw battery voltage")
    ax.scatter(dep["timestamp"], _series(dep, "batt_v_mean"), s=12, alpha=0.8, label="Decision-window batt_v_mean")
    ax.axhline(CUTOFF_V, color="#d62728", lw=1.0, ls="--", label="4.2 V cutoff")
    ax.axhline(5.0, color="#ff7f0e", lw=0.9, ls=":", label="5.0 V recover ref.")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", ncol=4)
    _setup_time_axis(ax)

    ax = axes[1]
    ax.plot(dep["timestamp"], _series(dep, "soc") * 100.0, lw=1.4, label="Deployment SoC")
    if "soc_calc" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "soc_calc") * 100.0, lw=1.0, alpha=0.8, label="Raw control SoC")
    if "soc_coulomb" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "soc_coulomb") * 100.0, lw=1.0, ls="--", alpha=0.8, label="Raw coulomb SoC")
    ax.axhline(20, color="#d62728", lw=0.9, ls="--")
    ax.axhline(80, color="#d62728", lw=0.9, ls="--")
    ax.set_ylabel("SoC (%)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[2]
    ax.step(dep["timestamp"], _series(dep, "action_raw_kw") * 1000.0, where="post", lw=1.3, label="Model raw action")
    ax.step(dep["timestamp"], _series(dep, "action_power_kw") * 1000.0, where="post", lw=1.5, label="Final action_power")
    ax.step(dep["timestamp"], _series(dep, "power_mw_cmd") / 1000.0, where="post", lw=1.0, ls="--", label="Command magnitude")
    neg = _series(dep, "action_raw_kw").lt(-1e-6)
    ax.scatter(dep.loc[neg, "timestamp"], np.zeros(int(neg.sum())), s=12, color="#d62728", label="Negative raw action")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Action (W)")
    ax.legend(loc="upper right", ncol=4)
    _setup_time_axis(ax)

    ax = axes[3]
    ax.plot(dep["timestamp"], _series(dep, "load_kw") * 1000.0, lw=1.4, label="Deployment load")
    ax.plot(dep["timestamp"], _series(dep, "pv_kw") * 1000.0, lw=1.2, label="Deployment PV support")
    ax.axhline(float(_series(dep, "flow_discharge_limit_kw").median()) * 1000.0, color="#d62728", lw=1.0, ls="--", label="Battery solo limit")
    if "grid_p_mw" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "grid_p_mw") / 1000.0, lw=1.0, alpha=0.75, label="Raw grid demand")
    if "load_p_mw" in raw_5:
        ax.plot(raw_5["timestamp"], _series(raw_5, "load_p_mw") / 1000.0, lw=0.9, alpha=0.6, label="Raw load")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper right", ncol=5)
    _setup_time_axis(ax)

    ax = axes[4]
    if "pv_support_ratio" in dep:
        ax.plot(dep["timestamp"], _series(dep, "pv_support_ratio"), lw=1.2, label="PV support ratio")
    if "pv_active" in dep:
        ax.step(dep["timestamp"], _series(dep, "pv_active"), where="post", lw=1.0, label="PV blocking state")
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

    ax = axes[5]
    _flag_band(ax, dep, "guard_block_load_over_discharge_limit", 1, "Load > solo discharge limit", "#d62728")
    _flag_band(ax, dep, "voltage_cutoff_active", 2, "Voltage cutoff active", "#ff7f0e")
    _flag_band(ax, dep, "guard_block_voltage_cutoff", 3, "Voltage cutoff blocked discharge", "#9467bd")
    _flag_band(ax, dep, "voltage_cutoff_day_locked", 4, "Voltage cutoff day locked", "#8c564b")
    _flag_band(ax, dep, "coral_clipped", 5, "CORAL clipped", "#1f77b4")
    _flag_band(ax, dep, "guard_block_pv_active_discharge", 6, "PV-state discharge block", "#2ca02c")
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels([
        "load limit",
        "cutoff active",
        "cutoff block",
        "day locked",
        "CORAL",
        "PV block",
    ])
    ax.set_ylim(0.4, 6.6)
    ax.set_ylabel("Flags")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    _setup_time_axis(ax)

    fig.suptitle("Zoom Diagnostics: 2026-06-18 to 2026-06-19", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "zoom_0618_0619_voltage_action_guard.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _metrics(dep: pd.DataFrame, raw: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    neg = dep[_series(dep, "action_raw_kw").lt(-1e-6)].copy()
    final_neg = dep[_series(dep, "action_power_kw").lt(-1e-6)].copy()
    cutoff_dep = _series(dep, "batt_v_mean").lt(CUTOFF_V)
    cutoff_raw = _series(raw, "voltage_v").lt(CUTOFF_V)

    by_day = []
    for date, d in dep.groupby(dep["timestamp"].dt.strftime("%Y-%m-%d")):
        r = raw[raw["timestamp"].dt.strftime("%Y-%m-%d") == date]
        n = d[_series(d, "action_raw_kw").lt(-1e-6)]
        by_day.append(
            {
                "date": date,
                "deployment_rows": int(len(d)),
                "raw_rows": int(len(r)),
                "model_negative_raw_count": int(len(n)),
                "final_negative_action_count": int(_series(d, "action_power_kw").lt(-1e-6).sum()),
                "neg_raw_with_batt_v_ge_cutoff": int(_series(n, "batt_v_mean").ge(CUTOFF_V).sum()),
                "neg_raw_with_load_limit_guard": int(_series(n, "guard_block_load_over_discharge_limit").fillna(0).gt(0).sum()),
                "neg_raw_with_voltage_cutoff_active": int(_series(n, "voltage_cutoff_active").fillna(0).gt(0).sum()),
                "dep_batt_v_below_cutoff_count": int(_series(d, "batt_v_mean").lt(CUTOFF_V).sum()),
                "raw_voltage_below_cutoff_count": int(_series(r, "voltage_v").lt(CUTOFF_V).sum()),
                "voltage_cutoff_active_count": int(_series(d, "voltage_cutoff_active").fillna(0).gt(0).sum()),
                "voltage_cutoff_day_locked_count": int(_series(d, "voltage_cutoff_day_locked").fillna(0).gt(0).sum()),
                "median_batt_v": float(_series(d, "batt_v_mean").median()),
                "min_batt_v": float(_series(d, "batt_v_mean").min()),
                "median_load_w_neg_raw": float((_series(n, "load_kw") * 1000.0).median()) if len(n) else np.nan,
                "battery_solo_limit_w": float((_series(d, "flow_discharge_limit_kw") * 1000.0).median()),
            }
        )

    summary = {
        "window": ["2026-06-18T00:00:00", "2026-06-19T23:59:59"],
        "cutoff_voltage_v": CUTOFF_V,
        "deployment_rows": int(len(dep)),
        "raw_rows": int(len(raw)),
        "batt_v_mean_quantiles": {str(k): float(v) for k, v in _series(dep, "batt_v_mean").quantile([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).items()},
        "raw_voltage_quantiles": {str(k): float(v) for k, v in _series(raw, "voltage_v").quantile([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).items()},
        "dep_batt_v_below_cutoff_count": int(cutoff_dep.sum()),
        "raw_voltage_below_cutoff_count": int(cutoff_raw.sum()),
        "voltage_cutoff_active_count": int(_series(dep, "voltage_cutoff_active").fillna(0).gt(0).sum()),
        "voltage_cutoff_day_locked_count": int(_series(dep, "voltage_cutoff_day_locked").fillna(0).gt(0).sum()),
        "model_negative_raw_count": int(len(neg)),
        "final_negative_action_count": int(len(final_neg)),
        "negative_raw_with_batt_v_ge_cutoff_count": int(_series(neg, "batt_v_mean").ge(CUTOFF_V).sum()),
        "negative_raw_with_load_limit_guard_count": int(_series(neg, "guard_block_load_over_discharge_limit").fillna(0).gt(0).sum()),
        "negative_raw_with_voltage_cutoff_active_count": int(_series(neg, "voltage_cutoff_active").fillna(0).gt(0).sum()),
        "negative_raw_with_pv_active_count": int(_series(neg, "pv_active").fillna(0).gt(0).sum()),
        "negative_raw_coral_clipped_count": int(_series(neg, "coral_clipped").fillna(0).gt(0).sum()),
        "battery_solo_discharge_limit_w": float((_series(dep, "flow_discharge_limit_kw") * 1000.0).median()),
        "negative_raw_load_w_quantiles": {str(k): float(v) for k, v in (_series(neg, "load_kw") * 1000.0).quantile([0, 0.25, 0.5, 0.75, 1]).items()},
        "negative_raw_pv_support_ratio_quantiles": {str(k): float(v) for k, v in _series(neg, "pv_support_ratio").quantile([0, 0.25, 0.5, 0.75, 1]).items()},
        "negative_raw_soc_quantiles": {str(k): float(v) for k, v in _series(neg, "soc").quantile([0, 0.25, 0.5, 0.75, 1]).items()},
    }
    return summary, pd.DataFrame(by_day)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dep_all, _ = base._read_daily("deployment")
    raw_all, _ = base._read_daily("raw_data")
    dep = dep_all[(dep_all["timestamp"] >= START) & (dep_all["timestamp"] < END)].copy()
    raw = raw_all[(raw_all["timestamp"] >= START) & (raw_all["timestamp"] < END)].copy()
    if dep.empty or raw.empty:
        raise SystemExit("No 2026-06-18 to 2026-06-19 data found.")

    fig_path = _save_zoom(dep, raw)
    summary, by_day = _metrics(dep, raw)
    summary["outputs"] = [str(fig_path.relative_to(ROOT))]

    summary_path = OUT_DIR / "zoom_0618_0619_diagnostic_summary.json"
    by_day_path = OUT_DIR / "zoom_0618_0619_daily_guard_breakdown.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    by_day.to_csv(by_day_path, index=False)
    print(f"Wrote {fig_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {by_day_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

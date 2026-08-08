from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "figures" / "recent_2026_06_15_16"
START_DATE = "2026-06-15"
END_DATE = "2026-06-16"


def _read_daily(prefix: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date in (START_DATE, END_DATE):
        path = RAW_DIR / f"{prefix}_v2_{date}.csv"
        if not path.exists():
            continue
        df = _read_csv_flexible(path, prefix)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df["date"] = date
        for col in df.columns:
            if col not in {"timestamp", "date", "battery_id", "session_id", "experiment_name", "model_file", "current_mode", "load_source"}:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def _read_csv_flexible(path: Path, prefix: str) -> pd.DataFrame:
    """Read logs that may switch deployment schema within the same day."""
    if prefix != "deployment":
        return pd.read_csv(path)

    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return pd.DataFrame()

    base_header = rows[0]
    headers = {len(base_header): base_header}

    # The 2026-06-16 deployment file uses the newer schema. Use it as a
    # reference so the 2026-06-15 file can include both logger versions.
    schema_path = RAW_DIR / "deployment_v2_2026-06-16.csv"
    if schema_path.exists():
        with schema_path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
            schema_rows = list(csv.reader(fh))
        if schema_rows:
            headers[len(schema_rows[0])] = schema_rows[0]

    records: list[dict[str, str]] = []
    for row in rows[1:]:
        if not row:
            continue
        if row[0] == "timestamp":
            headers[len(row)] = row
            continue

        header = headers.get(len(row))
        if header is None:
            # Keep partially written rows usable by mapping what exists.
            longest = max(headers.values(), key=len)
            header = longest[: len(row)]
        records.append(dict(zip(header, row)))

    return pd.DataFrame(records)


def _setup_time_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))


def _save_raw_overview(raw: pd.DataFrame) -> Path:
    raw_5 = raw.set_index("timestamp").resample("5min").mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(5, 1, figsize=(17, 12.5), sharex=True)

    ax = axes[0]
    ax.plot(raw_5["timestamp"], raw_5["mppt_p_mw"] / 1000.0, color="#f2b705", lw=1.8, label="MPPT power")
    ax.plot(raw_5["timestamp"], raw_5["bus_p_mw"] / 1000.0, color="#2ca02c", lw=1.5, label="Bus power")
    ax.plot(raw_5["timestamp"], raw_5["load_p_mw"] / 1000.0, color="#8c564b", lw=1.5, label="Load power")
    ax.plot(raw_5["timestamp"], raw_5["grid_p_mw"] / 1000.0, color="#7f7f7f", lw=1.2, alpha=0.85, label="Grid power")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper right", ncol=4)
    _setup_time_axis(ax)

    ax = axes[1]
    ax.plot(raw_5["timestamp"], raw_5["voltage_v"], color="#1f77b4", lw=1.6, label="Battery voltage")
    ax.plot(raw_5["timestamp"], raw_5["charge_voltage_v"], color="#17becf", lw=1.2, alpha=0.8, label="Charge voltage")
    ax.axhline(4.2, color="#d62728", lw=1.0, ls="--", alpha=0.8, label="4.2 V reference")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[2]
    ax.plot(raw_5["timestamp"], raw_5["current_raw_ma"], color="#ff7f0e", lw=1.2, label="Raw current")
    ax.plot(raw_5["timestamp"], raw_5["current_ma"], color="#d62728", lw=1.2, alpha=0.85, label="Signed current")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    ax = axes[3]
    ax.plot(raw_5["timestamp"], raw_5["soc_calc"] * 100.0, color="#2ca02c", lw=1.6, label="Energy SoE / control SoC")
    if "soc_coulomb" in raw_5:
        ax.plot(raw_5["timestamp"], raw_5["soc_coulomb"] * 100.0, color="#bcbd22", lw=1.1, ls="--", alpha=0.8, label="Coulomb SoC")
    ax.plot(raw_5["timestamp"], raw_5["soc_percent"], color="#7f7f7f", lw=1.0, alpha=0.75, label="Firmware SoC")
    ax.axhline(20, color="#d62728", lw=1.0, ls="--", alpha=0.7, label="20% lower bound")
    ax.axhline(80, color="#d62728", lw=1.0, ls="--", alpha=0.7, label="80% upper bound")
    ax.set_ylabel("SoC (%)")
    ax.legend(loc="upper right", ncol=4)
    _setup_time_axis(ax)

    ax = axes[4]
    if "speed_percent" in raw_5:
        ax.plot(raw_5["timestamp"], raw_5["speed_percent"], color="#9467bd", lw=1.4, label="Flow speed")
    if "situation_code" in raw_5:
        ax.step(raw_5["timestamp"], raw_5["situation_code"], where="post", color="#1f77b4", lw=1.0, alpha=0.7, label="Situation code")
    ax.set_ylabel("Flow / mode")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    fig.suptitle("Raw Data Overview: 2026-06-15 to 2026-06-16", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "raw_data_2026-06-15_to_2026-06-16_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _save_deployment_overview(dep: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(5, 1, figsize=(17, 12.5), sharex=True)

    ax = axes[0]
    ax.plot(dep["timestamp"], dep["load_kw"] * 1000.0, color="#8c564b", lw=1.8, label="Load")
    ax.plot(dep["timestamp"], dep["pv_kw"] * 1000.0, color="#f2b705", lw=1.8, label="PV")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    ax = axes[1]
    ax.plot(dep["timestamp"], dep["soc"] * 100.0, color="#2ca02c", lw=1.8, label="Energy SoE / deployment SoC")
    if "soc_coulomb" in dep:
        ax.plot(dep["timestamp"], dep["soc_coulomb"] * 100.0, color="#bcbd22", lw=1.1, ls="--", alpha=0.8, label="Coulomb SoC")
    ax.axhline(20, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(80, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("SoC (%)")
    ax.legend(loc="upper right")
    _setup_time_axis(ax)

    ax = axes[2]
    if "action_raw_kw" in dep:
        ax.step(dep["timestamp"], dep["action_raw_kw"] * 1000.0, where="post", color="#9467bd", lw=1.5, label="Raw action")
    ax.step(dep["timestamp"], dep["action_power_kw"] * 1000.0, where="post", color="#2ca02c", lw=1.8, label="Executed action")
    ax.step(dep["timestamp"], dep["power_mw_cmd"] / 1000.0, where="post", color="#1f77b4", lw=1.2, ls="--", alpha=0.9, label="Command magnitude")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Action (W)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[3]
    events = [
        ("coral_clipped", "CORAL clipped", "#ff7f0e"),
        ("guard_block_low_soc_discharge", "Low-SoC discharge block", "#d62728"),
        ("guard_block_pv_active_discharge", "PV-active discharge block", "#2ca02c"),
        ("guard_block_load_over_discharge_limit", "Load > discharge limit block", "#8c564b"),
        ("guard_block_no_pv_surplus_charge", "No-PV-surplus charge block", "#9467bd"),
        ("guard_flow_power_limited", "Flow power limited", "#1f77b4"),
    ]
    active_labels = []
    for idx, (col, label, color) in enumerate(events, start=1):
        if col in dep:
            active = pd.to_numeric(dep[col], errors="coerce").fillna(0).gt(0).to_numpy()
            ax.fill_between(
                dep["timestamp"],
                idx - 0.35,
                idx + 0.35,
                where=active,
                step="post",
                alpha=0.45,
                color=color,
                label=label,
            )
            active_labels.append((idx, label))
    ax.set_yticks([idx for idx, _ in active_labels])
    ax.set_yticklabels([label for _, label in active_labels])
    ax.set_ylim(0.4, max(1.6, len(active_labels) + 0.6))
    ax.set_ylabel("Safety events")
    _setup_time_axis(ax)

    ax = axes[4]
    if "action_flow_pct" in dep:
        ax.step(dep["timestamp"], dep["action_flow_pct"], where="post", color="#9467bd", lw=1.5, label="Model flow")
    if "flow_pct_cmd" in dep:
        ax.step(dep["timestamp"], dep["flow_pct_cmd"], where="post", color="#1f77b4", lw=1.5, ls="--", label="Command flow")
    if "completeness" in dep:
        ax2 = ax.twinx()
        ax2.plot(dep["timestamp"], dep["completeness"] * 100.0, color="#7f7f7f", lw=1.0, alpha=0.65, label="Completeness")
        ax2.set_ylabel("Completeness (%)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=3)
    else:
        ax.legend(loc="upper right", ncol=2)
    ax.set_ylabel("Flow (%)")
    ax.set_xlabel("Time")
    _setup_time_axis(ax)

    fig.suptitle("Deployment Control Overview: 2026-06-15 to 2026-06-16", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "deployment_2026-06-15_to_2026-06-16_control_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _save_requested_overview(raw: pd.DataFrame, dep: pd.DataFrame) -> Path:
    raw_5 = raw.set_index("timestamp").resample("5min").mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(5, 1, figsize=(17, 12.5), sharex=True)

    ax = axes[0]
    if "pv_support_ratio" in dep:
        ax.plot(dep["timestamp"], dep["pv_support_ratio"], color="#2ca02c", lw=1.8, label="PV/load ratio")
    elif {"pv_kw", "load_kw"}.issubset(dep.columns):
        ratio = dep["pv_kw"] / dep["load_kw"].replace(0, np.nan)
        ax.plot(dep["timestamp"], ratio, color="#2ca02c", lw=1.8, label="PV/load ratio")
    ax.axhline(0.8, color="#2ca02c", lw=1.0, ls=":", alpha=0.85, label="ratio=0.8")
    ax.set_ylabel("PV/load ratio")
    ax.set_ylim(0, 1.65)
    ax_load = ax.twinx()
    ax_load.plot(dep["timestamp"], dep["load_kw"] * 1000.0, color="#8c564b", lw=1.5, alpha=0.9, label="Load power")
    ax_load.set_ylabel("Load (W)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_load.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[1]
    ax.plot(dep["timestamp"], dep["soc"] * 100.0, color="#2ca02c", lw=1.8, label="Energy SoE / control SoC")
    if "soc_coulomb" in dep:
        ax.plot(dep["timestamp"], dep["soc_coulomb"] * 100.0, color="#bcbd22", lw=1.1, ls="--", alpha=0.8, label="Coulomb SoC")
    ax.axhline(20, color="#d62728", lw=1.0, ls="--", alpha=0.75, label="20%")
    ax.axhline(80, color="#d62728", lw=1.0, ls="--", alpha=0.75, label="80%")
    ax.set_ylabel("SoC (%)")
    ax.legend(loc="upper right", ncol=3)
    _setup_time_axis(ax)

    ax = axes[2]
    if "action_raw_kw" in dep:
        ax.step(dep["timestamp"], dep["action_raw_kw"] * 1000.0, where="post", color="#9467bd", lw=1.6, label="Raw action")
    ax.step(dep["timestamp"], dep["action_power_kw"] * 1000.0, where="post", color="#2ca02c", lw=1.8, label="Final action")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Action (W)")
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    ax = axes[3]
    if "action_flow_pct" in dep:
        ax.step(dep["timestamp"], dep["action_flow_pct"], where="post", color="#9467bd", lw=1.5, label="Raw flow rate")
    if "flow_pct_cmd" in dep:
        ax.step(dep["timestamp"], dep["flow_pct_cmd"], where="post", color="#1f77b4", lw=1.6, ls="--", label="Final flow rate")
    ax.set_ylabel("Flow rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right", ncol=2)
    _setup_time_axis(ax)

    ax = axes[4]
    ax.plot(raw_5["timestamp"], raw_5["voltage_v"], color="#1f77b4", lw=1.6, label="Battery voltage")
    ax.axhline(4.2, color="#d62728", lw=1.0, ls="--", alpha=0.75, label="4.2 V")
    ax.set_ylabel("Voltage (V)")
    ax_current = ax.twinx()
    ax_current.plot(raw_5["timestamp"], raw_5["current_ma"], color="#ff7f0e", lw=1.2, alpha=0.9, label="Battery current")
    ax_current.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax_current.set_ylabel("Current (mA)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_current.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=3)
    ax.set_xlabel("Time")
    _setup_time_axis(ax)

    fig.suptitle("Deployment Overview: 2026-06-15 to 2026-06-16", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "deployment_2026-06-15_to_2026-06-16_requested_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _daily_summary(raw: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in (START_DATE, END_DATE):
        raw_day = raw[raw["date"] == date]
        dep_day = dep[dep["date"] == date]
        row: dict[str, float | int | str] = {"date": date}
        row["raw_samples"] = int(len(raw_day))
        row["deployment_steps"] = int(len(dep_day))
        if not raw_day.empty:
            row["raw_start"] = str(raw_day["timestamp"].min())
            row["raw_end"] = str(raw_day["timestamp"].max())
            row["mean_mppt_w"] = float(raw_day["mppt_p_mw"].mean() / 1000.0)
            row["max_mppt_w"] = float(raw_day["mppt_p_mw"].max() / 1000.0)
            row["mean_load_w"] = float(raw_day["load_p_mw"].mean() / 1000.0)
            row["soc_calc_min_pct"] = float(raw_day["soc_calc"].min() * 100.0)
            row["soc_calc_max_pct"] = float(raw_day["soc_calc"].max() * 100.0)
        if not dep_day.empty:
            dep_day = dep_day.sort_values("timestamp").copy()
            dt_h = dep_day["timestamp"].shift(-1).sub(dep_day["timestamp"]).dt.total_seconds() / 3600.0
            fallback_dt = float(dt_h[(dt_h > 0) & (dt_h <= 0.25)].median())
            if not np.isfinite(fallback_dt):
                fallback_dt = 1.0 / 60.0
            dt_h = dt_h.fillna(fallback_dt).clip(lower=0.0, upper=0.25)
            row["load_energy_wh"] = float((dep_day["load_kw"] * 1000.0 * dt_h).sum())
            row["pv_energy_wh"] = float((dep_day["pv_kw"] * 1000.0 * dt_h).sum())
            executed = dep_day["action_power_kw"] * 1000.0
            row["charge_command_wh"] = float((executed.clip(lower=0) * dt_h).sum())
            row["discharge_command_wh"] = float((-executed.clip(upper=0) * dt_h).sum())
            row["coral_clipped_steps"] = int(
                pd.to_numeric(dep_day.get("coral_clipped", pd.Series(0, index=dep_day.index)), errors="coerce")
                .fillna(0)
                .gt(0)
                .sum()
            )
            guard_cols = [c for c in dep_day.columns if c.startswith("guard_block_")]
            row["guard_block_steps"] = int(dep_day[guard_cols].sum(axis=1).gt(0).sum()) if guard_cols else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _save_daily_summary(summary: pd.DataFrame) -> tuple[Path, Path]:
    csv_path = OUT_DIR / "daily_summary_2026-06-15_to_2026-06-16.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    x = np.arange(len(summary))
    labels = summary["date"].tolist()

    axes[0, 0].bar(x - 0.18, summary["load_energy_wh"], width=0.36, color="#8c564b", label="Load")
    axes[0, 0].bar(x + 0.18, summary["pv_energy_wh"], width=0.36, color="#f2b705", label="PV")
    axes[0, 0].set_title("Daily energy")
    axes[0, 0].set_ylabel("Wh")
    axes[0, 0].legend()

    axes[0, 1].bar(x - 0.18, summary["charge_command_wh"], width=0.36, color="#2ca02c", label="Charge command")
    axes[0, 1].bar(x + 0.18, summary["discharge_command_wh"], width=0.36, color="#d62728", label="Discharge command")
    axes[0, 1].set_title("Executed battery command energy")
    axes[0, 1].set_ylabel("Wh")
    axes[0, 1].legend()

    axes[1, 0].bar(x - 0.18, summary["soc_calc_min_pct"], width=0.36, color="#7f7f7f", label="Min")
    axes[1, 0].bar(x + 0.18, summary["soc_calc_max_pct"], width=0.36, color="#2ca02c", label="Max")
    axes[1, 0].axhline(20, color="#d62728", lw=1.0, ls="--")
    axes[1, 0].axhline(80, color="#d62728", lw=1.0, ls="--")
    axes[1, 0].set_title("Calculated SoC range")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].legend()

    axes[1, 1].bar(x - 0.18, summary["coral_clipped_steps"], width=0.36, color="#1f77b4", label="CORAL clipped")
    axes[1, 1].bar(x + 0.18, summary["guard_block_steps"], width=0.36, color="#ff7f0e", label="Any guard block")
    axes[1, 1].set_title("Safety-layer activity")
    axes[1, 1].set_ylabel("15-min steps")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Daily Summary: 2026-06-15 to 2026-06-16", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = OUT_DIR / "daily_summary_2026-06-15_to_2026-06-16.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path, csv_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.2,
        }
    )

    raw = _read_daily("raw_data")
    dep = _read_daily("deployment")
    if raw.empty:
        raise SystemExit("No raw data found for 2026-06-15 to 2026-06-16.")
    if dep.empty:
        raise SystemExit("No deployment data found for 2026-06-15 to 2026-06-16.")

    outputs = [_save_requested_overview(raw, dep)]

    for out in outputs:
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
ㄋ
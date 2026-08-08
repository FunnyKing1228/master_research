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
OUT_DIR = RAW_DIR / "figures" / "deployment_0520_0528_review"
RAW_DAYS = pd.date_range("2026-05-20", "2026-05-28", freq="D")
DEP_DAYS = pd.date_range("2026-05-20", "2026-05-29", freq="D")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data = rows[1:]
    inserts = [
        ("guard_block_load_over_discharge_limit", "guard_block_discharge_intent_threshold"),
        ("guard_block_firmware_override_discharge", "guard_block_isolated_load_bus_discharge"),
        ("firmware_override_discharge_samples_window", "isolated_load_bus_samples_window"),
    ]
    while any(len(r) > len(header) for r in data):
        for prev, new in inserts:
            if prev in header and new not in header:
                idx = header.index(prev) + 1
                header = header[:idx] + [new] + header[idx:]
                break
        else:
            break

    normalized: list[list[str]] = []
    for row in data:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        normalized.append(row)
    return pd.DataFrame(normalized, columns=header)


def _load_logs(prefix: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in days:
        path = RAW_DIR / f"{prefix}_{day:%Y-%m-%d}.csv"
        if not path.exists():
            continue
        df = _read_csv_flexible(path)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = day.date()
        df["source_file"] = path.name
        frames.append(df.dropna(subset=["timestamp"]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    for col in out.columns:
        if col not in {"timestamp", "date", "source_file", "session_id", "experiment_name", "model_file", "current_mode", "load_source", "battery_id"}:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().any():
                out[col] = converted
    return out


def _presence_summary() -> pd.DataFrame:
    days = pd.date_range("2026-05-19", "2026-05-29", freq="D")
    return pd.DataFrame(
        [
            {
                "date": f"{day:%Y-%m-%d}",
                "raw_exists": (RAW_DIR / f"raw_data_v2_{day:%Y-%m-%d}.csv").exists(),
                "deployment_exists": (RAW_DIR / f"deployment_v2_{day:%Y-%m-%d}.csv").exists(),
            }
            for day in days
        ]
    )


def _summaries(raw: pd.DataFrame, dep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    for date, g in raw.groupby("date"):
        v = pd.to_numeric(g["voltage_v"], errors="coerce")
        valid_v = v[v > 0]
        raw_rows.append(
            {
                "date": str(date),
                "raw_samples": len(g),
                "start": g["timestamp"].min(),
                "end": g["timestamp"].max(),
                "soc_start_pct": float(g["soc_calc"].iloc[0] * 100.0),
                "soc_end_pct": float(g["soc_calc"].iloc[-1] * 100.0),
                "soc_min_pct": float(g["soc_calc"].min() * 100.0),
                "soc_max_pct": float(g["soc_calc"].max() * 100.0),
                "voltage_min_v": float(valid_v.min()) if len(valid_v) else np.nan,
                "voltage_p05_v": float(valid_v.quantile(0.05)) if len(valid_v) else np.nan,
                "voltage_mean_v": float(valid_v.mean()) if len(valid_v) else np.nan,
                "low_voltage_samples_lt_4p2": int(((v > 0) & (v < 4.2)).sum()),
                "zero_voltage_samples": int((v == 0).sum()),
                "grid_w_mean": float(g["grid_p_mw"].mean() / 1000.0),
                "load_w_mean": float(g["load_p_mw"].mean() / 1000.0),
                "pv_w_mean": float(g["bus_p_mw"].mean() / 1000.0),
                "sit1_samples": int((g["situation_code"] == 1).sum()),
                "sit3_samples": int((g["situation_code"] == 3).sum()),
                "sit4_samples": int((g["situation_code"] == 4).sum()),
            }
        )

    dep_rows: list[dict[str, object]] = []
    guard_cols = [
        "guard_block_voltage_cutoff",
        "guard_block_low_soc_discharge",
        "guard_block_load_over_discharge_limit",
        "guard_block_discharge_intent_threshold",
        "guard_block_firmware_override_discharge",
        "guard_block_isolated_load_bus_discharge",
        "cutoff_soc_fallback_applied",
    ]
    if not dep.empty:
        for date, g in dep.groupby("date"):
            row: dict[str, object] = {
                "date": str(date),
                "dep_steps": len(g),
                "cmd_discharge_steps": int((g["action_power_kw"] < 0).sum()) if "action_power_kw" in g else 0,
                "cmd_charge_steps": int((g["action_power_kw"] > 0).sum()) if "action_power_kw" in g else 0,
                "cmd_standby_steps": int((g["action_power_kw"] == 0).sum()) if "action_power_kw" in g else 0,
            }
            for col in guard_cols:
                row[col] = int(pd.to_numeric(g.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            dep_rows.append(row)

    return pd.DataFrame(raw_rows), pd.DataFrame(dep_rows)


def _plot(raw: pd.DataFrame, dep: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    raw_plot = raw.set_index("timestamp").resample("10min").median(numeric_only=True).reset_index()
    dep_plot = dep.copy()

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.05, 0.95, 1.0, 0.75, 0.72]},
    )

    ax = axes[0]
    ax.plot(raw_plot["timestamp"], raw_plot["load_p_mw"] / 1000.0, color="#4C566A", lw=1.2, label="Load")
    ax.plot(raw_plot["timestamp"], raw_plot["bus_p_mw"] / 1000.0, color="#EBCB8B", lw=1.2, label="PV bus")
    ax.plot(raw_plot["timestamp"], raw_plot["grid_p_mw"] / 1000.0, color="#BF616A", lw=1.0, label="Grid")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper left", ncol=3, frameon=False)

    ax = axes[1]
    low_v = raw_plot["voltage_v"].between(0.01, 4.2)
    if low_v.any():
        ax.fill_between(
            raw_plot["timestamp"],
            0,
            9,
            where=low_v,
            color="#BF616A",
            alpha=0.10,
            step="mid",
            label="V < 4.2V",
        )
    ax.plot(raw_plot["timestamp"], raw_plot["voltage_v"], color="#B48EAD", lw=1.1, label="Battery voltage")
    ax.axhline(4.2, color="#BF616A", ls="--", lw=0.9, label="Cutoff 4.2V")
    ax.axhline(5.0, color="#D08770", ls="--", lw=0.8, label="Recover 5.0V")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper left", ncol=3, frameon=False)

    ax = axes[2]
    ax.plot(raw_plot["timestamp"], raw_plot["soc_calc"] * 100.0, color="#5E81AC", lw=1.2, label="Control SoC")
    ax.axhspan(20, 80, color="#A3BE8C", alpha=0.10)
    ax.axhline(20, color="#BF616A", ls="--", lw=0.9)
    ax.set_ylim(-2, 102)
    ax.set_ylabel("SoC (%)")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[3]
    ax.plot(raw_plot["timestamp"], raw_plot["current_raw_ma"], color="#2E3440", lw=0.9, label="Firmware current")
    ax.plot(raw_plot["timestamp"], raw_plot["current_ma"], color="#D08770", lw=0.85, alpha=0.85, label="SoC accounting current")
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel("Current (mA)")
    ax.legend(loc="upper left", ncol=2, frameon=False)

    ax = axes[4]
    if not dep_plot.empty:
        cmd_w = pd.to_numeric(dep_plot["action_power_kw"], errors="coerce") * 1000.0
        ax.fill_between(dep_plot["timestamp"], 0, cmd_w.clip(lower=0), step="post", color="#5E81AC", alpha=0.35, label="Charge cmd")
        ax.fill_between(dep_plot["timestamp"], 0, cmd_w.clip(upper=0), step="post", color="#BF616A", alpha=0.35, label="Discharge cmd")
        ax.step(dep_plot["timestamp"], cmd_w, where="post", color="#434C5E", lw=0.9)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel("Command\n(W)")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,
    )

    ax = axes[5]
    event_specs = [
        ("cutoff_soc_fallback_applied", "#BF616A", "SoC fallback"),
        ("guard_block_discharge_intent_threshold", "#EBCB8B", "Small discharge blocked"),
        ("guard_block_voltage_cutoff", "#D08770", "Voltage cutoff guard"),
        ("guard_block_firmware_override_discharge", "#5E81AC", "FW override block"),
        ("guard_block_isolated_load_bus_discharge", "#A3BE8C", "Load-bus block"),
    ]
    y_positions: list[int] = []
    y_labels: list[str] = []
    if not dep_plot.empty:
        for idx, (col, color, label) in enumerate(event_specs):
            y = len(event_specs) - idx
            y_positions.append(y)
            y_labels.append(label)
            if col not in dep_plot.columns:
                continue
            mask = pd.to_numeric(dep_plot[col], errors="coerce").fillna(0) > 0
            if mask.any():
                ax.scatter(
                    dep_plot.loc[mask, "timestamp"],
                    np.full(mask.sum(), y),
                    s=24,
                    color=color,
                    marker="o",
                    label=label,
                    zorder=3,
                )
    # Raw low-voltage samples are frequent; show them as a thin rug instead of
    # mixing them with command magnitudes.
    low_v_raw = raw_plot["voltage_v"].between(0.01, 4.2)
    if low_v_raw.any():
        y = 0
        y_positions.append(y)
        y_labels.append("Raw V<4.2")
        ax.scatter(
            raw_plot.loc[low_v_raw, "timestamp"],
            np.full(low_v_raw.sum(), y),
            s=5,
            color="#BF616A",
            alpha=0.35,
            marker="|",
            label="Raw V<4.2",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylim(-0.7, len(event_specs) + 1.0)
    ax.set_ylabel("Events")
    ax.grid(True, axis="x", color="0.92", lw=0.5)

    for ax in axes:
        ax.grid(True, axis="y", color="0.9", lw=0.6)

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    axes[-1].set_xlim(pd.Timestamp("2026-05-20 00:00"), pd.Timestamp("2026-05-29 00:00"))
    axes[-1].set_xlabel("Time")
    fig.suptitle("Deployment review 2026-05-20 to 2026-05-28: raw measurements, SoC, commands, and guards", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _load_logs("raw_data_v2", RAW_DAYS)
    dep = _load_logs("deployment_v2", DEP_DAYS)
    if raw.empty:
        raise FileNotFoundError("No raw_data_v2 files found for 2026-05-20..28")

    raw_summary, dep_summary = _summaries(raw, dep)
    presence = _presence_summary()
    raw_summary.to_csv(OUT_DIR / "raw_0520_0528_daily_summary.csv", index=False, encoding="utf-8-sig")
    dep_summary.to_csv(OUT_DIR / "deployment_0520_0529_daily_summary.csv", index=False, encoding="utf-8-sig")
    presence.to_csv(OUT_DIR / "log_presence_0519_0529.csv", index=False, encoding="utf-8-sig")
    _plot(raw, dep, OUT_DIR / "deployment_0520_0528_review_overview.png")

    print(f"Wrote outputs to: {OUT_DIR}")
    print("\nPresence:")
    print(presence.to_string(index=False))
    print("\nRaw summary:")
    print(raw_summary.to_string(index=False))
    print("\nDeployment summary:")
    print(dep_summary.to_string(index=False))


if __name__ == "__main__":
    main()


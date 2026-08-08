"""Report-style 2026-05-15..18 deployment plots.

Panels:
1) Load / PV
2) Battery voltage
3) SoC
4) Commanded charge/discharge power (from deployment decisions)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "figures" / "deployment_0515_0518_current_clean"
DAYS = pd.date_range("2026-05-15", "2026-05-18", freq="D")


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
        raise FileNotFoundError("No raw_data_v2 files found for 2026-05-15..18")

    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    numeric_cols = [
        "voltage_v",
        "current_raw_ma",
        "soc_calc",
        "bus_p_mw",
        "load_p_mw",
        "grid_p_mw",
        "mppt_p_mw",
        "situation_code",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read deployment logs across adjacent schema versions."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data_rows = rows[1:]

    if (
        "guard_block_discharge_intent_threshold" not in header
        and any(len(r) == len(header) + 1 for r in data_rows)
        and "guard_block_load_over_discharge_limit" in header
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


def load_deployment() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in DAYS:
        path = RAW_DIR / f"deployment_v2_{day:%Y-%m-%d}.csv"
        if not path.exists():
            continue
        df = _read_csv_flexible(path)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No deployment_v2 files found for 2026-05-15..18")

    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    numeric_cols = ["load_kw", "pv_kw", "action_power_kw", "power_mw_cmd", "situation_code"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def resample_for_plot(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    return (
        df.set_index("timestamp")
        .resample(rule)
        .median(numeric_only=True)
        .dropna(how="all")
        .reset_index()
    )


def setup_axes(nrows: int, figsize: tuple[float, float]):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    return plt.subplots(nrows, 1, figsize=figsize, sharex=True)


def plot_report(
    raw_df: pd.DataFrame,
    dep_df: pd.DataFrame,
    out_path: Path,
    title: str,
    xlim: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> None:
    plot_raw = resample_for_plot(raw_df)
    plot_dep = dep_df.copy().sort_values("timestamp")
    fig, axes = setup_axes(4, (14, 8))

    ax = axes[0]
    ax.step(plot_dep["timestamp"], plot_dep["load_kw"] * 1000.0, where="post", color="#4C566A", lw=1.3, label="Load")
    ax.step(plot_dep["timestamp"], plot_dep["pv_kw"] * 1000.0, where="post", color="#EBCB8B", lw=1.3, label="PV")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper left", ncol=2, frameon=False)

    ax = axes[1]
    ax.plot(plot_raw["timestamp"], plot_raw["voltage_v"], color="#B48EAD", lw=1.2)
    ax.axhline(4.2, color="#BF616A", ls="--", lw=0.9, label="Cutoff 4.2V")
    ax.set_ylabel("Battery V")
    ax.legend(loc="upper left", ncol=1, frameon=False)

    ax = axes[2]
    ax.plot(plot_raw["timestamp"], plot_raw["soc_calc"] * 100.0, color="#5E81AC", lw=1.4)
    ax.axhspan(10, 90, color="#A3BE8C", alpha=0.10)
    ax.axhline(20, color="#BF616A", ls="--", lw=0.9)
    ax.set_ylim(-2, 102)
    ax.set_ylabel("SoC (%)")

    ax = axes[3]
    cmd_w = plot_dep["action_power_kw"] * 1000.0
    charge_w = cmd_w.clip(lower=0.0)
    discharge_w = cmd_w.clip(upper=0.0)
    ax.fill_between(plot_dep["timestamp"], 0, charge_w, step="post", alpha=0.35, color="#5E81AC", label="Command charge (+)")
    ax.fill_between(plot_dep["timestamp"], 0, discharge_w, step="post", alpha=0.35, color="#BF616A", label="Command discharge (-)")
    ax.step(plot_dep["timestamp"], cmd_w, where="post", color="#2E3440", lw=1.0)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("Command (W)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left", ncol=2, frameon=False)

    for ax in axes:
        ax.grid(True, axis="y", color="0.9", lw=0.6)

    if xlim is None:
        xlim = (pd.Timestamp("2026-05-15 00:00"), pd.Timestamp("2026-05-18 23:59"))
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=8))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    else:
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlim(*xlim)

    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize(raw_df: pd.DataFrame, dep_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for date, g in raw_df.groupby(raw_df["timestamp"].dt.date):
        d = dep_df[dep_df["timestamp"].dt.date == date].copy()
        g = g.sort_values("timestamp")
        rows.append(
            {
                "date": str(date),
                "raw_samples": len(g),
                "dep_steps": len(d),
                "soc_start_pct": float(g["soc_calc"].iloc[0] * 100.0),
                "soc_end_pct": float(g["soc_calc"].iloc[-1] * 100.0),
                "voltage_min_v": float(g["voltage_v"].replace(0, pd.NA).min(skipna=True)),
                "cmd_charge_steps": int((d["action_power_kw"] > 0).sum()) if not d.empty else 0,
                "cmd_discharge_steps": int((d["action_power_kw"] < 0).sum()) if not d.empty else 0,
                "cmd_standby_steps": int((d["action_power_kw"] == 0).sum()) if not d.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    dep = load_deployment()
    summary = summarize(raw, dep)
    summary.to_csv(OUT_DIR / "deployment_0515_0518_report_clean_summary.csv", index=False, encoding="utf-8-sig")

    plot_report(
        raw,
        dep,
        OUT_DIR / "deployment_0515_0518_report_clean_overview.png",
        "Deployment 2026-05-15 to 2026-05-18: Load/PV, Battery Voltage, SoC, and Command",
    )

    for day in DAYS:
        start = pd.Timestamp(day)
        end = start + pd.Timedelta(days=1)
        daily_raw = raw[(raw["timestamp"] >= start) & (raw["timestamp"] < end)]
        daily_dep = dep[(dep["timestamp"] >= start) & (dep["timestamp"] < end)]
        if daily_raw.empty or daily_dep.empty:
            continue
        plot_report(
            daily_raw,
            daily_dep,
            OUT_DIR / f"deployment_{day:%Y-%m-%d}_report_clean.png",
            f"Deployment {day:%Y-%m-%d}: Load/PV, Battery Voltage, SoC, and Command",
            xlim=(start, end),
        )

    print(f"Wrote report clean plots to: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


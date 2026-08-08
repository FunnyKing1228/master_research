from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "experiments" / "seminar_baseline_results" / "dataset_overview"
OUT_PATH = OUT_DIR / "p302_dataset_overview_through_2026-05-18_clean_no_text_pv_ratio.png"
CURVE_OUT_PATH = OUT_DIR / "p302_dataset_overview_through_2026-05-18_clean_no_text_pv_ratio_soc_curve.png"
START_DATE = pd.Timestamp("2026-04-24")
END_DATE = pd.Timestamp("2026-05-18")


def _load_raw_day(day: pd.Timestamp) -> pd.DataFrame:
    path = RAW_DIR / f"raw_data_v2_{day:%Y-%m-%d}.csv"
    if not path.exists():
        return pd.DataFrame()
    usecols = [
        "timestamp",
        "soc_calc",
        "load_p_mw",
        "bus_p_mw",
        "mppt_p_mw",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["soc_calc", "load_p_mw", "bus_p_mw", "mppt_p_mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def _coverage_status(n_samples: int) -> str:
    if n_samples >= 8500:
        return "clean"
    if n_samples >= 7000:
        return "collected"
    return "partial"


def _daily_summary(days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []
    for day in days:
        df = _load_raw_day(day)
        if df.empty:
            rows.append(
                {
                    "date": day,
                    "n": 0,
                    "coverage": "partial",
                    "load_peak_w": np.nan,
                    "pv_peak_w": np.nan,
                    "pv_ratio_p90": np.nan,
                    "pv_ratio_max": np.nan,
                }
            )
            continue

        load_w = df.get("load_p_mw", pd.Series(dtype=float)) / 1000.0
        pv_w = df.get("bus_p_mw", pd.Series(dtype=float)) / 1000.0
        # Fallback to MPPT power if bus power is unavailable in older logs.
        if pv_w.notna().sum() == 0 and "mppt_p_mw" in df.columns:
            pv_w = df["mppt_p_mw"] / 1000.0
        ratio = pv_w / load_w.clip(lower=1e-6)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        rows.append(
            {
                "date": day,
                "n": len(df),
                "coverage": _coverage_status(len(df)),
                "load_peak_w": float(load_w.quantile(0.98)) if load_w.notna().any() else np.nan,
                "pv_peak_w": float(pv_w.quantile(0.98)) if pv_w.notna().any() else np.nan,
                "pv_ratio_p90": float(ratio.quantile(0.90)) if ratio.notna().any() else np.nan,
                "pv_ratio_max": float(ratio.quantile(0.99)) if ratio.notna().any() else np.nan,
            }
        )

        if "soc_calc" in df.columns:
            soc = df[["timestamp", "soc_calc"]].copy()
            soc["soc_pct"] = soc["soc_calc"].clip(0.0, 1.0) * 100.0
            all_frames.append(soc[["timestamp", "soc_pct"]])

    summary = pd.DataFrame(rows)
    soc_curve = (
        pd.concat(all_frames, ignore_index=True).sort_values("timestamp")
        if all_frames
        else pd.DataFrame(columns=["timestamp", "soc_pct"])
    )
    return summary, soc_curve


def _plot(summary: pd.DataFrame, soc_curve: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
        }
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [0.62, 1.2, 1.0, 1.35]},
    )

    color_clean = "#1b7f3a"
    color_collected = "#8b98aa"
    color_partial = "#f39c12"
    dates = summary["date"]

    ax = axes[0]
    for _, row in summary.iterrows():
        color = {
            "clean": color_clean,
            "collected": color_collected,
            "partial": color_partial,
        }[row["coverage"]]
        hatch = "///" if row["coverage"] == "partial" else None
        ax.bar(row["date"], 0.7, width=0.78, color=color, edgecolor="white", hatch=hatch)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Dataset\ncoverage")
    ax.legend(
        handles=[
            Patch(facecolor=color_clean, edgecolor="white", label="Clean full-day episodes"),
            Patch(facecolor=color_collected, edgecolor="white", label="Collected logs"),
            Patch(facecolor=color_partial, edgecolor="white", hatch="///", label="QC-needed / partial"),
        ],
        loc="upper left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.0, 1.18),
    )

    ax = axes[1]
    ax.plot(dates, summary["load_peak_w"], color="#1f5eff", marker="o", ms=3, lw=1.6, label="Load peak")
    ax.bar(dates, summary["pv_peak_w"], width=0.38, color="#f2a93b", alpha=0.95, label="PV peak")
    ax.set_ylabel("Daily peak\npower (W)")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.text(0.995, 0.78, "Lab-scale P302 testbed", transform=ax.transAxes, ha="right", color="#77849a", fontsize=8)

    ax = axes[2]
    ax.bar(dates, summary["pv_ratio_p90"], width=0.55, color="#8e5ce6", alpha=0.86, label="PV/load ratio (90th percentile)")
    ax.scatter(dates, summary["pv_ratio_max"], color="#4b2aa2", s=12, label="Daily max", zorder=3)
    ax.axhline(1.0, color="#8e5ce6", ls="--", lw=1.0)
    ax.set_ylabel("PV / load\nratio")
    ax.set_ylim(0, max(3.0, float(np.nanmax(summary["pv_ratio_max"].to_numpy(dtype=float))) * 1.08))
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.text(0.995, 0.78, "ratio > 1: PV can cover load", transform=ax.transAxes, ha="right", color="#8e5ce6", fontsize=8)

    ax = axes[3]
    ax.axhspan(20, 80, color="#9aa3ad", alpha=0.12, label="Training SoC target (20-80%)")
    ax.axhline(20, color="#8b98aa", ls="--", lw=0.9)
    ax.axhline(80, color="#8b98aa", ls="--", lw=0.9)
    if not soc_curve.empty:
        curve = (
            soc_curve.set_index("timestamp")["soc_pct"]
            .resample("10min")
            .median()
            .interpolate(limit=3)
            .reset_index()
        )
        # Break long gaps so unrelated days/sessions do not get connected visually.
        gap = curve["timestamp"].diff() > pd.Timedelta(hours=2)
        start = 0
        label_used = False
        for idx in list(np.flatnonzero(gap.to_numpy())) + [len(curve)]:
            part = curve.iloc[start:idx]
            if len(part) > 1:
                ax.plot(
                    part["timestamp"],
                    part["soc_pct"],
                    color="#1b7f3a",
                    lw=1.7,
                    alpha=0.95,
                    label="SoC trajectory" if not label_used else None,
                )
                label_used = True
            start = idx
    ax.set_ylabel("SoC (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axes[-1].set_xlim(START_DATE - pd.Timedelta(hours=12), END_DATE + pd.Timedelta(hours=18))
    axes[-1].set_xlabel("Date")

    for ax in axes:
        ax.grid(axis="x", visible=False)

    fig.tight_layout(h_pad=0.25)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    summary, soc_curve = _daily_summary(days)
    _plot(summary, soc_curve, OUT_PATH)
    _plot(summary, soc_curve, CURVE_OUT_PATH)
    summary.to_csv(OUT_DIR / "p302_dataset_overview_through_2026-05-18_soc_curve_summary.csv", index=False, encoding="utf-8-sig")
    print(OUT_PATH)
    print(CURVE_OUT_PATH)


if __name__ == "__main__":
    main()


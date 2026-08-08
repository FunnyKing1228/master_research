from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = RAW_DIR / "figures"


@dataclass
class Span:
    kind: str
    label: str
    severity: int
    start: pd.Timestamp
    end: pd.Timestamp
    duration_min: float
    n_points: int


SPAN_STYLES = {
    "battery_gt_9v": {
        "label": "Battery > 9V",
        "color": "#f6c65b",
        "alpha": 0.12,
        "severity": 1,
    },
    "battery_gt_10v": {
        "label": "Battery > 10V",
        "color": "#f28e2b",
        "alpha": 0.16,
        "severity": 2,
    },
    "battery_gt_15v": {
        "label": "Battery > 15V",
        "color": "#d62728",
        "alpha": 0.20,
        "severity": 3,
    },
    "grid_lt_10v": {
        "label": "Grid < 10V",
        "color": "#4e79a7",
        "alpha": 0.12,
        "severity": 1,
    },
    "grid_zeroish": {
        "label": "Grid ~ 0V",
        "color": "#1f3b73",
        "alpha": 0.18,
        "severity": 2,
    },
    "standby_code_4": {
        "label": "Standby / code 4",
        "color": "#8e8e8e",
        "alpha": 0.10,
        "severity": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a multi-day deployment health window with anomaly spans."
    )
    parser.add_argument("--start-date", required=True, help="Start date, e.g. 2026-04-10")
    parser.add_argument("--end-date", required=True, help="End date, e.g. 2026-04-20")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory containing deployment_v2_*.csv and raw_data_v2_*.csv (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for figure and CSV outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--raw-resample",
        default="5min",
        help="Resample rule for raw voltage traces (default: 5min)",
    )
    return parser.parse_args()


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_window(input_dir: Path, prefix: str, start_date: str, end_date: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    pattern = f"{prefix}_2026-04-*.csv"
    for path in sorted(input_dir.glob(pattern)):
        date = path.stem.split("_")[-1]
        if not (start_date <= date <= end_date):
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["source_date"] = date
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No files found for {prefix} between {start_date} and {end_date}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def merge_mask_to_spans(
    timestamps: pd.Series,
    mask: pd.Series,
    kind: str,
    *,
    max_gap_seconds: float,
    min_duration_seconds: float,
) -> List[Span]:
    ts = pd.to_datetime(timestamps)
    mask = mask.fillna(False).astype(bool)
    active_times = ts[mask]
    if active_times.empty:
        return []

    style = SPAN_STYLES[kind]
    spans: List[Span] = []
    start = active_times.iloc[0]
    prev = active_times.iloc[0]
    n_points = 1

    for current in active_times.iloc[1:]:
        gap = (current - prev).total_seconds()
        if gap <= max_gap_seconds:
            prev = current
            n_points += 1
            continue

        duration_seconds = (prev - start).total_seconds()
        if duration_seconds >= min_duration_seconds:
            spans.append(
                Span(
                    kind=kind,
                    label=style["label"],
                    severity=style["severity"],
                    start=start,
                    end=prev,
                    duration_min=duration_seconds / 60.0,
                    n_points=n_points,
                )
            )
        start = current
        prev = current
        n_points = 1

    duration_seconds = (prev - start).total_seconds()
    if duration_seconds >= min_duration_seconds:
        spans.append(
            Span(
                kind=kind,
                label=style["label"],
                severity=style["severity"],
                start=start,
                end=prev,
                duration_min=duration_seconds / 60.0,
                n_points=n_points,
            )
        )
    return spans


def build_spans(raw: pd.DataFrame, dep: pd.DataFrame) -> List[Span]:
    raw_spans: List[Span] = []
    raw_spans.extend(
        merge_mask_to_spans(
            raw["timestamp"],
            raw["voltage_v"] > 9.0,
            "battery_gt_9v",
            max_gap_seconds=45,
            min_duration_seconds=20,
        )
    )
    raw_spans.extend(
        merge_mask_to_spans(
            raw["timestamp"],
            raw["voltage_v"] > 10.0,
            "battery_gt_10v",
            max_gap_seconds=45,
            min_duration_seconds=20,
        )
    )
    raw_spans.extend(
        merge_mask_to_spans(
            raw["timestamp"],
            raw["voltage_v"] > 15.0,
            "battery_gt_15v",
            max_gap_seconds=45,
            min_duration_seconds=20,
        )
    )
    raw_spans.extend(
        merge_mask_to_spans(
            raw["timestamp"],
            raw["grid_v"] < 10.0,
            "grid_lt_10v",
            max_gap_seconds=45,
            min_duration_seconds=20,
        )
    )
    raw_spans.extend(
        merge_mask_to_spans(
            raw["timestamp"],
            raw["grid_v"] <= 0.1,
            "grid_zeroish",
            max_gap_seconds=45,
            min_duration_seconds=20,
        )
    )

    dep_spans = merge_mask_to_spans(
        dep["timestamp"],
        dep["situation_code"] == 4,
        "standby_code_4",
        max_gap_seconds=20 * 60,
        min_duration_seconds=15 * 60,
    )
    return sorted(raw_spans + dep_spans, key=lambda span: (span.start, span.severity, span.kind))


def prep_data(dep: pd.DataFrame, raw: pd.DataFrame, raw_resample: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _to_numeric(
        dep,
        [
            "soc",
            "action_power_kw",
            "pv_kw",
            "load_kw",
            "situation_code",
            "batt_v_mean",
            "price",
        ],
    )
    _to_numeric(
        raw,
        [
            "voltage_v",
            "grid_v",
            "load_v",
            "solar_p_mw",
            "load_p_mw",
            "grid_p_mw",
            "soc_calc",
        ],
    )

    dep["soc_pct"] = dep["soc"] * 100.0
    dep["load_w"] = dep["load_kw"] * 1000.0
    dep["pv_w"] = dep["pv_kw"] * 1000.0
    dep["action_w"] = dep["action_power_kw"] * 1000.0

    raw_resampled = (
        raw.set_index("timestamp")
        .resample(raw_resample)
        .median(numeric_only=True)
        .reset_index()
    )
    raw_resampled["battery_v"] = raw_resampled["voltage_v"]
    raw_resampled["grid_v"] = raw_resampled["grid_v"]
    return dep, raw_resampled


def configure_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, color="#d9d9d9", alpha=0.75, linestyle="--", linewidth=0.8)


def create_plot(
    dep: pd.DataFrame,
    raw_plot: pd.DataFrame,
    spans: List[Span],
    start_date: str,
    end_date: str,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft JhengHei", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 13,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )
    ax_power, ax_soc, ax_voltage = axes

    ax_power.plot(
        dep["timestamp"],
        dep["load_w"],
        color="#111111",
        linewidth=1.9,
        linestyle="--",
        label="Load",
    )
    ax_power.plot(
        dep["timestamp"],
        dep["pv_w"],
        color="#d4a000",
        linewidth=1.9,
        label="PV",
    )
    ax_power.plot(
        dep["timestamp"],
        dep["action_w"],
        color="#2e8b57",
        linewidth=1.7,
        label="Battery Action",
    )
    ax_power.axhline(0.0, color="black", linewidth=0.8)
    ax_power.set_ylabel("Power (W)")
    ax_power.set_title("10-Day Deployment Overview: Sim-to-Real Challenges (2026-04-10 ~ 04-20)")
    ax_power.legend(loc="upper left", ncol=3, frameon=False)

    ax_soc.plot(dep["timestamp"], dep["soc_pct"], color="#3f3c8f", linewidth=2.0, label="SoC")
    ax_soc.axhline(90.0, color="#cc2f2f", linestyle="--", linewidth=1.2, label="Upper limit (0.9)")
    ax_soc.axhline(10.0, color="#cc2f2f", linestyle="--", linewidth=1.2, label="Lower limit (0.1)")
    ax_soc.set_ylabel("SoC (%)")
    ax_soc.set_ylim(-2, 102)
    ax_soc.legend(loc="upper left", ncol=3, frameon=False)

    ax_voltage.plot(
        raw_plot["timestamp"],
        raw_plot["battery_v"],
        color="#ff7f0e",
        linewidth=1.6,
        label="Battery V",
    )
    ax_voltage.plot(
        raw_plot["timestamp"],
        raw_plot["grid_v"],
        color="#4e79a7",
        linewidth=1.6,
        label="Grid V",
    )
    ax_voltage.axhline(10.0, color="#cc2f2f", linestyle="--", linewidth=1.2, label="10V threshold")
    ax_voltage.set_ylabel("Voltage (V)")
    ax_voltage.set_xlabel("Date")
    ax_voltage.legend(loc="upper left", ncol=3, frameon=False)

    for ax in axes:
        configure_time_axis(ax)
        ax.margins(x=0.01)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def spans_to_frame(spans: List[Span]) -> pd.DataFrame:
    rows = [
        {
            "kind": span.kind,
            "label": span.label,
            "severity": span.severity,
            "start": span.start,
            "end": span.end,
            "duration_min": round(span.duration_min, 2),
            "n_points": span.n_points,
        }
        for span in spans
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dep = load_window(args.input_dir, "deployment_v2", args.start_date, args.end_date)
    raw = load_window(args.input_dir, "raw_data_v2", args.start_date, args.end_date)
    dep, raw_plot = prep_data(dep, raw, args.raw_resample)
    spans = build_spans(raw, dep)
    spans_df = spans_to_frame(spans)

    stem = f"deployment_overview_report_{args.start_date}_to_{args.end_date}"
    figure_path = args.output_dir / f"{stem}.png"
    csv_path = args.output_dir / f"{stem}_anomaly_spans.csv"
    pdf_path = args.output_dir / f"{stem}.pdf"

    create_plot(dep, raw_plot, spans, args.start_date, args.end_date, figure_path)
    create_plot(dep, raw_plot, spans, args.start_date, args.end_date, pdf_path)
    spans_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved figure: {figure_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved anomaly CSV: {csv_path}")
    if not spans_df.empty:
        top = spans_df.sort_values(["severity", "duration_min"], ascending=[False, False]).head(12)
        print("\nTop anomaly spans:")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()

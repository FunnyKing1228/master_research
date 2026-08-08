"""
Export clean voltage-current segments for the standalone SOH_Predictor.

The deployed microgrid logs are not standardized full laboratory cycles. This
script therefore exports conservative partial charge segments as offline inputs
for trend checks, not as ground-truth SoH labels.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = ROOT / "data" / "raw"
DEFAULT_OUT_DIR = ROOT / "data" / "soh_segments"


@dataclass
class Segment:
    direction: str
    rows: pd.DataFrame
    source_file: str
    index: int


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read deployment CSVs whose schema may have evolved across dates."""
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return pd.DataFrame()
        header = rows[0]
        width = len(header)
        fixed = []
        for row in rows[1:]:
            if len(row) < width:
                row = row + [""] * (width - len(row))
            elif len(row) > width:
                row = row[:width]
            fixed.append(row)
        return pd.DataFrame(fixed, columns=header)


def _load_raw_logs(log_dir: Path, start: str | None, end: str | None) -> pd.DataFrame:
    frames = []
    for path in sorted(log_dir.glob("raw_data_v2_*.csv")):
        date_part = path.stem.replace("raw_data_v2_", "")
        if start and date_part < start:
            continue
        if end and date_part > end:
            continue
        df = _read_csv_flexible(path)
        if df.empty:
            continue
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df.columns:
        raise ValueError("raw log missing timestamp column")

    numeric_cols = [
        "voltage_v",
        "current_ma",
        "current_raw_ma",
        "soc_calc",
        "charge_voltage_v",
        "situation_code",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "voltage_v", "current_ma"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _iter_segments(
    df: pd.DataFrame,
    *,
    direction: str,
    current_threshold_ma: float,
    min_samples: int,
    max_gap_sec: float,
    soc_min: float,
    soc_max: float,
    require_window_coverage: bool,
    soc_start_max: float,
    soc_end_min: float,
    min_soc_span: float,
    voltage_min_v: float,
) -> Iterable[Segment]:
    if df.empty:
        return

    sign = 1 if direction == "charge" else -1
    current = df["current_ma"].astype(float)
    mask = (
        (df["voltage_v"].astype(float) >= voltage_min_v)
        & (sign * current >= current_threshold_ma)
    )
    if "soc_calc" in df.columns:
        soc = df["soc_calc"].astype(float)
        mask &= soc.between(soc_min, soc_max)

    candidates = df.loc[mask].copy()
    if candidates.empty:
        return

    candidates["gap_sec"] = candidates["timestamp"].diff().dt.total_seconds().fillna(0.0)
    group_id = (candidates["gap_sec"] > max_gap_sec).cumsum()

    seg_idx = 0
    for _, seg in candidates.groupby(group_id):
        seg = seg.copy()
        if len(seg) < min_samples:
            continue
        if "soc_calc" in seg.columns:
            seg_soc_min = float(seg["soc_calc"].min())
            seg_soc_max = float(seg["soc_calc"].max())
            seg_soc_span = seg_soc_max - seg_soc_min
            if seg_soc_span < min_soc_span:
                continue
            if require_window_coverage and (
                seg_soc_min > soc_start_max or seg_soc_max < soc_end_min
            ):
                continue
        seg_idx += 1
        yield Segment(
            direction=direction,
            rows=seg,
            source_file=str(seg["source_file"].iloc[0]),
            index=seg_idx,
        )


def _write_segment(segment: Segment, cycles_dir: Path) -> dict:
    rows = segment.rows.copy()
    t0 = rows["timestamp"].iloc[0]
    out = pd.DataFrame(
        {
            "time": (rows["timestamp"] - t0).dt.total_seconds(),
            "voltage": rows["voltage_v"].astype(float),
            "current": rows["current_ma"].astype(float) / 1000.0,
        }
    )
    start_tag = pd.Timestamp(t0).strftime("%Y%m%d_%H%M%S")
    name = f"cycle_soh_{segment.direction}_{start_tag}_{segment.index:03d}.csv"
    path = cycles_dir / name
    out.to_csv(path, index=False)

    soc_min = float(rows["soc_calc"].min()) if "soc_calc" in rows.columns else np.nan
    soc_max = float(rows["soc_calc"].max()) if "soc_calc" in rows.columns else np.nan
    return {
        "file": name,
        "direction": segment.direction,
        "source_file": segment.source_file,
        "start": rows["timestamp"].iloc[0],
        "end": rows["timestamp"].iloc[-1],
        "samples": len(rows),
        "duration_sec": float(out["time"].iloc[-1]) if len(out) else 0.0,
        "voltage_min": float(rows["voltage_v"].min()),
        "voltage_max": float(rows["voltage_v"].max()),
        "current_ma_mean": float(rows["current_ma"].mean()),
        "soc_min": soc_min,
        "soc_max": soc_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export clean partial-cycle segments for SOH_Predictor."
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--soc-min", type=float, default=0.25)
    parser.add_argument("--soc-max", type=float, default=0.75)
    parser.add_argument(
        "--require-window-coverage",
        action="store_true",
        help="Require each exported segment to cover most of the requested SoC window.",
    )
    parser.add_argument(
        "--soc-start-max",
        type=float,
        default=0.30,
        help="With --require-window-coverage, segment min SoC must be <= this value.",
    )
    parser.add_argument(
        "--soc-end-min",
        type=float,
        default=0.70,
        help="With --require-window-coverage, segment max SoC must be >= this value.",
    )
    parser.add_argument(
        "--min-soc-span",
        type=float,
        default=0.0,
        help="Minimum SoC span required for each exported segment.",
    )
    parser.add_argument("--voltage-min-v", type=float, default=4.2)
    parser.add_argument("--current-threshold-ma", type=float, default=50.0)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--max-gap-sec", type=float, default=45.0)
    parser.add_argument("--include-discharge", action="store_true")
    args = parser.parse_args()

    df = _load_raw_logs(args.log_dir, args.start_date, args.end_date)
    if df.empty:
        raise SystemExit(f"No raw logs found in {args.log_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cycles_dir = args.out_dir / "cycles"
    cycles_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    directions = ["charge"]
    if args.include_discharge:
        directions.append("discharge")
    for direction in directions:
        for segment in _iter_segments(
            df,
            direction=direction,
            current_threshold_ma=args.current_threshold_ma,
            min_samples=args.min_samples,
            max_gap_sec=args.max_gap_sec,
            soc_min=args.soc_min,
            soc_max=args.soc_max,
            require_window_coverage=args.require_window_coverage,
            soc_start_max=args.soc_start_max,
            soc_end_min=args.soc_end_min,
            min_soc_span=args.min_soc_span,
            voltage_min_v=args.voltage_min_v,
        ):
            summaries.append(_write_segment(segment, cycles_dir))

    summary_df = pd.DataFrame(summaries)
    summary_path = args.out_dir / "soh_segment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Exported {len(summary_df)} segments to {cycles_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()


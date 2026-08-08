"""Convert high-frequency raw microgrid logs into 15-minute training rows.

Expected input columns are flexible. The script looks for timestamp, PV, load,
SoC, SoH, and flow-rate aliases, then writes a CSV compatible with
``MicrogridEnvironment(dataset_csv_path=...)``.

Example:
    python data/scripts/preprocessing/preprocess_raw_to_15min.py \
        --input data/raw/solar_20250101.csv \
        --output data/processed/solar_20250101_15min.csv \
        --window_min 15 \
        --load_kw 5.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_MIN = 15


def _first_available_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    timestamp_col = _first_available_column(df, ["timestamp", "time", "datetime", "Time", "Timestamp"])
    if timestamp_col is None:
        raise ValueError("Input CSV must contain a timestamp/time/datetime column.")
    df["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)
    df = df[df["timestamp"].notna()].copy()

    pv_candidates = [
        ("mppt_p_kw", 1.0),
        ("mppt_p_mw", 1e-6),
        ("solar_p_kw", 1.0),
        ("pv_kw", 1.0),
        ("Solar", 1.0),
        ("solar_p_mw", 1e-6),
    ]
    for col, scale in pv_candidates:
        if col in df.columns:
            df["pv_kw"] = df[col].astype(float).fillna(0.0) * scale
            break
    else:
        print("[WARN] No PV column found; using 0 kW.")
        df["pv_kw"] = 0.0

    load_col = _first_available_column(df, ["load_p_kw", "load_kw", "Consumption", "load"])
    df["load_kw"] = df[load_col].astype(float).fillna(0.0) if load_col else np.nan

    df["soc"] = df["battery_soc"].astype(float) if "battery_soc" in df.columns else np.nan
    df["soh"] = df["battery_soh"].astype(float) if "battery_soh" in df.columns else 1.0
    df["flow_lpm"] = df["flow_rate_lpm"].astype(float) if "flow_rate_lpm" in df.columns else 0.0

    return df.sort_values("timestamp").reset_index(drop=True)


def aggregate_to_window(
    df: pd.DataFrame,
    window_min: int,
    fixed_load_kw: float | None = None,
) -> pd.DataFrame:
    """Aggregate raw samples into fixed-width windows."""
    df = df.set_index("timestamp")
    rule = f"{window_min}min"

    if df["load_kw"].isna().all():
        if fixed_load_kw is None:
            raise ValueError("Input CSV has no load column. Provide --load_kw <kW>.")
        df["load_kw"] = float(fixed_load_kw)
    elif fixed_load_kw is not None:
        df["load_kw"] = float(fixed_load_kw)

    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    dt_sec = float(diffs.median()) if len(diffs) > 0 else 10.0
    dt_h = dt_sec / 3600.0

    agg = pd.DataFrame()
    agg["pv_mean"] = df["pv_kw"].resample(rule).mean()
    agg["pv_std"] = df["pv_kw"].resample(rule).std().fillna(0.0)
    agg["pv_max"] = df["pv_kw"].resample(rule).max()
    agg["load_mean"] = df["load_kw"].resample(rule).mean()
    agg["load_std"] = df["load_kw"].resample(rule).std().fillna(0.0)
    agg["load_max"] = df["load_kw"].resample(rule).max()
    agg["soc_mean"] = df["soc"].resample(rule).mean()
    agg["soc_end"] = df["soc"].resample(rule).last()
    agg["soh_mean"] = df["soh"].resample(rule).mean().fillna(1.0)
    agg["flow_rate_mean"] = df["flow_lpm"].resample(rule).mean().fillna(0.0)

    counts = df["pv_kw"].resample(rule).count()
    agg["energy_pv_kwh"] = agg["pv_mean"] * counts * dt_h
    agg["energy_load_kwh"] = agg["load_mean"] * counts * dt_h

    agg = agg.reset_index()
    agg["hour"] = agg["timestamp"].dt.hour
    agg["day_of_week"] = agg["timestamp"].dt.dayofweek
    agg["minute"] = agg["timestamp"].dt.minute

    base_price = 0.15
    agg["price"] = np.where(
        (agg["hour"] >= 8) & (agg["hour"] <= 18),
        base_price * 1.2,
        base_price * 0.8,
    )

    # Column aliases used by older environment-loading code.
    agg["Solar"] = agg["pv_mean"]
    agg["Consumption"] = agg["load_mean"]

    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw microgrid CSV logs into 15-minute rows.")
    parser.add_argument("--input", required=True, help="Input raw CSV path.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to data/processed.")
    parser.add_argument("--window_min", type=int, default=WINDOW_MIN, help="Aggregation window in minutes.")
    parser.add_argument("--load_kw", type=float, default=None, help="Fixed load in kW if no load column exists.")
    args = parser.parse_args()

    if args.output is None:
        base = Path(args.input).stem
        out_dir = Path(args.input).parent.parent / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"{base}_{args.window_min}min.csv")

    print(f"[preprocess] Input : {args.input}")
    print(f"[preprocess] Output: {args.output}")
    print(f"[preprocess] Window: {args.window_min} min")

    df_raw = load_raw_csv(args.input)
    print(
        f"[preprocess] Loaded {len(df_raw)} raw rows, "
        f"time range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}"
    )

    df_out = aggregate_to_window(df_raw, args.window_min, fixed_load_kw=args.load_kw)
    df_out.to_csv(args.output, index=False)

    print(f"[preprocess] Done. Wrote {len(df_out)} rows to {args.output}")
    print(f"\n  PV mean range  : {df_out['pv_mean'].min():.2f} to {df_out['pv_mean'].max():.2f} kW")
    print(f"  Load mean range: {df_out['load_mean'].min():.2f} to {df_out['load_mean'].max():.2f} kW")
    if df_out["soc_mean"].notna().any():
        print(f"  SoC range      : {df_out['soc_mean'].min():.3f} to {df_out['soc_mean'].max():.3f}")
    print(f"  SoH mean       : {df_out['soh_mean'].mean():.4f}")


if __name__ == "__main__":
    main()


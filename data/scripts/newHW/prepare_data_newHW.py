"""Prepare the provisional newHW LFP dataset without touching P302 data.

The source CSV has a malformed quoted header, an unusable Load_W channel,
an uncalibrated SoC channel, reversed ACS712_PV polarity, and invalid battery
voltage values.  Every non-measured substitution is marked TODO(newHW) here
and in docs/handover/newHW_pending_data.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "Date_Hour",
    "ACS712_Batt",
    "ACS712_PV",
    "MPPT_V_batt",
    "MPPT_I_Batt",
    "MPPT_W_PV",
    "SoC",
    "Load_W",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: Path) -> pd.DataFrame:
    # The whole header line is quoted as one CSV field.  Supplying explicit
    # names after skiprows=1 avoids silently parsing the file as one column.
    df = pd.read_csv(path, skiprows=1, names=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["Date_Hour"], errors="coerce")
    for column in COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    duplicate_rows = int(df["timestamp"].duplicated(keep=False).sum())
    if duplicate_rows:
        numeric = [column for column in COLUMNS[1:] if column in df]
        df = df.groupby("timestamp", as_index=False)[numeric].mean()
    return df.reset_index(drop=True)


def reconstruct_soc(
    df: pd.DataFrame,
    *,
    capacity_kwh: float,
    initial_soc: float,
    roundtrip_efficiency: float,
) -> pd.DataFrame:
    result = df.copy()
    voltage_valid = result["MPPT_V_batt"].between(20.0, 31.0)
    result["battery_voltage_valid"] = voltage_valid.astype(int)
    result["battery_voltage_v"] = result["MPPT_V_batt"].where(voltage_valid)

    # TODO(newHW): pack voltage is missing for long nighttime intervals.
    # Time interpolation is provisional and must be replaced after the voltage
    # channel is repaired.  The validity flag remains available for auditing.
    indexed_voltage = result.set_index("timestamp")["battery_voltage_v"]
    indexed_voltage = indexed_voltage.interpolate(method="time", limit_direction="both")
    result["battery_voltage_estimated_v"] = indexed_voltage.to_numpy()

    result["pv_current_corrected_a"] = -result["ACS712_PV"]
    result["battery_current_a"] = result["ACS712_Batt"]
    result["battery_power_kw"] = (
        result["battery_current_a"] * result["battery_voltage_estimated_v"] / 1000.0
    )

    dt_hours = result["timestamp"].diff().dt.total_seconds().fillna(0.0) / 3600.0
    result["source_gap_seconds"] = result["timestamp"].diff().dt.total_seconds().fillna(0.0)
    result["gap_over_60s"] = (result["source_gap_seconds"] > 60.0).astype(int)

    # Trapezoidal integration across observed timestamps.  TODO(newHW): this
    # assumes linear behavior through gaps (up to 163 s in the supplied file).
    previous_power = result["battery_power_kw"].shift(1).fillna(result["battery_power_kw"])
    interval_power = 0.5 * (previous_power + result["battery_power_kw"])
    charge_efficiency = float(np.sqrt(roundtrip_efficiency))
    discharge_efficiency = max(charge_efficiency, 1e-9)
    battery_energy_delta = np.where(
        interval_power >= 0.0,
        interval_power * dt_hours * charge_efficiency,
        interval_power * dt_hours / discharge_efficiency,
    )
    result["battery_energy_delta_kwh"] = battery_energy_delta
    result["soc_reconstructed_unclipped"] = (
        float(initial_soc) + np.cumsum(battery_energy_delta) / float(capacity_kwh)
    )
    result["soc_reconstructed"] = result["soc_reconstructed_unclipped"].clip(0.0, 1.0)
    result["soc_reconstruction_clipped"] = (
        result["soc_reconstructed_unclipped"].ne(result["soc_reconstructed"])
    ).astype(int)
    return result


def resample_15min(df: pd.DataFrame, *, fixed_load_w: float) -> pd.DataFrame:
    indexed = df.set_index("timestamp").sort_index()
    first_full = indexed.index.min().ceil("15min")
    last_full = indexed.index.max().floor("15min")
    indexed = indexed[(indexed.index >= first_full) & (indexed.index < last_full)]
    if indexed.empty:
        raise ValueError("No complete 15-minute interval exists in the source data")

    grouped = indexed.resample("15min")
    out = pd.DataFrame(index=grouped.size().index)
    out["Solar"] = grouped["MPPT_W_PV"].mean().clip(lower=0.0) / 1000.0
    # TODO(newHW): Load_W is invalid.  28.2 W comes from the supplied
    # ACS712 regression/night-state analysis, not a working load sensor.
    out["Consumption"] = float(fixed_load_w) / 1000.0
    out["battery_voltage_v"] = grouped["battery_voltage_estimated_v"].mean()
    out["battery_voltage_valid_fraction"] = grouped["battery_voltage_valid"].mean()
    out["battery_current_a"] = grouped["battery_current_a"].mean()
    out["pv_current_corrected_a"] = grouped["pv_current_corrected_a"].mean()
    out["soc_reconstructed"] = grouped["soc_reconstructed"].last()
    out["soc_reconstructed_unclipped"] = grouped["soc_reconstructed_unclipped"].last()
    out["source_samples"] = grouped.size().astype(int)
    out["max_source_gap_seconds"] = grouped["source_gap_seconds"].max().fillna(0.0)
    out["gap_over_60s_count"] = grouped["gap_over_60s"].sum().fillna(0).astype(int)
    out["soc_reconstruction_clipped"] = (
        grouped["soc_reconstruction_clipped"].max().fillna(0).astype(int)
    )
    # TODO(newHW): 1 W is only a provisional data-label threshold.  The
    # environment consumes continuous PV power instead of this boolean.
    out["pv_available"] = (out["Solar"] > 0.001).astype(int)
    out["pv_support_ratio"] = (
        out["Solar"] / out["Consumption"].clip(lower=1e-9)
    ).clip(0.0, 5.0)
    out["price"] = 0.0
    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out.index.name = "timestamp"
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare provisional newHW 15-minute data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    # TODO(newHW): capacity, initial SoC, load and efficiency all require
    # hardware confirmation; defaults reproduce this migration smoke only.
    parser.add_argument("--capacity-kwh", type=float, default=0.20)
    parser.add_argument("--initial-soc", type=float, default=1.0)
    parser.add_argument("--fixed-load-w", type=float, default=28.2)
    parser.add_argument("--roundtrip-efficiency", type=float, default=0.95)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Output must not overwrite the immutable source CSV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    source = read_source(args.input)
    source_with_soc = reconstruct_soc(
        source,
        capacity_kwh=args.capacity_kwh,
        initial_soc=args.initial_soc,
        roundtrip_efficiency=args.roundtrip_efficiency,
    )
    processed = resample_15min(source_with_soc, fixed_load_w=args.fixed_load_w)
    processed.to_csv(args.output, index=False)

    gaps = source_with_soc["source_gap_seconds"]
    first_night = source_with_soc[
        source_with_soc["timestamp"].between(
            pd.Timestamp("2026-08-14 17:45:00"),
            pd.Timestamp("2026-08-15 00:51:12"),
        )
        & source_with_soc["battery_voltage_valid"].eq(1)
    ]
    first_night_energy_kwh = float(
        np.trapezoid(
            first_night["battery_power_kw"].to_numpy(),
            first_night["timestamp"].astype("int64").to_numpy() / 1e9 / 3600.0,
        )
    )
    summary = {
        "status": "PROVISIONAL_IN_SAMPLE_ONLY",
        "todo_marker": "TODO(newHW)",
        "source_path": str(args.input),
        "source_sha256": sha256(args.input),
        "source_rows": int(len(source)),
        "source_start": str(source["timestamp"].min()),
        "source_end": str(source["timestamp"].max()),
        "source_duplicate_timestamp_rows_before_aggregation": int(
            pd.read_csv(args.input, skiprows=1, names=COLUMNS)["Date_Hour"].duplicated(
                keep=False
            ).sum()
        ),
        "gaps_over_60s": int((gaps > 60.0).sum()),
        "max_gap_seconds": float(gaps.max()),
        "invalid_voltage_rows": int((source_with_soc["battery_voltage_valid"] == 0).sum()),
        "raw_load_nonzero_rows": int((source["Load_W"].fillna(0.0) != 0.0).sum()),
        "raw_soc_zero_rows": int((source["SoC"].fillna(0.0) == 0.0).sum()),
        "processed_rows": int(len(processed)),
        "processed_start": str(processed["timestamp"].min()),
        "processed_end": str(processed["timestamp"].max()),
        "fixed_load_w_assumption": float(args.fixed_load_w),
        "capacity_kwh_assumption": float(args.capacity_kwh),
        "initial_soc_assumption": float(args.initial_soc),
        "roundtrip_efficiency_assumption": float(args.roundtrip_efficiency),
        "first_night_independent_discharge_wh": -first_night_energy_kwh * 1000.0,
        "first_night_last_valid_voltage_v": float(
            first_night["MPPT_V_batt"].iloc[-1]
        ),
        "first_night_last_valid_timestamp": str(first_night["timestamp"].iloc[-1]),
        "soc_unclipped_min": float(source_with_soc["soc_reconstructed_unclipped"].min()),
        "soc_unclipped_max": float(source_with_soc["soc_reconstructed_unclipped"].max()),
        "soc_clipped_rows": int(source_with_soc["soc_reconstruction_clipped"].sum()),
        "notes": [
            "Load_W was replaced by the supplied 28.2 W regression-derived baseline.",
            "ACS712_PV polarity was reversed.",
            "Battery voltage outside 20-31 V was masked then provisionally time-interpolated.",
            "SoC was reconstructed from ACS712_Batt energy integration.",
            "Initial SoC is unknown and provisionally anchored at 1.0.",
            "First-night integration gives about 199.07 Wh and ends at 25.42 V, consistent with the supplied 198.5 Wh / 25.43 V diagnosis.",
            "The 47-hour dataset cannot support train/validation/test separation.",
        ],
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

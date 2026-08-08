from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BASE_DATASET = PROCESSED_DIR / "training_v16_hybrid50.csv"
OUTPUT = PROCESSED_DIR / "training_v17_0504_curated.csv"

# Keep new post-deployment data that has useful PV/load behavior.
# Exclude 2026-04-30 because the system froze around 11:00 and repeated stale sensor values.
CURATED_DEPLOYMENT_DATES = (
    "2026-04-24",
    "2026-04-25",
    "2026-04-26",
    "2026-04-27",
    "2026-04-28",
    "2026-04-29",
)


def get_tou_price(timestamp: pd.Timestamp) -> float:
    hour = int(timestamp.hour)
    dow = int(timestamp.dayofweek)
    if dow >= 5:
        return 2.06
    if 0 <= hour < 9:
        return 2.06
    if 9 <= hour < 16:
        return 4.69
    if 16 <= hour < 22:
        return 7.13
    return 4.69


def load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATASET, parse_dates=["timestamp"])
    return df[["timestamp", "Solar", "PV_bool", "Consumption", "price", "hour", "day_of_week"]].copy()


def load_curated_deployment_days() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date in CURATED_DEPLOYMENT_DATES:
        path = RAW_DIR / f"deployment_v2_{date}.csv"
        df = pd.read_csv(path, usecols=["timestamp", "pv_kw", "load_kw"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["pv_kw"] = pd.to_numeric(df["pv_kw"], errors="coerce")
        df["load_kw"] = pd.to_numeric(df["load_kw"], errors="coerce")
        df = df.dropna(subset=["timestamp", "pv_kw", "load_kw"])
        frames.append(df)

    deployment = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    deployment = deployment.set_index("timestamp").resample("15min").median(numeric_only=True).dropna().reset_index()

    out = pd.DataFrame()
    out["timestamp"] = deployment["timestamp"]
    out["Solar"] = deployment["pv_kw"].clip(lower=0.0)
    out["Consumption"] = deployment["load_kw"].clip(lower=0.0001)
    ratio = out["Solar"].to_numpy(dtype=float) / np.clip(out["Consumption"].to_numpy(dtype=float), 1e-9, None)
    out["PV_bool"] = (ratio >= 0.8).astype(float)
    out["hour"] = out["timestamp"].dt.hour.astype(int)
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype(int)
    out["price"] = [get_tou_price(ts) for ts in out["timestamp"]]
    return out[["timestamp", "Solar", "PV_bool", "Consumption", "price", "hour", "day_of_week"]]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_dataset()
    curated = load_curated_deployment_days()
    combined = pd.concat([base, curated], ignore_index=True).sort_values("timestamp")
    combined.to_csv(OUTPUT, index=False)

    ratio = combined["Solar"].to_numpy(dtype=float) / np.clip(combined["Consumption"].to_numpy(dtype=float), 1e-9, None)
    curated_ratio = curated["Solar"].to_numpy(dtype=float) / np.clip(curated["Consumption"].to_numpy(dtype=float), 1e-9, None)
    print(f"Saved curated training dataset: {OUTPUT}")
    print(f"base_rows={len(base)} curated_rows={len(curated)} combined_rows={len(combined)}")
    print(f"curated_dates={', '.join(CURATED_DEPLOYMENT_DATES)}")
    print("excluded_dates=2026-04-30 (sensor freeze / stale values)")
    print(f"combined_steps_pv_ratio_ge_0.8={(ratio >= 0.8).sum()} fraction={(ratio >= 0.8).mean():.4f}")
    print(f"curated_steps_pv_ratio_ge_0.8={(curated_ratio >= 0.8).sum()} fraction={(curated_ratio >= 0.8).mean():.4f}")
    print(
        curated.assign(date=curated["timestamp"].dt.strftime("%Y-%m-%d"))
        .groupby("date")
        .agg(
            rows=("Solar", "size"),
            pv_max_kw=("Solar", "max"),
            load_max_kw=("Consumption", "max"),
            pv_ratio_max=("Solar", lambda s: float((s / curated.loc[s.index, "Consumption"]).max())),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()


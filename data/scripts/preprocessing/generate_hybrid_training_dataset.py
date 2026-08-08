from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE_CSV = ROOT / "data" / "processed" / "training_v16.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "training_v16_hybrid50.csv"

SUNNY_TEMPLATE_DATES = ("2026-04-09", "2026-04-10")


def get_tou_price(hour: int, day_of_week: int) -> float:
    if int(day_of_week) >= 5:
        return 2.06
    if 0 <= hour < 9:
        return 2.06
    if 9 <= hour < 16:
        return 4.69
    if 16 <= hour < 22:
        return 7.13
    return 4.69


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hybrid microgrid training dataset.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-days", type=int, default=20, help="Total synthetic days")
    parser.add_argument("--sunny-ratio", type=float, default=0.5, help="Target fraction of sunny days")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-05-04 00:00:00",
        help="Start timestamp for the generated sequential dataset",
    )
    return parser.parse_args()


def load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["Solar"] = df["Solar"].astype(float)
    df["Consumption"] = df["Consumption"].astype(float)
    return df


def build_daily_templates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    templates: dict[str, pd.DataFrame] = {}
    for date_key, group in df.groupby("date"):
        group = group.sort_values("timestamp").reset_index(drop=True).copy()
        if len(group) == 96:
            templates[date_key] = group
    return templates


def sample_day(
    template: pd.DataFrame,
    rng: np.random.Generator,
    is_sunny: bool,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    out = template.copy()

    if is_sunny:
        pv_scale = float(rng.uniform(0.92, 1.08))
        load_scale = float(rng.uniform(0.96, 1.04))
    else:
        pv_scale = float(rng.uniform(0.85, 1.05))
        load_scale = float(rng.uniform(0.96, 1.05))

    solar_noise = rng.normal(loc=0.0, scale=0.00012 if is_sunny else 0.00008, size=len(out))
    load_noise = rng.normal(loc=0.0, scale=0.00008, size=len(out))

    out["Solar"] = np.clip(out["Solar"].to_numpy(dtype=float) * pv_scale + solar_noise, 0.0, None)
    out["Consumption"] = np.clip(out["Consumption"].to_numpy(dtype=float) * load_scale + load_noise, 0.0001, None)

    time_offsets = pd.to_timedelta(np.arange(len(out)) * 15, unit="min")
    out["timestamp"] = target_date + time_offsets
    out["hour"] = out["timestamp"].dt.hour.astype(int)
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype(int)
    out["price"] = [get_tou_price(hour, dow) for hour, dow in zip(out["hour"], out["day_of_week"])]

    pv_ratio = out["Solar"].to_numpy(dtype=float) / np.clip(out["Consumption"].to_numpy(dtype=float), 1e-9, None)
    out["PV_bool"] = (pv_ratio >= 0.8).astype(float)
    out["source_template_date"] = template["date"].iloc[0]
    out["template_kind"] = "sunny" if is_sunny else "cloudy"
    out["generated_date"] = target_date.strftime("%Y-%m-%d")
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = load_source()
    templates = build_daily_templates(df)

    sunny_templates = [templates[d] for d in SUNNY_TEMPLATE_DATES if d in templates]
    cloudy_templates = [tpl for d, tpl in templates.items() if d not in SUNNY_TEMPLATE_DATES]
    if not sunny_templates:
        raise RuntimeError("No sunny templates found.")
    if not cloudy_templates:
        raise RuntimeError("No cloudy templates found.")

    total_days = int(args.total_days)
    sunny_days = int(round(total_days * float(args.sunny_ratio)))
    sunny_days = max(1, min(total_days - 1, sunny_days))
    cloudy_days = total_days - sunny_days

    day_labels = ["sunny"] * sunny_days + ["cloudy"] * cloudy_days
    rng.shuffle(day_labels)

    start_date = pd.Timestamp(args.start_date)
    generated_days: list[pd.DataFrame] = []
    for idx, label in enumerate(day_labels):
        target_date = start_date + pd.Timedelta(days=idx)
        if label == "sunny":
            template = sunny_templates[int(rng.integers(0, len(sunny_templates)))]
            generated_days.append(sample_day(template, rng, True, target_date))
        else:
            template = cloudy_templates[int(rng.integers(0, len(cloudy_templates)))]
            generated_days.append(sample_day(template, rng, False, target_date))

    out_df = pd.concat(generated_days, ignore_index=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_cols = ["timestamp", "Solar", "PV_bool", "Consumption", "price", "hour", "day_of_week"]
    out_df[save_cols].to_csv(output_path, index=False)

    pv_ratio = out_df["Solar"].to_numpy(dtype=float) / np.clip(out_df["Consumption"].to_numpy(dtype=float), 1e-9, None)
    daily = (
        out_df.assign(date=out_df["timestamp"].dt.strftime("%Y-%m-%d"), pv_ratio=pv_ratio)
        .groupby("date")
        .agg(
            max_ratio=("pv_ratio", "max"),
            steps_ge_080=("pv_ratio", lambda s: int((s >= 0.8).sum())),
            steps_ge_090=("pv_ratio", lambda s: int((s >= 0.9).sum())),
            steps_ge_100=("pv_ratio", lambda s: int((s >= 1.0).sum())),
        )
    )

    sunny_like_days = int((daily["max_ratio"] >= 0.8).sum())
    print(f"Saved hybrid dataset to: {output_path}")
    print(f"rows={len(out_df)} days={daily.shape[0]}")
    print(f"target_sunny_days={sunny_days} actual_days_max_ratio_ge_0.8={sunny_like_days}")
    print(f"steps_ge_0.8={(pv_ratio >= 0.8).sum()} fraction_ge_0.8={(pv_ratio >= 0.8).mean():.6f}")
    print(f"steps_ge_0.9={(pv_ratio >= 0.9).sum()} fraction_ge_0.9={(pv_ratio >= 0.9).mean():.6f}")
    print(f"steps_ge_1.0={(pv_ratio >= 1.0).sum()} fraction_ge_1.0={(pv_ratio >= 1.0).mean():.6f}")
    print(daily.to_string())


if __name__ == "__main__":
    main()

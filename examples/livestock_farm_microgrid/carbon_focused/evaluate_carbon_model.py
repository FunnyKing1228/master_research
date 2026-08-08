from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PARENT = THIS_DIR.parents[0]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from run_month_simulation import DEFAULT_CARBON_FACTOR_KG_PER_KWH, plot_daily_weekly_impact, plot_month, summarize  # noqa: E402
from train_sac_carbon import rollout_month  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved carbon-focused SAC model.")
    parser.add_argument("--model-path", default=str(THIS_DIR / "results" / "sac_carbon_focused_livestock.zip"))
    parser.add_argument("--start-date", default="2026-05-01")
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--initial-soc", type=float, default=0.50)
    parser.add_argument("--load-daily-energy-kwh", type=float, default=905.8064516129032)
    parser.add_argument("--load-peak-kw", type=float, default=104.0)
    parser.add_argument("--seed", type=int, default=10321)
    parser.add_argument("--use-weather-api", action="store_true")
    parser.add_argument("--carbon-factor", type=float, default=DEFAULT_CARBON_FACTOR_KG_PER_KWH)
    parser.add_argument("--output-dir", default=str(THIS_DIR / "results"))
    return parser.parse_args()


def main() -> None:
    from stable_baselines3 import SAC

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SAC.load(args.model_path)
    rollout = rollout_month(
        model=model,
        start_date=args.start_date,
        days=args.eval_days,
        initial_soc=args.initial_soc,
        seed=args.seed,
        use_weather_api=args.use_weather_api,
        load_daily_energy_kwh=args.load_daily_energy_kwh,
        load_peak_kw=args.load_peak_kw,
    )
    daily = summarize(rollout, carbon_factor_kg_per_kwh=args.carbon_factor)
    suffix = f"{args.start_date}_{args.eval_days}d_pv_only_charge_eval"
    rollout.to_csv(output_dir / f"carbon_month_rollout_{suffix}.csv", index=False)
    daily.to_csv(output_dir / f"carbon_month_daily_summary_{suffix}.csv", index=False)
    plot_month(rollout, daily, output_dir / f"carbon_month_simulation_{suffix}.png")
    plot_daily_weekly_impact(daily, output_dir / f"carbon_month_impact_summary_{suffix}.png")

    total_baseline_co2 = float(daily["co2_baseline_kg"].sum())
    total_saved_co2 = float(daily["co2_saved_kg"].sum())
    print("Carbon-focused evaluation finished")
    print(f"Total load: {float(daily['load_energy_kwh'].sum()):.1f} kWh")
    print(f"Total grid: {float(daily['grid_energy_kwh'].sum()):.1f} kWh")
    print(f"Total PV used: {float(daily['pv_used_kwh'].sum()):.1f} kWh")
    print(f"Total cost: {float(daily['energy_cost_twd'].sum()):.1f} TWD")
    print(f"Cost saved vs no-battery baseline: {float(daily['cost_saved_twd'].sum()):.1f} TWD")
    print(f"CO2 saved vs no-battery baseline: {total_saved_co2:.1f} kg")
    print(f"CO2 reduction: {100.0 * total_saved_co2 / max(total_baseline_co2, 1e-9):.2f}%")
    print(f"SoC range: {float(rollout['soc'].min()):.3f} - {float(rollout['soc'].max()):.3f}")
    print(f"Infeasible discharge attempts: {int(rollout['infeasible_discharge'].sum())}")
    print(f"SoC violation attempts: {int(rollout['attempted_soc_violation'].sum())}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

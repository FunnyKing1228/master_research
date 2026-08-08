from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from livestock_microgrid_env import MicrogridEnv


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_CARBON_FACTOR_KG_PER_KWH = 0.495


def heuristic_policy(env: MicrogridEnv, obs: np.ndarray) -> np.ndarray:
    """Simple deployment-style controller for environment validation.

    Positive action charges the battery. Negative action discharges only when
    the battery can fully take over the current load, matching the no-partial
    grid-support constraint.
    """

    row = env.profile.iloc[env.step_idx]
    soc = float(obs[0])
    load_kw = float(row["load_kw"])
    pv_kw = float(row["weather_adjusted_power_kw"])
    price = float(row["grid_price_twd_per_kwh"])
    pv_surplus_kw = max(pv_kw - load_kw, 0.0)

    if pv_surplus_kw > 2.0 and soc < env.battery.soc_max - 0.01:
        charge_kw = min(pv_surplus_kw, env.battery.max_power_kw)
        return np.array([charge_kw / env.battery.max_power_kw], dtype=np.float32)

    available_energy_kwh = max((soc - env.battery.soc_min) * env.battery.capacity_kwh, 0.0)
    max_discharge_kw_by_soc = available_energy_kwh * env.battery.one_way_efficiency / env.dt_hours
    can_fully_supply_load = load_kw <= min(env.battery.max_power_kw, max_discharge_kw_by_soc)
    is_peak_price = price >= 7.0

    if is_peak_price and can_fully_supply_load and soc > env.battery.soc_min + 0.03:
        discharge_kw = min(load_kw, env.battery.max_power_kw)
        return np.array([-discharge_kw / env.battery.max_power_kw], dtype=np.float32)

    return np.array([0.0], dtype=np.float32)


def run_month(
    start_date: str,
    days: int,
    initial_soc: float,
    use_weather_api: bool,
    seed: int,
    load_daily_energy_kwh: float,
    load_peak_kw: float,
) -> pd.DataFrame:
    env = MicrogridEnv(
        start_date=start_date,
        use_weather_api=use_weather_api,
        seed=seed,
        load_daily_energy_kwh=load_daily_energy_kwh,
        load_peak_kw=load_peak_kw,
    )
    records: List[Dict[str, Any]] = []
    soc = initial_soc

    for day_idx, day in enumerate(pd.date_range(start=start_date, periods=days, freq="D")):
        day_str = day.strftime("%Y-%m-%d")
        obs, _ = env.reset(seed=seed + day_idx, options={"start_date": day_str, "initial_soc": soc})

        while True:
            action = heuristic_policy(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            info = info.copy()
            info["date"] = day_str
            info["reward"] = reward
            info["action"] = float(action[0])
            records.append(info)

            if terminated or truncated:
                soc = float(info["soc"])
                break

    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, carbon_factor_kg_per_kwh: float = DEFAULT_CARBON_FACTOR_KG_PER_KWH) -> pd.DataFrame:
    dt_hr = 0.25
    baseline_grid_kw = (df["load_kw"] - df["weather_adjusted_power_kw"]).clip(lower=0.0)
    if "grid_price_twd_per_kwh" in df.columns:
        price = df["grid_price_twd_per_kwh"]
    else:
        price = (df["energy_cost_twd"] / (df["grid_import_kw"] * dt_hr).replace(0.0, np.nan)).ffill().bfill()
    baseline_cost_twd = (baseline_grid_kw * dt_hr * price).fillna(0.0)

    daily = (
        df.assign(
            grid_energy_kwh=df["grid_import_kw"] * dt_hr,
            baseline_grid_energy_kwh=baseline_grid_kw * dt_hr,
            pv_used_kwh=df["pv_used_kw"] * dt_hr,
            load_energy_kwh=df["load_kw"] * dt_hr,
            battery_charge_kwh=df["battery_power_kw"].clip(lower=0.0) * dt_hr,
            battery_discharge_kwh=(-df["battery_power_kw"].clip(upper=0.0)) * dt_hr,
            baseline_cost_twd=baseline_cost_twd,
        )
        .groupby("date")
        .agg(
            load_energy_kwh=("load_energy_kwh", "sum"),
            grid_energy_kwh=("grid_energy_kwh", "sum"),
            baseline_grid_energy_kwh=("baseline_grid_energy_kwh", "sum"),
            pv_used_kwh=("pv_used_kwh", "sum"),
            battery_charge_kwh=("battery_charge_kwh", "sum"),
            battery_discharge_kwh=("battery_discharge_kwh", "sum"),
            energy_cost_twd=("energy_cost_twd", "sum"),
            baseline_cost_twd=("baseline_cost_twd", "sum"),
            reward=("reward", "sum"),
            soc_min=("soc", "min"),
            soc_max=("soc", "max"),
            soc_end=("soc", "last"),
            infeasible_discharge=("infeasible_discharge", "sum"),
            attempted_soc_violation=("attempted_soc_violation", "sum"),
        )
        .reset_index()
    )
    daily["grid_energy_saved_kwh"] = daily["baseline_grid_energy_kwh"] - daily["grid_energy_kwh"]
    daily["cost_saved_twd"] = daily["baseline_cost_twd"] - daily["energy_cost_twd"]
    daily["co2_baseline_kg"] = daily["baseline_grid_energy_kwh"] * carbon_factor_kg_per_kwh
    daily["co2_actual_kg"] = daily["grid_energy_kwh"] * carbon_factor_kg_per_kwh
    daily["co2_saved_kg"] = daily["grid_energy_saved_kwh"] * carbon_factor_kg_per_kwh
    daily["co2_reduction_pct"] = (
        100.0 * daily["co2_saved_kg"] / daily["co2_baseline_kg"].replace(0.0, np.nan)
    ).fillna(0.0)
    return daily


def plot_month(df: pd.DataFrame, daily: pd.DataFrame, output_path: Path) -> None:
    timestamps = pd.to_datetime(df["timestamp"])
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(4, 1, figsize=(15, 10.5), sharex=False)
    ax_power, ax_soc, ax_batt, ax_daily = axes

    ax_power.plot(timestamps, df["load_kw"], color="#333333", linewidth=1.1, label="Load")
    ax_power.plot(timestamps, df["weather_adjusted_power_kw"], color="#2ca02c", linewidth=1.1, label="Weather-adjusted PV")
    ax_power.plot(timestamps, df["grid_import_kw"], color="#ff7f0e", linewidth=1.0, label="Grid import")
    ax_power.set_ylabel("Power (kW)")
    ax_power.legend(loc="upper right", ncol=3)
    ax_power.grid(True, linestyle="--", alpha=0.25)

    ax_soc.plot(timestamps, df["soc"], color="#1f77b4", linewidth=1.2, label="SoC")
    ax_soc.axhspan(0.2, 0.8, color="#2ca02c", alpha=0.10, label="20%-80% target")
    ax_soc.axhline(0.2, color="#aa0000", linestyle="--", linewidth=1.0)
    ax_soc.axhline(0.8, color="#aa0000", linestyle="--", linewidth=1.0)
    ax_soc.set_ylabel("SoC")
    ax_soc.set_ylim(0.0, 1.0)
    ax_soc.legend(loc="upper right")
    ax_soc.grid(True, linestyle="--", alpha=0.25)

    charge = df["battery_power_kw"].clip(lower=0.0)
    discharge = df["battery_power_kw"].clip(upper=0.0)
    ax_batt.fill_between(timestamps, 0, charge, color="#2ca02c", alpha=0.35, label="Charge")
    ax_batt.fill_between(timestamps, 0, discharge, color="#d62728", alpha=0.35, label="Discharge")
    ax_batt.axhline(0.0, color="#333333", linewidth=0.8)
    ax_batt.set_ylabel("Battery (kW)")
    ax_batt.legend(loc="upper right", ncol=2)
    ax_batt.grid(True, linestyle="--", alpha=0.25)

    day_index = pd.to_datetime(daily["date"])
    ax_daily.bar(day_index, daily["grid_energy_kwh"], color="#ff7f0e", alpha=0.75, label="Grid energy")
    ax_daily.plot(day_index, daily["pv_used_kwh"], color="#2ca02c", marker="o", linewidth=1.8, label="PV used")
    ax_daily.set_ylabel("Daily Energy (kWh)")
    ax_daily.set_xlabel("Date")
    ax_daily.legend(loc="upper right")
    ax_daily.grid(True, axis="y", linestyle="--", alpha=0.25)

    fig.suptitle("One-Month Livestock Farm Microgrid Simulation", y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_daily_weekly_impact(daily: pd.DataFrame, output_path: Path) -> None:
    """Plot daily and weekly energy/carbon impacts against the no-battery baseline."""

    plot_df = daily.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df = plot_df.sort_values("date").reset_index(drop=True)
    first_date = plot_df["date"].min()
    plot_df["week_idx"] = ((plot_df["date"] - first_date).dt.days // 7).astype(int)
    weekly = (
        plot_df.groupby("week_idx")
        .agg(
            week_start=("date", "min"),
            week_end=("date", "max"),
            baseline_grid_energy_kwh=("baseline_grid_energy_kwh", "sum"),
            grid_energy_kwh=("grid_energy_kwh", "sum"),
            co2_baseline_kg=("co2_baseline_kg", "sum"),
            co2_actual_kg=("co2_actual_kg", "sum"),
            grid_energy_saved_kwh=("grid_energy_saved_kwh", "sum"),
        )
        .reset_index()
    )
    weekly["co2_saved_kg"] = weekly["co2_baseline_kg"] - weekly["co2_actual_kg"]
    weekly["co2_reduction_pct"] = (
        100.0 * weekly["co2_saved_kg"] / weekly["co2_baseline_kg"].replace(0.0, np.nan)
    ).fillna(0.0)
    weekly["week_label"] = weekly.apply(
        lambda row: f"{pd.Timestamp(row['week_start']).strftime('%m%d')}~{pd.Timestamp(row['week_end']).strftime('%m%d')}",
        axis=1,
    )

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.2))
    ax_daily_grid, ax_weekly_grid, ax_daily_saved, ax_weekly_carbon = axes.ravel()

    dates = plot_df["date"]
    week_starts = pd.to_datetime(weekly["week_start"])
    week_labels = weekly["week_label"].tolist()
    width = 0.38
    x = np.arange(len(plot_df))
    ax_daily_grid.bar(x - width / 2, plot_df["baseline_grid_energy_kwh"], width=width, color="#bbbbbb", label="No battery baseline")
    ax_daily_grid.bar(x + width / 2, plot_df["grid_energy_kwh"], width=width, color="#ff7f0e", label="Controlled")
    ax_daily_grid.set_title("Daily Grid Energy")
    ax_daily_grid.set_ylabel("kWh")
    weekly_tick_positions = [int(np.argmin(np.abs((dates - day).dt.days.to_numpy()))) for day in week_starts]
    ax_daily_grid.set_xticks(weekly_tick_positions, week_labels, rotation=0)
    ax_daily_grid.legend(loc="upper right")
    ax_daily_grid.grid(True, axis="y", linestyle="--", alpha=0.25)

    wx = np.arange(len(weekly))
    ax_weekly_grid.bar(wx - width / 2, weekly["baseline_grid_energy_kwh"], width=width, color="#bbbbbb", label="No battery baseline")
    ax_weekly_grid.bar(wx + width / 2, weekly["grid_energy_kwh"], width=width, color="#ff7f0e", label="Controlled")
    ax_weekly_grid.set_title("Weekly Grid Energy")
    ax_weekly_grid.set_ylabel("kWh")
    ax_weekly_grid.set_xticks(wx, week_labels, rotation=0)
    ax_weekly_grid.legend(loc="upper right")
    ax_weekly_grid.grid(True, axis="y", linestyle="--", alpha=0.25)

    ax_daily_saved.bar(dates, plot_df["grid_energy_saved_kwh"], color=np.where(plot_df["grid_energy_saved_kwh"] >= 0, "#2ca02c", "#d62728"))
    ax_daily_saved.axhline(0.0, color="#333333", linewidth=0.8)
    ax_daily_saved.set_title("Daily Grid Energy Saved")
    ax_daily_saved.set_ylabel("kWh")
    ax_daily_saved.set_xticks(week_starts)
    ax_daily_saved.xaxis.set_major_formatter(mdates.DateFormatter("%m%d"))
    ax_daily_saved.grid(True, axis="y", linestyle="--", alpha=0.25)

    ax_weekly_carbon.bar(wx, weekly["co2_saved_kg"], color="#1f77b4", alpha=0.85, label="CO2 saved")
    ax_weekly_carbon.axhline(0.0, color="#333333", linewidth=0.8)
    ax_weekly_carbon.set_title("Weekly CO2 Saved")
    ax_weekly_carbon.set_ylabel("kg CO2")
    ax_weekly_carbon.set_xticks(wx, week_labels, rotation=0)
    ax_weekly_carbon.grid(True, axis="y", linestyle="--", alpha=0.25)
    max_weekly_co2 = max(float(weekly["co2_saved_kg"].max()), 1.0)
    min_weekly_co2 = min(float(weekly["co2_saved_kg"].min()), 0.0)
    ax_weekly_carbon.set_ylim(min_weekly_co2 * 1.10, max_weekly_co2 * 1.22)

    for idx, row in weekly.iterrows():
        ax_weekly_carbon.text(
            idx,
            row["co2_saved_kg"] + max_weekly_co2 * 0.035,
            f"{row['co2_reduction_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    total_energy_saved = float(plot_df["grid_energy_saved_kwh"].sum())
    total_co2_saved = float(plot_df["co2_saved_kg"].sum())
    total_co2_pct = 100.0 * total_co2_saved / max(float(plot_df["co2_baseline_kg"].sum()), 1e-9)
    fig.suptitle("Energy and Carbon Impact vs No-Battery Baseline", y=0.985)
    fig.text(
        0.5,
        0.018,
        f"Grid energy saved: {total_energy_saved:.0f} kWh | CO2 saved: {total_co2_saved:.0f} kg ({total_co2_pct:.1f}%)",
        ha="center",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), h_pad=2.2, w_pad=1.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-month livestock farm microgrid simulation.")
    parser.add_argument("--start-date", default="2026-05-01")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--initial-soc", type=float, default=0.50)
    parser.add_argument("--load-daily-energy-kwh", type=float, default=905.8064516129032)
    parser.add_argument("--load-peak-kw", type=float, default=104.0)
    parser.add_argument("--use-weather-api", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--carbon-factor", type=float, default=DEFAULT_CARBON_FACTOR_KG_PER_KWH)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = run_month(
        start_date=args.start_date,
        days=args.days,
        initial_soc=args.initial_soc,
        use_weather_api=args.use_weather_api,
        seed=args.seed,
        load_daily_energy_kwh=args.load_daily_energy_kwh,
        load_peak_kw=args.load_peak_kw,
    )
    daily = summarize(df, carbon_factor_kg_per_kwh=args.carbon_factor)

    suffix = f"{args.start_date}_{args.days}d"
    df.to_csv(output_dir / f"month_rollout_{suffix}.csv", index=False)
    daily.to_csv(output_dir / f"month_daily_summary_{suffix}.csv", index=False)
    plot_month(df, daily, output_dir / f"month_simulation_{suffix}.png")
    plot_daily_weekly_impact(daily, output_dir / f"month_impact_summary_{suffix}.png")

    print("Monthly simulation finished")
    print(f"Records: {len(df)}")
    print(f"Total load: {float(daily['load_energy_kwh'].sum()):.1f} kWh")
    print(f"Total grid: {float(daily['grid_energy_kwh'].sum()):.1f} kWh")
    print(f"Total PV used: {float(daily['pv_used_kwh'].sum()):.1f} kWh")
    print(f"Total cost: {float(daily['energy_cost_twd'].sum()):.1f} TWD")
    print(f"Cost saved vs no-battery baseline: {float(daily['cost_saved_twd'].sum()):.1f} TWD")
    print(f"CO2 saved vs no-battery baseline: {float(daily['co2_saved_kg'].sum()):.1f} kg")
    print(f"CO2 reduction: {100.0 * float(daily['co2_saved_kg'].sum()) / max(float(daily['co2_baseline_kg'].sum()), 1e-9):.2f}%")
    print(f"SoC range: {float(df['soc'].min()):.3f} - {float(df['soc'].max()):.3f}")
    print(f"Infeasible discharge attempts: {int(df['infeasible_discharge'].sum())}")
    print(f"SoC violation attempts: {int(df['attempted_soc_violation'].sum())}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

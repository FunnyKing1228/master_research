from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PARENT = THIS_DIR.parents[0]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from carbon_microgrid_env import CarbonFocusedMicrogridEnv  # noqa: E402
from run_month_simulation import (  # noqa: E402
    DEFAULT_CARBON_FACTOR_KG_PER_KWH,
    plot_daily_weekly_impact,
    plot_month,
    summarize,
)


DEFAULT_OUTPUT_DIR = THIS_DIR / "results"


class EpisodeLogger:
    def __init__(self) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.episode_rewards: List[float] = []
                self.episode_lengths: List[int] = []
                self.current_reward = 0.0
                self.current_length = 0

            def _on_step(self) -> bool:
                reward = float(np.asarray(self.locals["rewards"])[0])
                done = bool(np.asarray(self.locals["dones"])[0])
                self.current_reward += reward
                self.current_length += 1
                if done:
                    self.episode_rewards.append(self.current_reward)
                    self.episode_lengths.append(self.current_length)
                    self.current_reward = 0.0
                    self.current_length = 0
                return True

        self.callback = _Callback()


def make_env(
    start_date: str,
    seed: int,
    use_weather_api: bool,
    random_date_start: str | None = None,
    random_date_days: int = 0,
    load_daily_energy_kwh: float = 905.8064516129032,
    load_peak_kw: float = 104.0,
) -> CarbonFocusedMicrogridEnv:
    return CarbonFocusedMicrogridEnv(
        start_date=start_date,
        use_weather_api=use_weather_api,
        seed=seed,
        random_date_start=random_date_start,
        random_date_days=random_date_days,
        load_daily_energy_kwh=load_daily_energy_kwh,
        load_peak_kw=load_peak_kw,
    )


def plot_training_curve(log_df: pd.DataFrame, output_path: Path) -> None:
    if log_df.empty:
        return
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(log_df["episode"], log_df["episode_reward"], color="#1f77b4", alpha=0.5, label="Episode reward")
    if len(log_df) >= 5:
        ax.plot(
            log_df["episode"],
            log_df["episode_reward"].rolling(5, min_periods=1).mean(),
            color="#2ca02c",
            linewidth=2.0,
            label="Rolling mean (5 ep)",
        )
    ax.set_title("Carbon-Focused SAC Training Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train_sac(
    total_timesteps: int,
    start_date: str,
    seed: int,
    use_weather_api: bool,
    output_dir: Path,
    train_start_date: str | None,
    train_days: int,
    load_daily_energy_kwh: float,
    load_peak_kw: float,
) -> Any:
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(
        make_env(
            start_date=start_date,
            seed=seed,
            use_weather_api=use_weather_api,
            random_date_start=train_start_date,
            random_date_days=train_days,
            load_daily_energy_kwh=load_daily_energy_kwh,
            load_peak_kw=load_peak_kw,
        )
    )
    logger = EpisodeLogger()
    model = SAC(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=500,
        batch_size=128,
        gamma=0.98,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, callback=logger.callback, progress_bar=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "sac_carbon_focused_livestock.zip")
    log_df = pd.DataFrame(
        {
            "episode": np.arange(1, len(logger.callback.episode_rewards) + 1),
            "episode_reward": logger.callback.episode_rewards,
            "episode_length": logger.callback.episode_lengths,
        }
    )
    log_df.to_csv(output_dir / "carbon_training_log.csv", index=False)
    plot_training_curve(log_df, output_dir / "carbon_training_curve.png")
    return model


def rollout_month(
    model: Any,
    start_date: str,
    days: int,
    initial_soc: float,
    seed: int,
    use_weather_api: bool,
    load_daily_energy_kwh: float,
    load_peak_kw: float,
) -> pd.DataFrame:
    env = make_env(
        start_date=start_date,
        seed=seed,
        use_weather_api=use_weather_api,
        load_daily_energy_kwh=load_daily_energy_kwh,
        load_peak_kw=load_peak_kw,
    )
    records: List[Dict[str, Any]] = []
    soc = initial_soc

    for day_idx, day in enumerate(pd.date_range(start=start_date, periods=days, freq="D")):
        day_str = day.strftime("%Y-%m-%d")
        obs, _ = env.reset(seed=seed + day_idx, options={"start_date": day_str, "initial_soc": soc})
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            info = info.copy()
            info["date"] = day_str
            info["reward"] = reward
            info["action"] = float(np.asarray(action)[0])
            records.append(info)
            if terminated or truncated:
                soc = float(info["soc"])
                break

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a carbon-focused SAC policy.")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--start-date", default="2026-05-01")
    parser.add_argument("--train-start-date", default=None)
    parser.add_argument("--train-days", type=int, default=0)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--initial-soc", type=float, default=0.50)
    parser.add_argument("--load-daily-energy-kwh", type=float, default=905.8064516129032)
    parser.add_argument("--load-peak-kw", type=float, default=104.0)
    parser.add_argument("--use-weather-api", action="store_true")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--carbon-factor", type=float, default=DEFAULT_CARBON_FACTOR_KG_PER_KWH)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = train_sac(
        args.total_timesteps,
        args.start_date,
        args.seed,
        args.use_weather_api,
        output_dir,
        args.train_start_date,
        args.train_days,
        args.load_daily_energy_kwh,
        args.load_peak_kw,
    )
    rollout = rollout_month(
        model=model,
        start_date=args.start_date,
        days=args.eval_days,
        initial_soc=args.initial_soc,
        seed=args.seed + 10_000,
        use_weather_api=args.use_weather_api,
        load_daily_energy_kwh=args.load_daily_energy_kwh,
        load_peak_kw=args.load_peak_kw,
    )
    daily = summarize(rollout, carbon_factor_kg_per_kwh=args.carbon_factor)

    suffix = f"{args.start_date}_{args.eval_days}d_{args.total_timesteps}steps"
    rollout.to_csv(output_dir / f"carbon_month_rollout_{suffix}.csv", index=False)
    daily.to_csv(output_dir / f"carbon_month_daily_summary_{suffix}.csv", index=False)
    plot_month(rollout, daily, output_dir / f"carbon_month_simulation_{suffix}.png")
    plot_daily_weekly_impact(daily, output_dir / f"carbon_month_impact_summary_{suffix}.png")

    total_baseline_co2 = float(daily["co2_baseline_kg"].sum())
    total_saved_co2 = float(daily["co2_saved_kg"].sum())
    print("Carbon-focused SAC finished")
    print(f"Episodes logged: {len(pd.read_csv(output_dir / 'carbon_training_log.csv'))}")
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

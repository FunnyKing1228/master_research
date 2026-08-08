from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "data" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_thesis_behavior_figures import load_config, load_dataset, rollout_episode  # type: ignore
from weather_adjusted_pv_potential import (  # type: ignore
    LocationConfig,
    PVHardwareConfig,
    build_time_index,
    calculate_pv_potential,
    fetch_open_meteo_solar_weather,
)


MODEL_SPECS = [
    {
        "label": "Previous deployment candidate (v10 guided teacher, best)",
        "short": "v10_guided_best",
        "experiment": "v16sp_guided_teacher_v10_0505_toufix_occfix",
        "model": "best_sac_model.pth",
    },
    {
        "label": "New deployment-safe model (v12 no teacher, final)",
        "short": "v12_no_teacher_final",
        "experiment": "v16sp_no_teacher_v12_0505_s2block_deploysafe",
        "model": "final_sac_model.pth",
    },
]


def load_experiment_config(experiment: str) -> Dict[str, Any]:
    config_path = ROOT / "experiments" / experiment / "configs" / "experiment_config.yaml"
    config = load_config(str(config_path))
    if "config" in config and isinstance(config["config"], dict):
        config = config["config"]
    return config


def rollout_model(spec: Dict[str, str], date: str, dataset_override: str | None = None) -> pd.DataFrame:
    config = load_experiment_config(spec["experiment"])
    if dataset_override:
        config["env"]["dataset_csv_path"] = dataset_override
    dataset_df = load_dataset(config["env"]["dataset_csv_path"])
    model_path = ROOT / "experiments" / spec["experiment"] / "models" / spec["model"]
    df = rollout_episode(
        config,
        dataset_df,
        model_path=model_path,
        start_date=f"{date} 00:00:00",
        days=1,
        reset_seed=42,
    )
    df["model"] = spec["short"]
    df["model_label"] = spec["label"]
    df["projection_delta_w"] = (df["action_safe_w"] - df["action_raw_w"]).abs()
    return df


def summarize_rollout(df: pd.DataFrame) -> Dict[str, float | str]:
    dt_hr = 0.25
    baseline_grid_w = (df["load_w"] - df["pv_bus_w"]).clip(lower=0.0)
    grid_cost = ((df["grid_draw_w"] / 1000.0) * dt_hr * df["price"]).sum()
    baseline_cost = ((baseline_grid_w / 1000.0) * dt_hr * df["price"]).sum()
    savings = baseline_cost - grid_cost
    projection_events = int((df["projection_delta_w"] > 0.05).sum())
    discharge_steps = int((df["action_applied_w"] < -1e-6).sum())
    charge_steps = int((df["action_applied_w"] > 1e-6).sum())

    return {
        "model": str(df["model"].iloc[0]),
        "soc_start": float(df["soc_start"].iloc[0]),
        "soc_end": float(df["soc"].iloc[-1]),
        "soc_min": float(df["soc"].min()),
        "soc_max": float(df["soc"].max()),
        "charge_steps": charge_steps,
        "discharge_steps": discharge_steps,
        "pv_to_battery_wh": float(df["pv_to_battery_w"].clip(lower=0.0).sum() * dt_hr),
        "useful_discharge_wh": float(df["useful_discharge_w"].clip(lower=0.0).sum() * dt_hr),
        "grid_cost_twd": float(grid_cost),
        "baseline_cost_twd": float(baseline_cost),
        "grid_savings_twd": float(savings),
        "reward_sum": float(df["reward"].sum()),
        "projection_events": projection_events,
        "projection_delta_mean_w": float(df["projection_delta_w"].mean()),
        "projection_delta_max_w": float(df["projection_delta_w"].max()),
    }


def summarize_training_log(spec: Dict[str, str], last_n: int = 100) -> Dict[str, float | str]:
    log_path = ROOT / "experiments" / spec["experiment"] / "logs" / "episode_log.csv"
    df = pd.read_csv(log_path)
    tail = df.tail(min(last_n, len(df)))
    fields = [
        "ep_reward",
        "violations_attempted",
        "violations_realized",
        "safety_projected_meaningful",
        "projection_delta_mean_w",
        "projection_delta_max_w",
        "revenue",
        "cost",
        "net_profit",
        "soc_min",
        "soc_max",
    ]
    out: Dict[str, float | str] = {"model": spec["short"], "last_n": len(tail)}
    for field in fields:
        if field in tail.columns:
            out[field] = float(tail[field].mean())
    return out


def load_weather_potential(date: str, mode: str) -> pd.DataFrame:
    cache_path = ROOT / "data" / "raw" / "figures" / f"weather_adjusted_pv_{mode}_{date}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
        cached["timestamp"] = cached["timestamp"].dt.tz_convert("Asia/Taipei")
        return cached.set_index("timestamp")

    location = LocationConfig()
    hardware = PVHardwareConfig()
    times = build_time_index(
        date=date,
        timezone=location.timezone,
        start_time="00:00",
        end_time="23:45",
        freq="15min",
    )
    try:
        weather = fetch_open_meteo_solar_weather(date, location, mode=mode)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch Open-Meteo data and no cache was found at {cache_path}"
        ) from exc
    result = calculate_pv_potential(times, weather, location, hardware)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path)
    return result


def plot_comparison(
    rollouts: List[pd.DataFrame],
    weather: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    colors = {
        "v10_guided_best": "#d62728",
        "v12_no_teacher_final": "#1f77b4",
    }

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 13.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.15, 1.2, 1.2, 1.15]},
    )
    ax_weather, ax_soc, ax_action, ax_grid, ax_rain = axes

    ax_weather.plot(
        weather.index,
        weather["clear_sky_power_w"],
        color="#f2a900",
        linestyle="--",
        linewidth=2.0,
        label="Clear-sky upper bound",
    )
    ax_weather.plot(
        weather.index,
        weather["weather_adjusted_power_w"],
        color="#2ca02c",
        linewidth=2.3,
        label="Weather-adjusted PV potential",
    )
    for df in rollouts:
        ax_weather.plot(
            pd.to_datetime(df["timestamp"]).dt.tz_localize("Asia/Taipei"),
            df["pv_bus_w"],
            color=colors[df["model"].iloc[0]],
            linewidth=1.5,
            alpha=0.75,
            label=f"Observed bus PV ({df['model'].iloc[0]})",
        )
    ax_weather.set_ylabel("Solar / PV (W)")
    ax_weather.legend(loc="upper right", ncol=2)
    ax_weather.grid(True, linestyle="--", alpha=0.25)

    for df in rollouts:
        ts = pd.to_datetime(df["timestamp"]).dt.tz_localize("Asia/Taipei")
        color = colors[df["model"].iloc[0]]
        label = df["model"].iloc[0]
        ax_soc.plot(ts, df["soc"], color=color, linewidth=2.0, label=label)
        ax_action.plot(ts, df["action_applied_w"], color=color, linewidth=1.8, label=label)
        ax_grid.plot(ts, df["grid_draw_w"], color=color, linewidth=1.8, label=label)

    ax_soc.axhline(0.2, color="#aa0000", linestyle="--", linewidth=1.1)
    ax_soc.axhline(0.8, color="#aa0000", linestyle="--", linewidth=1.1)
    ax_soc.set_ylabel("SoC")
    ax_soc.legend(loc="upper right")
    ax_soc.grid(True, linestyle="--", alpha=0.25)

    ax_action.axhline(0, color="#333333", linewidth=0.9)
    ax_action.set_ylabel("Battery Action (W)")
    ax_action.legend(loc="upper right")
    ax_action.grid(True, linestyle="--", alpha=0.25)

    ax_grid.set_ylabel("Grid Draw (W)")
    ax_grid.legend(loc="upper right")
    ax_grid.grid(True, linestyle="--", alpha=0.25)

    rain_hourly = weather["rain_mm"].resample("1h").first()
    precip_hourly = weather["precipitation_mm"].resample("1h").first()
    ax_rain.bar(
        precip_hourly.index,
        precip_hourly,
        width=0.032,
        color="#4c78a8",
        alpha=0.5,
        label="Precipitation (mm)",
    )
    ax_rain.bar(
        rain_hourly.index,
        rain_hourly,
        width=0.021,
        color="#1f4e79",
        alpha=0.7,
        label="Rain (mm)",
    )
    ax_cloud = ax_rain.twinx()
    ax_cloud.plot(
        weather.index,
        weather["cloud_cover_pct"],
        color="#777777",
        linewidth=1.5,
        alpha=0.7,
        label="Cloud cover (%)",
    )
    ax_rain.set_ylabel("Rain / Precip. (mm)")
    ax_cloud.set_ylabel("Cloud (%)")
    ax_cloud.set_ylim(0, 100)
    lines1, labels1 = ax_rain.get_legend_handles_labels()
    lines2, labels2 = ax_cloud.get_legend_handles_labels()
    ax_rain.legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=3)
    ax_rain.grid(True, linestyle="--", alpha=0.25)
    ax_rain.set_xlabel("Time")

    fig.suptitle(title, y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved comparison plot: {output_path}")


def write_markdown_summary(
    output_path: Path,
    date: str,
    training_summary: pd.DataFrame,
    rollout_summary: pd.DataFrame,
) -> None:
    def to_markdown_table(df: pd.DataFrame) -> str:
        formatted = df.copy()
        for col in formatted.columns:
            if pd.api.types.is_float_dtype(formatted[col]):
                formatted[col] = formatted[col].map(lambda x: f"{x:.4f}")
        header = "| " + " | ".join(formatted.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in formatted.to_numpy()
        ]
        return "\n".join([header, sep] + rows)

    lines = [
        f"# Model Comparison Report ({date})",
        "",
        "## Training Log Summary (last 100 episodes)",
        "",
        to_markdown_table(training_summary),
        "",
        "## Same-Day Rollout Summary",
        "",
        to_markdown_table(rollout_summary),
        "",
        "Notes:",
        "- `projection_events` counts meaningful SafetyNet corrections over 0.05 W in the same-day rollout.",
        "- `grid_savings_twd` is baseline grid cost minus model grid cost for the same dataset day.",
        "- Weather/PV potential is from Open-Meteo plus pvlib using the mock PV panel specification.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown summary: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old/new model rollouts with weather evidence.")
    parser.add_argument("--date", default="2026-05-05")
    parser.add_argument("--weather-mode", choices=["forecast", "archive"], default="forecast")
    parser.add_argument("--dataset-override", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "raw" / "figures" / "model_comparison_0505"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weather = load_weather_potential(args.date, args.weather_mode)
    rollouts = [rollout_model(spec, args.date, args.dataset_override) for spec in MODEL_SPECS]

    training_summary = pd.DataFrame([summarize_training_log(spec) for spec in MODEL_SPECS])
    rollout_summary = pd.DataFrame([summarize_rollout(df) for df in rollouts])

    training_summary.to_csv(output_dir / f"training_summary_{args.date}.csv", index=False)
    rollout_summary.to_csv(output_dir / f"same_day_rollout_summary_{args.date}.csv", index=False)
    for df in rollouts:
        df.to_csv(output_dir / f"{df['model'].iloc[0]}_rollout_{args.date}.csv", index=False)
    weather.to_csv(output_dir / f"weather_potential_{args.date}.csv")

    plot_comparison(
        rollouts,
        weather,
        output_dir / f"model_rollout_weather_comparison_{args.date}.png",
        title=f"Old vs New Model Under Same Weather / Dataset Day ({args.date})",
    )
    write_markdown_summary(
        output_dir / f"model_comparison_report_{args.date}.md",
        args.date,
        training_summary,
        rollout_summary,
    )

    print("\nTraining summary")
    print(training_summary.to_string(index=False))
    print("\nSame-day rollout summary")
    print(rollout_summary.to_string(index=False))


if __name__ == "__main__":
    main()


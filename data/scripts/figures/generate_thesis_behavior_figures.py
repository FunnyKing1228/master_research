from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = ROOT / "core"
EXTERNAL_SN_DIR = ROOT.parent / "HaoYuResearch" / "mircrogrid_sac" / "src"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if EXTERNAL_SN_DIR.exists() and str(EXTERNAL_SN_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_SN_DIR))

from train_sac_microgrid import (  # type: ignore
    create_agent,
    create_environment,
    get_power_limits,
    load_config,
    norm_to_power_kw,
)

try:
    from safety_net import clear_residual_buffer, project as safety_project, set_conformal_params  # type: ignore
except Exception:
    clear_residual_buffer = None
    set_conformal_params = None
    safety_project = None


DEFAULT_EXPERIMENT_NAME = "v16s_aggr1000"


PRICE_BANDS = {
    2.06: ("Off-peak", "#d8ecff"),
    4.69: ("Mid-peak", "#fff0bf"),
    7.13: ("Peak", "#ffd9d9"),
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_dataset(dataset_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def timestamp_to_index(df: pd.DataFrame, start_date: str) -> int:
    target = pd.Timestamp(start_date)
    matches = df.index[df["timestamp"] == target].tolist()
    if not matches:
        raise ValueError(f"Start date {start_date} not found in dataset.")
    return int(matches[0])


def build_env_and_agent(config: Dict[str, Any], model_path: Path, device: str = "cpu"):
    env = create_environment(config)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = create_agent(config, state_dim=state_dim, action_dim=action_dim, device=device)
    agent.load(str(model_path))
    return env, agent


def rollout_episode(
    base_config: Dict[str, Any],
    dataset_df: pd.DataFrame,
    model_path: Path,
    start_date: str,
    days: int = 1,
    reset_seed: int = 42,
    use_safetynet: bool = True,
) -> pd.DataFrame:
    config = copy.deepcopy(base_config)
    steps_per_day = 96
    episode_length = int(days * steps_per_day)
    config["env"]["episode_length"] = episode_length
    env, agent = build_env_and_agent(config, model_path=model_path)

    start_idx = timestamp_to_index(dataset_df, start_date)
    env.fixed_start_idx = start_idx

    state, reset_info = env.reset(seed=reset_seed)
    start_soc = float(reset_info.get("soc", env.current_soc))

    if use_safetynet and set_conformal_params is not None and clear_residual_buffer is not None:
        window = int(config.get("conformal", {}).get("window", max(episode_length * 2, 10)))
        delta = float(config.get("conformal", {}).get("delta", 0.1))
        set_conformal_params(window=window, delta=delta)
        clear_residual_buffer()

    charge_limit_kw, discharge_limit_kw = get_power_limits(env)
    pmax = max(charge_limit_kw, discharge_limit_kw)
    prev_action_kw = 0.0
    records: List[Dict[str, Any]] = []

    for step in range(episode_length):
        timestamp = dataset_df.loc[start_idx + step, "timestamp"]
        action_norm = agent.select_action(state, evaluate=True)
        action_raw_kw = norm_to_power_kw(float(action_norm[0]), env)

        projected_kw = action_raw_kw
        if use_safetynet and safety_project is not None:
            projected_kw, _, _ = safety_project(
                state=state,
                action=np.array([action_raw_kw], dtype=np.float32),
                prev_action=prev_action_kw,
                pmax=pmax,
                pmin=discharge_limit_kw,
                pmax_positive=charge_limit_kw,
                ramp_kw=getattr(env, "safetynet_ramp_kw", None),
                soc_bounds=(float(getattr(env, "soc_min", 0.0)), float(getattr(env, "soc_max", 1.0))),
                env=env,
            )
            projected_kw = float(projected_kw)
        next_state, reward, terminated, truncated, info = env.step([projected_kw])

        if len(state) >= 7:
            pv_support_ratio = float(state[2])
            pv_bool = float(state[3])
        else:
            load_kw = float(info["load"])
            pv_kw = float(info["pv"])
            pv_support_ratio = float(np.clip(pv_kw / max(load_kw, 1e-9), 0.0, 1.5))
            pv_bool = float(1.0 if pv_kw > 1e-3 else 0.0)

        records.append(
            {
                "timestamp": timestamp,
                "hour": step * 0.25,
                "soc": float(info["current_soc"]),
                "soc_start": start_soc,
                "price": float(info["price"]),
                "pv_support_ratio": pv_support_ratio,
                "pv_bool": pv_bool,
                "action_raw_w": action_raw_kw * 1000.0,
                "action_safe_w": projected_kw * 1000.0,
                "action_applied_w": float(info["applied_action_kw"]) * 1000.0,
                "load_w": float(info["load"]) * 1000.0,
                "pv_bus_w": float(info["pv"]) * 1000.0,
                "grid_draw_w": float(info["grid_kw"]) * 1000.0,
                "pv_to_load_w": float(info["pv_to_load"]) * 1000.0,
                "pv_to_battery_w": float(info["pv_to_battery"]) * 1000.0,
                "useful_discharge_w": float(info["useful_discharge"]) * 1000.0,
                "reward": float(reward),
            }
        )

        prev_action_kw = projected_kw
        state = next_state
        if terminated or truncated:
            break

    out = pd.DataFrame.from_records(records)
    out.attrs["start_soc"] = start_soc
    out.attrs["start_date"] = start_date
    out.attrs["days"] = days
    out.attrs["reset_seed"] = reset_seed
    out.attrs["pv_bool_ratio_threshold"] = float(config["env"].get("pv_sufficient_ratio_threshold", 0.8))
    return out


def add_price_bands(ax: plt.Axes, timestamps: pd.Series, prices: pd.Series) -> None:
    if len(timestamps) == 0:
        return
    band_starts = [0]
    prices_np = prices.to_numpy()
    for idx in range(1, len(prices_np)):
        if not np.isclose(prices_np[idx], prices_np[idx - 1]):
            band_starts.append(idx)
    band_starts.append(len(prices_np))
    for start, end in zip(band_starts[:-1], band_starts[1:]):
        value = float(prices_np[start])
        color = PRICE_BANDS.get(round(value, 2), ("", "#f3f3f3"))[1]
        ax.axvspan(timestamps.iloc[start], timestamps.iloc[end - 1], color=color, alpha=0.55, lw=0)


def style_time_axis(ax: plt.Axes, days: int) -> None:
    if days <= 1:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 25, 3)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time of day")
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.set_xlabel("Time")


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_boolean_spans(
    ax: plt.Axes,
    timestamps: pd.Series,
    bool_values: pd.Series,
    color: str = "#f6c445",
    alpha: float = 0.2,
    label: str | None = None,
) -> None:
    if len(timestamps) == 0 or len(bool_values) == 0:
        return

    values = bool_values.fillna(0.0).to_numpy(dtype=float) > 0.5
    start_idx: int | None = None
    label_used = False
    for idx, is_true in enumerate(values):
        if is_true and start_idx is None:
            start_idx = idx
        is_last = idx == len(values) - 1
        if start_idx is not None and ((not is_true) or is_last):
            end_idx = idx if not is_true else idx + 1
            start_ts = timestamps.iloc[start_idx]
            if end_idx < len(timestamps):
                end_ts = timestamps.iloc[end_idx]
            else:
                delta = (
                    timestamps.iloc[-1] - timestamps.iloc[-2]
                    if len(timestamps) > 1
                    else pd.Timedelta(minutes=15)
                )
                end_ts = timestamps.iloc[-1] + delta
            ax.axvspan(
                start_ts,
                end_ts,
                color=color,
                alpha=alpha,
                lw=0,
                label=(label if (label and not label_used) else None),
            )
            label_used = True
            start_idx = None


def plot_signed_battery_power(ax: plt.Axes, x: pd.Series, action_w: pd.Series) -> None:
    charge_w = action_w.clip(lower=0.0)
    discharge_w = action_w.clip(upper=0.0)

    ax.fill_between(
        x,
        0.0,
        charge_w,
        step="mid",
        color="#0f9d8a",
        alpha=0.85,
        label="Battery charge",
    )
    ax.fill_between(
        x,
        0.0,
        discharge_w,
        step="mid",
        color="#e15759",
        alpha=0.85,
        label="Battery discharge",
    )
    ax.step(x, charge_w, where="mid", color="#0b7d6c", linewidth=1.0)
    ax.step(x, discharge_w, where="mid", color="#b33d3f", linewidth=1.0)


def build_load_share_frame(df: pd.DataFrame) -> pd.DataFrame:
    load_w = df["load_w"].clip(lower=1e-9)
    pv_to_load_w = df["pv_to_load_w"].clip(lower=0.0)
    battery_to_load_w = df["useful_discharge_w"].clip(lower=0.0)
    grid_to_load_w = (load_w - pv_to_load_w - battery_to_load_w).clip(lower=0.0)

    return pd.DataFrame(
        {
            "pv_load_ratio": np.clip(pv_to_load_w / load_w, 0.0, 1.0),
            "grid_load_ratio": np.clip(grid_to_load_w / load_w, 0.0, 1.0),
            "battery_load_ratio": np.clip(battery_to_load_w / load_w, 0.0, 1.0),
        },
        index=df.index,
    )


def plot_single_day(df: pd.DataFrame, out_stem: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.1, 0.8]},
    )

    timestamps = df["timestamp"]
    pv_supply_w = df["pv_to_load_w"].clip(lower=0.0)
    battery_supply_w = df["useful_discharge_w"].clip(lower=0.0)
    grid_supply_w = (df["load_w"] - pv_supply_w - battery_supply_w).clip(lower=0.0)

    axes[0].stackplot(
        timestamps,
        pv_supply_w,
        battery_supply_w,
        grid_supply_w,
        colors=["#f6c445", "#e15759", "#4e79a7"],
        alpha=0.88,
        labels=["PV supply", "Battery discharge", "Grid supply"],
    )
    axes[0].plot(
        timestamps,
        df["load_w"],
        color="#2f2f2f",
        linewidth=2.0,
        linestyle="--",
        label="Total load",
    )
    axes[0].plot(
        timestamps,
        df["pv_bus_w"],
        color="#ffb000",
        linewidth=2.4,
        linestyle="-",
        label="Total PV Available",
        zorder=5,
    )
    axes[0].set_ylabel("Power Supply (W)")
    axes[0].set_ylim(0.0, max(float(df["load_w"].max()), float(df["pv_bus_w"].max())) * 1.08)
    axes[0].set_title(
        f"Representative single-day behavior ({df.attrs['start_date']}, start SoC={df.attrs['start_soc']:.3f})",
        pad=10,
    )
    axes[0].legend(loc="upper left", ncols=2, frameon=True, framealpha=0.9)

    axes[1].plot(timestamps, df["soc"], color="#1f6f5f", linewidth=2.2, label="SoC")
    axes[1].axhline(0.1, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.85, label="SoC lower bound")
    axes[1].axhline(0.9, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.85, label="SoC upper bound")
    axes[1].set_ylabel("SoC")
    axes[1].set_ylim(0.0, 1.0)

    ax_soc_action = axes[1].twinx()
    charge_w = df["action_applied_w"].clip(lower=0.0)
    discharge_w = df["action_applied_w"].clip(upper=0.0)
    bar_width_days = 10 / (24 * 60)
    ax_soc_action.bar(
        timestamps,
        charge_w,
        width=bar_width_days,
        color="#17becf",
        alpha=0.75,
        label="Battery charge",
        align="center",
    )
    ax_soc_action.bar(
        timestamps,
        discharge_w,
        width=bar_width_days,
        color="#d62728",
        alpha=0.75,
        label="Battery discharge",
        align="center",
    )
    ax_soc_action.axhline(8.5, color="#17becf", linestyle=":", linewidth=1.0, alpha=0.45, label="Charge limit")
    ax_soc_action.axhline(-5.6, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.45, label="Discharge limit")
    ax_soc_action.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    ax_soc_action.set_ylabel("Battery Power (W)")

    lines_l, labels_l = axes[1].get_legend_handles_labels()
    lines_r, labels_r = ax_soc_action.get_legend_handles_labels()
    axes[1].legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", ncols=2, frameon=True, framealpha=0.9)

    axes[2].step(
        timestamps,
        df["price"],
        where="mid",
        color="#2f2f2f",
        linewidth=2.0,
        label="Price",
    )
    axes[2].set_ylabel("Price (TWD/kWh)")
    axes[2].legend(loc="upper left", frameon=True, framealpha=0.9)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    style_time_axis(axes[-1], days=1)
    plt.subplots_adjust(hspace=0.1, top=0.93)
    save_figure(fig, out_stem)


def plot_overlay(
    episodes: Dict[str, pd.DataFrame],
    out_stem: Path,
    title: str,
    include_seed_in_label: bool = False,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    for idx, (label, ep) in enumerate(episodes.items()):
        color = palette[idx % len(palette)]
        display_label = label if include_seed_in_label else str(pd.Timestamp(label).date())
        share_df = build_load_share_frame(ep)
        axes[0].plot(ep["hour"], ep["soc"], linewidth=2.0, color=color, label=display_label)
        axes[1].step(ep["hour"], ep["action_applied_w"], where="mid", linewidth=1.8, color=color, label=display_label)
        axes[2].plot(ep["hour"], share_df["pv_load_ratio"], linewidth=2.0, color=color, label=display_label)
        axes[3].plot(ep["hour"], share_df["grid_load_ratio"], linewidth=2.0, color=color, label=display_label)

    axes[0].axhline(0.1, color="#d62728", linestyle="--", linewidth=1.0)
    axes[0].axhline(0.9, color="#d62728", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("SoC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title(title, pad=10)

    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_ylabel("Battery\npower (W)")

    axes[2].set_ylabel("PV load\nshare")
    axes[2].set_ylim(0.0, 1.05)

    axes[3].set_ylabel("Grid load\nshare")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].set_xlabel("Hour of day")

    for ax in axes:
        ax.legend(loc="upper right", frameon=False)

    save_figure(fig, out_stem)


def plot_crossday(df: pd.DataFrame, out_stem: Path, title: str) -> None:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.1, 1.0, 1.0]},
        constrained_layout=True,
    )
    add_price_bands(axes[0], df["timestamp"], df["price"])
    axes[0].plot(df["timestamp"], df["price"], color="#2f2f2f", linewidth=2.0)
    axes[0].set_ylabel("Price")
    axes[0].set_title(title, pad=10)

    axes[1].plot(df["timestamp"], df["soc"], color="#1f6f5f", linewidth=2.1)
    axes[1].axhline(0.1, color="#d62728", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.9, color="#d62728", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("SoC")
    axes[1].set_ylim(0.0, 1.0)

    plot_signed_battery_power(axes[2], df["timestamp"], df["action_applied_w"])
    axes[2].axhline(0.0, color="#444444", linewidth=0.8)
    axes[2].axhline(8.5, color="#7fb3d5", linestyle=":", linewidth=1.0)
    axes[2].axhline(-5.6, color="#f1948a", linestyle=":", linewidth=1.0)
    axes[2].set_ylabel("Battery\npower (W)")
    axes[2].legend(loc="upper left", ncols=2, frameon=False)

    share_df = build_load_share_frame(df)
    axes[3].plot(df["timestamp"], share_df["grid_load_ratio"], color="#2c7be5", linewidth=1.8, label="Grid share")
    axes[3].plot(df["timestamp"], share_df["pv_load_ratio"], color="#f39c12", linewidth=1.8, label="PV share")
    axes[3].plot(
        df["timestamp"],
        share_df["battery_load_ratio"],
        color="#c0392b",
        linewidth=1.6,
        linestyle="--",
        label="Battery useful share",
    )
    axes[3].set_ylabel("Load share")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].legend(loc="upper right", frameon=False)

    style_time_axis(axes[-1], days=int(df.attrs["days"]))
    save_figure(fig, out_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-quality behavior figures.")
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT_NAME, help="Experiment folder name under experiments/")
    parser.add_argument("--model", type=str, default="best_sac_model.pth", help="Model filename under experiment models/")
    parser.add_argument("--output-subdir", type=str, default="thesis", help="Subdirectory under experiment results/")
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()
    experiment_name = args.experiment
    config_path = ROOT / "experiments" / experiment_name / "configs" / "experiment_config.yaml"
    model_path = ROOT / "experiments" / experiment_name / "models" / args.model
    out_dir = ROOT / "experiments" / experiment_name / "results" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(config_path))
    if "config" in config and isinstance(config["config"], dict):
        config = config["config"]
    dataset_df = load_dataset(config["env"]["dataset_csv_path"])

    single_day = rollout_episode(config, dataset_df, model_path, start_date="2026-03-31 00:00:00", days=1, reset_seed=42)
    plot_single_day(single_day, out_dir / "fig4_1_single_day_behavior")

    date_compare = {
        "2026-03-27 00:00:00": rollout_episode(config, dataset_df, model_path, "2026-03-27 00:00:00", days=1, reset_seed=42),
        "2026-03-31 00:00:00": rollout_episode(config, dataset_df, model_path, "2026-03-31 00:00:00", days=1, reset_seed=42),
        "2026-04-09 00:00:00": rollout_episode(config, dataset_df, model_path, "2026-04-09 00:00:00", days=1, reset_seed=42),
    }
    plot_overlay(
        date_compare,
        out_dir / "fig4_2_date_stability",
        title="Consistency across representative start dates",
    )

    init_soc_compare = {
        "2026-03-23 seed=1": rollout_episode(config, dataset_df, model_path, "2026-03-23 00:00:00", days=1, reset_seed=1),
        "2026-03-23 seed=7": rollout_episode(config, dataset_df, model_path, "2026-03-23 00:00:00", days=1, reset_seed=7),
    }
    plot_overlay(
        init_soc_compare,
        out_dir / "fig4_3_initial_soc_stability",
        title="Sensitivity to different initial SoC conditions",
        include_seed_in_label=True,
    )

    crossday_3 = rollout_episode(config, dataset_df, model_path, "2026-03-27 00:00:00", days=3, reset_seed=42)
    plot_crossday(
        crossday_3,
        out_dir / "fig4_4_crossday_3day",
        title="Three-day continuous simulation without SoC reset",
    )

    crossday_5 = rollout_episode(config, dataset_df, model_path, "2026-03-27 00:00:00", days=5, reset_seed=42)
    plot_crossday(
        crossday_5,
        out_dir / "fig4_5_crossday_5day",
        title="Five-day continuous simulation without SoC reset",
    )

    print(f"Saved thesis figures to: {out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

"""Plot multi-day SoC, power, and flow behavior for a trained SAC model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from train_sac_microgrid import create_agent, create_environment, load_config  # noqa: E402


def _first_midnight_start(dataset_path: Path, time_column: str, episode_length: int) -> int:
    df = pd.read_csv(dataset_path)
    if time_column not in df.columns:
        return 0
    ts = pd.to_datetime(df[time_column], errors="coerce")
    for idx, stamp in enumerate(ts):
        if pd.notna(stamp) and stamp.hour == 0 and stamp.minute == 0:
            if idx + episode_length <= len(df):
                return int(idx)
    return 0


def _decode_action(raw_action: np.ndarray, env) -> list[float]:
    power_kw = float(raw_action[0])
    power_kw = float(np.clip(power_kw, -env.battery_discharge_power_kw, env.battery_charge_power_kw))
    if getattr(env, "use_flow_rate_action", False) and raw_action.shape[0] >= 2:
        flow_fraction = float(np.clip((float(raw_action[1]) + 1.0) * 0.5, 0.0, 1.0))
        return [power_kw, flow_fraction]
    return [power_kw]


def run(args: argparse.Namespace) -> Path:
    cfg = load_config(str(args.config))
    env_cfg = cfg.setdefault("env", {})
    episode_length = int(args.episode_length or env_cfg.get("episode_length", 96))
    env_cfg["episode_length"] = episode_length
    dataset_path = ROOT / env_cfg.get("dataset_csv_path", "")
    if args.start_idx is None:
        start_idx = _first_midnight_start(
            dataset_path,
            str(env_cfg.get("dataset_time_column", "timestamp")),
            episode_length,
        )
    else:
        start_idx = int(args.start_idx)
    env_cfg["fixed_start_idx"] = start_idx

    time_column = str(env_cfg.get("dataset_time_column", "timestamp"))
    time_axis = np.arange(episode_length, dtype=float) * float(env_cfg.get("time_step", 0.25))
    if dataset_path.exists() and time_column:
        try:
            ts = pd.to_datetime(pd.read_csv(dataset_path, usecols=[time_column])[time_column], errors="coerce")
            ep_ts = ts.iloc[start_idx : start_idx + episode_length].reset_index(drop=True)
            if len(ep_ts) == episode_length and ep_ts.notna().all():
                time_axis = (ep_ts - ep_ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
        except Exception:
            pass

    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu"
    env = create_environment(cfg)
    if hasattr(env, "fixed_start_idx"):
        env.fixed_start_idx = int(start_idx)
    state, _ = env.reset(seed=args.seed)
    if args.init_soc is not None:
        env.current_soc = float(np.clip(args.init_soc, env.soc_min, env.soc_max))
        state = env._get_state()

    action_dim = int(env.action_space.shape[0])
    state_dim = int(env.observation_space.shape[0])
    agent = create_agent(cfg, state_dim, action_dim, device)
    checkpoint = torch.load(str(args.model), map_location=device)
    if isinstance(checkpoint, dict) and set(checkpoint.keys()) == {"actor"}:
        agent.actor.load_state_dict(checkpoint["actor"])
    else:
        agent.load(str(args.model))

    rows = []
    done = False
    step = 0
    while not done and step < episode_length:
        raw_action = agent.select_action(state, evaluate=True)
        decoded_action = _decode_action(raw_action, env)
        next_state, reward, terminated, truncated, info = env.step(decoded_action)
        load_kw = float(info.get("load", 0.0))
        pv_kw = float(info.get("pv", 0.0))
        rows.append(
            {
                "step": step,
                "hour": float(time_axis[step]) if step < len(time_axis) else step * float(env.time_step),
                "soc": float(info.get("current_soc", env.current_soc)),
                "raw_power_kw": float(raw_action[0]),
                "applied_power_kw": float(info.get("applied_action_kw", 0.0)),
                "flow_pct": float(info.get("flow_action", 0.0)) * 100.0,
                "load_w": load_kw * 1000.0,
                "pv_w": pv_kw * 1000.0,
                "pv_load_ratio": pv_kw / max(load_kw, 1e-9),
                "price": float(info.get("price", 0.0)),
                "situation_code": int(info.get("situation_code", 0)),
                "reward": float(reward),
            }
        )
        state = next_state
        done = bool(terminated or truncated)
        step += 1

    df = pd.DataFrame(rows)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output).with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)

    axes[0].plot(df["hour"], df["pv_load_ratio"], color="#2ca02c", linewidth=1.8)
    axes[0].axhline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("PV/load")
    duration_days = episode_length * float(env.time_step) / 24.0
    axes[0].set_title(f"{duration_days:g}-Day Flow-Control Behavior")

    axes[1].plot(df["hour"], df["soc"] * 100.0, color="#1f77b4", linewidth=2.0)
    axes[1].set_ylabel("SoC (%)")
    axes[1].set_ylim(15, 85)

    axes[2].plot(df["hour"], df["applied_power_kw"] * 1000.0, color="#d62728", linewidth=1.8)
    axes[2].fill_between(
        df["hour"],
        0,
        df["applied_power_kw"] * 1000.0,
        where=df["applied_power_kw"] >= 0,
        color="#d62728",
        alpha=0.16,
        label="Charge",
    )
    axes[2].fill_between(
        df["hour"],
        0,
        df["applied_power_kw"] * 1000.0,
        where=df["applied_power_kw"] < 0,
        color="#1f77b4",
        alpha=0.16,
        label="Discharge",
    )
    axes[2].set_ylabel("Battery power (W)")
    axes[2].legend(loc="upper right", frameon=False)

    axes[3].step(df["hour"], df["flow_pct"], where="post", color="#9467bd", linewidth=2.0)
    axes[3].set_ylabel("Flow (%)")
    axes[3].set_xlabel("Hour")
    axes[3].set_ylim(-5, 105)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(float(df["hour"].min()), float(df["hour"].max()))
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"Saved plot: {args.output}")
    print(f"Saved data: {csv_path}")
    print(
        "Summary: "
        f"charge_steps={(df['applied_power_kw'] > 1e-6).sum()}, "
        f"discharge_steps={(df['applied_power_kw'] < -1e-6).sum()}, "
        f"flow_min={df['flow_pct'].min():.1f}, flow_max={df['flow_pct'].max():.1f}"
    )
    return Path(args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--init-soc", type=float, default=0.70)
    parser.add_argument("--episode-length", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

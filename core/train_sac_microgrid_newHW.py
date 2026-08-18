"""Train a provisional newHW model using shared SAC components.

This entry point intentionally leaves core/train_sac_microgrid.py unchanged.
The resulting checkpoint is an in-sample smoke artifact, not a validated model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CORE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CORE_DIR.parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import train_sac_microgrid as shared
from experiment_manager import ExperimentManager
from microgrid_env_newHW import create_microgrid_env_newHW


def plot_results_newHW(
    metrics: dict,
    destination: Path,
    *,
    soc_min: float,
    soc_max: float,
    eval_every: int,
) -> None:
    def rolling_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        window = max(1, min(window, values.size))
        result = np.full(values.shape, np.nan, dtype=float)
        result[window - 1 :] = np.convolve(
            values,
            np.ones(window, dtype=float) / window,
            mode="valid",
        )
        return result

    episodes = np.arange(len(metrics["episode_rewards"]))
    episode_rewards = np.asarray(metrics["episode_rewards"], dtype=float)
    episode_lengths = np.maximum(
        np.asarray(metrics["episode_lengths"], dtype=float),
        1.0,
    )
    attempted = np.asarray(metrics["episode_attempted_violations"], dtype=float)
    realized = np.asarray(metrics["episode_realized_violations"], dtype=float)
    projection_fraction = (
        np.asarray(metrics["episode_safety_projected"], dtype=float) / episode_lengths
    )
    soc_trajectories = np.asarray(metrics["episode_soc_trajectories"], dtype=float)
    soc_episode_min = np.min(soc_trajectories, axis=1)
    soc_episode_mean = np.mean(soc_trajectories, axis=1)
    soc_episode_end = soc_trajectories[:, -1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "newHW provisional training — IN-SAMPLE SMOKE ONLY\n"
        "Economic profit: N/A (off-grid; no tariff/revenue model). "
        "Negative values below are objective scores, not money."
    )

    axes[0, 0].plot(
        episodes,
        episode_rewards,
        alpha=0.22,
        linewidth=1.0,
        label="Episode objective",
    )
    axes[0, 0].plot(
        episodes,
        rolling_mean(episode_rewards),
        linewidth=2.2,
        label="20-episode mean",
    )
    eval_rewards = np.asarray(metrics.get("eval_rewards", []), dtype=float)
    if eval_rewards.size:
        eval_episodes = np.arange(eval_rewards.size) * max(int(eval_every), 1)
        axes[0, 0].scatter(
            eval_episodes,
            eval_rewards,
            marker="o",
            s=28,
            label="Deterministic evaluation",
            zorder=3,
        )
    axes[0, 0].set_title("Provisional objective score (higher is better; not profit)")
    axes[0, 0].set_ylabel("Objective score")
    axes[0, 0].legend()

    axes[0, 1].plot(
        episodes,
        attempted,
        alpha=0.20,
        linewidth=1.0,
        label="Attempted out-of-bounds",
    )
    axes[0, 1].plot(
        episodes,
        rolling_mean(attempted),
        linewidth=2.2,
        label="Attempted (20-episode mean)",
    )
    axes[0, 1].plot(
        episodes,
        realized,
        linewidth=1.8,
        label="Realized out-of-bounds",
    )
    axes[0, 1].set_title("SoC boundary violations per episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].legend()

    axes[1, 0].plot(
        episodes,
        projection_fraction,
        alpha=0.20,
        linewidth=1.0,
        label="Episode fraction",
    )
    axes[1, 0].plot(
        episodes,
        rolling_mean(projection_fraction),
        linewidth=2.2,
        label="20-episode mean",
    )
    axes[1, 0].set_title("SafetyNet intervention rate")
    axes[1, 0].set_ylabel("Fraction of episode steps")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].legend()

    axes[1, 1].plot(episodes, soc_episode_min, label="Episode minimum")
    axes[1, 1].plot(episodes, soc_episode_mean, label="Episode mean")
    axes[1, 1].plot(episodes, soc_episode_end, label="Episode end")
    axes[1, 1].axhline(
        soc_min,
        color="#d62728",
        linestyle="--",
        label=f"Operating min ({soc_min:.0%})",
    )
    axes[1, 1].axhline(
        soc_max,
        color="#d62728",
        linestyle=":",
        label=f"Operating max ({soc_max:.0%})",
    )
    axes[1, 1].set_title("SoC behavior by episode")
    axes[1, 1].set_ylabel("SoC")
    axes[1, 1].legend(ncols=2)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("Episode")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train provisional newHW SAC model")
    parser.add_argument("--config", default="configs/config_newHW_sim.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    if not args.name.startswith("newHW_"):
        raise ValueError("newHW experiment names must start with 'newHW_'")
    experiment_dir = REPO_ROOT / "experiments" / args.name
    if experiment_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {experiment_dir}")

    config = shared.load_config(args.config)
    if config.get("env", {}).get("hardware_family") != "newHW":
        raise ValueError("Config is not explicitly marked hardware_family: newHW")
    if args.episodes is not None:
        config["training"]["total_episodes"] = int(args.episodes)

    shared.set_conformal_params(
        window=int(config.get("conformal", {}).get("window", 376)),
        delta=float(config.get("conformal", {}).get("delta", 0.1)),
    )
    torch.manual_seed(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]))

    manager = ExperimentManager(args.name)
    manager.save_config(config)
    env = create_microgrid_env_newHW(config)
    env.safetynet_ramp_kw = config.get("safetynet", {}).get("ramp_limit_kw")

    device_cfg = config.get("device", "auto")
    device = (
        "cuda"
        if device_cfg == "auto" and torch.cuda.is_available()
        else ("cpu" if device_cfg == "auto" else device_cfg)
    )
    agent = shared.create_agent(
        config,
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        device=device,
    )

    print("newHW status: PROVISIONAL_IN_SAMPLE_ONLY")
    print(f"Dataset rows: {len(env.episode_data)}")
    print(f"Observation/action dimensions: {env.observation_space.shape[0]}/1")
    started = time.time()
    metrics = shared.train_sac_with_microgrid(env, agent, config, manager)
    elapsed = time.time() - started

    metadata = {
        "hardware_family": "newHW",
        "validation_status": "IN_SAMPLE_SMOKE_ONLY_NOT_VALIDATED",
        "reward_status": "PROVISIONAL_REQUIRES_HUMAN_DECISION",
        "variant": config["training"]["variant"],
        "seed": int(config["random_seed"]),
        "elapsed_seconds": elapsed,
        "profit_status": "NOT_APPLICABLE_OFFGRID_NO_TARIFF_OR_REVENUE_MODEL",
    }
    manager.save_results(metrics, metadata=metadata)

    results_dir = Path(manager.results_dir)
    np.savez(results_dir / "sac_training_metrics_newHW.npz", **metrics)
    plot_results_newHW(
        metrics,
        results_dir / "training_results_newHW.png",
        soc_min=float(env.soc_min),
        soc_max=float(env.soc_max),
        eval_every=int(config["training"]["eval_every"]),
    )
    (results_dir / "VALIDATION_STATUS_newHW.md").write_text(
        "# newHW validation status\n\n"
        "**IN-SAMPLE SMOKE ONLY — NOT A GENERALIZATION OR DEPLOYMENT VALIDATION.**\n\n"
        "- Source data covers only about 47 hours.\n"
        "- No independent validation or test split exists.\n"
        "- Continuous 3-day and 5-day rollout is impossible with this dataset.\n"
        "- Reward weights, BMS thresholds, power limits, initial SoC and I/O remain TODO(newHW).\n",
        encoding="utf-8",
    )
    print(f"Completed provisional newHW training in {elapsed:.1f}s")
    print(f"Experiment: {experiment_dir}")


if __name__ == "__main__":
    # Keep shared imports deterministic when invoked from another directory.
    os.chdir(REPO_ROOT)
    main()

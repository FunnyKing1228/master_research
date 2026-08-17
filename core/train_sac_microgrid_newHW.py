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


def plot_results_newHW(metrics: dict, destination: Path) -> None:
    episodes = np.arange(1, len(metrics["episode_rewards"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "newHW provisional training — IN-SAMPLE SMOKE ONLY\n"
        "TODO(newHW): reward weights and hardware limits require human confirmation"
    )
    axes[0, 0].plot(episodes, metrics["episode_rewards"], alpha=0.8)
    axes[0, 0].set_title("Episode reward (provisional reliability objective)")
    axes[0, 1].plot(episodes, metrics["episode_realized_violations"], label="Realized")
    axes[0, 1].plot(episodes, metrics["episode_attempted_violations"], label="Attempted")
    axes[0, 1].set_title("SoC boundary events")
    axes[0, 1].legend()
    axes[1, 0].plot(episodes, metrics["episode_safety_projected"])
    axes[1, 0].set_title("Meaningful SafetyNet projections")
    axes[1, 1].plot(episodes, metrics["episode_actions_raw"], label="Raw")
    axes[1, 1].plot(episodes, metrics["episode_actions_safe"], label="Safe")
    axes[1, 1].set_title("Average normalized action magnitude")
    axes[1, 1].legend()
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
    }
    manager.save_results(metrics, metadata=metadata)

    results_dir = Path(manager.results_dir)
    np.savez(results_dir / "sac_training_metrics_newHW.npz", **metrics)
    plot_results_newHW(metrics, results_dir / "training_results_newHW.png")
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

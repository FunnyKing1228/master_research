"""Generate CORAL tuning configs and runnable commands.

This script does not launch training. It writes a small, explainable tuning grid
for the seminar deadline. The grid focuses on the observed 1000-episode result:
the no-curriculum OCC ablation was more profitable than the full safety-first
CORAL setting, so we sweep less conservative CORAL settings first.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / "configs" / "experiments" / "p302" / "config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml"
OUT_DIR = ROOT / "configs" / "baselines" / "research" / "coral_tuning"
COMMANDS_PATH = OUT_DIR / "run_coral_tuning_commands.ps1"


VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "coral_occ_nocurriculum_b08",
        "reason": "Use the strongest 1000-episode ablation signal: SafetyNet + OCC without long curriculum.",
        "beta_occ": 0.8,
        "warmup": 0,
        "attempted_penalty": 0.10,
        "projection_penalty": 0.025,
    },
    {
        "name": "coral_occ_nocurriculum_b05",
        "reason": "Reduce OCC conservatism while keeping correction awareness.",
        "beta_occ": 0.5,
        "warmup": 0,
        "attempted_penalty": 0.08,
        "projection_penalty": 0.018,
    },
    {
        "name": "coral_occ_nocurriculum_b03",
        "reason": "Profit-oriented CORAL candidate with weaker OCC pressure.",
        "beta_occ": 0.3,
        "warmup": 0,
        "attempted_penalty": 0.06,
        "projection_penalty": 0.012,
    },
    {
        "name": "coral_shortwarmup100_b05",
        "reason": "Keep a short pure-SAC phase, but avoid the 250-episode safety-first delay.",
        "beta_occ": 0.5,
        "warmup": 100,
        "attempted_penalty": 0.08,
        "projection_penalty": 0.018,
    },
    {
        "name": "coral_shortwarmup150_b04",
        "reason": "Middle point between current full CORAL and no-curriculum OCC.",
        "beta_occ": 0.4,
        "warmup": 150,
        "attempted_penalty": 0.08,
        "projection_penalty": 0.015,
    },
]


def load_base() -> Dict[str, Any]:
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(base: Dict[str, Any], variant: Dict[str, Any], episodes: int, seed: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["random_seed"] = int(seed)
    cfg["training"]["total_episodes"] = int(episodes)
    cfg["training"]["max_steps"] = 96
    cfg["training"]["eval_every"] = 20
    cfg["training"]["eval_episodes"] = 3
    cfg["training"]["save_every"] = 50
    cfg["training"]["variant"] = "sac_sn"
    cfg["training"]["safetynet_warmup_episodes"] = int(variant["warmup"])
    cfg["sac"]["beta_occ"] = float(variant["beta_occ"])
    cfg["reward"]["attempted_violation_penalty"] = float(variant["attempted_penalty"])
    cfg["reward"]["safety_projection_penalty"] = float(variant["projection_penalty"])
    cfg["guided_teacher"]["enabled"] = False
    cfg["guided_teacher"]["demo_episodes"] = 0
    cfg["logging"]["plot_results"] = False
    cfg["logging"]["save_models"] = True
    cfg["logging"]["save_metrics"] = True
    cfg["logging"]["csv_per_episode"] = True
    cfg["coral_tuning_note"] = variant["reason"]
    return cfg


def main() -> None:
    episodes = 1000
    seed = 42
    base = load_base()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    commands = [
        '$ROOT = Resolve-Path "$PSScriptRoot\\..\\..\\..\\.."',
        "Set-Location $ROOT",
        "",
    ]
    manifest = []
    for variant in VARIANTS:
        cfg = build_config(base, variant, episodes=episodes, seed=seed)
        cfg_path = OUT_DIR / f"{variant['name']}_s{seed}_{episodes}.yaml"
        exp_name = f"seminar_tune_{variant['name']}_s{seed}_{episodes}"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        relative_cfg_path = cfg_path.relative_to(ROOT)
        commands.append(
            f"py core\\train_sac_microgrid.py --config {relative_cfg_path} --episodes {episodes} --name {exp_name}"
        )
        manifest.append(
            {
                "experiment": exp_name,
                "config": str(relative_cfg_path),
                **variant,
            }
        )

    with open(COMMANDS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(commands))
        f.write("\n")
    with open(OUT_DIR / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote configs to {OUT_DIR}")
    print(f"Wrote commands to {COMMANDS_PATH}")


if __name__ == "__main__":
    main()


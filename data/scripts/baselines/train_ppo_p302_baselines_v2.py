"""Train corrected PPO and PPO+SafetyNet baselines for the P302 seminar table.

Compared with the first PPO script, this version gives PPO normalized
observations and a normalized [-1, 1] action interface. The underlying P302
environment, reward, safety projection, and evaluation metrics remain the same.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from safety_net import clear_residual_buffer, project as safety_project, set_conformal_params  # noqa: E402
from train_sac_microgrid import create_environment, get_power_limits  # noqa: E402


DEFAULT_BASE_CONFIG = ROOT / "configs" / "experiments" / "p302" / "config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml"
BASE_CONFIG = DEFAULT_BASE_CONFIG
OUT_DIR = ROOT / "experiments" / "seminar_baseline_results" / "ppo_baselines_v2"


def load_config(path: Path | None = None) -> Dict[str, Any]:
    config_path = BASE_CONFIG if path is None else path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_config(episodes: int, seed: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg["random_seed"] = int(seed)
    cfg["training"]["total_episodes"] = int(episodes)
    cfg["training"]["max_steps"] = 96
    cfg["env"]["episode_length"] = 96
    cfg["guided_teacher"]["enabled"] = False
    cfg["guided_teacher"]["demo_episodes"] = 0
    return cfg


class ObservationScaleWrapper(gym.ObservationWrapper):
    """Map finite Box observations to roughly [-1, 1] for PPO stability."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        finite = np.isfinite(low) & np.isfinite(high) & ((high - low) > 1e-8)
        self._low = low
        self._high = high
        self._finite = finite
        self.observation_space = gym.spaces.Box(
            low=-np.ones_like(low, dtype=np.float32),
            high=np.ones_like(high, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        scaled = obs.copy()
        scaled[self._finite] = 2.0 * (obs[self._finite] - self._low[self._finite]) / (
            self._high[self._finite] - self._low[self._finite]
        ) - 1.0
        scaled[~self._finite] = obs[~self._finite]
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)


class NormalizedActionWrapper(gym.ActionWrapper):
    """Expose a symmetric [-1, 1] action to PPO and map it to battery kW."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._low = np.asarray(env.action_space.low, dtype=np.float32)
        self._high = np.asarray(env.action_space.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-np.ones_like(self._low, dtype=np.float32),
            high=np.ones_like(self._high, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        mapped = self._low + (action_arr + 1.0) * 0.5 * (self._high - self._low)
        return np.clip(mapped, self._low, self._high).astype(np.float32)


class SafetyNetTrainingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env)
        self.config = config
        self.prev_action_kw = 0.0
        self.last_obs = None
        conformal_cfg = config.get("conformal", {})
        set_conformal_params(
            window=int(conformal_cfg.get("window", 1440)),
            delta=float(conformal_cfg.get("delta", 0.1)),
        )
        clear_residual_buffer()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action_kw = 0.0
        self.last_obs = obs
        return obs, info

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32)
        raw_kw = float(raw_action[0])
        charge_limit_kw, discharge_limit_kw = get_power_limits(self.env)
        pmax = max(charge_limit_kw, discharge_limit_kw)
        projected_kw, _, _ = safety_project(
            state=np.asarray(self.last_obs, dtype=np.float32),
            action=np.array([raw_kw], dtype=np.float32),
            prev_action=self.prev_action_kw,
            pmax=pmax,
            pmin=discharge_limit_kw,
            pmax_positive=charge_limit_kw,
            ramp_kw=getattr(self.env, "safetynet_ramp_kw", None),
            soc_bounds=(float(getattr(self.env, "soc_min", 0.0)), float(getattr(self.env, "soc_max", 1.0))),
            env=self.env,
        )
        safe_action = raw_action.copy()
        safe_action[0] = float(projected_kw)
        obs, reward, terminated, truncated, info = self.env.step(safe_action.tolist())
        self.prev_action_kw = float(projected_kw)
        self.last_obs = obs
        info = dict(info)
        info["ppo_raw_action_kw"] = raw_kw
        info["ppo_projected_action_kw"] = float(projected_kw)
        return obs, reward, terminated, truncated, info


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, print_every: int = 9600):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.print_every = int(print_every)
        self.next_print = int(print_every)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_print:
            print(f"[PPO v2] {self.num_timesteps}/{self.total_timesteps} timesteps")
            self.next_print += self.print_every
        return True


def scale_observation(env: Any, observation: np.ndarray) -> np.ndarray:
    low = np.asarray(env.observation_space.low, dtype=np.float32)
    high = np.asarray(env.observation_space.high, dtype=np.float32)
    obs = np.asarray(observation, dtype=np.float32)
    finite = np.isfinite(low) & np.isfinite(high) & ((high - low) > 1e-8)
    scaled = obs.copy()
    scaled[finite] = 2.0 * (obs[finite] - low[finite]) / (high[finite] - low[finite]) - 1.0
    scaled[~finite] = obs[~finite]
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def normalized_to_kw(env: Any, normalized_action: np.ndarray) -> float:
    return float(normalized_to_env_action(env, normalized_action)[0])


def normalized_to_env_action(env: Any, normalized_action: np.ndarray) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    action_arr = np.asarray(normalized_action, dtype=np.float32)
    mapped = low + (action_arr + 1.0) * 0.5 * (high - low)
    return np.clip(mapped, low, high).astype(np.float32)


def make_env(config: Dict[str, Any], use_safetynet: bool = False):
    env = create_environment(config)
    if use_safetynet:
        env = SafetyNetTrainingWrapper(env, config)
    env = ObservationScaleWrapper(env)
    env = NormalizedActionWrapper(env)
    return Monitor(env)


def attempted_violation(env: Any, state: np.ndarray, raw_kw: float) -> int:
    try:
        current_soc = float(state[0])
        soc_next_raw = float(env.predict_soc_raw(current_soc, raw_kw))
        soc_min = float(getattr(env, "soc_min_eff", getattr(env, "soc_min", 0.0)))
        soc_max = float(getattr(env, "soc_max_eff", getattr(env, "soc_max", 1.0)))
        return int(soc_next_raw < soc_min or soc_next_raw > soc_max)
    except Exception:
        return 0


def evaluate_model(
    model: PPO,
    config: Dict[str, Any],
    use_safetynet: bool,
    episodes: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    env = create_environment(config)
    conformal_cfg = config.get("conformal", {})
    set_conformal_params(
        window=int(conformal_cfg.get("window", 1440)),
        delta=float(conformal_cfg.get("delta", 0.1)),
    )
    clear_residual_buffer()

    episode_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for ep in range(int(episodes)):
        state, _ = env.reset(seed=seed + ep)
        prev_action_kw = 0.0
        prev_soc_violations = 0
        reward_sum = 0.0
        attempted_count = 0
        projected_count = 0
        meaningful_count = 0
        projection_deltas = []
        realized_count = 0
        soc_values = [float(state[0])]
        raw_actions_kw = []
        final_info: Dict[str, Any] = {}

        for step in range(int(config["training"]["max_steps"])):
            scaled_state = scale_observation(env, np.asarray(state, dtype=np.float32))
            normalized_action, _ = model.predict(scaled_state, deterministic=True)
            raw_env_action = normalized_to_env_action(env, np.asarray(normalized_action, dtype=np.float32))
            raw_kw = float(raw_env_action[0])
            raw_actions_kw.append(raw_kw)
            attempted = attempted_violation(env, np.asarray(state, dtype=np.float32), raw_kw)
            attempted_count += attempted

            safe_kw = raw_kw
            safe_env_action = raw_env_action.copy()
            delta_kw = 0.0
            did_project = False
            if use_safetynet:
                charge_limit_kw, discharge_limit_kw = get_power_limits(env)
                pmax = max(charge_limit_kw, discharge_limit_kw)
                safe_kw, did_project, delta_kw = safety_project(
                    state=np.asarray(state, dtype=np.float32),
                    action=np.array([raw_kw], dtype=np.float32),
                    prev_action=prev_action_kw,
                    pmax=pmax,
                    pmin=discharge_limit_kw,
                    pmax_positive=charge_limit_kw,
                    ramp_kw=getattr(env, "safetynet_ramp_kw", None),
                    soc_bounds=(float(getattr(env, "soc_min", 0.0)), float(getattr(env, "soc_max", 1.0))),
                    env=env,
                )
                did_project = bool(did_project)
                projected_count += int(did_project)
                threshold = float(config.get("reward", {}).get("safety_projection_event_threshold_kw", 0.0))
                meaningful_count += int(did_project and float(delta_kw) > threshold)
                if did_project:
                    projection_deltas.append(float(delta_kw))
                safe_env_action[0] = float(safe_kw)

            next_state, reward, terminated, truncated, info = env.step(safe_env_action.tolist())
            current_soc_violations = int(info.get("soc_violations", 0))
            realized = max(0, current_soc_violations - prev_soc_violations)
            prev_soc_violations = current_soc_violations
            realized_count += realized
            reward_sum += float(reward)
            soc_values.append(float(info.get("current_soc", next_state[0])))
            final_info = dict(info)
            step_rows.append(
                {
                    "episode": ep,
                    "step": step,
                    "raw_action_kw": raw_kw,
                    "safe_action_kw": float(safe_kw),
                    "reward": float(reward),
                    "soc": float(info.get("current_soc", next_state[0])),
                    "realized_violation": realized,
                    "attempted_violation": attempted,
                    "projected": int(did_project),
                    "projection_delta_w": float(delta_kw) * 1000.0,
                    "net_profit": float(info.get("total_revenue", 0.0)) - float(info.get("total_cost", 0.0)),
                    "flow_action": float(info.get("flow_action", 0.0)),
                    "flow_power_limited": int(info.get("flow_power_limited", 0)),
                    "pump_power_wh": float(info.get("pump_power_kw", 0.0)) * float(getattr(env, "time_step", 0.25)) * 1000.0,
                }
            )
            prev_action_kw = float(safe_kw)
            state = next_state
            if terminated or truncated:
                break

        soc_arr = np.asarray(soc_values, dtype=np.float32)
        raw_arr = np.asarray(raw_actions_kw, dtype=np.float32)
        ep_steps = pd.DataFrame([r for r in step_rows if r["episode"] == ep])
        episode_rows.append(
            {
                "episode": ep,
                "ep_reward": reward_sum,
                "ep_length": len(soc_values) - 1,
                "soc_min": float(soc_arr.min()),
                "soc_max": float(soc_arr.max()),
                "soc_mean": float(soc_arr.mean()),
                "soc_end": float(soc_arr[-1]),
                "raw_action_kw_mean": float(raw_arr.mean()) if len(raw_arr) else 0.0,
                "raw_action_kw_min": float(raw_arr.min()) if len(raw_arr) else 0.0,
                "raw_action_kw_max": float(raw_arr.max()) if len(raw_arr) else 0.0,
                "violations_realized": realized_count,
                "violations_attempted": attempted_count,
                "safety_projected": projected_count,
                "safety_projected_meaningful": meaningful_count,
                "projection_delta_mean_w": float(np.mean(projection_deltas) * 1000.0) if projection_deltas else 0.0,
                "projection_delta_max_w": float(np.max(projection_deltas) * 1000.0) if projection_deltas else 0.0,
                "revenue": float(final_info.get("total_revenue", 0.0)),
                "cost": float(final_info.get("total_cost", 0.0)),
                "net_profit": float(final_info.get("total_revenue", 0.0)) - float(final_info.get("total_cost", 0.0)),
                "flow_action_mean": float(ep_steps["flow_action"].mean()) if "flow_action" in ep_steps else 0.0,
                "flow_active_mean": float(ep_steps.loc[ep_steps["flow_action"] > 1e-9, "flow_action"].mean())
                if "flow_action" in ep_steps and (ep_steps["flow_action"] > 1e-9).any()
                else 0.0,
                "flow_power_limited_count": int(ep_steps["flow_power_limited"].sum()) if "flow_power_limited" in ep_steps else 0,
                "pump_power_wh": float(ep_steps["pump_power_wh"].sum()) if "pump_power_wh" in ep_steps else 0.0,
            }
        )
    return pd.DataFrame(episode_rows), pd.DataFrame(step_rows)


def write_summary(name: str, episode_df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    tail = episode_df.tail(min(50, len(episode_df)))
    row = {
        "baseline": name,
        "episodes": int(len(episode_df)),
        "mode": "eval_last50_mean",
        "net_profit": float(tail["net_profit"].mean()),
        "ep_reward": float(tail["ep_reward"].mean()),
        "violations_realized": float(tail["violations_realized"].mean()),
        "violations_attempted": float(tail["violations_attempted"].mean()),
        "safety_projected_meaningful": float(tail["safety_projected_meaningful"].mean()),
        "projection_delta_mean_w": float(tail["projection_delta_mean_w"].mean()),
        "projection_delta_max_w": float(tail["projection_delta_max_w"].max()),
        "soc_min": float(tail["soc_min"].mean()),
        "soc_max": float(tail["soc_max"].mean()),
        "raw_action_kw_mean": float(tail["raw_action_kw_mean"].mean()),
        "raw_action_kw_min": float(tail["raw_action_kw_min"].mean()),
        "raw_action_kw_max": float(tail["raw_action_kw_max"].mean()),
        "flow_action_mean": float(tail["flow_action_mean"].mean()) if "flow_action_mean" in tail else 0.0,
        "flow_active_mean": float(tail["flow_active_mean"].mean()) if "flow_active_mean" in tail else 0.0,
        "flow_power_limited_count": float(tail["flow_power_limited_count"].mean()) if "flow_power_limited_count" in tail else 0.0,
        "pump_power_wh": float(tail["pump_power_wh"].mean()) if "pump_power_wh" in tail else 0.0,
    }
    summary_path = output_dir / "summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(row, f, sort_keys=False, allow_unicode=True)
    return row


def train_one(name: str, use_safetynet: bool, episodes: int, seed: int, eval_episodes: int) -> Dict[str, Any]:
    config = prepare_config(episodes=episodes, seed=seed)
    output_dir = OUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    env = make_env(config, use_safetynet=use_safetynet)
    total_timesteps = int(episodes) * int(config["training"]["max_steps"])
    model = PPO(
        "MlpPolicy",
        env,
        seed=int(seed),
        learning_rate=1e-4,
        n_steps=96,
        batch_size=96,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device="auto",
    )
    model.learn(total_timesteps=total_timesteps, callback=ProgressCallback(total_timesteps))
    model.save(output_dir / "ppo_model.zip")

    episode_df, step_df = evaluate_model(
        model,
        config=config,
        use_safetynet=use_safetynet,
        episodes=eval_episodes,
        seed=seed + 10000,
    )
    episode_df.to_csv(output_dir / "eval_episode_log.csv", index=False)
    step_df.to_csv(output_dir / "eval_step_log.csv", index=False)
    return write_summary(name, episode_df, output_dir)


def write_combined_summary(rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ppo_baseline_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    cols = [
        "baseline",
        "net_profit",
        "violations_realized",
        "violations_attempted",
        "safety_projected_meaningful",
        "projection_delta_mean_w",
        "projection_delta_max_w",
        "soc_min",
        "soc_max",
        "raw_action_kw_mean",
        "raw_action_kw_min",
        "raw_action_kw_max",
        "flow_action_mean",
        "flow_active_mean",
        "flow_power_limited_count",
        "pump_power_wh",
    ]
    with open(OUT_DIR / "ppo_baseline_summary.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("# PPO Baseline Summary V2\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df[cols].iterrows():
            values = []
            for value in row:
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.4f}")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train corrected P302 PPO / PPO+SafetyNet baselines.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--which", choices=["ppo", "ppo_sn", "both"], default="both")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    global BASE_CONFIG, OUT_DIR
    args = parse_args()
    BASE_CONFIG = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    OUT_DIR = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows: List[Dict[str, Any]] = []
    if args.which in ("ppo", "both"):
        rows.append(train_one("PPO", use_safetynet=False, episodes=args.episodes, seed=args.seed, eval_episodes=args.eval_episodes))
    if args.which in ("ppo_sn", "both"):
        rows.append(train_one("PPO + SafetyNet", use_safetynet=True, episodes=args.episodes, seed=args.seed, eval_episodes=args.eval_episodes))
    write_combined_summary(rows)
    print(OUT_DIR / "ppo_baseline_summary.md")


if __name__ == "__main__":
    main()


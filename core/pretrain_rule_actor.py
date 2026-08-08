import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rule_expert import rule_expert_action_from_state
from train_sac_microgrid import load_config, create_environment, create_agent


def _reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_state, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return next_state, reward, done, info
    next_state, reward, done, info = out
    return next_state, reward, bool(done), info


def collect_rule_dataset(env, demo_episodes: int, expert_mode: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    charge_steps = 0
    discharge_steps = 0
    idle_steps = 0

    battery_charge_power_kw = float(getattr(env, 'battery_charge_power_kw', getattr(env, 'battery_power_kw', 0.0)))
    battery_discharge_power_kw = float(getattr(env, 'battery_discharge_power_kw', getattr(env, 'battery_power_kw', 0.0)))
    soc_min = float(getattr(env, 'soc_min', 0.10))
    soc_max = float(getattr(env, 'soc_max', 0.90))

    for _ in range(demo_episodes):
        state = _reset_env(env)
        done = False
        while not done:
            expert_action = rule_expert_action_from_state(
                state=state,
                battery_charge_power_kw=battery_charge_power_kw,
                battery_discharge_power_kw=battery_discharge_power_kw,
                soc_min=soc_min,
                soc_max=soc_max,
                mode=expert_mode,
            )
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(np.asarray([expert_action], dtype=np.float32))

            if expert_action > 1e-6:
                charge_steps += 1
            elif expert_action < -1e-6:
                discharge_steps += 1
            else:
                idle_steps += 1

            state, _, done, _ = _step_env(env, np.asarray([expert_action], dtype=np.float32))

    stats = {
        'samples': float(len(states)),
        'charge_ratio': charge_steps / max(1, len(states)),
        'discharge_ratio': discharge_steps / max(1, len(states)),
        'idle_ratio': idle_steps / max(1, len(states)),
    }
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32), stats


def train_behavior_cloning(agent, states: np.ndarray, actions: np.ndarray, epochs: int, batch_size: int) -> List[float]:
    device = agent.device
    actor = agent.actor
    optimizer = agent.actor_optimizer

    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    y = torch.as_tensor(actions, dtype=torch.float32, device=device)

    losses: List[float] = []
    n = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses: List[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x[idx]
            yb = y[idx]

            pred_mean, pred_logstd = actor(xb)
            loss_mean = F.mse_loss(pred_mean, yb)
            loss_std = 1e-3 * torch.mean(pred_logstd.pow(2))
            loss = loss_mean + loss_std

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        print(f"[BC] epoch {epoch + 1}/{epochs} loss={losses[-1]:.6f}")

    return losses


def evaluate_actor_alignment(agent, states: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    device = agent.device
    with torch.no_grad():
        x = torch.as_tensor(states, dtype=torch.float32, device=device)
        y = torch.as_tensor(actions, dtype=torch.float32, device=device)
        pred_mean, _ = agent.actor(x)
        mae = torch.mean(torch.abs(pred_mean - y)).item()
        mse = F.mse_loss(pred_mean, y).item()
    return {'bc_mae': float(mae), 'bc_mse': float(mse)}


def main():
    parser = argparse.ArgumentParser(description='Pretrain SAC actor from rule expert demonstrations')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--demo-episodes', type=int, default=80, help='Episodes to collect from rule expert')
    parser.add_argument('--bc-epochs', type=int, default=30, help='Behavior cloning epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='BC batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output checkpoint path')
    parser.add_argument(
        '--expert-mode',
        type=str,
        default='conservative_tou',
        choices=['conservative_tou', 'aggressive_discharge'],
        help='Rule expert mode for collecting demonstrations',
    )
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = create_environment(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device_config = config['device']
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    print(f"Using device: {device}")

    agent = create_agent(config, state_dim, action_dim, device)
    states, actions, demo_stats = collect_rule_dataset(env, args.demo_episodes, args.expert_mode)
    print(f"Collected {len(states)} demo samples")
    print(
        "Rule ratios:",
        f"charge={demo_stats['charge_ratio']:.3f}",
        f"discharge={demo_stats['discharge_ratio']:.3f}",
        f"idle={demo_stats['idle_ratio']:.3f}",
    )
    print(f"Expert mode: {args.expert_mode}")

    losses = train_behavior_cloning(agent, states, actions, args.bc_epochs, args.batch_size)
    fit_stats = evaluate_actor_alignment(agent, states, actions)
    print(f"BC fit: mae={fit_stats['bc_mae']:.6f} mse={fit_stats['bc_mse']:.6f}")

    output_path = args.output
    if not output_path:
        config_stem = os.path.splitext(os.path.basename(args.config))[0]
        output_path = os.path.join('..', 'experiments', f'{config_stem}_rule_bc_actor.pt')

    ckpt = {
        'actor': agent.actor.state_dict(),
        'meta': {
            'config': args.config,
            'demo_episodes': int(args.demo_episodes),
            'bc_epochs': int(args.bc_epochs),
            'expert_mode': args.expert_mode,
            'samples': int(len(states)),
            'final_bc_loss': float(losses[-1]) if losses else 0.0,
            **demo_stats,
            **fit_stats,
        }
    }
    torch.save(ckpt, output_path)
    print(f"Saved actor warm-start to: {output_path}")


if __name__ == '__main__':
    main()

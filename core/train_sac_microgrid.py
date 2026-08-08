import sys
import os
import csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'safe_fl_microgrid'))

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sac_agent import SACAgent
from microgrid_env import create_microgrid_env, MicrogridEnvironment
from experiment_manager import ExperimentManager, create_experiment_from_config
from compute_resources import collect_compute_resources, format_compute_resources
import time
from typing import List, Dict, Any
import argparse
from safety_net import project as safety_project, update_conformal_residual, set_conformal_params, clear_residual_buffer, get_residual_count


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
class EpisodeCSVLogger:
    """Documentation for this public API is provided in English."""

    HEADER = [
        'episode', 'wall_sec', 'ep_reward', 'ep_length',
        'soc_min', 'soc_max', 'soc_mean', 'soc_end',
        'violations_realized', 'violations_attempted', 'safety_projected',
        'strict_soc_violation_steps', 'strict_soc_violation_duration_h',
        'strict_soc_violation_kwh', 'strict_soc_violation_max_kwh',
        'safety_projected_meaningful', 'projection_delta_mean_w', 'projection_delta_max_w',
        'action_mean_abs', 'action_raw_mean', 'action_safe_mean',
        'flow_action_mean', 'flow_active_mean', 'flow_power_limited_count', 'pump_power_wh',
        'flow_too_low_active_count', 'flow_power_mismatch_count',
        'revenue', 'cost', 'net_profit',
        'buffer_size', 'conformal_delta', 'proj_penalty_mult',
        'alpha',
    ]

    def __init__(self, path: str):
        self.path = path
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(self.HEADER)

    def log(self, row: Dict[str, Any]):
        with open(self.path, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow([row.get(k, '') for k in self.HEADER])


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    repo_root = os.path.dirname(config_dir)

    env_cfg = config.get('env', {})
    dataset_path = env_cfg.get('dataset_csv_path')
    if isinstance(dataset_path, str) and dataset_path and not os.path.isabs(dataset_path):
        candidates = [
            os.path.abspath(dataset_path),
            os.path.abspath(os.path.join(config_dir, dataset_path)),
            os.path.abspath(os.path.join(repo_root, dataset_path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                env_cfg['dataset_csv_path'] = candidate
                break

    return config


def warmstart_actor_flexible(agent: SACAgent, actor_state: Dict[str, Any]) -> Dict[str, int]:
    """Load matching actor weights, allowing a 1D power policy to seed a 2D flow actor."""
    current = agent.actor.state_dict()
    loaded = 0
    partial = 0
    skipped = 0
    with torch.no_grad():
        for name, src in actor_state.items():
            if name not in current:
                skipped += 1
                continue
            dst = current[name]
            src = src.to(dst.device)
            if tuple(src.shape) == tuple(dst.shape):
                dst.copy_(src)
                loaded += 1
                continue
            if src.ndim == dst.ndim and all(src.shape[i] == dst.shape[i] for i in range(src.ndim - 1)):
                n = min(src.shape[-1], dst.shape[-1])
                if src.ndim == 1:
                    dst[:n].copy_(src[:n])
                else:
                    dst[..., :n].copy_(src[..., :n])
                partial += 1
                continue
            skipped += 1
    agent.actor.load_state_dict(current)
    return {"loaded": loaded, "partial": partial, "skipped": skipped}


def get_power_limits(env) -> tuple[float, float]:
    charge_kw = float(getattr(env, 'battery_charge_power_kw', getattr(env, 'battery_power_kw', 0.0)))
    discharge_kw = float(getattr(env, 'battery_discharge_power_kw', getattr(env, 'battery_power_kw', 0.0)))
    return charge_kw, discharge_kw


def norm_to_power_kw(power_norm: float, env) -> float:
    charge_kw, discharge_kw = get_power_limits(env)
    return float(power_norm) * (charge_kw if float(power_norm) >= 0.0 else discharge_kw)


def power_kw_to_norm(power_kw: float, env) -> float:
    charge_kw, discharge_kw = get_power_limits(env)
    if float(power_kw) >= 0.0:
        denom = max(charge_kw, 1e-9)
    else:
        denom = max(discharge_kw, 1e-9)
    return float(np.clip(float(power_kw) / denom, -1.0, 1.0))


def compute_occ_proxy(
    config: Dict[str, Any],
    env,
    state: np.ndarray,
    action_raw_kw: float,
    action_safe_kw: float,
    delta_kw: float,
    pmax: float,
) -> float:
    """Compute OCC proxy for actor shaping.

    Default behavior matches the legacy implementation:
      occ_proxy = delta_kw / Pmax

    When ``occ_proxy.mode == 'boundary_aware'``, we augment the proxy with
    soft boundary-risk terms so the actor can learn to avoid low-SoC discharge
    and high-SoC charge tendencies before hitting hard guards.
    """
    occ_cfg = config.get('occ_proxy', {}) or {}
    pmax_safe = max(float(pmax), 1e-9)
    projection_term = float(abs(delta_kw)) / pmax_safe

    mode = str(occ_cfg.get('mode', 'projection_only')).strip().lower()
    if mode != 'boundary_aware':
        return projection_term

    soc = float(state[0])
    soc_min = float(getattr(env, 'soc_min', 0.10))
    soc_max = float(getattr(env, 'soc_max', 0.90))
    low_thr = float(occ_cfg.get('low_soc_threshold', soc_min))
    high_thr = float(occ_cfg.get('high_soc_threshold', soc_max))
    low_thr = float(np.clip(low_thr, soc_min, soc_max))
    high_thr = float(np.clip(high_thr, soc_min, soc_max))

    # Keep thresholds ordered even if config is malformed.
    if high_thr <= low_thr:
        mid = 0.5 * (soc_min + soc_max)
        low_thr = min(low_thr, mid)
        high_thr = max(high_thr, mid)

    action_deadband_kw = float(
        occ_cfg.get(
            'action_deadband_kw',
            getattr(env, 'action_dead_zone_kw', 0.0),
        )
    )
    is_discharge = float(action_raw_kw) < -abs(action_deadband_kw)
    is_charge = float(action_raw_kw) > abs(action_deadband_kw)

    low_span = max(low_thr - soc_min, 1e-6)
    high_span = max(soc_max - high_thr, 1e-6)
    low_soc_risk = max(0.0, (low_thr - soc) / low_span) if is_discharge else 0.0
    high_soc_risk = max(0.0, (soc - high_thr) / high_span) if is_charge else 0.0

    delta_weight = float(occ_cfg.get('delta_weight', 1.0))
    low_soc_weight = float(occ_cfg.get('low_soc_weight', 0.0))
    high_soc_weight = float(occ_cfg.get('high_soc_weight', 0.0))
    clamp_max = float(occ_cfg.get('clamp_max', 2.0))

    occ_proxy = (
        delta_weight * projection_term
        + low_soc_weight * low_soc_risk
        + high_soc_weight * high_soc_risk
    )
    return float(np.clip(occ_proxy, 0.0, clamp_max))


def create_agent(config: Dict[str, Any], state_dim: int, action_dim: int, device: str) -> SACAgent:
    """Create SAC agent from configuration"""
    sac_config = config['sac']
    training_cfg = config.get('training', {})
    variant = training_cfg.get('variant', 'sac')
    evidential_enabled = variant == 'sac_sn_evi'
    lambda_evi = float(training_cfg.get('lambda_evi', 1e-3))
    beta_risk = float(training_cfg.get('beta_risk', 0.5))
    target_entropy = sac_config.get('target_entropy', -1.0)
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr_actor=sac_config['actor_lr'],
        lr_critic=sac_config['critic_lr'],
        lr_alpha=float(sac_config.get('alpha_lr', 1e-4)),
        gamma=sac_config['gamma'],
        tau=sac_config['tau'],
        alpha=sac_config['alpha'],
        target_entropy=target_entropy,
        hidden_dim=sac_config['hidden_dim'],
        buffer_size=sac_config['buffer_size'],
        batch_size=sac_config['batch_size'],
        evidential_enabled=evidential_enabled,
        lambda_evi=lambda_evi,
        beta_risk=beta_risk,
        beta_occ=float(sac_config.get('beta_occ', 0.3)),
        actor_update_interval=int(sac_config.get('actor_update_interval', 1)),
        alpha_update_interval=int(sac_config.get('alpha_update_interval', 1)),
        actor_warmup_updates=int(sac_config.get('actor_warmup_updates', 0)),
        freeze_alpha=bool(sac_config.get('freeze_alpha', False)),
    )
    
    return agent


def create_environment(config: Dict[str, Any]) -> MicrogridEnvironment:
    """Create microgrid environment from configuration"""
    env_config = config['env']
    safetynet_cfg = config.get('safetynet', {})
    stress_cfg = config.get('stress', {})
    
    stress_kwargs = {}
    if stress_cfg:
        stress_kwargs = {
            'stress_enable': bool(stress_cfg.get('enable', False)),
            'stress_efficiency_noise_std': float(stress_cfg.get('efficiency_noise_std', 0.0)),
            'stress_dt_jitter_std': float(stress_cfg.get('dt_jitter_std', 0.0)),
            'stress_action_lag_alpha': float(stress_cfg.get('action_lag_alpha', 0.0)),
            'stress_soc_obs_delay': int(stress_cfg.get('soc_obs_delay', 0)),
            'stress_soc_obs_noise_std': float(stress_cfg.get('soc_obs_noise_std', 0.0)),
            'stress_bounds_drift_std': float(stress_cfg.get('bounds_drift_std', 0.0)),
            'stress_external_pmax_shrink_prob': float(stress_cfg.get('external_pmax_shrink_prob', 0.0)),
            'stress_external_pmax_shrink_factor': float(stress_cfg.get('external_pmax_shrink_factor', 1.0)),
            'stress_power_loss_ratio': float(stress_cfg.get('power_loss_ratio', 0.0)),
            'stress_battery_response_noise_std': float(stress_cfg.get('battery_response_noise_std', 0.0)),
            'stress_battery_zero_response_prob': float(stress_cfg.get('battery_zero_response_prob', 0.0)),
        }

    env = create_microgrid_env(
        microgrid_id=env_config.get('microgrid_id', 0),
        episode_length=env_config['episode_length'],
        battery_capacity_kwh=env_config['battery_capacity_kwh'],
        battery_power_kw=env_config['battery_power_kw'],
        battery_charge_power_kw=env_config.get('battery_charge_power_kw', None),
        battery_discharge_power_kw=env_config.get('battery_discharge_power_kw', None),
        soc_min=env_config.get('soc_min', 0.1),
        soc_max=env_config.get('soc_max', 0.9),
        use_real_data=env_config['use_real_data'],
        time_step=env_config.get('time_step', 1.0),
        ramp_limit_kw=env_config.get('ramp_limit_kw', None),
        hard_guard=env_config.get('hard_guard', False),
        clip_soc_to_bounds=env_config.get('clip_soc_to_bounds', True),
        dataset_csv_path=env_config.get('dataset_csv_path', None),
        dataset_pv_join_wind=env_config.get('dataset_pv_join_wind', False),
        train_window_hours=env_config.get('train_window_hours', None),
        dataset_pv_column=env_config.get('dataset_pv_column', None),
        dataset_load_kw=env_config.get('dataset_load_kw', None),
        dataset_power_scale=env_config.get('dataset_power_scale', 1.0),
        dataset_time_column=env_config.get('dataset_time_column', None),
        use_dataset_timestamps_for_obs=env_config.get('use_dataset_timestamps_for_obs', True),
        deployment_observation_style=env_config.get('deployment_observation_style', False),
        deployment_window_steps=env_config.get('deployment_window_steps', 1),
        deployment_load_threshold_kw=env_config.get('deployment_load_threshold_kw', 0.0005),
        deployment_group_power_kw=env_config.get('deployment_group_power_kw', 0.0001),
        battery_delivered_load_per_group_kw=env_config.get('battery_delivered_load_per_group_kw', None),
        continuous_operation_mode=env_config.get('continuous_operation_mode', False),
        deployment_guard_style=env_config.get('deployment_guard_style', False),
        enforce_solo_discharge_load_limit=env_config.get('enforce_solo_discharge_load_limit', True),
        pre_measure_rest_flow_fraction=env_config.get('pre_measure_rest_flow_fraction', 0.0),
        pre_measure_flow_fraction=env_config.get('pre_measure_flow_fraction', 0.0),
        pre_measure_seconds=env_config.get('pre_measure_seconds', 0.0),
        synthetic_hourly_hold=env_config.get('synthetic_hourly_hold', False),
        synthetic_pv_peak_kw=env_config.get('synthetic_pv_peak_kw', 20.0),
        synthetic_pv_start_hour=env_config.get('synthetic_pv_start_hour', 6),
        synthetic_pv_end_hour=env_config.get('synthetic_pv_end_hour', 18),
        synthetic_load_base_kw=env_config.get('synthetic_load_base_kw', 10.0),
        synthetic_load_amp_kw=env_config.get('synthetic_load_amp_kw', 5.0),
        synthetic_price_base=env_config.get('synthetic_price_base', 0.12),
        synthetic_price_peak=env_config.get('synthetic_price_peak', 0.20),
        synthetic_price_peak_start=env_config.get('synthetic_price_peak_start', 8),
        synthetic_price_peak_end=env_config.get('synthetic_price_peak_end', 18),
        allow_grid_trading=env_config.get('allow_grid_trading', True),
        use_flow_rate_action=env_config.get('use_flow_rate_action', False),
        fixed_flow_fraction_when_uncontrolled=env_config.get(
            'fixed_flow_fraction_when_uncontrolled', 0.0
        ),
        flow_R_base_ohm=env_config.get('flow_R_base_ohm', 72.5),
        flow_P_max_pump_W=env_config.get('flow_P_max_pump_W', 0.0168),
        flow_k_R=env_config.get('flow_k_R', 0.5),
        flow_V_OCV_charge=env_config.get('flow_V_OCV_charge', 8.5),
        flow_V_OCV_discharge=env_config.get('flow_V_OCV_discharge', 5.5),
        flow_I_rated_A=env_config.get('flow_I_rated_A', 0.020),
        flow_min_active_fraction=env_config.get('flow_min_active_fraction', 0.01),
        flow_idle_fraction=env_config.get('flow_idle_fraction', 0.0),
        flow_pump_from_grid=env_config.get('flow_pump_from_grid', False),
        flow_charge_pump_free=env_config.get('flow_charge_pump_free', False),
        flow_limits_available_power=env_config.get('flow_limits_available_power', False),
        flow_power_min_fraction=env_config.get('flow_power_min_fraction', 0.0),
        flow_operating_rule_enabled=env_config.get('flow_operating_rule_enabled', True),
        battery_efficiency=env_config.get('battery_efficiency', 0.95),
        use_extended_obs=env_config.get('use_extended_obs', False),
        # ── TOU Reward Scale ─────────────────────────────────
        tou_reward_scale=env_config.get('tou_reward_scale', 3000.0),
        allow_grid_export=env_config.get('allow_grid_export', False),
        feed_in_tariff_ratio=env_config.get('feed_in_tariff_ratio', 0.5),
        discharge_auto=env_config.get('discharge_auto', False),
        discharge_mode=env_config.get('discharge_mode', 'solo_only'),
        voltage_cutoff_soc=env_config.get('voltage_cutoff_soc', 0.0),
        # ── v14: dead zone + reward version ────────────────────
        action_dead_zone_kw=env_config.get('action_dead_zone_kw', 0.0),
        discharge_intent_threshold_kw=env_config.get('discharge_intent_threshold_kw', 0.0),
        reward_version=env_config.get('reward_version', 'p302'),
        pv_obs_boolean=env_config.get('pv_obs_boolean', False),
        pv_obs_boolean_threshold_kw=env_config.get('pv_obs_boolean_threshold_kw', 0.001),
        pv_support_ratio_obs=env_config.get('pv_support_ratio_obs', False),
        pv_support_ratio_max=env_config.get('pv_support_ratio_max', 1.5),
        price_obs=env_config.get('price_obs', True),
        tou_onehot_obs=env_config.get('tou_onehot_obs', False),
        charge_requires_pv_surplus=env_config.get('charge_requires_pv_surplus', False),
        charge_limit_to_pv_surplus=env_config.get('charge_limit_to_pv_surplus', False),
        **stress_kwargs
    )
    try:
        _pfloor = env_config.get('soc_physical_floor', None)
        env.soc_physical_floor = None if _pfloor is None else float(_pfloor)
    except Exception:
        env.soc_physical_floor = None
    try:
        env.safetynet_ramp_kw = config.get('safetynet', {}).get('ramp_limit_kw', None)
    except Exception:
        env.safetynet_ramp_kw = None
    reward_cfg = config.get('reward', {})
    try:
        env.realized_violation_penalty = float(reward_cfg.get('realized_violation_penalty', 20.0))
    except Exception:
        env.realized_violation_penalty = 20.0
    try:
        env.blocked_by_pv_penalty = float(reward_cfg.get('blocked_by_pv_penalty', 0.10))
    except Exception:
        env.blocked_by_pv_penalty = 0.10
    try:
        env.blocked_by_load_penalty = float(reward_cfg.get('blocked_by_load_penalty', 0.05))
    except Exception:
        env.blocked_by_load_penalty = 0.05
    try:
        env.solar_storage_value_scale = float(reward_cfg.get('solar_storage_value_scale', 1.0))
    except Exception:
        env.solar_storage_value_scale = 1.0
    try:
        env.solar_storage_value_price = float(reward_cfg.get('solar_storage_value_price', 7.13))
    except Exception:
        env.solar_storage_value_price = 7.13
    try:
        env.offpeak_charge_soc_target = float(reward_cfg.get('offpeak_charge_soc_target', 0.85))
    except Exception:
        env.offpeak_charge_soc_target = 0.85
    try:
        env.peak_discharge_soc_floor = float(reward_cfg.get('peak_discharge_soc_floor', 0.15))
    except Exception:
        env.peak_discharge_soc_floor = 0.15
    try:
        env.v17_offpeak_charge_bonus = float(reward_cfg.get('v17_offpeak_charge_bonus', 0.6))
    except Exception:
        env.v17_offpeak_charge_bonus = 0.6
    try:
        env.v17_peak_discharge_bonus = float(reward_cfg.get('v17_peak_discharge_bonus', 1.2))
    except Exception:
        env.v17_peak_discharge_bonus = 1.2
    try:
        env.v17_peak_idle_penalty = float(reward_cfg.get('v17_peak_idle_penalty', 0.4))
    except Exception:
        env.v17_peak_idle_penalty = 0.4
    try:
        env.v17_solar_storage_bonus_scale = float(reward_cfg.get('v17_solar_storage_bonus_scale', 0.3))
    except Exception:
        env.v17_solar_storage_bonus_scale = 0.3
    try:
        env.no_pv_action_threshold_kw = float(reward_cfg.get('no_pv_action_threshold_kw', 0.001))
    except Exception:
        env.no_pv_action_threshold_kw = 0.001
    try:
        env.no_pv_throughput_penalty_per_kwh = float(reward_cfg.get('no_pv_throughput_penalty_per_kwh', 0.0))
    except Exception:
        env.no_pv_throughput_penalty_per_kwh = 0.0
    try:
        env.offpeak_no_pv_discharge_penalty_per_kwh = float(
            reward_cfg.get('offpeak_no_pv_discharge_penalty_per_kwh', 0.0)
        )
    except Exception:
        env.offpeak_no_pv_discharge_penalty_per_kwh = 0.0
    # v13: inject charge/discharge shaping params
    charge_cfg = config.get('charge_shaping', {})
    if charge_cfg:
        env._solar_charge_coeff = float(charge_cfg.get('solar_charge_coeff', 0.8))
        env._solar_waste_penalty = float(charge_cfg.get('solar_waste_penalty', 0.4))
        env._power_util_coeff = float(charge_cfg.get('power_util_coeff', 0.0))
    env._soc_init_mode = config.get('env', {}).get('soc_init_mode', 'balanced')
    
    return env


def get_safetynet_soc_bounds(config: Dict[str, Any], env: MicrogridEnvironment) -> tuple[float, float]:
    """Return possibly conservative SoC bounds used by SafetyNet projection."""
    low = float(getattr(env, 'soc_min_eff', getattr(env, 'soc_min', 0.0)))
    high = float(getattr(env, 'soc_max_eff', getattr(env, 'soc_max', 1.0)))
    margin = float(config.get('safetynet', {}).get('soc_margin', 0.0) or 0.0)
    if margin > 0.0 and high - low > 2.0 * margin:
        return low + margin, high - margin
    return low, high


def _price_is_close(price: float, target: float, atol: float = 1e-6) -> bool:
    return bool(abs(float(price) - float(target)) <= atol)


def guided_teacher_action_kw(env, teacher_cfg: Dict[str, Any]) -> float:
    """Heuristic teacher aligned with the deployment physics.

    Main policy:
      1. Charge only when PV can cover the load and still leaves usable surplus.
      2. Discharge only during expensive hours, without PV, and only if battery can solo-cover load.
      3. Otherwise keep the battery idle, except for an optional emergency off-peak top-up.
    """
    step = int(getattr(env, 'current_step', 0))
    soc = float(getattr(env, 'current_soc', 0.0))
    episode_data = getattr(env, 'episode_data', None) or {}

    load_series = episode_data.get('load', None)
    pv_series = episode_data.get('pv', None)
    price_series = episode_data.get('price', None)
    pv_bool_series = episode_data.get('pv_bool', None)

    load_kw = float(load_series[step]) if load_series is not None and step < len(load_series) else 0.0
    pv_kw = float(pv_series[step]) if pv_series is not None and step < len(pv_series) else 0.0
    price = float(price_series[step]) if price_series is not None and step < len(price_series) else 2.06
    if pv_bool_series is not None and step < len(pv_bool_series):
        pv_sufficient = bool(float(pv_bool_series[step]) > 0.5)
    else:
        pv_sufficient = bool(
            float(pv_kw) / max(float(load_kw), 1e-9)
            >= float(getattr(env, 'pv_sufficient_ratio_threshold', 0.8))
        )
    pv_present = bool(pv_kw > float(getattr(env, 'pv_obs_boolean_threshold_kw', 0.001)))

    battery_charge_power_kw = float(getattr(env, 'battery_charge_power_kw', getattr(env, 'battery_power_kw', 0.0)))
    battery_discharge_power_kw = float(getattr(env, 'battery_discharge_power_kw', getattr(env, 'battery_power_kw', 0.0)))

    solar_charge_soc_target = float(teacher_cfg.get('solar_charge_soc_target', 0.85))
    peak_discharge_soc_floor = float(teacher_cfg.get('peak_discharge_soc_floor', 0.20))
    emergency_grid_charge_soc = float(teacher_cfg.get('emergency_grid_charge_soc', 0.0))
    emergency_grid_charge_target = float(teacher_cfg.get('emergency_grid_charge_target', 0.12))
    pv_surplus_threshold_kw = float(teacher_cfg.get('pv_surplus_threshold_kw', 0.0002))
    emergency_charge_ratio = float(teacher_cfg.get('emergency_charge_ratio', 0.25))
    pv_cover_charge_threshold = float(teacher_cfg.get('pv_cover_charge_threshold', 0.95))
    pv_cover_charge_min_frac = float(teacher_cfg.get('pv_cover_charge_min_frac', 0.35))
    pv_bool_charge_min_frac = float(teacher_cfg.get('pv_bool_charge_min_frac', 0.22))
    pre_solar_discharge_enabled = bool(teacher_cfg.get('pre_solar_discharge_enabled', False))
    pre_solar_start_hour = int(teacher_cfg.get('pre_solar_start_hour', 5))
    pre_solar_end_hour = int(teacher_cfg.get('pre_solar_end_hour', 8))
    pre_solar_soc_floor = float(teacher_cfg.get('pre_solar_soc_floor', 0.55))

    is_peak = _price_is_close(price, float(teacher_cfg.get('peak_price', 7.13)))
    is_offpeak = _price_is_close(price, float(teacher_cfg.get('offpeak_price', 2.06)))
    try:
        current_hour, _ = env._get_obs_hour_dow(step)
    except Exception:
        steps_per_hour = max(1, int(round(1.0 / max(float(getattr(env, 'time_step', 0.25)), 1e-9))))
        current_hour = int((step // steps_per_hour) % 24)

    pv_surplus_kw = max(0.0, pv_kw - load_kw)
    pv_support_ratio = float(np.clip(pv_kw / max(load_kw, 1e-9), 0.0, 1.5))
    soc_deficit_frac = float(np.clip((solar_charge_soc_target - soc) / max(solar_charge_soc_target, 1e-9), 0.0, 1.0))
    if pv_sufficient and pv_surplus_kw > pv_surplus_threshold_kw and soc < solar_charge_soc_target:
        return float(min(battery_charge_power_kw, pv_surplus_kw))

    if pv_support_ratio >= pv_cover_charge_threshold and soc < solar_charge_soc_target:
        cover_span = max(1e-9, 1.0 - pv_cover_charge_threshold)
        cover_strength = float(np.clip((pv_support_ratio - pv_cover_charge_threshold) / cover_span, 0.0, 1.0))
        charge_frac = max(pv_cover_charge_min_frac, cover_strength)
        return float(battery_charge_power_kw * charge_frac * soc_deficit_frac)

    if pv_sufficient and soc < solar_charge_soc_target:
        support_strength = float(np.clip(pv_support_ratio / max(pv_cover_charge_threshold, 1e-9), 0.0, 1.0))
        charge_frac = max(pv_bool_charge_min_frac, support_strength * pv_cover_charge_min_frac)
        return float(battery_charge_power_kw * charge_frac * soc_deficit_frac)

    can_solo_discharge = load_kw <= battery_discharge_power_kw + 1e-9
    if (not pv_present) and is_peak and soc > peak_discharge_soc_floor and can_solo_discharge:
        return float(-min(load_kw, battery_discharge_power_kw))

    if (
        pre_solar_discharge_enabled
        and pre_solar_start_hour <= current_hour < pre_solar_end_hour
        and (not pv_present)
        and soc > pre_solar_soc_floor
        and can_solo_discharge
    ):
        return float(-min(load_kw, battery_discharge_power_kw))

    if (
        emergency_grid_charge_soc > 0.0
        and emergency_grid_charge_target > emergency_grid_charge_soc
        and is_offpeak
        and (not pv_present)
        and soc < emergency_grid_charge_soc
    ):
        deficit_frac = np.clip(
            (emergency_grid_charge_target - soc) / max(emergency_grid_charge_target, 1e-9),
            0.0,
            1.0,
        )
        return float(battery_charge_power_kw * emergency_charge_ratio * deficit_frac)

    return 0.0


def teacher_action_to_actor_vector(action_kw: float, env, action_dim: int) -> np.ndarray:
    charge_limit_kw, discharge_limit_kw = get_power_limits(env)
    if action_kw >= 0.0:
        action_norm = action_kw / max(charge_limit_kw, 1e-9)
    else:
        action_norm = action_kw / max(discharge_limit_kw, 1e-9)
    action_norm = float(np.clip(action_norm, -1.0, 1.0))

    if action_dim <= 1:
        return np.array([action_norm], dtype=np.float32)

    if abs(action_kw) <= 1e-12:
        flow_norm = -1.0
    else:
        if action_kw >= 0.0:
            power_frac = abs(action_kw) / max(charge_limit_kw, 1e-9)
        else:
            power_frac = abs(action_kw) / max(discharge_limit_kw, 1e-9)
        flow_fraction = 0.60 + 0.40 * float(np.clip(power_frac, 0.0, 1.0))
        flow_norm = 2.0 * flow_fraction - 1.0
    return np.array([action_norm, float(np.clip(flow_norm, -1.0, 1.0))], dtype=np.float32)


def collect_guided_teacher_demos(
    env: MicrogridEnvironment,
    guided_cfg: Dict[str, Any],
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    demo_episodes = int(guided_cfg.get('demo_episodes', 80))
    max_steps = int(guided_cfg.get('demo_max_steps', getattr(env, 'episode_length', 288)))
    action_dim = int(env.action_space.shape[0])

    states: List[np.ndarray] = []
    teacher_actions: List[np.ndarray] = []

    fixed_start_backup = getattr(env, 'fixed_start_idx', None)
    try:
        for ep in range(demo_episodes):
            state, _ = env.reset(seed=random_seed + ep)
            for _ in range(max_steps):
                teacher_kw = guided_teacher_action_kw(env, guided_cfg)
                teacher_vec = teacher_action_to_actor_vector(teacher_kw, env, action_dim)
                states.append(np.asarray(state, dtype=np.float32))
                teacher_actions.append(teacher_vec.astype(np.float32))

                env_action = [float(teacher_kw)]
                if action_dim > 1:
                    env_action.append(0.5)
                next_state, _, terminated, truncated, _ = env.step(env_action)
                state = next_state
                if terminated or truncated:
                    break
    finally:
        env.fixed_start_idx = fixed_start_backup

    return np.asarray(states, dtype=np.float32), np.asarray(teacher_actions, dtype=np.float32)


def guided_behavior_cloning_pretrain(
    env: MicrogridEnvironment,
    agent: SACAgent,
    config: Dict[str, Any],
    exp_manager: ExperimentManager,
) -> Dict[str, Any]:
    guided_cfg = config.get('guided_teacher', {}) or {}
    if not guided_cfg.get('enabled', False):
        return {}

    random_seed = int(config.get('random_seed', 0))
    states_np, actions_np = collect_guided_teacher_demos(env, guided_cfg, random_seed=random_seed)
    if len(states_np) == 0:
        print("Guided teacher demo set is empty; skipping BC pretrain.")
        return {}

    epochs = int(guided_cfg.get('bc_epochs', 20))
    batch_size = int(guided_cfg.get('bc_batch_size', 512))
    bc_lr = float(guided_cfg.get('bc_lr', 3e-4))

    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=bc_lr)
    states = torch.as_tensor(states_np, dtype=torch.float32, device=agent.device)
    targets = torch.as_tensor(actions_np, dtype=torch.float32, device=agent.device)

    final_loss = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(states.shape[0], device=agent.device)
        epoch_losses = []
        for start in range(0, states.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            batch_states = states[idx]
            batch_targets = targets[idx]
            pred_mean, _ = agent.actor(batch_states)
            loss = F.mse_loss(pred_mean, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        final_loss = float(np.mean(epoch_losses)) if epoch_losses else final_loss
        if epoch == 0 or (epoch + 1) == epochs or ((epoch + 1) % max(1, epochs // 5) == 0):
            print(
                f"[GUIDED-BC] epoch {epoch + 1:>3d}/{epochs} | "
                f"samples={states.shape[0]} | loss={final_loss:.6f}"
            )

    if bool(guided_cfg.get('save_actor_checkpoint', True)):
        actor_path = os.path.join(exp_manager.models_dir, "guided_teacher_actor.pth")
        torch.save({'actor': agent.actor.state_dict()}, actor_path)
        print(f"[GUIDED-BC] saved actor checkpoint to: {actor_path}")

    return {
        'demo_samples': int(states.shape[0]),
        'bc_epochs': int(epochs),
        'bc_final_loss': float(final_loss),
    }


def train_sac_with_microgrid(
    env: MicrogridEnvironment,
    agent: SACAgent,
    config: Dict[str, Any],
    exp_manager: ExperimentManager
) -> Dict[str, List[float]]:
    """
    Train SAC agent using microgrid environment
    
    Args:
        env: Microgrid environment
        agent: SAC agent
        config: Training configuration
        exp_manager: Experiment manager for organizing outputs
    
    Returns:
        Dictionary containing training metrics
    """
    
    training_config = config['training']
    logging_config = config['logging']
    
    total_episodes = training_config['total_episodes']
    max_steps = training_config['max_steps']
    update_every = config['sac']['update_every']
    eval_every = training_config['eval_every']
    save_every = training_config['save_every']
    eval_episodes = training_config['eval_episodes']
    log_interval = logging_config['log_interval']
    warmup_steps = config['sac']['warmup_steps']
    variant = config.get('training', {}).get('variant', 'sac')
    variant_needs_sn = variant in ('sac_sn', 'sac_sn_evi')
    variant_penalty_only = variant == 'sac_penalty'

    sn_warmup_eps = int(config.get('training', {}).get('safetynet_warmup_episodes', 0))
    if variant_needs_sn and sn_warmup_eps > 0:
        use_safetynet = False
        curriculum_enabled = True
        curriculum_switched = False
    else:
        use_safetynet = variant_needs_sn
        curriculum_enabled = False
        curriculum_switched = not variant_needs_sn

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_soc_violations = []
    episode_action_violations = []
    episode_actions = []
    actions_raw_series = []
    actions_safe_series = []
    episode_soc_trajectories = []
    episode_revenues = []
    episode_costs = []
    episode_attempted_violations = []
    episode_safety_projected = []
    episode_realized_violations = []
    
    # Evaluation metrics
    eval_rewards = []
    eval_soc_violations = []
    eval_revenues = []
    eval_costs = []
    
    best_eval_reward = float('-inf')
    # Adaptive conformal/penalty parameters
    conformal_cfg = config.get('conformal', {})
    current_conformal_window = int(conformal_cfg.get('window', 2880))
    current_conformal_delta = float(conformal_cfg.get('delta', 0.1))
    projected_penalty_mult = 1.0
    
    # ── Per-episode CSV logger ─────────────────────────────────
    csv_log_path = os.path.join(exp_manager.logs_dir, "episode_log.csv")
    csv_logger = EpisodeCSVLogger(csv_log_path)
    train_start_wall = time.time()

    print(f"\n{'='*70}")
    print(f"  SAC Training — P302 Microgrid Simulation")
    print(f"{'='*70}")
    print(f"  Episodes      : {total_episodes}")
    print(f"  Steps/episode : {max_steps}")
    print(f"  Total steps   : {total_episodes * max_steps:,}")
    print(f"  Variant       : {variant}")
    if curriculum_enabled:
        print(f"  SafetyNet     : CURRICULUM (OFF for EP 0~{sn_warmup_eps-1}, ON from EP {sn_warmup_eps})")
    else:
        print(f"  SafetyNet     : {'ON' if use_safetynet else 'OFF'}")
    print(f"  Device        : {agent.device}")
    print(f"  Warmup        : {warmup_steps} steps")
    print(f"  Eval every    : {eval_every} episodes ({eval_episodes} eps each)")
    print(f"  Log every     : {log_interval} episodes")
    print(f"  Save every    : {save_every} episodes")
    print(f"  Experiment    : {exp_manager.experiment_dir}")
    print(f"  CSV log       : {csv_log_path}")
    print(f"  State dim     : {env.observation_space.shape[0]}")
    print(f"  Action dim    : {env.action_space.shape[0]}")
    print(f"{'='*70}\n")
    
    for episode in range(total_episodes):
        if curriculum_enabled and not curriculum_switched and episode >= sn_warmup_eps:
            use_safetynet = True
            curriculum_switched = True
            try:
                clear_residual_buffer()
                set_conformal_params(
                    window=current_conformal_window,
                    delta=float(conformal_cfg.get('delta', 0.1))
                )
            except Exception:
                pass
            current_conformal_delta = float(conformal_cfg.get('delta', 0.1))
            projected_penalty_mult = 1.0
            print(f"\n{'*'*70}")
            print(f"  ★ CURRICULUM SWITCH @ Episode {episode}")
            print(f"    Phase 1 (Pure SAC) → Phase 2 (SAC + CRTSN + OCC + Adaptive)")
            print(f"    SafetyNet: ON | Conformal buffer: cleared")
            print(f"    Conformal delta: {current_conformal_delta:.3f}")
            print(f"{'*'*70}\n")

        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        soc_violations = 0
        action_violations = 0
        actions = []  # legacy avg (will use normalized)
        actions_raw = []       # normalized [-1,1]
        actions_safe = []      # normalized [-1,1]
        actions_safe_kw = []
        soc_trajectory = [state[0]]  # First element is SoC
        episode_revenue = 0.0
        episode_cost = 0.0
        # Scenario (situation code) distribution per episode
        sit_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        attempted_count = 0
        projected_count = 0
        projected_meaningful_count = 0
        projection_deltas_kw = []
        realized_count = 0
        strict_soc_violation_steps = 0
        strict_soc_violation_duration_h = 0.0
        strict_soc_violation_kwh = 0.0
        strict_soc_violation_max_kwh = 0.0
        flow_actions = []
        flow_active_actions = []
        flow_power_limited_count = 0
        flow_too_low_active_count = 0
        flow_power_mismatch_count = 0
        pump_power_kwh = 0.0
        prev_soc_violations_cum = 0
        prev_action_violations_cum = 0

        use_flow = bool(getattr(env, 'use_flow_rate_action', False))
        action_dim = 2 if use_flow else 1

        for step in range(max_steps):
            # Select action
            action_norm = agent.select_action(state, evaluate=False)  # [-1, 1]^action_dim
            action_norm_val = float(action_norm[0])
            actions.append(action_norm_val)
            actions_raw.append(action_norm_val)
            charge_limit_kw, discharge_limit_kw = get_power_limits(env)
            power_scale = max(charge_limit_kw, discharge_limit_kw)
            a_raw_kw = norm_to_power_kw(action_norm_val, env)

            if use_flow and len(action_norm) >= 2:
                flow_norm_raw = float(action_norm[1])
                flow_fraction = float(np.clip((flow_norm_raw + 1.0) * 0.5, 0.0, 1.0))
            else:
                flow_norm_raw = 0.0
                flow_fraction = float(getattr(env, 'flow_idle_fraction', 0.0))
            
            try:
                current_soc = float(state[0])
                soc_next_raw = float(env.predict_soc_raw(current_soc, a_raw_kw)) if hasattr(env, 'predict_soc_raw') else None
                soc_min = float(getattr(env, 'soc_min_eff', getattr(env, 'soc_min', 0.0)))
                soc_max = float(getattr(env, 'soc_max_eff', getattr(env, 'soc_max', 1.0)))
                attempted = int(soc_next_raw is not None and (soc_next_raw < soc_min or soc_next_raw > soc_max))
            except Exception:
                attempted = 0
            attempted_count += attempted

            pmax = power_scale
            if use_safetynet:
                ramp_kw = getattr(env, 'safetynet_ramp_kw', None)
                soc_bounds = get_safetynet_soc_bounds(config, env)
                prev_a_kw = float(actions_safe_kw[-1]) if actions_safe_kw else 0.0
                a_safe_kw, did_project, delta_kw = safety_project(
                    state=state,
                    action=np.array([a_raw_kw], dtype=np.float32),
                    prev_action=prev_a_kw,
                    pmax=pmax,
                    pmin=discharge_limit_kw,
                    pmax_positive=charge_limit_kw,
                    ramp_kw=ramp_kw,
                    soc_bounds=soc_bounds,
                    env=env,
                )
                safety_projected = int(did_project)
                projection_event_threshold_kw = float(
                    config.get('reward', {}).get(
                        'safety_projection_event_threshold_kw',
                        getattr(env, 'action_dead_zone_kw', 0.0),
                    )
                )
                meaningful_projection = int(safety_projected and float(delta_kw) > projection_event_threshold_kw)
                projected_count += safety_projected
                projected_meaningful_count += meaningful_projection
                if safety_projected:
                    projection_deltas_kw.append(float(delta_kw))
                a_safe_norm = power_kw_to_norm(a_safe_kw, env)
                actions_safe.append(a_safe_norm)
                actions_safe_kw.append(a_safe_kw)
            else:
                delta_kw = 0.0
                safety_projected = 0
                a_safe_kw = a_raw_kw
                a_safe_norm = action_norm_val
                actions_safe.append(a_safe_norm)
                actions_safe_kw.append(a_safe_kw)

            soc_pred_next = None
            try:
                if hasattr(env, 'predict_soc_raw'):
                    soc_pred_next = float(env.predict_soc_raw(float(state[0]), float(a_safe_kw)))
            except Exception:
                soc_pred_next = None

            if use_flow:
                env_action = [a_safe_kw, flow_fraction]
            else:
                env_action = [a_safe_kw]
            next_state, reward, terminated, truncated, step_info = env.step(env_action)
            flow_step = float(step_info.get('flow_action', 0.0))
            flow_actions.append(flow_step)
            if flow_step > 1e-9:
                flow_active_actions.append(flow_step)
            flow_power_limited_count += int(step_info.get('flow_power_limited', 0))
            flow_too_low_active_count += int(step_info.get('flow_too_low_active', 0))
            flow_power_mismatch_count += int(step_info.get('flow_power_mismatch', 0))
            pump_power_kwh += float(step_info.get('pump_power_kw', 0.0)) * float(getattr(env, 'time_step', 0.25))
            # Apply safety shaping only for SafetyNet variants, and scale consistently with env reward
            # Note: env reward already multiplied by reward_scaling; we scale penalties by the same factor
            scale_guard = max(float(getattr(env, 'reward_scaling', 1.0)), 1e-9)
            if use_safetynet or variant_penalty_only:
                attempted_penalty_val = float(config.get('reward', {}).get('attempted_violation_penalty', 0.1)) if attempted else 0.0
                pmax_guard = max(pmax, 1e-9)
                proj_unit = float(config.get('reward', {}).get('safety_projection_penalty', 0.001))
                projected_penalty_val = (
                    projected_penalty_mult * proj_unit * (float(delta_kw) / pmax_guard)
                    if use_safetynet and safety_projected
                    else 0.0
                )
                reward -= scale_guard * (attempted_penalty_val + projected_penalty_val)
            done = terminated or truncated
            
            occ_proxy = compute_occ_proxy(
                config=config,
                env=env,
                state=state,
                action_raw_kw=a_raw_kw,
                action_safe_kw=a_safe_kw,
                delta_kw=delta_kw,
                pmax=pmax,
            )
            if use_flow:
                action_to_store = np.array([a_safe_norm, flow_norm_raw], dtype=np.float32)
                occ_action_to_store = np.array([action_norm_val, flow_norm_raw], dtype=np.float32)
            else:
                action_to_store = np.array([a_safe_norm], dtype=np.float32)
                occ_action_to_store = np.array([action_norm_val], dtype=np.float32)
            agent.store_transition(
                state,
                action_to_store,
                reward,
                next_state,
                done,
                occ_proxy=occ_proxy,
                occ_action=occ_action_to_store,
            )
            
            # Update networks (only after warmup)
            if (step % update_every == 0 and 
                len(agent.replay_buffer) >= warmup_steps and
                len(agent.replay_buffer) >= agent.batch_size):
                update_info = agent.update()
                if episode % log_interval == 0 and step == 0:
                    print(f"Episode {episode}, Step {step}: {update_info}")
            
            # Update episode metrics
            episode_reward += reward
            episode_length += 1
            soc_trajectory.append(next_state[0])
            try:
                if soc_pred_next is not None:
                    update_conformal_residual(float(next_state[0]) - float(soc_pred_next))
            except Exception:
                pass
            
            current_soc_violations_cum = int(step_info.get('soc_violations', 0))
            realized = max(0, current_soc_violations_cum - prev_soc_violations_cum)
            prev_soc_violations_cum = current_soc_violations_cum
            realized_count += realized
            soc_violations += realized
            strict_soc_violation_steps = int(step_info.get('strict_soc_violation_steps', 0))
            strict_soc_violation_duration_h = float(step_info.get('strict_soc_violation_duration_h', 0.0))
            strict_soc_violation_kwh = float(step_info.get('strict_soc_violation_kwh', 0.0))
            strict_soc_violation_max_kwh = float(step_info.get('strict_soc_violation_max_kwh', 0.0))

            current_action_violations_cum = int(step_info.get('action_violations', 0))
            action_violation_step = max(0, current_action_violations_cum - prev_action_violations_cum)
            prev_action_violations_cum = current_action_violations_cum
            action_violations += action_violation_step
 
            # Remove duplicate per-episode penalty application; shaping already applied above when enabled

            episode_revenue = step_info.get('total_revenue', 0)
            episode_cost = step_info.get('total_cost', 0)
            
            # Track situation codes (1-4)
            sit = int(step_info.get('situation_code', 4))
            if sit in sit_counts:
                sit_counts[sit] += 1
            
            if done:
                break
                
            state = next_state
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_soc_violations.append(soc_violations)
        episode_action_violations.append(action_violations)
        episode_actions.append(np.mean(np.abs(actions)))
        actions_raw_series.append(np.mean(np.abs(actions_raw)) if actions_raw else 0.0)
        actions_safe_series.append(np.mean(np.abs(actions_safe)) if actions_safe else 0.0)
        episode_soc_trajectories.append(soc_trajectory)
        episode_revenues.append(episode_revenue)
        episode_costs.append(episode_cost)
        episode_attempted_violations.append(attempted_count)
        episode_safety_projected.append(projected_meaningful_count)
        episode_realized_violations.append(realized_count)
        projection_delta_mean_w = float(np.mean(projection_deltas_kw) * 1000.0) if projection_deltas_kw else 0.0
        projection_delta_max_w = float(np.max(projection_deltas_kw) * 1000.0) if projection_deltas_kw else 0.0
        flow_action_mean = float(np.mean(flow_actions)) if flow_actions else 0.0
        flow_active_mean = float(np.mean(flow_active_actions)) if flow_active_actions else 0.0
        
        soc_arr = np.array(soc_trajectory)
        ep_wall = time.time() - train_start_wall
        try:
            current_alpha = float(agent.log_alpha.exp().item()) if hasattr(agent, 'log_alpha') else 0.0
        except Exception:
            current_alpha = 0.0
        csv_logger.log({
            'episode': episode,
            'wall_sec': f'{ep_wall:.1f}',
            'ep_reward': f'{episode_reward:.4f}',
            'ep_length': episode_length,
            'soc_min': f'{soc_arr.min():.4f}',
            'soc_max': f'{soc_arr.max():.4f}',
            'soc_mean': f'{soc_arr.mean():.4f}',
            'soc_end': f'{soc_arr[-1]:.4f}',
            'violations_realized': realized_count,
            'violations_attempted': attempted_count,
            'safety_projected': projected_count,
            'strict_soc_violation_steps': strict_soc_violation_steps,
            'strict_soc_violation_duration_h': f'{strict_soc_violation_duration_h:.4f}',
            'strict_soc_violation_kwh': f'{strict_soc_violation_kwh:.4f}',
            'strict_soc_violation_max_kwh': f'{strict_soc_violation_max_kwh:.4f}',
            'safety_projected_meaningful': projected_meaningful_count,
            'projection_delta_mean_w': f'{projection_delta_mean_w:.4f}',
            'projection_delta_max_w': f'{projection_delta_max_w:.4f}',
            'action_mean_abs': f'{np.mean(np.abs(actions)):.4f}',
            'action_raw_mean': f'{np.mean(np.abs(actions_raw)):.4f}' if actions_raw else '0',
            'action_safe_mean': f'{np.mean(np.abs(actions_safe)):.4f}' if actions_safe else '0',
            'flow_action_mean': f'{flow_action_mean:.4f}',
            'flow_active_mean': f'{flow_active_mean:.4f}',
            'flow_power_limited_count': flow_power_limited_count,
            'pump_power_wh': f'{pump_power_kwh * 1000.0:.4f}',
            'flow_too_low_active_count': flow_too_low_active_count,
            'flow_power_mismatch_count': flow_power_mismatch_count,
            'revenue': f'{episode_revenue:.4f}',
            'cost': f'{episode_cost:.4f}',
            'net_profit': f'{episode_revenue - episode_cost:.4f}',
            'buffer_size': len(agent.replay_buffer),
            'conformal_delta': f'{current_conformal_delta:.4f}',
            'proj_penalty_mult': f'{projected_penalty_mult:.3f}',
            'alpha': f'{current_alpha:.4f}',
            'sit1_solo': sit_counts.get(1, 0),
            'sit2_grid_sup': sit_counts.get(2, 0),
            'sit3_charge': sit_counts.get(3, 0),
            'sit4_standby': sit_counts.get(4, 0),
        })

        try:
            target_low, target_high = 5, 10
            if realized_count > target_high:
                current_conformal_delta = max(0.01, current_conformal_delta - 0.005)
                projected_penalty_mult = min(2.0, projected_penalty_mult * 1.1)
                set_conformal_params(window=current_conformal_window, delta=current_conformal_delta)
            elif realized_count <= target_low:
                current_conformal_delta = min(0.15, current_conformal_delta + 0.005)
                projected_penalty_mult = max(0.5, projected_penalty_mult * 0.95)
                set_conformal_params(window=current_conformal_window, delta=current_conformal_delta)
        except Exception:
            pass

        # Evaluation
        if episode % eval_every == 0:
            eval_reward, eval_violations, eval_revenue, eval_cost = evaluate_microgrid_agent(
                agent, env, n_episodes=eval_episodes, use_safetynet=use_safetynet,
                config=config,
                stress_eval_seed=training_config.get('stress_eval_seed', None)
            )
            eval_rewards.append(eval_reward)
            eval_soc_violations.append(eval_violations)
            eval_revenues.append(eval_revenue)
            eval_costs.append(eval_cost)
            
            is_best = "★ BEST" if eval_reward > best_eval_reward else ""
            print(f"   [EVAL] Ep {episode:4d} | "
                  f"Eval={eval_reward:8.2f} (Train={episode_reward:8.2f}) | "
                  f"Violations={eval_violations:3d} | "
                  f"Rev=${eval_revenue:.4f} Cost=${eval_cost:.4f} {is_best}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                if logging_config['save_models']:
                    # Save to experiment directory
                    best_model_path = os.path.join(exp_manager.models_dir, "best_sac_model.pth")
                    agent.save(best_model_path)
                    # Suppress verbose: best model silently saved
        
        # Save checkpoint
        if episode % save_every == 0 and logging_config['save_models']:
            checkpoint_path = os.path.join(exp_manager.models_dir, f"sac_checkpoint_ep{episode}.pth")
            agent.save(checkpoint_path)
        
        if episode % log_interval == 0:
            n_back = min(log_interval, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-n_back:])
            avg_violations = np.mean(episode_soc_violations[-n_back:])
            avg_revenue = np.mean(episode_revenues[-n_back:])
            avg_cost = np.mean(episode_costs[-n_back:])
            avg_attempted = np.mean(episode_attempted_violations[-n_back:])
            avg_projected = np.mean(episode_safety_projected[-n_back:])
            avg_realized = np.mean(episode_realized_violations[-n_back:])
            soc_last = np.array(soc_trajectory)
            elapsed = time.time() - train_start_wall
            eps_per_sec = max(episode + 1, 1) / max(elapsed, 1)
            eta_sec = (total_episodes - episode - 1) / max(eps_per_sec, 1e-9)
            pct = (episode + 1) / total_episodes * 100

            phase_tag = "Phase2:CORAL" if use_safetynet else "Phase1:SAC"
            print(f"\n── Ep {episode:4d}/{total_episodes} ({pct:5.1f}%) [{phase_tag}] "
                  f"| {elapsed/60:.1f}min elapsed | ETA {eta_sec/60:.1f}min ──")
            print(f"   Reward(avg{n_back})={avg_reward:8.2f}  "
                  f"Violations: att={avg_attempted:.1f} proj_sig={avg_projected:.1f} real={avg_realized:.1f}")
            print(f"   Scenarios[last]: S1={sit_counts[1]} S2={sit_counts[2]} "
                  f"S3={sit_counts[3]} S4={sit_counts[4]}")
            print(f"   SoC[last]: {soc_last.min():.3f}~{soc_last.max():.3f} "
                  f"(mean={soc_last.mean():.3f}, end={soc_last[-1]:.3f})")
            print(f"   Revenue=${avg_revenue:.4f}  Cost=${avg_cost:.4f}  "
                  f"Net=${avg_revenue-avg_cost:.4f}  Buffer={len(agent.replay_buffer)}")
    
    # Save final model
    if logging_config['save_models']:
        final_model_path = os.path.join(exp_manager.models_dir, "final_sac_model.pth")
        agent.save(final_model_path)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_soc_violations': episode_soc_violations,
        'episode_action_violations': episode_action_violations,
        'episode_actions': episode_actions,
        'episode_actions_raw': actions_raw_series,
        'episode_actions_safe': actions_safe_series,
        'episode_soc_trajectories': episode_soc_trajectories,
        'episode_revenues': episode_revenues,
        'episode_costs': episode_costs,
        'episode_attempted_violations': episode_attempted_violations,
        'episode_safety_projected': episode_safety_projected,
        'episode_realized_violations': episode_realized_violations,
        'eval_rewards': eval_rewards,
        'eval_soc_violations': eval_soc_violations,
        'eval_revenues': eval_revenues,
        'eval_costs': eval_costs
    }


def evaluate_microgrid_agent(agent: SACAgent, env: MicrogridEnvironment, n_episodes: int = 5, n_steps: int | None = None, use_safetynet: bool = False, stress_eval_seed: int | None = None, config: Dict[str, Any] | None = None) -> tuple:
    """Evaluate agent over multiple episodes"""
    total_reward = 0
    total_violations = 0
    total_revenue = 0
    total_cost = 0
    if n_steps is None:
        n_steps = getattr(env, 'episode_length', 24)
    try:
        conformal_window = int(getattr(env, 'episode_length', n_steps)) * 2 if n_steps is not None else 2880
        set_conformal_params(window=conformal_window, delta=float(0.1))
        clear_residual_buffer()
        print(f"[EVAL] conformal residuals cleared; count={get_residual_count()}")
    except Exception:
        pass
    if stress_eval_seed is not None:
        try:
            np.random.seed(int(stress_eval_seed))
        except Exception:
            pass
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        try:
            if hasattr(env, 'episode_data') and hasattr(env, 'load_data'):
                total_len = int(min(len(env.load_data), len(env.pv_data), len(env.price_data)))
                if n_steps is not None and total_len >= int(n_steps):
                    start_idx_eval = int(total_len - int(n_steps))
                    setattr(env, 'fixed_start_idx', start_idx_eval)
                    state, _ = env.reset()
        except Exception:
            pass
        episode_reward = 0
        episode_violations = 0
        episode_revenue = 0
        episode_cost = 0
        prev_soc_violations_cum = 0
        prev_a_eval = 0.0
        
        use_flow_eval = bool(getattr(env, 'use_flow_rate_action', False))
        for step in range(int(n_steps)):
            action = agent.select_action(state, evaluate=True)  # [-1, 1]^action_dim
            charge_limit_kw, discharge_limit_kw = get_power_limits(env)
            pmax = max(charge_limit_kw, discharge_limit_kw)
            a_raw_kw = norm_to_power_kw(float(action[0]), env)
            if use_flow_eval and len(action) >= 2:
                flow_fraction_eval = float(np.clip((float(action[1]) + 1.0) * 0.5, 0.0, 1.0))
            else:
                flow_fraction_eval = float(getattr(env, 'flow_idle_fraction', 0.0))
            if use_safetynet:
                ramp_kw = getattr(env, 'safetynet_ramp_kw', None)
                soc_bounds = get_safetynet_soc_bounds(config or {}, env)
                a_safe_kw, _, _ = safety_project(
                    state=state,
                    action=np.array([a_raw_kw], dtype=np.float32),
                    prev_action=prev_a_eval,
                    pmax=pmax,
                    pmin=discharge_limit_kw,
                    pmax_positive=charge_limit_kw,
                    ramp_kw=ramp_kw,
                    soc_bounds=soc_bounds,
                    env=env,
                )
                need_reset = False
                if hasattr(env, 'hard_guard') and not bool(getattr(env, 'hard_guard')):
                    setattr(env, 'hard_guard', True)
                    need_reset = True
                env_action_eval = [a_safe_kw, flow_fraction_eval] if use_flow_eval else [a_safe_kw]
                next_state, reward, terminated, truncated, info = env.step(env_action_eval)
                if need_reset:
                    setattr(env, 'hard_guard', False)
                prev_a_eval = a_safe_kw
            else:
                env_action_eval = [a_raw_kw, flow_fraction_eval] if use_flow_eval else [a_raw_kw]
                next_state, reward, terminated, truncated, info = env.step(env_action_eval)
            done = terminated or truncated
            
            episode_reward += reward
            current_soc_violations_cum = int(info.get('soc_violations', 0))
            episode_violations += max(0, current_soc_violations_cum - prev_soc_violations_cum)
            prev_soc_violations_cum = current_soc_violations_cum
            episode_revenue = info.get('total_revenue', 0)
            episode_cost = info.get('total_cost', 0)
            
            if done:
                break
                
            state = next_state

        try:
            if hasattr(env, 'fixed_start_idx'):
                setattr(env, 'fixed_start_idx', None)
        except Exception:
            pass
        total_reward += episode_reward
        total_violations += episode_violations
        total_revenue += episode_revenue
        total_cost += episode_cost
 
    return (total_reward / n_episodes, total_violations, 
            total_revenue / n_episodes, total_cost / n_episodes)


def plot_microgrid_training_results(metrics: Dict[str, List[float]], config: Dict[str, Any], 
                                   exp_manager: ExperimentManager, save_path: str = "training_results.png"):
    """Plot training results for microgrid environment"""
    def sliding_average(values, window: int = 20):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        window = max(1, min(int(window), arr.size))
        kernel = np.ones(window, dtype=float) / float(window)
        smooth = np.convolve(arr, kernel, mode='valid')
        if window == 1:
            return smooth
        pad_left = np.full(window - 1, np.nan, dtype=float)
        return np.concatenate([pad_left, smooth])

    try:
        num_eps_guard = int(len(metrics.get('episode_rewards', [])))
    except Exception:
        num_eps_guard = 0
    if num_eps_guard == 0:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.axis('off')
        present_keys = list(metrics.keys()) if isinstance(metrics, dict) else []
        msg_lines = [
            'No training data available to plot.',
            'This diagnostic page is generated to avoid a blank image.',
            f"Available keys: {present_keys}",
            'Expected keys include: episode_rewards, episode_lengths, episode_realized_violations,',
            'episode_attempted_violations, episode_safety_projected, episode_revenues, episode_costs.'
        ]
        ax.text(0.02, 0.98, "\n".join(msg_lines), va='top', ha='left', fontsize=12)
        plt.tight_layout()
        exp_plot_path = os.path.join(exp_manager.results_dir, "training_results.png")
        plt.savefig(exp_plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Episode rewards
    reward_window = int(config.get('plotting', {}).get('reward_smoothing_window', 20))
    reward_series = np.asarray(metrics['episode_rewards'], dtype=float)
    reward_smoothed = sliding_average(reward_series, window=reward_window)
    axes[0, 0].plot(reward_series, color='lightgray', linewidth=0.9, alpha=0.7, label='Raw reward')
    axes[0, 0].plot(reward_smoothed, color='tab:blue', linewidth=2.2, label=f'Sliding avg ({reward_window})')
    axes[0, 0].set_title('Episode Rewards (Smoothed)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # Episode lengths
    axes[0, 1].plot(metrics['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # SoC violations
    soc_series = metrics.get('episode_realized_violations') or metrics.get('episode_soc_violations')
    axes[0, 2].plot(soc_series)
    axes[0, 2].set_title('SoC Violations per Episode (Realized)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Violations')
    axes[0, 2].grid(True)
    
    # Evaluation rewards
    if metrics['eval_rewards']:
        eval_episodes = np.arange(0, len(metrics['episode_rewards']), config['training']['eval_every'])
        axes[1, 0].plot(eval_episodes, metrics['eval_rewards'], 'ro-')
        axes[1, 0].set_title('Evaluation Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
    
    # Average actions
    axes[1, 1].plot(metrics['episode_actions'], label='|Action| (legacy avg)', color='gray', alpha=0.5)
    if 'episode_actions_raw' in metrics:
        axes[1, 1].plot(metrics['episode_actions_raw'], label='|a_raw|', color='orange')
    if 'episode_actions_safe' in metrics:
        axes[1, 1].plot(metrics['episode_actions_safe'], label='|a_safe|', color='blue')
    axes[1, 1].set_title('Average Action Magnitude (Raw vs Safe)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('|Action|')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    # Revenue vs Cost
    if metrics['episode_revenues'] and metrics['episode_costs']:
        axes[1, 2].plot(metrics['episode_revenues'], label='Revenue', color='green')
        axes[1, 2].plot(metrics['episode_costs'], label='Cost', color='red')
        axes[1, 2].set_title('Episode Revenue vs Cost')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Amount ($)')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
    
    # SoC trajectory example (last episode)
    if metrics['episode_soc_trajectories']:
        last_soc = metrics['episode_soc_trajectories'][-1]
        steps = range(len(last_soc))
        axes[2, 0].plot(steps, last_soc, 'b-', linewidth=2)
        axes[2, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='SoC Min')
        axes[2, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='SoC Max')
        axes[2, 0].set_title('SoC Trajectory (Last Episode)')
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('SoC')
        axes[2, 0].grid(True)
        axes[2, 0].legend()
    
    num_eps = len(metrics.get('episode_rewards', []))
    def pad_series(key: str):
        arr = metrics.get(key, []) or []
        if num_eps <= 0:
            return []
        if len(arr) < num_eps:
            arr = list(arr) + [0] * (num_eps - len(arr))
        elif len(arr) > num_eps:
            arr = list(arr[:num_eps])
        return arr
    att_series = pad_series('episode_attempted_violations')
    proj_series = pad_series('episode_safety_projected')
    real_series = pad_series('episode_realized_violations') or pad_series('episode_soc_violations')
    if num_eps > 0:
        att = np.array(att_series, dtype=float)
        proj = np.array(proj_series, dtype=float)
        real = np.array(real_series, dtype=float)
        steps = float(config['training'].get('max_steps', len(real)))
        overlap_score = (np.mean(np.abs(att - proj)) + np.mean(np.abs(proj - real))) / max(1.0, steps)
        if overlap_score < 0.02:
            no_intervention = np.clip(att - proj, 0, None)
            prevented = np.clip(proj - real, 0, None)
            failed = np.clip(real, 0, None)
            x = np.arange(num_eps)
            axes[2, 1].bar(x, no_intervention, label='Attempted w/o SN', color='gray', alpha=0.5)
            axes[2, 1].bar(x, prevented, bottom=no_intervention, label='SN Prevented', color='tab:blue', alpha=0.7)
            axes[2, 1].bar(x, failed, bottom=no_intervention+prevented, label='Realized', color='tab:red', alpha=0.8)
            axes[2, 1].set_title('SafetyNet Outcomes per Episode (Stacked)')
        else:
            axes[2, 1].plot(att_series, label='Attempted', color='orange', marker='o', markevery=max(1, num_eps//10), alpha=0.7)
            axes[2, 1].plot(proj_series, label='Projected', color='blue', marker='s', markevery=max(1, num_eps//10), alpha=0.7)
            axes[2, 1].plot(real_series, label='Realized', color='red', marker='x', markevery=max(1, num_eps//10), alpha=0.8)
            d1 = (att - proj).tolist()
            d2 = (proj - real).tolist()
            axes[2, 1].plot(d1, label='Δ(Att-Proj)', color='tab:gray', linestyle='--', alpha=0.4)
            axes[2, 1].plot(d2, label='Δ(Proj-Real)', color='tab:green', linestyle='--', alpha=0.4)
            axes[2, 1].set_title('SafetyNet KPIs per Episode (Lines + Deltas)')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Count (<= steps)')
        axes[2, 1].grid(True)
        axes[2, 1].legend()

    # Net profit (Revenue - Cost)
    if metrics['episode_revenues'] and metrics['episode_costs']:
        net_profits = [r - c for r, c in zip(metrics['episode_revenues'], metrics['episode_costs'])]
        axes[2, 2].plot(net_profits, color='purple')
        axes[2, 2].set_title('Net Profit per Episode')
        axes[2, 2].set_xlabel('Episode')
        axes[2, 2].set_ylabel('Net Profit ($)')
        axes[2, 2].grid(True)
        axes[2, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save to experiment directory
    exp_plot_path = os.path.join(exp_manager.results_dir, "training_results.png")
    plt.savefig(exp_plot_path, dpi=300, bbox_inches='tight')
    print(f"Training plot saved to: {exp_plot_path}")
    
    # Also save locally for display
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function with microgrid environment"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SAC agent with Microgrid Environment')
    parser.add_argument('--config', type=str, default='../configs/config_microgrid.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override total episodes from config')
    parser.add_argument('--name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--variant', type=str, choices=['sac', 'sac_penalty', 'sac_sn', 'sac_sn_evi'], default=None,
                       help='Training variant: sac, sac_penalty, sac_sn, or sac_sn_evi')
    parser.add_argument('--actor-warmstart', type=str, default=None,
                       help='Optional path to actor-only checkpoint or full SAC checkpoint for actor warm-start')
    parser.add_argument('--warmstart-mode', type=str, default='actor_only',
                       choices=['actor_only', 'actor_critics', 'full_agent'],
                       help='Warm-start scope: actor_only, actor_critics, or full_agent')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    # Setup conformal parameters if provided
    conformal_cfg = config.get('conformal', {})
    try:
        set_conformal_params(
            window=int(conformal_cfg.get('window', 2880)),
            delta=float(conformal_cfg.get('delta', 0.1))
        )
    except Exception:
        set_conformal_params()
    
    # Override episodes if specified
    if args.episodes is not None:
        config['training']['total_episodes'] = args.episodes
    # Apply variant override if provided
    if args.variant is not None:
        config.setdefault('training', {})['variant'] = args.variant
    
    # Create experiment manager after CLI overrides so saved config matches the actual run.
    exp_manager = ExperimentManager(args.name)
    exp_manager.save_config(config)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Create microgrid environment
    env = create_environment(config)
    
    # Agent parameters
    state_dim = env.observation_space.shape[0]  # 6D state space
    action_dim = env.action_space.shape[0]      # 1D action space
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Device
    device_config = config['device']
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    print(f"Using device: {device}")
    
    # Create SAC agent
    agent = create_agent(config, state_dim, action_dim, device)

    # Optional warm-start
    if args.actor_warmstart:
        ckpt = torch.load(args.actor_warmstart, map_location=device)
        if args.warmstart_mode == 'full_agent':
            agent.load(args.actor_warmstart)
            print(f"Loaded full agent warm-start: {args.actor_warmstart}")
        else:
            actor_state = ckpt.get('actor') if isinstance(ckpt, dict) and 'actor' in ckpt else ckpt
            actor_stats = warmstart_actor_flexible(agent, actor_state)
            if args.warmstart_mode == 'actor_critics' and isinstance(ckpt, dict):
                if 'critic1' in ckpt:
                    agent.critic1.load_state_dict(ckpt['critic1'], strict=True)
                if 'critic2' in ckpt:
                    agent.critic2.load_state_dict(ckpt['critic2'], strict=True)
                if 'target_critic1' in ckpt:
                    agent.target_critic1.load_state_dict(ckpt['target_critic1'], strict=True)
                else:
                    agent.target_critic1.load_state_dict(agent.critic1.state_dict())
                if 'target_critic2' in ckpt:
                    agent.target_critic2.load_state_dict(ckpt['target_critic2'], strict=True)
                else:
                    agent.target_critic2.load_state_dict(agent.critic2.state_dict())
                if 'occ_head' in ckpt:
                    agent.occ_head.load_state_dict(ckpt['occ_head'], strict=True)
                if 'log_alpha' in ckpt:
                    with torch.no_grad():
                        agent.log_alpha.data.copy_(ckpt['log_alpha'].to(device))
                    agent.alpha = agent.log_alpha.exp()
                print(f"Loaded actor+critics warm-start: {args.actor_warmstart}")
            else:
                print(
                    f"Loaded actor warm-start: {args.actor_warmstart} "
                    f"(loaded={actor_stats['loaded']}, partial={actor_stats['partial']}, "
                    f"skipped={actor_stats['skipped']})"
                )

    guided_stats = guided_behavior_cloning_pretrain(env, agent, config, exp_manager)
    if guided_stats:
        print(
            "[GUIDED-BC] completed | "
            f"samples={guided_stats['demo_samples']} | "
            f"epochs={guided_stats['bc_epochs']} | "
            f"final_loss={guided_stats['bc_final_loss']:.6f}"
        )
    
    # Train agent
    start_time = time.time()
    metrics = train_sac_with_microgrid(env, agent, config, exp_manager)
    training_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  Training completed in {training_time/60:.1f} min ({training_time:.0f} sec)")
    print(f"{'='*70}")
    if metrics['eval_rewards']:
        print(f"  Final eval reward : {metrics['eval_rewards'][-1]:.4f}")
        print(f"  Best eval reward  : {max(metrics['eval_rewards']):.4f}")
    print(f"  Avg SoC violations: {np.mean(metrics['episode_soc_violations']):.2f}")
    print(f"  Avg action viol.  : {np.mean(metrics['episode_action_violations']):.2f}")
    
    # Collect compute resources information
    compute_res = collect_compute_resources(agent, device, training_time)
    
    # Save results to experiment directory
    _variant = config.get('training', {}).get('variant', 'sac')
    _sn_warmup = int(config.get('training', {}).get('safetynet_warmup_episodes', 0))
    _curriculum = _variant in ('sac_sn', 'sac_sn_evi') and _sn_warmup > 0
    metadata = {
        'variant': _variant,
        'curriculum': _curriculum,
        'safetynet_warmup_episodes': _sn_warmup if _curriculum else 0,
        'time_step': float(getattr(env, 'time_step', 1.0)),
        'seed': int(config.get('random_seed', 0)),
    }
    exp_manager.save_results(metrics, metadata=metadata, compute_resources=compute_res)
    
    # Print compute resources summary
    print("\n" + "="*50)
    print(format_compute_resources(compute_res))
    print("="*50)
    
    # Plot results
    if config['logging']['plot_results']:
        plot_microgrid_training_results(metrics, config, exp_manager)
    
    # Save metrics
    if config['logging']['save_metrics']:
        metrics_path = "sac_training_metrics.npz"
        np.savez(metrics_path, **metrics)
        exp_manager.save_metrics(metrics_path)
        print("Training metrics saved to experiment directory")
    
    # Print final statistics
    print("\nFinal Training Statistics:")
    print(f"Total episodes: {len(metrics['episode_rewards'])}")
    print(f"Average reward: {np.mean(metrics['episode_rewards']):.2f}")
    print(f"Best reward: {np.max(metrics['episode_rewards']):.2f}")
    print(f"Average SoC violations: {np.mean(metrics['episode_soc_violations']):.2f}")
    print(f"Total SoC violations: {np.sum(metrics['episode_soc_violations'])}")
    print(f"Average revenue: ${np.mean(metrics['episode_revenues']):.2f}")
    print(f"Average cost: ${np.mean(metrics['episode_costs']):.2f}")
    print(f"Net profit: ${np.mean(metrics['episode_revenues']) - np.mean(metrics['episode_costs']):.2f}")
    
    # Print experiment summary
    exp_manager.print_experiment_summary()
    
    # Clean up temporary files
    exp_manager.cleanup_temp_files()


if __name__ == "__main__":
    main() 
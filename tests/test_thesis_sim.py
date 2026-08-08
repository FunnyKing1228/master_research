import os
import sys

import numpy as np
import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THESIS_CODE = os.path.join(PROJECT_ROOT, "thesis_sim", "code")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THESIS_CODE)

from core.microgrid_env import create_microgrid_env
from thesis_config import DEFAULT_CONFIG, create_thesis_environment, load_thesis_config, validate_thesis_config
from run_eval import run_rollout


def _make_thesis_env(**overrides):
    defaults = dict(
        microgrid_id=0,
        episode_length=4,
        time_step=0.25,
        battery_capacity_kwh=0.0112,
        battery_power_kw=0.0056,
        battery_charge_power_kw=0.0085,
        battery_discharge_power_kw=0.0056,
        battery_efficiency=0.95,
        soc_min=0.20,
        soc_max=0.80,
        use_real_data=False,
        synthetic_hourly_hold=True,
        synthetic_pv_peak_kw=0.0,
        synthetic_load_base_kw=0.0092,
        synthetic_load_amp_kw=0.0,
        deployment_observation_style=True,
        deployment_group_power_kw=0.0023,
        battery_delivered_load_per_group_kw=0.00058,
        deployment_guard_style=True,
        enforce_solo_discharge_load_limit=False,
        discharge_auto=True,
        discharge_mode="solo_only",
        voltage_cutoff_soc=0.20,
        pv_obs_boolean=True,
        pv_obs_boolean_threshold_kw=0.001,
        pv_support_ratio_obs=True,
        pv_sufficient_ratio_threshold=0.8,
        use_flow_rate_action=False,
        pre_measure_rest_flow_fraction=0.0,
        pre_measure_flow_fraction=0.5,
        pre_measure_seconds=25,
    )
    defaults.update(overrides)
    env = create_microgrid_env(**defaults)
    env.reset(seed=0)
    env.episode_data["load"] = np.full(env.episode_length, 0.0092, dtype=float)
    env.episode_data["pv"] = np.zeros(env.episode_length, dtype=float)
    env.episode_data["price"] = np.full(env.episode_length, 7.13, dtype=float)
    env.episode_data["hour"] = np.full(env.episode_length, 14, dtype=int)
    env.episode_data["dow"] = np.zeros(env.episode_length, dtype=int)
    env._refresh_episode_pv_bool()
    env.current_soc = 0.5
    return env


def test_base_thesis_config_loads_and_validates_without_warnings():
    cfg = load_thesis_config(DEFAULT_CONFIG)

    warnings = validate_thesis_config(cfg)

    assert warnings == []
    assert cfg["env"]["battery_delivered_load_per_group_kw"] == pytest.approx(0.00058)
    assert cfg["env"]["enforce_solo_discharge_load_limit"] is False


def test_base_thesis_config_can_construct_environment():
    cfg = load_thesis_config(DEFAULT_CONFIG)
    cfg["env"]["episode_length"] = 2
    cfg["training"]["max_steps"] = 2
    cfg["stress"]["enable"] = False

    env = create_thesis_environment(cfg)
    obs, info = env.reset(seed=0)

    assert obs.shape[0] >= 6
    assert info["step"] == 0
    assert env.battery_delivered_load_per_group_kw == pytest.approx(0.00058)


def test_short_thesis_rollout_smoke():
    rows = run_rollout(DEFAULT_CONFIG, steps=2, policy="no_pv_discharge")

    assert len(rows) == 2
    assert "battery_delivered_load_kw" in rows[0]
    assert "pre_measure_event" in rows[0]


def test_no_pv_discharge_uses_battery_delivered_load_scale_without_old_load_guard():
    env = _make_thesis_env()

    _, _, _, _, info = env.step([-0.001])

    assert info["grid_pv_load_kw"] == pytest.approx(0.0092)
    assert info["battery_delivered_load_kw"] == pytest.approx(4 * 0.00058)
    assert info["guard_block_load_over_discharge_limit"] == 0
    assert info["blocked_by_load"] is False
    assert info["useful_discharge"] == pytest.approx(4 * 0.00058)
    assert info["applied_action_kw"] == pytest.approx(-(4 * 0.00058))


def test_pv_active_still_blocks_thesis_discharge():
    env = _make_thesis_env()
    env.episode_data["pv"] = np.full(env.episode_length, 0.002, dtype=float)
    env._refresh_episode_pv_bool()

    _, _, _, _, info = env.step([-0.001])

    assert info["guard_block_pv_active_discharge"] == 1
    assert info["applied_action_kw"] == pytest.approx(0.0)
    assert info["useful_discharge"] == pytest.approx(0.0)

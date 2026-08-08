"""
Unit tests for core/microgrid_env.py
=====================================
Covers:
  - observation/action space dimensions
  - SoC boundary clamping
  - charge, discharge, and standby logic
  - discharge_auto mode
  - voltage_cutoff_soc
"""
import os
import sys
import pytest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)

from core.microgrid_env import create_microgrid_env


def _make_env(**overrides):
    """Create a synthetic-data test environment using 15-minute SLFB settings."""
    defaults = dict(
        microgrid_id=0,
        episode_length=96,               # 24h / 0.25h = 96 steps
        time_step=0.25,                   # 15 min
        battery_capacity_kwh=0.00990,
        battery_power_kw=0.000825,
        battery_efficiency=0.95,
        soc_min=0.1,
        soc_max=0.9,
        use_real_data=False,
        synthetic_hourly_hold=True,
        synthetic_pv_peak_kw=0.001,       # 1W peak
        synthetic_load_base_kw=0.0004,    # 400 mW
        synthetic_load_amp_kw=0.0,        # constant load
        use_flow_rate_action=True,
        flow_R_base_ohm=10.0,
        flow_P_max_pump_W=0.124,
        flow_I_rated_A=0.150,
    )
    defaults.update(overrides)
    return create_microgrid_env(**defaults)


class TestEnvBasics:

    def test_observation_space(self):
        env = _make_env()
        obs, _ = env.reset()
        assert obs.shape == (6,), f"Expected 6D state, got {obs.shape}"

    def test_observation_space_without_price_obs(self):
        env = _make_env(use_flow_rate_action=False, price_obs=False)
        obs, _ = env.reset()
        assert obs.shape == (5,), f"Expected 5D state without price, got {obs.shape}"

    def test_observation_space_ratio_without_price_obs(self):
        env = _make_env(
            use_flow_rate_action=False,
            pv_support_ratio_obs=True,
            pv_obs_boolean=True,
            price_obs=False,
        )
        obs, _ = env.reset()
        assert obs.shape == (6,), f"Expected 6D state with pv ratio but no price, got {obs.shape}"

    def test_observation_space_with_tou_onehot(self):
        env = _make_env(
            use_flow_rate_action=False,
            pv_obs_boolean=True,
            price_obs=False,
            tou_onehot_obs=True,
        )
        obs, _ = env.reset()
        assert obs.shape == (8,), f"Expected 8D state with TOU onehot, got {obs.shape}"

    def test_observation_space_ratio_with_tou_onehot(self):
        env = _make_env(
            use_flow_rate_action=False,
            pv_support_ratio_obs=True,
            pv_obs_boolean=True,
            price_obs=False,
            tou_onehot_obs=True,
        )
        obs, _ = env.reset()
        assert obs.shape == (9,), f"Expected 9D state with pv ratio + TOU onehot, got {obs.shape}"

    def test_action_space_2d(self):
        env = _make_env(use_flow_rate_action=True)
        assert env.action_space.shape[0] == 2  # [power, flow]

    def test_action_space_1d(self):
        env = _make_env(use_flow_rate_action=False)
        assert env.action_space.shape[0] == 1  # [power]

    def test_reset_soc_in_state(self):
        env = _make_env()
        obs, _ = env.reset()
        # obs[0] = SoC, should be between soc_min and soc_max
        assert 0.0 <= obs[0] <= 1.0

    def test_state_uses_dataset_timestamp_hour_and_dow_when_available(self):
        env = _make_env(use_flow_rate_action=False)
        env.load_data = np.full(env.episode_length, 1.0, dtype=float)
        env.pv_data = np.zeros(env.episode_length, dtype=float)
        env.price_data = np.full(env.episode_length, 2.06, dtype=float)
        env.hour_data = np.full(env.episode_length, 17, dtype=int)
        env.dow_data = np.full(env.episode_length, 6, dtype=int)

        obs, _ = env.reset(seed=0)

        assert obs[4] == pytest.approx(17.0, abs=1e-6)
        assert obs[5] == pytest.approx(6.0, abs=1e-6)

    def test_deployment_observation_style_uses_schedule_fallback_load(self):
        env = _make_env(
            use_flow_rate_action=False,
            deployment_observation_style=True,
            deployment_load_threshold_kw=0.0005,
            deployment_group_power_kw=0.0001,
        )
        env.load_data = np.full(env.episode_length, 0.0001, dtype=float)
        env.pv_data = np.zeros(env.episode_length, dtype=float)
        env.price_data = np.full(env.episode_length, 2.06, dtype=float)
        env.hour_data = np.full(env.episode_length, 14, dtype=int)
        env.dow_data = np.zeros(env.episode_length, dtype=int)

        obs, _ = env.reset(seed=0)

        assert obs[1] == pytest.approx(4 * 0.0001, abs=1e-9)

    def test_pv_bool_uses_ratio_threshold(self):
        env = _make_env(
            use_flow_rate_action=False,
            pv_support_ratio_obs=True,
            pv_obs_boolean=True,
            price_obs=True,
            pv_sufficient_ratio_threshold=0.8,
        )
        env.load_data = np.full(env.episode_length, 1.0, dtype=float)
        env.pv_data = np.full(env.episode_length, 0.79, dtype=float)
        env.price_data = np.full(env.episode_length, 2.06, dtype=float)

        obs, _ = env.reset(seed=0)
        assert obs.shape == (7,)
        assert obs[2] == pytest.approx(0.79, abs=1e-6)
        assert obs[3] == pytest.approx(0.0, abs=1e-6)
        assert env.episode_data["pv_bool"][0] == pytest.approx(0.0, abs=1e-6)

        env.pv_data = np.full(env.episode_length, 0.80, dtype=float)
        obs2, _ = env.reset(seed=0)
        assert obs2[2] == pytest.approx(0.80, abs=1e-6)
        assert obs2[3] == pytest.approx(1.0, abs=1e-6)
        assert env.episode_data["pv_bool"][0] == pytest.approx(1.0, abs=1e-6)

    def test_reset_ignores_legacy_pv_bool_column(self):
        env = _make_env(
            use_flow_rate_action=False,
            pv_support_ratio_obs=True,
            pv_obs_boolean=True,
            price_obs=True,
            pv_sufficient_ratio_threshold=0.8,
        )
        env.load_data = np.full(env.episode_length, 1.0, dtype=float)
        env.pv_data = np.full(env.episode_length, 0.50, dtype=float)
        env.price_data = np.full(env.episode_length, 2.06, dtype=float)
        env.pv_bool_data = np.ones(env.episode_length, dtype=float)

        obs, _ = env.reset(seed=0)
        assert obs[2] == pytest.approx(0.50, abs=1e-6)
        assert obs[3] == pytest.approx(0.0, abs=1e-6)
        assert env.episode_data["pv_bool"][0] == pytest.approx(0.0, abs=1e-6)

    def test_discharge_is_blocked_when_pv_is_present_but_not_sufficient(self):
        env = _make_env(
            use_flow_rate_action=False,
            discharge_auto=False,
            discharge_mode="solo_only",
            pv_support_ratio_obs=True,
            pv_obs_boolean=True,
            price_obs=True,
            pv_sufficient_ratio_threshold=0.8,
            battery_power_kw=1.2,
            battery_discharge_power_kw=1.2,
            battery_charge_power_kw=1.2,
            synthetic_pv_peak_kw=0.0,
            synthetic_load_base_kw=1.0,
        )
        env.load_data = np.full(env.episode_length, 1.0, dtype=float)
        env.pv_data = np.full(env.episode_length, 0.3, dtype=float)
        env.price_data = np.full(env.episode_length, 4.69, dtype=float)

        obs, _ = env.reset(seed=0)
        env.current_soc = 0.5
        obs2, _, _, _, info = env.step([-1.0])

        assert obs[2] == pytest.approx(0.3, abs=1e-6)
        assert obs[3] == pytest.approx(0.0, abs=1e-6)
        assert info["blocked_by_pv"] is True
        assert info["useful_discharge"] == pytest.approx(0.0, abs=1e-9)
        assert obs2[0] == pytest.approx(0.5, abs=1e-6)

    def test_continuous_operation_mode_preserves_soc_and_previous_action(self):
        env = _make_env(
            use_flow_rate_action=False,
            continuous_operation_mode=True,
            synthetic_pv_peak_kw=0.0,
            synthetic_load_base_kw=0.0004,
            battery_power_kw=0.5,
            battery_charge_power_kw=0.5,
            battery_discharge_power_kw=0.5,
        )
        env.reset(seed=0)
        env.current_soc = 0.4
        env.step([0.2])

        soc_after_step = env.current_soc
        prev_action_after_step = env.prev_action_kw

        env.reset(seed=1)

        assert env.current_soc == pytest.approx(soc_after_step, abs=1e-6)
        assert env.prev_action_kw == pytest.approx(prev_action_after_step, abs=1e-6)

    def test_deployment_guard_style_blocks_low_soc_discharge(self):
        env = _make_env(
            use_flow_rate_action=False,
            deployment_guard_style=True,
            discharge_auto=False,
            synthetic_pv_peak_kw=0.0,
            synthetic_load_base_kw=1.0,
            battery_power_kw=1.0,
            battery_charge_power_kw=1.0,
            battery_discharge_power_kw=1.0,
        )
        env.reset(seed=0)
        env.current_soc = 0.10

        obs2, _, _, _, info = env.step([-0.5])

        assert info["guard_block_low_soc_discharge"] == 1
        assert info["applied_action_kw"] == pytest.approx(0.0, abs=1e-9)
        assert obs2[0] == pytest.approx(0.10, abs=1e-6)

    def test_zero_response_stress_can_nullify_battery_effect(self):
        env = _make_env(
            use_flow_rate_action=False,
            stress_enable=True,
            stress_battery_zero_response_prob=1.0,
            synthetic_pv_peak_kw=0.0,
            battery_power_kw=1.0,
            battery_charge_power_kw=1.0,
            battery_discharge_power_kw=1.0,
        )
        obs, _ = env.reset(seed=0)
        initial_soc = obs[0]

        obs2, _, _, _, info = env.step([0.5])

        assert obs2[0] == pytest.approx(initial_soc, abs=1e-6)
        assert info["applied_action_kw"] == pytest.approx(0.0, abs=1e-9)

    def test_episode_length(self):
        env = _make_env(episode_length=10)
        obs, _ = env.reset()
        for i in range(10):
            action = [0.0, 0.5]  # standby
            obs, reward, done, truncated, info = env.step(action)
        assert done or truncated


class TestEnergyFlow:

    def test_charge_increases_soc(self):
        """Charging action increases SoC."""
        env = _make_env()
        obs, _ = env.reset()
        initial_soc = obs[0]

        # Public-facing comment translated to English.
        action = [0.8, 0.5]
        obs2, _, _, _, _ = env.step(action)
        assert obs2[0] > initial_soc, \
            f"SoC should increase from {initial_soc} after charging, got {obs2[0]}"

    def test_discharge_decreases_soc(self):
        """Discharging action decreases SoC in non-auto mode."""
        env = _make_env(discharge_auto=False)
        env.reset()
        env.current_soc = 0.5
        initial_soc = env.current_soc

        action = [-0.8, 0.5]  # discharge
        obs2, _, _, _, _ = env.step(action)
        assert obs2[0] < initial_soc, \
            f"SoC should decrease from {initial_soc} after discharging, got {obs2[0]}"

    def test_standby_minimal_soc_change(self):
        """Standby action causes minimal SoC change."""
        env = _make_env()
        obs, _ = env.reset()
        initial_soc = obs[0]
        action = [0.0, 0.5]
        obs2, _, _, _, _ = env.step(action)
        # Public-facing comment translated to English.
        assert abs(obs2[0] - initial_soc) < 0.01

    def test_soc_stays_in_range(self):
        """SoC remains within [soc_min, soc_max] after multiple steps."""
        env = _make_env(soc_min=0.1, soc_max=0.9)
        obs, _ = env.reset()

        # Public-facing comment translated to English.
        for _ in range(50):
            action = [1.0, 0.5]
            obs, _, done, trunc, _ = env.step(action)
            if done or trunc:
                break
        assert obs[0] <= 0.9 + 0.01

        # Public-facing comment translated to English.
        env2 = _make_env(soc_min=0.1, soc_max=0.9, discharge_auto=False)
        obs2, _ = env2.reset()
        for _ in range(50):
            action = [-1.0, 0.5]
            obs2, _, done, trunc, _ = env2.step(action)
            if done or trunc:
                break
        assert obs2[0] >= 0.1 - 0.01


class TestDischargeAuto:
    """discharge_auto=True: discharge amount is determined by load."""

    def test_discharge_auto_no_crash(self):
        """Auto mode does not crash."""
        env = _make_env(discharge_auto=True)
        obs, _ = env.reset()
        for _ in range(10):
            action = [-0.5, 0.5]
            obs, _, done, trunc, _ = env.step(action)
            if done or trunc:
                break

    def test_solo_only_blocks_partial_discharge(self):
        """In solo_only mode, discharge is blocked when load exceeds Pmax."""
        env = _make_env(
            discharge_auto=True,
            discharge_mode="solo_only",
            use_flow_rate_action=False,
            synthetic_pv_peak_kw=0.0,
            synthetic_load_base_kw=0.0012,
            battery_power_kw=0.000825,
            battery_discharge_power_kw=0.000825,
        )
        obs, _ = env.reset()
        env.current_soc = 0.5
        obs2, _, _, _, info = env.step([-1.0])

        assert info["blocked_by_load"] is True
        assert info["useful_discharge"] == pytest.approx(0.0, abs=1e-9)
        assert info["situation_code"] == 4
        assert obs2[0] == pytest.approx(0.5, abs=1e-6)

    def test_partial_assist_falls_back_to_solo_only(self):
        """Legacy partial_assist falls back to hardware-aligned solo_only mode."""
        env = _make_env(
            discharge_auto=True,
            discharge_mode="partial_assist",
            use_flow_rate_action=False,
            synthetic_pv_peak_kw=0.0,
            synthetic_load_base_kw=0.0012,
            battery_power_kw=0.000825,
            battery_discharge_power_kw=0.000825,
        )
        obs, _ = env.reset()
        env.current_soc = 0.5
        obs2, _, _, _, info = env.step([-1.0])

        assert env.discharge_mode == "solo_only"
        assert info["blocked_by_load"] is True
        assert info["useful_discharge"] == pytest.approx(0.0, abs=1e-9)
        assert info["situation_code"] == 4
        assert obs2[0] == pytest.approx(0.5, abs=1e-6)


class TestSocPhysicalFloor:
    """The optional physical floor limits discharge without changing legacy behavior."""

    def test_no_floor_allows_negative_soc_and_default_matches_explicit_none(self):
        kwargs = dict(
            use_flow_rate_action=False,
            clip_soc_to_bounds=False,
            battery_capacity_kwh=0.001,
            battery_power_kw=0.001,
            battery_discharge_power_kw=0.001,
            battery_efficiency=1.0,
        )
        default_env = _make_env(**kwargs)
        explicit_env = _make_env(**kwargs)
        explicit_env.soc_physical_floor = None
        for env in (default_env, explicit_env):
            env.current_soc = 0.05

        default_soc, default_action = default_env._update_battery_soc(-0.001)
        explicit_soc, explicit_action = explicit_env._update_battery_soc(-0.001)

        assert default_soc < 0.0
        assert default_soc == pytest.approx(explicit_soc)
        assert default_action == pytest.approx(explicit_action)

    def test_floor_truncates_discharge_at_zero(self):
        env = _make_env(
            use_flow_rate_action=False,
            clip_soc_to_bounds=False,
            battery_capacity_kwh=0.001,
            battery_power_kw=0.001,
            battery_discharge_power_kw=0.001,
            battery_efficiency=1.0,
        )
        env.soc_physical_floor = 0.0
        env.current_soc = 0.05

        soc, actual_action_kw = env._update_battery_soc(-0.001)

        assert soc == pytest.approx(0.0)
        assert actual_action_kw == pytest.approx(-0.0002)
        assert env.soc_violations == 1

    def test_floor_leaves_charge_side_unchanged(self):
        kwargs = dict(
            use_flow_rate_action=False,
            clip_soc_to_bounds=False,
            battery_capacity_kwh=0.001,
            battery_power_kw=0.001,
            battery_charge_power_kw=0.001,
            battery_efficiency=1.0,
        )
        without_floor = _make_env(**kwargs)
        with_floor = _make_env(**kwargs)
        with_floor.soc_physical_floor = 0.0
        for env in (without_floor, with_floor):
            env.current_soc = -0.10

        expected = without_floor._update_battery_soc(0.0002)
        actual = with_floor._update_battery_soc(0.0002)

        assert actual == pytest.approx(expected)

    def test_grid_import_and_cost_capture_truncated_discharge_shortfall(self):
        env = _make_env(
            episode_length=4,
            time_step=0.25,
            use_flow_rate_action=False,
            clip_soc_to_bounds=False,
            battery_capacity_kwh=0.001,
            battery_power_kw=0.001,
            battery_discharge_power_kw=0.001,
            battery_efficiency=1.0,
            synthetic_pv_peak_kw=0.0,
            synthetic_load_base_kw=0.001,
            allow_grid_export=True,
            discharge_auto=False,
            voltage_cutoff_soc=0.0,
        )
        env.soc_physical_floor = 0.0
        env.reset(seed=0)
        env.current_soc = 0.10

        _, _, _, _, info = env.step([-0.001])

        assert info["current_soc"] == pytest.approx(0.0)
        assert info["applied_action_kw"] == pytest.approx(-0.0004)
        assert info["grid_import_kw"] == pytest.approx(0.0006)
        assert info["total_cost"] == pytest.approx(
            info["grid_import_kw"] * env.time_step * env.episode_data["price"][0]
        )


class TestVoltageCutoff:
    """voltage_cutoff_soc > 0 blocks discharge below the threshold."""

    def test_cutoff_prevents_excessive_discharge(self):
        env = _make_env(
            voltage_cutoff_soc=0.15,
            discharge_auto=False,
        )
        obs, _ = env.reset()

        # Public-facing comment translated to English.
        for _ in range(200):
            action = [-1.0, 0.5]
            obs, _, done, trunc, _ = env.step(action)
            if done or trunc:
                break

        # Public-facing comment translated to English.
        # Public-facing comment translated to English.
        assert obs[0] >= 0.09

    def test_cutoff_allows_charge(self):
        """Charging remains allowed below cutoff."""
        env = _make_env(voltage_cutoff_soc=0.15)
        obs, _ = env.reset()

        # Public-facing comment translated to English.
        action = [1.0, 0.5]
        obs2, _, _, _, _ = env.step(action)
        # Public-facing comment translated to English.
        assert obs2[0] >= obs[0] - 0.001


class TestV16spNoPvPenalty:

    def test_no_pv_throughput_penalty_reduces_reward(self):
        env = _make_env(use_flow_rate_action=False)
        env.current_step = 0  # off-peak
        env.current_soc = 0.5
        env.no_pv_action_threshold_kw = 0.001
        env.no_pv_throughput_penalty_per_kwh = 4.0
        env.offpeak_no_pv_discharge_penalty_per_kwh = 0.0

        penalized = env._calculate_reward_v16sp(
            action_kw=0.0004,
            load_kw=0.0004,
            pv_kw=0.0,
            price=2.06,
            net_load_after_pv=0.0004,
            useful_discharge=0.0,
            charge_kw=0.0004,
            grid_kw=0.0008,
            baseline_grid_kw=0.0004,
            pump_power_kw=0.0,
            pv_to_battery=0.0,
        )

        env.no_pv_throughput_penalty_per_kwh = 0.0
        baseline = env._calculate_reward_v16sp(
            action_kw=0.0004,
            load_kw=0.0004,
            pv_kw=0.0,
            price=2.06,
            net_load_after_pv=0.0004,
            useful_discharge=0.0,
            charge_kw=0.0004,
            grid_kw=0.0008,
            baseline_grid_kw=0.0004,
            pump_power_kw=0.0,
            pv_to_battery=0.0,
        )

        assert penalized < baseline

    def test_offpeak_no_pv_discharge_penalty_only_hits_offpeak_discharge(self):
        env = _make_env(use_flow_rate_action=False)
        env.no_pv_action_threshold_kw = 0.001
        env.no_pv_throughput_penalty_per_kwh = 0.0
        env.offpeak_no_pv_discharge_penalty_per_kwh = 6.0
        env.current_soc = 0.7

        env.current_step = 0  # off-peak
        offpeak = env._calculate_reward_v16sp(
            action_kw=-0.0004,
            load_kw=0.0004,
            pv_kw=0.0,
            price=2.06,
            net_load_after_pv=0.0004,
            useful_discharge=0.0004,
            charge_kw=0.0,
            grid_kw=0.0,
            baseline_grid_kw=0.0004,
            pump_power_kw=0.0,
            pv_to_battery=0.0,
        )

        env.current_step = 40  # 10:00, mid-peak weekday
        midpeak = env._calculate_reward_v16sp(
            action_kw=-0.0004,
            load_kw=0.0004,
            pv_kw=0.0,
            price=4.69,
            net_load_after_pv=0.0004,
            useful_discharge=0.0004,
            charge_kw=0.0,
            grid_kw=0.0,
            baseline_grid_kw=0.0004,
            pump_power_kw=0.0,
            pv_to_battery=0.0,
        )

        assert offpeak < midpeak


class TestPumpAccounting:

    def test_uncontrolled_flow_charges_full_pump_power_every_step(self):
        env = _make_env(
            use_flow_rate_action=False,
            fixed_flow_fraction_when_uncontrolled=1.0,
            flow_P_max_pump_W=1000.0,
        )
        env.reset(seed=0)

        _, _, _, _, info = env.step([0.0])

        assert env.action_space.shape == (1,)
        assert info["flow_action"] == pytest.approx(1.0)
        assert info["pump_power_kw"] == pytest.approx(1.0)
        assert info["pump_energy_kwh"] == pytest.approx(0.25)
        assert info["grid_import_kw"] == pytest.approx(info["net_load_after_pv"] + 1.0)
        assert info["pump_cost"] == pytest.approx(0.25 * info["price"])

    def test_flow_control_uses_cubic_pump_curve_when_active(self):
        env = _make_env(
            use_flow_rate_action=True,
            flow_pump_from_grid=True,
            flow_P_max_pump_W=1000.0,
        )
        env.reset(seed=0)

        _, _, _, _, info = env.step([0.0005, 0.5])

        expected_pump_kw = 1.0 * (0.5 ** 3)
        assert info["flow_action"] == pytest.approx(0.5)
        assert info["pump_power_kw"] == pytest.approx(expected_pump_kw)
        assert info["pump_energy_kwh"] == pytest.approx(expected_pump_kw * 0.25)
        assert info["grid_import_kw"] == pytest.approx(
            info["net_load_after_pv"] + info["applied_action_kw"] + expected_pump_kw
        )

    def test_grid_supplied_pump_is_not_counted_twice_while_charging(self):
        env = _make_env(
            use_flow_rate_action=True,
            flow_pump_from_grid=True,
            flow_P_max_pump_W=1.0,
        )
        env.reset(seed=0)

        _, _, _, _, info = env.step([0.0005, 0.5])

        expected_pump_kw = 0.001 * (0.5 ** 3)
        assert info["net_power_kw"] == pytest.approx(info["applied_action_kw"])
        assert info["costed_pump_power_kw"] == pytest.approx(expected_pump_kw)
        assert info["grid_import_kw"] == pytest.approx(
            info["net_load_after_pv"]
            + info["applied_action_kw"]
            + expected_pump_kw
        )

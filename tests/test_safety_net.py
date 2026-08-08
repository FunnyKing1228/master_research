"""
Unit tests for core/safety_net.py
==================================
Covers:
  - SafetyNet bounds
  - CORAL projection into the safe set
  - near-boundary behavior: low-SoC discharge block and high-SoC charge block
  - no intervention in the middle SoC range
"""
import os
import sys
import pytest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)

from core.safety_net import SafetyNet


def _make_state(soc=0.5, load=0.0004, pv=0.0003, price=0.5, hour=12.0, dow=0.0):
    """Create a 6D state vector."""
    return np.array([soc, load, pv, price, hour, dow], dtype=np.float32)


class TestSafetyNetInit:

    def test_default_init(self):
        sn = SafetyNet(
            battery_capacity_kwh=0.00990,
            battery_power_kw=0.000825,
            battery_efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9,
        )
        assert sn.battery_capacity_kwh == 0.00990
        assert sn.battery_power_kw == 0.000825
        assert sn.safe_soc_min > sn.soc_min  # See test assertion for expected behavior.
        assert sn.safe_soc_max < sn.soc_max  # See test assertion for expected behavior.


class TestSafetyNetBounds:

    @pytest.fixture
    def sn(self):
        return SafetyNet(
            battery_capacity_kwh=0.00990,
            battery_power_kw=0.000825,
            battery_efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9,
        )

    def test_bounds_mid_soc(self, sn):
        """Middle SoC bounds should be close to maximum power."""
        state = _make_state(soc=0.5)
        low, high = sn.bounds(state)
        assert low < 0  # See test assertion for expected behavior.
        assert high > 0  # See test assertion for expected behavior.

    def test_bounds_low_soc(self, sn):
        """Near lower SoC bound, discharge bound should shrink."""
        state = _make_state(soc=0.12)
        low, high = sn.bounds(state)
        # Public-facing comment translated to English.
        assert low > -sn.battery_power_kw  # See test assertion for expected behavior.

    def test_bounds_high_soc(self, sn):
        """Near upper SoC bound, charge bound should shrink."""
        state = _make_state(soc=0.88)
        low, high = sn.bounds(state)
        # Public-facing comment translated to English.
        assert high < sn.battery_power_kw  # See test assertion for expected behavior.

    def test_bounds_valid_range(self, sn):
        """Lower bound should not exceed upper bound."""
        for soc in [0.1, 0.2, 0.5, 0.8, 0.9]:
            state = _make_state(soc=soc)
            low, high = sn.bounds(state)
            assert low <= high + 1e-9, f"Invalid bounds at SoC={soc}: [{low}, {high}]"


class TestSafetyNetProject:

    @pytest.fixture
    def sn(self):
        return SafetyNet(
            battery_capacity_kwh=0.00990,
            battery_power_kw=0.000825,
            battery_efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9,
        )

    def test_safe_action_unchanged(self, sn):
        """Actions inside the safe range are unchanged."""
        state = _make_state(soc=0.5)
        action_raw = np.array([0.0], dtype=np.float32)  # See test assertion for expected behavior.
        action_safe, info = sn.project(state, action_raw)
        assert not info['clipped']
        assert action_safe[0] == pytest.approx(0.0, abs=1e-6)

    def test_discharge_clipped_at_low_soc(self, sn):
        """Discharge is clipped at low SoC."""
        state = _make_state(soc=0.11)
        action_raw = np.array([-0.000825], dtype=np.float32)  # See test assertion for expected behavior.
        action_safe, info = sn.project(state, action_raw)
        # Public-facing comment translated to English.
        assert info['clipped'] or abs(action_safe[0]) < abs(action_raw[0])

    def test_charge_clipped_at_high_soc(self, sn):
        """Charge is clipped at high SoC."""
        state = _make_state(soc=0.89)
        action_raw = np.array([0.000825], dtype=np.float32)  # See test assertion for expected behavior.
        action_safe, info = sn.project(state, action_raw)
        # Public-facing comment translated to English.
        assert info['clipped'] or action_safe[0] < action_raw[0]

    def test_mid_soc_no_clip(self, sn):
        """Small actions at middle SoC are not clipped."""
        state = _make_state(soc=0.5)
        # Public-facing comment translated to English.
        action_raw = np.array([0.0001], dtype=np.float32)
        action_safe, info = sn.project(state, action_raw)
        assert not info['clipped']
        assert action_safe[0] == pytest.approx(0.0001, abs=1e-6)

    def test_project_returns_correct_shape(self, sn):
        state = _make_state(soc=0.5)
        action_raw = np.array([0.0], dtype=np.float32)
        action_safe, info = sn.project(state, action_raw)
        assert action_safe.shape == (1,)
        assert isinstance(info, dict)
        assert 'clipped' in info
        assert 'safe_bounds' in info


class TestSafetyMetrics:

    def test_safety_level_safe(self):
        sn = SafetyNet(
            battery_capacity_kwh=0.00990,
            battery_power_kw=0.000825,
            soc_min=0.1, soc_max=0.9,
        )
        state = _make_state(soc=0.5)
        metrics = sn.get_safety_metrics(state)
        assert metrics['safety_level'] == 'safe'

    def test_safety_level_danger_low(self):
        sn = SafetyNet(
            battery_capacity_kwh=0.00990,
            battery_power_kw=0.000825,
            soc_min=0.1, soc_max=0.9,
        )
        state = _make_state(soc=0.05)  # below soc_min
        metrics = sn.get_safety_metrics(state)
        assert 'danger' in metrics['safety_level'] or 'warning' in metrics['safety_level']

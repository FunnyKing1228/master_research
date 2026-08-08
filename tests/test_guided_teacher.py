import os
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core'))

from core.train_sac_microgrid import guided_teacher_action_kw


def _make_dummy_env(load_kw, pv_kw, price, soc, pv_bool=None):
    if pv_bool is None:
        pv_bool = 1.0 if pv_kw > 0.001 else 0.0
    return SimpleNamespace(
        current_step=0,
        current_soc=soc,
        battery_charge_power_kw=0.0085,
        battery_discharge_power_kw=0.0056,
        pv_obs_boolean_threshold_kw=0.001,
        episode_data={
            'load': [load_kw],
            'pv': [pv_kw],
            'price': [price],
            'pv_bool': [pv_bool],
        },
    )


def test_teacher_charges_when_pv_can_almost_cover_load():
    env = _make_dummy_env(load_kw=0.0023, pv_kw=0.0022, price=2.06, soc=0.30, pv_bool=1.0)
    action_kw = guided_teacher_action_kw(
        env,
        {
            'solar_charge_soc_target': 0.85,
            'pv_cover_charge_threshold': 0.95,
            'pv_cover_charge_min_frac': 0.35,
        },
    )
    assert action_kw > 0.0


def test_teacher_does_not_charge_without_pv_signal():
    env = _make_dummy_env(load_kw=0.0023, pv_kw=0.0016, price=2.06, soc=0.30, pv_bool=0.0)
    action_kw = guided_teacher_action_kw(
        env,
        {
            'solar_charge_soc_target': 0.85,
            'pv_cover_charge_threshold': 0.95,
            'pv_cover_charge_min_frac': 0.35,
        },
    )
    assert action_kw == pytest.approx(0.0, abs=1e-9)


def test_teacher_discharges_at_peak_without_pv_when_load_is_feasible():
    env = _make_dummy_env(load_kw=0.0023, pv_kw=0.0, price=7.13, soc=0.70, pv_bool=0.0)
    action_kw = guided_teacher_action_kw(env, {'peak_discharge_soc_floor': 0.20})
    assert action_kw == pytest.approx(-0.0023, abs=1e-6)


def test_teacher_keeps_battery_idle_offpeak_without_pv():
    env = _make_dummy_env(load_kw=0.0023, pv_kw=0.0, price=2.06, soc=0.70, pv_bool=0.0)
    action_kw = guided_teacher_action_kw(env, {'peak_discharge_soc_floor': 0.20})
    assert action_kw == pytest.approx(0.0, abs=1e-9)


def test_teacher_does_not_discharge_when_pv_is_present_but_not_sufficient():
    env = _make_dummy_env(load_kw=0.0023, pv_kw=0.0012, price=7.13, soc=0.70, pv_bool=0.0)
    action_kw = guided_teacher_action_kw(env, {'peak_discharge_soc_floor': 0.20})
    assert action_kw == pytest.approx(0.0, abs=1e-9)

import os
import sys

import numpy as np
import pytest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core'))

from core.train_sac_microgrid import compute_occ_proxy


class DummyEnv:
    soc_min = 0.10
    soc_max = 0.90
    action_dead_zone_kw = 0.00005


def test_occ_proxy_projection_only_matches_legacy_behavior():
    env = DummyEnv()
    state = np.array([0.11], dtype=np.float32)

    occ_proxy = compute_occ_proxy(
        config={},
        env=env,
        state=state,
        action_raw_kw=-0.002,
        action_safe_kw=0.0,
        delta_kw=0.0014,
        pmax=0.0056,
    )

    assert occ_proxy == pytest.approx(0.25, abs=1e-6)


def test_occ_proxy_boundary_aware_penalizes_low_soc_discharge():
    env = DummyEnv()
    state = np.array([0.11], dtype=np.float32)
    config = {
        'occ_proxy': {
            'mode': 'boundary_aware',
            'delta_weight': 1.0,
            'low_soc_weight': 0.8,
            'high_soc_weight': 0.6,
            'low_soc_threshold': 0.12,
            'high_soc_threshold': 0.88,
            'clamp_max': 2.0,
        }
    }

    occ_proxy = compute_occ_proxy(
        config=config,
        env=env,
        state=state,
        action_raw_kw=-0.002,
        action_safe_kw=0.0,
        delta_kw=0.0014,
        pmax=0.0056,
    )

    # legacy projection = 0.25; low-SoC discharge risk adds 0.4
    assert occ_proxy == pytest.approx(0.65, abs=1e-6)


def test_occ_proxy_boundary_aware_penalizes_high_soc_charge():
    env = DummyEnv()
    state = np.array([0.89], dtype=np.float32)
    config = {
        'occ_proxy': {
            'mode': 'boundary_aware',
            'delta_weight': 1.0,
            'low_soc_weight': 0.8,
            'high_soc_weight': 0.6,
            'low_soc_threshold': 0.12,
            'high_soc_threshold': 0.88,
            'clamp_max': 2.0,
        }
    }

    occ_proxy = compute_occ_proxy(
        config=config,
        env=env,
        state=state,
        action_raw_kw=0.002,
        action_safe_kw=0.0,
        delta_kw=0.0014,
        pmax=0.0056,
    )

    # legacy projection = 0.25; high-SoC charge risk adds 0.3
    assert occ_proxy == pytest.approx(0.55, abs=1e-6)

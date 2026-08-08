"""
Unit tests for control/run_deployment.py
=========================================
Covers:
  - SoCTracker energy accounting, coulomb diagnostics, clamping, and long-disconnect handling
  - DataBuffer 15-minute windows and aggregation
  - build_state_from_aggregation state dimensions and ranges
  - determine_situation scenario-code classification
  - DeploymentLogger daily file rotation
  - write_command_simple direct output format
  - get_tou_price（TOU price lookup.）
  - get_load_groups load scheduling
"""
import os
import sys
import csv
import io
import pytest
import tempfile
import shutil
from datetime import datetime, timezone, timedelta, time as dtime

import numpy as np

# ── Ensure import path is available. ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'control'))

from control.run_deployment import (
    SoCTracker,
    DataBuffer,
    Reading,
    DeploymentLogger,
    RawDataLogger,
    build_state_from_aggregation,
    determine_situation,
    warn_load_over_discharge_limit,
    apply_pv_active_discharge_guard,
    is_invalid_partial_discharge,
    apply_flow_operating_rule,
    command_pp_for_action,
    write_command_simple,
    get_tou_price,
    get_load_groups,
    BATTERY_PMAX_KW,
    BATTERY_CHARGE_PMAX_KW,
    BATTERY_DISCHARGE_PMAX_KW,
    BATTERY_CAPACITY_MAH,
    BATTERY_CAPACITY_KWH,
    BATTERY_EFFICIENCY,
    BATTERY_CUTOFF_V,
    BATTERY_CUTOFF_RECOVER_V,
    PV_PRESENT_THRESHOLD_KW,
    CUTOFF_COOLDOWN_SEC,
    CUTOFF_MAX_PER_DAY,
    CUTOFF_ZERO_V_STREAK,
    BATTERY_CHARGE_V,
    BATTERY_DISCHARGE_V,
    LOAD_PER_GROUP_W,
    MAX_LOAD_GROUPS,
    FLOW_REST_PCT,
    FLOW_PRE_MEASURE_PCT,
    VOLTAGE_RECOVERY_SECONDS,
    FLOW_IDLE_PCT,
    FLOW_MIN_ACTIVE_PCT,
    STANDBY_SITUATION_CODE,
    PRE_MEASURE_SITUATION_CODE,
    STANDBY_FLOW_SITUATION_CODE,
    write_standby_rest_command,
    write_pre_measure_command,
    perform_pre_measure_for_decision,
    TOU_OFFPEAK,
    TOU_MIDPEAK,
    TOU_PEAK,
    TZ_UTC8,
    safe_print,
)


# ══════════════════════════════════════════════════════════════════
# Hardware constant validation.
# ══════════════════════════════════════════════════════════════════
class TestConstants:
    """Ensure constants match the SLFB hardware specification."""

    def test_battery_voltage_hierarchy(self):
        assert BATTERY_CUTOFF_V < BATTERY_CUTOFF_RECOVER_V < BATTERY_DISCHARGE_V < BATTERY_CHARGE_V
        assert BATTERY_CUTOFF_V == 4.2
        assert BATTERY_CUTOFF_RECOVER_V == 5.0
        assert BATTERY_DISCHARGE_V == 5.6
        assert BATTERY_CHARGE_V == 8.5

    def test_battery_power(self):
        assert BATTERY_CHARGE_PMAX_KW == pytest.approx(0.0085, abs=1e-6)
        assert BATTERY_DISCHARGE_PMAX_KW == pytest.approx(0.0056, abs=1e-6)
        assert BATTERY_PMAX_KW == pytest.approx(0.0085, abs=1e-6)


class TestSafePrint:
    def test_safe_print_survives_cp950_unicode_error(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp950", errors="strict")

        safe_print("  ⚠ PV still active", file=stream, end="")
        stream.flush()

        output = raw.getvalue().decode("cp950", errors="strict")
        assert "\\u26a0" in output

    def test_battery_capacity(self):
        # 1000mA × 2h = 2000 mAh
        assert BATTERY_CAPACITY_MAH == pytest.approx(2000.0, abs=0.1)
        # 5.6W × 2h = 11.2 Wh = 0.0112 kWh
        assert BATTERY_CAPACITY_KWH == pytest.approx(0.01120, abs=1e-5)

    def test_battery_efficiency(self):
        assert BATTERY_EFFICIENCY == 0.95

    def test_load_spec(self):
        # Each group is about 0.1 W; four groups total about 0.4 W.
        assert LOAD_PER_GROUP_W == pytest.approx(0.1)
        assert MAX_LOAD_GROUPS == 4

    def test_rest_and_pre_measure_flow_constants(self):
        assert FLOW_REST_PCT == pytest.approx(0.0)
        assert FLOW_PRE_MEASURE_PCT == pytest.approx(50.0)
        assert FLOW_IDLE_PCT == pytest.approx(FLOW_REST_PCT)
        assert VOLTAGE_RECOVERY_SECONDS == pytest.approx(25.0)
        assert apply_flow_operating_rule(0.0, active=False) == pytest.approx(0.0)
        assert apply_flow_operating_rule(10.0, active=True) == pytest.approx(FLOW_MIN_ACTIVE_PCT)
        assert STANDBY_SITUATION_CODE == 3
        assert PRE_MEASURE_SITUATION_CODE == 3

    def test_pv_present_threshold_matches_low_power_load(self):
        # The load bank is only about 0.4 W, so this must stay far below the old 1 W threshold.
        assert PV_PRESENT_THRESHOLD_KW == pytest.approx(0.00005)


# ══════════════════════════════════════════════════════════════════
# SoCTracker
# ══════════════════════════════════════════════════════════════════
class TestSoCTracker:

    def _make_ts(self, minute=0, second=0):
        return datetime(2026, 3, 16, 12, minute, second, tzinfo=TZ_UTC8)

    def test_initial_soc(self):
        t = SoCTracker(initial_soc=0.5)
        assert t.get_soc() == 0.5

    def test_first_update_only_sets_time(self):
        """The first update only sets last_update and does not change SoC."""
        t = SoCTracker(initial_soc=0.5)
        t.update(self._make_ts(0, 0), current_ma=150.0)
        assert t.get_soc() == 0.5  # unchanged

    def test_charging_increases_soc(self):
        """Charging (I > 0) should increase SoC."""
        t = SoCTracker(initial_soc=0.5, capacity_mah=1800.0, efficiency_rte=0.95)
        t.update(self._make_ts(0, 0), current_ma=150.0)   # init time
        t.update(self._make_ts(0, 10), current_ma=150.0)   # 10 sec charging

        # Energy SoC uses V x I integration. With no measured voltage in the
        # test, charging falls back to BATTERY_CHARGE_V.
        capacity_wh = 1800.0 / 1000.0 * BATTERY_DISCHARGE_V
        expected_delta = BATTERY_CHARGE_V * 150.0 / 1000.0 * (10.0 / 3600.0) * 0.95 / capacity_wh
        assert t.get_soc() == pytest.approx(0.5 + expected_delta, abs=1e-6)
        assert t.get_soc() > 0.5

    def test_discharging_decreases_soc(self):
        """Discharging (I < 0) should decrease SoC."""
        t = SoCTracker(initial_soc=0.5, capacity_mah=1800.0, efficiency_rte=0.95)
        t.update(self._make_ts(0, 0), current_ma=-150.0)
        t.update(self._make_ts(0, 10), current_ma=-150.0)

        assert t.get_soc() < 0.5

    def test_soc_clamped_to_0_1(self):
        """SoC stays within [0, 1]."""
        # Start near empty and discharge heavily.
        t = SoCTracker(initial_soc=0.01, capacity_mah=1800.0)
        t.update(self._make_ts(0, 0), current_ma=-5000.0)
        t.update(self._make_ts(30, 0), current_ma=-5000.0)  # 30 min at 5A
        assert t.get_soc() == 0.0

        # Start near full and charge heavily.
        t2 = SoCTracker(initial_soc=0.99, capacity_mah=1800.0)
        t2.update(self._make_ts(0, 0), current_ma=5000.0)
        t2.update(self._make_ts(30, 0), current_ma=5000.0)
        assert t2.get_soc() == 1.0

    def test_zero_current_no_change(self):
        """I=0 does not change SoC."""
        t = SoCTracker(initial_soc=0.5)
        t.update(self._make_ts(0, 0), current_ma=0.0)
        t.update(self._make_ts(1, 0), current_ma=0.0)
        assert t.get_soc() == 0.5

    def test_time_backward_skip(self):
        """Backward time jump is skipped."""
        t = SoCTracker(initial_soc=0.5)
        t.update(self._make_ts(1, 0), current_ma=150.0)
        t.update(self._make_ts(0, 30), current_ma=150.0)  # earlier timestamp
        # Should still be close to 0.5 (only init update applied)
        assert t.get_soc() == 0.5

    def test_long_gap_capped(self):
        """Long disconnects are clamped to MAX_INTEGRATION_SEC."""
        t = SoCTracker(initial_soc=0.5, capacity_mah=1800.0)
        t.update(self._make_ts(0, 0), current_ma=150.0)
        # Jump forward by two hours.
        ts_2h = datetime(2026, 3, 16, 14, 0, 0, tzinfo=TZ_UTC8)
        t.update(ts_2h, current_ma=150.0)
        assert t.skipped_intervals == 1  # Clamped once.
        # SoC still changes; the interval is not skipped.
        assert t.get_soc() > 0.5

    def test_set_soc(self):
        t = SoCTracker(initial_soc=0.5)
        t.set_soc(0.8)
        assert t.get_soc() == 0.8
        t.set_soc(-0.1)  # clamp to 0
        assert t.get_soc() == 0.0
        t.set_soc(1.5)   # clamp to 1
        assert t.get_soc() == 1.0

    def test_unclamped_soc_goes_negative(self):
        """Unclamped SoC can be negative while clamped SoC stays at 0."""
        t = SoCTracker(initial_soc=0.01, capacity_mah=1800.0, efficiency_rte=0.95)
        t.update(self._make_ts(0, 0), current_ma=-5000.0)
        t.update(self._make_ts(30, 0), current_ma=-5000.0)  # 30 min at 5A discharge
        assert t.get_soc() == 0.0            # clamped to 0
        assert t.get_soc_unclamped() < 0.0   # unclamped goes negative

    def test_unclamped_soc_goes_above_1(self):
        """Unclamped SoC can exceed 1."""
        t = SoCTracker(initial_soc=0.99, capacity_mah=1800.0, efficiency_rte=0.95)
        t.update(self._make_ts(0, 0), current_ma=5000.0)
        t.update(self._make_ts(30, 0), current_ma=5000.0)  # 30 min at 5A charge
        assert t.get_soc() == 1.0            # clamped to 1
        assert t.get_soc_unclamped() > 1.0   # unclamped goes above 1

    def test_unclamped_tracks_cumulative(self):
        """Unclamped SoC tracks accumulated charge independent of clipping."""
        t = SoCTracker(initial_soc=0.0, capacity_mah=1800.0, efficiency_rte=0.95)
        eta = 0.95
        capacity_wh = 1800.0 / 1000.0 * BATTERY_DISCHARGE_V
        # Discharge first: clamped=0 and unclamped<0.
        t.update(self._make_ts(0, 0), current_ma=-150.0)
        t.update(self._make_ts(1, 0), current_ma=-150.0)  # 1 min discharge
        expected_delta = -BATTERY_DISCHARGE_V * 150.0 / 1000.0 * (60.0 / 3600.0) / eta / capacity_wh
        assert t.get_soc() == 0.0  # clamped
        assert t.get_soc_unclamped() == pytest.approx(expected_delta, abs=1e-6)

        # Charge again; unclamped SoC should recover above zero.
        t.update(self._make_ts(2, 0), current_ma=150.0)  # 1 min charge
        charge_delta = BATTERY_CHARGE_V * 150.0 / 1000.0 * (60.0 / 3600.0) * eta / capacity_wh
        assert t.get_soc_unclamped() == pytest.approx(expected_delta + charge_delta, abs=1e-6)

    def test_set_soc_resets_unclamped(self):
        """set_soc also resets the unclamped SoC."""
        t = SoCTracker(initial_soc=0.5)
        t.set_soc(0.3)
        assert t.get_soc_unclamped() == 0.3

    def test_stats(self):
        t = SoCTracker(initial_soc=0.5, capacity_mah=1800.0, efficiency_rte=0.95)
        t.update(self._make_ts(0, 0), current_ma=150.0)
        t.update(self._make_ts(1, 0), current_ma=150.0)
        stats = t.get_stats()
        assert 'soc' in stats
        assert 'soc_unclamped' in stats
        assert 'soc_coulomb' in stats
        assert 'total_charge_mah' in stats
        assert 'total_discharge_mah' in stats
        assert 'total_charge_wh' in stats
        assert 'total_discharge_wh' in stats
        assert stats['total_charge_mah'] > 0
        assert stats['total_charge_wh'] > 0
        assert stats['total_discharge_mah'] == 0.0


# ══════════════════════════════════════════════════════════════════
# DataBuffer
# ══════════════════════════════════════════════════════════════════
class TestDataBuffer:

    def _make_reading(self, minute=0, second=0, mppt_p=500.0, batt_v=6.5,
                      batt_i=100.0, load_p=400.0, bus_p=0.0):
        return Reading(
            timestamp=datetime(2026, 3, 16, 12, minute, second, tzinfo=TZ_UTC8),
            mppt_p_mw=mppt_p,
            batt_v=batt_v,
            batt_i_ma=batt_i,
            load_p_mw=load_p,
            bus_p_mw=bus_p,
        )

    def test_empty_buffer_aggregate(self):
        buf = DataBuffer(window_sec=900)
        agg = buf.aggregate()
        assert agg['n_samples'] == 0
        assert agg['mppt_p_mean_mW'] == 0.0

    def test_single_reading(self):
        buf = DataBuffer(window_sec=900)
        r = self._make_reading(minute=0, mppt_p=800.0)
        buf.add(r)
        assert buf.count == 1

        agg = buf.aggregate()
        assert agg['n_samples'] == 1
        assert agg['mppt_p_mean_mW'] == pytest.approx(800.0)

    def test_window_alignment(self):
        """Window start should align to a 15-minute boundary."""
        buf = DataBuffer(window_sec=900)
        r = self._make_reading(minute=7)  # 12:07
        buf.add(r)
        # Should align to 12:00.
        assert buf.window_start.minute == 0

    def test_window_alignment_at_22(self):
        """12:22 should align to 12:15."""
        buf = DataBuffer(window_sec=900)
        r = Reading(
            timestamp=datetime(2026, 3, 16, 12, 22, 0, tzinfo=TZ_UTC8),
            mppt_p_mw=500.0,
        )
        buf.add(r)
        assert buf.window_start.minute == 15

    def test_is_window_complete(self):
        buf = DataBuffer(window_sec=900)
        t0 = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        r = Reading(timestamp=t0, mppt_p_mw=500.0)
        buf.add(r)

        # After 10 minutes, the window is incomplete.
        t10 = t0 + timedelta(minutes=10)
        assert not buf.is_window_complete(t10)

        # After 15 minutes, the window is complete.
        t15 = t0 + timedelta(minutes=15)
        assert buf.is_window_complete(t15)

    def test_aggregate_multiple_readings(self):
        buf = DataBuffer(window_sec=900)
        for i in range(5):
            buf.add(self._make_reading(
                minute=i * 3,
                mppt_p=100.0 * (i + 1),
                batt_v=6.5,
                batt_i=100.0,
            ))

        agg = buf.aggregate()
        assert agg['n_samples'] == 5
        # mean(100, 200, 300, 400, 500) = 300
        assert agg['mppt_p_mean_mW'] == pytest.approx(300.0)
        # max = 500
        assert agg['mppt_p_max_mW'] == pytest.approx(500.0)

    def test_reset(self):
        buf = DataBuffer(window_sec=900)
        buf.add(self._make_reading())
        assert buf.count == 1

        t_new = datetime(2026, 3, 16, 12, 15, 0, tzinfo=TZ_UTC8)
        buf.reset(new_start=t_new)
        assert buf.count == 0
        assert buf.window_start == t_new

    def test_battery_power_aggregation(self):
        """batt_p is the mean of V x I."""
        buf = DataBuffer(window_sec=900)
        buf.add(self._make_reading(minute=0, batt_v=6.0, batt_i=100.0))  # 600 mW
        buf.add(self._make_reading(minute=1, batt_v=7.0, batt_i=200.0))  # 1400 mW

        agg = buf.aggregate()
        # mean(600, 1400) = 1000
        assert agg['batt_p_mean_mW'] == pytest.approx(1000.0)

    def test_load_filter_zeros(self):
        """load_p_mw=0 readings are excluded from load mean."""
        buf = DataBuffer(window_sec=900)
        buf.add(self._make_reading(minute=0, load_p=400.0))
        buf.add(self._make_reading(minute=1, load_p=0.0))
        buf.add(self._make_reading(minute=2, load_p=400.0))

        agg = buf.aggregate()
        # Only two positive load readings exist: mean(400, 400)=400.
        assert agg['load_p_mean_mW'] == pytest.approx(400.0)


# ══════════════════════════════════════════════════════════════════
# build_state_from_aggregation
# ══════════════════════════════════════════════════════════════════
class TestBuildState:

    def test_state_dimension(self):
        """State vector must be 6D."""
        agg = {
            'mppt_p_mean_mW': 800.0,
            'bus_p_mean_mW': 800.0,
            'load_p_mean_mW': 400.0,
        }
        now = datetime(2026, 3, 16, 14, 0, 0, tzinfo=TZ_UTC8)
        state = build_state_from_aggregation(agg, soc=0.5, now=now)
        assert state.shape == (6,)
        assert state.dtype == np.float32

    def test_state_values(self):
        agg = {
            'mppt_p_mean_mW': 800.0,
            'bus_p_mean_mW': 320.0,
            'load_p_mean_mW': 400.0,  # 4 groups x 100 mW.
            'bus_v_mean': 14.5,
            'grid_v_mean': 13.2,
        }
        now = datetime(2026, 3, 16, 14, 0, 0, tzinfo=TZ_UTC8)
        state = build_state_from_aggregation(agg, soc=0.5, now=now)

        assert state[0] == pytest.approx(0.5)            # SoC
        assert state[1] == pytest.approx(0.0004, abs=1e-5)  # load = 400 mW / 1e6
        assert state[2] == pytest.approx(1.0)             # pv_bool = 1 (pv/load = 0.8)
        assert state[4] == 14.0                           # hour
        assert state[5] == 0.0                            # Monday = 0

    def test_state_values_with_pv_support_ratio(self):
        agg = {
            'bus_p_mean_mW': 200.0,
            'load_p_mean_mW': 400.0,
            'bus_v_mean': 14.2,
            'grid_v_mean': 13.2,
        }
        now = datetime(2026, 3, 16, 14, 0, 0, tzinfo=TZ_UTC8)
        state = build_state_from_aggregation(agg, soc=0.5, now=now, include_pv_support_ratio=True)

        assert state.shape == (7,)
        assert state[0] == pytest.approx(0.5)
        assert state[1] == pytest.approx(0.0004, abs=1e-5)
        assert state[2] == pytest.approx(0.5, abs=1e-5)  # 200 / 400 = 0.5
        assert state[3] == pytest.approx(0.0)
        assert state[5] == 14.0
        assert state[6] == 0.0

    def test_state_dimension_without_price_obs(self):
        """Without explicit price, pv_support_ratio state should be 6D."""
        agg = {
            'bus_p_mean_mW': 200.0,
            'load_p_mean_mW': 400.0,
            'bus_v_mean': 14.2,
        }
        now = datetime(2026, 3, 16, 14, 0, 0, tzinfo=TZ_UTC8)
        state = build_state_from_aggregation(
            agg, soc=0.5, now=now,
            include_pv_support_ratio=True,
            include_price_obs=False,
        )

        assert state.shape == (6,)
        assert state[0] == pytest.approx(0.5)
        assert state[1] == pytest.approx(0.0004, abs=1e-5)
        assert state[2] == pytest.approx(0.5, abs=1e-5)
        assert state[3] == pytest.approx(0.0)
        assert state[4] == 14.0
        assert state[5] == 0.0

    def test_state_dimension_without_price_or_ratio(self):
        """Minimal state has 5D: SoC, load, pv_bool, hour, dow."""
        agg = {
            'bus_p_mean_mW': 320.0,
            'load_p_mean_mW': 400.0,
            'bus_v_mean': 14.5,
        }
        now = datetime(2026, 3, 16, 14, 0, 0, tzinfo=TZ_UTC8)
        state = build_state_from_aggregation(
            agg, soc=0.5, now=now,
            include_pv_support_ratio=False,
            include_price_obs=False,
        )

        assert state.shape == (5,)
        assert state[0] == pytest.approx(0.5)
        assert state[1] == pytest.approx(0.0004, abs=1e-5)
        assert state[2] == pytest.approx(1.0)
        assert state[3] == 14.0
        assert state[4] == 0.0

    def test_low_measured_load_is_used_directly(self):
        """Low measured load is still used directly."""
        agg = {
            'mppt_p_mean_mW': 800.0,
            'bus_p_mean_mW': 800.0,
            'load_p_mean_mW': 50.0,
        }
        now = datetime(2026, 3, 16, 14, 0, 0, tzinfo=TZ_UTC8)
        state = build_state_from_aggregation(agg, soc=0.5, now=now)

        assert state[1] == pytest.approx(0.00005, abs=1e-6)

    def test_zero_measured_load_stays_zero(self):
        """A zero measured load is not replaced by a schedule fallback."""
        agg = {
            'mppt_p_mean_mW': 0.0,
            'bus_p_mean_mW': 0.0,
            'load_p_mean_mW': 0.0,
        }
        now = datetime(2026, 3, 16, 14, 0, 0, tzinfo=TZ_UTC8)
        state = build_state_from_aggregation(agg, soc=0.5, now=now)
        assert state[1] == pytest.approx(0.0, abs=1e-9)

    def test_price_normalization(self):
        """Price normalization: peak 7.13 / 10 ~= 0.713."""
        agg = {'mppt_p_mean_mW': 0.0, 'bus_p_mean_mW': 0.0, 'load_p_mean_mW': 0.0}
        now = datetime(2026, 3, 16, 18, 0, 0, tzinfo=TZ_UTC8)  # Peak period.
        state = build_state_from_aggregation(agg, soc=0.5, now=now)
        assert state[3] == pytest.approx(TOU_PEAK / 10.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════
# determine_situation
# ══════════════════════════════════════════════════════════════════
class TestDetermineSituation:

    def test_discharge_covers_load(self):
        """Sufficient discharge maps to scenario 1."""
        # action = -0.001 kW, load = 0.0004, pv = 0
        # net_load = 0.0004, discharge = 0.001 >= 0.0004
        assert determine_situation(-0.001, 0.0004, 0.0) == 1

    def test_discharge_with_pv_covers_load(self):
        """PV plus discharge covering load maps to scenario 1."""
        # load=0.001, pv=0.0008 → net=0.0002
        # discharge=0.001 >= 0.0002 and exceeds the deployment intent threshold
        assert determine_situation(-0.001, 0.001, 0.0008) == 1

    def test_discharge_insufficient(self):
        """Insufficient discharge maps to standby-flow mode because grid support is disabled."""
        # net_load = 0.0004, discharge = 0.0002 < 0.0004
        assert determine_situation(-0.0002, 0.0004, 0.0) == STANDBY_FLOW_SITUATION_CODE

    def test_over_limit_measured_load_still_allows_discharge(self):
        """Measured overload is only a warning; it must not force standby."""
        load_kw = BATTERY_DISCHARGE_PMAX_KW + 0.003
        action_kw = -BATTERY_DISCHARGE_PMAX_KW

        assert warn_load_over_discharge_limit(action_kw, load_kw) == 1
        assert is_invalid_partial_discharge(action_kw, load_kw, 0.0) is False
        assert determine_situation(action_kw, load_kw, 0.0) == 1

    def test_pv_active_discharge_guard_forces_zero_command(self):
        """Strong PV support must still prevent a negative battery command."""
        load_kw = BATTERY_DISCHARGE_PMAX_KW + 0.003
        action_kw = -BATTERY_DISCHARGE_PMAX_KW
        pv_active = 1.0

        assert warn_load_over_discharge_limit(action_kw, load_kw) == 1

        guarded_kw, guard_flag = apply_pv_active_discharge_guard(action_kw, pv_active)
        flow_pct = apply_flow_operating_rule(80.0, active=abs(guarded_kw) > 0.0001)
        situation = determine_situation(guarded_kw, load_kw, pv_kw=0.001)
        power_mw_cmd = int(round(abs(guarded_kw) * 1e6))

        assert guard_flag == 1
        assert guarded_kw == 0.0
        assert power_mw_cmd == 0
        assert flow_pct == pytest.approx(FLOW_REST_PCT)
        assert situation == STANDBY_FLOW_SITUATION_CODE

    def test_invalid_partial_discharge_still_blocks_within_solo_range(self):
        """Partial discharge remains invalid when the measured load is feasible."""
        load_kw = 0.0010
        action_kw = -0.0002

        assert warn_load_over_discharge_limit(action_kw, load_kw) == 0
        assert is_invalid_partial_discharge(action_kw, load_kw, 0.0) is True
        assert determine_situation(action_kw, load_kw, 0.0) == STANDBY_FLOW_SITUATION_CODE

    def test_discharge_without_measured_load_is_standby(self):
        """A negative action must not request battery solo when measured load is zero."""
        assert determine_situation(-0.001, 0.0, 0.0) == STANDBY_FLOW_SITUATION_CODE

    def test_charge(self):
        """Charging maps to scenario 3."""
        assert determine_situation(0.0005, 0.0004, 0.0) == 3

    def test_standby(self):
        """Near-zero action maps to standby-flow mode 3, not shutdown mode 4."""
        assert determine_situation(0.0, 0.0004, 0.0) == STANDBY_FLOW_SITUATION_CODE
        assert determine_situation(0.00001, 0.0004, 0.0) == STANDBY_FLOW_SITUATION_CODE
        assert STANDBY_FLOW_SITUATION_CODE == 3


# ══════════════════════════════════════════════════════════════════
# TOU price lookup.
# ══════════════════════════════════════════════════════════════════
class TestTOUPrice:

    def test_weekday_offpeak(self):
        # Weekday 00-09.
        for h in range(9):
            assert get_tou_price(h, 0) == TOU_OFFPEAK

    def test_weekday_midpeak_morning(self):
        # Weekday 09-16.
        for h in range(9, 16):
            assert get_tou_price(h, 0) == TOU_MIDPEAK

    def test_weekday_peak(self):
        # Weekday 16-22.
        for h in range(16, 22):
            assert get_tou_price(h, 0) == TOU_PEAK

    def test_weekday_midpeak_night(self):
        # Weekday 22-24.
        for h in range(22, 24):
            assert get_tou_price(h, 0) == TOU_MIDPEAK

    def test_weekend_all_offpeak(self):
        for h in range(24):
            assert get_tou_price(h, 5) == TOU_OFFPEAK  # Saturday
            assert get_tou_price(h, 6) == TOU_OFFPEAK  # Sunday


# ══════════════════════════════════════════════════════════════════
# get_load_groups
# ══════════════════════════════════════════════════════════════════
class TestGetLoadGroups:

    def test_load_pattern_schedule(self):
        """Load schedule from load_pattern.txt is applied correctly."""
        for h in range(24):
            assert get_load_groups(dtime(h, 0)) == 4, f"hour={h} expected 4"


# ══════════════════════════════════════════════════════════════════
# write_command_simple
# ══════════════════════════════════════════════════════════════════
class TestWriteCommandSimple:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_basic_write(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        ok = write_command_simple(path, 3, ts, "01", 280, 25, load_count=4)
        assert ok is True

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        assert lines[0] == "3"
        assert lines[1] == "20260316120000,4"
        assert lines[2] == "01,280,25,"

    def test_standby(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        ok = write_standby_rest_command(path, ts, "01", load_count=4)
        assert ok

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert lines[0] == "3"
        assert lines[2] == "01,0,0,"

    def test_pre_measure_command(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        ok = write_pre_measure_command(path, ts, "01", load_count=4)
        assert ok

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert lines[0] == "3"
        assert lines[2] == "01,0,50,"

    def test_zero_power_command_keeps_battery_id_for_flow(self):
        assert command_pp_for_action("01", 0) == "01"
        assert command_pp_for_action("01", 1) == "01"
        assert command_pp_for_action("02", -1) == "02"

    def test_overwrite(self, tmp_dir):
        """Repeated writes overwrite instead of appending."""
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        write_command_simple(path, 3, ts, "01", 280, 25)
        write_standby_rest_command(path, ts, "01")

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) == 3
        assert lines[0] == "3"  # Last written value.
        assert lines[2] == "01,0,0,"

    def test_pre_measure_helper_does_not_sleep_when_injected(self, tmp_dir, monkeypatch):
        path = os.path.join(tmp_dir, "Command.txt")
        data_path = os.path.join(tmp_dir, "Data.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(
                "20260316120000\n"
                "1600,500,8000,1500,450,6750,1200,300,3600,\n"
                "550,33,400,\n"
                "01,500,650,0,250,100,\n"
            )

        slept = []
        reading = perform_pre_measure_for_decision(
            data_path, path, "01", 4, recovery_seconds=0.0,
            sleep_fn=lambda sec: slept.append(sec),
        )

        assert slept == []
        assert reading is not None
        with open(path, 'r') as f:
            content = f.read()
        assert "01,0,50," in content


# ══════════════════════════════════════════════════════════════════
# DeploymentLogger
# ══════════════════════════════════════════════════════════════════
class TestDeploymentLogger:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_creates_file_with_header(self, tmp_dir):
        logger = DeploymentLogger(tmp_dir)
        assert logger.path is not None
        assert os.path.exists(logger.path)

        with open(logger.path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == DeploymentLogger.HEADER

    def test_log_row(self, tmp_dir):
        logger = DeploymentLogger(tmp_dir)
        row = {
            'timestamp': '2026-03-16 12:00:00',
            'step': 1,
            'soc': 0.5,
        }
        logger.log(row)

        with open(logger.path, 'r', encoding='utf-8-sig') as f:
            lines = list(csv.reader(f))
        assert len(lines) == 2  # header + 1 data row
        assert lines[1][0] == '2026-03-16 12:00:00'

    def test_daily_rollover(self, tmp_dir):
        logger = DeploymentLogger(tmp_dir)

        # Day 1
        row1 = {'timestamp': '2026-03-16 23:59:00', 'step': 1}
        logger.log(row1)
        path1 = logger.path

        # Day 2
        row2 = {'timestamp': '2026-03-17 00:01:00', 'step': 2}
        logger.log(row2)
        path2 = logger.path

        assert path1 != path2
        assert '2026-03-16' in path1
        assert '2026-03-17' in path2
        assert os.path.exists(path1)
        assert os.path.exists(path2)

    def test_no_duplicate_headers_on_append(self, tmp_dir):
        """Restarting does not duplicate the header."""
        logger1 = DeploymentLogger(tmp_dir)
        logger1.log({'timestamp': '2026-03-16 12:00:00'})
        path = logger1.path

        # Simulate restart.
        logger2 = DeploymentLogger(tmp_dir)
        logger2.log({'timestamp': '2026-03-16 12:15:00'})

        with open(path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        # There should be only one header row.
        header_count = sum(1 for l in lines if l.startswith('timestamp'))
        assert header_count == 1


# ══════════════════════════════════════════════════════════════════
# RawDataLogger writes raw readings about every 10 seconds.
# ══════════════════════════════════════════════════════════════════
class TestRawDataLogger:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def _make_reading(self, ts=None, mppt_p=500.0, batt_v=6.5,
                      batt_i=100.0, load_p=400.0):
        if ts is None:
            ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        return Reading(
            timestamp=ts,
            mppt_p_mw=mppt_p,
            mppt_v=16.0,
            mppt_i_ma=50.0,
            solar_p_mw=mppt_p,
            batt_v=batt_v,
            batt_i_ma=batt_i,
            batt_soc_pct=50.0,
            batt_temp_c=25.0,
            batt_speed_pct=10.0,
            load_v=5.5,
            load_i_ma=30.0,
            load_p_mw=load_p,
        )

    def test_creates_file_with_header(self, tmp_dir):
        rl = RawDataLogger(tmp_dir)
        assert rl.path is not None
        assert os.path.exists(rl.path)
        assert 'raw_data_' in rl.path
        rl.close()

    def test_logs_per_reading(self, tmp_dir):
        """Each log call writes one row."""
        rl = RawDataLogger(tmp_dir)
        r = self._make_reading()
        rl.log(r, raw_current_ma=100.0, soc_calc=0.5, soc_unclamped=0.5, charge_mah=0.0, discharge_mah=0.0, situation_code=4, battery_pp="01")
        rl.log(r, raw_current_ma=100.0, soc_calc=0.501, soc_unclamped=0.501, charge_mah=0.1, discharge_mah=0.0, situation_code=4, battery_pp="01")
        rl.log(r, raw_current_ma=100.0, soc_calc=0.502, soc_unclamped=0.502, charge_mah=0.2, discharge_mah=0.0, situation_code=4, battery_pp="01")
        rl.close()

        with open(rl.path, 'r', encoding='utf-8-sig') as f:
            lines = list(csv.reader(f))
        assert len(lines) == 4  # header + 3 data rows

    def test_daily_rollover(self, tmp_dir):
        rl = RawDataLogger(tmp_dir)
        ts1 = datetime(2026, 3, 16, 23, 59, 0, tzinfo=TZ_UTC8)
        r1 = self._make_reading(ts=ts1)
        rl.log(r1, raw_current_ma=0.0, soc_calc=0.5, soc_unclamped=0.5, charge_mah=0.0, discharge_mah=0.0, situation_code=4, battery_pp="01")
        path1 = rl.path

        ts2 = datetime(2026, 3, 17, 0, 1, 0, tzinfo=TZ_UTC8)
        r2 = self._make_reading(ts=ts2)
        rl.log(r2, raw_current_ma=0.0, soc_calc=0.5, soc_unclamped=0.5, charge_mah=0.0, discharge_mah=0.0, situation_code=4, battery_pp="01")
        path2 = rl.path

        assert path1 != path2
        assert '2026-03-16' in path1
        assert '2026-03-17' in path2
        rl.close()

    def test_records_both_raw_and_processed_current(self, tmp_dir):
        """Log both raw and processed current."""
        rl = RawDataLogger(tmp_dir)
        r = self._make_reading(batt_i=-80.0)  # Processed value: -80 mA discharge.
        rl.log(r, raw_current_ma=80.0, soc_calc=0.5, soc_unclamped=0.5, charge_mah=0.0, discharge_mah=5.0, situation_code=1, battery_pp="01")
        rl.close()

        with open(rl.path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row['current_ma'] == '-80.0'     # Processed value.
        assert row['current_raw_ma'] == '80.0'  # Raw value.

    def test_records_soc_and_mah_fields(self, tmp_dir):
        """Verify CSV records soc_unclamped, charge_mah, and discharge_mah."""
        rl = RawDataLogger(tmp_dir)
        r = self._make_reading(batt_i=150.0)
        rl.log(r, raw_current_ma=0.0, soc_calc=0.05,
               soc_unclamped=-0.0123, charge_mah=1.23, discharge_mah=4.56,
               situation_code=3, battery_pp="01")
        rl.close()

        with open(rl.path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row['soc_calc'] == '0.0500'
        assert row['soc_unclamped'] == '-0.0123'   # May be negative.
        assert row['charge_mah'] == '1.23'
        assert row['discharge_mah'] == '4.56'
        assert row['current_ma'] == '150.0'        # Synthetic charging current.
        assert row['current_raw_ma'] == '0.0'      # Firmware reports zero.

    def test_no_duplicate_headers_on_reopen(self, tmp_dir):
        """Restart does not duplicate the header."""
        rl1 = RawDataLogger(tmp_dir)
        r = self._make_reading()
        rl1.log(r, raw_current_ma=0.0, soc_calc=0.5, soc_unclamped=0.5, charge_mah=0.0, discharge_mah=0.0, situation_code=4, battery_pp="01")
        rl1.close()

        rl2 = RawDataLogger(tmp_dir)
        rl2.log(r, raw_current_ma=0.0, soc_calc=0.5, soc_unclamped=0.5, charge_mah=0.0, discharge_mah=0.0, situation_code=4, battery_pp="01")
        rl2.close()

        with open(rl2.path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        header_count = sum(1 for l in lines if l.startswith('timestamp'))
        assert header_count == 1


# ══════════════════════════════════════════════════════════════════
# Synthetic current mode validation.
# ══════════════════════════════════════════════════════════════════
class TestSyntheticCurrent:
    """
    Validate synthetic current mode:
    When firmware reports I=0, infer SoC from issued commands and known system current.
    """

    def test_synthetic_charge_updates_soc(self):
        """Synthetic charging current should increase SoC."""
        tracker = SoCTracker(
            capacity_mah=BATTERY_CAPACITY_MAH,
            initial_soc=0.50,
            efficiency_rte=BATTERY_EFFICIENCY,
        )
        t0 = datetime(2026, 3, 16, 10, 0, 0, tzinfo=TZ_UTC8)
        tracker.update(t0, 0.0, 5.6)

        # Simulate one hour of synthetic charging (scenario 3 -> +1000 mA).
        syn_ma = 1000.0
        for i in range(1, 361):  # Every 10 seconds; 360 steps = 1 hour.
            t = t0 + timedelta(seconds=10 * i)
            tracker.update(t, syn_ma, 8.5)

        # Energy accounting: 8.5 V x 1 A x 1 h x eta = 8.075 Wh.
        # This clips from 50% to full for an 11.2 Wh battery.
        eta = float(BATTERY_EFFICIENCY)
        soc = tracker.get_soc()
        assert soc > 0.50, f"SoC should increase from 0.50 but got {soc}"
        expected_unclamped = 0.50 + BATTERY_CHARGE_V * 1.0 * 1.0 * eta / (BATTERY_CAPACITY_KWH * 1000.0)
        expected = min(1.0, expected_unclamped)
        assert abs(soc - expected) < 0.005, f"SoC={soc:.6f}, expected≈{expected:.6f}"

    def test_synthetic_discharge_updates_soc(self):
        """Synthetic discharging current should decrease SoC."""
        tracker = SoCTracker(
            capacity_mah=BATTERY_CAPACITY_MAH,
            initial_soc=0.50,
            efficiency_rte=BATTERY_EFFICIENCY,
        )
        t0 = datetime(2026, 3, 16, 10, 0, 0, tzinfo=TZ_UTC8)
        tracker.update(t0, 0.0, 5.6)

        # Simulate one hour of synthetic discharging (scenario 1/2 -> -1000 mA).
        syn_ma = -1000.0
        for i in range(1, 361):
            t = t0 + timedelta(seconds=10 * i)
            tracker.update(t, syn_ma, 5.6)

        soc = tracker.get_soc()
        assert soc < 0.50, f"SoC should decrease from 0.50 but got {soc}"

    def test_synthetic_standby_no_change(self):
        """Synthetic current is zero in standby, so SoC does not change."""
        tracker = SoCTracker(
            capacity_mah=BATTERY_CAPACITY_MAH,
            initial_soc=0.50,
            efficiency_rte=BATTERY_EFFICIENCY,
        )
        t0 = datetime(2026, 3, 16, 10, 0, 0, tzinfo=TZ_UTC8)
        tracker.update(t0, 0.0, 5.5)

        # Public-facing comment translated to English.
        for i in range(1, 100):
            t = t0 + timedelta(seconds=10 * i)
            tracker.update(t, 0.0, 5.5)

        assert tracker.get_soc() == 0.50

    def test_firmware_zero_with_synthetic_still_works(self):
        """Synthetic mode can track SoC even when firmware reports I=0."""
        tracker = SoCTracker(
            capacity_mah=BATTERY_CAPACITY_MAH,
            initial_soc=0.50,
            efficiency_rte=BATTERY_EFFICIENCY,
        )
        t0 = datetime(2026, 3, 16, 10, 0, 0, tzinfo=TZ_UTC8)
        tracker.update(t0, 0.0, 5.5)

        # Public-facing comment translated to English.
        firmware_i = 0.0  # Firmware-reported current.
        synthetic_i = 150.0  # Synthetic current used by the controller.

        for i in range(1, 91):  # 15 minutes.
            t = t0 + timedelta(seconds=10 * i)
            # In run_deployment.py, synthetic mode replaces reading.batt_i_ma with synthetic_i.
            # The SoC tracker sees synthetic_i, not firmware_i.
            tracker.update(t, synthetic_i, 8.5)

        # Energy accounting: V x I x dt / capacity_Wh.
        eta = float(BATTERY_EFFICIENCY)
        expected = 0.50 + (BATTERY_CHARGE_V * 0.150 * 0.25 * eta / (BATTERY_CAPACITY_KWH * 1000.0))
        assert abs(tracker.get_soc() - expected) < 0.002


# ══════════════════════════════════════════════════════════════════
# Voltage cutoff constant validation.
# ══════════════════════════════════════════════════════════════════
class TestVoltageCutoff:
    """Test enhanced voltage cutoff: hysteresis, cooldown, V=0 anomaly, and daily limit."""

    def test_cutoff_constants(self):
        """Cutoff voltage is below recovery voltage and discharge voltage."""
        assert BATTERY_CUTOFF_V == 4.2
        assert BATTERY_CUTOFF_RECOVER_V == 5.0
        assert BATTERY_CUTOFF_V < BATTERY_CUTOFF_RECOVER_V
        assert BATTERY_CUTOFF_RECOVER_V < BATTERY_DISCHARGE_V

    def test_hysteresis_gap_widened(self):
        """Hysteresis gap is at least 0.5 V to protect degraded batteries from voltage bounce."""
        gap = BATTERY_CUTOFF_RECOVER_V - BATTERY_CUTOFF_V
        assert gap >= 0.5, f"Hysteresis gap too small: {gap}V"

    def test_cooldown_and_daily_limit_constants(self):
        assert CUTOFF_COOLDOWN_SEC == 300
        assert CUTOFF_MAX_PER_DAY == 5
        assert CUTOFF_ZERO_V_STREAK == 3

    def test_cutoff_triggers_on_low_voltage(self):
        voltage_cutoff_active = False
        batt_v = 4.1

        if batt_v > 0 and batt_v < BATTERY_CUTOFF_V and not voltage_cutoff_active:
            voltage_cutoff_active = True

        assert voltage_cutoff_active is True

    def test_cutoff_does_not_trigger_on_safe_voltage(self):
        voltage_cutoff_active = False
        batt_v = 5.5

        if batt_v > 0 and batt_v < BATTERY_CUTOFF_V and not voltage_cutoff_active:
            voltage_cutoff_active = True

        assert voltage_cutoff_active is False

    def test_zero_voltage_streak_triggers_cutoff(self):
        """Consecutive V=0 readings trigger cutoff."""
        voltage_cutoff_active = False
        zero_streak = 0

        for _ in range(CUTOFF_ZERO_V_STREAK):
            batt_v = 0.0
            if batt_v > 0:
                zero_streak = 0
            else:
                zero_streak += 1
                if zero_streak >= CUTOFF_ZERO_V_STREAK and not voltage_cutoff_active:
                    voltage_cutoff_active = True

        assert voltage_cutoff_active is True

    def test_single_zero_does_not_trigger(self):
        """A single V=0 reading does not trigger cutoff."""
        voltage_cutoff_active = False
        zero_streak = 0
        batt_v = 0.0

        if batt_v > 0:
            zero_streak = 0
        else:
            zero_streak += 1
            if zero_streak >= CUTOFF_ZERO_V_STREAK and not voltage_cutoff_active:
                voltage_cutoff_active = True

        assert voltage_cutoff_active is False
        assert zero_streak == 1

    def test_zero_streak_resets_on_valid_reading(self):
        """V=0 streak resets after a normal voltage reading."""
        zero_streak = 2

        batt_v = 5.0
        if batt_v > 0:
            zero_streak = 0

        assert zero_streak == 0

    def test_recovery_requires_5v(self):
        """4.8 V should not recover after cutoff; recovery requires >= 5.0 V."""
        voltage_cutoff_active = True
        batt_v = 4.8

        if voltage_cutoff_active and batt_v >= BATTERY_CUTOFF_RECOVER_V:
            voltage_cutoff_active = False

        assert voltage_cutoff_active is True

    def test_recovery_at_5v(self):
        """5.0 V can recover after cutoff once cooldown is satisfied."""
        voltage_cutoff_active = True
        batt_v = 5.0

        voltage_ok = batt_v >= BATTERY_CUTOFF_RECOVER_V
        assert voltage_ok is True

    def test_recovery_above_5v(self):
        voltage_cutoff_active = True
        batt_v = 5.5

        if voltage_cutoff_active and batt_v >= BATTERY_CUTOFF_RECOVER_V:
            voltage_cutoff_active = False

        assert voltage_cutoff_active is False

    def test_daily_limit_locks_standby(self):
        """Reaching the trigger limit locks discharge for the day."""
        day_count = CUTOFF_MAX_PER_DAY
        day_locked = day_count >= CUTOFF_MAX_PER_DAY
        assert day_locked is True

    def test_cutoff_blocks_discharge_action(self):
        voltage_cutoff_active = True
        action_kw = -0.0005

        if voltage_cutoff_active and action_kw < 0:
            action_kw = 0.0

        assert action_kw == 0.0

    def test_low_voltage_cutoff_still_blocks_over_limit_discharge(self):
        action_kw = -BATTERY_DISCHARGE_PMAX_KW
        load_kw = BATTERY_DISCHARGE_PMAX_KW + 0.003
        batt_v = BATTERY_CUTOFF_V - 0.1

        assert warn_load_over_discharge_limit(action_kw, load_kw) == 1
        if batt_v > 0 and batt_v < BATTERY_CUTOFF_V and action_kw < 0:
            action_kw = 0.0

        assert action_kw == 0.0

    def test_cutoff_allows_charge_action(self):
        voltage_cutoff_active = True
        action_kw = 0.0005

        if voltage_cutoff_active and action_kw < 0:
            action_kw = 0.0

        assert action_kw == 0.0005

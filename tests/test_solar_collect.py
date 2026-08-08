"""
Unit tests for control/solar_test_collect.py
=============================================
Covers:
  - Data.txt parsing for new/old formats and negative current
  - direct Command.txt writes without temp files
  - load power calculation using 0.1 W/group, not 8 W/group
"""
import os
import sys
import pytest
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'control'))

from control.solar_test_collect import (
    LOAD_PER_GROUP_W,
    FLOW_REST_PCT,
    FLOW_PRE_MEASURE_PCT,
    FLOW_IDLE_PCT,
    DEFAULT_STANDBY_SCENARIO,
    command_pp_for_action,
    parse_data_txt,
    write_command_txt,
    write_rest_command,
    write_pre_measure_command,
)

TZ_UTC8 = timezone(timedelta(hours=8))


def test_load_power_constant_matches_current_hardware():
    assert LOAD_PER_GROUP_W == pytest.approx(0.1)


def test_rest_and_pre_measure_flow_constants():
    assert FLOW_REST_PCT == 0
    assert FLOW_PRE_MEASURE_PCT == 50
    assert FLOW_IDLE_PCT == FLOW_REST_PCT


# ══════════════════════════════════════════════════════════════════
# parse_data_txt
# ══════════════════════════════════════════════════════════════════
class TestParseDataTxt:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def _write(self, tmp_dir, content):
        path = os.path.join(tmp_dir, "Data.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_newest_format_with_grid(self, tmp_dir):
        """2026/03/20+ format: 9 MPPT + 6 Load/Grid + 7 battery fields."""
        content = (
            "20260320120000\n"
            "1600,500,8000,1500,450,6750,1200,300,3600,\n"
            "550,33,400,2200,100,220000,\n"
            "1,101,720,500,1200,332,1000,\n"
        )
        path = self._write(tmp_dir, content)
        ts, mppt, bus, load, grid, batts = parse_data_txt(path)

        assert ts == "20260320120000"
        assert mppt is not None
        assert mppt['mppt_p_mw'] == 6750.0
        assert bus is not None
        assert bus['bus_p_mw'] == 3600.0
        assert load is not None
        assert load['load_p_mw'] == 400.0
        assert grid is not None
        assert grid['grid_v'] == pytest.approx(22.0)
        assert grid['grid_p_mw'] == 220000.0
        assert len(batts) == 1
        assert batts[0]['pp'] == '01'
        assert batts[0]['soc_pct'] == pytest.approx(10.1)
        assert batts[0]['volt_v'] == pytest.approx(7.20)
        assert batts[0]['charge_v'] == pytest.approx(5.00)
        assert batts[0]['curr_ma'] == 1200.0

    def test_previous_format_3field_load(self, tmp_dir):
        """Early 2026/03 format: 9 MPPT + 3 Load + 6 battery fields."""
        content = (
            "20260316120000\n"
            "1600,500,8000,1500,450,6750,1200,300,3600,\n"
            "550,33,400,\n"
            "01,500,650,120,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        ts, mppt, bus, load, grid, batts = parse_data_txt(path)

        assert ts == "20260316120000"
        assert load is not None
        assert grid is None
        assert len(batts) == 1
        assert batts[0]['charge_v'] == 0.0
        assert batts[0]['curr_ma'] == 120.0

    def test_6field_mppt_with_3field_load(self, tmp_dir):
        """6-field MPPT can still be followed by a measured load line."""
        content = (
            "20260316120000\n"
            "1600,500,8000,1500,450,6750,\n"
            "550,33,400,\n"
            "01,500,650,-80,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        ts, mppt, bus, load, grid, batts = parse_data_txt(path)

        assert mppt is not None
        assert bus is None
        assert load is not None
        assert load["load_p_mw"] == pytest.approx(400.0)
        assert grid is None
        assert batts[0]["curr_ma"] == -80.0

    def test_old_format(self, tmp_dir):
        content = (
            "20260316120000\n"
            "1600,500,8000,1500,450,6750,\n"
            "01,500,650,120,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        ts, mppt, bus, load, grid, batts = parse_data_txt(path)

        assert mppt is not None
        assert bus is None
        assert load is None
        assert grid is None
        assert len(batts) == 1

    def test_negative_battery_current(self, tmp_dir):
        """Battery current is negative during discharge."""
        content = (
            "20260316120000\n"
            "0,0,0,0,0,0,\n"
            "01,500,550,-80,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        ts, mppt, bus, load, grid, batts = parse_data_txt(path)
        assert batts[0]['curr_ma'] == -80.0

    def test_empty_file(self, tmp_dir):
        path = self._write(tmp_dir, "")
        ts, mppt, bus, load, grid, batts = parse_data_txt(path)
        assert ts is None
        assert mppt is None
        assert batts == []

    def test_nonexistent(self):
        ts, mppt, bus, load, grid, batts = parse_data_txt("/nonexistent/Data.txt")
        assert ts is None


# ══════════════════════════════════════════════════════════════════
# write_command_txt (direct write, no temp file)
# ══════════════════════════════════════════════════════════════════
class TestWriteCommandTxt:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_basic_standby(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        ok = write_rest_command(path, DEFAULT_STANDBY_SCENARIO, ts, 4, "01")
        assert ok is True

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        assert lines[0] == "3"
        assert lines[1] == "20260316120000,4"
        assert lines[2] == "01,0,0,"
        assert DEFAULT_STANDBY_SCENARIO == 3

    def test_pre_measure_command(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        ok = write_pre_measure_command(path, DEFAULT_STANDBY_SCENARIO, ts, 4, "01")
        assert ok is True

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        assert lines[0] == "3"
        assert lines[1] == "20260316120000,4"
        assert lines[2] == "01,0,50,"

    def test_zero_power_command_keeps_battery_id_for_flow(self):
        assert command_pp_for_action("01", 0) == "01"
        assert command_pp_for_action("01", 280) == "01"

    def test_no_temp_file_created(self, tmp_dir):
        """Ensure no .tmp files are left behind."""
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        write_rest_command(path, DEFAULT_STANDBY_SCENARIO, ts, 4, "01")

        files = os.listdir(tmp_dir)
        tmp_files = [f for f in files if f.endswith('.tmp')]
        assert len(tmp_files) == 0, f"Temp files found: {tmp_files}"

    def test_overwrite(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        write_command_txt(path, 3, ts, 4, "01", 280, 25)
        write_rest_command(path, DEFAULT_STANDBY_SCENARIO, ts, 4, "01")

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) == 3  # Only the final write remains.
        assert lines[0] == "3"
        assert lines[2] == "01,0,0,"

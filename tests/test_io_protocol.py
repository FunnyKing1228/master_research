"""
Unit tests for control/io_protocol.py
======================================
Covers:
  - Data.txt parsing for new/old formats, negative current, and edge values
  - Command.txt write/read behavior
  - _VendorDataResult backward compatibility
  - MPPT, Load, and Battery field parsing
"""
import os
import sys
import io
import pytest
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

# ── Ensure import path is available. ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'control'))

from control.io_protocol import (
    parse_ts, format_ts,
    parse_signed_field, format_signed_field,
    parse_unsigned_field, format_unsigned_field,
    parse_vendor_data_line,
    parse_mppt_line, parse_mppt_line_v2,
    parse_load_line,
    read_vendor_data_file,
    _VendorDataResult,
    write_control_file_vendor,
    TZ_UTC8,
)


# ══════════════════════════════════════════════════════════════════
# Timestamp parsing
# ══════════════════════════════════════════════════════════════════
class TestTimestamp:
    def test_parse_ts_basic(self):
        dt = parse_ts("20260316143022")
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 16
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 22
        assert dt.tzinfo == TZ_UTC8

    def test_format_ts_roundtrip(self):
        dt = datetime(2026, 3, 16, 8, 5, 0, tzinfo=TZ_UTC8)
        s = format_ts(dt)
        assert s == "20260316080500"
        dt2 = parse_ts(s)
        assert dt2 == dt


# ══════════════════════════════════════════════════════════════════
# Signed / unsigned fields
# ══════════════════════════════════════════════════════════════════
class TestSignedField:
    def test_positive(self):
        assert parse_signed_field("00012345") == 123.45

    def test_negative(self):
        assert parse_signed_field("10012345") == -123.45

    def test_zero(self):
        assert parse_signed_field("00000000") == 0.0

    def test_roundtrip(self):
        for val in [0.0, 1.23, -4.56, 999.99, -0.01]:
            s = format_signed_field(val)
            assert len(s) == 8
            parsed = parse_signed_field(s)
            assert abs(parsed - val) < 0.01

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_signed_field("20012345")  # See test assertion for expected behavior.
        with pytest.raises(ValueError):
            parse_signed_field("0001234")  # See test assertion for expected behavior.


class TestUnsignedField:
    def test_basic(self):
        assert parse_unsigned_field("00012345") == 123.45

    def test_zero(self):
        assert parse_unsigned_field("00000000") == 0.0

    def test_roundtrip(self):
        for val in [0.0, 1.23, 999.99]:
            s = format_unsigned_field(val)
            parsed = parse_unsigned_field(s)
            assert abs(parsed - val) < 0.01


# ══════════════════════════════════════════════════════════════════
# Battery data line (parse_vendor_data_line)
# ══════════════════════════════════════════════════════════════════
class TestParseVendorDataLine:
    """
    New format (7 fields): ID,SOC,BV,ChargeV,ChargeI,Temp,Speed
    Old format (6 fields): ID,SOC,BV,BI,Temp,Speed
    Units: SOC=0.1%, BV/ChargeV=0.01V, BI/ChargeI=1mA, Temp=0.1C, Speed=0.1%
    """

    def test_new_format_7field(self):
        """2026/03/20+ format includes charge voltage."""
        pp, soc, v, cv, i, t, spd = parse_vendor_data_line("1,101,720,500,1200,332,1000,")
        assert pp == "01"        # zfill(2)
        assert soc == pytest.approx(10.1)
        assert v == pytest.approx(7.20)
        assert cv == pytest.approx(5.00)
        assert i == 1200.0
        assert t == pytest.approx(33.2)
        assert spd == pytest.approx(100.0)

    def test_old_format_6field_positive_current(self):
        pp, soc, v, cv, i, t, spd = parse_vendor_data_line("01,500,650,120,250,100,")
        assert pp == "01"
        assert soc == 50.0       # 500 / 10
        assert v == 6.50         # 650 / 100
        assert cv == 0.0  # See test assertion for expected behavior.
        assert i == 120.0        # 120 mA (positive = charging)
        assert t == 25.0         # 250 / 10
        assert spd == 10.0       # 100 / 10

    def test_old_format_negative_current(self):
        """Regression: isdigit() ignored negative signs and parsed current as 0."""
        pp, soc, v, cv, i, t, spd = parse_vendor_data_line("01,500,550,-80,250,100,")
        assert cv == 0.0
        assert i == -80.0  # See test assertion for expected behavior.

    def test_old_format_large_negative_current(self):
        pp, soc, v, cv, i, t, spd = parse_vendor_data_line("01,500,550,-1200,250,100,")
        assert i == -1200.0

    def test_zero_everything_6field(self):
        """Disconnected battery reports all zeros in the old format."""
        pp, soc, v, cv, i, t, spd = parse_vendor_data_line("01,0,0,0,0,0,")
        assert soc == 0.0
        assert v == 0.0
        assert cv == 0.0
        assert i == 0.0

    def test_invalid_field_fallback(self):
        """Non-numeric BI defaults to 0 in the old format."""
        pp, soc, v, cv, i, t, spd = parse_vendor_data_line("01,500,550,abc,250,100,")
        assert i == 0.0

    def test_zfill_single_digit_id(self):
        """Battery ID "1" is normalized to "01"."""
        pp, soc, v, cv, i, t, spd = parse_vendor_data_line("1,0,0,0,0,0,0,")
        assert pp == "01"


# ══════════════════════════════════════════════════════════════════
# MPPT line parsing
# ══════════════════════════════════════════════════════════════════
class TestParseMPPTLine:
    """
    Old 6-field format: SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,
    Units: V=0.01V, I=mA, P=mW
    """

    def test_basic_6field(self):
        sv, si, sp, mv, mi, mp = parse_mppt_line("1600,500,8000,1500,450,6750,")
        assert sv == 16.00       # 1600 / 100
        assert si == 500.0       # 500 mA
        assert sp == 8000.0      # 8000 mW
        assert mv == 15.00       # 1500 / 100
        assert mi == 450.0
        assert mp == 6750.0

    def test_zeros(self):
        """Nighttime case with no solar power."""
        sv, si, sp, mv, mi, mp = parse_mppt_line("0,0,0,0,0,0,")
        assert sv == 0.0
        assert mp == 0.0


class TestParseMPPTLineV2:
    """New 9-field format includes BusV, BusI, and BusP."""

    def test_9field(self):
        mppt_6, bus = parse_mppt_line_v2("1600,500,8000,1500,450,6750,1200,300,3600,")
        assert mppt_6 == (16.00, 500.0, 8000.0, 15.00, 450.0, 6750.0)
        assert bus is not None
        assert bus == (12.00, 300.0, 3600.0)

    def test_6field_no_bus(self):
        mppt_6, bus = parse_mppt_line_v2("1600,500,8000,1500,450,6750,")
        assert bus is None

    def test_zeros(self):
        mppt_6, bus = parse_mppt_line_v2("0,0,0,0,0,0,0,0,0,")
        assert mppt_6 == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert bus == (0.0, 0.0, 0.0)


# ══════════════════════════════════════════════════════════════════
# Load line parsing
# ══════════════════════════════════════════════════════════════════
class TestParseLoadLine:
    """
    New format: LoadV,LoadI,LoadP,GridV,GridI,GridP (6 fields)
    Old format: LoadV,LoadI,LoadP (3 fields)
    """

    def test_new_format_6field(self):
        (lv, li, lp), grid = parse_load_line("1200,5500,6600,2200,100,220000,")
        assert lv == 12.00
        assert li == 5500.0
        assert lp == 6600.0
        assert grid is not None
        gv, gi, gp = grid
        assert gv == 22.00
        assert gi == 100.0
        assert gp == 220000.0

    def test_old_format_3field(self):
        (lv, li, lp), grid = parse_load_line("1200,5500,6600,")
        assert lv == 12.00
        assert li == 5500.0
        assert lp == 6600.0
        assert grid is None

    def test_zeros(self):
        (lv, li, lp), grid = parse_load_line("0,0,0,")
        assert lv == 0.0
        assert lp == 0.0
        assert grid is None

    def test_zeros_with_grid(self):
        (lv, li, lp), grid = parse_load_line("0,0,0,0,0,0,")
        assert lv == 0.0
        assert grid is not None
        assert grid == (0.0, 0.0, 0.0)


# ══════════════════════════════════════════════════════════════════
# read_vendor_data_file (full Data.txt parsing)
# ══════════════════════════════════════════════════════════════════
class TestReadVendorDataFile:
    """
    Test complete Data.txt reads across new and old formats.
    """

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

    def test_newest_format_with_grid_and_charge_v(self, tmp_dir):
        """2026/03/20+ format: 9 MPPT + 6 Load/Grid + 7 battery fields."""
        content = (
            "20260320120000\n"
            "1600,500,8000,1500,450,6750,1200,300,3600,\n"
            "550,33,400,2200,100,220000,\n"
            "1,101,720,500,1200,332,1000,\n"
        )
        path = self._write(tmp_dir, content)
        result = read_vendor_data_file(path, clear_after_read=False)

        # MPPT
        assert result['mppt'] is not None
        sv, si, sp, mv, mi, mp = result['mppt']
        assert mp == 6750.0

        # MPPT-Bus
        assert result['mppt_bus'] is not None
        bv, bi, bp = result['mppt_bus']
        assert bp == 3600.0

        # Load
        assert result['load'] is not None
        lv, li, lp = result['load']
        assert lp == 400.0

        # Grid (new)
        assert result['grid'] is not None
        gv, gi, gp = result['grid']
        assert gv == pytest.approx(22.0)
        assert gp == 220000.0

        # Battery (new 7-tuple)
        assert '01' in result['batteries']
        ts, soc, v, cv, i, t, spd = result['batteries']['01']
        assert soc == pytest.approx(10.1)
        assert v == pytest.approx(7.20)
        assert cv == pytest.approx(5.00)
        assert i == 1200.0

    def test_previous_format_3field_load(self, tmp_dir):
        """Early 2026/03 format: 9 MPPT + 3 Load + 6 battery fields."""
        content = (
            "20260316120000\n"
            "1600,500,8000,1500,450,6750,1200,300,3600,\n"
            "550,33,400,\n"
            "01,500,650,120,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        result = read_vendor_data_file(path, clear_after_read=False)

        assert result['load'] is not None
        assert result['grid'] is None  # See test assertion for expected behavior.
        assert '01' in result['batteries']
        ts, soc, v, cv, i, t, spd = result['batteries']['01']
        assert cv == 0.0  # See test assertion for expected behavior.
        assert i == 120.0

    def test_6field_mppt_with_3field_load(self, tmp_dir):
        """6-field MPPT can still be followed by a measured load line."""
        content = (
            "20260316120000\n"
            "1600,500,8000,1500,450,6750,\n"
            "550,33,400,\n"
            "01,500,650,-80,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        result = read_vendor_data_file(path, clear_after_read=False)

        assert result['mppt'] is not None
        assert result['mppt_bus'] is None
        assert result['load'] is not None
        assert result['load'][2] == pytest.approx(400.0)
        assert result['grid'] is None
        assert result['batteries']['01'][4] == -80.0

    def test_old_format_6field(self, tmp_dir):
        """2025/12 old format: 6 MPPT fields + battery, without Load."""
        content = (
            "20260316120000\n"
            "1600,500,8000,1500,450,6750,\n"
            "01,500,650,120,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        result = read_vendor_data_file(path, clear_after_read=False)

        assert result['mppt'] is not None
        assert result['mppt_bus'] is None
        assert result['load'] is None
        assert result['grid'] is None
        assert '01' in result['batteries']

    def test_negative_battery_current(self, tmp_dir):
        """Battery current is negative during discharge in the old format."""
        content = (
            "20260316120000\n"
            "0,0,0,0,0,0,\n"
            "01,500,550,-80,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        result = read_vendor_data_file(path, clear_after_read=False)
        ts, soc, v, cv, i, t, spd = result['batteries']['01']
        assert i == -80.0, "Negative current must be parsed correctly!"

    def test_empty_file(self, tmp_dir):
        path = self._write(tmp_dir, "")
        result = read_vendor_data_file(path, clear_after_read=False)
        assert result['mppt'] is None
        assert result['batteries'] == {}

    def test_nonexistent_file(self):
        result = read_vendor_data_file("/nonexistent/path/Data.txt")
        assert result['mppt'] is None
        assert result['batteries'] == {}

    def test_backward_compat_tuple_unpack(self, tmp_dir):
        """Legacy tuple unpacking: mppt_data, batt_data = result."""
        content = (
            "20260316120000\n"
            "1600,500,8000,1500,450,6750,\n"
            "01,500,650,120,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        result = read_vendor_data_file(path, clear_after_read=False)

        # Tuple unpacking.
        mppt_data, batt_data = result
        assert mppt_data is not None
        assert isinstance(batt_data, dict)
        assert len(result) == 2  # See test assertion for expected behavior.

    def test_clear_after_read(self, tmp_dir):
        content = (
            "20260316120000\n"
            "0,0,0,0,0,0,\n"
            "01,500,650,120,250,100,\n"
        )
        path = self._write(tmp_dir, content)
        result = read_vendor_data_file(path, clear_after_read=True)
        assert '01' in result['batteries']

        # The file should be cleared.
        with open(path, 'r') as f:
            assert f.read().strip() == ""

    def test_multiple_batteries(self, tmp_dir):
        content = (
            "20260316120000\n"
            "0,0,0,0,0,0,\n"
            "01,500,650,120,250,100,\n"
            "02,300,420,-50,260,80,\n"
        )
        path = self._write(tmp_dir, content)
        result = read_vendor_data_file(path, clear_after_read=False)
        assert '01' in result['batteries']
        assert '02' in result['batteries']
        _, _, _, _, i2, _, _ = result['batteries']['02']
        assert i2 == -50.0


# ══════════════════════════════════════════════════════════════════
# write_control_file_vendor
# ══════════════════════════════════════════════════════════════════
class TestWriteControlFileVendor:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_basic_write(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        commands = {'01': (ts, 0.28, 25.0)}  # 0.28 W, 25%

        ok = write_control_file_vendor(
            path, commands,
            global_ts=ts,
            situation_code=3,
            load_count=4,
            require_empty=False,
        )
        assert ok is True

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        assert lines[0] == "3"          # situation code
        assert "20260316120000" in lines[1]
        assert ",4" in lines[1]         # load count
        assert lines[2].startswith("01,")
        assert "280," in lines[2]       # 0.28 W = 280 mW
        assert ",25," in lines[2]       # 25%

    def test_standby_rest_uses_mode3_zero_flow(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        commands = {'01': (ts, 0.0, 0.0)}

        ok = write_control_file_vendor(
            path, commands,
            global_ts=ts,
            situation_code=3,
            load_count=4,
            require_empty=False,
        )
        assert ok is True

        with open(path, 'r') as f:
            content = f.read()
        assert content.splitlines()[0] == "3"
        assert "01,0,0," in content

    def test_pre_measure_uses_mode3_physical_pp_and_50_flow(self, tmp_dir):
        path = os.path.join(tmp_dir, "Command.txt")
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=TZ_UTC8)
        commands = {'01': (ts, 0.0, 50.0)}

        ok = write_control_file_vendor(
            path, commands,
            global_ts=ts,
            situation_code=3,
            load_count=4,
            require_empty=False,
        )
        assert ok is True

        with open(path, 'r') as f:
            content = f.read()
        assert content.splitlines()[0] == "3"
        assert "01,0,50," in content

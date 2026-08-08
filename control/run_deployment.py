#!/usr/bin/env python3
"""
====================

Data Flow:
  ┌──────────┐    Data.txt    ┌────────────────┐
  │ Hardware  │ ───(10-12s)──→│  DataBuffer     │
  └──────────┘                │  (15min window) │
                              └───────┬────────┘
                                      │ every 15min
                              ┌───────▼────────┐
                              │  Aggregation    │
                              │  mean/std/max   │
                              └───────┬────────┘
                                      │
                              ┌───────▼────────┐
                              │  State Builder  │
                              │  [SoC, load,    │
                              │   pv, price,    │
                              │   hour, dow]    │
                              └───────┬────────┘
                                      │
                              ┌───────▼────────┐
                              │  SAC Agent      │
                              │  → raw action   │
                              └───────┬────────┘
                                      │
                              ┌───────▼────────┐
                              │  CORAL Safety   │
                              │  CRTSN → OCC    │
                              │  → safe action  │
                              └───────┬────────┘
                                      │
  ┌──────────┐  Command.txt   ┌───────▼────────┐
  │ Hardware  │ ←──(1s)──────│  Command Writer │
  └──────────┘                └────────────────┘

  - MPPT_P in Data.txt : mW（800 = 800 mW = 0.8 W）
  - power in Command.txt : mW（8500 = 8500 mW = 8.5 W max charge）
  - flow in Command.txt  : %（25 = 25%）

  python control/run_deployment.py ^
      --data-file ./Data.txt ^
      --command-file ./Command.txt ^
      --model-path ./models/best_sac_model.pth ^
      --battery-id 01
"""

import os
import sys
import time
import math
import argparse
import json
import csv
import inspect
import builtins
import hashlib
from datetime import datetime, timezone, timedelta, time as dtime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import yaml


def _make_console_streams_safe() -> None:
    """Avoid crashing the control loop on console encoding errors."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="backslashreplace")
            except Exception:
                pass


def safe_print(*args, **kwargs) -> None:
    """Best-effort print that never raises UnicodeEncodeError."""
    try:
        builtins.print(*args, **kwargs)
        return
    except UnicodeEncodeError:
        pass

    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    flush = kwargs.get("flush", False)
    file = kwargs.get("file", sys.stdout)
    if file is None:
        file = sys.stdout

    text = sep.join(str(arg) for arg in args) + end
    encoding = getattr(file, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="backslashreplace").decode(
        encoding,
        errors="replace",
    )
    try:
        file.write(safe_text)
        if flush and hasattr(file, "flush"):
            file.flush()
    except Exception:
        fallback = getattr(sys, "__stdout__", None) or sys.stdout
        fallback.write(text.encode("ascii", errors="backslashreplace").decode("ascii"))
        if flush and hasattr(fallback, "flush"):
            fallback.flush()


_make_console_streams_safe()
print = safe_print


def _patch_inspect_for_frozen_torch() -> None:
    """
    """
    if not getattr(sys, "frozen", False):
        return

    original_getsource = inspect.getsource
    original_getsourcelines = inspect.getsourcelines

    def _read_module_source(obj: Any) -> Optional[str]:
        module = obj if inspect.ismodule(obj) else inspect.getmodule(obj)
        if module is None:
            return None

        module_name = getattr(module, "__name__", None)
        loader = getattr(module, "__loader__", None)
        if loader is not None and hasattr(loader, "get_source") and module_name:
            try:
                source = loader.get_source(module_name)
                if source:
                    return source
            except Exception:
                pass

        source_path = getattr(module, "__file__", None)
        if source_path and os.path.isfile(source_path):
            for encoding in ("utf-8", "utf-8-sig", "cp950"):
                try:
                    with open(source_path, "r", encoding=encoding) as f:
                        return f.read()
                except Exception:
                    continue
        return None

    def _safe_getsource(obj: Any) -> str:
        try:
            return original_getsource(obj)
        except OSError:
            source = _read_module_source(obj)
            if source is not None:
                return source
            module = obj if inspect.ismodule(obj) else inspect.getmodule(obj)
            module_name = getattr(module, "__name__", "")
            if module_name.startswith("torch."):
                return "\n"
            raise

    def _safe_getsourcelines(obj: Any):
        try:
            return original_getsourcelines(obj)
        except OSError:
            source = _read_module_source(obj)
            if source is not None:
                return source.splitlines(True), 1
            module = obj if inspect.ismodule(obj) else inspect.getmodule(obj)
            module_name = getattr(module, "__name__", "")
            if module_name.startswith("torch."):
                return ["\n"], 1
            raise

    inspect.getsource = _safe_getsource
    inspect.getsourcelines = _safe_getsourcelines


_patch_inspect_for_frozen_torch()
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core'))
sys.path.insert(0, SCRIPT_DIR)

from sac_agent import SACAgent
from io_protocol import (
    read_vendor_data_file,
    format_ts,
    parse_ts,
    TZ_UTC8,
)


def norm_to_power_kw(power_norm: float) -> float:
    power_norm = float(np.clip(power_norm, -1.0, 1.0))
    if power_norm >= 0.0:
        return power_norm * BATTERY_CHARGE_PMAX_KW
    return power_norm * BATTERY_DISCHARGE_PMAX_KW


def apply_flow_operating_rule(flow_pct: float, active: bool) -> float:
    """Use zero-flow rest unless an active battery action is being commanded."""
    if not active:
        return FLOW_REST_PCT
    return float(np.clip(max(float(flow_pct), FLOW_MIN_ACTIVE_PCT), 0.0, 100.0))


STANDBY_SITUATION_CODE = 3
PRE_MEASURE_SITUATION_CODE = 3
STANDBY_FLOW_SITUATION_CODE = STANDBY_SITUATION_CODE  # Backward-compatible public alias.


def command_pp_for_action(pp: str, power_mw: int) -> str:
    """Keep the battery PP even at zero power so flow commands still target it."""
    return f"{int(pp):02d}" if str(pp).isdigit() else str(pp)


def write_command_simple(path: str, situation_code: int, ts: 'datetime',
                         pp: str, power_mw: int, flow_pct: int,
                         load_count: int = 4) -> bool:
    """
      {situation_code}
      YYYYMMDDhhmmss,{load_count}
      PP,power_mW,flow_pct,
    """
    command_pp = command_pp_for_action(pp, power_mw)
    content = (
        f"{situation_code}\n"
        f"{ts.strftime('%Y%m%d%H%M%S')},{load_count}\n"
        f"{command_pp},{power_mw},{flow_pct},\n"
    )
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except (IOError, OSError, PermissionError):
        return False


def write_standby_rest_command(path: str, ts: 'datetime', pp: str,
                               load_count: int = 4) -> bool:
    """Write normal standby/rest: mode 3, physical PP, zero power, zero flow."""
    return write_command_simple(
        path, STANDBY_SITUATION_CODE, ts,
        pp, 0, int(round(FLOW_REST_PCT)),
        load_count=load_count,
    )


def write_pre_measure_command(path: str, ts: 'datetime', pp: str,
                              load_count: int = 4) -> bool:
    """Write voltage recovery / pre-measure command before model decisions."""
    return write_command_simple(
        path, PRE_MEASURE_SITUATION_CODE, ts,
        pp, 0, int(round(FLOW_PRE_MEASURE_PCT)),
        load_count=load_count,
    )
from safety_net import (
    SafetyNet,
    update_conformal_residual,
    set_conformal_params,
    get_residual_count,
    clear_residual_buffer,
)
from soh_predictor import OnlineSoHPredictor

# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
BATTERY_CHARGE_V      = 8.5
BATTERY_DISCHARGE_V   = 5.6
BATTERY_CUTOFF_V      = 4.2
BATTERY_CHARGE_CUTOFF_V = 8.8
BATTERY_CUTOFF_RECOVER_V = 5.0
BATTERY_CUTOFF_SOC    = 0.20        # default control SoC fallback when voltage cutoff trips
CUTOFF_COOLDOWN_SEC   = 300
CUTOFF_MAX_PER_DAY    = 5
CUTOFF_ZERO_V_STREAK  = 3          # consecutive V=0 readings are treated as noisy/missing, not cutoff

# SoH-aware passive monitoring. This is a deployable health monitor/proxy layer,
# not a calibrated laboratory SoH estimator. It records usable segments when the
# deployment naturally satisfies the condition; it does not shape actions.
SOH_HEALTH_LOW_VOLTAGE_V = BATTERY_CUTOFF_V
SOH_HEALTH_RECOVER_V = BATTERY_CUTOFF_RECOVER_V
SOH_HEALTH_LOW_VOLTAGE_SAMPLES = 3
SOH_HEALTH_RECOVERY_SAMPLES = 12
SOH_HEALTH_SAG_WARNING_V = 0.8
SOH_RECORD_SOC_MIN = 0.25
SOH_RECORD_SOC_MAX = 0.75
SOH_RECORD_CURRENT_THRESHOLD_MA = 50.0
SOH_RECORD_MIN_SAMPLES = 30

_system_current_a     = 1.0
DISCHARGE_HOURS       = 2.0         # 2Ah / 1A = 2hr

BATTERY_CHARGE_PMAX_KW    = (_system_current_a * BATTERY_CHARGE_V) / 1000
BATTERY_DISCHARGE_PMAX_KW = (_system_current_a * BATTERY_DISCHARGE_V) / 1000
BATTERY_PMAX_KW           = max(BATTERY_CHARGE_PMAX_KW, BATTERY_DISCHARGE_PMAX_KW)  # backward compatibility
DISCHARGE_INTENT_THRESHOLD_KW = 0.00005  # 50 mW, below one 100 mW load group.
FLOW_MIN_ACTIVE_PCT = 60.0
FLOW_REST_PCT = 0.0
FLOW_PRE_MEASURE_PCT = 50.0
VOLTAGE_RECOVERY_SECONDS = float(os.environ.get("P302_VOLTAGE_RECOVERY_SECONDS", "25.0"))
FLOW_IDLE_PCT = FLOW_REST_PCT  # Backward-compatible public alias for rest flow.

BATTERY_CAPACITY_MAH  = _system_current_a * 1000 * DISCHARGE_HOURS        # 2000.0 mAh
BATTERY_CAPACITY_WH   = (_system_current_a * BATTERY_DISCHARGE_V) * DISCHARGE_HOURS  # 11.20 Wh
BATTERY_CAPACITY_KWH  = BATTERY_CAPACITY_WH / 1000                        # 0.01120 kWh

BATTERY_EFFICIENCY    = 0.95
FIRMWARE_OVERRIDE_DISCHARGE_CURRENT_MA = 20.0
FIRMWARE_OVERRIDE_DISCHARGE_VOLTAGE_V = 6.0
ISOLATED_LOAD_DROP_LOAD_MAX_W = 0.05
ISOLATED_LOAD_DROP_VOLTAGE_MAX_V = 5.0
ISOLATED_LOAD_DROP_CURRENT_MAX_MA = 200.0
ISOLATED_LOAD_DROP_GRID_MAX_W = 1.0
HYBRID_MEASURED_LOAD_MIN_W = 0.05

# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
LOAD_PER_GROUP_W      = 0.1
MAX_LOAD_GROUPS       = 4
PV_PRESENT_THRESHOLD_KW = 0.00005
PV_SURPLUS_CHARGE_THRESHOLD_KW = 0.0002
PV_SUFFICIENT_RATIO_THRESHOLD = 0.8
PV_SUPPORT_RATIO_MAX = 1.5

_LOAD_PATTERN_FALLBACK = [(dtime(0, 0), 4)]


def _load_schedule_from_file() -> List[Tuple[dtime, int]]:
    """
    """
    candidates = []
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(exe_dir, 'load_pattern.txt'))
        candidates.append(os.path.join(exe_dir, '_internal', 'load_pattern.txt'))
    candidates.append(os.path.join(PROJECT_ROOT, 'load_pattern.txt'))
    candidates.append(os.path.join(SCRIPT_DIR, '..', 'load_pattern.txt'))

    for path in candidates:
        if os.path.isfile(path):
            try:
                schedule = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split(',')
                        if len(parts) < 2:
                            continue
                        hm = parts[0].strip().split(':')
                        h, m = int(hm[0]), int(hm[1]) if len(hm) > 1 else 0
                        groups = int(parts[1].strip())
                        schedule.append((dtime(h, m), groups))
                if schedule:
                    schedule.sort(key=lambda x: x[0])
                    print(f"[LOAD] Loaded schedule from {path} ({len(schedule)} entries)")
                    return schedule
            except Exception as e:
                print(f"[LOAD] Failed to parse {path}: {e}")
    print("[LOAD] No load_pattern.txt found, using fallback (4 groups all day)")
    return _LOAD_PATTERN_FALLBACK


LOAD_SCHEDULE = _load_schedule_from_file()
for _t, _n in LOAD_SCHEDULE:
    print(f"  [LOAD]   {_t.strftime('%H:%M')} -> {_n} groups")

TOU_OFFPEAK  = 2.06   # TWD/kWh
TOU_MIDPEAK  = 4.69
TOU_PEAK     = 7.13


def get_load_groups(t: dtime) -> int:
    """Documentation for this public API is provided in English."""
    result = 0
    for sched_t, n in LOAD_SCHEDULE:
        if t >= sched_t:
            result = n
    return result


def get_tou_price(hour: int, day_of_week: int = 0) -> float:
    """
    
    Args:
        hour: 0~23
        day_of_week: 0=Mon ... 6=Sun
    
    Returns:
    """
    if day_of_week >= 5:  # Sat=5, Sun=6
        return TOU_OFFPEAK
    if hour < 9:
        return TOU_OFFPEAK      # 00:00 - 09:00
    elif hour < 16:
        return TOU_MIDPEAK      # 09:00 - 16:00
    elif hour < 22:
        return TOU_PEAK         # 16:00 - 22:00
    else:
        return TOU_MIDPEAK      # 22:00 - 24:00


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
@dataclass
class Reading:
    """Documentation for this public API is provided in English."""
    timestamp: datetime
    solar_v: float = 0.0
    solar_i_ma: float = 0.0
    solar_p_mw: float = 0.0
    mppt_p_mw: float = 0.0
    mppt_v: float = 0.0
    mppt_i_ma: float = 0.0
    bus_v: float = 0.0
    bus_i_ma: float = 0.0
    bus_p_mw: float = 0.0
    load_v: float = 0.0
    load_i_ma: float = 0.0
    load_p_mw: float = 0.0
    grid_v: float = 0.0
    grid_i_ma: float = 0.0
    grid_p_mw: float = 0.0
    batt_soc_pct: float = 50.0
    batt_v: float = 0.0
    batt_charge_v: float = 0.0
    batt_i_ma: float = 0.0
    batt_temp_c: float = 25.0
    batt_speed_pct: float = 0.0


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
class DataBuffer:
    """
    """

    def __init__(self, window_sec: int = 900):
        self.window_sec = window_sec   # 15 min = 900 sec
        self.readings: List[Reading] = []
        self.window_start: Optional[datetime] = None

    def add(self, reading: Reading):
        """Documentation for this public API is provided in English."""
        now = reading.timestamp
        if self.window_start is None:
            minute = now.minute
            aligned_min = (minute // 15) * 15
            self.window_start = now.replace(minute=aligned_min, second=0, microsecond=0)
        self.readings.append(reading)

    def is_window_complete(self, now: datetime) -> bool:
        """Documentation for this public API is provided in English."""
        if self.window_start is None:
            return False
        elapsed = (now - self.window_start).total_seconds()
        return elapsed >= self.window_sec

    def aggregate(self) -> Dict[str, float]:
        """
        
        Returns:
            {
            }
        """
        if not self.readings:
            return {
                'mppt_p_mean_mW': 0.0, 'mppt_p_std_mW': 0.0, 'mppt_p_max_mW': 0.0,
                'bus_p_mean_mW': 0.0,
                'load_p_mean_mW': 0.0,
                'batt_p_mean_mW': 0.0,
                'batt_v_mean': 0.0, 'batt_v_min': 0.0, 'batt_v_max': 0.0,
                'batt_i_mean_ma': 0.0,
                'n_samples': 0, 'completeness': 0.0,
            }

        mppt_vals = [r.mppt_p_mw for r in self.readings]
        batt_p_vals = [r.batt_v * r.batt_i_ma for r in self.readings if r.batt_v > 0]  # V × mA = mW
        batt_v_vals = [r.batt_v for r in self.readings if r.batt_v > 0]
        batt_i_vals = [r.batt_i_ma for r in self.readings]

        bus_p_vals = [r.bus_p_mw for r in self.readings if r.bus_v > 0 or r.bus_p_mw > 0]
        load_p_vals = [r.load_p_mw for r in self.readings if r.load_v > 0 or r.load_p_mw > 0]
        bus_v_vals_raw = [r.bus_v for r in self.readings if r.bus_v > 0]
        grid_v_vals_raw = [r.grid_v for r in self.readings if r.grid_v > 0]

        n = len(self.readings)
        expected = self.window_sec / 11.0
        completeness = min(n / expected, 1.0) if expected > 0 else 0.0

        mppt_mean = float(np.mean(mppt_vals)) if mppt_vals else 0.0
        mppt_std = float(np.std(mppt_vals)) if len(mppt_vals) > 1 else 0.0
        mppt_max = float(np.max(mppt_vals)) if mppt_vals else 0.0

        bus_p_mean = float(np.mean(bus_p_vals)) if bus_p_vals else 0.0
        load_p_mean = float(np.mean(load_p_vals)) if load_p_vals else 0.0

        return {
            'mppt_p_mean_mW': mppt_mean,
            'mppt_p_std_mW': mppt_std,
            'mppt_p_max_mW': mppt_max,
            'bus_p_mean_mW': bus_p_mean,
            'load_p_mean_mW': load_p_mean,
            'batt_p_mean_mW': float(np.mean(batt_p_vals)) if batt_p_vals else 0.0,
            'batt_v_mean': float(np.mean(batt_v_vals)) if batt_v_vals else 0.0,
            'batt_v_min': float(np.min(batt_v_vals)) if batt_v_vals else 0.0,
            'batt_v_max': float(np.max(batt_v_vals)) if batt_v_vals else 0.0,
            'batt_i_mean_ma': float(np.mean(batt_i_vals)) if batt_i_vals else 0.0,
            'bus_v_mean': float(np.mean(bus_v_vals_raw)) if bus_v_vals_raw else 0.0,
            'grid_v_mean': float(np.mean(grid_v_vals_raw)) if grid_v_vals_raw else 0.0,
            'n_samples': n,
            'completeness': completeness,
        }

    def reset(self, new_start: Optional[datetime] = None):
        """Documentation for this public API is provided in English."""
        self.readings.clear()
        self.window_start = new_start

    @property
    def count(self) -> int:
        return len(self.readings)


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
class SoCTracker:
    """
    Track deployment state of charge with energy as the primary accounting path.

    The control-facing SoC uses battery terminal energy:
        ΔSoE = (V × I × Δt) / capacity_Wh

    A coulomb-counted SoC is kept separately for diagnostics.
    """

    MAX_INTEGRATION_SEC = 3600.0

    def __init__(self, initial_soc: float = 0.5,
                 capacity_mah: float = BATTERY_CAPACITY_MAH,
                 efficiency_rte: float = BATTERY_EFFICIENCY):
        self.soc = initial_soc
        self.soc_unclamped = initial_soc
        self.capacity_mah = capacity_mah
        self.capacity_wh = max(1e-9, self.capacity_mah / 1000.0 * BATTERY_DISCHARGE_V)
        self.eta = float(np.clip(efficiency_rte, 0.01, 1.0))
        self.last_update: Optional[datetime] = None

        self.total_charge_mah = 0.0
        self.total_discharge_mah = 0.0
        self.total_charge_wh = 0.0
        self.total_discharge_wh = 0.0
        self.soc_coulomb = initial_soc
        self.soc_coulomb_unclamped = initial_soc
        self.skipped_intervals = 0

    def update(self, timestamp: datetime, current_ma: float, voltage_v: float = 0.0):
        """
        
        Args:
        """
        if self.last_update is None:
            self.last_update = timestamp
            return

        dt_sec = (timestamp - self.last_update).total_seconds()
        self.last_update = timestamp

        if dt_sec <= 0:
            return

        if dt_sec > self.MAX_INTEGRATION_SEC:
            self.skipped_intervals += 1
            dt_sec = self.MAX_INTEGRATION_SEC

        dt_h = dt_sec / 3600.0

        delta_mah_raw = current_ma * dt_h  # mA × h = mAh
        if voltage_v > 0:
            effective_voltage_v = float(voltage_v)
        else:
            effective_voltage_v = BATTERY_CHARGE_V if current_ma > 0 else BATTERY_DISCHARGE_V
        delta_wh_raw = effective_voltage_v * current_ma / 1000.0 * dt_h

        if current_ma > 0:
            effective_mah = delta_mah_raw * self.eta
            effective_wh = delta_wh_raw * self.eta
            self.total_charge_mah += effective_mah
            self.total_charge_wh += effective_wh
        elif current_ma < 0:
            effective_mah = delta_mah_raw / self.eta
            effective_wh = delta_wh_raw / self.eta
            self.total_discharge_mah += abs(effective_mah)
            self.total_discharge_wh += abs(effective_wh)
        else:
            effective_mah = 0.0
            effective_wh = 0.0

        if self.capacity_mah > 0:
            delta_soc_coulomb = effective_mah / self.capacity_mah
            self.soc_coulomb_unclamped += delta_soc_coulomb
            self.soc_coulomb = float(np.clip(self.soc_coulomb + delta_soc_coulomb, 0.0, 1.0))

        if self.capacity_wh > 0:
            delta_soc_energy = effective_wh / self.capacity_wh
            self.soc_unclamped += delta_soc_energy
            self.soc = float(np.clip(self.soc + delta_soc_energy, 0.0, 1.0))

    def update_from_buffer(self, readings: List[Reading]):
        """Documentation for this public API is provided in English."""
        for r in readings:
            self.update(r.timestamp, r.batt_i_ma, r.batt_v)

    def get_soc(self) -> float:
        return self.soc

    def set_soc(self, soc: float):
        """Documentation for this public API is provided in English."""
        self.soc_unclamped = soc
        self.soc = float(np.clip(soc, 0.0, 1.0))
        self.soc_coulomb_unclamped = soc
        self.soc_coulomb = float(np.clip(soc, 0.0, 1.0))

    def set_capacity_mah(self, capacity_mah: float):
        """Documentation for this public API is provided in English."""
        if capacity_mah > 0:
            self.capacity_mah = float(capacity_mah)
            self.capacity_wh = max(1e-9, self.capacity_mah / 1000.0 * BATTERY_DISCHARGE_V)

    def get_soc_unclamped(self) -> float:
        """Documentation for this public API is provided in English."""
        return self.soc_unclamped

    def get_stats(self) -> Dict[str, float]:
        """Documentation for this public API is provided in English."""
        return {
            'soc': self.soc,
            'soc_unclamped': self.soc_unclamped,
            'soc_coulomb': self.soc_coulomb,
            'soc_coulomb_unclamped': self.soc_coulomb_unclamped,
            'total_charge_mah': self.total_charge_mah,
            'total_discharge_mah': self.total_discharge_mah,
            'total_charge_wh': self.total_charge_wh,
            'total_discharge_wh': self.total_discharge_wh,
            'skipped_intervals': self.skipped_intervals,
            'eta_coulombic': self.eta,
            'capacity_mah': self.capacity_mah,
            'capacity_wh': self.capacity_wh,
        }


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
SOLAR_HOUR_START = 7
SOLAR_HOUR_END   = 18


def derive_load_kw_from_aggregation(
    agg: Dict[str, float],
    now: datetime,
) -> Tuple[float, str, int]:
    """

    Returns:
        load_kw, load_source, load_groups
    """
    load_measured_mw = max(0.0, float(agg.get('load_p_mean_mW', 0.0)))
    load_groups = get_load_groups(now.time())

    return load_measured_mw / 1e6, "measured", load_groups


def derive_pv_features_from_aggregation(
    agg: Dict[str, float],
    load_kw: float,
    now: datetime,
) -> Tuple[float, float, float]:
    """

    Returns:
        pv_kw, pv_support_ratio, pv_bool
    """
    bus_power_kw = max(0.0, agg.get('bus_p_mean_mW', 0.0) / 1e6)
    pv_support_ratio = float(np.clip(
        bus_power_kw / max(load_kw, 1e-9),
        0.0,
        PV_SUPPORT_RATIO_MAX,
    ))
    pv_bool = 1.0 if pv_support_ratio >= PV_SUFFICIENT_RATIO_THRESHOLD else 0.0
    return bus_power_kw, pv_support_ratio, pv_bool


def build_state_from_aggregation(
    agg: Dict[str, float],
    soc: float,
    now: datetime,
    include_pv_support_ratio: bool = False,
    include_price_obs: bool = True,
) -> np.ndarray:
    """

    5D:
      [SoC, load_kW, pv_bool, hour, day_of_week]

    6D:
      [SoC, load_kW, pv_bool, price_norm, hour, day_of_week]

    6D:
      [SoC, load_kW, pv_support_ratio, pv_bool, hour, day_of_week]

    7D:
      [SoC, load_kW, pv_support_ratio, pv_bool, price_norm, hour, day_of_week]
    
    pv_bool: 1.0 if pv/load >= 0.8, matching the training environment.
    
    Args:
    
    Returns:
        6D or 7D numpy array (float32)
    """
    load_kw, _load_source, _load_groups = derive_load_kw_from_aggregation(agg, now)

    price = get_tou_price(now.hour, now.weekday())
    price_norm = float(np.clip(price / 10.0, 0.0, 1.0))

    hour = float(now.hour)
    dow = float(now.weekday())  # 0=Mon, 6=Sun

    _pv_kw, pv_support_ratio, pv_bool = derive_pv_features_from_aggregation(agg, load_kw, now)

    if include_pv_support_ratio:
        if include_price_obs:
            state = np.array([
                soc,               # 0: SoC (0~1)
                load_kw,
                pv_support_ratio,
                pv_bool,           # 3: PV boolean (0/1)
                price_norm,
                hour,
                dow,
            ], dtype=np.float32)
        else:
            state = np.array([
                soc,               # 0: SoC (0~1)
                load_kw,
                pv_support_ratio,
                pv_bool,           # 3: PV boolean (0/1)
                hour,
                dow,
            ], dtype=np.float32)
    else:
        if include_price_obs:
            state = np.array([
                soc,         # 0: SoC (0~1)
                load_kw,
                pv_bool,     # 2: PV boolean (0/1)
                price_norm,
                hour,
                dow,
            ], dtype=np.float32)
        else:
            state = np.array([
                soc,         # 0: SoC (0~1)
                load_kw,
                pv_bool,     # 2: PV boolean (0/1)
                hour,
                dow,
            ], dtype=np.float32)

    return state


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
def determine_situation(action_kw: float, load_kw: float, pv_kw: float) -> int:
    """
    
    
    
    """
    if action_kw < -DISCHARGE_INTENT_THRESHOLD_KW:
        if load_kw <= 0.0:
            return STANDBY_FLOW_SITUATION_CODE  # No measured load; rest on grid/PV path.
        if is_invalid_partial_discharge(action_kw, load_kw, pv_kw):
            return STANDBY_FLOW_SITUATION_CODE  # Invalid discharge -> rest/grid supply.
        return 1  # Battery discharge is allowed by deployment guard semantics.
    elif action_kw > 0.0001:
        return 3
    else:
        return STANDBY_FLOW_SITUATION_CODE


def warn_load_over_discharge_limit(
    action_kw: float,
    load_kw: float,
    discharge_limit_kw: float = BATTERY_DISCHARGE_PMAX_KW,
) -> int:
    """Flag measured overload without blocking deployment discharge."""
    return int(
        action_kw < -DISCHARGE_INTENT_THRESHOLD_KW
        and load_kw > discharge_limit_kw
    )


def apply_pv_active_discharge_guard(action_kw: float, pv_active: float) -> Tuple[float, int]:
    """Block battery discharge while PV support is still active."""
    if pv_active > 0.5 and action_kw < 0:
        return 0.0, 1
    return action_kw, 0


def is_invalid_partial_discharge(
    action_kw: float,
    load_kw: float,
    pv_kw: float,
    discharge_limit_kw: float = BATTERY_DISCHARGE_PMAX_KW,
    tolerance_kw: float = 0.0001,
) -> bool:
    """
    Block only clear partial-discharge requests within the feasible solo range.

    Measured load above the battery solo limit is now a diagnostic warning only:
    the deployed load estimate can be conservative when all load banks are on.
    """
    if action_kw >= -DISCHARGE_INTENT_THRESHOLD_KW or load_kw <= 0.0:
        return False

    net_load = max(0.0, load_kw - pv_kw)
    if net_load > discharge_limit_kw + tolerance_kw:
        return False

    return abs(action_kw) < net_load - tolerance_kw


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
def load_agent(model_path: str, state_dim: int | None = 6, action_dim: int = 2,
               device: str = "cpu") -> SACAgent:
    """Documentation for this public API is provided in English."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    hidden_dim = 128
    if 'actor' in checkpoint:
        actor_state = checkpoint['actor']
        if 'fc1.weight' in actor_state:
            hidden_dim = actor_state['fc1.weight'].shape[0]
            if state_dim is None:
                state_dim = int(actor_state['fc1.weight'].shape[1])
            if 'log_std_layer.weight' in actor_state:
                action_dim = actor_state['log_std_layer.weight'].shape[0]
            elif 'fc_mean.weight' in actor_state:
                action_dim = actor_state['fc_mean.weight'].shape[0]
            elif 'fc_logstd.weight' in actor_state:
                action_dim = actor_state['fc_logstd.weight'].shape[0]

    if state_dim is None:
        state_dim = 6

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=hidden_dim,
    )
    agent.load(model_path)
    agent.device = device
    print(f"  Model loaded: state_dim={state_dim}, action_dim={action_dim}, "
          f"hidden_dim={hidden_dim}")
    return agent, action_dim


def load_model_experiment_config(model_path: str) -> Dict[str, Any]:
    """Load the experiment config saved beside a trained model, if present."""
    model_file = Path(model_path).resolve()
    config_candidates = []
    if len(model_file.parents) >= 2:
        config_candidates.append(model_file.parents[1] / "configs" / "experiment_config.yaml")
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None
    if exe_dir is not None:
        config_candidates.append(exe_dir / "_internal" / "configs" / "experiment_config.yaml")
        config_candidates.append(exe_dir / "configs" / "experiment_config.yaml")
    config_candidates.append(Path(PROJECT_ROOT).resolve() / "configs" / "experiment_config.yaml")

    config_path = next((p for p in config_candidates if p.exists()), None)
    if config_path is None:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "config" in cfg and isinstance(cfg["config"], dict):
            cfg = cfg["config"]
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        print(f"  Warning: failed to load model experiment_config.yaml: {e}")
        return {}


def infer_state_layout(model_path: str, model_state_dim: int) -> Tuple[bool, bool]:
    """Documentation for this public API is provided in English."""
    cfg = load_model_experiment_config(model_path)

    if cfg:
        env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
        use_pv_support_ratio_state = bool(env_cfg.get("pv_support_ratio_obs", model_state_dim == 7))
        use_price_obs_state = bool(env_cfg.get("price_obs", True))
        print(f"  State layout loaded: pv_support_ratio={use_pv_support_ratio_state}, "
              f"price_obs={use_price_obs_state}")
        return use_pv_support_ratio_state, use_price_obs_state

    #   7D -> [soc, load, pv_support_ratio, pv_bool, price, hour, dow]
    #   6D -> [soc, load, pv_bool, price, hour, dow]
    #   5D -> [soc, load, pv_bool, hour, dow]
    if model_state_dim >= 7:
        return True, True
    if model_state_dim == 6:
        return False, True
    if model_state_dim == 5:
        return False, False
    return False, True


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
def read_data_txt(path: str, battery_pp: str = "01") -> Optional[Reading]:
    """
    
    
    Returns:
    """
    try:
        result = read_vendor_data_file(
            path, max_age_sec=60, clear_after_read=False
        )
        mppt_data = result['mppt']
        batt_data = result['batteries']
        mppt_bus = result.get('mppt_bus')
        load_hw  = result.get('load')
        grid_hw  = result.get('grid')
    except Exception:
        return None

    now = datetime.now(TZ_UTC8)
    reading = Reading(timestamp=now)

    if mppt_data is not None:
        solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw = mppt_data
        reading.solar_v = solar_v
        reading.solar_i_ma = solar_i_ma
        reading.solar_p_mw = solar_p_mw
        reading.mppt_p_mw = mppt_p_mw
        reading.mppt_v = mppt_v
        reading.mppt_i_ma = mppt_i_ma

    if mppt_bus is not None:
        reading.bus_v = mppt_bus[0]
        reading.bus_i_ma = mppt_bus[1]
        reading.bus_p_mw = mppt_bus[2]

    if load_hw is not None:
        reading.load_v = load_hw[0]
        reading.load_i_ma = load_hw[1]
        reading.load_p_mw = load_hw[2]

    if grid_hw is not None:
        reading.grid_v = grid_hw[0]
        reading.grid_i_ma = grid_hw[1]
        reading.grid_p_mw = grid_hw[2]

    if battery_pp in batt_data:
        batt_tuple = batt_data[battery_pp]
        if len(batt_tuple) == 7:
            ts, soc_pct, volt_v, charge_v, curr_ma, temp_c, speed = batt_tuple
            reading.batt_charge_v = charge_v
        else:
            ts, soc_pct, volt_v, curr_ma, temp_c, speed = batt_tuple
        reading.batt_soc_pct = soc_pct
        reading.batt_v = volt_v
        reading.batt_i_ma = curr_ma
        reading.batt_temp_c = temp_c
        reading.batt_speed_pct = speed
        if ts is not None:
            reading.timestamp = ts

    return reading


def perform_pre_measure_for_decision(
    data_file: str,
    command_file: str,
    battery_pp: str,
    load_count: int,
    recovery_seconds: float = VOLTAGE_RECOVERY_SECONDS,
    sleep_fn=time.sleep,
    dry_run: bool = False,
) -> Optional[Reading]:
    """
    Before a model decision, briefly run mode 3 at zero power and 50% flow,
    then read a fresh Data.txt so voltage/cutoff/model inputs are observable.
    """
    now_ts = datetime.now(TZ_UTC8)
    if not dry_run:
        write_pre_measure_command(command_file, now_ts, battery_pp, load_count=load_count)
    if recovery_seconds > 0:
        sleep_fn(float(recovery_seconds))
    return read_data_txt(data_file, battery_pp=battery_pp)


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
class DeploymentLogger:
    """
    
    """

    HEADER = [
        'timestamp', 'step',
        'session_id', 'experiment_name', 'model_file', 'current_mode',
        'hybrid_min_current_ma', 'coral_delta', 'coral_buffer', 'coral_window',
        'soc', 'soc_unclamped',
        'charge_mah', 'discharge_mah',
        'soc_coulomb', 'soc_coulomb_unclamped',
        'charge_wh', 'discharge_wh',
        'load_kw', 'pv_kw', 'price', 'hour', 'dow',
        'price_norm', 'pv_support_ratio', 'pv_bool', 'pv_active',
        'load_source', 'load_fallback_used',
        'mppt_mean_mW', 'mppt_max_mW', 'mppt_std_mW',
        'bus_p_mean_mW', 'load_p_mean_mW',
        'batt_p_mean_mW', 'batt_v_mean', 'batt_i_mean_ma', 'bus_v_mean', 'grid_v_mean',
        'n_samples', 'completeness',
        'action_power_kw', 'action_flow_pct',
        'power_mw_cmd', 'flow_pct_cmd', 'situation_code',
        'load_groups', 'pv_surplus_kw', 'guard_delta_mW',
        'guard_force_charge_low_soc', 'guard_block_low_soc_discharge',
        'guard_block_high_soc_charge', 'guard_block_pv_active_discharge',
        'guard_block_voltage_cutoff', 'warn_load_over_discharge_limit',
        'guard_block_invalid_discharge',
        'guard_block_no_pv_surplus_charge',
        'guard_flow_power_limited', 'flow_charge_limit_kw', 'flow_discharge_limit_kw',
        'guard_block_discharge_intent_threshold', 'guard_block_firmware_override_discharge',
        'guard_block_isolated_load_bus_discharge',
        'guard_block_health_lock_discharge',
        'firmware_override_discharge_samples_window',
        'isolated_load_bus_samples_window',
        'voltage_cutoff_active', 'voltage_cutoff_day_locked', 'voltage_cutoff_day_count',
        'cutoff_soc_fallback_enabled', 'cutoff_soc_fallback_percent',
        'cutoff_soc_fallback_applied', 'cutoff_soc_before', 'cutoff_soc_after',
        'soh_health_enabled', 'soh_health_lock_active', 'soh_health_lock_reason',
        'soh_record_candidate', 'soh_last_record_time', 'soh_record_reason',
        'soh_low_voltage_samples_window', 'soh_low_voltage_streak',
        'soh_recovery_samples', 'soh_voltage_sag_v', 'soh_proxy_score',
        'soh_prediction_enabled', 'soh_model_path', 'soh_last_prediction_time',
        'soh_last_value', 'soh_prediction_status', 'soh_prediction_method',
        'soh_effective_capacity_mah', 'soh_use_for_capacity',
        'coral_active', 'coral_clipped', 'coral_delta_mW',
        'coral_interventions', 'coral_residual_count', 'action_raw_kw',
    ]

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._current_date: Optional[str] = None
        self.path: Optional[str] = None
        self._ensure_file(datetime.now(TZ_UTC8).strftime('%Y-%m-%d'))
        print(f"  日誌: {self.path}")

    def _ensure_file(self, date_str: str):
        """Documentation for this public API is provided in English."""
        if date_str == self._current_date:
            return
        self._current_date = date_str
        self.path = os.path.join(self.log_dir, f'deployment_v2_{date_str}.csv')
        need_header = (not os.path.exists(self.path) or
                       os.path.getsize(self.path) == 0)
        if need_header:
            with open(self.path, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(self.HEADER)
            print(f"  [Logger] 新日誌: {self.path}")

    def log(self, row: Dict[str, Any]):
        ts_str = row.get('timestamp', '')
        if ts_str:
            date_part = ts_str[:10]  # 'YYYY-MM-DD'
            self._ensure_file(date_part)
        with open(self.path, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow([row.get(k, '') for k in self.HEADER])


class RawDataLogger:
    """
    
    
    """

    HEADER = [
        'timestamp',
        'battery_id',
        'soc_percent',
        'voltage_v',
        'charge_voltage_v',
        'current_ma',
        'current_raw_ma',
        'temp_c',
        'speed_percent',
        'solar_v',
        'solar_i_ma',
        'solar_p_mw',
        'mppt_v',
        'mppt_i_ma',
        'mppt_p_mw',
        'bus_v',
        'bus_i_ma',
        'bus_p_mw',
        'load_v',
        'load_i_ma',
        'load_p_mw',
        'grid_v',
        'grid_i_ma',
        'grid_p_mw',
        'soc_calc',
        'soc_unclamped',
        'soc_coulomb',
        'soc_coulomb_unclamped',
        'charge_mah',
        'discharge_mah',
        'charge_wh',
        'discharge_wh',
        'situation_code',
    ]

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._current_date: Optional[str] = None
        self._file = None
        self._writer = None
        self.path: Optional[str] = None
        self._open_for_date(datetime.now(TZ_UTC8).strftime('%Y-%m-%d'))

    def _open_for_date(self, date_str: str):
        """Documentation for this public API is provided in English."""
        if date_str == self._current_date:
            return
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._current_date = date_str
        self.path = os.path.join(self.log_dir, f'raw_data_v2_{date_str}.csv')
        file_exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        self._file = open(self.path, 'a', newline='', encoding='utf-8-sig')
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
        if not file_exists:
            self._writer.writeheader()
            print(f"  [RawLog] New raw log: {self.path}")

    def log(self, reading: 'Reading', raw_current_ma: float,
            soc_calc: float, soc_unclamped: float,
            charge_mah: float, discharge_mah: float,
            situation_code: int, battery_pp: str,
            soc_coulomb: float = 0.0, soc_coulomb_unclamped: float = 0.0,
            charge_wh: float = 0.0, discharge_wh: float = 0.0):
        """Documentation for this public API is provided in English."""
        date_str = reading.timestamp.strftime('%Y-%m-%d')
        self._open_for_date(date_str)
        self._writer.writerow({
            'timestamp': reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'battery_id': battery_pp,
            'soc_percent': f'{reading.batt_soc_pct:.1f}',
            'voltage_v': f'{reading.batt_v:.2f}',
            'charge_voltage_v': f'{reading.batt_charge_v:.2f}',
            'current_ma': f'{reading.batt_i_ma:.1f}',
            'current_raw_ma': f'{raw_current_ma:.1f}',
            'temp_c': f'{reading.batt_temp_c:.1f}',
            'speed_percent': f'{reading.batt_speed_pct:.1f}',
            'solar_v': f'{reading.solar_v:.2f}',
            'solar_i_ma': f'{reading.solar_i_ma:.1f}',
            'solar_p_mw': f'{reading.solar_p_mw:.1f}',
            'mppt_v': f'{reading.mppt_v:.2f}',
            'mppt_i_ma': f'{reading.mppt_i_ma:.1f}',
            'mppt_p_mw': f'{reading.mppt_p_mw:.1f}',
            'bus_v': f'{reading.bus_v:.2f}',
            'bus_i_ma': f'{reading.bus_i_ma:.1f}',
            'bus_p_mw': f'{reading.bus_p_mw:.1f}',
            'load_v': f'{reading.load_v:.2f}',
            'load_i_ma': f'{reading.load_i_ma:.1f}',
            'load_p_mw': f'{reading.load_p_mw:.1f}',
            'grid_v': f'{reading.grid_v:.2f}',
            'grid_i_ma': f'{reading.grid_i_ma:.1f}',
            'grid_p_mw': f'{reading.grid_p_mw:.1f}',
            'soc_calc': f'{soc_calc:.4f}',
            'soc_unclamped': f'{soc_unclamped:.4f}',
            'soc_coulomb': f'{soc_coulomb:.4f}',
            'soc_coulomb_unclamped': f'{soc_coulomb_unclamped:.4f}',
            'charge_mah': f'{charge_mah:.2f}',
            'discharge_mah': f'{discharge_mah:.2f}',
            'charge_wh': f'{charge_wh:.4f}',
            'discharge_wh': f'{discharge_wh:.4f}',
            'situation_code': situation_code,
        })
        self._file.flush()

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="P302 即時部署控制迴圈（15 分鐘聚合 + SAC 推論）")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Data.txt 路徑（韌體寫入）")
    parser.add_argument("--command-file", type=str, required=True,
                        help="Command.txt 路徑（AI 輸出）")
    parser.add_argument("--model-path", type=str, required=True,
                        help="SAC 模型 .pth 路徑")
    parser.add_argument("--battery-pp", type=str, default="01",
                        help="目標電池 PP 編號（01-10）")
    parser.add_argument("--initial-soc", type=float, default=0.0,
                        help="初始 SoC (0~1)")
    parser.add_argument("--poll-sec", type=float, default=10.0,
                        help="Data.txt 輪詢間隔（秒），預設 10")
    parser.add_argument("--voltage-recovery-sec", type=float,
                        default=VOLTAGE_RECOVERY_SECONDS,
                        help="每次模型決策前 mode 3 / 50%% pre-measure 等待秒數，預設 25，可用 P302_VOLTAGE_RECOVERY_SECONDS 覆寫")
    parser.add_argument("--window-min", type=int, default=15,
                        help="聚合窗格（分鐘），預設 15")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--log-dir", type=str, default=None,
                        help="日誌輸出目錄（預設: results/deployment/）")
    parser.add_argument("--dry-run", action="store_true",
                        help="乾跑模式：不寫 Command.txt，僅印出決策")
    parser.add_argument("--pv-surplus-charge-only", action="store_true", default=True,
                        help="僅允許 PV surplus 充電，並將充電功率限制在 PV surplus 內（預設啟用）")
    parser.add_argument("--no-pv-surplus-charge-only", dest="pv_surplus_charge_only",
                        action="store_false",
                        help="停用 PV surplus charging guard，允許模型要求市電充電")
    parser.add_argument("--pv-surplus-charge-threshold-kw", type=float,
                        default=PV_SURPLUS_CHARGE_THRESHOLD_KW,
                        help="PV surplus 低於此值時禁止充電，預設 0.0002kW")
    parser.add_argument("--initial-action", type=str, default="standby",
                        choices=["standby", "random"],
                        help="首次 15 分鐘的動作（standby=待機, random=隨機）")
    parser.add_argument("--coral", action="store_true", default=True,
                        help="啟用 CORAL SafetyNet（預設啟用）")
    parser.add_argument("--no-coral", dest="coral", action="store_false",
                        help="停用 CORAL SafetyNet")
    parser.add_argument("--coral-delta", type=float, default=0.15,
                        help="Conformal 分位數 delta（越小越保守，預設 0.15）")
    parser.add_argument("--coral-buffer", type=float, default=0.03,
                        help="SoC 安全緩衝區比例（預設 0.03）")
    parser.add_argument("--coral-window", type=int, default=96,
                        help="Conformal 殘差視窗（步數，預設 96 = 1 天）")
    parser.add_argument("--current-mode", type=str, default="hybrid",
                        choices=["signed", "invert", "unsigned", "synthetic", "hybrid"],
                        help="電池電流符號模式: "
                             "signed=正充負放(預設), "
                             "invert=正放負充(反轉), "
                             "unsigned=全正值由Scenario推斷方向, "
                             "synthetic=忽略韌體電流,用已知系統電流+指令方向合成, "
                             "hybrid=優先用實測電流大小+Scenario方向,缺失才合成")
    parser.add_argument("--synthetic-current-ma", type=float, default=1000.0,
                        help="合成電流模式下的系統電流 (mA)，預設 1000 (V16: 1A)")
    parser.add_argument("--hybrid-min-current-ma", type=float, default=50.0,
                        help="hybrid 模式下低於此值視為電流雜訊，必要時才 fallback synthetic")
    parser.add_argument("--cutoff-soc-fallback", action="store_true", default=True,
                        help="電壓 cutoff 觸發時，將 control SoC fallback 到指定百分比（預設啟用）")
    parser.add_argument("--no-cutoff-soc-fallback", dest="cutoff_soc_fallback",
                        action="store_false",
                        help="停用電壓 cutoff 後的 control SoC fallback，只保留禁止放電保護")
    parser.add_argument("--cutoff-soc-fallback-percent", type=float,
                        default=BATTERY_CUTOFF_SOC * 100.0,
                        help="cutoff 觸發時 control SoC fallback 百分比，預設 20")
    # ── SoH-aware passive monitoring / optional protection ──
    parser.add_argument("--soh-health-protection", action="store_true", default=False,
                        help="啟用可選 SoH health lock：偵測疑似不健康電池時禁止放電（預設關閉）")
    parser.add_argument("--no-soh-health-protection", dest="soh_health_protection",
                        action="store_false",
                        help="停用 SoH-aware health lock，只被動記錄健康指標")
    parser.add_argument("--soh-low-voltage-v", type=float,
                        default=SOH_HEALTH_LOW_VOLTAGE_V,
                        help="健康鎖低電壓門檻，預設等於 cutoff 4.2V")
    parser.add_argument("--soh-low-voltage-samples", type=int,
                        default=SOH_HEALTH_LOW_VOLTAGE_SAMPLES,
                        help="連續幾筆低電壓後啟動 health lock，預設 3")
    parser.add_argument("--soh-recover-v", type=float,
                        default=SOH_HEALTH_RECOVER_V,
                        help="解除 health lock 的恢復電壓，預設 5.0V")
    parser.add_argument("--soh-recovery-samples", type=int,
                        default=SOH_HEALTH_RECOVERY_SAMPLES,
                        help="連續幾筆恢復電壓後解除 health lock，預設 12")
    parser.add_argument("--soh-sag-warning-v", type=float,
                        default=SOH_HEALTH_SAG_WARNING_V,
                        help="15 分鐘窗格內電池電壓 sag 警示門檻，預設 0.8V")
    parser.add_argument("--soh-record-soc-min", type=float,
                        default=SOH_RECORD_SOC_MIN,
                        help="自然符合 SoH 片段記錄的 SoC 下限，預設 0.25")
    parser.add_argument("--soh-record-soc-max", type=float,
                        default=SOH_RECORD_SOC_MAX,
                        help="自然符合 SoH 片段記錄的 SoC 上限，預設 0.75")
    parser.add_argument("--soh-record-current-ma", type=float,
                        default=SOH_RECORD_CURRENT_THRESHOLD_MA,
                        help="自然符合 SoH 片段記錄的平均充電電流門檻，預設 50mA")
    parser.add_argument("--soh-record-min-samples", type=int,
                        default=SOH_RECORD_MIN_SAMPLES,
                        help="自然符合 SoH 片段記錄的窗格最少樣本數，預設 30")
    parser.add_argument("--soh-prediction", action="store_true", default=False,
                        help="啟用線上 SoH prediction：合格片段形成後輸出並預測 SoH（預設關閉）")
    parser.add_argument("--no-soh-prediction", dest="soh_prediction",
                        action="store_false",
                        help="停用線上 SoH prediction，只記錄候選片段")
    parser.add_argument("--soh-model-path", type=str, default="",
                        help="SoH 模型或 predictor 資料夾路徑；可指向含 predict.py 的資料夾")
    parser.add_argument("--soh-use-for-capacity", action="store_true", default=False,
                        help="用最近一次 SoH prediction 修正 Coulomb counting 有效容量（實驗性，預設關閉）")
    parser.add_argument("--no-soh-use-for-capacity", dest="soh_use_for_capacity",
                        action="store_false",
                        help="不把 SoH prediction 套進 SoC 容量計算")
    args = parser.parse_args()
    model_path_resolved = Path(args.model_path).resolve()
    session_id = datetime.now(TZ_UTC8).strftime('%Y%m%d_%H%M%S')
    experiment_name = model_path_resolved.parents[1].name if len(model_path_resolved.parents) >= 2 else model_path_resolved.stem
    model_file = model_path_resolved.name
    model_hash = hashlib.sha1(str(model_path_resolved).encode('utf-8')).hexdigest()[:10]
    cutoff_soc_fallback = float(np.clip(args.cutoff_soc_fallback_percent / 100.0, 0.0, 1.0))

    print("=" * 70)
    print("  P302 realtime deployment control loop + CORAL SafetyNet")
    print(f"  Battery: {BATTERY_CAPACITY_MAH:.0f}mAh / {_system_current_a*1000:.0f}mA / "
          f"{BATTERY_CHARGE_V}V(charge) / {BATTERY_DISCHARGE_V}V(discharge) / {BATTERY_CUTOFF_V}V(cutoff)")
    print(f"  Charge voltage cutoff: {BATTERY_CHARGE_CUTOFF_V:.2f}V")
    print(f"  Discharge intent threshold: {DISCHARGE_INTENT_THRESHOLD_KW*1000:.2f}W")
    print(f"  Load: {MAX_LOAD_GROUPS} groups x {LOAD_PER_GROUP_W}W = {MAX_LOAD_GROUPS*LOAD_PER_GROUP_W}W")
    print("  CORAL: CRTSN + OCC + Adaptive Loop")
    print(
        f"  Cutoff SoC fallback: "
        f"{'ON' if args.cutoff_soc_fallback else 'OFF'}"
        f"{f' -> {cutoff_soc_fallback*100:.1f}%' if args.cutoff_soc_fallback else ''}"
    )
    print(
        f"  SoH passive recording: ON "
        f"(SoC {args.soh_record_soc_min*100:.0f}-{args.soh_record_soc_max*100:.0f}%, "
        f"I_charge>{args.soh_record_current_ma:.0f}mA, n>={args.soh_record_min_samples})"
    )
    print(
        f"  Optional SoH health lock: "
        f"{'ON' if args.soh_health_protection else 'OFF'} "
        f"(low<{args.soh_low_voltage_v:.2f}V x{args.soh_low_voltage_samples}, "
        f"recover>{args.soh_recover_v:.2f}V x{args.soh_recovery_samples})"
    )
    print(
        f"  Online SoH prediction: "
        f"{'ON' if args.soh_prediction else 'OFF'}"
        f"{f' ({args.soh_model_path})' if args.soh_prediction and args.soh_model_path else ''}"
    )
    print(
        f"  SoH-adjusted SoC capacity: "
        f"{'ON' if args.soh_use_for_capacity else 'OFF'}"
    )
    print("=" * 70)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    agent, action_dim = load_agent(args.model_path, state_dim=None, device=device)
    model_state_dim = int(agent.actor.fc1.weight.shape[1])
    model_hidden_dim = int(agent.actor.fc1.weight.shape[0])
    model_cfg = load_model_experiment_config(args.model_path)
    model_env_cfg = model_cfg.get("env", {}) if isinstance(model_cfg, dict) else {}
    use_pv_support_ratio_state, use_price_obs_state = infer_state_layout(args.model_path, model_state_dim)
    is_2d = (action_dim >= 2)
    model_use_flow = bool(model_env_cfg.get("use_flow_rate_action", is_2d))
    model_pv_surplus_charge_only = bool(
        model_env_cfg.get("charge_requires_pv_surplus", False)
        or model_env_cfg.get("charge_limit_to_pv_surplus", False)
    )
    model_flow_limits_available_power = bool(model_env_cfg.get("flow_limits_available_power", False))
    model_flow_power_min_fraction = float(np.clip(model_env_cfg.get("flow_power_min_fraction", 0.0), 0.0, 1.0))
    model_flow_min_pct = float(model_env_cfg.get("flow_min_active_fraction", FLOW_MIN_ACTIVE_PCT / 100.0)) * 100.0
    model_group_power_kw = model_env_cfg.get("deployment_group_power_kw", None)
    current_group_power_kw = LOAD_PER_GROUP_W / 1000.0
    print(f"  Action dimension: {action_dim} ({'power+flow' if is_2d else 'power only'})")
    print(f"  Model architecture: state_dim={model_state_dim}, hidden_dim={model_hidden_dim}")
    print(
        "  Model deployment assumptions: "
        f"flow_action={'ON' if model_use_flow else 'OFF'}, "
        f"min_active_flow={model_flow_min_pct:.0f}%, "
        f"flow_limits_power={'ON' if model_flow_limits_available_power else 'OFF'}, "
        f"pv_surplus_charge={'ON' if model_pv_surplus_charge_only else 'OFF'}"
    )
    if model_group_power_kw is not None:
        try:
            model_group_power_kw_f = float(model_group_power_kw)
        except (TypeError, ValueError):
            model_group_power_kw_f = current_group_power_kw
        if abs(model_group_power_kw_f - current_group_power_kw) > 1e-9:
            print(
                "  WARNING: model load scale mismatch: "
                f"trained deployment_group_power_kw={model_group_power_kw_f:.6f}, "
                f"current measured group power={current_group_power_kw:.6f}. "
                "Safety guards still run, but policy behavior may be out-of-distribution."
            )
    print(
        "  Deployment guards: "
        f"pv_surplus_charge={'ON' if args.pv_surplus_charge_only else 'OFF'}, "
        f"threshold={args.pv_surplus_charge_threshold_kw*1000:.1f}W"
    )
    if model_use_flow and not is_2d:
        print("  WARNING: model config expects flow action, but checkpoint action_dim is not 2D.")
    if model_pv_surplus_charge_only and not args.pv_surplus_charge_only:
        print("  WARNING: model was trained with PV-surplus charging, but deployment guard is OFF.")

    safety_net = None
    if args.coral:
        set_conformal_params(window=args.coral_window, delta=args.coral_delta)
        clear_residual_buffer()
        safety_net = SafetyNet(
            battery_capacity_kwh=BATTERY_CAPACITY_KWH,
            battery_power_kw=BATTERY_PMAX_KW,
            battery_efficiency=BATTERY_EFFICIENCY,
            soc_min=0.20,
            soc_max=0.80,
            initial_buffer_ratio=args.coral_buffer,
            min_buffer_ratio=0.01,
            boundary_epsilon=0.005,
            time_step=0.25,
            n_step_preview=2,
            enable_n_step_preview=True,
        )
        print(f"  CORAL SafetyNet: ON (delta={args.coral_delta}, buffer={args.coral_buffer})")
    else:
        print(f"  CORAL SafetyNet: OFF")

    coral_stats = {'interventions': 0, 'total_steps': 0, 'total_delta_kw': 0.0}

    soc_tracker = SoCTracker(initial_soc=args.initial_soc)
    buffer = DataBuffer(window_sec=args.window_min * 60)

    log_dir = args.log_dir or os.path.join(PROJECT_ROOT, 'results', 'deployment')
    logger = DeploymentLogger(log_dir)
    raw_logger = RawDataLogger(log_dir)
    print(f"  Raw log: {raw_logger.path}")
    soh_predictor = None
    if args.soh_prediction:
        soh_predictor = OnlineSoHPredictor(
            args.soh_model_path,
            log_dir,
            nominal_capacity_mah=BATTERY_CAPACITY_MAH,
            nominal_capacity_wh=BATTERY_CAPACITY_WH,
        )
        print(f"  SoH online predictions: {soh_predictor.prediction_csv}")

    # ── Startup self-test: model -> state -> action pipeline ─────────────
    print("\n  [self-test] Verifying deployment pipeline...")
    try:
        test_soc = 0.5
        test_now = datetime.now(TZ_UTC8)
        test_agg = {
            'mppt_p_mean_mW': 200.0, 'mppt_p_std_mW': 10.0, 'mppt_p_max_mW': 300.0,
            'bus_p_mean_mW': 0.0, 'load_p_mean_mW': 400.0,
            'batt_p_mean_mW': 0.0, 'batt_v_mean': 5.5, 'batt_i_mean_ma': 0.0,
            'bus_v_mean': 14.5, 'grid_v_mean': 12.0,
            'n_samples': 82, 'completeness': 1.0,
        }
        test_state = build_state_from_aggregation(
            test_agg, test_soc, test_now,
            include_pv_support_ratio=use_pv_support_ratio_state,
            include_price_obs=use_price_obs_state,
        )
        expected_dim = 2 + (2 if use_pv_support_ratio_state else 1) + (1 if use_price_obs_state else 0) + 2
        assert test_state.shape == (expected_dim,), (
            f"State dimension mismatch: got {test_state.shape}, expected ({expected_dim},)"
        )
        assert 0.0 <= test_state[0] <= 1.0, f"SoC out of range: {test_state[0]}"

        with torch.no_grad():
            test_action = agent.select_action(test_state, evaluate=True)
        assert len(test_action) == action_dim, (
            f"Action dimension mismatch: got {len(test_action)}, expected {action_dim}"
        )

        test_power_kw = norm_to_power_kw(float(test_action[0]))
        if action_dim >= 2:
            test_flow_pct_raw = float(np.clip((float(test_action[1]) + 1.0) * 50.0, 0.0, 100.0))
        else:
            test_flow_pct_raw = abs(float(test_action[0])) * 100.0
        test_flow_pct = apply_flow_operating_rule(
            test_flow_pct_raw,
            active=abs(test_power_kw) > 0.0001,
        )
        limit_kw = BATTERY_CHARGE_PMAX_KW if test_power_kw >= 0 else BATTERY_DISCHARGE_PMAX_KW
        assert abs(test_power_kw) <= limit_kw * 1.01, \
            f"Power command out of range: {test_power_kw*1e6:.1f}mW"
        assert 0.0 <= test_flow_pct <= 100.0, f"Flow command out of range: {test_flow_pct:.1f}%"
        if abs(test_power_kw) <= 0.0001:
            assert test_flow_pct == FLOW_REST_PCT, "Standby/rest flow is not zero"
        else:
            assert test_flow_pct >= FLOW_MIN_ACTIVE_PCT, "Active flow below safe minimum"

        test_sit = determine_situation(test_power_kw, 0.0002, 0.0002)
        assert test_sit in (1, 2, 3, 4), f"Invalid situation code: {test_sit}"

        tracker_test = SoCTracker(initial_soc=0.5, capacity_mah=BATTERY_CAPACITY_MAH)
        t0 = test_now
        tracker_test.update(t0, 0.0, 5.5)
        tracker_test.update(t0 + timedelta(seconds=60), 50.0, 5.5)
        assert tracker_test.get_soc() > 0.5, "SoCTracker charge SoC did not increase"
        tracker_test.update(t0 + timedelta(seconds=120), -50.0, 5.5)
        assert tracker_test.get_soc() < 0.5 + 0.1, "SoCTracker discharge SoC abnormal"

        print(f"  [self-test] PASS")
        print(f"    State: {test_state}")
        print(
            f"    Action: {test_action} -> {test_power_kw*1e6:.1f}mW, "
            f"flow={test_flow_pct:.1f}%, situation {test_sit}"
        )
        print(f"    SoCTracker: charge/discharge direction OK")
    except Exception as e:
        print(f"  [self-test] FAIL: {e}")
        print(f"    Deployment aborted; please check before running hardware control.")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    pp = f"{int(args.battery_pp):02d}"
    window_sec = args.window_min * 60
    step_count = 0

    last_power_w = 0.0
    last_flow_pct = 0.0
    last_sit_code = STANDBY_FLOW_SITUATION_CODE
    last_command_write: Optional[datetime] = None
    last_action_kw = 0.0
    batt_sign_warnings = 0
    batt_zero_readings = 0
    batt_zero_warned = False
    voltage_cutoff_active = False
    voltage_cutoff_count = 0
    voltage_cutoff_time: Optional[datetime] = None
    voltage_cutoff_day_count = 0
    voltage_cutoff_day_locked = False
    voltage_cutoff_day_date: Optional[object] = None
    voltage_zero_streak = 0
    voltage_cutoff_fallback_applied_this_event = False
    cutoff_soc_fallback_applied_window = 0
    cutoff_soc_before_window: Optional[float] = None
    cutoff_soc_after_window: Optional[float] = None
    firmware_override_discharge_samples_window = 0
    firmware_override_discharge_warn_count = 0
    isolated_load_bus_samples_window = 0
    isolated_load_bus_warn_count = 0
    soh_health_lock_active = False
    soh_health_lock_reason = ""
    soh_low_voltage_streak = 0
    soh_recovery_samples = 0
    soh_low_voltage_samples_window = 0
    soh_health_lock_warn_count = 0
    soh_last_record_time: Optional[datetime] = None
    soh_record_reason = ""
    soh_last_prediction_time: Optional[datetime] = None
    soh_last_value: Optional[float] = None
    soh_prediction_status = ""
    soh_prediction_method = ""
    soh_effective_capacity_mah = BATTERY_CAPACITY_MAH

    def _apply_cutoff_soc_fallback(trigger: str):
        """Apply control SoC fallback once per cutoff event, if enabled."""
        nonlocal voltage_cutoff_fallback_applied_this_event
        nonlocal cutoff_soc_fallback_applied_window
        nonlocal cutoff_soc_before_window, cutoff_soc_after_window

        if not args.cutoff_soc_fallback or voltage_cutoff_fallback_applied_this_event:
            return

        before = soc_tracker.get_soc()
        soc_tracker.set_soc(cutoff_soc_fallback)
        after = soc_tracker.get_soc()
        voltage_cutoff_fallback_applied_this_event = True
        cutoff_soc_fallback_applied_window = 1
        cutoff_soc_before_window = before
        cutoff_soc_after_window = after
        print(
            f"     Cutoff SoC fallback ({trigger}): "
            f"{before*100:.1f}% -> {after*100:.1f}%"
        )

    print(f"\n  Data.txt  : {args.data_file}")
    print(f"  Command   : {args.command_file}")
    print(f"  模型      : {args.model_path}")
    print(f"  Session   : {session_id}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Model tag : {model_file}#{model_hash}")
    print(f"  電池 PP   : {pp}")
    print(f"  初始 SoC  : {args.initial_soc * 100:.0f}%")
    print(f"  輪詢間隔  : {args.poll_sec}s")
    print(f"  Pre-measure: mode {PRE_MEASURE_SITUATION_CODE}, power=0mW, "
          f"flow={FLOW_PRE_MEASURE_PCT:.0f}%, wait={args.voltage_recovery_sec:.1f}s")
    print(f"  Rest mode  : mode {STANDBY_SITUATION_CODE}, power=0mW, "
          f"flow={FLOW_REST_PCT:.0f}%")
    print(f"  聚合窗格  : {args.window_min} min")
    print(f"  首次動作  : {args.initial_action}")
    _mode_labels = {
        "signed": "正充負放",
        "invert": "正放負充(反轉)",
        "unsigned": "全正值(由Scenario推斷)",
        "synthetic": f"合成電流({args.synthetic_current_ma:.0f}mA, 忽略韌體)",
        "hybrid": f"混合電流(實測優先, 低於{args.hybrid_min_current_ma:.0f}mA才合成)",
    }
    print(f"  Current mode: {args.current_mode} ({_mode_labels[args.current_mode]})")
    print(f"  Dry run     : {args.dry_run}")
    print()
    print("-" * 70)
    print(f"  開始時間: {datetime.now(TZ_UTC8).strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    if args.initial_action == "random":
        rnd_power_kw = float(np.random.uniform(-BATTERY_PMAX_KW, BATTERY_PMAX_KW))
        rnd_flow_pct = float(np.random.uniform(FLOW_MIN_ACTIVE_PCT, 100.0))
        last_power_w = abs(rnd_power_kw) * 1e3  # kW → W
        last_flow_pct = apply_flow_operating_rule(rnd_flow_pct, active=abs(rnd_power_kw) > 0.0001)
        last_sit_code = determine_situation(rnd_power_kw, 0.0, 0.0)
        print(f"  [初始] 隨機動作: power={last_power_w*1000:.1f}mW ({last_power_w:.4f}W), flow={last_flow_pct:.0f}%")
    else:
        last_flow_pct = FLOW_REST_PCT
        print(f"  [初始] 待機/rest: power=0mW, flow={FLOW_REST_PCT:.0f}%")

    try:
        while True:
            loop_start = time.time()
            now = datetime.now(TZ_UTC8)

            reading = read_data_txt(args.data_file, battery_pp=pp)
            if reading is not None:
                raw_i_orig = reading.batt_i_ma
                firmware_override_discharge_now = False
                isolated_load_bus_now = False
                if args.current_mode == "invert":
                    reading.batt_i_ma = -reading.batt_i_ma
                elif args.current_mode == "unsigned":
                    magnitude = abs(reading.batt_i_ma)
                    if last_sit_code == 1:
                        reading.batt_i_ma = -magnitude
                    elif last_sit_code == 3:
                        reading.batt_i_ma = magnitude
                    else:
                        pass
                elif args.current_mode == "synthetic":
                    #
                    if last_sit_code == 3:
                        max_power_w = BATTERY_PMAX_KW * 1e3  # 5.6 W
                        ratio = min(last_power_w / max_power_w, 1.0) if max_power_w > 0 else 0.0
                        reading.batt_i_ma = args.synthetic_current_ma * ratio
                    elif last_sit_code == 1:
                        load_w = get_load_groups(now.time()) * LOAD_PER_GROUP_W  # e.g. 4 × 2.5W = 10W
                        discharge_ma = load_w / BATTERY_DISCHARGE_V * 1000.0  # e.g. 10/5.6*1000 ≈ 1786 mA
                        reading.batt_i_ma = -discharge_ma
                    else:
                        reading.batt_i_ma = 0.0
                elif args.current_mode == "hybrid":
                    measured_mag = abs(raw_i_orig)
                    min_current_ma = max(0.0, float(args.hybrid_min_current_ma))

                    if last_sit_code == 3:
                        max_power_w = BATTERY_CHARGE_PMAX_KW * 1e3
                        ratio = min(last_power_w / max(max_power_w, 1e-9), 1.0)
                        fallback_ma = args.synthetic_current_ma * ratio
                        reading.batt_i_ma = measured_mag if measured_mag >= min_current_ma else fallback_ma
                    elif last_sit_code == 1:
                        measured_load_w = max(0.0, float(reading.load_p_mw) / 1000.0)
                        measured_grid_w = max(0.0, float(reading.grid_p_mw) / 1000.0)
                        load_w = measured_load_w
                        fallback_ma = load_w / BATTERY_DISCHARGE_V * 1000.0
                        discharge_mag = measured_mag if measured_mag >= min_current_ma else fallback_ma
                        if (
                            reading.batt_v > 0.0
                            and reading.batt_v <= ISOLATED_LOAD_DROP_VOLTAGE_MAX_V
                            and abs(raw_i_orig) <= ISOLATED_LOAD_DROP_CURRENT_MAX_MA
                            and measured_load_w <= ISOLATED_LOAD_DROP_LOAD_MAX_W
                            and measured_grid_w <= ISOLATED_LOAD_DROP_GRID_MAX_W
                        ):
                            isolated_load_bus_now = True
                            isolated_load_bus_samples_window += 1
                            reading.batt_i_ma = 0.0
                        elif (
                            raw_i_orig > FIRMWARE_OVERRIDE_DISCHARGE_CURRENT_MA
                            and reading.batt_v >= FIRMWARE_OVERRIDE_DISCHARGE_VOLTAGE_V
                        ):
                            firmware_override_discharge_now = True
                            firmware_override_discharge_samples_window += 1
                            # Commanded discharge but firmware reports charge-like current.
                            # Do not integrate this as charge; the next window blocks discharge.
                            reading.batt_i_ma = 0.0
                        else:
                            reading.batt_i_ma = -discharge_mag
                    else:
                        reading.batt_i_ma = 0.0

                if firmware_override_discharge_now:
                    firmware_override_discharge_warn_count += 1
                    if firmware_override_discharge_warn_count <= 5:
                        print(
                            "  ⚠ 偵測韌體覆蓋放電：Scenario1 但韌體回報正電流，"
                            f"V={reading.batt_v:.2f}V I_raw={raw_i_orig:.0f}mA"
                        )
                if isolated_load_bus_now:
                    isolated_load_bus_warn_count += 1
                    if isolated_load_bus_warn_count <= 5:
                        print(
                            "  ⚠ 偵測 sit=1 斷載：load/grid 幾乎為 0，暫停放電積分，"
                            f"V={reading.batt_v:.2f}V I_raw={raw_i_orig:.0f}mA "
                            f"load={reading.load_p_mw/1000.0:.3f}W grid={reading.grid_p_mw/1000.0:.3f}W"
                        )

                if args.current_mode == "signed" and abs(last_action_kw) > 1e-6 and abs(raw_i_orig) > 1:
                    cmd_charging = (last_action_kw > 0)
                    meas_positive = (raw_i_orig > 0)
                    if cmd_charging != meas_positive:
                        batt_sign_warnings += 1
                        if batt_sign_warnings <= 5:
                            print(f"  ⚠ 電流方向不一致！命令={'充電' if cmd_charging else '放電'}"
                                  f"  量測I={raw_i_orig:.0f}mA（{'正' if meas_positive else '負'}）")
                            if batt_sign_warnings == 5:
                                print(f"    → 建議切換為 --current-mode invert 或 unsigned")

                buffer.add(reading)

                soc_tracker.update(
                    reading.timestamp, reading.batt_i_ma, reading.batt_v
                )

                soc_stats = soc_tracker.get_stats()
                raw_logger.log(
                    reading, raw_current_ma=raw_i_orig,
                    soc_calc=soc_tracker.get_soc(),
                    soc_unclamped=soc_tracker.get_soc_unclamped(),
                    charge_mah=soc_stats['total_charge_mah'],
                    discharge_mah=soc_stats['total_discharge_mah'],
                    situation_code=last_sit_code,
                    battery_pp=pp,
                    soc_coulomb=soc_stats['soc_coulomb'],
                    soc_coulomb_unclamped=soc_stats['soc_coulomb_unclamped'],
                    charge_wh=soc_stats['total_charge_wh'],
                    discharge_wh=soc_stats['total_discharge_wh'],
                )

                if reading.batt_v == 0.0 and raw_i_orig == 0.0:
                    batt_zero_readings += 1
                    if batt_zero_readings == 30 and not batt_zero_warned:
                        batt_zero_warned = True
                        print("\n" + "!" * 60)
                        print("  !! 警告：電池 V=0, I=0 已持續 30 筆 !!")
                        print("  !! 可能原因：")
                        print("  !!   1. 電池未接上 / PP 編號不符")
                        print("  !!   2. Data.txt 格式不含電池行")
                        print("  !!   3. 韌體尚未回報電池數據")
                        print("  !! SoC 將維持在初始值，模型決策可能不準確")
                        print("!" * 60 + "\n")
                else:
                    batt_zero_readings = 0

                _now_dt = datetime.now(TZ_UTC8)
                _today = _now_dt.date()

                if voltage_cutoff_day_date != _today:
                    voltage_cutoff_day_date = _today
                    voltage_cutoff_day_count = 0
                    voltage_cutoff_day_locked = False

                voltage_observation_reliable = (
                    abs(last_power_w) > 0.0
                    or last_flow_pct >= FLOW_PRE_MEASURE_PCT
                )

                def _force_standby():
                    nonlocal last_sit_code, last_power_w, last_flow_pct, last_action_kw
                    last_sit_code = STANDBY_SITUATION_CODE
                    last_power_w = 0.0
                    last_flow_pct = FLOW_REST_PCT
                    last_action_kw = 0.0
                    write_standby_rest_command(
                        args.command_file, _now_dt, pp,
                        load_count=get_load_groups(now.time()),
                    )

                # ── SoH-aware passive monitor / optional health lock ─────
                # Always record low-voltage evidence. Only the explicit optional
                # health lock may change actions; passive SoH recording never does.
                if reading.batt_v > 0:
                    if reading.batt_v < args.soh_low_voltage_v:
                        soh_low_voltage_streak += 1
                        soh_recovery_samples = 0
                        soh_low_voltage_samples_window += 1
                    else:
                        soh_low_voltage_streak = 0
                        if soh_health_lock_active and reading.batt_v >= args.soh_recover_v:
                            soh_recovery_samples += 1
                        elif soh_health_lock_active:
                            soh_recovery_samples = 0

                    if (
                        args.soh_health_protection
                        and
                        not soh_health_lock_active
                        and soh_low_voltage_streak >= max(1, args.soh_low_voltage_samples)
                    ):
                        soh_health_lock_active = True
                        soh_health_lock_reason = (
                            f"low_voltage_streak:{soh_low_voltage_streak}"
                            f"@{reading.batt_v:.2f}V"
                        )
                        soh_health_lock_warn_count += 1
                        print(
                            "\n  !! SoH health lock ON: "
                            f"{soh_health_lock_reason}; discharge disabled"
                        )

                    if (
                        args.soh_health_protection
                        and
                        soh_health_lock_active
                        and soh_recovery_samples >= max(1, args.soh_recovery_samples)
                    ):
                        print(
                            "\n  SoH health lock OFF: "
                            f"V recovered >= {args.soh_recover_v:.2f}V "
                            f"for {soh_recovery_samples} samples"
                        )
                        soh_health_lock_active = False
                        soh_health_lock_reason = ""
                        soh_recovery_samples = 0
                        soh_low_voltage_streak = 0

                    if args.soh_health_protection and soh_health_lock_active and last_sit_code in (1, 2):
                        _force_standby()

                if voltage_cutoff_day_locked:
                    _force_standby()
                elif reading.batt_v > 0 and voltage_observation_reliable:
                    voltage_zero_streak = 0
                    if reading.batt_v < BATTERY_CUTOFF_V and not voltage_cutoff_active:
                        voltage_cutoff_active = True
                        voltage_cutoff_time = _now_dt
                        voltage_cutoff_count += 1
                        voltage_cutoff_day_count += 1
                        print(f"\n  !! 電壓截止 #{voltage_cutoff_count} !! "
                              f"V={reading.batt_v:.2f}V < {BATTERY_CUTOFF_V}V "
                              f"-> 強制待機 (今日 {voltage_cutoff_day_count}/{CUTOFF_MAX_PER_DAY})")
                        print(f"     恢復條件: V >= {BATTERY_CUTOFF_RECOVER_V}V "
                              f"+ 冷卻 {CUTOFF_COOLDOWN_SEC}s")
                        _apply_cutoff_soc_fallback("realtime")
                        _force_standby()
                        if voltage_cutoff_day_count >= CUTOFF_MAX_PER_DAY:
                            voltage_cutoff_day_locked = True
                            print(f"  !! 今日截止次數 {voltage_cutoff_day_count} 達上限 "
                                  f"-> 整天鎖定 standby，電池疑似退化！")
                    elif voltage_cutoff_active:
                        cooldown_ok = (
                            voltage_cutoff_time is not None
                            and (_now_dt - voltage_cutoff_time).total_seconds()
                            >= CUTOFF_COOLDOWN_SEC
                        )
                        if (reading.batt_v >= BATTERY_CUTOFF_RECOVER_V
                                and cooldown_ok):
                            voltage_cutoff_active = False
                            voltage_cutoff_fallback_applied_this_event = False
                            print(f"\n  電壓恢復: V={reading.batt_v:.2f}V >= "
                                  f"{BATTERY_CUTOFF_RECOVER_V}V, "
                                  f"冷卻 {CUTOFF_COOLDOWN_SEC}s 已過, 解除截止")
                        else:
                            if last_sit_code in (1, 2):
                                _force_standby()
                elif voltage_observation_reliable:
                    voltage_zero_streak += 1
                    if voltage_zero_streak == CUTOFF_ZERO_V_STREAK:
                        print(
                            f"\n  !! V=0 noisy/missing reading streak: "
                            f"{voltage_zero_streak}; ignore for voltage cutoff"
                        )
                else:
                    voltage_zero_streak = 0

                if buffer.count % 10 == 1:
                    ts_str = reading.timestamp.strftime('%H:%M:%S')
                    cv_tag = f" CV={reading.batt_charge_v:.2f}V" if reading.batt_charge_v > 0 else ""
                    batt_status = f"V={reading.batt_v:.2f}V{cv_tag} I={reading.batt_i_ma:.0f}mA"
                    if reading.batt_v == 0 and reading.batt_i_ma == 0:
                        batt_status += " [NO BATT!]"
                    cutoff_tag = " [V-CUTOFF!]" if voltage_cutoff_active else ""
                    health_tag = " [HEALTH-LOCK!]" if soh_health_lock_active else ""
                    print(f"  [{ts_str}] #{buffer.count:3d}  "
                          f"MPPT={reading.mppt_p_mw:6.0f}mW  {batt_status}  "
                          f"SoC={soc_tracker.get_soc()*100:.1f}%{cutoff_tag}{health_tag}")
            else:
                ts_str = now.strftime('%H:%M:%S')
                if buffer.count == 0:
                    print(f"  [{ts_str}] 等待 Data.txt...")

            if buffer.is_window_complete(now) and buffer.count > 0:
                decision_load_groups = get_load_groups(now.time())
                print(
                    f"\n  [pre-measure] mode {PRE_MEASURE_SITUATION_CODE}, "
                    f"PP={pp}, power=0mW, flow={FLOW_PRE_MEASURE_PCT:.0f}% "
                    f"for {args.voltage_recovery_sec:.1f}s before decision"
                )
                fresh_reading = perform_pre_measure_for_decision(
                    args.data_file,
                    args.command_file,
                    battery_pp=pp,
                    load_count=decision_load_groups,
                    recovery_seconds=max(0.0, float(args.voltage_recovery_sec)),
                    dry_run=args.dry_run,
                )
                if fresh_reading is not None:
                    reading = fresh_reading
                    buffer.add(fresh_reading)
                    soc_tracker.update(
                        fresh_reading.timestamp,
                        fresh_reading.batt_i_ma,
                        fresh_reading.batt_v,
                    )
                    print(
                        f"  [pre-measure] fresh Data.txt: "
                        f"V={fresh_reading.batt_v:.2f}V "
                        f"I={fresh_reading.batt_i_ma:.0f}mA "
                        f"MPPT={fresh_reading.mppt_p_mw:.0f}mW"
                    )
                else:
                    print("  [pre-measure] fresh Data.txt not available; using latest buffered data")

                step_count += 1
                print(f"\n{'='*60}")
                print(f"  [Step {step_count}] 15 分鐘聚合 ({buffer.count} 筆)")
                print(f"{'='*60}")

                agg = buffer.aggregate()
                soc = soc_tracker.get_soc()

                mppt_mw = agg['mppt_p_mean_mW']
                print(f"  MPPT 平均: {mppt_mw:.1f} mW ({mppt_mw/1000:.4f} W)")
                print(f"  MPPT 最大: {agg['mppt_p_max_mW']:.1f} mW  標準差: {agg['mppt_p_std_mW']:.1f} mW")
                if agg['bus_p_mean_mW'] > 0:
                    print(f"  MPPT-Bus: {agg['bus_p_mean_mW']:.1f} mW")
                if agg['load_p_mean_mW'] > 0:
                    flag = "" if agg['load_p_mean_mW'] >= 50.0 else " [低實測值,仍直接使用]"
                    print(f"  負載實測: {agg['load_p_mean_mW']:.1f} mW{flag}")
                print(f"  電池功率: {agg['batt_p_mean_mW']:.1f} mW "
                      f"(V={agg['batt_v_mean']:.2f}V, I={agg['batt_i_mean_ma']:.1f}mA)")
                print(f"  完整度: {agg['completeness']*100:.0f}% ({agg['n_samples']} 筆)")
                print(f"  SoC (自算): {soc*100:.1f}%")
                soh_voltage_sag_v = max(0.0, agg.get('batt_v_max', 0.0) - agg.get('batt_v_min', 0.0))
                soh_low_ratio = (
                    soh_low_voltage_samples_window / max(1, int(agg.get('n_samples', 0)))
                )
                soh_proxy_score = float(np.clip(
                    100.0
                    - 60.0 * soh_low_ratio
                    - 20.0 * max(0.0, soh_voltage_sag_v - args.soh_sag_warning_v)
                    - (25.0 if soh_health_lock_active else 0.0),
                    0.0,
                    100.0,
                ))
                soh_record_candidate = int(
                    args.soh_record_soc_min <= soc <= args.soh_record_soc_max
                    and agg.get('n_samples', 0) >= max(1, args.soh_record_min_samples)
                    and agg.get('batt_v_mean', 0.0) >= args.soh_low_voltage_v
                    and agg.get('batt_i_mean_ma', 0.0) >= args.soh_record_current_ma
                )
                if soh_record_candidate:
                    soh_last_record_time = now
                    soh_record_reason = (
                        f"natural_charge_window:"
                        f"soc={soc:.3f},"
                        f"i={agg.get('batt_i_mean_ma', 0.0):.1f}mA,"
                        f"v={agg.get('batt_v_mean', 0.0):.2f}V"
                    )
                    print(
                        "  SoH record candidate: "
                        f"{soh_record_reason}; last_record={soh_last_record_time:%Y-%m-%d %H:%M:%S}"
                    )
                    if soh_predictor is not None:
                        try:
                            pred = soh_predictor.predict_from_readings(
                                list(buffer.readings),
                                step=step_count,
                                timestamp=now,
                            )
                            soh_prediction_status = str(pred.get("status", ""))
                            soh_prediction_method = str(pred.get("method", ""))
                            if pred.get("soh") != "":
                                soh_last_value = float(pred["soh"])
                                soh_last_prediction_time = now
                                soh_effective_capacity_mah = float(pred.get("capacity_mah", BATTERY_CAPACITY_MAH))
                                print(
                                    "  SoH prediction: "
                                    f"{soh_last_value*100:.1f}% "
                                    f"({soh_prediction_method}), "
                                    f"effective_capacity={soh_effective_capacity_mah:.0f}mAh"
                                )
                                if args.soh_use_for_capacity:
                                    soc_tracker.set_capacity_mah(soh_effective_capacity_mah)
                                    if safety_net is not None:
                                        safety_net.battery_capacity_kwh = (
                                            BATTERY_CAPACITY_KWH
                                            * soh_effective_capacity_mah
                                            / max(BATTERY_CAPACITY_MAH, 1e-9)
                                        )
                                    print(
                                        "  SoH capacity applied to SoC tracker "
                                        f"(capacity={soh_effective_capacity_mah:.0f}mAh)"
                                    )
                        except Exception as exc:
                            soh_prediction_status = "ERROR"
                            soh_prediction_method = "ONLINE"
                            print(f"  !! SoH prediction failed: {exc}")
                if soh_low_voltage_samples_window > 0 or soh_health_lock_active:
                    print(
                        f"  Health: score={soh_proxy_score:.1f}/100, "
                        f"lowV_samples={soh_low_voltage_samples_window}, "
                        f"sag={soh_voltage_sag_v:.2f}V, "
                        f"lock={'ON' if soh_health_lock_active else 'OFF'}"
                    )

                state = build_state_from_aggregation(
                    agg, soc, now,
                    include_pv_support_ratio=use_pv_support_ratio_state,
                    include_price_obs=use_price_obs_state,
                )
                load_kw, load_source, load_groups = derive_load_kw_from_aggregation(agg, now)
                price = get_tou_price(now.hour, now.weekday())
                price_norm = float(np.clip(price / 10.0, 0.0, 1.0))
                if use_pv_support_ratio_state:
                    pv_support_ratio = state[2]
                    pv_bool = state[3]
                    state_hour = state[5] if use_price_obs_state else state[4]
                    state_dow = state[6] if use_price_obs_state else state[5]
                else:
                    _, pv_support_ratio, pv_bool = derive_pv_features_from_aggregation(agg, load_kw, now)
                    pv_bool = state[2]
                    state_hour = state[4] if use_price_obs_state else state[3]
                    state_dow = state[5] if use_price_obs_state else state[4]

                pv_kw, _pv_support_ratio_check, _pv_bool_check = derive_pv_features_from_aggregation(agg, load_kw, now)
                pv_active = float(pv_kw > PV_PRESENT_THRESHOLD_KW)
                load_fallback_used = int(load_source != "measured")

                bus_v_str = f"bus={agg.get('bus_v_mean', 0):.1f}V"
                grid_v_str = f"grid={agg.get('grid_v_mean', 0):.1f}V"
                print(f"\n  State: SoC={state[0]:.3f}, Load={load_kw*1e6:.1f}mW({load_groups}組), "
                      f"PV_ratio={pv_support_ratio:.2f}, PV_bool={pv_bool:.0f}, PV_active={pv_active:.0f} "
                      f"({bus_v_str} {grid_v_str}), "
                      f"Price={price:.2f}, Hour={int(state_hour)}, DoW={int(state_dow)}")

                with torch.no_grad():
                    action_norm = agent.select_action(state, evaluate=True)

                if is_2d:
                    power_norm = float(action_norm[0])   # [-1, 1]
                    flow_norm = float(action_norm[1])    # [0, 1] or [-1, 1]
                    flow_pct = float(np.clip((flow_norm + 1) / 2 * 100, 0, 100))
                else:
                    power_norm = float(action_norm[0])
                    flow_pct = abs(power_norm) * 100.0

                action_kw_raw = norm_to_power_kw(power_norm)
                action_kw = action_kw_raw
                guard_force_charge_low_soc = 0
                guard_block_low_soc_discharge = 0
                guard_block_high_soc_charge = 0
                guard_block_pv_active_discharge = 0
                guard_block_voltage_cutoff = 0
                warn_load_over_discharge_limit_flag = 0
                guard_block_invalid_discharge = 0
                guard_block_no_pv_surplus_charge = 0
                guard_flow_power_limited = 0
                flow_charge_limit_kw = BATTERY_CHARGE_PMAX_KW
                flow_discharge_limit_kw = BATTERY_DISCHARGE_PMAX_KW
                guard_block_discharge_intent_threshold = 0
                guard_block_firmware_override_discharge = 0
                guard_block_isolated_load_bus_discharge = 0
                guard_block_health_lock_discharge = 0

                coral_clipped = False
                coral_delta_kw = 0.0
                if safety_net is not None:
                    state_soc = np.array([soc], dtype=np.float32)
                    action_arr = np.array([action_kw_raw], dtype=np.float32)
                    proj_result = safety_net.project(state_soc, action_arr)
                    action_kw_safe = float(proj_result[0]) if isinstance(proj_result[0], (int, float)) else float(proj_result[0][0])
                    info_proj = proj_result[1] if len(proj_result) > 1 else {}
                    coral_clipped = info_proj.get('clipped', abs(action_kw_safe - action_kw_raw) > 1e-8)
                    coral_delta_kw = abs(action_kw_safe - action_kw_raw)
                    action_kw = action_kw_safe

                    coral_stats['total_steps'] += 1
                    if coral_clipped:
                        coral_stats['interventions'] += 1
                        coral_stats['total_delta_kw'] += coral_delta_kw
                        print(f"  [CORAL] clipped: {action_kw_raw*1e6:.1f}mW -> {action_kw*1e6:.1f}mW "
                              f"(Δ={coral_delta_kw*1e6:.1f}mW, SoC={soc*100:.1f}%)")

                    update_conformal_residual(coral_delta_kw)

                latest_batt_v = agg.get('batt_v_mean', 5.5) if agg else 5.5
                if reading is not None and reading.batt_v > 0:
                    latest_batt_v = reading.batt_v
                latest_charge_v = reading.batt_charge_v if reading is not None else 0.0
                charge_guard_v = max(float(latest_batt_v or 0.0), float(latest_charge_v or 0.0))

                if soc <= 0.20 and action_kw < 0:
                    action_kw = 0.0
                    guard_block_low_soc_discharge = 1
                    print(f"  ⚠ SoC 過低 ({soc*100:.1f}%)，禁止放電")
                elif soc >= 0.80 and action_kw > 0:
                    action_kw = 0.0
                    guard_block_high_soc_charge = 1
                    print(f"  ⚠ SoC 過高 ({soc*100:.1f}%)，禁止充電")
                elif charge_guard_v >= BATTERY_CHARGE_CUTOFF_V and action_kw > 0:
                    action_kw = 0.0
                    guard_block_high_soc_charge = 1
                    print(
                        f"  ⚠ 充電電壓過高 (V={charge_guard_v:.2f}V >= "
                        f"{BATTERY_CHARGE_CUTOFF_V:.2f}V)，禁止充電"
                    )

                if (
                    action_kw < 0
                    and abs(action_kw) < DISCHARGE_INTENT_THRESHOLD_KW
                ):
                    action_kw = 0.0
                    guard_block_discharge_intent_threshold = 1
                    print(
                        f"  ⚠ 放電意圖過小 "
                        f"({abs(action_kw_raw)*1000:.2f}W < "
                        f"{DISCHARGE_INTENT_THRESHOLD_KW*1000:.2f}W)，視為待機"
                    )

                pv_surplus_kw = max(0.0, float(pv_kw) - float(load_kw))
                if args.pv_surplus_charge_only and action_kw > 0:
                    threshold_kw = max(0.0, float(args.pv_surplus_charge_threshold_kw))
                    if pv_surplus_kw <= threshold_kw:
                        action_kw = 0.0
                        guard_block_no_pv_surplus_charge = 1
                        print(
                            f"  ⚠ PV surplus 不足 "
                            f"({pv_surplus_kw*1000:.2f}W <= {threshold_kw*1000:.2f}W)，禁止充電"
                        )
                    elif action_kw > pv_surplus_kw:
                        print(
                            f"  [pv-surplus-guard] charge limited "
                            f"{action_kw*1000:.2f}W -> {pv_surplus_kw*1000:.2f}W"
                        )
                        action_kw = pv_surplus_kw

                warn_load_over_discharge_limit_flag = warn_load_over_discharge_limit(
                    action_kw,
                    load_kw,
                )
                if warn_load_over_discharge_limit_flag:
                    print(
                        f"  ⚠ 負載 {load_kw*1000:.2f}W > 電池放電上限 "
                        f"{BATTERY_DISCHARGE_PMAX_KW*1000:.2f}W，僅記錄警告，仍允許模型放電"
                    )

                action_kw, pv_active_guard_flag = apply_pv_active_discharge_guard(action_kw, pv_active)
                if pv_active_guard_flag:
                    guard_block_pv_active_discharge = pv_active_guard_flag
                    print(f"  ⚠ PV 仍在供電中 (pv_kw={pv_kw*1e6:.0f}mW)，禁止放電")

                if (
                    firmware_override_discharge_samples_window > 0
                    and action_kw < 0
                ):
                    action_kw = 0.0
                    guard_block_firmware_override_discharge = 1
                    print(
                        "  ⚠ 上個窗格偵測到韌體覆蓋放電（實際在充電），"
                        "本窗格禁止放電避免誤扣 SoC"
                    )
                if (
                    isolated_load_bus_samples_window > 0
                    and action_kw < 0
                ):
                    action_kw = 0.0
                    guard_block_isolated_load_bus_discharge = 1
                    print(
                        "  ⚠ 上個窗格偵測到 sit=1 斷載（load bus 幾乎 0W），"
                        "本窗格禁止放電避免假放電積分"
                    )

                if soh_health_lock_active and action_kw < 0:
                    action_kw = 0.0
                    guard_block_health_lock_discharge = 1
                    print(
                        "  ⚠ SoH health lock 生效，禁止放電 "
                        f"({soh_health_lock_reason or 'health_lock'})"
                    )

                if voltage_cutoff_active:
                    if action_kw < 0:
                        action_kw = 0.0
                        guard_block_voltage_cutoff = 1
                        print(f"  !! 電壓截止中 (V={latest_batt_v:.2f}V)，禁止放電")
                elif latest_batt_v > 0 and latest_batt_v < BATTERY_CUTOFF_V and action_kw < 0:
                    action_kw = 0.0
                    voltage_cutoff_active = True
                    voltage_cutoff_time = datetime.now(TZ_UTC8)
                    voltage_cutoff_count += 1
                    voltage_cutoff_day_count += 1
                    guard_block_voltage_cutoff = 1
                    print(f"  !! 電壓截止 ({latest_batt_v:.2f}V < {BATTERY_CUTOFF_V}V)，禁止放電")
                    _apply_cutoff_soc_fallback("decision")
                    soc = soc_tracker.get_soc()
                    if voltage_cutoff_day_count >= CUTOFF_MAX_PER_DAY:
                        voltage_cutoff_day_locked = True
                        print(f"  !! 今日截止達上限 -> 整天鎖定 standby")

                if action_kw < -DISCHARGE_INTENT_THRESHOLD_KW:
                    solo_target_kw = min(load_kw, BATTERY_DISCHARGE_PMAX_KW)
                    if solo_target_kw > 0:
                        action_kw = -solo_target_kw
                        print(
                            f"  [discharge-auto] negative action -> Battery Solo target "
                            f"{solo_target_kw*1000:.2f}W"
                        )

                if is_invalid_partial_discharge(action_kw, load_kw, pv_kw):
                    action_kw = 0.0
                    guard_block_invalid_discharge = 1
                    print("  ⚠ 放電無法獨立接管負載（Grid Support 已禁用），改為待機")

                flow_pct = apply_flow_operating_rule(flow_pct, active=abs(action_kw) > 0.0001)
                if model_flow_limits_available_power and abs(action_kw) > 0.0001:
                    flow_fraction_for_power = float(np.clip(
                        flow_pct / 100.0,
                        model_flow_power_min_fraction,
                        1.0,
                    ))
                    flow_charge_limit_kw = BATTERY_CHARGE_PMAX_KW * flow_fraction_for_power
                    flow_discharge_limit_kw = BATTERY_DISCHARGE_PMAX_KW * flow_fraction_for_power
                    before_flow_power_limit_kw = action_kw
                    action_kw = float(np.clip(
                        action_kw,
                        -flow_discharge_limit_kw,
                        flow_charge_limit_kw,
                    ))
                    if abs(action_kw - before_flow_power_limit_kw) > 1e-9:
                        guard_flow_power_limited = 1
                        print(
                            "  [flow-power-guard] "
                            f"flow={flow_pct:.0f}% limits power "
                            f"{before_flow_power_limit_kw*1000:.2f}W -> {action_kw*1000:.2f}W"
                        )
                else:
                    flow_charge_limit_kw = BATTERY_CHARGE_PMAX_KW
                    flow_discharge_limit_kw = BATTERY_DISCHARGE_PMAX_KW

                sit_code = determine_situation(action_kw, load_kw, pv_kw)
                flow_pct = apply_flow_operating_rule(flow_pct, active=abs(action_kw) > 0.0001)

                power_w = abs(action_kw) * 1e3  # kW → W
                power_mw_display = power_w * 1000.0
                guard_delta_mw = abs(action_kw - action_kw_raw) * 1e6
                direction = "充電" if action_kw > 0.0001 else ("放電" if action_kw < -0.0001 else "待機")

                print(f"\n  決策: 情況{sit_code}({direction})")
                print(f"    功率: {power_mw_display:.1f} mW = {power_w:.4f} W")
                print(f"    流速: {flow_pct:.0f}%")
                print(f"    raw action: {action_norm}")
                if safety_net is not None:
                    raw_mw = abs(action_kw_raw) * 1e6
                    safe_mw = abs(action_kw) * 1e6
                    tag = "[CORAL clipped]" if coral_clipped else "[CORAL pass]"
                    print(f"    {tag}: raw={raw_mw:.1f}mW → safe={safe_mw:.1f}mW "
                          f"(殘差池={get_residual_count()}筆)")

                last_power_w = power_w
                last_flow_pct = flow_pct
                last_sit_code = sit_code
                last_action_kw = action_kw

                if not args.dry_run:
                    write_ts = datetime.now(TZ_UTC8)
                    power_mw_int = int(round(power_w * 1000.0))
                    flow_int = int(round(max(0.0, min(100.0, flow_pct))))
                    success = write_command_simple(
                        args.command_file, sit_code, write_ts,
                        pp, power_mw_int, flow_int,
                        load_count=get_load_groups(write_ts.time()),
                    )
                    if success:
                        last_command_write = write_ts
                        print(f"  Command.txt updated")
                    else:
                        print(f"  Command.txt write FAILED")
                else:
                    print(f"  [DRY RUN] 跳過 Command.txt 寫入")

                soc_stats_15m = soc_tracker.get_stats()
                logger.log({
                    'session_id': session_id,
                    'experiment_name': experiment_name,
                    'model_file': f'{model_file}#{model_hash}',
                    'current_mode': args.current_mode,
                    'hybrid_min_current_ma': f'{args.hybrid_min_current_ma:.1f}',
                    'coral_delta': f'{args.coral_delta:.4f}',
                    'coral_buffer': f'{args.coral_buffer:.4f}',
                    'coral_window': args.coral_window,
                    'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'step': step_count,
                    'soc': f'{soc:.4f}',
                    'soc_unclamped': f'{soc_tracker.get_soc_unclamped():.4f}',
                    'charge_mah': f'{soc_stats_15m["total_charge_mah"]:.2f}',
                    'discharge_mah': f'{soc_stats_15m["total_discharge_mah"]:.2f}',
                    'soc_coulomb': f'{soc_stats_15m["soc_coulomb"]:.4f}',
                    'soc_coulomb_unclamped': f'{soc_stats_15m["soc_coulomb_unclamped"]:.4f}',
                    'charge_wh': f'{soc_stats_15m["total_charge_wh"]:.4f}',
                    'discharge_wh': f'{soc_stats_15m["total_discharge_wh"]:.4f}',
                    'load_kw': f'{load_kw:.6f}',
                    'pv_kw': f'{pv_kw:.8f}',
                    'price': f'{price:.3f}',
                    'hour': int(state_hour),
                    'dow': int(state_dow),
                    'price_norm': f'{price_norm:.4f}',
                    'pv_support_ratio': f'{pv_support_ratio:.4f}',
                    'pv_bool': f'{pv_bool:.1f}',
                    'pv_active': f'{pv_active:.1f}',
                    'load_source': load_source,
                    'load_fallback_used': load_fallback_used,
                    'mppt_mean_mW': f'{agg["mppt_p_mean_mW"]:.2f}',
                    'mppt_max_mW': f'{agg["mppt_p_max_mW"]:.2f}',
                    'mppt_std_mW': f'{agg["mppt_p_std_mW"]:.2f}',
                    'bus_p_mean_mW': f'{agg["bus_p_mean_mW"]:.2f}',
                    'load_p_mean_mW': f'{agg["load_p_mean_mW"]:.2f}',
                    'batt_p_mean_mW': f'{agg["batt_p_mean_mW"]:.2f}',
                    'batt_v_mean': f'{agg["batt_v_mean"]:.3f}',
                    'batt_i_mean_ma': f'{agg["batt_i_mean_ma"]:.1f}',
                    'bus_v_mean': f'{agg.get("bus_v_mean", 0.0):.3f}',
                    'grid_v_mean': f'{agg.get("grid_v_mean", 0.0):.3f}',
                    'n_samples': agg['n_samples'],
                    'completeness': f'{agg["completeness"]:.3f}',
                    'action_power_kw': f'{action_kw:.8f}',
                    'action_flow_pct': f'{flow_pct:.1f}',
                    'power_mw_cmd': f'{power_mw_display:.1f}',
                    'flow_pct_cmd': f'{flow_pct:.1f}',
                    'situation_code': sit_code,
                    'load_groups': load_groups,
                    'pv_surplus_kw': f'{pv_surplus_kw:.8f}',
                    'guard_delta_mW': f'{guard_delta_mw:.2f}',
                    'guard_force_charge_low_soc': guard_force_charge_low_soc,
                    'guard_block_low_soc_discharge': guard_block_low_soc_discharge,
                    'guard_block_high_soc_charge': guard_block_high_soc_charge,
                    'guard_block_pv_active_discharge': guard_block_pv_active_discharge,
                    'guard_block_voltage_cutoff': guard_block_voltage_cutoff,
                    'warn_load_over_discharge_limit': warn_load_over_discharge_limit_flag,
                    'guard_block_invalid_discharge': guard_block_invalid_discharge,
                    'guard_block_no_pv_surplus_charge': guard_block_no_pv_surplus_charge,
                    'guard_flow_power_limited': guard_flow_power_limited,
                    'flow_charge_limit_kw': f'{flow_charge_limit_kw:.8f}',
                    'flow_discharge_limit_kw': f'{flow_discharge_limit_kw:.8f}',
                    'guard_block_discharge_intent_threshold': guard_block_discharge_intent_threshold,
                    'guard_block_firmware_override_discharge': guard_block_firmware_override_discharge,
                    'guard_block_isolated_load_bus_discharge': guard_block_isolated_load_bus_discharge,
                    'guard_block_health_lock_discharge': guard_block_health_lock_discharge,
                    'firmware_override_discharge_samples_window': firmware_override_discharge_samples_window,
                    'isolated_load_bus_samples_window': isolated_load_bus_samples_window,
                    'voltage_cutoff_active': 1 if voltage_cutoff_active else 0,
                    'voltage_cutoff_day_locked': 1 if voltage_cutoff_day_locked else 0,
                    'voltage_cutoff_day_count': voltage_cutoff_day_count,
                    'cutoff_soc_fallback_enabled': 1 if args.cutoff_soc_fallback else 0,
                    'cutoff_soc_fallback_percent': f'{cutoff_soc_fallback*100:.1f}',
                    'cutoff_soc_fallback_applied': cutoff_soc_fallback_applied_window,
                    'cutoff_soc_before': (
                        f'{cutoff_soc_before_window:.4f}'
                        if cutoff_soc_before_window is not None else ''
                    ),
                    'cutoff_soc_after': (
                        f'{cutoff_soc_after_window:.4f}'
                        if cutoff_soc_after_window is not None else ''
                    ),
                    'soh_health_enabled': 1 if args.soh_health_protection else 0,
                    'soh_health_lock_active': 1 if soh_health_lock_active else 0,
                    'soh_health_lock_reason': soh_health_lock_reason,
                    'soh_record_candidate': soh_record_candidate,
                    'soh_last_record_time': (
                        soh_last_record_time.strftime('%Y-%m-%d %H:%M:%S')
                        if soh_last_record_time is not None else ''
                    ),
                    'soh_record_reason': soh_record_reason,
                    'soh_low_voltage_samples_window': soh_low_voltage_samples_window,
                    'soh_low_voltage_streak': soh_low_voltage_streak,
                    'soh_recovery_samples': soh_recovery_samples,
                    'soh_voltage_sag_v': f'{soh_voltage_sag_v:.3f}',
                    'soh_proxy_score': f'{soh_proxy_score:.1f}',
                    'soh_prediction_enabled': 1 if args.soh_prediction else 0,
                    'soh_model_path': args.soh_model_path,
                    'soh_last_prediction_time': (
                        soh_last_prediction_time.strftime('%Y-%m-%d %H:%M:%S')
                        if soh_last_prediction_time is not None else ''
                    ),
                    'soh_last_value': (
                        f'{soh_last_value:.6f}' if soh_last_value is not None else ''
                    ),
                    'soh_prediction_status': soh_prediction_status,
                    'soh_prediction_method': soh_prediction_method,
                    'soh_effective_capacity_mah': f'{soh_effective_capacity_mah:.2f}',
                    'soh_use_for_capacity': 1 if args.soh_use_for_capacity else 0,
                    'coral_active': 1 if safety_net is not None else 0,
                    'coral_clipped': 1 if coral_clipped else 0,
                    'coral_delta_mW': f'{coral_delta_kw*1e6:.2f}',
                    'coral_interventions': coral_stats['interventions'],
                    'coral_residual_count': get_residual_count(),
                    'action_raw_kw': f'{action_kw_raw:.8f}',
                })
                cutoff_soc_fallback_applied_window = 0
                cutoff_soc_before_window = None
                cutoff_soc_after_window = None
                firmware_override_discharge_samples_window = 0
                isolated_load_bus_samples_window = 0
                soh_low_voltage_samples_window = 0

                curr_aligned_min = (now.minute // args.window_min) * args.window_min
                next_start = now.replace(minute=curr_aligned_min, second=0, microsecond=0)
                if next_start <= buffer.window_start:
                    next_min = curr_aligned_min + args.window_min
                    if next_min >= 60:
                        next_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    else:
                        next_start = now.replace(minute=next_min, second=0, microsecond=0)
                buffer.reset(new_start=next_start)
                print(f"\n  下一個窗格: {next_start.strftime('%H:%M:%S')}")
                print("-" * 60)

            elif not args.dry_run:
                now_ts = datetime.now(TZ_UTC8)
                should_update = (
                    last_command_write is None or
                    (now_ts - last_command_write).total_seconds() >= 1.0
                )
                if should_update:
                    last_power_mw_int = int(round(last_power_w * 1000.0))
                    last_flow_int = int(round(max(0.0, min(100.0, last_flow_pct))))
                    write_command_simple(
                        args.command_file, last_sit_code, now_ts,
                        pp, last_power_mw_int, last_flow_int,
                        load_count=get_load_groups(now_ts.time()),
                    )
                    last_command_write = now_ts

            elapsed = time.time() - loop_start
            sleep_time = max(0.5, args.poll_sec - elapsed)
            if buffer.window_start is not None:
                remaining = (buffer.window_start.timestamp() + window_sec) - time.time()
                if 0 < remaining < args.poll_sec:
                    sleep_time = min(sleep_time, max(0.5, remaining))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"  控制迴圈已停止 (Ctrl+C)")
        print(f"  總步數: {step_count}")
        print(f"  Final SoC: {soc_tracker.get_soc()*100:.1f}%")
        if voltage_cutoff_count > 0:
            print(f"  Voltage cutoff count: {voltage_cutoff_count}")
            print(f"  Cutoff status: {'active' if voltage_cutoff_active else 'recovered'}")
        if safety_net is not None and coral_stats['total_steps'] > 0:
            rate = coral_stats['interventions'] / coral_stats['total_steps'] * 100
            print(f"  CORAL interventions: {coral_stats['interventions']}/{coral_stats['total_steps']} ({rate:.1f}%)")
            if coral_stats['interventions'] > 0:
                avg_delta = coral_stats['total_delta_kw'] / coral_stats['interventions']
                print(f"  CORAL average correction: {avg_delta*1e6:.1f} mW")
        print(f"  Decision log: {logger.path}")
        print(f"  Raw log: {raw_logger.path}")
        print(f"{'='*60}")

        raw_logger.close()

        if not args.dry_run:
            try:
                now_ts = datetime.now(TZ_UTC8)
                write_standby_rest_command(
                    args.command_file, now_ts, pp,
                    load_count=get_load_groups(now_ts.time()),
                )
                print("  Standby/rest command written")
            except Exception:
                pass

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()

        if not args.dry_run:
            try:
                now_ts = datetime.now(TZ_UTC8)
                write_standby_rest_command(
                    args.command_file, now_ts, pp,
                    load_count=get_load_groups(now_ts.time()),
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()


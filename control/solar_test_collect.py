#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Solar-Only Data Collection Script (Scenario 3 - Rest / Pre-measure Flow)
=========================================================
Purpose:
  - Collect MPPT / solar data without using the battery
  - Battery power = 0, rest flow = 0%
  - Optional pre-measure uses flow = 50% before reading voltage
  - Load groups = 4 (constant, via load_count)
  - Scenario 3 with zero battery power keeps the command on the physical PP
  - Logs all MPPT readings to CSV for training data

Command.txt output format:
  3                          <-- Scenario 3 (rest / pre-measure mode)
  YYYYMMDDhhmmss,4           <-- timestamp, load_count=4
  01,0,0,                    <-- PP=01, power=0mW, rest flow=0%

Usage (on deployment machine):
  1. Place this script next to Data.txt / Command.txt
     (or specify paths via --data-file / --command-file)
  2. Run: python solar_test_collect.py
  3. Press Ctrl+C to stop. CSV log saved to ./solar_log_YYYYMMDD_HHMMSS.csv

No dependencies beyond Python standard library + (optional) numpy for stats.
"""

import os
import sys
import io
import csv
import time
import argparse
from datetime import datetime, timezone, timedelta

# Optional: numpy for better stats
try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

TZ_UTC8 = timezone(timedelta(hours=8))
LOAD_PER_GROUP_W = 0.1
FLOW_REST_PCT = 0
FLOW_PRE_MEASURE_PCT = 50
FLOW_IDLE_PCT = FLOW_REST_PCT  # Backward-compatible public alias.
DEFAULT_STANDBY_SCENARIO = 3


# ====================================================================
# Minimal Data.txt parser (standalone, no external dependencies)
# ====================================================================
def parse_data_txt(path):
    """
    Parse vendor Data.txt format (supports old/new formats).
    
    Newest format (2026/03/20+):
        Line 1: YYYYMMDDHHmmSS
        Line 2: SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,BusV,BusI,BusP,
        Line 3: LoadV,LoadI,LoadP,GridV,GridI,GridP,        (6 fields)
        Line 4+: ID,SOC,BV,ChargeV,ChargeI,Temp,Speed,      (7 fields)
    
    Previous format (2026/03 early):
        Line 3: LoadV,LoadI,LoadP,                            (3 fields)
        Battery: ID,SOC,BV,BI,Temp,Speed,                    (6 fields)
    
    Old format (2025/12):
        Line 2: SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,  (6 fields)
        Line 3+: ID,SOC,BV,BI,Temp,Speed,
    
    Returns: (timestamp_str, mppt_dict, mppt_bus_dict, load_dict, grid_dict, battery_list)
    
    mppt_dict keys:     solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw
    mppt_bus_dict keys: bus_v, bus_i_ma, bus_p_mw  (or None if old format)
    load_dict keys:     load_v, load_i_ma, load_p_mw  (or None if old format)
    grid_dict keys:     grid_v, grid_i_ma, grid_p_mw  (or None if not newest format)
    battery_list: [{pp, soc_pct, volt_v, charge_v, curr_ma, temp_c, speed_pct}, ...]
    """
    if not os.path.exists(path):
        return None, None, None, None, None, []
    try:
        with io.open(path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return None, None, None, None, None, []

    if not lines:
        return None, None, None, None, None, []

    # Line 1: timestamp (14 digits, optionally followed by ,load_count)
    ts_str = None
    idx = 0
    first = lines[0]
    ts_part = first.split(',')[0].strip()
    if len(ts_part) >= 14 and ts_part[:14].isdigit():
        ts_str = ts_part[:14]
        idx = 1

    # Line 2: MPPT data (6 or 9 numeric fields, first field is NOT a small battery ID)
    mppt = None
    mppt_bus = None
    if idx < len(lines):
        parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
        if len(parts) >= 6:
            is_battery_id = parts[0].isdigit() and 1 <= int(parts[0]) <= 10
            if not is_battery_id:
                try:
                    vals = [float(p) for p in parts]
                    mppt = {
                        'solar_v': vals[0] / 100.0,
                        'solar_i_ma': vals[1],
                        'solar_p_mw': vals[2],
                        'mppt_v': vals[3] / 100.0,
                        'mppt_i_ma': vals[4],
                        'mppt_p_mw': vals[5],
                    }
                    # New format: 9+ fields → MPPT-Bus
                    if len(vals) >= 9:
                        mppt_bus = {
                            'bus_v': vals[6] / 100.0,
                            'bus_i_ma': vals[7],
                            'bus_p_mw': vals[8],
                        }
                    idx += 1
                except (ValueError, IndexError):
                    idx += 1

    # Line 3: Load [+ Grid] data. Some firmware versions emit 6-field MPPT
    # without Bus fields but still provide a measured load line.
    load = None
    grid = None
    if idx < len(lines):
        parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
        if len(parts) >= 3:
            is_battery = (parts[0].isdigit() and 1 <= int(parts[0]) <= 10
                          and 6 <= len(parts) <= 7)
            if not is_battery:
                try:
                    vals = [float(p) for p in parts]
                    load = {
                        'load_v': vals[0] / 100.0,    # 0.01V
                        'load_i_ma': vals[1],           # mA
                        'load_p_mw': vals[2],           # mW
                    }
                    # Newest format: 6+ fields → Grid data
                    if len(vals) >= 6:
                        grid = {
                            'grid_v': vals[3] / 100.0,
                            'grid_i_ma': vals[4],
                            'grid_p_mw': vals[5],
                        }
                    idx += 1
                except (ValueError, IndexError):
                    idx += 1

    # Remaining lines: battery data (6 or 7 fields)
    batteries = []
    while idx < len(lines):
        parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
        idx += 1
        if len(parts) < 6:
            continue
        try:
            if len(parts) >= 7:
                # Newest format: ID,SOC,BV,ChargeV,ChargeI,Temp,Speed
                batteries.append({
                    'pp': parts[0].zfill(2),
                    'soc_pct': float(parts[1]) / 10.0,
                    'volt_v': float(parts[2]) / 100.0,
                    'charge_v': float(parts[3]) / 100.0,
                    'curr_ma': float(parts[4]),
                    'temp_c': float(parts[5]) / 10.0,
                    'speed_pct': float(parts[6]) / 10.0,
                })
            else:
                # Old format: ID,SOC,BV,BI,Temp,Speed
                batteries.append({
                    'pp': parts[0].zfill(2),
                    'soc_pct': float(parts[1]) / 10.0,
                    'volt_v': float(parts[2]) / 100.0,
                    'charge_v': 0.0,
                    'curr_ma': float(parts[3]),
                    'temp_c': float(parts[4]) / 10.0,
                    'speed_pct': float(parts[5]) / 10.0,
                })
        except (ValueError, IndexError):
            continue

    return ts_str, mppt, mppt_bus, load, grid, batteries


# ====================================================================
# Minimal Command.txt writer (standalone)
# ====================================================================
def command_pp_for_action(pp, power_mw):
    """Keep the battery PP even at zero power so flow commands still target it."""
    return f"{int(pp):02d}" if str(pp).isdigit() else str(pp)


def write_command_txt(path, scenario, timestamp_dt, load_count, pp, power_mw, flow_pct):
    """
    Write Command.txt in vendor format (direct write, no temp file):
      {scenario}
      YYYYMMDDhhmmss,{load_count}
      PP,power_mW,flow_pct,
    """
    ts_str = timestamp_dt.strftime('%Y%m%d%H%M%S')
    command_pp = command_pp_for_action(pp, power_mw)
    content = (
        f"{scenario}\n"
        f"{ts_str},{load_count}\n"
        f"{command_pp},{int(power_mw)},{int(flow_pct)},\n"
    )
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except (IOError, OSError, PermissionError):
        return False


def write_rest_command(path, scenario, timestamp_dt, load_count, pp):
    """Write normal rest command: mode 3, physical PP, zero power, zero flow."""
    return write_command_txt(
        path, scenario, timestamp_dt, load_count, pp, 0, FLOW_REST_PCT
    )


def write_pre_measure_command(path, scenario, timestamp_dt, load_count, pp):
    """Write pre-measure command: mode 3, physical PP, zero power, 50% flow."""
    return write_command_txt(
        path, scenario, timestamp_dt, load_count, pp, 0, FLOW_PRE_MEASURE_PCT
    )


# ====================================================================
# Main loop
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Solar-only data collection (Scenario 3 - rest / optional pre-measure flow)')
    parser.add_argument('--data-file', type=str, default='./Data.txt',
                        help='Data.txt path (vendor writes)')
    parser.add_argument('--command-file', type=str, default='./Command.txt',
                        help='Command.txt path (we write)')
    parser.add_argument('--battery-pp', type=str, default='01',
                        help='Battery PP ID (default: 01)')
    parser.add_argument('--load-count', type=int, default=4,
                        help='Load groups (0-4, default: 4)')
    parser.add_argument('--poll-sec', type=float, default=10.0,
                        help='Read Data.txt + write CSV interval (sec, default: 10)')
    parser.add_argument('--pre-measure-sec', type=float, default=0.0,
                        help='Optional mode 3 / 50%% flow voltage recovery before each read (sec, default: 0)')
    parser.add_argument('--log-dir', type=str, default='.',
                        help='CSV log output directory')
    parser.add_argument('--scenario', type=int, default=DEFAULT_STANDBY_SCENARIO, choices=[1, 2, 3, 4],
                        help='Scenario code (default: 3=rest/pre-measure with zero battery power; 4=explicit motor stop)')
    args = parser.parse_args()

    CSV_HEADER = [
        'timestamp',
        'battery_id',
        'soc_percent',         # SoC (%)
        'voltage_v',
        'charge_voltage_v',
        'current_ma',
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
        'load_count',
        'load_power_w',
        'data_txt_ts',
        'elapsed_sec',
    ]

    os.makedirs(args.log_dir, exist_ok=True)

    def open_csv_for_date(date_str):
        """Documentation for this public API is provided in English."""
        path = os.path.join(args.log_dir, f'collected_data_v2_{date_str}.csv')
        if os.path.exists(path):
            f = open(path, 'a', newline='', encoding='utf-8-sig')
            w = csv.DictWriter(f, fieldnames=CSV_HEADER)
            print(f'  [CSV] 接續寫入: {path}')
        else:
            f = open(path, 'w', newline='', encoding='utf-8-sig')
            w = csv.DictWriter(f, fieldnames=CSV_HEADER)
            w.writeheader()
            print(f'  [CSV] 建立新檔: {path}')
        return f, w, path

    current_date_str = datetime.now(TZ_UTC8).strftime('%Y-%m-%d')
    csv_file, csv_writer, csv_path = open_csv_for_date(current_date_str)

    pp = f'{int(args.battery_pp):02d}'
    total_reads = 0
    start_time = time.time()

    print('=' * 70)
    print('  P302 Solar-Only Data Collection')
    print('=' * 70)
    print(f'  Scenario  : {args.scenario} ({"Rest/pre-measure" if args.scenario == 3 else "S" + str(args.scenario)})')
    print(f'  Load      : {args.load_count} groups ({args.load_count * 100}mW)')
    print(f'  Rest      : PP={pp}, Power=0mW, Flow={FLOW_REST_PCT}%')
    print(f'  Pre-meas. : Flow={FLOW_PRE_MEASURE_PCT}% for {args.pre_measure_sec:.1f}s when enabled')
    print(f'  Data.txt  : {os.path.abspath(args.data_file)}')
    print(f'  Command   : {os.path.abspath(args.command_file)}')
    print(f'  CSV Log   : {os.path.abspath(csv_path)}')
    print(f'  Poll      : {args.poll_sec}s')
    print('=' * 70)
    print()
    print('  Press Ctrl+C to stop.')
    print()
    print('-' * 70)

    # Running stats
    mppt_values = []

    try:
        while True:
            loop_start = time.time()
            now = datetime.now(TZ_UTC8)

            today_str = now.strftime('%Y-%m-%d')
            if today_str != current_date_str:
                csv_file.close()
                current_date_str = today_str
                csv_file, csv_writer, csv_path = open_csv_for_date(current_date_str)

            # 1) Normal rest is zero flow. If requested, briefly pre-flow before reading voltage.
            if args.pre_measure_sec > 0:
                ok = write_pre_measure_command(
                    args.command_file, args.scenario, now, args.load_count, pp
                )
                time.sleep(float(args.pre_measure_sec))
            else:
                ok = write_rest_command(
                    args.command_file, args.scenario, now, args.load_count, pp
                )

            # 2) Read Data.txt (supports old/new formats incl. grid + charge_v)
            ts_str, mppt, mppt_bus, load, grid, batteries = parse_data_txt(args.data_file)
            elapsed = time.time() - start_time

            if load is not None:
                load_power_w = load['load_p_mw'] / 1000.0  # mW → W
            else:
                load_power_w = args.load_count * LOAD_PER_GROUP_W

            row = {
                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                'battery_id': pp,
                'soc_percent': '',
                'voltage_v': '',
                'charge_voltage_v': '',
                'current_ma': '',
                'temp_c': '',
                'speed_percent': '',
                'solar_v': '',
                'solar_i_ma': '',
                'solar_p_mw': '',
                'mppt_v': '',
                'mppt_i_ma': '',
                'mppt_p_mw': '',
                'bus_v': '',
                'bus_i_ma': '',
                'bus_p_mw': '',
                'load_v': '',
                'load_i_ma': '',
                'load_p_mw': '',
                'grid_v': '',
                'grid_i_ma': '',
                'grid_p_mw': '',
                'load_count': str(args.load_count),
                'load_power_w': f'{load_power_w:.1f}',
                'data_txt_ts': ts_str or '',
                'elapsed_sec': f'{elapsed:.1f}',
            }

            if mppt:
                row.update({
                    'solar_v': f'{mppt["solar_v"]:.2f}',
                    'solar_i_ma': f'{mppt["solar_i_ma"]:.0f}',
                    'solar_p_mw': f'{mppt["solar_p_mw"]:.0f}',
                    'mppt_v': f'{mppt["mppt_v"]:.2f}',
                    'mppt_i_ma': f'{mppt["mppt_i_ma"]:.0f}',
                    'mppt_p_mw': f'{mppt["mppt_p_mw"]:.0f}',
                })
                mppt_values.append(mppt['mppt_p_mw'])
                total_reads += 1

            if mppt_bus:
                row.update({
                    'bus_v': f'{mppt_bus["bus_v"]:.2f}',
                    'bus_i_ma': f'{mppt_bus["bus_i_ma"]:.0f}',
                    'bus_p_mw': f'{mppt_bus["bus_p_mw"]:.0f}',
                })

            if load:
                row.update({
                    'load_v': f'{load["load_v"]:.2f}',
                    'load_i_ma': f'{load["load_i_ma"]:.0f}',
                    'load_p_mw': f'{load["load_p_mw"]:.0f}',
                })

            if grid:
                row.update({
                    'grid_v': f'{grid["grid_v"]:.2f}',
                    'grid_i_ma': f'{grid["grid_i_ma"]:.0f}',
                    'grid_p_mw': f'{grid["grid_p_mw"]:.0f}',
                })

            # Battery info (if available)
            batt = None
            for b in batteries:
                if b['pp'] == pp or b['pp'] == args.battery_pp:
                    batt = b
                    break
            if batt:
                row.update({
                    'soc_percent': f'{batt["soc_pct"]:.2f}',
                    'voltage_v': f'{batt["volt_v"]:.2f}',
                    'charge_voltage_v': f'{batt["charge_v"]:.2f}',
                    'current_ma': f'{batt["curr_ma"]:.0f}',
                    'temp_c': f'{batt["temp_c"]:.1f}',
                    'speed_percent': f'{batt["speed_pct"]:.1f}',
                })

            csv_writer.writerow(row)
            csv_file.flush()

            # Console output (every read)
            ts_display = now.strftime('%H:%M:%S')
            if mppt:
                mppt_w = mppt['mppt_p_mw'] / 1000.0
                solar_w = mppt['solar_p_mw'] / 1000.0
                
                extra = ''
                if mppt_bus:
                    bus_w = mppt_bus['bus_p_mw'] / 1000.0
                    extra += f'  Bus={bus_w:.3f}W'
                if load:
                    load_w = load['load_p_mw'] / 1000.0
                    extra += f'  Load={load_w:.3f}W'
                if grid:
                    grid_w = grid['grid_p_mw'] / 1000.0
                    extra += f'  Grid={grid_w:.3f}W'
                
                batt_info = ''
                if batt:
                    cv_tag = f' CV={batt["charge_v"]:.2f}V' if batt.get("charge_v", 0) > 0 else ''
                    batt_info = (f'  Batt: {batt["volt_v"]:.2f}V{cv_tag} '
                                 f'{batt["curr_ma"]:.0f}mA '
                                 f'SoC={batt["soc_pct"]:.1f}% '
                                 f'T={batt["temp_c"]:.1f}C')

                # Stats
                stats = ''
                if len(mppt_values) > 5:
                    recent = mppt_values[-30:]
                    avg = sum(recent) / len(recent)
                    if HAS_NP:
                        std = float(np.std(recent))
                        stats = f'  [avg={avg:.0f} std={std:.0f}]'
                    else:
                        stats = f'  [avg={avg:.0f}]'

                cmd_ok = 'OK' if ok else 'FAIL'
                print(f'  [{ts_display}] #{total_reads:5d}  '
                      f'MPPT={mppt["mppt_p_mw"]:6.0f}mW ({mppt_w:.3f}W)  '
                      f'Solar={solar_w:.3f}W'
                      f'{extra}'
                      f'{batt_info}{stats}  [Cmd:{cmd_ok}]')
            else:
                print(f'  [{ts_display}] No MPPT data (waiting for Data.txt...)'
                      f'  [Cmd:{"OK" if ok else "FAIL"}]')

            # Summary every 5 minutes
            if total_reads > 0 and total_reads % 30 == 0:
                mins = elapsed / 60.0
                rate = total_reads / mins if mins > 0 else 0
                print(f'\n  --- {mins:.1f} min | {total_reads} reads '
                      f'({rate:.1f}/min) | CSV: {csv_path} ---\n')

            # Sleep
            loop_elapsed = time.time() - loop_start
            sleep_time = max(0.1, args.poll_sec - loop_elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f'\n\n{"=" * 70}')
        print(f'  Data collection stopped (Ctrl+C)')
        print(f'  Duration  : {elapsed/60:.1f} minutes')
        print(f'  Readings  : {total_reads}')
        if mppt_values:
            avg_mw = sum(mppt_values) / len(mppt_values)
            max_mw = max(mppt_values)
            print(f'  MPPT avg  : {avg_mw:.0f} mW ({avg_mw/1000:.3f} W)')
            print(f'  MPPT max  : {max_mw:.0f} mW ({max_mw/1000:.3f} W)')
        print(f'  CSV saved : {os.path.abspath(csv_path)}')
        print(f'{"=" * 70}')

        # Write zero-flow rest command one last time.
        now = datetime.now(TZ_UTC8)
        write_rest_command(
            args.command_file, DEFAULT_STANDBY_SCENARIO, now,
            args.load_count, pp,
        )

    finally:
        csv_file.close()


if __name__ == '__main__':
    main()

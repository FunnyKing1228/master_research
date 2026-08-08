"""
build_training_dataset.py
=========================




"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, time as dtime, date as ddate

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT  = SCRIPT_DIR.parent
RAW_DIR    = DATA_ROOT / 'raw'
OUT_DIR    = DATA_ROOT / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
BATTERY_CAPACITY_MAH  = 600.0      # 50mA × 12hr
BATTERY_CHARGE_V      = 8.5        # V
BATTERY_DISCHARGE_V   = 5.6        # V
BATTERY_CHARGE_I_MA   = 50.0
BATTERY_CAPACITY_WH   = 3.36       # 0.280W × 12hr
BATTERY_CAPACITY_KWH  = BATTERY_CAPACITY_WH / 1000  # 0.00336 kWh

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
LOAD_PER_GROUP_W = 0.1    # W (100mW per group, measured avg)
LOAD_VOLTAGE     = 5.0    # V
MAX_GROUPS       = 4

LOAD_SCHEDULE = [
    (dtime( 0, 0), 0),
    (dtime( 6, 0), 1),
    (dtime( 7, 0), 2),
    (dtime( 8, 0), 3),
    (dtime( 9, 0), 4),
    (dtime(12, 0), 3),
    (dtime(13, 0), 4),
    (dtime(17, 0), 3),
    (dtime(18, 0), 2),
    (dtime(20, 0), 1),
    (dtime(22, 0), 0),
]


def get_load_groups(t: dtime) -> int:
    """Documentation for this public API is provided in English."""
    result = 0
    for sched_t, n in LOAD_SCHEDULE:
        if t >= sched_t:
            result = n
    return result


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'mppt_p_mw' in df.columns:
        for old, new in [('mppt_p_mw', 'mppt_p'), ('solar_p_mw', 'solar_p')]:
            if old in df.columns:
                df[new] = df[old].astype(float) / 1000.0
        for old, new in [('current_ma', 'current_a'), ('solar_i_ma', 'solar_i'),
                         ('mppt_i_ma', 'mppt_i')]:
            if old in df.columns:
                df[new] = df[old].astype(float) / 1000.0
    for col in ['mppt_p', 'mppt_v', 'mppt_i', 'solar_p', 'solar_v', 'solar_i',
                'current_a', 'voltage_v', 'soc_percent', 'temp_c']:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
def load_and_select(raw_dir: Path, force_dates=None) -> pd.DataFrame:
    """

    """
    files = sorted(raw_dir.glob('collected_data_*.csv'))
    daily = {}
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=['timestamp'])
            df = normalize_columns(df)
            date = df['timestamp'].dt.date.iloc[0]
            diffs = df['timestamp'].diff().dt.total_seconds().dropna()
            dt_med = diffs.median() if len(diffs) > 0 else 11.0
            expected = 24 * 3600 / dt_med
            completeness = min(len(df) / expected, 1.0)
            dur_h = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            mask_d = (df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour <= 16)
            mppt_day_mean = df.loc[mask_d, 'mppt_p'].mean() if mask_d.sum() > 0 else 0.0
            mppt_day_max  = df.loc[mask_d, 'mppt_p'].max()  if mask_d.sum() > 0 else 0.0
            daily[date] = {
                'file': f, 'df': df, 'n_rows': len(df),
                'completeness': completeness, 'duration_h': dur_h,
                'mppt_day_mean_W': mppt_day_mean,
                'mppt_day_max_W': mppt_day_max,
            }
            mppt_ok = 'OK' if mppt_day_mean > 0.01 else 'low'
            print(f"  {f.name:45s}  {len(df):>5d} rows  {dur_h:5.1f}h  "
                  f"完整度 {completeness*100:5.1f}%  MPPT:{mppt_ok}({mppt_day_mean*1000:.0f}mW)")
        except Exception as e:
            print(f"  ERR {f.name}: {e}")

    if force_dates:
        selected = sorted([d for d in force_dates if d in daily])
        missing = [d for d in force_dates if d not in daily]
        if missing:
            print(f"\n  [警告] 指定日期但找不到資料：{missing}")
        print(f"\n  使用指定日期 ({len(selected)} 天)：")
    else:
        from datetime import timedelta
        good_dates = sorted([d for d, v in daily.items()
                             if v['completeness'] >= 0.90 and v['mppt_day_mean_W'] > 0.01])
        print(f"\n  品質合格日期（完整度≥90% & MPPT>10mW）：{good_dates}")

        if good_dates:
            best_run = [good_dates[0]]
            current_run = [good_dates[0]]
            for i in range(1, len(good_dates)):
                if (good_dates[i] - good_dates[i-1]).days == 1:
                    current_run.append(good_dates[i])
                else:
                    if len(current_run) > len(best_run):
                        best_run = current_run
                    current_run = [good_dates[i]]
            if len(current_run) > len(best_run):
                best_run = current_run
            selected = best_run
        else:
            ranked = sorted(daily.items(), key=lambda x: x[1]['completeness'], reverse=True)
            selected = sorted([d for d, _ in ranked[:6]])

    print(f"\n  選定 {len(selected)} 天（最長連續段）：")
    for d in selected:
        v = daily[d]
        print(f"    {d}  完整度 {v['completeness']*100:5.1f}%  "
              f"MPPT day avg {v['mppt_day_mean_W']*1000:.0f}mW  max {v['mppt_day_max_W']*1000:.0f}mW")

    frames = [daily[d]['df'] for d in selected]
    df_all = pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f"  合計 {len(df_all):,} 筆")
    return df_all, selected


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
def simulate_soc(df: pd.DataFrame) -> np.ndarray:
    """


    """
    cap_wh = BATTERY_CAPACITY_WH  # 3.36 Wh
    max_charge_w = BATTERY_CHARGE_I_MA * BATTERY_CHARGE_V / 1000  # 0.425 W
    efficiency = 0.85

    soc = np.zeros(len(df), dtype=float)
    soc[0] = 0.5

    timestamps = df['timestamp'].values
    mppt_p = df['mppt_p'].fillna(0.0).values

    for i in range(1, len(df)):
        dt_sec = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 's')
        if dt_sec <= 0 or dt_sec > 600:
            soc[i] = soc[i-1]
            continue

        dt_h = dt_sec / 3600.0

        p_charge_w = min(float(mppt_p[i]), max_charge_w)
        p_charge_w = max(p_charge_w, 0.0)

        delta_wh = p_charge_w * dt_h * efficiency
        delta_soc = delta_wh / cap_wh if cap_wh > 0 else 0.0

        soc[i] = float(np.clip(soc[i-1] + delta_soc, 0.0, 1.0))

    return soc


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
WINDOW_MIN = 15

def aggregate_15min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().set_index('timestamp').sort_index()
    rule = f'{WINDOW_MIN}min'

    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    dt_sec = float(diffs.median()) if len(diffs) > 0 else 11.0
    dt_h   = dt_sec / 3600.0

    agg = pd.DataFrame()

    # MPPT（W）
    agg['mppt_p_mean_W']  = df['mppt_p'].resample(rule).mean()
    agg['mppt_p_std_W']   = df['mppt_p'].resample(rule).std().fillna(0.0)
    agg['mppt_p_max_W']   = df['mppt_p'].resample(rule).max()
    agg['mppt_v_mean_V']  = df['mppt_v'].resample(rule).mean()
    agg['mppt_i_mean_A']  = df['mppt_i'].resample(rule).mean()

    # Solar
    agg['solar_p_mean_W'] = df['solar_p'].resample(rule).mean()

    if 'temp_c' in df.columns and df['temp_c'].notna().any():
        agg['temp_mean_c'] = df['temp_c'].resample(rule).mean()

    agg['soc_mean']       = df['soc_estimated'].resample(rule).mean()
    agg['soc_end']        = df['soc_estimated'].resample(rule).last()

    agg['load_groups']    = df['load_groups'].resample(rule).mean()
    agg['load_W']         = df['load_W'].resample(rule).mean()
    agg['load_std_W']     = df['load_W'].resample(rule).std().fillna(0.0)
    agg['load_max_W']     = df['load_W'].resample(rule).max()

    counts = df['mppt_p'].resample(rule).count()
    agg['n_samples'] = counts
    expected = WINDOW_MIN * 60 / max(dt_sec, 1.0)
    agg['completeness'] = (counts / expected).clip(0.0, 1.0).round(3)
    agg['has_gap'] = (agg['completeness'] < 0.5).astype(int)

    agg['energy_mppt_Wh'] = agg['mppt_p_mean_W'] * counts * dt_h
    agg['energy_load_Wh'] = agg['load_W'] * WINDOW_MIN / 60.0  # W × h

    agg = agg.reset_index()

    agg['hour']        = agg['timestamp'].dt.hour
    agg['minute']      = agg['timestamp'].dt.minute
    agg['day_of_week'] = agg['timestamp'].dt.dayofweek
    agg['date']        = agg['timestamp'].dt.date

    agg['Solar']       = agg['mppt_p_mean_W'] / 1000.0  # W → kW
    agg['Consumption'] = agg['load_W'] / 1000.0          # W → kW

    def get_tou_price(hour, dow):
        if dow >= 5:
            return 2.06
        if 0 <= hour < 9:
            return 2.06
        elif 9 <= hour < 16:
            return 4.69
        elif 16 <= hour < 22:
            return 7.13
        else:
            return 4.69
    agg['price'] = [get_tou_price(h, d) for h, d in zip(agg['hour'], agg['day_of_week'])]

    return agg


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dates', nargs='+', help='指定日期 e.g. 2026-03-05 2026-03-06 ...')
    args = parser.parse_args()

    force_dates = None
    if args.dates:
        force_dates = [ddate.fromisoformat(d) for d in args.dates]

    print("=" * 60)
    print("  Training Dataset Builder")
    print("  電池：600 mAh / 50 mA / 8.5V(充) / 5.6V(放)")
    print("  負載：4 組 x 100mW")
    print("=" * 60)

    print("\n[1/4] 載入資料 & 選取日期...")
    df, selected_dates = load_and_select(RAW_DIR, force_dates=force_dates)

    print("\n[2/4] 注入負載模式...")
    df['load_groups'] = df['timestamp'].apply(
        lambda ts: get_load_groups(ts.time())
    )
    df['load_W'] = df['load_groups'] * LOAD_PER_GROUP_W
    load_summary = df.groupby('load_groups')['load_W'].count()
    print("  負載分佈（採樣數）：")
    for n, cnt in load_summary.items():
        print(f"    {int(n)} 組 ({int(n)*100}mW)：{cnt:,} 筆")

    print("\n[3/4] SoC 模擬（MPPT→電池被動充電）...")
    df['soc_estimated'] = simulate_soc(df)
    soc = df['soc_estimated']
    print(f"  SoC 範圍：{soc.min():.4f} ~ {soc.max():.4f}")
    print(f"  SoC 平均：{soc.mean():.4f}")
    print(f"  SoC 末端：{soc.iloc[-1]:.4f}")

    print("\n[4/4] 15 分鐘聚合...")
    df_agg = aggregate_15min(df)

    n_before = len(df_agg)
    df_agg = df_agg[df_agg['completeness'] > 0.3].reset_index(drop=True)
    n_after = len(df_agg)
    print(f"  聚合窗格：{n_before} → {n_after}（移除 {n_before - n_after} 個低品質窗格）")

    out_path = OUT_DIR / 'training_7day_15min.csv'
    df_agg.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  輸出 → {out_path}")

    raw_out = OUT_DIR / 'training_7day_raw.csv'
    keep = ['timestamp', 'mppt_p', 'mppt_v', 'mppt_i', 'solar_p',
            'load_groups', 'load_W', 'soc_estimated']
    df[[c for c in keep if c in df.columns]].to_csv(raw_out, index=False, encoding='utf-8-sig')
    print(f"  原始資料 → {raw_out}")

    print("\n" + "=" * 60)
    print("  Dataset 統計摘要")
    print("=" * 60)
    print(f"  選定日期        ：{selected_dates[0]} ~ {selected_dates[-1]}")
    print(f"  原始採樣數      ：{len(df):,}")
    print(f"  15-min 窗格數   ：{len(df_agg)}")
    print(f"  MPPT 發電總量   ：{df_agg['energy_mppt_Wh'].sum():.3f} Wh")
    print(f"  負載耗電總量    ：{df_agg['energy_load_Wh'].sum():.3f} Wh")
    print(f"  MPPT 峰值功率   ：{df_agg['mppt_p_max_W'].max():.3f} W")
    print(f"  平均完整度      ：{df_agg['completeness'].mean()*100:.1f}%")

    print(f"\n  電池參數（供 RL env 使用）：")
    print(f"    battery_capacity_kwh  = {BATTERY_CAPACITY_KWH:.6f}")
    print(f"    battery_power_kw      = {BATTERY_CHARGE_I_MA * BATTERY_CHARGE_V / 1e6:.6f}  (充電功率)")
    print(f"    time_step             = 0.25  (15 分鐘)")

    print("\n  日別統計：")
    for d, grp in df_agg.groupby('date'):
        mppt_wh = grp['energy_mppt_Wh'].sum()
        load_wh = grp['energy_load_Wh'].sum()
        soc_s   = grp['soc_mean'].iloc[0] if len(grp) > 0 else 0
        soc_e   = grp['soc_end'].iloc[-1] if len(grp) > 0 else 0
        print(f"    {d}  MPPT {mppt_wh:6.3f} Wh  Load {load_wh:6.1f} Wh  "
              f"SoC {soc_s:.3f}→{soc_e:.3f}  windows {len(grp)}")

    print("\n  繪圖...")
    plot_training_summary(df, df_agg)


def plot_training_summary(df_raw: pd.DataFrame, df_agg: pd.DataFrame):
    """Documentation for this public API is provided in English."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig_dir = OUT_DIR / 'analysis' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    dates = df_raw.index if isinstance(df_raw.index, pd.DatetimeIndex) else pd.to_datetime(df_raw['timestamp'])
    fig.suptitle('Training Dataset — MPPT (real) + Load (controlled) + SoC (simulated)',
                 fontsize=14, fontweight='bold')

    ax0, ax1, ax2 = axes

    ax0.plot(df_raw['timestamp'], df_raw['mppt_p'] * 1000, color='#F4A460', linewidth=0.5)
    ax0.set_ylabel('MPPT (mW)')
    ax0.grid(alpha=0.3)
    ax0.set_title('MPPT Power (real solar data)')

    ax1.step(df_raw['timestamp'], df_raw['load_groups'], color='#DC143C', linewidth=0.8, where='post')
    ax1.set_ylabel('Load Groups')
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.grid(alpha=0.3)
    ax1.set_title('Load Pattern (controlled, 0-4 groups x 100mW)')

    ax2.plot(df_raw['timestamp'], df_raw['soc_estimated'] * 100, color='#4682B4', linewidth=0.8)
    ax2.set_ylabel('SoC (%)')
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.3)
    ax2.set_title('SoC (simulated passive charging, 600 mAh battery)')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    out = fig_dir / 'training_overview.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  training_overview.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    for d, grp in df_raw.groupby(df_raw['timestamp'].dt.date):
        hours = (grp['timestamp'] - grp['timestamp'].dt.normalize()).dt.total_seconds() / 3600
        ax.plot(hours, grp['soc_estimated'] * 100, linewidth=0.8, label=str(d), alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('SoC (%)')
    ax.set_title('SoC Daily Cycles (Coulomb Counting)', fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = fig_dir / 'soc_daily_cycles.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  soc_daily_cycles.png")

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(df_agg['timestamp'], df_agg['mppt_p_mean_W'] * 1000, width=0.008,
            color='#DAA520', alpha=0.8, label='MPPT (mW)')
    ax1.set_ylabel('MPPT Power (mW)')
    ax2 = ax1.twinx()
    ax2.step(df_agg['timestamp'], df_agg['load_groups'], color='#DC143C',
             linewidth=1.2, where='post', label='Load Groups', alpha=0.7)
    ax2.set_ylabel('Load Groups')
    ax2.set_ylim(-0.5, 5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')
    ax1.set_title('15-min Aggregated: MPPT vs Load', fontweight='bold')
    ax1.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    out = fig_dir / 'mppt_vs_load_15min.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  mppt_vs_load_15min.png")


if __name__ == '__main__':
    main()

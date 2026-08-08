"""
Generate 14-day synthetic training data for V14 model.

Variable load (1-4 groups @ ~150 mW each), realistic PV profile
based on observed P302 data, and Taipower 2026 TOU pricing.

Load schedule reference:
  - IEC 62257-9 (Small renewable energy/hybrid systems) load profiles
  - IEEE Std 1547-2018 distributed generation load curves
  - Palma-Behnke et al. (2013), IEEE Trans. Ind. Electron. — microgrid EMS
  - Observed P302 lab data: 4 groups, each ~100-155 mW
  - User spec: 4 groups × 150 mW = 600 mW full load
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

DAYS = 14
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
TOTAL_STEPS = DAYS * STEPS_PER_DAY

LOAD_PER_GROUP_W = 0.150   # 150 mW = 0.150 W

# Observed P302 hourly mean solar PV (W)
PV_HOURLY_W = {
    0: 0.015, 1: 0.015, 2: 0.015, 3: 0.015, 4: 0.015, 5: 0.015,
    6: 0.247, 7: 0.514, 8: 0.594, 9: 0.690, 10: 1.013, 11: 1.137,
    12: 1.411, 13: 1.417, 14: 1.120, 15: 0.795, 16: 0.575, 17: 0.351,
    18: 0.018, 19: 0.015, 20: 0.015, 21: 0.015, 22: 0.015, 23: 0.015,
}

# Typical daily load schedule (number of load groups per hour)
LOAD_GROUPS_SCHEDULE = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 3,
    9: 3, 10: 4, 11: 4,
    12: 3, 13: 3,
    14: 4, 15: 4, 16: 3,
    17: 3, 18: 2, 19: 2,
    20: 2, 21: 1, 22: 1, 23: 1,
}


def get_tou_price(hour: int, is_weekend: bool) -> float:
    if is_weekend:
        return 2.06
    if 0 <= hour < 9:
        return 2.06
    elif 9 <= hour < 16:
        return 4.69
    elif 16 <= hour < 22:
        return 7.13
    else:
        return 4.69


start_time = datetime(2026, 3, 1, 0, 0, 0)
records = []

for day in range(DAYS):
    # Day-level weather & load variation
    pv_scale = np.random.uniform(0.5, 1.4)
    day_load_bias = np.random.choice([-1, 0, 0, 0, 1])
    # Weekend: lighter load
    is_day_weekend = (start_time + timedelta(days=day)).weekday() >= 5

    for step_in_day in range(STEPS_PER_DAY):
        hour = step_in_day // STEPS_PER_HOUR
        minute = (step_in_day % STEPS_PER_HOUR) * 15
        ts = start_time + timedelta(days=day, hours=hour, minutes=minute)
        is_weekend = ts.weekday() >= 5

        # Solar PV
        pv_base_w = PV_HOURLY_W.get(hour, 0.015)
        pv_noise = 1.0 + np.random.uniform(-0.15, 0.15)
        pv_w = max(0.0, pv_base_w * pv_scale * pv_noise)

        # Load groups
        base_groups = LOAD_GROUPS_SCHEDULE.get(hour, 1)
        if is_day_weekend:
            base_groups = max(1, base_groups - 1)
        step_noise = np.random.choice([-1, 0, 0, 1])
        total_groups = int(np.clip(base_groups + day_load_bias + step_noise, 1, 4))
        load_w = total_groups * LOAD_PER_GROUP_W
        load_w += np.random.uniform(-0.015, 0.015)
        load_w = max(0.050, load_w)

        price = get_tou_price(hour, is_weekend)

        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'Solar': pv_w / 1000.0,
            'Consumption': load_w / 1000.0,
            'price': price,
            'hour': hour,
            'day_of_week': ts.weekday(),
            'load_groups': total_groups,
        })

df = pd.DataFrame(records)
out_path = 'data/processed/training_v14_14day.csv'
df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f'Generated {len(df)} rows -> {out_path}')
print(f'Time: {df.timestamp.iloc[0]} ~ {df.timestamp.iloc[-1]}')
print(f'Solar (kW): mean={df.Solar.mean():.6f}, max={df.Solar.max():.6f}')
print(f'Load  (kW): mean={df.Consumption.mean():.6f}, max={df.Consumption.max():.6f}')
print(f'Load groups dist: {df.load_groups.value_counts().sort_index().to_dict()}')

# Hourly breakdown
print('\n--- Load groups by hour (avg) ---')
for h in range(24):
    hd = df[df['hour'] == h]
    print(f'  H{h:02d}: {hd.load_groups.mean():.1f} groups, '
          f'load={hd.Consumption.mean()*1e6:.0f} mW, '
          f'solar={hd.Solar.mean()*1e6:.0f} mW')

# Weekend vs weekday
wd = df[df['day_of_week'] < 5]
we = df[df['day_of_week'] >= 5]
print(f'\nWeekday: load={wd.Consumption.mean()*1e6:.0f}mW, solar={wd.Solar.mean()*1e6:.0f}mW')
print(f'Weekend: load={we.Consumption.mean()*1e6:.0f}mW, solar={we.Solar.mean()*1e6:.0f}mW')

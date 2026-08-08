# Livestock Farm Microgrid Gymnasium Environment

This folder contains a standalone custom Gymnasium environment for a livestock
farm microgrid in Taoyuan, Taiwan.

## Files

- `livestock_microgrid_env.py`: modular Python implementation with:
  - `LoadGenerator`
  - `SolarPredictor`
  - `MicrogridEnv`

## Key System Specs

- Location: lat `25.02702`, lon `121.12371`
- Time step: 15 minutes
- PV: 300 kWp, tilt 25 deg, azimuth 180 deg
- Battery: Vanadium redox flow battery, 50 kW / 100 kWh
- Battery charge/discharge efficiency: 95% / 95%
- Equivalent round-trip efficiency: 90.25%
- Safe SoC range: 20% to 80%

## Billing-Based Load Sizing

The two electricity bills are treated as two load sources in the same farm
site:

- Bill A: `16320 kWh`, contracted/regular peak demand `72 kW`
- Bill B: `11760 kWh`, peak demand assumed from the second bill as `32 kW`
- Combined energy: `28080 kWh` over 31 days, or `905.81 kWh/day`
- Combined peak cap: `72 + 32 = 104 kW`

The load generator therefore uses `905.81 kWh/day` and a peak cap of `104 kW`
for the combined site profile. If only Bill A is modeled, use `526.45 kWh/day`
and `72 kW`.

## Battery Sizing Note

The battery is set to `50 kW / 100 kWh` as requested. With a 20%-80% safe SoC
band, the usable energy is:

```text
100 kWh * (0.8 - 0.2) = 60 kWh
```

## Important Note About the Load Requirement

The requested load segment ranges and the default combined daily energy target
(`905.81 kWh`) are mathematically inconsistent if all segment bounds are enforced
strictly. The minimum energy implied by the requested segment ranges is already:

```text
6h * 20kW + 3h * 70kW + 7h * 60kW + 3h * 90kW + 5h * 20kW = 1120 kWh
```

Therefore the default generator preserves the livestock-farm daily shape,
injects bounded noise, normalizes the peak to `104 kW`, and calibrates the daily
integral to about `905.81 kWh`.

If strict segment bounds are required, instantiate:

```python
LoadGenerator(strict_segment_bounds=True)
```

but the resulting daily energy will be higher than `905 kWh`.

## Smoke Test

From the repository root:

```powershell
py -3 examples\livestock_farm_microgrid\livestock_microgrid_env.py
```

## Minimal Usage

```python
from examples.livestock_farm_microgrid.livestock_microgrid_env import MicrogridEnv

env = MicrogridEnv(use_weather_api=False, seed=42)
obs, info = env.reset(options={"initial_soc": 0.5})

for _ in range(96):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Observation

The observation vector is:

```text
[soc,
 hour_sin,
 hour_cos,
 load_kw / 104,
 observed_pv_kw / 300,
 clear_sky_power_kw / 300,
 weather_adjusted_power_kw / 300,
 grid_price / 10,
 cloud_cover_pct / 100]
```

The two solar potential features are included to reduce the
demand-censored partial observability problem:

- `clear_sky_power_kw`: physical upper bound from `pvlib`
- `weather_adjusted_power_kw`: weather-degraded potential from Open-Meteo data

## Action

The action is continuous:

```text
Box([-1], [1])
```

- positive action: charge battery
- negative action: discharge battery
- zero action: standby

This planning environment allows partial battery support. The battery offsets
as much load as its power and SoC allow, and the grid supplies the remaining
demand. This is different from the P302 hardware-specific deployment constraint.

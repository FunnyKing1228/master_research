"""P302 real-hardware configuration helper.

This module maps the zinc-air SLFB microgrid testbed into environment keyword
arguments that can be passed to ``MicrogridEnvironment``. It describes the
hardware assumptions used for data collected under ``data/raw/``.

Hardware summary:
  - Battery: SLFB zinc-air battery, charge 8.5 V, discharge 5.6 V.
  - MPPT: measured peak around 1.75 W, typical daytime 700-800 mW.
  - Electronic load: four controlled groups.
  - Sampling: about 11 seconds, aggregated into 15-minute windows.

Flow-rate equivalent model:
  - R_base = (V_charge - V_discharge) / (2 * I_rated).
  - P_pump(Q) = P_max * Q^3.
  - R_eq(Q) = R_base * (1 + k_R * (1 - Q) / Q).
  - Low flow saves pump power but increases resistance and losses.
  - High flow improves reaction stability but increases pump parasitic power.
"""

import os

# ──────────────────────────────────────────────────────────────
# Battery parameters: 50 mA system current and 12-hour charge/discharge.
# ──────────────────────────────────────────────────────────────
_SYSTEM_CURRENT_A       = 0.05           # System current: 50 mA = 0.05 A.
DISCHARGE_HOURS         = 12.0           # Maximum charge/discharge time.
BATTERY_CHARGE_V        = 8.5            # Charge voltage.
BATTERY_DISCHARGE_V     = 5.6            # Discharge voltage, 1.4 V x 4 cells.
BATTERY_AVG_V           = (BATTERY_CHARGE_V + BATTERY_DISCHARGE_V) / 2  # 7.05 V
BATTERY_CHARGE_I_MA     = _SYSTEM_CURRENT_A * 1000  # 20.0 mA
BATTERY_CAPACITY_MAH    = BATTERY_CHARGE_I_MA * DISCHARGE_HOURS  # 240.0 mAh
BATTERY_CAPACITY_WH     = (_SYSTEM_CURRENT_A * BATTERY_DISCHARGE_V) * DISCHARGE_HOURS  # 1.344 Wh
BATTERY_CAPACITY_KWH    = BATTERY_CAPACITY_WH / 1000   # 0.001344 kWh
BATTERY_POWER_W         = _SYSTEM_CURRENT_A * BATTERY_DISCHARGE_V  # 0.112 W
BATTERY_POWER_KW        = BATTERY_POWER_W / 1000        # 0.000112 kW
BATTERY_EFFICIENCY      = 0.95           # System conversion efficiency, updated 2026/03/16.

# ──────────────────────────────────────────────────────────────
# Flow-rate electrochemical equivalent model parameters.
# ──────────────────────────────────────────────────────────────
# Baseline internal resistance: R_base = (V_charge - V_discharge) / (2 x I_rated).
# Formula: (8.5 - 5.6) / (2 x 0.05) = 29.0 ohm.
FLOW_R_BASE_OHM         = (BATTERY_CHARGE_V - BATTERY_DISCHARGE_V) / (2 * _SYSTEM_CURRENT_A)  # 72.5 Ω
# Max pump parasitic power is about 15% of discharge power.
# Formula: 0.280 W x 0.15 = 0.042 W (42 mW).
FLOW_P_MAX_PUMP_W       = BATTERY_POWER_W * 0.15
# Resistance growth factor, used as a tunable hyperparameter.
FLOW_K_R                = 0.5
# Open-circuit voltage assumptions.
FLOW_V_OCV_CHARGE       = BATTERY_CHARGE_V     # 8.5 V
FLOW_V_OCV_DISCHARGE    = BATTERY_DISCHARGE_V  # 5.6 V
# Rated current.
FLOW_I_RATED_A          = _SYSTEM_CURRENT_A    # 0.050 A

# ──────────────────────────────────────────────────────────────
# Load parameters.
# ──────────────────────────────────────────────────────────────
LOAD_GROUPS             = 4              # Number of load groups.
LOAD_PER_GROUP_W        = 0.1            # Per-group power: 100 mW = 0.1 W.
LOAD_VOLTAGE            = 5.0            # V
LOAD_MAX_W              = LOAD_GROUPS * LOAD_PER_GROUP_W   # 0.4 W (400 mW)
LOAD_MAX_KW             = LOAD_MAX_W / 1000                # 0.0004 kW
# BATTERY_POWER_W vs LOAD_MAX_W indicates how much load the battery can cover.
# Battery-solo scenario can be feasible when PV also contributes to the load.

# ──────────────────────────────────────────────────────────────
# MPPT/PV parameters from measured statistics.
# ──────────────────────────────────────────────────────────────
MPPT_PEAK_W             = 1.75           # Historical maximum power in W.
MPPT_DAY_AVG_W          = 0.75           # Daytime average power in W.
MPPT_PEAK_KW            = MPPT_PEAK_W / 1000
PV_START_HOUR           = 6
PV_END_HOUR             = 18

# ──────────────────────────────────────────────────────────────
# Time parameters.
# ──────────────────────────────────────────────────────────────
SAMPLE_INTERVAL_SEC     = 11.0           # Raw sampling interval.
AGGREGATION_WINDOW_MIN  = 15             # Aggregation window.
TIME_STEP_H             = AGGREGATION_WINDOW_MIN / 60.0  # 0.25 h
STEPS_PER_DAY           = int(24 / TIME_STEP_H)          # 96 steps/day
EPISODE_LENGTH_STEPS    = STEPS_PER_DAY                   # One day per episode.

# ──────────────────────────────────────────────────────────────
# TOU price model.
# ──────────────────────────────────────────────────────────────
PRICE_OFF_PEAK          = 0.10           # Off-peak price in $/kWh.
PRICE_ON_PEAK           = 0.18           # On-peak price in $/kWh, 08:00-18:00.

# ──────────────────────────────────────────────────────────────
# Dataset path.
# ──────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
TRAINING_CSV = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'training_7day_15min.csv')


def get_env_kwargs(use_extended_obs: bool = True) -> dict:
    """
    Return keyword arguments for ``MicrogridEnvironment(**kwargs)``.

    Example:
        from configs.p302_real_hw import get_env_kwargs
        env = MicrogridEnvironment(**get_env_kwargs())
    """
    kw = dict(
        # Battery.
        battery_capacity_kwh    = BATTERY_CAPACITY_KWH,
        battery_power_kw        = BATTERY_POWER_KW,
        battery_efficiency      = BATTERY_EFFICIENCY,
        soc_min                 = 0.05,         # Avoid deep discharge for zinc-air cells.
        soc_max                 = 0.95,

        # Time.
        episode_length          = EPISODE_LENGTH_STEPS,
        time_step               = TIME_STEP_H,

        # Synthetic-data parameters used as fallback if no CSV exists.
        synthetic_pv_peak_kw    = MPPT_PEAK_KW,
        synthetic_pv_start_hour = PV_START_HOUR,
        synthetic_pv_end_hour   = PV_END_HOUR,
        synthetic_load_base_kw  = LOAD_PER_GROUP_W / 1000,  # One group as baseline.
        synthetic_load_amp_kw   = (LOAD_MAX_W - LOAD_PER_GROUP_W) / 1000,
        synthetic_price_base    = PRICE_OFF_PEAK,
        synthetic_price_peak    = PRICE_ON_PEAK,
        synthetic_price_peak_start = 8,
        synthetic_price_peak_end   = 18,

        # External CSV dataset.
        dataset_csv_path        = TRAINING_CSV,
        dataset_pv_column       = 'Solar',           # kW
        dataset_time_column     = 'timestamp',
        dataset_power_scale     = 1.0,                # Already converted to kW in CSV.

        # Extended state space.
        use_extended_obs        = use_extended_obs,
        initial_soh             = 1.0,
        soh_degradation_per_kwh = 0.001,  # Zinc-air cells degrade relatively quickly.
        initial_flow_rate_lpm   = 0.0,    # No liquid cooling by default.

        # Column mapping.
        dataset_pv_std_column   = 'mppt_p_std_W',     # Environment converts W to kW.
        dataset_pv_max_column   = 'mppt_p_max_W',
        dataset_load_std_column = 'load_std_W',
        dataset_load_max_column = 'load_max_W',

        # Flow-rate electrochemical equivalent model.
        use_flow_rate_action    = use_extended_obs,  # Extended mode enables 2D action.
        flow_R_base_ohm         = FLOW_R_BASE_OHM,
        flow_P_max_pump_W       = FLOW_P_MAX_PUMP_W,
        flow_k_R                = FLOW_K_R,
        flow_V_OCV_charge       = FLOW_V_OCV_CHARGE,
        flow_V_OCV_discharge    = FLOW_V_OCV_DISCHARGE,
        flow_I_rated_A          = FLOW_I_RATED_A,

        # Safety bounds.
        ramp_limit_kw           = BATTERY_POWER_KW * 0.5,  # Ramp limit.
        allow_grid_trading      = True,
    )
    return kw


# ──────────────────────────────────────────────────────────────
# Print summary.
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("P302 Real Hardware Configuration")
    print("=" * 60)
    print(f"Battery Capacity  : {BATTERY_CAPACITY_MAH} mAh = {BATTERY_CAPACITY_WH:.4f} Wh = {BATTERY_CAPACITY_KWH:.6f} kWh")
    print(f"Battery Power     : {BATTERY_POWER_W:.4f} W = {BATTERY_POWER_KW:.6f} kW")
    print(f"Battery Efficiency: {BATTERY_EFFICIENCY}")
    print(f"Charge V / I      : {BATTERY_CHARGE_V} V / {BATTERY_CHARGE_I_MA} mA")
    print(f"Discharge V       : {BATTERY_DISCHARGE_V} V")
    print(f"MPPT Peak         : {MPPT_PEAK_W} W = {MPPT_PEAK_KW:.6f} kW")
    print(f"Load Max          : {LOAD_MAX_W} W = {LOAD_MAX_KW:.4f} kW ({LOAD_GROUPS} groups)")
    print(f"Time Step         : {TIME_STEP_H} h = {AGGREGATION_WINDOW_MIN} min")
    print(f"Steps/Day         : {STEPS_PER_DAY}")
    print(f"Training CSV      : {TRAINING_CSV}")
    print(f"\nFull charge time from 0%: {BATTERY_CAPACITY_MAH/BATTERY_CHARGE_I_MA*60:.1f} min")
    print(f"Energy per charge cycle : {BATTERY_CAPACITY_WH:.4f} Wh")

    print(f"\n{'='*60}")
    print("Flow Rate Model (SLFB Synthetic)")
    print(f"{'='*60}")
    print(f"R_base            : {FLOW_R_BASE_OHM:.1f} Ohm")
    print(f"P_max_pump        : {FLOW_P_MAX_PUMP_W*1000:.1f} mW ({FLOW_P_MAX_PUMP_W:.4f} W)")
    print(f"k_R               : {FLOW_K_R}")
    print(f"V_OCV charge      : {FLOW_V_OCV_CHARGE} V")
    print(f"V_OCV discharge   : {FLOW_V_OCV_DISCHARGE} V")
    print(f"I_rated           : {FLOW_I_RATED_A*1000:.0f} mA")

    # Show model behavior under different flow rates.
    print(f"\n{'Q%':>5} | {'R_eq(Ohm)':>10} | {'V_dis(V)':>8} | {'V_chg(V)':>8} | {'P_pump(mW)':>10} | {'eta_dis':>7} | {'eta_chg':>7}")
    print("-" * 75)
    I = FLOW_I_RATED_A
    for q_pct in [1, 5, 10, 25, 50, 75, 100]:
        Q = q_pct / 100.0
        R_eq = FLOW_R_BASE_OHM * (1.0 + FLOW_K_R * (1.0 - Q) / Q)
        V_dis = max(0, FLOW_V_OCV_DISCHARGE - I * R_eq)
        V_chg = FLOW_V_OCV_CHARGE + I * R_eq
        P_pump = FLOW_P_MAX_PUMP_W * Q**3 * 1000  # mW
        eta_dis = V_dis / max(FLOW_V_OCV_DISCHARGE, 1e-6)
        eta_chg = FLOW_V_OCV_CHARGE / max(V_chg, 1e-6)
        print(f"{q_pct:>4}% | {R_eq:>10.1f} | {V_dis:>8.3f} | {V_chg:>8.3f} | {P_pump:>10.3f} | {eta_dis:>7.3f} | {eta_chg:>7.3f}")

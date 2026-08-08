import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from pymgrid import MicrogridGenerator
    from pymgrid.microgrid import Microgrid
    PYTHON_MICROGRID_AVAILABLE = True
except ImportError:
    PYTHON_MICROGRID_AVAILABLE = False
    print("Info: python-microgrid not installed. Running MicrogridEnvironment in python-microgrid synthetic mode.")


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
def get_taipower_tou_price(hour: int, is_weekend: bool) -> float:
    """
    2026 Taipower Summer Time-of-Use electricity rate (TWD/kWh).
    
    Weekdays:
      00:00 - 09:00  Off-Peak    2.06 TWD/kWh
      09:00 - 16:00  Mid-Peak    4.69 TWD/kWh
      16:00 - 22:00  Peak        7.13 TWD/kWh
      22:00 - 24:00  Mid-Peak    4.69 TWD/kWh
    
    Weekends / Holidays:
      All day                     2.06 TWD/kWh (flat, no arbitrage)
    
    Args:
        hour: 0-23
        is_weekend: True if Saturday(5) or Sunday(6), using Monday=0 convention
    Returns:
        Price in TWD/kWh
    """
    if is_weekend:
        return 2.06
    # Weekday TOU
    if 0 <= hour < 9:
        return 2.06   # Off-Peak
    elif 9 <= hour < 16:
        return 4.69   # Mid-Peak
    elif 16 <= hour < 22:
        return 7.13   # Peak
    else:  # 22-24
        return 4.69   # Mid-Peak (Late)


def determine_situation_code(action_kw: float, net_load_after_pv: float,
                             pv_kw: float = 0.0, load_kw: float = 0.0) -> int:
    """
    
    
    
    Args:
    Returns:
        1, 2, 3, or 4
    """
    if action_kw < -1e-6:  # Discharge
        discharge_kw = abs(action_kw)
        if discharge_kw >= load_kw * 0.99:
            return 1  # Battery Solo (battery alone covers full load)
        else:
            return 4  # Invalid discharge request -> standby/grid supply
    elif action_kw > 1e-6:  # Charge
        return 3  # Grid Charge
    else:
        return 4  # Standby


class MicrogridEnvironment(gym.Env):
    """
    """
    
    def __init__(
        self,
        microgrid_id: int = 0,
        episode_length: int = 24,
        time_step: int = 1,  # hours
        battery_capacity_kwh: float = 100.0,
        battery_power_kw: float = 50.0,
        battery_charge_power_kw: Optional[float] = None,
        battery_discharge_power_kw: Optional[float] = None,
        battery_efficiency: float = 0.95,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        clip_soc_to_bounds: bool = True,
        price_scaling: float = 1.0,
        reward_scaling: float = 1.0,
        use_real_data: bool = True,
        ramp_limit_kw: float = None,
        hard_guard: bool = False,
        allow_grid_trading: bool = True,
        # External dataset (CSV) options
        dataset_csv_path: Optional[str] = None,
        dataset_pv_join_wind: bool = False,
        train_window_hours: Optional[int] = None,
        dataset_pv_column: Optional[str] = None,
        dataset_load_kw: Optional[float] = None,
        dataset_power_scale: float = 1.0,
        dataset_time_column: Optional[str] = None,
        use_dataset_timestamps_for_obs: bool = True,
        deployment_observation_style: bool = False,
        deployment_window_steps: int = 1,
        deployment_load_threshold_kw: float = 0.0005,
        deployment_group_power_kw: float = 0.0001,
        battery_delivered_load_per_group_kw: Optional[float] = None,
        continuous_operation_mode: bool = False,
        deployment_guard_style: bool = False,
        enforce_solo_discharge_load_limit: bool = True,
        pre_measure_rest_flow_fraction: float = 0.0,
        pre_measure_flow_fraction: float = 0.0,
        pre_measure_seconds: float = 0.0,
        # Weather variation (synthetic data only)
        weather_pv_scale_std: float = 0.0,
        weather_load_scale_std: float = 0.0,
        weather_pv_noise_std: float = 0.0,
        # Synthetic hourly-hold pattern (for 15-min actions)
        synthetic_hourly_hold: bool = False,
        synthetic_pv_peak_kw: float = 20.0,
        synthetic_pv_start_hour: int = 6,
        synthetic_pv_end_hour: int = 18,
        synthetic_load_base_kw: float = 10.0,
        synthetic_load_amp_kw: float = 5.0,
        synthetic_price_base: float = 0.12,
        synthetic_price_peak: float = 0.20,
        synthetic_price_peak_start: int = 8,
        synthetic_price_peak_end: int = 18,
        stress_enable: bool = False,
        stress_efficiency_noise_std: float = 0.0,
        stress_dt_jitter_std: float = 0.0,
        stress_action_lag_alpha: float = 0.0,
        stress_soc_obs_delay: int = 0,
        stress_soc_obs_noise_std: float = 0.0,
        stress_bounds_drift_std: float = 0.0,
        stress_external_pmax_shrink_prob: float = 0.0,
        stress_external_pmax_shrink_factor: float = 1.0,
        stress_power_loss_ratio: float = 0.0,
        stress_battery_response_noise_std: float = 0.0,
        stress_battery_zero_response_prob: float = 0.0,
        use_extended_obs: bool = False,
        initial_soh: float = 1.0,
        soh_degradation_per_kwh: float = 0.0,
        initial_flow_rate_lpm: float = 0.0,
        dataset_pv_std_column: Optional[str] = None,
        dataset_pv_max_column: Optional[str] = None,
        dataset_load_std_column: Optional[str] = None,
        dataset_load_max_column: Optional[str] = None,
        dataset_soh_column: Optional[str] = None,
        dataset_flow_rate_column: Optional[str] = None,
        use_flow_rate_action: bool = False,
        fixed_flow_fraction_when_uncontrolled: float = 0.0,
        flow_R_base_ohm: float = 72.5,
        flow_P_max_pump_W: float = 0.0168,     # W（16.8 mW）
        flow_k_R: float = 0.5,
        flow_V_OCV_charge: float = 8.5,
        flow_V_OCV_discharge: float = 5.5,
        flow_I_rated_A: float = 0.020,        # A（20 mA）
        flow_min_active_fraction: float = 0.01,
        flow_idle_fraction: float = 0.0,
        flow_pump_from_grid: bool = False,
        flow_charge_pump_free: bool = False,
        flow_limits_available_power: bool = False,
        flow_power_min_fraction: float = 0.0,
        flow_operating_rule_enabled: bool = True,
        # ── TOU Reward Scale ────────────────────────────────────────────
        tou_reward_scale: float = 3000.0,
        allow_grid_export: bool = False,
        feed_in_tariff_ratio: float = 0.5,
        discharge_auto: bool = False,
        # discharge_mode:
        discharge_mode: str = 'solo_only',
        voltage_cutoff_soc: float = 0.0,
        action_dead_zone_kw: float = 0.0,
        # discharge_intent_threshold_kw: in discharge_auto mode, a negative action
        # must exceed this threshold before it is treated as Battery Solo intent.
        discharge_intent_threshold_kw: float = 0.0,
        # reward_version: 'p302' (v6-v13) | 'v14' | 'v16*' | 'v17'
        reward_version: str = 'p302',
        pv_obs_boolean: bool = False,
        pv_obs_boolean_threshold_kw: float = 0.001,  # legacy fallback only; preferred rule uses pv/load ratio
        pv_support_ratio_obs: bool = False,
        pv_support_ratio_max: float = 1.5,
        pv_sufficient_ratio_threshold: float = 0.8,
        price_obs: bool = True,
        tou_onehot_obs: bool = False,
        charge_requires_pv_surplus: bool = False,
        charge_limit_to_pv_surplus: bool = False,
    ):
        super().__init__()
        
        self.episode_length = episode_length
        self.time_step = time_step
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_charge_power_kw = float(
            battery_charge_power_kw if battery_charge_power_kw is not None else battery_power_kw
        )
        self.battery_discharge_power_kw = float(
            battery_discharge_power_kw if battery_discharge_power_kw is not None else battery_power_kw
        )
        # Backward-compatible aggregate power used by legacy reward code / generic scaling.
        self.battery_power_kw = max(self.battery_charge_power_kw, self.battery_discharge_power_kw)
        self.battery_efficiency = battery_efficiency
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.clip_soc_to_bounds = bool(clip_soc_to_bounds)
        # Optional hard physical SoC floor (e.g. 0.0). When set, discharge is
        # truncated to the energy actually available down to this floor so the
        # battery cannot deliver energy it does not have; the unmet load is
        # picked up by the grid in the downstream accounting. Default None keeps
        # every existing scenario byte-for-byte unchanged. Set via config key
        # env.soc_physical_floor (wired in train_sac_microgrid.create_environment).
        self.soc_physical_floor = None
        self.price_scaling = price_scaling
        self.reward_scaling = reward_scaling
        self.use_real_data = use_real_data
        self.ramp_limit_kw = ramp_limit_kw
        self.hard_guard = hard_guard
        self.allow_grid_trading = bool(allow_grid_trading)
        self.tou_reward_scale = float(tou_reward_scale)
        self.discharge_auto = bool(discharge_auto)
        discharge_mode = str(discharge_mode).strip().lower()
        if discharge_mode == 'partial_assist':
            discharge_mode = 'solo_only'
        if discharge_mode != 'solo_only':
            discharge_mode = 'solo_only'
        self.discharge_mode = discharge_mode
        self.voltage_cutoff_soc = float(voltage_cutoff_soc)
        self.action_dead_zone_kw = float(action_dead_zone_kw)
        self.discharge_intent_threshold_kw = float(discharge_intent_threshold_kw)
        self.reward_version = str(reward_version)
        self.pv_obs_boolean = bool(pv_obs_boolean)
        self.pv_obs_boolean_threshold_kw = float(pv_obs_boolean_threshold_kw)
        self.pv_support_ratio_obs = bool(pv_support_ratio_obs)
        self.pv_support_ratio_max = float(max(0.1, pv_support_ratio_max))
        self.pv_sufficient_ratio_threshold = float(max(0.0, pv_sufficient_ratio_threshold))
        self.price_obs = bool(price_obs)
        self.tou_onehot_obs = bool(tou_onehot_obs)
        self.charge_requires_pv_surplus = bool(charge_requires_pv_surplus)
        self.charge_limit_to_pv_surplus = bool(charge_limit_to_pv_surplus)
        self.pv_bool_data = None
        self._solar_charge_coeff = 0.8
        self._solar_waste_penalty = 0.4
        self._low_soc_charge_thresh = 0.15
        self._low_soc_charge_bonus = 1.0
        self._power_util_coeff = 0.0
        self.offpeak_charge_soc_target = 0.85
        self.peak_discharge_soc_floor = 0.15
        self.v17_offpeak_charge_bonus = 0.6
        self.v17_peak_discharge_bonus = 1.2
        self.v17_peak_idle_penalty = 0.4
        self.v17_solar_storage_bonus_scale = 0.3
        self.no_pv_action_threshold_kw = 0.001
        self.no_pv_throughput_penalty_per_kwh = 0.0
        self.offpeak_no_pv_discharge_penalty_per_kwh = 0.0
        self.dataset_csv_path = dataset_csv_path
        self.dataset_pv_join_wind = bool(dataset_pv_join_wind)
        self.train_window_hours = int(train_window_hours) if train_window_hours is not None else None
        self.dataset_pv_column = dataset_pv_column
        self.dataset_load_kw = float(dataset_load_kw) if dataset_load_kw is not None else None
        self.dataset_power_scale = float(dataset_power_scale)
        self.dataset_time_column = dataset_time_column
        self.use_dataset_timestamps_for_obs = bool(use_dataset_timestamps_for_obs)
        self.deployment_observation_style = bool(deployment_observation_style)
        self.deployment_window_steps = max(1, int(deployment_window_steps))
        self.deployment_load_threshold_kw = float(max(0.0, deployment_load_threshold_kw))
        self.deployment_group_power_kw = float(max(0.0, deployment_group_power_kw))
        self.battery_delivered_load_per_group_kw = (
            None
            if battery_delivered_load_per_group_kw is None
            else float(max(0.0, battery_delivered_load_per_group_kw))
        )
        self.continuous_operation_mode = bool(continuous_operation_mode)
        self.deployment_guard_style = bool(deployment_guard_style)
        self.enforce_solo_discharge_load_limit = bool(enforce_solo_discharge_load_limit)
        self.pre_measure_rest_flow_fraction = float(np.clip(pre_measure_rest_flow_fraction, 0.0, 1.0))
        self.pre_measure_flow_fraction = float(np.clip(pre_measure_flow_fraction, 0.0, 1.0))
        self.pre_measure_seconds = float(max(0.0, pre_measure_seconds))
        # Weather variation settings
        self.weather_pv_scale_std = float(weather_pv_scale_std)
        self.weather_load_scale_std = float(weather_load_scale_std)
        self.weather_pv_noise_std = float(weather_pv_noise_std)
        self.synthetic_hourly_hold = bool(synthetic_hourly_hold)
        self.synthetic_pv_peak_kw = float(synthetic_pv_peak_kw)
        self.synthetic_pv_start_hour = int(synthetic_pv_start_hour)
        self.synthetic_pv_end_hour = int(synthetic_pv_end_hour)
        self.synthetic_load_base_kw = float(synthetic_load_base_kw)
        self.synthetic_load_amp_kw = float(synthetic_load_amp_kw)
        self.synthetic_price_base = float(synthetic_price_base)
        self.synthetic_price_peak = float(synthetic_price_peak)
        self.synthetic_price_peak_start = int(synthetic_price_peak_start)
        self.synthetic_price_peak_end = int(synthetic_price_peak_end)
        self.fixed_start_idx: Optional[int] = None
        self._valid_episode_start_indices: Optional[np.ndarray] = None
        self._valid_episode_window_count = 0
        # Stress settings
        self.stress_enable = bool(stress_enable)
        self.stress_efficiency_noise_std = float(stress_efficiency_noise_std)
        self.stress_dt_jitter_std = float(stress_dt_jitter_std)
        self.stress_action_lag_alpha = float(stress_action_lag_alpha)
        self.stress_soc_obs_delay = int(stress_soc_obs_delay)
        self.stress_soc_obs_noise_std = float(stress_soc_obs_noise_std)
        self.stress_bounds_drift_std = float(stress_bounds_drift_std)
        self.stress_external_pmax_shrink_prob = float(stress_external_pmax_shrink_prob)
        self.stress_external_pmax_shrink_factor = float(stress_external_pmax_shrink_factor)
        self.stress_power_loss_ratio = float(np.clip(stress_power_loss_ratio, 0.0, 0.3))
        self.stress_battery_response_noise_std = float(max(0.0, stress_battery_response_noise_std))
        self.stress_battery_zero_response_prob = float(np.clip(stress_battery_zero_response_prob, 0.0, 1.0))
        self.use_extended_obs = bool(use_extended_obs)
        self.current_soh = float(np.clip(initial_soh, 0.0, 1.0))
        self._initial_soh = float(np.clip(initial_soh, 0.0, 1.0))
        self.soh_degradation_per_kwh = float(soh_degradation_per_kwh)
        self.current_flow_rate_lpm = float(initial_flow_rate_lpm)
        self._initial_flow_rate_lpm = float(initial_flow_rate_lpm)
        self.dataset_pv_std_column   = dataset_pv_std_column
        self.dataset_pv_max_column   = dataset_pv_max_column
        self.dataset_load_std_column = dataset_load_std_column
        self.dataset_load_max_column = dataset_load_max_column
        self.dataset_soh_column      = dataset_soh_column
        self.dataset_flow_rate_column = dataset_flow_rate_column
        self.pv_std_data  : Optional[np.ndarray] = None
        self.pv_max_data  : Optional[np.ndarray] = None
        self.load_std_data: Optional[np.ndarray] = None
        self.load_max_data: Optional[np.ndarray] = None
        self.soh_data     : Optional[np.ndarray] = None
        self.flow_rate_data: Optional[np.ndarray] = None
        self.use_flow_rate_action = bool(use_flow_rate_action)
        self.fixed_flow_fraction_when_uncontrolled = float(
            np.clip(fixed_flow_fraction_when_uncontrolled, 0.0, 1.0)
        )
        self.flow_R_base_ohm      = float(flow_R_base_ohm)
        self.flow_P_max_pump_W    = float(flow_P_max_pump_W)
        self.flow_k_R             = float(flow_k_R)
        self.flow_V_OCV_charge    = float(flow_V_OCV_charge)
        self.flow_V_OCV_discharge = float(flow_V_OCV_discharge)
        self.flow_I_rated_A       = float(flow_I_rated_A)
        self.flow_min_active_fraction = float(np.clip(flow_min_active_fraction, 0.0, 1.0))
        self.flow_idle_fraction = float(np.clip(flow_idle_fraction, 0.0, 1.0))
        self.flow_pump_from_grid = bool(flow_pump_from_grid)
        self.flow_charge_pump_free = bool(flow_charge_pump_free)
        self.flow_limits_available_power = bool(flow_limits_available_power)
        self.flow_power_min_fraction = float(np.clip(flow_power_min_fraction, 0.0, 1.0))
        self.flow_operating_rule_enabled = bool(flow_operating_rule_enabled)
        self.allow_grid_export = bool(allow_grid_export)
        self.feed_in_tariff_ratio = float(np.clip(feed_in_tariff_ratio, 0.0, 1.0))
        self._current_flow_action = self.flow_idle_fraction
        # Effective parameters for stress
        self.soc_min_eff = self.soc_min
        self.soc_max_eff = self.soc_max
        self._effective_time_step = float(self.time_step)
        self._effective_efficiency = float(self.battery_efficiency)
        self._prev_exec_action_kw = 0.0
        self._soc_obs_buffer = []
        self.realized_violation_penalty = 20.0
        self.blocked_by_pv_penalty = 0.10
        self.blocked_by_load_penalty = 0.05
        
        # Initialize attributes
        self.microgrid = None
        self.microgrid_generator = None
        self.load_data = None
        self.pv_data = None
        self.price_data = None
        self.hour_data = None
        self.dow_data = None
        
        # Initialize microgrid
        self._init_microgrid(microgrid_id)
        # If user specified an external CSV dataset, override time series
        try:
            if isinstance(self.dataset_csv_path, str) and len(self.dataset_csv_path) > 0:
                self._load_external_csv(self.dataset_csv_path, self.dataset_pv_join_wind)
                print(f"Loaded external CSV dataset: {self.dataset_csv_path}")
        except Exception as e:
            print(f"Warning: failed loading external CSV dataset: {e}")
        
        # Environment state and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.current_soc = 0.1  # Will be randomized in reset()
        self.episode_data = None
        self.prev_action_kw = 0.0
        self._continuous_next_start_idx = 0
        self._continuous_initialized = False
        self._deployment_voltage_cutoff_active = False
        self._deployment_guard_day = None
        
        # Statistics
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.soc_violations = 0
        self.action_violations = 0
        self.strict_soc_violation_steps = 0
        self.strict_soc_violation_duration_h = 0.0
        self.strict_soc_violation_kwh = 0.0
        self.strict_soc_violation_max_kwh = 0.0
        
        self._last_grid_kw = 0.0
        self._last_net_load_after_pv = 0.0
        self._last_useful_discharge = 0.0
        self._last_pv_to_load = 0.0
        
    def _init_microgrid(self, microgrid_id: int):
        """Documentation for this public API is provided in English."""
        if not PYTHON_MICROGRID_AVAILABLE:
            self.microgrid = object()
            self._generate_synthetic_data()
            print(f"Microgrid {microgrid_id} initialized (synthetic mode)")
            print(f"  - Battery capacity: {self.battery_capacity_kwh:.1f} kWh")
            print(f"  - Battery power: charge={self.battery_charge_power_kw:.1f} kW, discharge={self.battery_discharge_power_kw:.1f} kW")
            print(f"  - Episode length: {self.episode_length} steps")
            return
            
        try:
            # Try different ways to initialize microgrid
            if hasattr(MicrogridGenerator, 'generate'):
                # Old API
                self.microgrid_generator = MicrogridGenerator(nb_microgrid=1)
                self.microgrid_generator.generate(microgrid_id)
                self.microgrid = self.microgrid_generator.microgrids[microgrid_id]
            elif hasattr(MicrogridGenerator, 'create'):
                # New API
                self.microgrid_generator = MicrogridGenerator()
                self.microgrid = self.microgrid_generator.create(nb_microgrid=1)[microgrid_id]
            else:
                # Try direct creation
                self.microgrid = Microgrid()
                print("Warning: Using basic Microgrid instance")
            
            # Get time series data
            self._load_time_series_data()
            
            print(f"Microgrid {microgrid_id} initialized successfully")
            print(f"  - Battery capacity: {self.battery_capacity_kwh:.1f} kWh")
            print(f"  - Battery power: charge={self.battery_charge_power_kw:.1f} kW, discharge={self.battery_discharge_power_kw:.1f} kW")
            print(f"  - Episode length: {self.episode_length} steps")
            
        except Exception as e:
            print(f"Warning: Failed to initialize microgrid: {e}")
            if isinstance(self.dataset_csv_path, str) and self.dataset_csv_path:
                print(
                    "Continuing with custom MicrogridEnvironment; the configured "
                    "external CSV overrides optional pymgrid time series"
                )
            else:
                print("Falling back to synthetic custom environment")
            self.microgrid = None
    
    def _load_time_series_data(self):
        """Documentation for this public API is provided in English."""
        if self.microgrid is None:
            return
            
        try:
            # Get load and PV data
            self.load_data = self.microgrid.load_ts
            self.pv_data = self.microgrid.pv_ts
            
            # Get price data (if available)
            if hasattr(self.microgrid, 'price_ts'):
                self.price_data = self.microgrid.price_ts
            else:
                # Generate synthetic price data
                self.price_data = self._generate_synthetic_prices()
                
            print(f"Loaded time series data:")
            print(f"  - Load data: {len(self.load_data)} points")
            print(f"  - PV data: {len(self.pv_data)} points")
            print(f"  - Price data: {len(self.price_data)} points")
            
        except Exception as e:
            print(f"Warning: Failed to load time series data: {e}")
            self._generate_synthetic_data()

    def _load_external_csv(self, csv_path: str, pv_join_wind: bool = False):
        """
        Load external CSV dataset.
        - Standard format: index/Consumption/Solar/Wind
        - Raw acquisition format: timestamp, solar_p_mw/mppt_p_mw, with optional dataset_load_kw
        """
        import pandas as pd
        p = csv_path
        df = pd.read_csv(p)
        if 'Consumption' in df.columns:
            load = df['Consumption'].astype(float).values
            pv = None
            if 'Solar' in df.columns:
                pv = df['Solar'].astype(float).values
            if pv_join_wind and 'Wind' in df.columns:
                wind = df['Wind'].astype(float).fillna(0.0).values
                pv = (pv if pv is not None else 0.0) + wind
            if pv is None:
                pv = np.zeros_like(load)
            time_col = self.dataset_time_column or ('index' if 'index' in df.columns else None)
        else:
            # Raw acquisition format (e.g., DATA_Acquisition)
            pv_col = self.dataset_pv_column
            if not pv_col:
                if 'solar_p_mw' in df.columns:
                    pv_col = 'solar_p_mw'
                elif 'mppt_p_mw' in df.columns:
                    pv_col = 'mppt_p_mw'
            if not pv_col or pv_col not in df.columns:
                raise ValueError('CSV missing PV column (set dataset_pv_column)')
            pv_raw = df[pv_col].astype(float).fillna(0.0).values
            pv = pv_raw * float(self.dataset_power_scale)
            if self.dataset_load_kw is None:
                raise ValueError('CSV missing Consumption column and dataset_load_kw not provided')
            load = np.ones_like(pv) * float(self.dataset_load_kw)
            time_col = self.dataset_time_column or ('timestamp' if 'timestamp' in df.columns else None)

        # Build price time series using 2026 Taipower TOU rates (TWD/kWh)
        N = int(len(load))
        price = np.ones(N, dtype=float) * 2.06  # default off-peak
        hours = np.zeros(N, dtype=int)
        dows = np.zeros(N, dtype=int)
        try:
            if time_col and time_col in df.columns:
                t = pd.to_datetime(df[time_col], errors='coerce')
                hours = t.dt.hour.fillna(0).astype(int).values
                # day_of_week: Monday=0 … Sunday=6
                dows = t.dt.dayofweek.fillna(0).astype(int).values
                for i in range(N):
                    is_wknd = (dows[i] >= 5)
                    price[i] = get_taipower_tou_price(int(hours[i]), is_wknd)
                print(f"  TOU pricing applied: Off={2.06}, Mid={4.69}, Peak={7.13} TWD/kWh")
            else:
                # fallback: build TOU from step index (assume weekday)
                steps_per_hour = max(1, int(round(1.0 / max(self.time_step, 1e-9)))) if self.time_step < 1.0 else 1
                for i in range(N):
                    h = int((i // steps_per_hour) % 24)
                    day = int((i // (steps_per_hour * 24)) % 7)
                    hours[i] = h
                    dows[i] = day
                    price[i] = get_taipower_tou_price(h, day >= 5)
        except Exception as e:
            print(f"  Warning: TOU pricing fallback to flat 2.06 TWD/kWh: {e}")
            price = np.ones(N, dtype=float) * 2.06
        # Assign series
        self.load_data  = np.asarray(load,  dtype=float)
        self.pv_data    = np.asarray(pv,    dtype=float)
        self.price_data = np.asarray(price, dtype=float)
        self.hour_data  = np.asarray(hours, dtype=int)
        self.dow_data   = np.asarray(dows, dtype=int)
        self._valid_episode_start_indices = None
        self._valid_episode_window_count = 0
        if 'training_window_id' in df.columns and int(self.episode_length) > 0:
            starts: list[int] = []
            window_count = 0
            for _, group in df.groupby('training_window_id', sort=False):
                idx = np.asarray(group.index, dtype=int)
                if len(idx) < int(self.episode_length):
                    continue
                contiguous = np.all(np.diff(idx) == 1)
                if not contiguous:
                    continue
                window_count += 1
                first = int(idx[0])
                last_start = int(idx[-1]) - int(self.episode_length) + 1
                starts.extend(range(first, last_start + 1))
            if starts:
                self._valid_episode_start_indices = np.asarray(starts, dtype=int)
                self._valid_episode_window_count = window_count
                print(
                    f"  Window-bounded episode starts: {len(starts)} "
                    f"across {window_count} training_window_id groups"
                )

        def _try_col(col_name_hint: Optional[str], fallback_cols: list) -> Optional[np.ndarray]:
            """Documentation for this public API is provided in English."""
            candidates = ([col_name_hint] if col_name_hint else []) + fallback_cols
            for c in candidates:
                if c and c in df.columns:
                    return np.asarray(df[c].astype(float).fillna(0.0).values, dtype=float)
            return None

        self.pv_std_data   = _try_col(self.dataset_pv_std_column,   ['pv_std'])
        self.pv_max_data   = _try_col(self.dataset_pv_max_column,   ['pv_max'])
        self.load_std_data = _try_col(self.dataset_load_std_column, ['load_std'])
        self.load_max_data = _try_col(self.dataset_load_max_column, ['load_max'])
        self.soh_data      = _try_col(self.dataset_soh_column,      ['soh_mean', 'battery_soh'])
        self.flow_rate_data= _try_col(self.dataset_flow_rate_column,['flow_rate_mean', 'flow_rate_lpm'])
        # Legacy PV_bool columns are intentionally ignored.
        # pv_bool is now regenerated from pv/load at reset time so that the
        # observation, teacher, and validation plots all follow one rule.
        self.pv_bool_data = None
    
    def _generate_synthetic_data(self):
        """Documentation for this public API is provided in English."""
        print(f"Generating synthetic microgrid data for {self.episode_length} steps...")

        if self.synthetic_hourly_hold and self.time_step < 1.0:
            steps_per_hour = max(1, int(round(1.0 / max(self.time_step, 1e-9))))
            hours = np.arange(24, dtype=float)
            pv_hourly = np.zeros_like(hours)
            start = int(self.synthetic_pv_start_hour)
            end = int(self.synthetic_pv_end_hour)
            if end > start:
                x = (hours - start) / max(1.0, (end - start))
                pv_hourly = self.synthetic_pv_peak_kw * np.sin(np.pi * np.clip(x, 0.0, 1.0))
                pv_hourly[(hours < start) | (hours > end)] = 0.0
            load_hourly = self.synthetic_load_base_kw + self.synthetic_load_amp_kw * (1.0 + np.sin(2 * np.pi * (hours - 7) / 24.0)) / 2.0
            price_hourly = np.ones_like(hours) * self.synthetic_price_base
            peak_mask = (hours >= self.synthetic_price_peak_start) & (hours <= self.synthetic_price_peak_end)
            price_hourly[peak_mask] = self.synthetic_price_peak

            pv_series = np.repeat(pv_hourly, steps_per_hour)
            load_series = np.repeat(load_hourly, steps_per_hour)
            price_series = np.repeat(price_hourly, steps_per_hour)

            total_steps = int(self.episode_length)
            if total_steps > len(pv_series):
                reps = int(np.ceil(total_steps / len(pv_series)))
                pv_series = np.tile(pv_series, reps)[:total_steps]
                load_series = np.tile(load_series, reps)[:total_steps]
                price_series = np.tile(price_series, reps)[:total_steps]
            else:
                pv_series = pv_series[:total_steps]
                load_series = load_series[:total_steps]
                price_series = price_series[:total_steps]

            self.load_data = np.maximum(load_series, 0.0)
            self.pv_data = np.maximum(pv_series, 0.0)
            self.price_data = np.maximum(price_series, 0.05)
            self._refresh_time_arrays()
            print("Synthetic hourly-hold data generated")
            return
        
        # Generate longer time series based on episode length
        if self.episode_length <= 24:
            # Daily pattern (24 hours)
            hours = np.arange(24)
            base_load = 30.0  # kW
            load_pattern = base_load + 20 * np.sin(2 * np.pi * (hours - 6) / 24) + 10 * np.random.randn(24)
            self.load_data = np.maximum(load_pattern, 5.0)  # Minimum 5 kW
            
            # Generate PV profile (solar pattern)
            solar_pattern = 40 * np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) + 5 * np.random.randn(24)
            self.pv_data = np.maximum(solar_pattern, 0.0)  # No negative PV
            
            # Generate price profile (time-of-use pricing)
            base_price = 0.15  # $/kWh
            price_pattern = base_price + 0.05 * np.sin(2 * np.pi * (hours - 12) / 24) + 0.02 * np.random.randn(24)
            self.price_data = np.maximum(price_pattern, 0.05)  # Minimum 5 cents/kWh
            
        elif self.episode_length <= 720:
            # Monthly pattern (30 days × 24 hours)
            self._generate_monthly_data()
        else:
            # Yearly pattern (365 days × 24 hours)
            self._generate_yearly_data()
        
        self._refresh_time_arrays()
        print("Synthetic data generated")
    
    def _generate_monthly_data(self):
        """Documentation for this public API is provided in English."""
        days = np.arange(30)
        hours = np.arange(24)
        
        # Base load with weekly pattern
        base_load = 30.0  # kW
        weekly_pattern = base_load + 5 * np.sin(2 * np.pi * days / 7)  # Weekly variation
        
        # Daily load pattern
        daily_pattern = 20 * np.sin(2 * np.pi * (hours - 6) / 24)  # Daily variation
        
        # Combine patterns
        load_data = np.zeros(720)
        for day in range(30):
            for hour in range(24):
                idx = day * 24 + hour
                base = weekly_pattern[day]
                daily = daily_pattern[hour]
                noise = 8 * np.random.randn()
                load_data[idx] = base + daily + noise
        
        self.load_data = np.maximum(load_data, 5.0)  # Minimum 5 kW
        
        # PV data with seasonal variation
        pv_data = np.zeros(720)
        for day in range(30):
            for hour in range(24):
                idx = day * 24 + hour
                # Seasonal factor (assuming month 1-12)
                seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day / 30 - 0.5))  # Peak in summer
                # Daily solar pattern
                solar_hour = hour - 6  # Solar noon at 12
                if 0 <= solar_hour <= 12:
                    solar_intensity = 40 * np.sin(np.pi * solar_hour / 12) * seasonal_factor
                else:
                    solar_intensity = 0
                
                pv_data[idx] = max(0, solar_intensity + 3 * np.random.randn())
        
        self.pv_data = pv_data
        
        # Price data with weekly and daily patterns
        price_data = np.zeros(720)
        base_price = 0.15  # $/kWh
        
        for day in range(30):
            for hour in range(24):
                idx = day * 24 + hour
                # Weekly pattern (weekend vs weekday)
                is_weekend = (day % 7) >= 5
                weekly_factor = 0.9 if is_weekend else 1.1
                
                # Daily pattern (peak hours)
                is_peak_hour = 8 <= hour <= 18
                daily_factor = 1.2 if is_peak_hour else 0.8
                
                # Base price with variations
                price = base_price * weekly_factor * daily_factor
                price += 0.02 * np.random.randn()  # Random noise
                price_data[idx] = max(0.05, price)
        
        self.price_data = price_data
    
    def _generate_yearly_data(self):
        """Documentation for this public API is provided in English."""
        days = np.arange(365)
        hours = np.arange(24)
        
        # Base load with seasonal and weekly patterns
        base_load = 30.0  # kW
        
        # Seasonal pattern (winter vs summer)
        seasonal_pattern = base_load + 15 * np.sin(2 * np.pi * (days - 172) / 365)  # Peak in summer
        
        # Weekly pattern
        weekly_pattern = 5 * np.sin(2 * np.pi * days / 7)  # Weekly variation
        
        # Daily load pattern
        daily_pattern = 20 * np.sin(2 * np.pi * (hours - 6) / 24)  # Daily variation
        
        # Combine all patterns
        load_data = np.zeros(8760)
        for day in range(365):
            for hour in range(24):
                idx = day * 24 + hour
                seasonal = seasonal_pattern[day]
                weekly = weekly_pattern[day]
                daily = daily_pattern[hour]
                noise = 10 * np.random.randn()
                load_data[idx] = seasonal + weekly + daily + noise
        
        self.load_data = np.maximum(load_data, 5.0)  # Minimum 5 kW
        
        # PV data with strong seasonal variation
        pv_data = np.zeros(8760)
        for day in range(365):
            for hour in range(24):
                idx = day * 24 + hour
                # Strong seasonal factor
                seasonal_factor = 0.3 + 0.7 * np.sin(2 * np.pi * (days[day] - 172) / 365)  # Peak in summer
                
                # Daily solar pattern
                solar_hour = hour - 6  # Solar noon at 12
                if 0 <= solar_hour <= 12:
                    solar_intensity = 50 * np.sin(np.pi * solar_hour / 12) * seasonal_factor
                else:
                    solar_intensity = 0
                
                pv_data[idx] = max(0, solar_intensity + 5 * np.random.randn())
        
        self.pv_data = pv_data
        
        # Price data with seasonal, weekly, and daily patterns
        price_data = np.zeros(8760)
        base_price = 0.15  # $/kWh
        
        for day in range(365):
            for hour in range(24):
                idx = day * 24 + hour
                
                # Seasonal pattern (higher prices in winter)
                seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (days[day] - 172) / 365)
                
                # Weekly pattern (weekend vs weekday)
                is_weekend = (day % 7) >= 5
                weekly_factor = 0.9 if is_weekend else 1.1
                
                # Daily pattern (peak hours)
                is_peak_hour = 8 <= hour <= 18
                daily_factor = 1.3 if is_peak_hour else 0.7
                
                # Base price with all variations
                price = base_price * seasonal_factor * weekly_factor * daily_factor
                price += 0.03 * np.random.randn()  # Random noise
                price_data[idx] = max(0.05, price)
        
        self.price_data = price_data
    
    def _generate_synthetic_prices(self):
        """Documentation for this public API is provided in English."""
        hours = np.arange(24)
        price_pattern = np.array([get_taipower_tou_price(int(h), False) for h in hours])
        return price_pattern

    def _refresh_time_arrays(self):
        """Documentation for this public API is provided in English."""
        try:
            total_len = int(min(
                len(self.load_data) if self.load_data is not None else 0,
                len(self.pv_data) if self.pv_data is not None else 0,
                len(self.price_data) if self.price_data is not None else 0,
            ))
        except Exception:
            total_len = 0
        if total_len <= 0:
            self.hour_data = None
            self.dow_data = None
            return
        if self.hour_data is not None and self.dow_data is not None and len(self.hour_data) == total_len and len(self.dow_data) == total_len:
            return
        steps_per_hour = max(1, int(round(1.0 / max(self.time_step, 1e-9)))) if self.time_step < 1.0 else 1
        idx = np.arange(total_len, dtype=int)
        self.hour_data = ((idx // steps_per_hour) % 24).astype(int)
        self.dow_data = ((idx // (steps_per_hour * 24)) % 7).astype(int)
    
    def _setup_spaces(self):
        """Documentation for this public API is provided in English.
        

          [SoC, SoH, flow_rate_norm,
           pv_mean, pv_std, pv_max,
           load_mean, load_std, load_max,
           price_norm, hour, day_of_week,
           energy_pv_kwh_norm, energy_load_kwh_norm]
        """
        if self.use_extended_obs:
            #        3=pv_mean, 4=pv_std, 5=pv_max,
            #        6=load_mean, 7=load_std, 8=load_max,
            #        9=price_norm, 10=hour, 11=dow,
            #        12=energy_pv_norm, 13=energy_load_norm
            pw = max(self.battery_charge_power_kw, self.battery_discharge_power_kw) * 2
            self.observation_space = spaces.Box(
                low=np.array([
                    0.0, 0.0, 0.0,           # SoC, SoH, flow_norm
                    0.0, 0.0, 0.0,           # pv_mean, pv_std, pv_max
                    0.0, 0.0, 0.0,           # load_mean, load_std, load_max
                    0.0, 0.0, 0.0,           # price_norm, hour, dow
                    0.0, 0.0,                # energy_pv_norm, energy_load_norm
                ], dtype=np.float32),
                high=np.array([
                    1.0,  1.0,  1.0,         # SoC, SoH, flow_norm
                    pw,   pw,   pw,           # pv_mean, pv_std, pv_max
                    pw,   pw,   pw,           # load_mean, load_std, load_max
                    1.0,  23.0, 6.0,          # price_norm, hour, dow
                    1.0,  1.0,
                ], dtype=np.float32)
            )
        else:
            if self.pv_support_ratio_obs:
                if self.tou_onehot_obs:
                    low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], dtype=np.float32)
                    high = np.array([1.0, 100.0, self.pv_support_ratio_max, 1.0, 1.0, 1.0, 1.0, 23, 6], dtype=np.float32)
                elif self.price_obs:
                    low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], dtype=np.float32)
                    high = np.array([1.0, 100.0, self.pv_support_ratio_max, 1.0, 1.0, 23, 6], dtype=np.float32)
                else:
                    low = np.array([0.0, 0.0, 0.0, 0.0, 0, 0], dtype=np.float32)
                    high = np.array([1.0, 100.0, self.pv_support_ratio_max, 1.0, 23, 6], dtype=np.float32)
                self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            else:
                pv_high = 1.0 if self.pv_obs_boolean else 50.0
                if self.tou_onehot_obs:
                    low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0], dtype=np.float32)
                    high = np.array([1.0, 100.0, pv_high, 1.0, 1.0, 1.0, 23, 6], dtype=np.float32)
                elif self.price_obs:
                    low = np.array([0.0, 0.0, 0.0, 0.0, 0, 0], dtype=np.float32)
                    high = np.array([1.0, 100.0, pv_high, 1.0, 23, 6], dtype=np.float32)
                else:
                    low = np.array([0.0, 0.0, 0.0, 0, 0], dtype=np.float32)
                    high = np.array([1.0, 100.0, pv_high, 23, 6], dtype=np.float32)
                self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Action space
        if self.use_flow_rate_action:
            self.action_space = spaces.Box(
                low=np.array([-self.battery_discharge_power_kw, 0.00], dtype=np.float32),
                high=np.array([ self.battery_charge_power_kw, 1.00], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-self.battery_discharge_power_kw]),
                high=np.array([self.battery_charge_power_kw]),
                dtype=np.float32
            )

    def _compute_pv_support_ratio(self, load: float, pv: float) -> float:
        return float(np.clip(
            float(pv) / max(float(load), 1e-9),
            0.0,
            self.pv_support_ratio_max,
        ))

    def _compute_pv_bool(self, load: float, pv: float) -> float:
        pv_support_ratio = float(pv) / max(float(load), 1e-9)
        return float(1.0 if pv_support_ratio >= self.pv_sufficient_ratio_threshold else 0.0)

    def _compute_pv_active(self, pv: float) -> float:
        return float(1.0 if float(pv) > self.pv_obs_boolean_threshold_kw else 0.0)

    def _refresh_episode_pv_bool(self) -> None:
        if self.episode_data is None:
            return
        load_arr = np.asarray(self.episode_data.get('load', []), dtype=float)
        pv_arr = np.asarray(self.episode_data.get('pv', []), dtype=float)
        n = int(min(len(load_arr), len(pv_arr)))
        if n <= 0:
            self.episode_data['pv_bool'] = np.asarray([], dtype=float)
            return
        pv_ratio = np.divide(
            pv_arr[:n],
            np.clip(load_arr[:n], 1e-9, None),
            out=np.zeros(n, dtype=float),
            where=np.ones(n, dtype=bool),
        )
        self.episode_data['pv_bool'] = (pv_ratio >= self.pv_sufficient_ratio_threshold).astype(float)

    def _get_schedule_load_groups(self, hour: int) -> int:
        hour = int(np.clip(hour, 0, 23))
        if hour < 6:
            return 1
        if hour < 8:
            return 2
        if hour < 14:
            return 3
        if hour < 17:
            return 4
        if hour < 20:
            return 2
        return 1

    def _get_battery_delivered_load_kw(self, load_kw: float, load_groups: int) -> float:
        """Return the load scale seen when the battery is the solo supplier."""
        if self.battery_delivered_load_per_group_kw is None:
            return float(max(0.0, load_kw))
        groups = int(max(0, load_groups))
        return float(groups) * float(self.battery_delivered_load_per_group_kw)

    def _get_obs_hour_dow(self, step: int) -> tuple[int, int]:
        if (
            self.use_dataset_timestamps_for_obs
            and self.episode_data is not None
            and 'hour' in self.episode_data
            and 'dow' in self.episode_data
            and step < len(self.episode_data['hour'])
            and step < len(self.episode_data['dow'])
        ):
            return int(self.episode_data['hour'][step]), int(self.episode_data['dow'][step])
        steps_per_hour = max(1, int(round(1.0 / max(self.time_step, 1e-9)))) if self.time_step < 1.0 else 1
        current_hour = int((step // steps_per_hour) % 24)
        current_day = int((step // (steps_per_hour * 24)) % 7)
        return current_hour, current_day

    def _build_observation_snapshot(self, step: int) -> Dict[str, float]:
        if self.episode_data is not None:
            load_series = self.episode_data.get('load', [])
            pv_series = self.episode_data.get('pv', [])
            price_series = self.episode_data.get('price', [])
            load_now = float(load_series[step]) if step < len(load_series) else 0.0
            pv_now = float(pv_series[step]) if step < len(pv_series) else 0.0
            price_now = float(price_series[step]) if step < len(price_series) else 2.06
        else:
            h = step % 24
            load_now = float(self.load_data[h]) if self.load_data is not None else 30.0
            pv_now = float(self.pv_data[h]) if self.pv_data is not None else 20.0
            price_now = float(self.price_data[h]) if self.price_data is not None else 0.15

        hour, dow = self._get_obs_hour_dow(step)
        load_source = 'measured'
        load_groups = self._get_schedule_load_groups(hour)

        if not self.deployment_observation_style or self.episode_data is None:
            return {
                'load_kw': load_now,
                'pv_kw': pv_now,
                'price': price_now,
                'hour': hour,
                'dow': dow,
                'load_source': load_source,
                'load_groups': load_groups,
            }

        window = max(1, int(self.deployment_window_steps))
        start = max(0, step - window + 1)
        load_window = np.asarray(self.episode_data.get('load', [load_now])[start:step + 1], dtype=float)
        pv_window = np.asarray(self.episode_data.get('pv', [pv_now])[start:step + 1], dtype=float)
        load_measured_kw = float(np.mean(load_window)) if load_window.size > 0 else load_now
        pv_measured_kw = float(np.mean(pv_window)) if pv_window.size > 0 else pv_now
        load_kw = load_measured_kw
        if load_measured_kw < self.deployment_load_threshold_kw:
            load_kw = float(load_groups) * self.deployment_group_power_kw
            load_source = 'schedule_fallback'
        return {
            'load_kw': load_kw,
            'pv_kw': pv_measured_kw,
            'price': price_now,
            'hour': hour,
            'dow': dow,
            'load_source': load_source,
            'load_groups': load_groups,
        }

    def _apply_deployment_style_guards(
        self,
        action_kw: float,
        pv_kw: float,
        load_kw: float,
        battery_load_kw: float,
        current_day: int,
    ) -> tuple[float, Dict[str, int]]:
        flags = {
            'guard_force_charge_low_soc': 0,
            'guard_block_low_soc_discharge': 0,
            'guard_block_high_soc_charge': 0,
            'guard_block_pv_active_discharge': 0,
            'guard_block_voltage_cutoff': 0,
            'guard_block_load_over_discharge_limit': 0,
        }
        if not self.deployment_guard_style:
            return action_kw, flags

        if self._deployment_guard_day is None or int(current_day) != int(self._deployment_guard_day):
            self._deployment_guard_day = int(current_day)
            self._deployment_voltage_cutoff_active = False

        guarded_action_kw = float(action_kw)
        if self.current_soc <= self.soc_min and guarded_action_kw < 0:
            guarded_action_kw = 0.0
            flags['guard_block_low_soc_discharge'] = 1
        elif self.current_soc >= self.soc_max and guarded_action_kw > 0:
            guarded_action_kw = 0.0
            flags['guard_block_high_soc_charge'] = 1

        if (
            guarded_action_kw < 0
            and self.discharge_mode == "solo_only"
            and self.enforce_solo_discharge_load_limit
            and float(battery_load_kw) > self.battery_discharge_power_kw
        ):
            guarded_action_kw = 0.0
            flags['guard_block_load_over_discharge_limit'] = 1

        if self._compute_pv_active(pv_kw) > 0.5 and guarded_action_kw < 0:
            guarded_action_kw = 0.0
            flags['guard_block_pv_active_discharge'] = 1

        if self.voltage_cutoff_soc > 0.0 and self.current_soc <= self.voltage_cutoff_soc and guarded_action_kw < 0:
            self._deployment_voltage_cutoff_active = True
        if self._deployment_voltage_cutoff_active and guarded_action_kw < 0:
            guarded_action_kw = 0.0
            flags['guard_block_voltage_cutoff'] = 1

        return guarded_action_kw, flags
    
    def _get_state(self) -> np.ndarray:
        """Documentation for this public API is provided in English."""
        step = self.current_step

        snapshot = self._build_observation_snapshot(step)
        load = float(snapshot['load_kw'])
        pv = float(snapshot['pv_kw'])
        price = float(snapshot['price'])

        if self.stress_enable and self.stress_soc_obs_delay > 0 and len(self._soc_obs_buffer) > 0:
            soc_obs = float(self._soc_obs_buffer[0])
        else:
            soc_obs = float(self.current_soc)
        if self.stress_enable and self.stress_soc_obs_noise_std > 0.0:
            soc_obs += float(np.random.randn()) * self.stress_soc_obs_noise_std
            soc_obs = float(np.clip(soc_obs, 0.0, 1.0))

        current_hour = int(snapshot['hour'])
        current_day = int(snapshot['dow'])

        price_norm = float(np.clip(price / 10.0, 0.0, 1.0))
        tou_offpeak = float(np.isclose(price, 2.06, atol=1e-6))
        tou_midpeak = float(np.isclose(price, 4.69, atol=1e-6))
        tou_peak = float(np.isclose(price, 7.13, atol=1e-6))

        pv_bool = self._compute_pv_bool(load, pv)

        if not self.use_extended_obs:
            if self.pv_support_ratio_obs:
                pv_support_ratio = self._compute_pv_support_ratio(load, pv)
                if self.tou_onehot_obs:
                    return np.array([
                        soc_obs,          # SoC
                        load,
                        pv_support_ratio,
                        pv_bool,          # PV sufficient boolean
                        tou_offpeak,
                        tou_midpeak,
                        tou_peak,
                        current_hour,
                        current_day,
                    ], dtype=np.float32)
                if self.price_obs:
                    return np.array([
                        soc_obs,          # SoC
                        load,
                        pv_support_ratio,
                        pv_bool,          # PV sufficient boolean
                        price_norm,
                        current_hour,
                        current_day,
                    ], dtype=np.float32)
                return np.array([
                    soc_obs,          # SoC
                    load,
                    pv_support_ratio,
                    pv_bool,          # PV sufficient boolean
                    current_hour,
                    current_day,
                ], dtype=np.float32)

            if self.pv_obs_boolean:
                pv_obs = pv_bool
            else:
                pv_obs = pv
            if self.tou_onehot_obs:
                return np.array([
                    soc_obs,        # SoC
                    load,
                    pv_obs,         # PV: boolean(0/1) or kW
                    tou_offpeak,
                    tou_midpeak,
                    tou_peak,
                    current_hour,
                    current_day,
                ], dtype=np.float32)
            if self.price_obs:
                return np.array([
                    soc_obs,        # SoC
                    load,
                    pv_obs,         # PV: boolean(0/1) or kW
                    price_norm,
                    current_hour,
                    current_day,
                ], dtype=np.float32)
            return np.array([
                soc_obs,        # SoC
                load,
                pv_obs,         # PV: boolean(0/1) or kW
                current_hour,
                current_day,
            ], dtype=np.float32)

        ep_pv_std  = self.episode_data.get('pv_std',  None) if self.episode_data else None
        ep_pv_max  = self.episode_data.get('pv_max',  None) if self.episode_data else None
        ep_ld_std  = self.episode_data.get('load_std',None) if self.episode_data else None
        ep_ld_max  = self.episode_data.get('load_max',None) if self.episode_data else None

        pv_std  = float(ep_pv_std[step])  if ep_pv_std  is not None and step < len(ep_pv_std)  else float(pv * 0.15)
        pv_max  = float(ep_pv_max[step])  if ep_pv_max  is not None and step < len(ep_pv_max)  else float(pv * 1.20)
        ld_std  = float(ep_ld_std[step])  if ep_ld_std  is not None and step < len(ep_ld_std)  else float(load * 0.10)
        ld_max  = float(ep_ld_max[step])  if ep_ld_max  is not None and step < len(ep_ld_max)  else float(load * 1.15)

        ep_soh = self.episode_data.get('soh', None) if self.episode_data else None
        if ep_soh is not None and step < len(ep_soh):
            soh_obs = float(ep_soh[step])
        else:
            soh_obs = float(np.clip(self.current_soh, 0.0, 1.0))

        if self.use_flow_rate_action:
            flow_norm = float(np.clip(self._current_flow_action, 0.0, 1.0))
        else:
            ep_flow = self.episode_data.get('flow_rate', None) if self.episode_data else None
            if ep_flow is not None and step < len(ep_flow):
                flow_raw = float(ep_flow[step])
            else:
                flow_raw = float(self.current_flow_rate_lpm)
            flow_norm = float(np.clip(flow_raw / 20.0, 0.0, 1.0))

        dt_h = float(self.time_step)
        energy_pv_kwh   = pv   * dt_h
        energy_load_kwh = load * dt_h
        cap = max(self.battery_capacity_kwh, 1e-6)
        energy_pv_norm   = float(np.clip(energy_pv_kwh   / cap, 0.0, 1.0))
        energy_load_norm = float(np.clip(energy_load_kwh / cap, 0.0, 1.0))

        return np.array([
            soc_obs,          # 0  SoC
            soh_obs,
            flow_norm,
            pv,
            pv_std,
            pv_max,
            load,
            ld_std,
            ld_max,
            price_norm,
            float(current_hour),
            float(current_day),
            energy_pv_norm,
            energy_load_norm,
        ], dtype=np.float32)
    
    # ══════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════
    def _pump_power_W(self, Q: float) -> float:
        """Documentation for this public API is provided in English.
        
        P_pump(Q) = P_max × Q³
        
        Args:
        Returns:
        """
        Q = float(np.clip(Q, 0.01, 1.0))
        return self.flow_P_max_pump_W * (Q ** 3)

    def _apply_flow_operating_rule(self, requested_q: float, active: bool) -> float:
        """Map a requested flow fraction to the hardware-safe operating range."""
        requested_q = float(np.clip(requested_q, 0.0, 1.0))
        if not self.flow_operating_rule_enabled:
            return requested_q
        if not active:
            return self.flow_idle_fraction
        return float(np.clip(max(requested_q, self.flow_min_active_fraction), 0.0, 1.0))
    
    def _equivalent_resistance(self, Q: float) -> float:
        """Documentation for this public API is provided in English.
        
        R_eq(Q) = R_base × (1 + k_R × (1 - Q) / Q)
        
        
        Args:
        Returns:
        """
        Q = float(np.clip(Q, 0.01, 1.0))
        return self.flow_R_base_ohm * (1.0 + self.flow_k_R * (1.0 - Q) / Q)
    
    def _cell_voltage(self, Q: float, is_charging: bool) -> float:
        """Documentation for this public API is provided in English.
        
        
        Args:
        Returns:
        """
        R_eq = self._equivalent_resistance(Q)
        I = self.flow_I_rated_A
        if is_charging:
            return self.flow_V_OCV_charge + I * R_eq
        else:
            return max(0.0, self.flow_V_OCV_discharge - I * R_eq)
    
    def _flow_efficiency(self, Q: float, is_charging: bool) -> float:
        """Documentation for this public API is provided in English.
        
        
        Args:
        Returns:
        """
        R_eq = self._equivalent_resistance(Q)
        I = self.flow_I_rated_A
        if is_charging:
            V_cell = self.flow_V_OCV_charge + I * R_eq
            return float(np.clip(self.flow_V_OCV_charge / max(V_cell, 1e-6), 0.01, 1.0))
        else:
            V_cell = max(0.0, self.flow_V_OCV_discharge - I * R_eq)
            return float(np.clip(V_cell / max(self.flow_V_OCV_discharge, 1e-6), 0.01, 1.0))
    
    def _net_power_W(self, gross_power_W: float, Q: float) -> float:
        """Documentation for this public API is provided in English.
        
        P_net = P_gross - P_pump(Q)
        
        Args:
        Returns:
        """
        if self.flow_pump_from_grid:
            return gross_power_W
        P_pump = self._pump_power_W(Q)
        if gross_power_W >= 0:
            return gross_power_W - P_pump
        else:
            return gross_power_W + P_pump

    def _flow_limited_power_bounds(self, Q: float) -> tuple[float, float]:
        """Return charge/discharge limits after flow-rate availability scaling."""
        if not self.flow_limits_available_power:
            return self.battery_charge_power_kw, self.battery_discharge_power_kw
        q_eff = float(np.clip(Q, self.flow_power_min_fraction, 1.0))
        return self.battery_charge_power_kw * q_eff, self.battery_discharge_power_kw * q_eff

    def _update_battery_soc(self, action: float):
        """Documentation for this public API is provided in English.
        
        """
        action = self._clip_action_by_direction(float(action))
        # Convert action from kW to kWh using effective dt
        dt_eff = float(getattr(self, '_effective_time_step', self.time_step))
        eta_eff = float(getattr(self, '_effective_efficiency', self.battery_efficiency))
        
        energy_change_kwh = action * dt_eff
        # Apply efficiency
        if energy_change_kwh > 0:  # Charging
            energy_change_kwh *= eta_eff
        else:  # Discharging
            eta_safe = eta_eff if eta_eff > 1e-9 else self.battery_efficiency
            energy_change_kwh /= eta_safe
        
        # Update SoC
        new_soc = self.current_soc + energy_change_kwh / self.battery_capacity_kwh
        
        low = float(getattr(self, 'soc_min_eff', self.soc_min))
        high = float(getattr(self, 'soc_max_eff', self.soc_max))
        actual_action_kw = action
        
        if new_soc < low:
            self.soc_violations += 1
            if self.clip_soc_to_bounds:
                actual_energy_kwh = (low - self.current_soc) * self.battery_capacity_kwh
                if dt_eff > 1e-9:
                    actual_action_kw = actual_energy_kwh * eta_eff / dt_eff
                new_soc = low
        elif new_soc > high:
            self.soc_violations += 1
            if self.clip_soc_to_bounds:
                actual_energy_kwh = (high - self.current_soc) * self.battery_capacity_kwh
                if dt_eff > 1e-9 and eta_eff > 1e-9:
                    actual_action_kw = actual_energy_kwh / (dt_eff * eta_eff)
                new_soc = high

        # Hard physical SoC floor (lower side only). Applied independently of the
        # operational clip so that strict-band (e.g. 20-80%) violations are still
        # counted, but the battery can never discharge below the physical floor.
        # The commanded discharge is truncated to the energy available down to
        # the floor: actual_action_kw is recomputed so SoC lands exactly on the
        # floor, and the resulting discharge shortfall is served by the grid in
        # the step's downstream import/cost accounting.
        pfloor = getattr(self, 'soc_physical_floor', None)
        if pfloor is not None and action < 0.0 and new_soc < float(pfloor):
            actual_energy_kwh = (float(pfloor) - self.current_soc) * self.battery_capacity_kwh
            if dt_eff > 1e-9:
                actual_action_kw = actual_energy_kwh * eta_eff / dt_eff
            new_soc = float(pfloor)

        return new_soc, self._clip_action_by_direction(actual_action_kw)

    def _soc_violation_excess_kwh(self, soc: float) -> float:
        """Return how far SoC sits outside the active bounds, expressed as kWh."""
        low = float(getattr(self, 'soc_min_eff', self.soc_min))
        high = float(getattr(self, 'soc_max_eff', self.soc_max))
        if soc < low:
            return float((low - soc) * self.battery_capacity_kwh)
        if soc > high:
            return float((soc - high) * self.battery_capacity_kwh)
        return 0.0

    def predict_soc_raw(self, soc: float, action: float) -> float:
        """Documentation for this public API is provided in English.
        Args:
        Returns:
        """
        energy_change_kwh = action * self.time_step
        if energy_change_kwh > 0:
            energy_change_kwh *= self.battery_efficiency
        else:
            energy_change_kwh /= self.battery_efficiency
        return soc + energy_change_kwh / self.battery_capacity_kwh

    def _clip_action_by_direction(self, action_kw: float) -> float:
        """Clip positive / negative power using asymmetric charge/discharge limits."""
        return float(np.clip(action_kw, -self.battery_discharge_power_kw, self.battery_charge_power_kw))
    
    def _calculate_reward(self, action: float, net_load: float, price: float) -> float:
        """Documentation for this public API is provided in English."""
        reward = 0.0
        
        dt = self.time_step
        if action < 0:  # Discharging (selling energy)
            reward += abs(action) * dt * price
        else:  # Charging (buying energy)
            reward -= action * dt * price

        reward -= 0.02 * abs(action) * dt
        
        net_load_penalty = -0.05 * (abs(net_load) / 100.0) ** 2
        reward += net_load_penalty
        
        soc_bonus_low, soc_bonus_high = 0.3, 0.8
        if soc_bonus_low <= self.current_soc <= soc_bonus_high:
            reward += 0.05
        
        if self.current_soc < self.soc_min or self.current_soc > self.soc_max:
            reward -= 2.0
        
        price_normalized = (price - 0.05) / 0.45
        if price_normalized > 0.5 and action < 0:
            reward += 0.5
        elif price_normalized < 0.3 and action > 0:
            reward += 0.3
        
        action_smoothness_penalty = -0.001 * (abs(action) / self.battery_power_kw) ** 2
        reward += action_smoothness_penalty
        
        if self.episode_length > 24:
            time_factor = min(1.0, self.current_step / (self.episode_length * 0.1))
            if time_factor < 1.0:
                if abs(action) < self.battery_power_kw * 0.3:
                    reward += 0.1 * (1.0 - time_factor)
                else:
                    reward -= 0.05 * (1.0 - time_factor)
        
        if self.episode_length > 720:
            day_of_year = self.current_step // 24
            season = (day_of_year // 91) % 4
            
            if season == 1:
                if action > 0 and self.current_soc < 0.7:
                    reward += 0.2
            elif season == 3:
                if action < 0 and self.current_soc > 0.3:
                    reward += 0.1
        
        return reward * self.reward_scaling
    
    def _calculate_reward_phase1(self, action: float, net_load: float, price: float) -> float:
        """Documentation for this public API is provided in English."""
        reward = 0.0
        
        dt = self.time_step
        if action < 0:  # Discharging (selling energy)
            reward += abs(action) * dt * price
        else:  # Charging (buying energy)
            reward -= action * dt * price

        reward -= 0.02 * abs(action) * dt
        
        net_load_penalty = -0.05 * (abs(net_load) / 100.0) ** 2
        reward += net_load_penalty
        
        soc_bonus_low, soc_bonus_high = 0.3, 0.8
        if soc_bonus_low <= self.current_soc <= soc_bonus_high:
            reward += 0.05
        
        
        # 5. Action efficiency reward
        price_normalized = (price - 0.05) / 0.45
        if price_normalized > 0.5 and action < 0:
            reward += 0.5
        elif price_normalized < 0.3 and action > 0:
            reward += 0.3
        
        # 6. Action smoothness penalty
        action_smoothness_penalty = -0.001 * (abs(action) / self.battery_power_kw) ** 2
        reward += action_smoothness_penalty
        
        # 7. Long-term planning reward
        if self.episode_length > 24:
            time_factor = min(1.0, self.current_step / (self.episode_length * 0.1))
            if time_factor < 1.0:
                if abs(action) < self.battery_power_kw * 0.3:
                    reward += 0.1 * (1.0 - time_factor)
                else:
                    reward -= 0.05 * (1.0 - time_factor)
        
        # 8. Seasonal awareness reward
        if self.episode_length > 720:
            day_of_year = self.current_step // 24
            season = (day_of_year // 91) % 4
            
            if season == 1:
                if action > 0 and self.current_soc < 0.7:
                    reward += 0.2
            elif season == 3:
                if action < 0 and self.current_soc > 0.3:
                    reward += 0.1
        
        return reward * self.reward_scaling
    
    def _calculate_reward_no_grid(self, action: float, net_load: float, load_kw: float, 
                                   pv_kw: float, price: float) -> float:
        """
        
        
        Args:
        """
        reward = 0.0
        dt = self.time_step
        
        unserved_load = max(0.0, net_load)
        if unserved_load > 0:
            unserved_penalty = -10.0 * unserved_load * price * dt
            unserved_penalty -= 5.0 * (unserved_load / max(load_kw, 1e-6)) ** 2 * dt
            reward += unserved_penalty
        
        if pv_kw > 1e-6:
            pv_used = min(pv_kw, load_kw + max(0, action))
            pv_utilization = pv_used / pv_kw
            pv_reward = 0.5 * pv_utilization
            reward += pv_reward
        
        soc_target_range = (0.4, 0.7)
        if soc_target_range[0] <= self.current_soc <= soc_target_range[1]:
            reward += 0.1
        elif self.current_soc < 0.2:
            reward -= 0.05 * (0.2 - self.current_soc) / 0.2
        elif self.current_soc > 0.9:
            reward -= 0.02 * (self.current_soc - 0.9) / 0.1
        
        if self.current_soc < self.soc_min or self.current_soc > self.soc_max:
            reward -= 5.0
        
        steps_per_hour = max(1, int(round(1.0 / max(self.time_step, 1e-9)))) if self.time_step < 1.0 else 1
        hour = int((self.current_step // steps_per_hour) % 24)
        if 8 <= hour <= 16:
            if action > 0 and pv_kw > load_kw * 1.1:
                reward += 0.2
            if action < -0.3 * self.battery_power_kw and pv_kw > load_kw:
                reward -= 0.1
        elif 18 <= hour <= 23 or 0 <= hour <= 6:
            if action < 0 and load_kw > pv_kw * 1.1:
                reward += 0.15
            if action > 0.3 * self.battery_power_kw and pv_kw < load_kw * 0.5:
                reward -= 0.05
        
        if hasattr(self, 'prev_action_kw'):
            action_change = abs(action - self.prev_action_kw)
            smoothness_penalty = -0.001 * (action_change / self.battery_power_kw) ** 2
            reward += smoothness_penalty
        
        if self.current_soc > 0.8 and action > 0.2 * self.battery_power_kw and load_kw < pv_kw * 0.5:
            reward -= 0.05
        if self.current_soc < 0.2 and action < -0.2 * self.battery_power_kw and load_kw < pv_kw * 1.1:
            reward -= 0.1
        
        reward -= 0.01 * abs(action) * dt
        
        return reward * self.reward_scaling
    
    def _calculate_reward_p302(self, action_kw: float, load_kw: float, pv_kw: float,
                               price: float, net_load_after_pv: float,
                               useful_discharge: float, charge_kw: float,
                               grid_kw: float, baseline_grid_kw: float,
                               pump_power_kw: float) -> float:
        """
        P302 SLFB TOU Arbitrage Reward（v7）
        ======================================
        
        
          - ✗ SoC centering reward
          - ✗ Action efficiency / smoothness penalty
          - ✗ Throughput penalty
        
        Args:
        """
        reward = 0.0
        soc = self.current_soc
        pmax = max(self.battery_power_kw, 1e-9)
        dt = self.time_step
        
        steps_per_hour = max(1, int(round(1.0 / max(dt, 1e-9)))) if dt < 1.0 else 1
        hour = int((self.current_step // steps_per_hour) % 24)
        day = int((self.current_step // (steps_per_hour * 24)) % 7)
        is_weekend = (day >= 5)  # Saturday=5, Sunday=6
        is_peak_window = (16 <= hour < 22) and not is_weekend
        is_offpeak = (0 <= hour < 9) and not is_weekend
        
        # ══════════════════════════════════════════════════════════
        #    (Baseline_Cost - Agent_Cost) × tou_reward_scale
        # ══════════════════════════════════════════════════════════
        baseline_cost_twd = baseline_grid_kw * dt * price  # TWD
        agent_cost_twd = grid_kw * dt * price              # TWD
        economic_reward = (baseline_cost_twd - agent_cost_twd) * self.tou_reward_scale
        reward += economic_reward
        
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        if is_peak_window:
            if action_kw < -0.05 * pmax and soc > 0.2:
                reward += 1.0
            elif action_kw > 0.05 * pmax:
                reward -= 1.0
            else:
                reward -= 0.3
        
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        if is_offpeak:
            if action_kw > 0.05 * pmax and soc < 0.75:
                reward += 0.5
            elif action_kw < -0.05 * pmax:
                reward -= 0.3
        
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        pv_excess = max(0.0, pv_kw - load_kw)
        _solar_coeff = float(getattr(self, '_solar_charge_coeff', 0.8))
        _solar_waste_pen = float(getattr(self, '_solar_waste_penalty', 0.4))
        if pv_excess > 0.01 * pmax and action_kw > 0.05 * pmax and soc < 0.85:
            power_frac = min(1.0, action_kw / pmax)
            solar_charge_bonus = _solar_coeff * min(1.0, pv_excess / pmax) * (0.5 + 0.5 * power_frac)
            reward += solar_charge_bonus
        elif pv_excess > 0.1 * pmax and action_kw <= 0 and soc < 0.5:
            reward -= _solar_waste_pen
        
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        if soc < 0.15 and action_kw > 0.05 * pmax:
            reward += 1.0
        elif soc < 0.15 and action_kw < -0.05 * pmax:
            reward -= 2.0
        
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        _power_util_coeff = float(getattr(self, '_power_util_coeff', 0.0))
        if _power_util_coeff > 0:
            power_frac = min(1.0, abs(action_kw) / pmax)
            if is_peak_window and action_kw < -0.05 * pmax and soc > 0.2:
                reward += _power_util_coeff * power_frac
            elif is_offpeak and action_kw > 0.05 * pmax and soc < 0.75:
                reward += _power_util_coeff * 0.5 * power_frac
        
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        if is_weekend and abs(action_kw) > 0.05 * pmax:
            reward -= 0.5
        
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        if soc < self.soc_min or soc > self.soc_max:
            reward -= 5.0
        elif soc > 0.88:
            reward -= 1.0 * ((soc - 0.85) / 0.05) ** 2
        elif soc < 0.12:
            reward -= 1.0 * ((0.15 - soc) / 0.05) ** 2
        
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        if action_kw < 0:
            wasted = abs(action_kw) - useful_discharge
            if wasted > 0.01 * pmax:
                reward -= 0.5 * min(1.0, wasted / pmax)
        
        return reward
    
    def _calculate_reward_v14(self, action_kw: float, load_kw: float, pv_kw: float,
                              price: float, net_load_after_pv: float,
                              useful_discharge: float, charge_kw: float,
                              grid_kw: float, baseline_grid_kw: float,
                              pump_power_kw: float) -> float:
        """
        V14: Grid Minimization Reward
        ==============================
        Core objective: minimize grid electricity usage while meeting load.
        
        Design principles:
          1. Economic savings signal (baseline - agent cost) × scale
          2. Strong discharge incentive during peak hours
          3. Solar charge incentive (free energy capture)
          4. Off-peak charge incentive (cheap preparation)
          5. Penalty for peak-time charging (buying expensive power)
          6. Dead zone awareness (commands < threshold = standby)
          7. No blanket "charge for points" — only context-aware rewards
        """
        reward = 0.0
        soc = self.current_soc
        pmax = max(self.battery_power_kw, 1e-9)
        dt = self.time_step

        steps_per_hour = max(1, int(round(1.0 / max(dt, 1e-9)))) if dt < 1.0 else 1
        hour = int((self.current_step // steps_per_hour) % 24)
        day = int((self.current_step // (steps_per_hour * 24)) % 7)
        is_weekend = (day >= 5)
        is_peak = (16 <= hour < 22) and not is_weekend
        is_midpeak = ((9 <= hour < 16) or (22 <= hour < 24)) and not is_weekend
        is_offpeak = (0 <= hour < 9) and not is_weekend

        # ── 1. Core economic signal (scaled) ──────────────────────
        baseline_cost = baseline_grid_kw * dt * price
        agent_cost = grid_kw * dt * price
        reward += (baseline_cost - agent_cost) * self.tou_reward_scale

        # ── 2. Discharge reward: proportional to grid savings ─────
        if action_kw < -0.01 * pmax:
            if useful_discharge > 0:
                price_tier = price / 7.13
                power_frac = min(1.0, useful_discharge / pmax)
                reward += 3.0 * power_frac * price_tier
            else:
                reward -= 1.0

        # ── 3. Solar charge: capture free energy ──────────────────
        pv_excess = max(0.0, pv_kw - load_kw)
        if action_kw > 0.05 * pmax and pv_excess > 0.01 * pmax and soc < 0.85:
            solar_frac = min(1.0, pv_excess / max(charge_kw, 1e-9))
            reward += 1.5 * solar_frac * min(1.0, action_kw / pmax)
        elif pv_excess > 0.1 * pmax and action_kw <= 0 and soc < 0.60:
            reward -= 0.5

        # ── 4. Peak behavior (the money window) ──────────────────
        if is_peak:
            if action_kw < -0.05 * pmax and soc > 0.15:
                reward += 2.0
            elif action_kw > 0.05 * pmax:
                reward -= 2.0
            elif abs(action_kw) < 0.05 * pmax and soc > 0.30:
                reward -= 0.5

        # ── 5. Off-peak charge (prepare for peak) ────────────────
        if is_offpeak and action_kw > 0.05 * pmax and soc < 0.75:
            reward += 0.5 * min(1.0, action_kw / pmax)

        # ── 6. Mid-peak: moderate signals ─────────────────────────
        if is_midpeak:
            if action_kw < -0.05 * pmax and soc > 0.20:
                reward += 0.8
            elif action_kw > 0.05 * pmax and soc > 0.60:
                reward -= 0.3

        # ── 7. SoC safety ────────────────────────────────────────
        if soc < self.soc_min or soc > self.soc_max:
            reward -= 5.0
        elif soc > 0.92:
            reward -= 1.0
        elif soc < 0.08:
            reward -= 0.5

        # ── 8. Wasted discharge (voltage blocked) ────────────────
        if action_kw < 0:
            wasted = abs(action_kw) - useful_discharge
            if wasted > 0.01 * pmax:
                reward -= 0.5 * min(1.0, wasted / pmax)

        # ── 9. Weekend protection (no arbitrage opportunity) ─────
        if is_weekend and abs(action_kw) > 0.1 * pmax:
            reward -= 0.3

        return reward

    def _calculate_reward_v16(self, action_kw: float, load_kw: float, pv_kw: float,
                              price: float, net_load_after_pv: float,
                              useful_discharge: float, charge_kw: float,
                              grid_kw: float, baseline_grid_kw: float,
                              pump_power_kw: float,
                              blocked_by_pv: bool = False,
                              blocked_by_load: bool = False) -> float:
        """
        V16: Profit + Solar Guidance
        ============================
        Primary: (baseline_grid_cost − agent_grid_cost) × scale
        Guidance: small bonus for free solar charging, penalty for wasting it.

        SoC boundaries are enforced by CORAL/SafetyNet.
        """
        dt = self.time_step
        pmax = self.battery_power_kw
        soc = self.current_soc

        baseline_cost = baseline_grid_kw * dt * price
        agent_cost = grid_kw * dt * price
        reward = (baseline_cost - agent_cost) * self.tou_reward_scale

        pv_excess = max(0.0, pv_kw - load_kw)
        if charge_kw > 0.05 * pmax and pv_excess > 0.01 * pmax and soc < 0.85:
            solar_frac = min(1.0, pv_excess / max(charge_kw, 1e-9))
            power_frac = min(1.0, charge_kw / pmax)
            reward += 1.5 * solar_frac * power_frac
        elif pv_excess > 0.1 * pmax and charge_kw <= 0 and soc < 0.60:
            reward -= 0.5

        if blocked_by_pv:
            reward -= float(getattr(self, 'blocked_by_pv_penalty', 0.10))
        if blocked_by_load:
            reward -= float(getattr(self, 'blocked_by_load_penalty', 0.05))

        return reward

    def _calculate_reward_v16m(self, action_kw: float, load_kw: float, pv_kw: float,
                               price: float, net_load_after_pv: float,
                               useful_discharge: float, charge_kw: float,
                               grid_kw: float, baseline_grid_kw: float,
                               pump_power_kw: float,
                               blocked_by_pv: bool = False,
                               blocked_by_load: bool = False) -> float:
        """
        V16m: Minimal explainable reward
        =================================
        Keep only:
          1. Profit signal: baseline grid cost - agent grid cost
          2. Small penalties for physically infeasible discharge commands

        No hand-crafted incentives for solar charging or peak discharge timing.
        SoC safety remains handled by SafetyNet / violation penalties.
        """
        dt = self.time_step
        baseline_cost = baseline_grid_kw * dt * price
        agent_cost = grid_kw * dt * price
        reward = (baseline_cost - agent_cost) * self.tou_reward_scale

        if blocked_by_pv:
            reward -= float(getattr(self, 'blocked_by_pv_penalty', 0.10))
        if blocked_by_load:
            reward -= float(getattr(self, 'blocked_by_load_penalty', 0.05))

        return reward

    def _calculate_reward_v16s(self, action_kw: float, load_kw: float, pv_kw: float,
                               price: float, net_load_after_pv: float,
                               useful_discharge: float, charge_kw: float,
                               grid_kw: float, baseline_grid_kw: float,
                               pump_power_kw: float, pv_to_battery: float,
                               blocked_by_pv: bool = False,
                               blocked_by_load: bool = False) -> float:
        """
        V16s: Minimal reward + stored free solar value
        ==============================================
        Keep the explainable pieces only:
          1. Grid-cost savings vs baseline
          2. Small penalties for infeasible discharge commands
          3. Positive value when excess PV is actually stored in battery

        The solar storage term only fires on real `pv_to_battery`, so it does not
        reward fake charging from the grid.
        """
        dt = self.time_step
        baseline_cost = baseline_grid_kw * dt * price
        agent_cost = grid_kw * dt * price
        reward = (baseline_cost - agent_cost) * self.tou_reward_scale

        if blocked_by_pv:
            reward -= float(getattr(self, 'blocked_by_pv_penalty', 0.10))
        if blocked_by_load:
            reward -= float(getattr(self, 'blocked_by_load_penalty', 0.05))

        storage_price = float(getattr(self, 'solar_storage_value_price', 7.13))
        storage_scale = float(getattr(self, 'solar_storage_value_scale', 1.0))
        if pv_to_battery > 1e-9 and storage_scale > 0.0:
            reward += pv_to_battery * dt * storage_price * self.tou_reward_scale * storage_scale

        return reward

    def _calculate_reward_v16sp(self, action_kw: float, load_kw: float, pv_kw: float,
                                price: float, net_load_after_pv: float,
                                useful_discharge: float, charge_kw: float,
                                grid_kw: float, baseline_grid_kw: float,
                                pump_power_kw: float, pv_to_battery: float,
                                blocked_by_pv: bool = False,
                                blocked_by_load: bool = False) -> float:
        """
        V16sp: Stable minimum-standard reward on top of v16s
        ================================================
        Keep the explainable v16s core, then add only three conservative signals:
          1. Off-peak top-up bonus when SoC is still clearly low.
          2. Peak discharge bonus when discharge actually reduces grid demand.
          3. Peak idle penalty only when discharge is physically available.

        The design intentionally avoids depending on exact PV attribution.
        """
        reward = self._calculate_reward_v16s(
            action_kw, load_kw, pv_kw, price,
            net_load_after_pv, useful_discharge, charge_kw,
            grid_kw, baseline_grid_kw, pump_power_kw, pv_to_battery,
            blocked_by_pv=blocked_by_pv,
            blocked_by_load=blocked_by_load,
        )

        dt = self.time_step
        steps_per_hour = max(1, int(round(1.0 / max(dt, 1e-9)))) if dt < 1.0 else 1
        hour = int((self.current_step // steps_per_hour) % 24)
        day = int((self.current_step // (steps_per_hour * 24)) % 7)
        is_weekend = (day >= 5)
        is_peak = (16 <= hour < 22) and not is_weekend
        is_midpeak = ((9 <= hour < 16) or (22 <= hour < 24)) and not is_weekend
        is_offpeak = (0 <= hour < 9) and not is_weekend

        soc = self.current_soc
        charge_target = float(getattr(self, 'offpeak_charge_soc_target', 0.85))
        discharge_floor = float(getattr(self, 'peak_discharge_soc_floor', 0.15))
        offpeak_bonus = float(getattr(self, 'v17_offpeak_charge_bonus', 0.6))
        peak_bonus = float(getattr(self, 'v17_peak_discharge_bonus', 1.2))
        peak_idle_penalty = float(getattr(self, 'v17_peak_idle_penalty', 0.4))

        charge_frac = min(1.0, charge_kw / max(self.battery_charge_power_kw, 1e-9))
        discharge_frac = min(1.0, useful_discharge / max(self.battery_discharge_power_kw, 1e-9))
        need_charge_frac = np.clip((charge_target - soc) / max(charge_target, 1e-9), 0.0, 1.0)
        extra_energy_frac = np.clip((soc - discharge_floor) / max(1.0 - discharge_floor, 1e-9), 0.0, 1.0)
        throughput_kwh = max(0.0, charge_kw + useful_discharge) * dt
        no_pv_threshold_kw = float(getattr(self, 'no_pv_action_threshold_kw', 0.001))
        no_pv = pv_kw <= no_pv_threshold_kw

        # In partial_assist, any non-zero load without PV block is a usable discharge opportunity.
        discharge_available = (load_kw > 1e-9) and (not blocked_by_pv) and soc > discharge_floor + 1e-6

        if is_offpeak and soc < charge_target and charge_kw > 0.05 * self.battery_charge_power_kw:
            reward += offpeak_bonus * need_charge_frac * charge_frac

        if is_peak and useful_discharge > 1e-9 and extra_energy_frac > 0.0:
            reward += peak_bonus * extra_energy_frac * discharge_frac
        elif (
            is_peak
            and discharge_available
            and useful_discharge <= 1e-9
            and charge_kw <= 1e-9
            and extra_energy_frac > 0.0
        ):
            reward -= peak_idle_penalty * extra_energy_frac

        # Penalize economically irrational battery throughput when there is no PV support.
        throughput_penalty_per_kwh = float(getattr(self, 'no_pv_throughput_penalty_per_kwh', 0.0))
        if no_pv and throughput_kwh > 1e-12 and throughput_penalty_per_kwh > 0.0:
            reward -= throughput_kwh * throughput_penalty_per_kwh * self.tou_reward_scale

        # Extra penalty: discharging during off-peak with no PV usually destroys arbitrage value.
        offpeak_discharge_penalty_per_kwh = float(
            getattr(self, 'offpeak_no_pv_discharge_penalty_per_kwh', 0.0)
        )
        if (
            is_offpeak
            and no_pv
            and useful_discharge > 1e-12
            and offpeak_discharge_penalty_per_kwh > 0.0
        ):
            reward -= useful_discharge * dt * offpeak_discharge_penalty_per_kwh * self.tou_reward_scale

        return reward

    def _calculate_reward_v16e(self, action_kw: float, load_kw: float, pv_kw: float,
                               price: float, net_load_after_pv: float,
                               useful_discharge: float, charge_kw: float,
                               grid_kw: float, baseline_grid_kw: float,
                               pump_power_kw: float,
                               blocked_by_pv: bool = False,
                               blocked_by_load: bool = False) -> float:
        """
        V16e: Episodic Profit Reward
        =============================
        Accumulate baseline vs agent grid cost over the episode.
        Return 0 for all non-terminal steps.
        At terminal step: return (total_baseline_cost - total_agent_cost) × scale.

        This makes "solar charging = less grid over the episode" a single clear signal.
        CORAL/SafetyNet penalties still apply per-step (added externally).
        """
        dt = self.time_step
        self._ep_baseline_cost += baseline_grid_kw * dt * price
        self._ep_agent_cost += grid_kw * dt * price

        blocked_penalty = 0.0
        if blocked_by_pv:
            blocked_penalty -= float(getattr(self, 'blocked_by_pv_penalty', 0.10))
        if blocked_by_load:
            blocked_penalty -= float(getattr(self, 'blocked_by_load_penalty', 0.05))

        is_terminal = (self.current_step >= self.episode_length - 1)
        if not is_terminal:
            return blocked_penalty

        episode_savings = self._ep_baseline_cost - self._ep_agent_cost
        return blocked_penalty + episode_savings * self.tou_reward_scale

    def _calculate_reward_v17(self, action_kw: float, load_kw: float, pv_kw: float,
                              price: float, net_load_after_pv: float,
                              useful_discharge: float, charge_kw: float,
                              grid_kw: float, baseline_grid_kw: float,
                              pump_power_kw: float, pv_to_battery: float,
                              blocked_by_pv: bool = False,
                              blocked_by_load: bool = False) -> float:
        """
        V17: Off-peak preparation + peak discharge
        ==========================================
        Keep the profit signal as the main objective, then add only three
        explainable nudges:
          1. Reward charging during cheap hours when SoC is still below target.
          2. Reward effective discharge during expensive peak hours.
          3. Penalize idling during peak hours when discharge is physically feasible.

        Solar remains a secondary helper: only real `pv_to_battery` gets a small
        bonus. We do not assume "sunlight means free charging opportunity".
        """
        dt = self.time_step
        baseline_cost = baseline_grid_kw * dt * price
        agent_cost = grid_kw * dt * price
        reward = (baseline_cost - agent_cost) * self.tou_reward_scale

        if blocked_by_pv:
            reward -= float(getattr(self, 'blocked_by_pv_penalty', 0.10))
        if blocked_by_load:
            reward -= float(getattr(self, 'blocked_by_load_penalty', 0.05))

        steps_per_hour = max(1, int(round(1.0 / max(dt, 1e-9)))) if dt < 1.0 else 1
        hour = int((self.current_step // steps_per_hour) % 24)
        day = int((self.current_step // (steps_per_hour * 24)) % 7)
        is_weekend = (day >= 5)
        is_peak = (16 <= hour < 22) and not is_weekend
        is_midpeak = ((9 <= hour < 16) or (22 <= hour < 24)) and not is_weekend
        is_offpeak = (0 <= hour < 9) and not is_weekend

        soc = self.current_soc
        charge_target = float(getattr(self, 'offpeak_charge_soc_target', 0.85))
        discharge_floor = float(getattr(self, 'peak_discharge_soc_floor', 0.15))
        offpeak_bonus = float(getattr(self, 'v17_offpeak_charge_bonus', 0.6))
        peak_bonus = float(getattr(self, 'v17_peak_discharge_bonus', 1.2))
        peak_idle_penalty = float(getattr(self, 'v17_peak_idle_penalty', 0.4))
        solar_storage_scale = float(getattr(self, 'v17_solar_storage_bonus_scale', 0.3))
        charge_frac = min(1.0, charge_kw / max(self.battery_charge_power_kw, 1e-9))
        discharge_frac = min(1.0, useful_discharge / max(self.battery_discharge_power_kw, 1e-9))
        need_charge_frac = np.clip((charge_target - soc) / max(charge_target, 1e-9), 0.0, 1.0)
        extra_energy_frac = np.clip((soc - discharge_floor) / max(1.0 - discharge_floor, 1e-9), 0.0, 1.0)
        discharge_feasible = (
            (not blocked_by_pv)
            and (not blocked_by_load)
            and (load_kw <= self.battery_discharge_power_kw + 1e-9)
        )

        if is_offpeak and charge_kw > 0.01 * self.battery_charge_power_kw and need_charge_frac > 0.0:
            reward += offpeak_bonus * need_charge_frac * charge_frac

        if is_peak and useful_discharge > 1e-9 and extra_energy_frac > 0.0:
            reward += peak_bonus * extra_energy_frac * discharge_frac
        elif is_peak and discharge_feasible and soc > discharge_floor + 1e-6 and useful_discharge <= 1e-9:
            reward -= peak_idle_penalty * extra_energy_frac

        if pv_to_battery > 1e-9 and solar_storage_scale > 0.0:
            reward += (
                pv_to_battery
                * dt
                * float(getattr(self, 'solar_storage_value_price', 7.13))
                * self.tou_reward_scale
                * solar_storage_scale
            )

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Documentation for this public API is provided in English."""
        super().reset(seed=seed)
        rng = getattr(self, 'np_random', None)
        if rng is None:
            rng = np.random.default_rng(seed)
        options = options or {}
        force_cold_reset = bool(options.get('force_cold_reset', False))
        preserve_persistent_state = bool(
            self.continuous_operation_mode
            and self._continuous_initialized
            and not force_cold_reset
        )
        
        # Reset episode state
        self.current_step = 0
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.soc_violations = 0
        self.action_violations = 0
        self.strict_soc_violation_steps = 0
        self.strict_soc_violation_duration_h = 0.0
        self.strict_soc_violation_kwh = 0.0
        self.strict_soc_violation_max_kwh = 0.0
        self._last_grid_kw = 0.0
        self._last_net_load_after_pv = 0.0
        self._last_useful_discharge = 0.0
        self._last_pv_to_load = 0.0
        self._ep_baseline_cost = 0.0
        self._ep_agent_cost = 0.0
        if not preserve_persistent_state:
            eps = 1e-3
            _soc_init_mode = getattr(self, '_soc_init_mode', 'balanced')
            r = float(rng.random())
            if _soc_init_mode == 'full_random':
                lo = float(self.soc_min) + eps
                hi = float(self.soc_max) - eps
            elif _soc_init_mode == 'low_bias':
                if r < 0.7:
                    lo = max(float(self.soc_min), 0.0)
                    hi = min(float(self.soc_max), 0.15)
                else:
                    lo = max(float(self.soc_min), 0.15)
                    hi = min(float(self.soc_max), 0.50)
            else:  # 'balanced' (v13)
                if r < 0.4:
                    lo = max(float(self.soc_min), 0.10)
                    hi = min(float(self.soc_max), 0.25)
                elif r < 0.7:
                    lo = max(float(self.soc_min), 0.25)
                    hi = min(float(self.soc_max), 0.55)
                else:
                    lo = max(float(self.soc_min), 0.55)
                    hi = min(float(self.soc_max), 0.85)
            if hi <= lo + eps:
                lo = float(self.soc_min) + eps
                hi = float(self.soc_max) - eps
            if hi < lo:
                lo = float(self.soc_min)
                hi = float(self.soc_min)
            self.current_soc = float(rng.uniform(lo, hi))
            self.current_soh = float(self._initial_soh)
            self.current_flow_rate_lpm = float(self._initial_flow_rate_lpm)
            self._current_flow_action = self.flow_idle_fraction
            self.prev_action_kw = 0.0
            self._prev_exec_action_kw = 0.0
            self._deployment_voltage_cutoff_active = False
            self._deployment_guard_day = None
        # Reset effective bounds with drift if stress
        if self.stress_enable and self.stress_bounds_drift_std > 0.0:
            drift = float(rng.normal()) * self.stress_bounds_drift_std
            # Shrink and shift within [soc_min, soc_max]
            base_low = self.soc_min + max(0.0, drift)
            base_high = self.soc_max - max(0.0, drift)
            margin = 1e-3
            self.soc_min_eff = float(np.clip(base_low, self.soc_min + margin, self.soc_max - 2*margin))
            self.soc_max_eff = float(np.clip(base_high, self.soc_min + 2*margin, self.soc_max - margin))
        else:
            self.soc_min_eff = self.soc_min
            self.soc_max_eff = self.soc_max
        self.current_soc = float(np.clip(self.current_soc, float(self.soc_min_eff), float(self.soc_max_eff)))
        # Initialize observation buffer for delayed/noisy SoC
        if not preserve_persistent_state or len(self._soc_obs_buffer) == 0:
            self._soc_obs_buffer = [self.current_soc] * (max(0, self.stress_soc_obs_delay) + 1)
        
        if self.load_data is None:
            self._generate_synthetic_data()
        self._refresh_time_arrays()

        _ld  = self.load_data  if self.load_data  is not None else np.ones(self.episode_length) * 30.0
        _pv  = self.pv_data    if self.pv_data    is not None else np.ones(self.episode_length) * 20.0
        _pr  = self.price_data if self.price_data is not None else np.ones(self.episode_length) * 0.15
        _hr  = self.hour_data  if self.hour_data  is not None else None
        _dow = self.dow_data   if self.dow_data   is not None else None

        self.episode_data = {
            'load' : _ld,
            'pv'   : _pv,
            'price': _pr,
        }
        if _hr is not None:
            self.episode_data['hour'] = _hr
        if _dow is not None:
            self.episode_data['dow'] = _dow
        
        if self.use_extended_obs:
            n = self.episode_length
            pv_arr   = self.episode_data['pv']
            load_arr = self.episode_data['load']
            self.episode_data['pv_std']   = self.pv_std_data[:n]   if self.pv_std_data   is not None else pv_arr   * 0.15
            self.episode_data['pv_max']   = self.pv_max_data[:n]   if self.pv_max_data   is not None else pv_arr   * 1.20
            self.episode_data['load_std'] = self.load_std_data[:n] if self.load_std_data is not None else load_arr * 0.10
            self.episode_data['load_max'] = self.load_max_data[:n] if self.load_max_data is not None else load_arr * 1.15
            self.episode_data['soh']      = self.soh_data[:n]      if self.soh_data      is not None else np.full(n, self._initial_soh)
            self.episode_data['flow_rate']= self.flow_rate_data[:n]if self.flow_rate_data is not None else np.full(n, self._initial_flow_rate_lpm)
        
        try:
            total_len = int(min(len(self.episode_data['load']), len(self.episode_data['pv']), len(self.episode_data['price'])))
            if total_len >= int(self.episode_length) and int(self.episode_length) > 0:
                max_start_global = max(0, total_len - int(self.episode_length))
                if self.fixed_start_idx is not None:
                    start_idx = int(max(0, min(self.fixed_start_idx, max_start_global)))
                elif preserve_persistent_state and self.continuous_operation_mode:
                    start_idx = int(max(0, min(self._continuous_next_start_idx, max_start_global)))
                else:
                    valid_starts = getattr(self, '_valid_episode_start_indices', None)
                    if valid_starts is not None and len(valid_starts) > 0:
                        start_idx = int(valid_starts[int(rng.integers(0, len(valid_starts)))])
                    elif self.train_window_hours is not None:
                        max_start = int(max(0, min(self.train_window_hours, max_start_global)))
                        start_idx = int(rng.integers(0, max_start + 1))
                    else:
                        max_start = max_start_global
                        start_idx = int(rng.integers(0, max_start + 1))
                sl = slice(start_idx, start_idx + self.episode_length)
                new_ep: dict = {
                    'load' : self.episode_data['load'][sl],
                    'pv'   : self.episode_data['pv'][sl],
                    'price': self.episode_data['price'][sl],
                }
                for _k in ('hour', 'dow', 'pv_bool', 'pv_std', 'pv_max', 'load_std', 'load_max', 'soh', 'flow_rate'):
                    if _k in self.episode_data:
                        new_ep[_k] = self.episode_data[_k][sl]
                self.episode_data = new_ep
                self.current_step = 0
                if self.continuous_operation_mode and self.fixed_start_idx is None:
                    if max_start_global > 0:
                        next_start = start_idx + int(self.episode_length)
                        self._continuous_next_start_idx = 0 if next_start > max_start_global else next_start
                    else:
                        self._continuous_next_start_idx = 0
        except Exception:
            self.current_step = 0

        if not self.use_real_data and self.episode_data is not None:
            pv = self.episode_data.get('pv', None)
            load = self.episode_data.get('load', None)
            if pv is not None and self.weather_pv_scale_std > 0.0:
                pv_scale = max(0.0, 1.0 + np.random.randn() * self.weather_pv_scale_std)
                pv = pv * pv_scale
            if load is not None and self.weather_load_scale_std > 0.0:
                load_scale = max(0.0, 1.0 + np.random.randn() * self.weather_load_scale_std)
                load = load * load_scale
            if pv is not None and self.weather_pv_noise_std > 0.0:
                pv = pv * (1.0 + np.random.randn(len(pv)) * self.weather_pv_noise_std)
            if pv is not None:
                self.episode_data['pv'] = np.maximum(pv, 0.0)
            if load is not None:
                self.episode_data['load'] = np.maximum(load, 0.0)
        self._refresh_episode_pv_bool()

        # Get initial state
        initial_state = self._get_state()
        
        info = {
            'soc': self.current_soc,
            'step': self.current_step,
            'load': self.episode_data['load'][0],
            'pv': self.episode_data['pv'][0],
            'price': self.episode_data['price'][0]
        }
        self._continuous_initialized = True
        
        return initial_state, info
    
    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Documentation for this public API is provided in English.
        
        """
        if self.current_step >= self.episode_length:
            return self._get_state(), 0.0, True, False, {}
        
        action_kw = float(action[0])
        action_kw = self._clip_action_by_direction(action_kw)
        if self.use_flow_rate_action and len(action) >= 2:
            requested_flow_action = float(np.clip(action[1], 0.0, 1.0))
            flow_action = self._apply_flow_operating_rule(requested_flow_action, active=False)
        else:
            requested_flow_action = self.fixed_flow_fraction_when_uncontrolled
            flow_action = requested_flow_action
        self._current_flow_action = flow_action

        if self.action_dead_zone_kw > 0 and abs(action_kw) < self.action_dead_zone_kw:
            action_kw = 0.0

        discharge_intent_below_threshold = 0
        if (
            self.discharge_auto
            and self.discharge_intent_threshold_kw > 0
            and action_kw < 0
            and abs(action_kw) < self.discharge_intent_threshold_kw
        ):
            action_kw = 0.0
            discharge_intent_below_threshold = 1

        if self.discharge_auto and action_kw < 0:
            auto_snapshot = self._build_observation_snapshot(self.current_step)
            load_kw_now = self._get_battery_delivered_load_kw(
                float(auto_snapshot['load_kw']),
                int(auto_snapshot['load_groups']),
            )
            auto_discharge = min(self.battery_discharge_power_kw, load_kw_now)
            action_kw = -auto_discharge

        if self.ramp_limit_kw is not None:
            action_diff = abs(action_kw - self.prev_action_kw)
            if action_diff > self.ramp_limit_kw:
                self.action_violations += 1
                if action_kw > self.prev_action_kw:
                    action_kw = self.prev_action_kw + self.ramp_limit_kw
                else:
                    action_kw = self.prev_action_kw - self.ramp_limit_kw
        
        applied_action_kw = action_kw
        if self.stress_enable and self.stress_action_lag_alpha > 0.0:
            alpha = float(np.clip(self.stress_action_lag_alpha, 0.0, 0.99))
            applied_action_kw = alpha * self._prev_exec_action_kw + (1.0 - alpha) * action_kw
        if self.stress_enable and self.stress_external_pmax_shrink_prob > 0.0:
            if np.random.rand() < self.stress_external_pmax_shrink_prob:
                shrink = float(np.clip(self.stress_external_pmax_shrink_factor, 0.1, 1.0))
                applied_action_kw = float(np.clip(
                    applied_action_kw,
                    -self.battery_discharge_power_kw * shrink,
                    self.battery_charge_power_kw * shrink,
                ))
        
        if self.stress_enable and self.stress_power_loss_ratio > 0.0:
            loss_ratio = float(self.stress_power_loss_ratio)
            if applied_action_kw > 0:
                applied_action_kw = applied_action_kw / max(1e-6, 1.0 - loss_ratio)
            elif applied_action_kw < 0:
                applied_action_kw = applied_action_kw * (1.0 - loss_ratio)
            applied_action_kw = self._clip_action_by_direction(applied_action_kw)

        obs_snapshot = self._build_observation_snapshot(self.current_step)
        actual_load_kw = self.episode_data['load'][self.current_step]
        actual_pv_kw = self.episode_data['pv'][self.current_step]
        battery_delivered_load_kw = self._get_battery_delivered_load_kw(
            float(obs_snapshot['load_kw']),
            int(obs_snapshot['load_groups']),
        )
        current_day = int(obs_snapshot['dow'])
        applied_action_kw, deployment_guard_flags = self._apply_deployment_style_guards(
            applied_action_kw,
            float(obs_snapshot['pv_kw']),
            float(obs_snapshot['load_kw']),
            battery_delivered_load_kw,
            current_day,
        )

        if applied_action_kw > 0 and (self.charge_requires_pv_surplus or self.charge_limit_to_pv_surplus):
            pv_surplus_kw = max(0.0, float(actual_pv_kw) - float(actual_load_kw))
            if self.charge_requires_pv_surplus and pv_surplus_kw <= 1e-9:
                applied_action_kw = 0.0
            elif self.charge_limit_to_pv_surplus:
                applied_action_kw = min(applied_action_kw, pv_surplus_kw)

        if self.voltage_cutoff_soc > 0 and self.current_soc <= self.voltage_cutoff_soc and applied_action_kw < 0:
            applied_action_kw = 0.0

        if self.stress_enable:
            if (
                self.stress_battery_zero_response_prob > 0.0
                and np.random.rand() < self.stress_battery_zero_response_prob
            ):
                applied_action_kw = 0.0
            elif self.stress_battery_response_noise_std > 0.0 and abs(applied_action_kw) > 1e-12:
                applied_action_kw *= 1.0 + float(np.random.randn()) * self.stress_battery_response_noise_std
                applied_action_kw = self._clip_action_by_direction(applied_action_kw)

        if self.use_flow_rate_action:
            flow_action = self._apply_flow_operating_rule(
                requested_flow_action,
                active=abs(applied_action_kw) > max(self.action_dead_zone_kw, 1e-9),
            )
            self._current_flow_action = flow_action

        flow_charge_limit_kw, flow_discharge_limit_kw = self._flow_limited_power_bounds(flow_action)
        flow_power_limited = 0
        if self.use_flow_rate_action and self.flow_limits_available_power and abs(applied_action_kw) > 1e-12:
            before_flow_limit_kw = applied_action_kw
            applied_action_kw = float(np.clip(
                applied_action_kw,
                -flow_discharge_limit_kw,
                flow_charge_limit_kw,
            ))
            flow_power_limited = int(abs(applied_action_kw - before_flow_limit_kw) > 1e-12)
        
        pump_power_kw = 0.0
        flow_eta_modifier = 1.0
        net_power_kw = applied_action_kw
        
        if self.use_flow_rate_action and flow_action > 0.0:
            Q = flow_action
            pump_power_kw = self._pump_power_W(Q) / 1000.0
            is_charging = (applied_action_kw >= 0)
            flow_eta_modifier = self._flow_efficiency(Q, is_charging)
            net_power_kw = self._net_power_W(applied_action_kw * 1000.0, Q) / 1000.0
        elif not self.use_flow_rate_action and flow_action > 0.0:
            # In the uncontrolled-flow scenario the pump is an always-on
            # auxiliary load. It does not alter the policy action or battery
            # power, but it must be purchased from the grid at every step.
            pump_power_kw = self._pump_power_W(flow_action) / 1000.0
        
        if self.stress_enable:
            dt_jitter = 1.0 + float(np.random.randn()) * self.stress_dt_jitter_std
            self._effective_time_step = max(1e-3, float(self.time_step) * dt_jitter)
            eta_noise = 1.0 + float(np.random.randn()) * self.stress_efficiency_noise_std
            self._effective_efficiency = float(np.clip(
                self.battery_efficiency * eta_noise * flow_eta_modifier, 1e-3, 1.0))
        else:
            self._effective_time_step = float(self.time_step)
            self._effective_efficiency = float(np.clip(
                self.battery_efficiency * flow_eta_modifier, 1e-3, 1.0))

        old_soc = self.current_soc
        try:
            soc_next_raw = float(self.predict_soc_raw(old_soc, applied_action_kw))
        except Exception:
            soc_next_raw = old_soc
        energy_violate_kwh = 0.0
        if soc_next_raw < self.soc_min:
            energy_violate_kwh = (self.soc_min - soc_next_raw) * self.battery_capacity_kwh
        elif soc_next_raw > self.soc_max:
            energy_violate_kwh = (soc_next_raw - self.soc_max) * self.battery_capacity_kwh

        if self.hard_guard and (soc_next_raw < self.soc_min or soc_next_raw > self.soc_max):
            dt = float(self.time_step)
            eta = float(self.battery_efficiency)
            cap = float(self.battery_capacity_kwh)
            target_soc = self.soc_min if soc_next_raw < self.soc_min else self.soc_max
            delta_e = (target_soc - old_soc) * cap  # kWh
            if delta_e >= 0:
                action_kw = delta_e / (dt * eta)
            else:
                action_kw = delta_e * eta / dt
            action_kw = self._clip_action_by_direction(action_kw)
            soc_next_raw = float(self.predict_soc_raw(old_soc, action_kw))

        soc_before_action = self.current_soc
        self.current_soc, actual_action_kw = self._update_battery_soc(applied_action_kw)
        self._prev_exec_action_kw = applied_action_kw
        self.current_flow_rate_lpm = flow_action * 20.0

        if self.use_flow_rate_action:
            actual_net_power_kw = self._net_power_W(actual_action_kw * 1000.0, flow_action) / 1000.0
        else:
            actual_net_power_kw = actual_action_kw

        # ═══════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════
        load_kw = actual_load_kw
        pv_kw = actual_pv_kw
        price = self.episode_data['price'][self.current_step]
        battery_load_kw = battery_delivered_load_kw
        pv_to_load = min(pv_kw, load_kw)
        net_load_after_pv = max(0.0, load_kw - pv_kw)

        pv_active = bool(self._compute_pv_active(pv_kw) > 0.5)
        voltage_blocked = False
        blocked_by_pv = False
        blocked_by_load = False
        discharge_kw_net = 0.0
        if actual_action_kw < 0:
            discharge_kw_net = abs(actual_net_power_kw) if actual_net_power_kw < 0 else 0.0
            if self.allow_grid_export:
                useful_discharge = min(discharge_kw_net, load_kw)
            elif pv_active:
                useful_discharge = 0.0
                voltage_blocked = True
                blocked_by_pv = True
            else:
                if self.discharge_mode == 'partial_assist':
                    useful_discharge = min(discharge_kw_net, load_kw)
                    if useful_discharge <= 1e-9:
                        useful_discharge = 0.0
                        voltage_blocked = True
                else:
                    discharge_kw_gross = abs(actual_action_kw)
                    if discharge_kw_gross >= battery_load_kw * 0.99:
                        useful_discharge = min(discharge_kw_net, battery_load_kw)
                    else:
                        useful_discharge = 0.0
                        voltage_blocked = True
                        blocked_by_load = True

            if voltage_blocked:
                self.current_soc = soc_before_action
                actual_action_kw = 0.0
                actual_net_power_kw = 0.0
                discharge_kw_net = 0.0
                if self.use_flow_rate_action:
                    flow_action = self.flow_idle_fraction
                    self._current_flow_action = flow_action
                    self.current_flow_rate_lpm = flow_action * 20.0
                    pump_power_kw = 0.0
                flow_eta_modifier = 1.0
                net_power_kw = 0.0

            charge_kw = 0.0
            pv_to_battery = 0.0
        else:

            useful_discharge = 0.0
            charge_kw = actual_net_power_kw
            pv_excess = max(0.0, pv_kw - load_kw)
            pv_to_battery = min(pv_excess, charge_kw)

        strict_violation_excess_kwh = self._soc_violation_excess_kwh(self.current_soc)
        if strict_violation_excess_kwh > 0.0:
            self.strict_soc_violation_steps += 1
            self.strict_soc_violation_duration_h += float(self.time_step)
            self.strict_soc_violation_kwh += strict_violation_excess_kwh
            self.strict_soc_violation_max_kwh = max(
                self.strict_soc_violation_max_kwh,
                strict_violation_excess_kwh,
            )
            if not self.clip_soc_to_bounds:
                energy_violate_kwh = max(energy_violate_kwh, strict_violation_excess_kwh)
        
        if self.soh_degradation_per_kwh > 0.0:
            throughput_kwh = abs(actual_action_kw) * float(self._effective_time_step)
            self.current_soh = float(np.clip(
                self.current_soh - self.soh_degradation_per_kwh * throughput_kwh,
                0.0, 1.0
            ))

        physical_pump_power_kw = pump_power_kw
        costed_pump_power_kw = physical_pump_power_kw
        if self.flow_charge_pump_free and actual_action_kw > 1e-6:
            costed_pump_power_kw = 0.0
        elif self.use_flow_rate_action and not self.flow_pump_from_grid:
            # Battery-side pump loss is already embedded in actual_net_power_kw.
            # Do not add it to grid import a second time.
            costed_pump_power_kw = 0.0
        pump_energy_kwh = physical_pump_power_kw * self.time_step

        if self.allow_grid_export:
            sell_price = price * self.feed_in_tariff_ratio
            net_grid_kw = load_kw - pv_kw - discharge_kw_net + (charge_kw - pv_to_battery) + costed_pump_power_kw
            grid_import_kw = max(0.0, net_grid_kw)
            grid_export_kw = max(0.0, -net_grid_kw)
            grid_cost = grid_import_kw * self.time_step * price
            export_revenue = grid_export_kw * self.time_step * sell_price

            baseline_net_grid_kw = load_kw - pv_kw
            baseline_import_kw = max(0.0, baseline_net_grid_kw)
            baseline_export_kw = max(0.0, -baseline_net_grid_kw)
            baseline_cost = (
                baseline_import_kw * self.time_step * price
                - baseline_export_kw * self.time_step * sell_price
            )
            agent_net_cost = grid_cost - export_revenue
            grid_kw = agent_net_cost / max(self.time_step * price, 1e-9)
            baseline_grid_kw = baseline_cost / max(self.time_step * price, 1e-9)
            grid_savings = baseline_cost - agent_net_cost
        elif useful_discharge > 0 and self.discharge_mode == 'solo_only':
            grid_kw = costed_pump_power_kw
            pv_to_load = 0.0
            grid_import_kw = grid_kw
            grid_export_kw = 0.0
            export_revenue = 0.0
            baseline_grid_kw = net_load_after_pv
            grid_cost = grid_kw * self.time_step * price
            baseline_cost = baseline_grid_kw * self.time_step * price
            grid_savings = baseline_cost - grid_cost
        else:
            grid_kw = net_load_after_pv - useful_discharge + (charge_kw - pv_to_battery) + costed_pump_power_kw
            grid_kw = max(0.0, grid_kw)
            grid_import_kw = grid_kw
            grid_export_kw = 0.0
            export_revenue = 0.0
            baseline_grid_kw = net_load_after_pv
            grid_cost = grid_kw * self.time_step * price
            baseline_cost = baseline_grid_kw * self.time_step * price
            grid_savings = baseline_cost - grid_cost
        
        # Keep profit accounting identical across market modes:
        # profit = export revenue - grid import cost. Grid savings remains a
        # reward component, not a second monetary revenue stream.
        self.total_revenue += export_revenue
        self.total_cost += grid_cost
        
        self._last_grid_kw = grid_kw
        self._last_net_load_after_pv = net_load_after_pv
        self._last_useful_discharge = useful_discharge
        self._last_pv_to_load = pv_to_load
        
        if self.reward_version == 'v16':
            reward = self._calculate_reward_v16(
                action_kw, load_kw, pv_kw, price,
                net_load_after_pv, useful_discharge, charge_kw,
                grid_kw, baseline_grid_kw, pump_power_kw,
                blocked_by_pv=blocked_by_pv,
                blocked_by_load=blocked_by_load,
            )
        elif self.reward_version == 'v16m':
            reward = self._calculate_reward_v16m(
                action_kw, load_kw, pv_kw, price,
                net_load_after_pv, useful_discharge, charge_kw,
                grid_kw, baseline_grid_kw, pump_power_kw,
                blocked_by_pv=blocked_by_pv,
                blocked_by_load=blocked_by_load,
            )
        elif self.reward_version == 'v16s':
            reward = self._calculate_reward_v16s(
                action_kw, load_kw, pv_kw, price,
                net_load_after_pv, useful_discharge, charge_kw,
                grid_kw, baseline_grid_kw, pump_power_kw, pv_to_battery,
                blocked_by_pv=blocked_by_pv,
                blocked_by_load=blocked_by_load,
            )
        elif self.reward_version == 'v16sp':
            reward = self._calculate_reward_v16sp(
                action_kw, load_kw, pv_kw, price,
                net_load_after_pv, useful_discharge, charge_kw,
                grid_kw, baseline_grid_kw, pump_power_kw, pv_to_battery,
                blocked_by_pv=blocked_by_pv,
                blocked_by_load=blocked_by_load,
            )
        elif self.reward_version == 'v17':
            reward = self._calculate_reward_v17(
                action_kw, load_kw, pv_kw, price,
                net_load_after_pv, useful_discharge, charge_kw,
                grid_kw, baseline_grid_kw, pump_power_kw, pv_to_battery,
                blocked_by_pv=blocked_by_pv,
                blocked_by_load=blocked_by_load,
            )
        elif self.reward_version == 'v16e':
            reward = self._calculate_reward_v16e(
                action_kw, load_kw, pv_kw, price,
                net_load_after_pv, useful_discharge, charge_kw,
                grid_kw, baseline_grid_kw, pump_power_kw,
                blocked_by_pv=blocked_by_pv,
                blocked_by_load=blocked_by_load,
            )
        elif self.reward_version == 'v14':
            reward = self._calculate_reward_v14(
                action_kw, load_kw, pv_kw, price,
                net_load_after_pv, useful_discharge, charge_kw,
                grid_kw, baseline_grid_kw, pump_power_kw,
            )
        elif self.allow_grid_trading:
            net_load = load_kw - pv_kw + net_power_kw
            reward = self._calculate_reward_phase1(action_kw, net_load, price)
        else:
            reward = self._calculate_reward_p302(
                action_kw, load_kw, pv_kw, price,
                net_load_after_pv, useful_discharge, charge_kw,
                grid_kw, baseline_grid_kw, pump_power_kw,
            )
        
        if self.use_flow_rate_action:
            if abs(applied_action_kw) > 1e-9:
                pump_ratio = pump_power_kw / max(abs(applied_action_kw), 1e-9)
                reward -= 0.1 * pump_ratio
            if flow_action < 0.1:
                R_eq = self._equivalent_resistance(flow_action)
                R_penalty = (R_eq / self.flow_R_base_ohm - 1.0) * 0.01
                reward -= R_penalty
        
        if energy_violate_kwh > 0.0:
            penalty_per_kwh = float(getattr(self, 'realized_violation_penalty', 20.0))
            scale_guard = max(float(self.reward_scaling), 1e-9)
            reward -= (penalty_per_kwh * energy_violate_kwh) / scale_guard
        
        if useful_discharge > 0:
            if self.discharge_mode == 'partial_assist' and grid_kw > 1e-9:
                situation_code = 2  # Battery assists load but grid still supports the rest
            else:
                situation_code = 1  # Battery solo successfully supplies load
        elif action_kw < -1e-6 and voltage_blocked:
            situation_code = 4  # Invalid discharge request is blocked before execution
        elif actual_action_kw > 1e-6:
            situation_code = 3  # Charging
        else:
            situation_code = 4  # Standby
        
        self.current_step += 1
        self.prev_action_kw = action_kw
        
        done = self.current_step >= self.episode_length
        flow_too_low_active = int(
            self.use_flow_rate_action
            and (not self.flow_operating_rule_enabled)
            and abs(actual_action_kw) > max(self.action_dead_zone_kw, 1e-9)
            and flow_action < self.flow_min_active_fraction
        )
        required_flow_fraction = 0.0
        if self.use_flow_rate_action and abs(actual_action_kw) > max(self.action_dead_zone_kw, 1e-9):
            if actual_action_kw > 0:
                required_flow_fraction = abs(actual_action_kw) / max(self.battery_charge_power_kw, 1e-9)
            else:
                required_flow_fraction = abs(actual_action_kw) / max(self.battery_discharge_power_kw, 1e-9)
        flow_power_mismatch = int(
            self.use_flow_rate_action
            and (not self.flow_operating_rule_enabled)
            and required_flow_fraction > 0.0
            and flow_action + 1e-9 < min(1.0, required_flow_fraction)
        )
        
        info = {
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'grid_import_kw': grid_import_kw,
            'grid_export_kw': grid_export_kw,
            'export_revenue': export_revenue,
            'feed_in_tariff_ratio': self.feed_in_tariff_ratio,
            'soc_violations': self.soc_violations,
            'action_violations': self.action_violations,
            'strict_soc_violation_steps': self.strict_soc_violation_steps,
            'strict_soc_violation_duration_h': self.strict_soc_violation_duration_h,
            'strict_soc_violation_kwh': self.strict_soc_violation_kwh,
            'strict_soc_violation_max_kwh': self.strict_soc_violation_max_kwh,
            'strict_soc_violation_excess_kwh': strict_violation_excess_kwh,
            'clip_soc_to_bounds': int(self.clip_soc_to_bounds),
            'current_soc': self.current_soc,
            'current_soh': self.current_soh,
            'flow_rate_lpm': self.current_flow_rate_lpm,
            'flow_action': flow_action,
            'pump_power_kw': pump_power_kw,
            'costed_pump_power_kw': costed_pump_power_kw,
            'pump_energy_kwh': pump_energy_kwh,
            'pump_cost': costed_pump_power_kw * self.time_step * price,
            'fixed_flow_fraction_when_uncontrolled': self.fixed_flow_fraction_when_uncontrolled,
            'flow_efficiency': flow_eta_modifier,
            'flow_power_limited': flow_power_limited,
            'flow_too_low_active': flow_too_low_active,
            'flow_power_mismatch': flow_power_mismatch,
            'required_flow_fraction': required_flow_fraction,
            'flow_charge_limit_kw': flow_charge_limit_kw,
            'flow_discharge_limit_kw': flow_discharge_limit_kw,
            'net_power_kw': net_power_kw,
            'gross_power_kw': applied_action_kw,
            'R_eq_ohm': self._equivalent_resistance(flow_action) if self.use_flow_rate_action else 0.0,
            'net_load_after_pv': self._last_net_load_after_pv,
            'grid_kw': self._last_grid_kw,
            'useful_discharge': self._last_useful_discharge,
            'pv_to_load': self._last_pv_to_load,
            'pv_to_battery': pv_to_battery,
            'baseline_grid_kw': baseline_grid_kw,
            'price': price,
            'situation_code': situation_code,
            'voltage_blocked': voltage_blocked,
            'blocked_by_pv': blocked_by_pv,
            'blocked_by_load': blocked_by_load,
            'guard_force_charge_low_soc': deployment_guard_flags['guard_force_charge_low_soc'],
            'guard_block_low_soc_discharge': deployment_guard_flags['guard_block_low_soc_discharge'],
            'guard_block_high_soc_charge': deployment_guard_flags['guard_block_high_soc_charge'],
            'guard_block_pv_active_discharge': deployment_guard_flags['guard_block_pv_active_discharge'],
            'guard_block_voltage_cutoff': deployment_guard_flags['guard_block_voltage_cutoff'],
            'guard_block_load_over_discharge_limit': deployment_guard_flags['guard_block_load_over_discharge_limit'],
            'guard_block_discharge_intent_threshold': discharge_intent_below_threshold,
            'voltage_cutoff_active': int(self._deployment_voltage_cutoff_active),
            'discharge_mode': self.discharge_mode,
            'discharge_intent_threshold_kw': self.discharge_intent_threshold_kw,
            'step': self.current_step,
            'load': load_kw,
            'grid_pv_load_kw': load_kw,
            'battery_delivered_load_kw': battery_load_kw,
            'battery_delivered_load_per_group_kw': (
                0.0
                if self.battery_delivered_load_per_group_kw is None
                else self.battery_delivered_load_per_group_kw
            ),
            'pv': pv_kw,
            'obs_load_kw': float(obs_snapshot['load_kw']),
            'obs_pv_kw': float(obs_snapshot['pv_kw']),
            'obs_load_source': obs_snapshot['load_source'],
            'obs_load_groups': int(obs_snapshot['load_groups']),
            'enforce_solo_discharge_load_limit': int(self.enforce_solo_discharge_load_limit),
            'pre_measure_rest_flow_fraction': self.pre_measure_rest_flow_fraction,
            'pre_measure_flow_fraction': self.pre_measure_flow_fraction,
            'pre_measure_seconds': self.pre_measure_seconds,
            'pre_measure_event': int(self.pre_measure_seconds > 0.0),
            'action_kw': action_kw,
            'requested_action_kw': applied_action_kw,
            'applied_action_kw': actual_action_kw,
        }
        
        if self.stress_enable and self.stress_soc_obs_delay > 0:
            self._soc_obs_buffer.pop(0)
            self._soc_obs_buffer.append(self.current_soc)

        return self._get_state(), reward, done, False, info
    
    def render(self):
        """Documentation for this public API is provided in English."""
        pass
    
    def close(self):
        """Documentation for this public API is provided in English."""
        pass


class MicrogridEnvWrapper(gym.Env):
    """
    """
    
    def __init__(self, env: MicrogridEnvironment):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    @property
    def unwrapped(self):
        return self.env


def create_microgrid_env(
    microgrid_id: int = 0,
    episode_length: int = 24,
    time_step: float = 1.0,
    battery_capacity_kwh: float = 100.0,
    battery_power_kw: float = 50.0,
    use_real_data: bool = True,
    ramp_limit_kw: float = None,
    hard_guard: bool = False,
    allow_grid_trading: bool = True,
    **kwargs
) -> MicrogridEnvironment:
    """Documentation for this public API is provided in English."""
    return MicrogridEnvironment(
        microgrid_id=microgrid_id,
        episode_length=episode_length,
        time_step=time_step,
        allow_grid_trading=allow_grid_trading,
        battery_capacity_kwh=battery_capacity_kwh,
        battery_power_kw=battery_power_kw,
        use_real_data=use_real_data,
        ramp_limit_kw=ramp_limit_kw,
        hard_guard=hard_guard,
        **kwargs
    ) 
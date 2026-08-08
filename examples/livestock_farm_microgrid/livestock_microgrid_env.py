from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import pvlib
from gymnasium import spaces


try:
    import openmeteo_requests
except ImportError:  # pragma: no cover - optional runtime dependency
    openmeteo_requests = None

try:
    import requests
except ImportError:  # pragma: no cover - only needed for fallback API calls
    requests = None


@dataclass(frozen=True)
class LocationConfig:
    """Farm location used for solar geometry and weather queries."""

    name: str = "Taoyuan Livestock Farm"
    latitude: float = 25.02702
    longitude: float = 121.12371
    timezone: str = "Asia/Taipei"
    altitude_m: float = 100.0


@dataclass(frozen=True)
class PVConfig:
    """PV array configuration.

    capacity_kwp is the DC nameplate capacity. For an RL environment we convert
    irradiance to AC-like potential with a simple PVWatts-style scaling.
    """

    capacity_kwp: float = 300.0
    surface_tilt: float = 25.0
    surface_azimuth: float = 180.0
    performance_ratio: float = 0.85


@dataclass(frozen=True)
class BatteryConfig:
    """Vanadium redox flow battery configuration."""

    capacity_kwh: float = 100.0
    max_power_kw: float = 50.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soc_min: float = 0.20
    soc_max: float = 0.80

    @property
    def one_way_efficiency(self) -> float:
        return self.charge_efficiency

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency


class LoadGenerator:
    """Generate a realistic 24-hour livestock farm load profile.

    The requested segment ranges and target daily energy are numerically
    inconsistent if every segment bound is enforced strictly:
      min energy = 6*20 + 3*70 + 7*60 + 3*90 + 5*20 = 1120 kWh.

    Therefore the default strategy preserves the farm-like daily shape, injects
    +/-5% noise, fixes the peak at 104 kW, and calibrates the integral to about
    905 kWh. Use strict_segment_bounds=True to enforce all segment ranges and
    accept the physically implied higher daily energy.
    """

    def __init__(
        self,
        dt_minutes: int = 15,
        target_daily_energy_kwh: float = 905.0,
        target_peak_kw: float = 104.0,
        noise_pct: float = 0.05,
        seed: Optional[int] = None,
        strict_segment_bounds: bool = False,
    ) -> None:
        self.dt_minutes = dt_minutes
        self.steps_per_day = int(24 * 60 / dt_minutes)
        self.dt_hours = dt_minutes / 60.0
        self.target_daily_energy_kwh = target_daily_energy_kwh
        self.target_peak_kw = target_peak_kw
        self.noise_pct = noise_pct
        self.strict_segment_bounds = strict_segment_bounds
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        hours = np.arange(self.steps_per_day) * self.dt_hours
        load_kw = np.zeros(self.steps_per_day, dtype=np.float64)

        for i, hour in enumerate(hours):
            load_kw[i] = self._sample_segment_load(hour)

        # Livestock loads are not perfectly smooth, but the noise is bounded.
        noise = self.rng.uniform(1.0 - self.noise_pct, 1.0 + self.noise_pct, size=self.steps_per_day)
        load_kw *= noise

        if self.strict_segment_bounds:
            load_kw = self._clip_to_segment_bounds(hours, load_kw)
            load_kw *= self.target_peak_kw / max(load_kw.max(), 1e-9)
            achieved = float(load_kw.sum() * self.dt_hours)
            if achieved < self.target_daily_energy_kwh:
                warnings.warn(
                    "Strict segment bounds and 905 kWh target are infeasible together; "
                    f"generated {achieved:.1f} kWh after peak normalization.",
                    RuntimeWarning,
                )
        else:
            # Shape-preserving calibration: solve y = a*x + b so that peak and
            # daily energy are both matched. This is the practical RL default.
            load_kw = self._match_peak_and_energy(load_kw)

        index = pd.timedelta_range(start="0min", periods=self.steps_per_day, freq=f"{self.dt_minutes}min")
        return pd.DataFrame({"time_of_day": index, "hour": hours, "load_kw": load_kw})

    def _sample_segment_load(self, hour: float) -> float:
        if 0 <= hour < 6:
            return self.rng.uniform(20.0, 30.0)
        if 6 <= hour < 9:
            return self.rng.uniform(70.0, 85.0)
        if 9 <= hour < 16:
            # Daytime cooling follows a smooth temperature-like hump.
            cooling_shape = math.sin((hour - 9.0) / 7.0 * math.pi)
            return 60.0 + 15.0 * cooling_shape + self.rng.uniform(-2.0, 2.0)
        if 16 <= hour < 19:
            return self.rng.uniform(90.0, 104.0)
        return self.rng.uniform(20.0, 30.0)

    @staticmethod
    def _clip_to_segment_bounds(hours: np.ndarray, load_kw: np.ndarray) -> np.ndarray:
        clipped = load_kw.copy()
        for i, hour in enumerate(hours):
            if 0 <= hour < 6 or 19 <= hour < 24:
                clipped[i] = np.clip(clipped[i], 20.0, 30.0)
            elif 6 <= hour < 9:
                clipped[i] = np.clip(clipped[i], 70.0, 85.0)
            elif 9 <= hour < 16:
                clipped[i] = np.clip(clipped[i], 60.0, 75.0)
            elif 16 <= hour < 19:
                clipped[i] = np.clip(clipped[i], 90.0, 104.0)
        return clipped

    def _match_peak_and_energy(self, load_kw: np.ndarray) -> np.ndarray:
        # First lock the peak exactly. Then adjust only non-peak points so the
        # integral reaches the requested daily energy without changing the peak.
        calibrated = load_kw * (self.target_peak_kw / max(float(load_kw.max()), 1e-9))
        peak_idx = int(np.argmax(calibrated))
        target_sum = self.target_daily_energy_kwh / self.dt_hours

        for _ in range(200):
            current_sum = float(calibrated.sum())
            diff = current_sum - target_sum
            if abs(diff) < 1e-6:
                break

            if diff > 0.0:
                # Remove energy from all non-peak points that are still above 0.
                active = np.ones_like(calibrated, dtype=bool)
                active[peak_idx] = False
                active &= calibrated > 1e-9
                if not active.any():
                    break
                reduction = diff / active.sum()
                calibrated[active] = np.maximum(calibrated[active] - reduction, 0.0)
            else:
                # Add energy to all non-peak points that are still below the peak.
                active = np.ones_like(calibrated, dtype=bool)
                active[peak_idx] = False
                active &= calibrated < self.target_peak_kw - 1e-9
                if not active.any():
                    break
                addition = (-diff) / active.sum()
                calibrated[active] = np.minimum(calibrated[active] + addition, self.target_peak_kw)

        calibrated[peak_idx] = self.target_peak_kw
        return calibrated


class SolarPredictor:
    """Two-layer solar feature provider.

    Layer 1: clear_sky_power_kw from pvlib solar geometry and clear-sky model.
    Layer 2: weather_adjusted_power_kw from Open-Meteo irradiance/cloud data.

    If the weather API is unavailable, the class falls back to a deterministic
    cloud degradation curve so that RL training remains reproducible.
    """

    def __init__(
        self,
        location: LocationConfig = LocationConfig(),
        pv: PVConfig = PVConfig(),
        dt_minutes: int = 15,
        use_weather_api: bool = True,
        weather_mode: str = "auto",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.location = location
        self.pv = pv
        self.dt_minutes = dt_minutes
        self.use_weather_api = use_weather_api
        self.weather_mode = weather_mode
        self.cache_dir = cache_dir

    def predict_day(self, day: str) -> pd.DataFrame:
        times = pd.date_range(
            start=pd.Timestamp(f"{day} 00:00:00", tz=self.location.timezone),
            periods=int(24 * 60 / self.dt_minutes),
            freq=f"{self.dt_minutes}min",
        )
        clear = self._clear_sky_power(times)
        weather = self._fetch_weather(day)

        if weather is None:
            cloud_cover = self._fallback_cloud_cover(times)
            weather_adjusted_kw = clear["clear_sky_power_kw"] * (1.0 - 0.75 * cloud_cover / 100.0)
            temperature_c = np.full(len(times), 25.0)
        else:
            weather_15min = self._resample_weather(weather, times)
            weather_adjusted_kw = self._weather_adjusted_power(times, weather_15min)
            cloud_cover = weather_15min["cloud_cover"].to_numpy()
            temperature_c = weather_15min["temperature_2m"].to_numpy()

        result = clear.copy()
        result["weather_adjusted_power_kw"] = np.clip(weather_adjusted_kw, 0.0, self.pv.capacity_kwp)
        result["cloud_cover_pct"] = np.clip(cloud_cover, 0.0, 100.0)
        result["temperature_c"] = temperature_c
        result.index.name = "timestamp"
        return result

    def _clear_sky_power(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        site = pvlib.location.Location(
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            tz=self.location.timezone,
            altitude=self.location.altitude_m,
            name=self.location.name,
        )
        solar_position = site.get_solarposition(times)
        clear_sky = site.get_clearsky(times, model="ineichen")
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.pv.surface_tilt,
            surface_azimuth=self.pv.surface_azimuth,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=clear_sky["dni"],
            ghi=clear_sky["ghi"],
            dhi=clear_sky["dhi"],
        )
        clear_kw = self.pv.capacity_kwp * self.pv.performance_ratio * poa["poa_global"].clip(lower=0.0) / 1000.0
        return pd.DataFrame(
            {
                "solar_zenith_deg": solar_position["apparent_zenith"],
                "clear_sky_power_kw": np.clip(clear_kw, 0.0, self.pv.capacity_kwp),
            },
            index=times,
        )

    def _fetch_weather(self, day: str) -> Optional[pd.DataFrame]:
        if not self.use_weather_api:
            return None

        mode = self._resolve_weather_mode(day)
        cache_path = self._cache_path(day, mode)
        if cache_path is not None and cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()

        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            if mode == "archive"
            else "https://api.open-meteo.com/v1/forecast"
        )
        params = {
            "latitude": self.location.latitude,
            "longitude": self.location.longitude,
            "timezone": self.location.timezone,
            "start_date": day,
            "end_date": day,
            "hourly": [
                "shortwave_radiation",
                "direct_normal_irradiance",
                "diffuse_radiation",
                "cloud_cover",
                "temperature_2m",
            ],
        }

        try:
            if openmeteo_requests is not None:
                weather = self._fetch_weather_openmeteo_requests(url, params)
            else:
                weather = self._fetch_weather_requests(url, params)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                weather.reset_index().to_csv(cache_path, index=False)
            return weather
        except Exception as exc:  # pragma: no cover - network dependent
            warnings.warn(f"Weather API unavailable, using fallback cloud model: {exc}", RuntimeWarning)
            return None

    def _cache_path(self, day: str, mode: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe_name = f"open_meteo_{mode}_{day}.csv"
        return self.cache_dir / safe_name

    def _fetch_weather_openmeteo_requests(self, url: str, params: Dict[str, Any]) -> pd.DataFrame:
        client = openmeteo_requests.Client()
        responses = client.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert(self.location.timezone),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert(self.location.timezone),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        variable_names = params["hourly"]
        data = {"timestamp": times}
        for idx, name in enumerate(variable_names):
            data[name] = hourly.Variables(idx).ValuesAsNumpy()
        return pd.DataFrame(data).set_index("timestamp").sort_index()

    def _fetch_weather_requests(self, url: str, params: Dict[str, Any]) -> pd.DataFrame:
        if requests is None:
            raise RuntimeError("Neither openmeteo-requests nor requests is installed.")
        request_params = params.copy()
        request_params["hourly"] = ",".join(params["hourly"])
        response = requests.get(url, params=request_params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        hourly = payload["hourly"]
        df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
        for name in params["hourly"]:
            df[name] = pd.to_numeric(pd.Series(hourly.get(name, [])), errors="coerce")
        df["timestamp"] = df["timestamp"].dt.tz_localize(self.location.timezone)
        return df.set_index("timestamp").sort_index()

    def _resolve_weather_mode(self, day: str) -> str:
        if self.weather_mode in {"archive", "forecast"}:
            return self.weather_mode
        requested = pd.Timestamp(day).date()
        today = date_cls.today()
        return "archive" if requested < today else "forecast"

    @staticmethod
    def _resample_weather(weather: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
        combined_index = weather.index.union(times)
        return (
            weather.reindex(combined_index)
            .sort_index()
            .interpolate(method="time")
            .reindex(times)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

    def _weather_adjusted_power(self, times: pd.DatetimeIndex, weather: pd.DataFrame) -> np.ndarray:
        site = pvlib.location.Location(
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            tz=self.location.timezone,
            altitude=self.location.altitude_m,
        )
        solar_position = site.get_solarposition(times)
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.pv.surface_tilt,
            surface_azimuth=self.pv.surface_azimuth,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=weather["direct_normal_irradiance"].clip(lower=0.0),
            ghi=weather["shortwave_radiation"].clip(lower=0.0),
            dhi=weather["diffuse_radiation"].clip(lower=0.0),
        )
        temp_coeff = np.clip(1.0 - 0.004 * (weather["temperature_2m"].to_numpy() - 25.0), 0.75, 1.10)
        power_kw = self.pv.capacity_kwp * self.pv.performance_ratio * poa["poa_global"].clip(lower=0.0) / 1000.0
        return power_kw.to_numpy() * temp_coeff

    @staticmethod
    def _fallback_cloud_cover(times: pd.DatetimeIndex) -> np.ndarray:
        hours = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
        daily_wave = 35.0 + 20.0 * np.sin((hours - 7.0) / 24.0 * 2.0 * np.pi)
        return np.clip(daily_wave, 5.0, 85.0)


class MicrogridEnv(gym.Env):
    """Gymnasium environment for a livestock farm microgrid.

    Action:
      Box([-1], [1])
      positive = charge battery, negative = discharge battery.

    Observation:
      [soc, hour_sin, hour_cos, load_kw/104, observed_pv_kw/PV_kWp,
       clear_sky_power_kw/PV_kWp, weather_adjusted_power_kw/PV_kWp,
       grid_price_norm, cloud_cover_pct/100]

    The environment is ready for Stable Baselines 3 and RLlib because it follows
    Gymnasium's reset/step signatures and uses continuous Box spaces.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        start_date: str = "2026-05-05",
        seed: Optional[int] = None,
        use_weather_api: bool = False,
        strict_load_segment_bounds: bool = False,
        random_date_start: Optional[str] = None,
        random_date_days: int = 0,
        load_daily_energy_kwh: float = 905.8064516129032,
        load_peak_kw: float = 104.0,
    ) -> None:
        super().__init__()
        self.dt_minutes = 15
        self.dt_hours = self.dt_minutes / 60.0
        self.steps_per_day = int(24 * 60 / self.dt_minutes)
        self.start_date = start_date
        self.rng = np.random.default_rng(seed)
        self.random_dates = self._build_random_dates(random_date_start, random_date_days)
        self.load_daily_energy_kwh = load_daily_energy_kwh
        self.load_peak_kw = load_peak_kw

        self.location = LocationConfig()
        self.pv_config = PVConfig()
        self.battery = BatteryConfig()
        self.load_generator = LoadGenerator(
            dt_minutes=self.dt_minutes,
            target_daily_energy_kwh=self.load_daily_energy_kwh,
            target_peak_kw=self.load_peak_kw,
            seed=seed,
            strict_segment_bounds=strict_load_segment_bounds,
        )
        self.solar_predictor = SolarPredictor(
            location=self.location,
            pv=self.pv_config,
            dt_minutes=self.dt_minutes,
            use_weather_api=use_weather_api,
            cache_dir=Path(__file__).resolve().parent / "weather_cache",
        )

        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.profile: pd.DataFrame
        self.step_idx = 0
        self.soc = 0.50
        self.last_info: Dict[str, Any] = {}
        self._build_profiles()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        if "start_date" in options:
            self.start_date = options["start_date"]
        elif self.random_dates:
            self.start_date = str(self.rng.choice(self.random_dates))
        self._build_profiles()

        self.step_idx = 0
        if "initial_soc" in options:
            self.soc = float(np.clip(options["initial_soc"], self.battery.soc_min, self.battery.soc_max))
        else:
            self.soc = float(self.rng.uniform(0.35, 0.65))

        self.last_info = {"soc": self.soc}
        return self._get_obs(), self.last_info.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        row = self.profile.iloc[self.step_idx]
        command_kw = float(np.clip(action[0], -1.0, 1.0) * self.battery.max_power_kw)
        load_kw = float(row["load_kw"])
        pv_potential_kw = float(row["weather_adjusted_power_kw"])
        price = float(row["grid_price_twd_per_kwh"])

        battery_power_kw = 0.0  # positive charge, negative discharge
        grid_import_kw = 0.0
        pv_used_kw = 0.0
        safety_penalty = 0.0
        infeasible_discharge = False
        attempted_soc_violation = False
        situation = "standby"

        if command_kw > 1e-6:
            # Charging is limited by battery power and remaining safe capacity.
            room_kwh = (self.battery.soc_max - self.soc) * self.battery.capacity_kwh
            max_charge_kw_by_soc = room_kwh / (self.battery.one_way_efficiency * self.dt_hours)
            battery_power_kw = min(command_kw, self.battery.max_power_kw, max_charge_kw_by_soc)
            if command_kw > max_charge_kw_by_soc + 1e-6:
                attempted_soc_violation = True
                safety_penalty += 25.0

            # PV is consumed by load first; remaining charge can come from PV surplus
            # and then grid. This exposes economic trade-offs to the agent.
            pv_to_load_kw = min(pv_potential_kw, load_kw)
            pv_surplus_kw = max(pv_potential_kw - pv_to_load_kw, 0.0)
            pv_to_battery_kw = min(pv_surplus_kw, battery_power_kw)
            grid_to_battery_kw = max(battery_power_kw - pv_to_battery_kw, 0.0)
            pv_used_kw = pv_to_load_kw + pv_to_battery_kw
            grid_import_kw = max(load_kw - pv_to_load_kw, 0.0) + grid_to_battery_kw
            self.soc += battery_power_kw * self.battery.one_way_efficiency * self.dt_hours / self.battery.capacity_kwh
            situation = "charge"

        elif command_kw < -1e-6:
            requested_discharge_kw = abs(command_kw)
            available_energy_kwh = max((self.soc - self.battery.soc_min) * self.battery.capacity_kwh, 0.0)
            max_discharge_kw_by_soc = available_energy_kwh * self.battery.discharge_efficiency / self.dt_hours
            available_discharge_kw = min(self.battery.max_power_kw, max_discharge_kw_by_soc)

            actual_discharge_kw = min(requested_discharge_kw, available_discharge_kw, load_kw)
            if requested_discharge_kw > available_discharge_kw + 1e-6:
                attempted_soc_violation = True
                safety_penalty += 25.0

            # Planning-scale livestock microgrid: partial battery support is
            # allowed. The battery offsets part of the load, and the grid covers
            # any remaining demand.
            battery_power_kw = -actual_discharge_kw
            pv_used_kw = min(pv_potential_kw, max(load_kw - actual_discharge_kw, 0.0))
            grid_import_kw = max(load_kw - actual_discharge_kw - pv_used_kw, 0.0)
            self.soc -= actual_discharge_kw * self.dt_hours / (
                self.battery.discharge_efficiency * self.battery.capacity_kwh
            )
            situation = "battery_support" if actual_discharge_kw > 1e-6 else "standby"

        else:
            pv_used_kw = min(pv_potential_kw, load_kw)
            grid_import_kw = max(load_kw - pv_used_kw, 0.0)
            situation = "standby"

        if self.soc < self.battery.soc_min - 1e-9 or self.soc > self.battery.soc_max + 1e-9:
            attempted_soc_violation = True
            safety_penalty += 100.0
        self.soc = float(np.clip(self.soc, self.battery.soc_min, self.battery.soc_max))

        reward = self._calculate_reward(
            grid_import_kw=grid_import_kw,
            price=price,
            pv_used_kw=pv_used_kw,
            battery_power_kw=battery_power_kw,
            safety_penalty=safety_penalty,
        )

        info = {
            "timestamp": row.name,
            "situation": situation,
            "soc": self.soc,
            "command_kw": command_kw,
            "battery_power_kw": battery_power_kw,
            "load_kw": load_kw,
            "observed_pv_kw": self._observed_pv(pv_potential_kw, load_kw, battery_power_kw),
            "clear_sky_power_kw": float(row["clear_sky_power_kw"]),
            "weather_adjusted_power_kw": pv_potential_kw,
            "grid_import_kw": grid_import_kw,
            "pv_used_kw": pv_used_kw,
            "grid_price_twd_per_kwh": price,
            "energy_cost_twd": grid_import_kw * self.dt_hours * price,
            "infeasible_discharge": infeasible_discharge,
            "attempted_soc_violation": attempted_soc_violation,
            "safety_penalty": safety_penalty,
        }
        self.last_info = info

        self.step_idx += 1
        terminated = self.step_idx >= self.steps_per_day
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, info

    def _build_profiles(self) -> None:
        load_df = self.load_generator.generate()
        solar_df = self.solar_predictor.predict_day(self.start_date)
        profile = solar_df.copy()
        profile["load_kw"] = load_df["load_kw"].to_numpy()
        profile["grid_price_twd_per_kwh"] = [self._tou_price(ts) for ts in profile.index]
        self.profile = profile

    @staticmethod
    def _build_random_dates(start: Optional[str], days: int) -> list[str]:
        if not start or days <= 0:
            return []
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start, periods=days, freq="D")]

    def _get_obs(self) -> np.ndarray:
        row = self.profile.iloc[self.step_idx]
        ts = row.name
        hour = ts.hour + ts.minute / 60.0
        observed_pv_kw = self._observed_pv(float(row["weather_adjusted_power_kw"]), float(row["load_kw"]), 0.0)
        obs = np.array(
            [
                self.soc,
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
                float(row["load_kw"]) / self.load_peak_kw,
                observed_pv_kw / self.pv_config.capacity_kwp,
                float(row["clear_sky_power_kw"]) / self.pv_config.capacity_kwp,
                float(row["weather_adjusted_power_kw"]) / self.pv_config.capacity_kwp,
                float(row["grid_price_twd_per_kwh"]) / 10.0,
                float(row["cloud_cover_pct"]) / 100.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _calculate_reward(
        self,
        grid_import_kw: float,
        price: float,
        pv_used_kw: float,
        battery_power_kw: float,
        safety_penalty: float,
    ) -> float:
        # Primary objective: minimize grid energy cost.
        grid_cost = grid_import_kw * self.dt_hours * price

        # Secondary objective: prefer local PV utilization, especially charging
        # from otherwise curtailed solar energy.
        pv_bonus = 0.02 * pv_used_kw * self.dt_hours

        # Mild cycling cost prevents meaningless high-frequency battery motion.
        cycling_cost = 0.005 * abs(battery_power_kw) * self.dt_hours

        return -grid_cost + pv_bonus - cycling_cost - safety_penalty

    @staticmethod
    def _tou_price(timestamp: pd.Timestamp) -> float:
        hour = timestamp.hour + timestamp.minute / 60.0
        if 0 <= hour < 9:
            return 2.06
        if 16 <= hour < 22:
            return 7.13
        return 4.69

    @staticmethod
    def _observed_pv(pv_potential_kw: float, load_kw: float, battery_power_kw: float) -> float:
        # Demand-censored observation: measured PV cannot exceed what the farm
        # and battery currently request from the PV/bus.
        pv_demand_kw = load_kw + max(battery_power_kw, 0.0)
        return min(pv_potential_kw, pv_demand_kw)


if __name__ == "__main__":
    env = MicrogridEnv(use_weather_api=False, seed=7)
    obs, info = env.reset(seed=7, options={"initial_soc": 0.50})
    total_reward = 0.0
    for _ in range(env.steps_per_day):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    energy = float(env.profile["load_kw"].sum() * env.dt_hours)
    peak = float(env.profile["load_kw"].max())
    print(f"Smoke test finished. reward={total_reward:.2f}, load_energy={energy:.1f} kWh, peak={peak:.1f} kW")

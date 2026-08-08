from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pvlib
import requests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "figures"
FONT_SCALE = 1.5


@dataclass(frozen=True)
class LocationConfig:
    name: str = "Hong-Zhi Livestock Farm"
    latitude: float = 25.02702
    longitude: float = 121.12371
    timezone: str = "Asia/Taipei"
    altitude_m: float = 100.0


@dataclass(frozen=True)
class PVHardwareConfig:
    surface_tilt: float = 25.0
    surface_azimuth: float = 180.0
    panel_area_m2: float = 2.0
    panel_efficiency: float = 0.20


def build_time_index(
    date: str,
    timezone: str,
    start_time: str = "05:00",
    end_time: str = "19:00",
    freq: str = "15min",
) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{date} {start_time}", tz=timezone)
    end = pd.Timestamp(f"{date} {end_time}", tz=timezone)
    return pd.date_range(start=start, end=end, freq=freq)


def open_meteo_endpoint(mode: str) -> str:
    if mode == "archive":
        return "https://archive-api.open-meteo.com/v1/archive"
    if mode == "forecast":
        return "https://api.open-meteo.com/v1/forecast"
    raise ValueError(f"Unsupported Open-Meteo mode: {mode}")


def fetch_open_meteo_solar_weather(
    date: str,
    location_cfg: LocationConfig,
    mode: str = "forecast",
    timeout_sec: int = 30,
) -> pd.DataFrame:
    """
    Fetch hourly weather/irradiance data from Open-Meteo.

    Variables:
      - shortwave_radiation: global horizontal irradiance proxy (GHI), W/m^2
      - direct_normal_irradiance: DNI, W/m^2
      - diffuse_radiation: DHI-like diffuse irradiance, W/m^2
      - cloud_cover: total cloud cover, %
      - precipitation: total precipitation, mm
      - rain: rain amount, mm
      - temperature_2m: ambient temperature, degC

    Use `mode=archive` for historical records and `mode=forecast` for future/current days.
    """

    hourly_vars = [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "cloud_cover",
        "precipitation",
        "rain",
        "temperature_2m",
    ]
    params: Dict[str, Any] = {
        "latitude": location_cfg.latitude,
        "longitude": location_cfg.longitude,
        "timezone": location_cfg.timezone,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(hourly_vars),
    }

    response = requests.get(open_meteo_endpoint(mode), params=params, timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise RuntimeError(f"Open-Meteo response has no hourly data: {payload}")

    hourly = payload["hourly"]
    df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for variable in hourly_vars:
        df[variable] = pd.to_numeric(pd.Series(hourly.get(variable, [])), errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_localize(location_cfg.timezone)
    return df.set_index("timestamp").sort_index()


def resample_weather_to_times(weather_df: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate hourly Open-Meteo data onto the RL/control time grid."""

    combined_index = weather_df.index.union(times)
    interpolated = (
        weather_df.reindex(combined_index)
        .sort_index()
        .interpolate(method="time")
        .reindex(times)
    )
    return interpolated.fillna(0.0)


def calculate_pv_potential(
    times: pd.DatetimeIndex,
    weather_df: pd.DataFrame,
    location_cfg: LocationConfig,
    hardware_cfg: PVHardwareConfig,
) -> pd.DataFrame:
    """Calculate clear-sky and weather-adjusted PV power potential."""

    site = pvlib.location.Location(
        latitude=location_cfg.latitude,
        longitude=location_cfg.longitude,
        tz=location_cfg.timezone,
        altitude=location_cfg.altitude_m,
        name=location_cfg.name,
    )
    solar_position = site.get_solarposition(times)
    clear_sky = site.get_clearsky(times, model="ineichen")

    weather = resample_weather_to_times(weather_df, times)
    ghi = weather["shortwave_radiation"].clip(lower=0.0)
    dni = weather["direct_normal_irradiance"].clip(lower=0.0)
    dhi = weather["diffuse_radiation"].clip(lower=0.0)

    clear_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=hardware_cfg.surface_tilt,
        surface_azimuth=hardware_cfg.surface_azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clear_sky["dni"],
        ghi=clear_sky["ghi"],
        dhi=clear_sky["dhi"],
    )
    weather_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=hardware_cfg.surface_tilt,
        surface_azimuth=hardware_cfg.surface_azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
    )

    scale = hardware_cfg.panel_area_m2 * hardware_cfg.panel_efficiency
    result = pd.DataFrame(
        {
            "solar_zenith_deg": solar_position["apparent_zenith"],
            "clear_sky_ghi_w_m2": clear_sky["ghi"],
            "weather_ghi_w_m2": ghi,
            "weather_dni_w_m2": dni,
            "weather_dhi_w_m2": dhi,
            "cloud_cover_pct": weather["cloud_cover"],
            "precipitation_mm": weather["precipitation"],
            "rain_mm": weather["rain"],
            "temperature_2m_c": weather["temperature_2m"],
            "clear_sky_poa_w_m2": clear_poa["poa_global"].clip(lower=0.0),
            "weather_poa_w_m2": weather_poa["poa_global"].clip(lower=0.0),
        },
        index=times,
    )
    result["clear_sky_power_w"] = result["clear_sky_poa_w_m2"] * scale
    result["weather_adjusted_power_w"] = result["weather_poa_w_m2"] * scale
    result["weather_to_clear_sky_ratio"] = (
        result["weather_adjusted_power_w"]
        / result["clear_sky_power_w"].where(result["clear_sky_power_w"] > 1e-9)
    ).clip(lower=0.0, upper=1.5)
    result.index.name = "timestamp"
    return result


def summarize(result: pd.DataFrame, power_column: str) -> tuple[pd.Timestamp, float]:
    peak_time = result[power_column].idxmax()
    peak_power = float(result.loc[peak_time, power_column])
    return peak_time, peak_power


def plot_pv_potential(
    result: pd.DataFrame,
    output_path: Optional[Path],
    title: str,
    show: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10 * FONT_SCALE,
            "axes.titlesize": 13 * FONT_SCALE,
            "axes.labelsize": 11 * FONT_SCALE,
            "legend.fontsize": 9 * FONT_SCALE,
            "xtick.labelsize": 9 * FONT_SCALE,
            "ytick.labelsize": 9 * FONT_SCALE,
        }
    )
    fig, (ax_power, ax_irr, ax_weather) = plt.subplots(
        3,
        1,
        figsize=(13.5, 10.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.35, 1.25]},
    )

    ax_power.plot(
        result.index,
        result["clear_sky_power_w"],
        color="#f2a900",
        linewidth=2.0,
        linestyle="--",
        label="Clear-sky upper bound",
    )
    ax_power.plot(
        result.index,
        result["weather_adjusted_power_w"],
        color="#1f77b4",
        linewidth=2.4,
        label="Weather-adjusted estimate",
    )
    ax_power.fill_between(
        result.index,
        result["weather_adjusted_power_w"],
        color="#1f77b4",
        alpha=0.18,
    )
    ax_power.set_ylabel("PV Power (W)")
    ax_power.grid(True, linestyle="--", alpha=0.25)
    ax_power.legend(loc="upper right")

    ax_irr.plot(
        result.index,
        result["weather_ghi_w_m2"],
        color="#2ca02c",
        linewidth=2.0,
        label="GHI / shortwave",
    )
    ax_irr.plot(
        result.index,
        result["weather_dni_w_m2"],
        color="#ff7f0e",
        linewidth=1.6,
        label="DNI",
    )
    ax_irr.plot(
        result.index,
        result["weather_dhi_w_m2"],
        color="#9467bd",
        linewidth=1.6,
        label="DHI / diffuse",
    )
    ax_irr.set_ylabel("Irradiance (W/m2)")
    ax_irr.grid(True, linestyle="--", alpha=0.25)
    ax_irr.legend(loc="upper right", ncol=3)

    # Open-Meteo precipitation/rain are hourly values. Plot one bar per hour so
    # the visual evidence and total rainfall are not inflated by 15-minute interpolation.
    hourly_precip = result["precipitation_mm"].resample("1h").first()
    hourly_rain = result["rain_mm"].resample("1h").first()
    bar_width_days = 0.032  # ~46 minutes; readable hourly bars on a day plot
    ax_weather.bar(
        hourly_precip.index,
        hourly_precip,
        width=bar_width_days,
        color="#4c78a8",
        alpha=0.55,
        label="Precipitation (mm)",
    )
    ax_weather.bar(
        hourly_rain.index,
        hourly_rain,
        width=bar_width_days * 0.65,
        color="#1f4e79",
        alpha=0.65,
        label="Rain (mm)",
    )
    ax_weather.set_ylabel("Rain / Precip. (mm)")
    ax_weather.grid(True, linestyle="--", alpha=0.25)

    ax_cloud = ax_weather.twinx()
    ax_cloud.plot(
        result.index,
        result["cloud_cover_pct"],
        color="#777777",
        alpha=0.55,
        linewidth=1.5,
        label="Cloud cover (%)",
    )
    ax_cloud.set_ylabel("Cloud Cover (%)")
    ax_cloud.set_ylim(0, 100)

    lines1, labels1 = ax_weather.get_legend_handles_labels()
    lines2, labels2 = ax_cloud.get_legend_handles_labels()
    ax_weather.legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=3)
    ax_weather.set_xlabel("Time")
    ax_power.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220)
        print(f"Saved plot: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Open-Meteo weather data and calculate weather-adjusted PV potential."
    )
    parser.add_argument("--date", default="2026-05-06")
    parser.add_argument("--mode", choices=["forecast", "archive"], default="forecast")
    parser.add_argument("--start-time", default="05:00")
    parser.add_argument("--end-time", default="19:00")
    parser.add_argument("--freq", default="15min")
    parser.add_argument("--latitude", type=float, default=25.02702)
    parser.add_argument("--longitude", type=float, default=121.12371)
    parser.add_argument("--timezone", default="Asia/Taipei")
    parser.add_argument("--altitude-m", type=float, default=100.0)
    parser.add_argument("--surface-tilt", type=float, default=25.0)
    parser.add_argument("--surface-azimuth", type=float, default=180.0)
    parser.add_argument("--panel-area-m2", type=float, default=2.0)
    parser.add_argument("--panel-efficiency", type=float, default=0.20)
    parser.add_argument("--csv-output", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    location_cfg = LocationConfig(
        latitude=args.latitude,
        longitude=args.longitude,
        timezone=args.timezone,
        altitude_m=args.altitude_m,
    )
    hardware_cfg = PVHardwareConfig(
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        panel_area_m2=args.panel_area_m2,
        panel_efficiency=args.panel_efficiency,
    )
    times = build_time_index(
        args.date,
        timezone=location_cfg.timezone,
        start_time=args.start_time,
        end_time=args.end_time,
        freq=args.freq,
    )
    weather_df = fetch_open_meteo_solar_weather(args.date, location_cfg, mode=args.mode)
    result = calculate_pv_potential(times, weather_df, location_cfg, hardware_cfg)

    clear_peak_time, clear_peak = summarize(result, "clear_sky_power_w")
    weather_peak_time, weather_peak = summarize(result, "weather_adjusted_power_w")
    mean_ratio = float(result["weather_to_clear_sky_ratio"].dropna().mean())

    print("Weather-adjusted PV potential")
    print(f"  Source   : Open-Meteo {args.mode}")
    print(f"  Date     : {args.date}")
    print(f"  ClearSky : {clear_peak:.2f} W at {clear_peak_time.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"  Weather  : {weather_peak:.2f} W at {weather_peak_time.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"  Avg weather / clear-sky ratio: {mean_ratio:.3f}")
    total_precip = float(result["precipitation_mm"].resample("1h").first().sum())
    total_rain = float(result["rain_mm"].resample("1h").first().sum())
    print(f"  Total precipitation: {total_precip:.2f} mm")
    print(f"  Total rain         : {total_rain:.2f} mm")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(csv_path)
        print(f"Saved CSV: {csv_path}")

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"weather_adjusted_pv_{args.mode}_{args.date}.png"
    )
    plot_pv_potential(
        result,
        output_path=output_path,
        title=f"Weather-Adjusted PV Potential ({args.date}, Open-Meteo {args.mode})",
        show=args.show,
    )


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

try:
    import pvlib
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "Missing dependency: pvlib. Install it with `pip install pvlib` "
        "or `pip install -r requirements.txt`."
    ) from exc


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "figures"


@dataclass(frozen=True)
class LocationConfig:
    """Site information needed by pvlib."""

    name: str = "Hong-Zhi Livestock Farm"
    latitude: float = 25.02702
    longitude: float = 121.12371
    timezone: str = "Asia/Taipei"
    altitude_m: float = 100.0


@dataclass(frozen=True)
class PVHardwareConfig:
    """Panel-level assumptions. Replace these values once specs are known."""

    surface_tilt: float = 25.0
    surface_azimuth: float = 180.0
    panel_area_m2: float = 2.0
    panel_efficiency: float = 0.20


def build_time_index(
    date: str = "2026-05-06",
    timezone: str = "Asia/Taipei",
    start_time: str = "05:00",
    end_time: str = "19:00",
    freq: str = "15min",
) -> pd.DatetimeIndex:
    """Create a timezone-aware time index for one clear-sky analysis day."""

    start = pd.Timestamp(f"{date} {start_time}", tz=timezone)
    end = pd.Timestamp(f"{date} {end_time}", tz=timezone)
    return pd.date_range(start=start, end=end, freq=freq)


def calculate_clear_sky_pv_upper_bound(
    times: pd.DatetimeIndex,
    location_cfg: LocationConfig,
    hardware_cfg: PVHardwareConfig,
) -> pd.DataFrame:
    """
    Calculate clear-sky irradiance, POA irradiance, and theoretical panel power.

    The output is a theoretical upper bound under clear-sky conditions. It does
    not include inverter losses, wiring losses, temperature derating, dust,
    shading, MPPT clipping, or demand-censored bus observations.
    """

    site = pvlib.location.Location(
        latitude=location_cfg.latitude,
        longitude=location_cfg.longitude,
        tz=location_cfg.timezone,
        altitude=location_cfg.altitude_m,
        name=location_cfg.name,
    )

    solar_position = site.get_solarposition(times)
    clear_sky = site.get_clearsky(times, model="ineichen")

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=hardware_cfg.surface_tilt,
        surface_azimuth=hardware_cfg.surface_azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clear_sky["dni"],
        ghi=clear_sky["ghi"],
        dhi=clear_sky["dhi"],
    )

    theoretical_power_w = (
        poa["poa_global"].clip(lower=0.0)
        * hardware_cfg.panel_area_m2
        * hardware_cfg.panel_efficiency
    )

    result = pd.DataFrame(
        {
            "solar_zenith_deg": solar_position["apparent_zenith"],
            "solar_azimuth_deg": solar_position["azimuth"],
            "ghi_w_m2": clear_sky["ghi"],
            "dni_w_m2": clear_sky["dni"],
            "dhi_w_m2": clear_sky["dhi"],
            "poa_global_w_m2": poa["poa_global"],
            "theoretical_power_w": theoretical_power_w,
        },
        index=times,
    )
    result.index.name = "timestamp"
    return result


def summarize_peak(result: pd.DataFrame) -> tuple[pd.Timestamp, float]:
    """Return peak theoretical power timestamp and value."""

    peak_time = result["theoretical_power_w"].idxmax()
    peak_power_w = float(result.loc[peak_time, "theoretical_power_w"])
    return peak_time, peak_power_w


def plot_theoretical_power(
    result: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot theoretical clear-sky PV power over the selected day."""

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        result.index,
        result["theoretical_power_w"],
        color="#f2a900",
        linewidth=2.4,
        label="Theoretical Power Output (W)",
    )
    ax.fill_between(
        result.index,
        result["theoretical_power_w"],
        color="#f2a900",
        alpha=0.22,
    )
    ax.set_title("Clear-Sky Theoretical PV Power Upper Bound")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate clear-sky theoretical PV power upper bound with pvlib."
    )
    parser.add_argument("--date", default="2026-05-06", help="Analysis date, e.g. 2026-05-06")
    parser.add_argument("--start-time", default="05:00", help="Start time in local timezone")
    parser.add_argument("--end-time", default="19:00", help="End time in local timezone")
    parser.add_argument("--freq", default="15min", help="DatetimeIndex frequency")
    parser.add_argument("--latitude", type=float, default=25.02702)
    parser.add_argument("--longitude", type=float, default=121.12371)
    parser.add_argument("--timezone", default="Asia/Taipei")
    parser.add_argument("--altitude-m", type=float, default=100.0)
    parser.add_argument("--surface-tilt", type=float, default=25.0)
    parser.add_argument("--surface-azimuth", type=float, default=180.0)
    parser.add_argument("--panel-area-m2", type=float, default=2.0)
    parser.add_argument("--panel-efficiency", type=float, default=0.20)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "clear_sky_theoretical_pv_2026-05-06.png"),
        help="Output plot path. Use an empty string to skip saving.",
    )
    parser.add_argument("--csv-output", default="", help="Optional CSV output path")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
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
        date=args.date,
        timezone=location_cfg.timezone,
        start_time=args.start_time,
        end_time=args.end_time,
        freq=args.freq,
    )
    result = calculate_clear_sky_pv_upper_bound(times, location_cfg, hardware_cfg)

    peak_time, peak_power_w = summarize_peak(result)
    print("Clear-sky theoretical PV upper bound")
    print(f"  Location : {location_cfg.name}")
    print(f"  Date     : {args.date}")
    print(f"  Tilt/Az  : {hardware_cfg.surface_tilt:.1f} deg / {hardware_cfg.surface_azimuth:.1f} deg")
    print(f"  Area/Eff : {hardware_cfg.panel_area_m2:.2f} m^2 / {hardware_cfg.panel_efficiency:.1%}")
    print(f"  Peak     : {peak_power_w:.2f} W at {peak_time.strftime('%Y-%m-%d %H:%M %Z')}")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(csv_path)
        print(f"Saved CSV: {csv_path}")

    output_path = Path(args.output) if args.output else None
    plot_theoretical_power(result, output_path=output_path, show=args.show)


if __name__ == "__main__":
    main()


"""Create the standard four-panel figures without Monte Carlo processing."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HANDOFF_DIR = Path(__file__).resolve().parents[1]

from outdata_folder_loader import load_outdata_directory  # noqa: E402


DEFAULT_DATA_DIR = HANDOFF_DIR / "dataset"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "example_output"
BATTERY_CAPACITY_WH = 11.2
POWER_DEADBAND_W = 0.10

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)


def _date_value(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD") from exc


def _time_value(value: str) -> time:
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, pattern).time()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("Use HH:MM or HH:MM:SS")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create four-panel figures without Monte Carlo replay."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-date", type=_date_value)
    parser.add_argument("--end-date", type=_date_value)
    parser.add_argument("--start-time", type=_time_value, default=time(0, 0))
    parser.add_argument("--end-time", type=_time_value, default=time(23, 59, 59))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--view",
        choices=("both", "power", "voltage-current"),
        default="both",
        help="Select the fourth-panel content",
    )
    parser.add_argument("--name", help="Output filename prefix")
    return parser.parse_args()


def _plot_segments(frame: pd.DataFrame) -> list[np.ndarray]:
    gap = frame["timestamp"].diff().dt.total_seconds().fillna(0).to_numpy(float)
    starts = np.r_[0, np.flatnonzero(gap[1:] > 30 * 60) + 1]
    stops = np.r_[starts[1:], len(frame)]
    return [np.arange(start, stop) for start, stop in zip(starts, stops)]


def _soc_equivalent_power_w(frame: pd.DataFrame) -> np.ndarray:
    timestamps = pd.to_datetime(frame["timestamp"])
    soc = frame["soc_pct"].to_numpy(float) / 100.0
    dt_h = timestamps.diff().dt.total_seconds().div(3600.0).to_numpy(float)
    power = np.zeros(len(frame), dtype=float)
    valid = np.isfinite(dt_h) & (dt_h > 0) & (dt_h <= 0.5)
    power[valid] = (
        np.diff(soc, prepend=soc[0])[valid]
        * BATTERY_CAPACITY_WH
        / dt_h[valid]
    )
    power = np.clip(power, -6.0, 9.0)
    power[np.abs(power) < POWER_DEADBAND_W] = 0.0
    return power


def _legend_right(ax: plt.Axes, *, ncol: int = 1) -> None:
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=ncol,
        borderaxespad=0.0,
    )


def _save_figure(
    frame: pd.DataFrame,
    output: Path,
    view: str,
) -> Path:
    x = frame["x_day"].to_numpy(float)
    segments = _plot_segments(frame)
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    ax = axes[0]
    for number, idx in enumerate(segments):
        ax.plot(
            x[idx], frame["load_w"].to_numpy(float)[idx],
            color="#1f77b4", lw=1.45,
            label="Load demand (W)" if number == 0 else None,
        )
        ax.plot(
            x[idx], frame["pv_w"].to_numpy(float)[idx],
            color="#ff9900", lw=1.45,
            label="PV power (W)" if number == 0 else None,
        )
    ax.set_ylabel("Power (W)")
    ax.set_title("(a) PV and load", loc="left", pad=15)
    _legend_right(ax, ncol=2)

    ax = axes[1]
    for number, idx in enumerate(segments):
        ax.plot(
            x[idx], frame["soc_pct"].to_numpy(float)[idx] / 100.0,
            color="#2ca02c", lw=1.55,
            label="Reference SoC" if number == 0 else None,
        )
    ax.axhline(0.20, color="#777777", lw=0.9, ls=":", label="20/80% bounds")
    ax.axhline(0.80, color="#777777", lw=0.9, ls=":")
    ax.set_ylabel("SoC")
    ax.set_title("(b) SoC", loc="left", pad=15)
    _legend_right(ax, ncol=2)

    ax = axes[2]
    for number, idx in enumerate(segments):
        ax.step(
            x[idx], frame["flow_pct"].to_numpy(float)[idx],
            where="post", color="#1f77b4", lw=1.35,
            label="Flow rate" if number == 0 else None,
        )
    ax.set_ylabel("Flow (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(c) Flow rate", loc="left", pad=15)
    _legend_right(ax)

    ax = axes[3]
    if view == "power":
        equivalent_power = _soc_equivalent_power_w(frame)
        for number, idx in enumerate(segments):
            ax.plot(
                x[idx], equivalent_power[idx],
                color="#2ca02c", lw=1.5,
                label="SoC-equivalent battery power" if number == 0 else None,
            )
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_ylabel("Power (W)")
        ax.set_title(
            "(d) Battery power inferred from SoC",
            loc="left",
            pad=15,
        )
        _legend_right(ax)
    else:
        current_ax = ax.twinx()
        for number, idx in enumerate(segments):
            ax.plot(
                x[idx], frame["voltage_v"].to_numpy(float)[idx],
                color="#1f77b4", lw=1.35,
                label="Battery voltage" if number == 0 else None,
            )
            current_ax.plot(
                x[idx], frame["current_ma"].to_numpy(float)[idx],
                color="#ff7f0e", lw=1.05,
                label="Battery current" if number == 0 else None,
            )
        ax.axhline(
            4.2, color="#d62728", lw=0.9, ls="--", label="4.2 V reference"
        )
        ax.set_ylabel("Voltage (V)")
        current_ax.set_ylabel("Current (mA)")
        ax.set_title("(d) Battery voltage and current", loc="left", pad=15)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = current_ax.get_legend_handles_labels()
        ax.legend(
            handles + handles2,
            labels + labels2,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.01),
            ncol=3,
            borderaxespad=0.0,
        )

    ticks: list[float] = []
    labels: list[str] = []
    for _day, group in frame.groupby("day_number", sort=True):
        ticks.append(float(group["x_day"].mean()))
        labels.append(str(group["date_label"].iloc[0]))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("Day index")
    for ax in axes:
        ax.grid(color="#d9d9d9", alpha=0.65, ls="--", lw=0.7)

    title = (
        f"{frame['timestamp'].min():%Y-%m-%d %H:%M} "
        f"to {frame['timestamp'].max():%Y-%m-%d %H:%M} — "
        f"{'power view' if view == 'power' else 'voltage/current view'}"
    )
    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    frame = load_outdata_directory(
        data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        start_time=args.start_time,
        end_time=args.end_time,
        apply_causal_lag=True,
    )
    output_dir = args.output_dir.expanduser().resolve()
    name = args.name or (
        f"{frame['timestamp'].min():%Y%m%d_%H%M}_"
        f"{frame['timestamp'].max():%Y%m%d_%H%M}"
    )
    figures: list[Path] = []
    if args.view in {"both", "power"}:
        figures.append(
            _save_figure(
                frame,
                output_dir / f"{name}_power.png",
                "power",
            )
        )
    if args.view in {"both", "voltage-current"}:
        figures.append(
            _save_figure(
                frame,
                output_dir / f"{name}_voltage_current.png",
                "voltage-current",
            )
        )
    print(f"Selected: {frame['timestamp'].min()} to {frame['timestamp'].max()}")
    for figure in figures:
        print(f"Saved: {figure}")


if __name__ == "__main__":
    main()

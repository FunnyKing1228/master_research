"""Four panels: PV/load, SoC, flow rate, and battery voltage/current."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common_4panel_plotting import (
    OUTPUT_DIR,
    draw_first_three_panels,
    finish_figure,
    load_data,
    monte_carlo_runs,
    prepare_data,
    segment_indices,
)


OUTPUT = OUTPUT_DIR / "example_4panel_voltage_current.png"


def save_voltage_current_figure(
    input_data: Path | pd.DataFrame,
    output_path: Path,
    title: str = "Four-panel MC replay — voltage/current view",
) -> Path:
    """Convert one canonical CSV dataset into the voltage/current-view PNG."""
    frame = (
        prepare_data(input_data)
        if isinstance(input_data, pd.DataFrame)
        else load_data(input_data)
    )
    runs = monte_carlo_runs(frame)
    x = frame["x_day"].to_numpy(dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    draw_first_three_panels(axes, frame, runs)

    ax = axes[3]
    voltage = frame["voltage_v"].to_numpy(dtype=float)
    current = frame["current_ma"].to_numpy(dtype=float)
    segments = segment_indices(frame)
    for number, idx in enumerate(segments):
        ax.plot(
            x[idx], voltage[idx], color="#1f77b4", lw=1.35,
            label="Battery voltage" if number == 0 else None,
        )
    ax.axhline(4.2, color="#d62728", lw=0.9, ls="--", label="4.2 V reference")
    current_ax = ax.twinx()
    for number, idx in enumerate(segments):
        current_ax.plot(
            x[idx], current[idx], color="#ff7f0e", lw=1.05,
            label="Battery current" if number == 0 else None,
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

    finish_figure(fig, axes, frame, title)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a four-panel voltage/current figure.")
    parser.add_argument("--input", type=Path, required=True, help="Input converted CSV dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output PNG path")
    parser.add_argument("--title", default="Four-panel MC replay — voltage/current view")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = save_voltage_current_figure(args.input, args.output, args.title)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

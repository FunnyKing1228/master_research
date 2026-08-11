"""Four panels: PV/load, SoC, flow rate, and battery power command."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_4panel_plotting import (
    OUTPUT_DIR,
    command_runs_from_soc,
    draw_first_three_panels,
    equivalent_command_w,
    finish_figure,
    load_data,
    monte_carlo_runs,
    prepare_data,
    segment_indices,
)


OUTPUT = OUTPUT_DIR / "example_4panel_mc_command.png"


def save_command_figure(
    input_data: Path | pd.DataFrame,
    output_path: Path,
    title: str = "Four-panel MC replay — command view",
) -> Path:
    """Convert one canonical CSV dataset into the command-view PNG."""
    frame = (
        prepare_data(input_data)
        if isinstance(input_data, pd.DataFrame)
        else load_data(input_data)
    )
    runs = monte_carlo_runs(frame)
    command_runs = command_runs_from_soc(frame, runs)
    c05, c50, c95 = np.percentile(command_runs, [5, 50, 95], axis=0)
    x = frame["x_day"].to_numpy(dtype=float)
    reference = equivalent_command_w(
        frame, frame["soc_pct"].to_numpy(dtype=float) / 100.0
    )

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    draw_first_three_panels(axes, frame, runs)

    ax = axes[3]
    for number, idx in enumerate(segment_indices(frame)):
        ax.fill_between(
            x[idx], c05[idx], c95[idx], color="#d62728", alpha=0.22,
            label="MC 5–95%" if number == 0 else None,
        )
        ax.plot(
            x[idx], reference[idx], color="#2ca02c", lw=1.4,
            label="Reference command" if number == 0 else None,
        )
        ax.plot(
            x[idx], c50[idx], color="#d62728", lw=1.2, ls="--",
            label="MC median" if number == 0 else None,
        )
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_ylabel("Command (W)")
    ax.set_title("(d) Battery power command", loc="left", pad=15)
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
        ncol=3, borderaxespad=0.0,
    )

    finish_figure(fig, axes, frame, title)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a four-panel battery command figure.")
    parser.add_argument("--input", type=Path, required=True, help="Input converted CSV dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output PNG path")
    parser.add_argument("--title", default="Four-panel MC replay — command view")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = save_command_figure(args.input, args.output, args.title)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

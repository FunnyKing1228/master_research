from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "raw" / "figures" / "report_0504"
OUTPUT = OUT_DIR / "simple_pv_observation_censoring_schematic.png"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    x = np.linspace(0, 10, 300)
    demand = np.full_like(x, 10.0)

    # Simple conceptual behavior:
    # below demand, observed PV is the real available PV;
    # at demand, observed PV is capped by what the system asks for.
    observed_pv = np.piecewise(
        x,
        [x < 4.2, x >= 4.2],
        [lambda t: 2.5 + 1.75 * t, 10.0],
    )
    hidden_available = np.where(x >= 4.2, 10.0 + 2.5 * (1 - np.exp(-(x - 4.2) / 1.6)), np.nan)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(x, demand, color="black", lw=3.0)
    ax.plot(x, observed_pv, color="#1f77b4", lw=3.2)
    ax.plot(x, hidden_available, color="#f28e2b", lw=2.6, linestyle="--")

    ax.fill_between(x, observed_pv, demand, where=observed_pv < demand, color="#1f77b4", alpha=0.12)
    ax.fill_between(x, demand, hidden_available, where=x >= 4.2, color="#f28e2b", alpha=0.14)

    ax.axvline(4.2, color="0.55", lw=1.5, linestyle=":")

    ax.text(0.45, 11.45, "Power demand\n(load + charging)", color="black", fontsize=13, va="bottom")
    ax.text(1.10, 3.15, "Observed PV", color="#1f77b4", fontsize=13, va="bottom")
    ax.text(8.45, 13.1, "Unknown extra PV\nmay exist", color="#b35c00", fontsize=13, va="center")

    ax.text(
        1.65,
        0.45,
        "Observed PV < demand\n??solar is truly insufficient",
        color="#1f77b4",
        fontsize=13,
        ha="center",
    )
    ax.text(
        7.15,
        0.45,
        "Observed PV = demand\n??measurement is capped by demand",
        color="#b35c00",
        fontsize=13,
        ha="center",
    )

    ax.set_title("Why Controller-Observed PV Can Be Incomplete")
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Increasing solar availability")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_xticks([])
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.88)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {OUTPUT}")


if __name__ == "__main__":
    main()


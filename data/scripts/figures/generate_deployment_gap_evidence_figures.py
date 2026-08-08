"""Generate slide-ready deployment-gap evidence figures."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experiments" / "seminar_baseline_results" / "deployment_gap_figures"
FIG_DIR = ROOT / "data" / "raw" / "figures"


def wrap(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_card(ax, x, y, w, h, title, bullets, color):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", linewidth=1.4)
    ax.add_patch(rect)
    ax.text(x + 0.04 * w, y + h - 0.12 * h, title, fontsize=17, weight="bold", va="top")
    bullet_text = "\n".join([f"- {b}" for b in bullets])
    ax.text(x + 0.04 * w, y + h - 0.28 * h, bullet_text, fontsize=12.5, va="top", linespacing=1.28)


def figure_overview() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.95,
        "Deployment Gap Evidence: Why Simulation Assumptions Break",
        ha="center",
        va="top",
        fontsize=23,
        weight="bold",
    )
    draw_card(
        ax,
        0.04,
        0.12,
        0.28,
        0.72,
        "Noisy & Censored",
        [
            "PV is not a perfect known profile",
            "nighttime MPPT offsets: ~11-14 mW",
            "SoC is estimated by coulomb counting",
            "old datasets had nighttime PV artifacts",
        ],
        "#E8F3FF",
    )
    draw_card(
        ax,
        0.36,
        0.12,
        0.28,
        0.72,
        "Hardware Timing",
        [
            "1 RL step = 15-minute window",
            "expected readings: 15 min / 11 s ~= 82 samples",
            "long integration gaps are clamped",
            "voltage cutoff has 300 s cooldown",
        ],
        "#FFF4D8",
    )
    draw_card(
        ax,
        0.68,
        0.12,
        0.28,
        0.72,
        "Messy Data",
        [
            "low grid-voltage day excluded",
            "sensor freeze / stale values excluded",
            "malformed CSV rows skipped",
            "incomplete days removed",
        ],
        "#F1E8FF",
    )
    ax.text(
        0.5,
        0.045,
        "Takeaway: deployment-safe RL must handle censored observations, timing gaps, and imperfect real logs, not only ideal power-flow equations.",
        ha="center",
        fontsize=13.5,
        weight="bold",
    )
    save(fig, "deployment_gap_evidence_overview.png")


def figure_messy_data_table() -> None:
    quality_path = FIG_DIR / "clean_daily_grid_load_solar_0511" / "daily_grid_load_solar_quality_summary.csv"
    df = pd.read_csv(quality_path)
    included = int(df["included_for_clean_training"].sum())
    excluded = int((df["included_for_clean_training"] == 0).sum())
    issue_rows = df[df["included_for_clean_training"] == 0][
        ["date", "excluded_reason", "raw_rows", "resampled_15min_steps", "grid_v_min"]
    ].copy()
    issue_rows["excluded_reason"] = issue_rows["excluded_reason"].map(lambda s: wrap(s, 42))
    issue_rows["grid_v_min"] = issue_rows["grid_v_min"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")

    fig, ax = plt.subplots(figsize=(14.5, 7.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.97,
        "Messy Deployment Data: Cleaning Was Required Before RL Training",
        ha="center",
        va="top",
        fontsize=22,
        weight="bold",
    )
    ax.text(
        0.5,
        0.90,
        f"Clean full days used: {included}    |    Excluded/problematic days: {excluded}",
        ha="center",
        fontsize=15,
        weight="bold",
    )

    col_labels = ["Date", "Exclusion reason", "Raw rows", "15-min steps", "Min grid V"]
    table = ax.table(
        cellText=issue_rows.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colWidths=[0.13, 0.49, 0.12, 0.13, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.15)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4B5563")
        elif r % 2 == 0:
            cell.set_facecolor("#F3F4F6")
        else:
            cell.set_facecolor("#FFFFFF")
    ax.text(
        0.5,
        0.055,
        "Evidence source: daily_grid_load_solar_quality_summary.csv",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    save(fig, "messy_data_quality_evidence.png")


def figure_pv_audit() -> None:
    pv_path = FIG_DIR / "pv_distribution_audit_2026-04-24_to_2026-05-11_summary.csv"
    processed_path = FIG_DIR / "pv_distribution_audit_processed_datasets.csv"
    pv = pd.read_csv(pv_path)
    processed = pd.read_csv(processed_path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    fig.suptitle("PV Observation Audit: Measured PV Is Not a Perfect Simulator Profile", fontsize=20, weight="bold")

    x = range(len(pv))
    axes[0].bar(x, pv["day_pv_w_max_10_15"], label="daytime PV max (W)", color="#FDBA74")
    axes[0].plot(x, pv["night_pv_w_max"], marker="o", color="#2563EB", label="night PV max (W)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(pv["date"], rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("W")
    axes[0].set_title("Raw deployment audit")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    show = processed[processed["segment"].isin(["night_20_05", "deep_night_00_04"])].copy()
    labels = [f"{r.dataset}\n{r.segment}" for r in show.itertuples()]
    axes[1].bar(range(len(show)), show["solar_gt_0p1w_frac"] * 100, color="#A78BFA")
    axes[1].set_xticks(range(len(show)))
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Samples with solar > 0.1 W (%)")
    axes[1].set_title("Processed dataset artifacts")
    axes[1].grid(axis="y", alpha=0.25)
    for i, v in enumerate(show["duplicate_timestamps_total"]):
        if v > 0:
            axes[1].text(i, show.iloc[i]["solar_gt_0p1w_frac"] * 100 + 1.2, f"dup={int(v)}", ha="center", fontsize=9)

    fig.text(
        0.5,
        0.02,
        "Evidence: raw PV has small nighttime offsets; older processed datasets contained nighttime artifacts and duplicate timestamps.",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(fig, "pv_observation_audit_evidence.png")


def figure_timing_soc() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.96, "Hardware Timing & Estimated SoC: Not Instant Simulator Transitions", ha="center", va="top", fontsize=22, weight="bold")

    # Timeline
    ax.hlines(0.72, 0.08, 0.92, color="#111827", linewidth=2)
    sample_x = [0.10 + i * 0.035 for i in range(12)]
    for sx in sample_x:
        ax.plot(sx, 0.72, "o", color="#2563EB", markersize=7)
    ax.annotate("", xy=(0.92, 0.72), xytext=(0.08, 0.72), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.5, 0.78, "15-minute RL control window", ha="center", fontsize=17, weight="bold")
    ax.text(0.5, 0.66, "Expected sensor samples: 15 min / 11 s ~= 82 readings", ha="center", fontsize=13)

    # Boxes
    boxes = [
        (0.08, 0.36, 0.25, 0.20, "WindowAggregator", ["aggregates voltage/current/power", "records n_samples", "records completeness"]),
        (0.375, 0.36, 0.25, 0.20, "SoCTracker", ["coulomb counting", "uses current x elapsed time", "clamps long gaps to 1 hour"]),
        (0.67, 0.36, 0.25, 0.20, "Hardware Protection", ["voltage cutoff", "300 s cooldown", "hysteresis, not instant recovery"]),
    ]
    colors = ["#DBEAFE", "#DCFCE7", "#FEE2E2"]
    for (x, y, w, h, title, bullets), color in zip(boxes, colors):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#374151", linewidth=1.3))
        ax.text(x + 0.02, y + h - 0.04, title, fontsize=15, weight="bold", va="top")
        ax.text(x + 0.02, y + h - 0.10, "\n".join(f"- {b}" for b in bullets), fontsize=11.5, va="top", linespacing=1.25)

    ax.text(
        0.5,
        0.17,
        "Deployment implication: the RL agent sees aggregated, delayed, and protected dynamics, not a perfect instantaneous simulator state.",
        ha="center",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "Evidence source: control/run_deployment.py (WindowAggregator, SoCTracker, cutoff cooldown)",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    save(fig, "hardware_timing_soc_evidence.png")


def main() -> None:
    figure_overview()
    figure_messy_data_table()
    figure_pv_audit()
    figure_timing_soc()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()


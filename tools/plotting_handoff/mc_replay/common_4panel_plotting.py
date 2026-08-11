"""Shared data, Monte Carlo, and first-three-panel helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


BUNDLE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "example_output"
CAPACITY_WH = 11.2
POWER_DEADBAND_W = 0.10
N_RUNS = 1000
SEED = 20260722
MC_CAPACITY_STD_WH = 0.35
MC_TRANSFER_GAIN_STD = 0.030
MC_COMMAND_GAIN_STD = 0.030
MC_SOC_BOUND_STD = 0.004
OUTDATA_CHARGE_GAIN = 0.93
OUTDATA_DISCHARGE_GAIN = 1.175
GAP_RESET_H = 0.5
SEAM_SOC_JUMP = 0.05


def load_data(path: Path) -> pd.DataFrame:
    """Load one plotting dataset and create missing day-index helper columns."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    frame = pd.read_csv(path)
    return prepare_data(frame)


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an in-memory plotting dataset."""
    frame = frame.copy()
    required = {
        "timestamp",
        "load_w",
        "pv_w",
        "soc_pct",
        "flow_pct",
        "voltage_v",
        "current_ma",
        "battery_power_w",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Input is missing columns: {sorted(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        bad_rows = frame.index[frame["timestamp"].isna()].tolist()[:5]
        raise RuntimeError(f"Invalid timestamp values at rows: {bad_rows}")
    if frame["timestamp"].duplicated().any():
        raise RuntimeError("Input contains duplicated timestamps")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    dates = frame["timestamp"].dt.normalize()
    if "day_number" not in frame:
        frame["day_number"] = pd.factorize(dates, sort=True)[0] + 1
    else:
        frame["day_number"] = pd.to_numeric(frame["day_number"], errors="raise")

    if "x_day" not in frame:
        seconds = (
            frame["timestamp"].dt.hour * 3600
            + frame["timestamp"].dt.minute * 60
            + frame["timestamp"].dt.second
        )
        frame["x_day"] = frame["day_number"] - 1 + seconds / 86400.0
    else:
        frame["x_day"] = pd.to_numeric(frame["x_day"], errors="raise")

    if "segment_id" not in frame:
        frame["segment_id"] = 0
    else:
        frame["segment_id"] = pd.to_numeric(frame["segment_id"], errors="raise")

    numeric = required.difference({"timestamp"})
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(numeric)].isna().any().any():
        invalid = frame[list(numeric)].columns[frame[list(numeric)].isna().any()].tolist()
        raise RuntimeError(f"Input contains missing or non-numeric values in: {invalid}")

    if len(frame) < 2:
        raise RuntimeError("Input must contain at least two rows")
    return frame.sort_values("x_day").reset_index(drop=True)


def boundary_mask(frame: pd.DataFrame) -> np.ndarray:
    """Match the thesis replay: reset only at real gaps or unexplained seams."""
    timestamps = pd.to_datetime(frame["timestamp"])
    dt_h = timestamps.diff().dt.total_seconds().div(3600.0).fillna(0.0).to_numpy(float)
    soc = frame["soc_pct"].to_numpy(dtype=float) / 100.0
    command = frame["battery_power_w"].fillna(0.0).to_numpy(dtype=float)
    boundary = np.zeros(len(frame), dtype=bool)
    for idx in range(1, len(frame)):
        if dt_h[idx] > GAP_RESET_H:
            boundary[idx] = True
            continue
        soc_jump = abs(soc[idx] - soc[idx - 1])
        if (
            soc_jump >= SEAM_SOC_JUMP
            and abs(command[idx - 1]) < 0.015
            and abs(command[idx]) < 0.015
        ):
            boundary[idx] = True
    return boundary


def monte_carlo_runs(frame: pd.DataFrame) -> np.ndarray:
    """Use the calibrated 1000-run, gap-aware thesis replay method."""
    rng = np.random.default_rng(SEED)
    timestamps = pd.to_datetime(frame["timestamp"])
    dt_h = timestamps.diff().dt.total_seconds().div(3600.0).fillna(0.0).to_numpy(float)
    reference = frame["soc_pct"].to_numpy(dtype=float) / 100.0
    command = frame["battery_power_w"].fillna(0.0).to_numpy(dtype=float)
    boundary = boundary_mask(frame)
    same_row_command = (
        "causal_lag_applied" in frame
        and bool(frame["causal_lag_applied"].iloc[0])
    )
    runs = np.empty((N_RUNS, len(frame)), dtype=float)
    for run in range(N_RUNS):
        capacity = float(
            np.clip(rng.normal(CAPACITY_WH, MC_CAPACITY_STD_WH), 9.8, 12.8)
        )
        charge_gain = float(
            np.clip(
                rng.normal(OUTDATA_CHARGE_GAIN, MC_TRANSFER_GAIN_STD),
                0.70,
                1.25,
            )
        )
        discharge_gain = float(
            np.clip(
                rng.normal(OUTDATA_DISCHARGE_GAIN, MC_TRANSFER_GAIN_STD),
                0.70,
                1.45,
            )
        )
        command_gain = float(
            np.clip(rng.normal(1.0, MC_COMMAND_GAIN_STD), 0.85, 1.15)
        )
        soc_floor = float(
            np.clip(rng.normal(0.20, MC_SOC_BOUND_STD), 0.18, 0.23)
        )
        soc_ceiling = float(
            np.clip(rng.normal(0.80, MC_SOC_BOUND_STD), 0.77, 0.82)
        )
        runs[run, 0] = float(
            np.clip(reference[0] + rng.normal(0.0, 0.006), soc_floor, soc_ceiling)
        )
        command_noise = rng.normal(0.0, 0.025, len(frame))
        for idx in range(1, len(frame)):
            if boundary[idx]:
                runs[run, idx] = float(
                    np.clip(
                        reference[idx] + rng.normal(0.0, 0.006),
                        soc_floor,
                        soc_ceiling,
                    )
                )
                continue
            command_idx = idx if same_row_command else idx - 1
            power = (
                command[command_idx] * command_gain
                + command_noise[command_idx]
            )
            if abs(power) < 0.015:
                power = 0.0
            transfer_gain = charge_gain if power >= 0 else discharge_gain
            step_h = dt_h[idx] if dt_h[idx] > 0 else 0.25
            runs[run, idx] = float(
                np.clip(
                    runs[run, idx - 1]
                    + power * transfer_gain * step_h / capacity,
                    soc_floor,
                    soc_ceiling,
                )
            )
    return runs


def equivalent_command_w(
    frame: pd.DataFrame,
    soc: pd.Series | np.ndarray,
) -> np.ndarray:
    """Convert a SoC trajectory to the equivalent command used by thesis plots."""
    timestamps = pd.to_datetime(frame["timestamp"])
    values = pd.Series(np.asarray(soc, dtype=float), index=frame.index)
    dt_h = timestamps.diff().dt.total_seconds().div(3600.0)
    command = values.diff() * CAPACITY_WH / dt_h
    output = (
        command.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=-6.0, upper=9.0)
        .to_numpy(dtype=float, copy=True)
    )
    output[boundary_mask(frame)] = 0.0
    output[np.abs(output) < POWER_DEADBAND_W] = 0.0
    return output


def command_runs_from_soc(frame: pd.DataFrame, runs: np.ndarray) -> np.ndarray:
    return np.vstack([equivalent_command_w(frame, run) for run in runs])


def segment_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    """Split plotted lines only at real timestamp gaps, not daily MC resets."""
    timestamps = pd.to_datetime(frame["timestamp"])
    gap_seconds = timestamps.diff().dt.total_seconds().fillna(0).to_numpy()
    starts = np.r_[0, np.flatnonzero(gap_seconds[1:] > GAP_RESET_H * 3600) + 1]
    stops = np.r_[starts[1:], len(frame)]
    return [np.arange(start, stop) for start, stop in zip(starts, stops)]


def draw_first_three_panels(
    axes: np.ndarray,
    frame: pd.DataFrame,
    runs: np.ndarray,
) -> None:
    x = frame["x_day"].to_numpy(dtype=float)
    soc = frame["soc_pct"].to_numpy(dtype=float) / 100.0
    p05, p50, p95 = np.percentile(runs, [5, 50, 95], axis=0)
    segments = segment_indices(frame)

    ax = axes[0]
    load = frame["load_w"].to_numpy(dtype=float)
    pv = frame["pv_w"].to_numpy(dtype=float)
    for number, idx in enumerate(segments):
        ax.plot(
            x[idx], load[idx], color="#1f77b4", lw=1.45,
            label="Load demand (W)" if number == 0 else None,
        )
        ax.plot(
            x[idx], pv[idx], color="#ff9900", lw=1.45,
            label="PV power (W)" if number == 0 else None,
        )
    ax.set_ylabel("Power (W)")
    ax.set_title("(a) PV and load", loc="left", pad=15)
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
        ncol=2, borderaxespad=0.0,
    )

    ax = axes[1]
    for number, idx in enumerate(segments):
        ax.fill_between(
            x[idx], p05[idx], p95[idx], color="#d62728", alpha=0.22,
            label="MC 5–95%" if number == 0 else None,
        )
        ax.plot(
            x[idx], soc[idx], color="#2ca02c", lw=1.5,
            label="Reference SoC" if number == 0 else None,
        )
        ax.plot(
            x[idx], p50[idx], color="#d62728", lw=1.2, ls="--",
            label="MC median" if number == 0 else None,
        )
    ax.axhline(0.20, color="#777777", lw=0.8, ls=":")
    ax.axhline(0.80, color="#777777", lw=0.8, ls=":")
    ax.set_ylabel("SoC")
    ax.set_title("(b) SoC", loc="left", pad=15)
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
        ncol=3, borderaxespad=0.0,
    )

    ax = axes[2]
    flow = frame["flow_pct"].to_numpy(dtype=float)
    for number, idx in enumerate(segments):
        ax.step(
            x[idx], flow[idx], where="post", color="#1f77b4", lw=1.3,
            label="Flow rate" if number == 0 else None,
        )
    ax.set_ylabel("Flow (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(c) Flow rate", loc="left", pad=15)
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
        borderaxespad=0.0,
    )


def finish_figure(fig: plt.Figure, axes: np.ndarray, frame: pd.DataFrame, title: str) -> None:
    ticks: list[float] = []
    labels: list[str] = []
    for day, group in frame.groupby("day_number", sort=True):
        ticks.append(float(group["x_day"].mean()))
        if "date_label" in group:
            labels.append(str(group["date_label"].iloc[0]))
        else:
            labels.append(f"Day {int(day)}")
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("Day index")
    for ax in axes:
        ax.grid(color="#d9d9d9", alpha=0.65, ls="--", lw=0.7)
    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""Compute transparent energy and dispatch upper bounds for newHW.

The chronological greedy oracle always stores available PV surplus and serves
load deficit whenever battery energy is available.  Under the fixed model
assumptions this maximizes served energy, but it is not a hardware-certified
bound because load, capacity, efficiency, limits, and initial SoC are TODO(newHW).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import yaml


def calculate_energy_bounds(
    dataset: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    env = config["env"]
    dt_h = float(env["time_step"])
    capacity_kwh = float(env["battery_capacity_kwh"])
    charge_limit_kw = float(env["battery_charge_power_kw"])
    discharge_limit_kw = float(env["battery_discharge_power_kw"])
    efficiency = float(env["battery_efficiency"])
    soc_min = float(env["soc_min"])
    soc_max = float(env["soc_max"])
    initial_soc = float(env["initial_soc"])

    load_kw = dataset["Consumption"].astype(float).clip(lower=0.0).to_numpy()
    pv_kw = dataset["Solar"].astype(float).clip(lower=0.0).to_numpy()
    timestamps = pd.to_datetime(dataset["timestamp"])
    total_load_kwh = float(load_kw.sum() * dt_h)
    total_pv_kwh = float(pv_kw.sum() * dt_h)
    direct_pv_kw = np.minimum(pv_kw, load_kw)
    direct_pv_kwh = float(direct_pv_kw.sum() * dt_h)
    raw_energy_gap_kwh = float(total_load_kwh - total_pv_kwh)

    initial_deliverable_kwh = max(
        0.0, (initial_soc - soc_min) * capacity_kwh * efficiency
    )
    energy_only_upper_kwh = min(
        total_load_kwh, total_pv_kwh + initial_deliverable_kwh
    )

    soc = initial_soc
    records: list[dict[str, Any]] = []
    for timestamp, pv_step_kw, load_step_kw in zip(timestamps, pv_kw, load_kw):
        pv_to_load_kw = min(pv_step_kw, load_step_kw)
        surplus_kw = max(0.0, pv_step_kw - load_step_kw)
        deficit_kw = max(0.0, load_step_kw - pv_step_kw)
        charge_room_kw = max(
            0.0,
            (soc_max - soc) * capacity_kwh / (dt_h * max(efficiency, 1e-9)),
        )
        charge_kw = min(surplus_kw, charge_limit_kw, charge_room_kw)
        soc += charge_kw * dt_h * efficiency / capacity_kwh
        discharge_room_kw = max(
            0.0, (soc - soc_min) * capacity_kwh * efficiency / dt_h
        )
        discharge_kw = min(deficit_kw, discharge_limit_kw, discharge_room_kw)
        soc -= discharge_kw * dt_h / (efficiency * capacity_kwh)
        served_kw = pv_to_load_kw + discharge_kw
        unmet_kw = max(0.0, load_step_kw - served_kw)
        curtailed_kw = max(0.0, surplus_kw - charge_kw)
        records.append(
            {
                "timestamp": timestamp,
                "oracle_soc": soc,
                "oracle_charge_w": charge_kw * 1000.0,
                "oracle_discharge_w": discharge_kw * 1000.0,
                "oracle_served_load_w": served_kw * 1000.0,
                "oracle_unmet_load_w": unmet_kw * 1000.0,
                "oracle_pv_curtailed_w": curtailed_kw * 1000.0,
            }
        )

    trace = pd.DataFrame(records)
    oracle_served_kwh = float(trace["oracle_served_load_w"].sum() * dt_h / 1000.0)
    oracle_unmet_kwh = float(trace["oracle_unmet_load_w"].sum() * dt_h / 1000.0)
    oracle_curtailment_kwh = float(
        trace["oracle_pv_curtailed_w"].sum() * dt_h / 1000.0
    )

    # Sustainable-window bound: maximize served deficit while requiring final
    # SoC >= initial SoC, so the battery cannot contribute net energy created
    # before the 47-hour accounting window.
    n_steps = len(dataset)
    surplus_kw = np.maximum(pv_kw - load_kw, 0.0)
    deficit_kw = np.maximum(load_kw - pv_kw, 0.0)
    objective = np.r_[
        np.zeros(n_steps),
        -np.ones(n_steps) * dt_h,
        np.zeros(n_steps),
    ]
    equality = np.zeros((n_steps, 3 * n_steps))
    equality_rhs = np.zeros(n_steps)
    for step in range(n_steps):
        equality[step, step] = -efficiency * dt_h / capacity_kwh
        equality[step, n_steps + step] = dt_h / (
            max(efficiency, 1e-9) * capacity_kwh
        )
        equality[step, 2 * n_steps + step] = 1.0
        if step == 0:
            equality_rhs[step] = initial_soc
        else:
            equality[step, 2 * n_steps + step - 1] = -1.0
    bounds = (
        [(0.0, min(charge_limit_kw, value)) for value in surplus_kw]
        + [(0.0, min(discharge_limit_kw, value)) for value in deficit_kw]
        + [(soc_min, soc_max) for _ in range(n_steps - 1)]
        + [(initial_soc, soc_max)]
    )
    cyclic_result = linprog(
        objective,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )
    if not cyclic_result.success:
        raise RuntimeError(f"Terminal-SoC-neutral oracle failed: {cyclic_result.message}")
    cyclic_charge_kw = cyclic_result.x[:n_steps]
    cyclic_discharge_kw = cyclic_result.x[n_steps : 2 * n_steps]
    cyclic_soc = cyclic_result.x[2 * n_steps :]
    cyclic_served_kw = direct_pv_kw + cyclic_discharge_kw
    cyclic_unmet_kw = np.maximum(load_kw - cyclic_served_kw, 0.0)
    cyclic_served_kwh = float(cyclic_served_kw.sum() * dt_h)
    cyclic_unmet_kwh = float(cyclic_unmet_kw.sum() * dt_h)
    trace["cyclic_oracle_soc"] = cyclic_soc
    trace["cyclic_oracle_charge_w"] = cyclic_charge_kw * 1000.0
    trace["cyclic_oracle_discharge_w"] = cyclic_discharge_kw * 1000.0
    trace["cyclic_oracle_served_load_w"] = cyclic_served_kw * 1000.0
    trace["cyclic_oracle_unmet_load_w"] = cyclic_unmet_kw * 1000.0

    summary = {
        "status": "PROVISIONAL_PHYSICAL_BOUND_UNDER_CURRENT_ASSUMPTIONS",
        "todo_marker": "TODO(newHW)",
        "total_load_kwh": total_load_kwh,
        "total_pv_kwh": total_pv_kwh,
        "pv_to_load_energy_ratio_if_all_pv_shiftable": (
            total_pv_kwh / max(total_load_kwh, 1e-9)
        ),
        "raw_load_minus_pv_gap_kwh": raw_energy_gap_kwh,
        "direct_pv_only_served_kwh": direct_pv_kwh,
        "direct_pv_only_served_fraction": direct_pv_kwh
        / max(total_load_kwh, 1e-9),
        "initial_battery_deliverable_kwh_assumption": initial_deliverable_kwh,
        "energy_only_upper_bound_kwh": energy_only_upper_kwh,
        "energy_only_upper_bound_fraction": energy_only_upper_kwh
        / max(total_load_kwh, 1e-9),
        "chronological_oracle_served_kwh": oracle_served_kwh,
        "chronological_oracle_served_fraction": oracle_served_kwh
        / max(total_load_kwh, 1e-9),
        "chronological_oracle_unmet_kwh": oracle_unmet_kwh,
        "chronological_oracle_loss_of_load_step_fraction": float(
            (trace["oracle_unmet_load_w"] > 1e-6).mean()
        ),
        "chronological_oracle_pv_curtailment_kwh": oracle_curtailment_kwh,
        "oracle_soc_min": float(trace["oracle_soc"].min()),
        "oracle_soc_max": float(trace["oracle_soc"].max()),
        "oracle_soc_end": float(trace["oracle_soc"].iloc[-1]),
        "terminal_soc_neutral_oracle_served_kwh": cyclic_served_kwh,
        "terminal_soc_neutral_oracle_served_fraction": cyclic_served_kwh
        / max(total_load_kwh, 1e-9),
        "terminal_soc_neutral_oracle_unmet_kwh": cyclic_unmet_kwh,
        "terminal_soc_neutral_oracle_loss_of_load_step_fraction": float(
            (cyclic_unmet_kw > 1e-6).mean()
        ),
        "terminal_soc_neutral_oracle_end_soc": float(cyclic_soc[-1]),
        "assumptions": {
            "fixed_load_w": float(load_kw[0] * 1000.0),
            "battery_capacity_kwh": capacity_kwh,
            "battery_charge_power_kw": charge_limit_kw,
            "battery_discharge_power_kw": discharge_limit_kw,
            "battery_efficiency": efficiency,
            "soc_min": soc_min,
            "soc_max": soc_max,
            "initial_soc": initial_soc,
            "standby_load_added_separately": False,
        },
        "interpretation": [
            "The 4.8 W standby state is not added to the fixed 28.2 W load.",
            "The energy-only bound ignores chronology, finite storage and power limits.",
            "The chronological oracle includes current timing, storage, power and efficiency assumptions.",
            "The terminal-SoC-neutral oracle also requires end SoC >= initial SoC, so the battery only shifts in-window PV energy.",
            "Neither bound is hardware-certified until TODO(newHW) inputs are confirmed.",
        ],
    }
    return summary, trace


def plot_bounds(summary: dict[str, Any], output: Path) -> None:
    labels = [
        "Load demand",
        "PV generation",
        "Direct PV",
        "Energy-only bound",
        "Chronological oracle",
        "Terminal-SoC-neutral",
    ]
    values = [
        summary["total_load_kwh"],
        summary["total_pv_kwh"],
        summary["direct_pv_only_served_kwh"],
        summary["energy_only_upper_bound_kwh"],
        summary["chronological_oracle_served_kwh"],
        summary["terminal_soc_neutral_oracle_served_kwh"],
    ]
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(
        labels,
        values,
        color=["#333333", "#f2a900", "#7fba00", "#5b9bd5", "#2e75b6", "#7030a0"],
    )
    axis.axhline(summary["total_load_kwh"], color="#d62728", linestyle="--")
    axis.set_ylabel("Energy over 47-hour window (kWh)")
    axis.set_title(
        "newHW provisional energy bounds — NOT HARDWARE-CERTIFIED\n"
        "TODO(newHW): fixed load, capacity, limits, efficiency and initial SoC"
    )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate provisional newHW energy bounds")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    dataset = pd.read_csv(args.dataset)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "config" in config and isinstance(config["config"], dict):
        config = config["config"]
    summary, trace = calculate_energy_bounds(dataset, config)
    for path in (args.output_json, args.output_trace, args.output_plot):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    trace.to_csv(args.output_trace, index=False)
    plot_bounds(summary, args.output_plot)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

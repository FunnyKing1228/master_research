from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "control") not in sys.path:
    sys.path.insert(0, str(ROOT / "control"))

from control.run_deployment import (  # type: ignore
    BATTERY_CAPACITY_KWH,
    BATTERY_CHARGE_PMAX_KW,
    BATTERY_DISCHARGE_PMAX_KW,
    BATTERY_EFFICIENCY,
    BATTERY_CUTOFF_V,
    DataBuffer,
    Reading,
    SafetyNet,
    SoCTracker,
    build_state_from_aggregation,
    clear_residual_buffer,
    derive_load_kw_from_aggregation,
    derive_pv_features_from_aggregation,
    determine_situation,
    get_load_groups,
    get_residual_count,
    get_tou_price,
    infer_state_layout,
    load_agent,
    norm_to_power_kw,
    set_conformal_params,
    update_conformal_residual,
    PV_PRESENT_THRESHOLD_KW,
)


DEFAULT_MODEL = ROOT / "experiments" / "v16sp_guided_teacher_v4_hybrid50_2000_pvactivefix" / "models" / "best_sac_model.pth"
OBS_LOAD_EPS_KW = 3e-4
OBS_PV_EPS_KW = 3e-4
OBS_SOC_EPS = 5e-3
ACTION_EPS_KW = 3e-4


@dataclass
class ReplayContext:
    buffer: DataBuffer
    soc_tracker: SoCTracker
    safety_net: Optional[SafetyNet]
    agent: Any
    action_dim: int
    use_pv_support_ratio_state: bool
    use_price_obs_state: bool
    use_coral: bool
    coral_delta: float = 0.15
    coral_buffer: float = 0.03
    coral_window: int = 96
    voltage_cutoff_active: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline replay of deployment raw logs into 15-minute aggregated states."
    )
    parser.add_argument("--raw-csv", required=True, help="Path to raw_data_v2_*.csv")
    parser.add_argument("--deployment-csv", default=None, help="Optional deployment_v2_*.csv for comparison")
    parser.add_argument("--model-path", default=None, help="Optional model path for action replay")
    parser.add_argument("--initial-soc", type=float, default=None, help="Override initial SoC (0~1)")
    parser.add_argument("--window-min", type=int, default=15, help="Aggregation window minutes")
    parser.add_argument("--no-coral", action="store_true", help="Disable CORAL replay even if model is loaded")
    parser.add_argument("--coral-delta", type=float, default=0.15)
    parser.add_argument("--coral-buffer", type=float, default=0.03)
    parser.add_argument("--coral-window", type=int, default=96)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-csv", default=None, help="Output replay CSV path")
    return parser.parse_args()


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    _to_numeric(
        df,
        [
            "soc_percent",
            "voltage_v",
            "charge_voltage_v",
            "current_ma",
            "current_raw_ma",
            "temp_c",
            "speed_percent",
            "solar_v",
            "solar_i_ma",
            "solar_p_mw",
            "mppt_v",
            "mppt_i_ma",
            "mppt_p_mw",
            "bus_v",
            "bus_i_ma",
            "bus_p_mw",
            "load_v",
            "load_i_ma",
            "load_p_mw",
            "grid_v",
            "grid_i_ma",
            "grid_p_mw",
            "soc_calc",
            "soc_unclamped",
            "charge_mah",
            "discharge_mah",
            "situation_code",
        ],
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def load_deployment_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    _to_numeric(
        df,
        [
            "soc",
            "soc_unclamped",
            "load_kw",
            "pv_kw",
            "price",
            "hour",
            "dow",
            "price_norm",
            "pv_support_ratio",
            "pv_bool",
            "pv_active",
            "load_fallback_used",
            "mppt_mean_mW",
            "mppt_max_mW",
            "mppt_std_mW",
            "bus_p_mean_mW",
            "load_p_mean_mW",
            "batt_p_mean_mW",
            "batt_v_mean",
            "batt_i_mean_ma",
            "bus_v_mean",
            "grid_v_mean",
            "n_samples",
            "completeness",
            "action_power_kw",
            "action_flow_pct",
            "power_mw_cmd",
            "flow_pct_cmd",
            "situation_code",
            "load_groups",
            "guard_delta_mW",
            "guard_force_charge_low_soc",
            "guard_block_low_soc_discharge",
            "guard_block_high_soc_charge",
            "guard_block_pv_active_discharge",
            "guard_block_voltage_cutoff",
            "voltage_cutoff_active",
            "voltage_cutoff_day_locked",
            "voltage_cutoff_day_count",
            "coral_active",
            "coral_clipped",
            "coral_delta_mW",
            "coral_interventions",
            "coral_residual_count",
            "action_raw_kw",
        ],
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def row_to_reading(row: pd.Series) -> Reading:
    return Reading(
        timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
        solar_v=float(row.get("solar_v", 0.0) or 0.0),
        solar_i_ma=float(row.get("solar_i_ma", 0.0) or 0.0),
        solar_p_mw=float(row.get("solar_p_mw", 0.0) or 0.0),
        mppt_p_mw=float(row.get("mppt_p_mw", 0.0) or 0.0),
        mppt_v=float(row.get("mppt_v", 0.0) or 0.0),
        mppt_i_ma=float(row.get("mppt_i_ma", 0.0) or 0.0),
        bus_v=float(row.get("bus_v", 0.0) or 0.0),
        bus_i_ma=float(row.get("bus_i_ma", 0.0) or 0.0),
        bus_p_mw=float(row.get("bus_p_mw", 0.0) or 0.0),
        load_v=float(row.get("load_v", 0.0) or 0.0),
        load_i_ma=float(row.get("load_i_ma", 0.0) or 0.0),
        load_p_mw=float(row.get("load_p_mw", 0.0) or 0.0),
        grid_v=float(row.get("grid_v", 0.0) or 0.0),
        grid_i_ma=float(row.get("grid_i_ma", 0.0) or 0.0),
        grid_p_mw=float(row.get("grid_p_mw", 0.0) or 0.0),
        batt_soc_pct=float(row.get("soc_percent", 0.0) or 0.0),
        batt_v=float(row.get("voltage_v", 0.0) or 0.0),
        batt_charge_v=float(row.get("charge_voltage_v", 0.0) or 0.0),
        batt_i_ma=float(row.get("current_ma", 0.0) or 0.0),
        batt_temp_c=float(row.get("temp_c", 0.0) or 0.0),
        batt_speed_pct=float(row.get("speed_percent", 0.0) or 0.0),
    )


def build_context(args: argparse.Namespace, raw_df: pd.DataFrame) -> ReplayContext:
    initial_soc = args.initial_soc
    if initial_soc is None:
        initial_soc = float(raw_df["soc_calc"].dropna().iloc[0]) if "soc_calc" in raw_df.columns else 0.0

    buffer = DataBuffer(window_sec=args.window_min * 60)
    soc_tracker = SoCTracker(initial_soc=float(initial_soc))
    agent = None
    action_dim = 0
    use_pv_support_ratio_state = True
    use_price_obs_state = True
    safety_net = None
    use_coral = (not args.no_coral) and bool(args.model_path)

    if args.model_path:
        device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        agent, action_dim = load_agent(args.model_path, state_dim=None, device=device)
        model_state_dim = int(agent.actor.fc1.weight.shape[1])
        use_pv_support_ratio_state, use_price_obs_state = infer_state_layout(args.model_path, model_state_dim)

        if use_coral:
            set_conformal_params(window=args.coral_window, delta=args.coral_delta)
            clear_residual_buffer()
            safety_net = SafetyNet(
                battery_capacity_kwh=BATTERY_CAPACITY_KWH,
                battery_power_kw=max(BATTERY_CHARGE_PMAX_KW, BATTERY_DISCHARGE_PMAX_KW),
                battery_efficiency=BATTERY_EFFICIENCY,
                soc_min=0.10,
                soc_max=0.90,
                initial_buffer_ratio=args.coral_buffer,
                min_buffer_ratio=0.01,
                boundary_epsilon=0.005,
                time_step=0.25,
                n_step_preview=2,
                enable_n_step_preview=True,
            )

    return ReplayContext(
        buffer=buffer,
        soc_tracker=soc_tracker,
        safety_net=safety_net,
        agent=agent,
        action_dim=action_dim,
        use_pv_support_ratio_state=use_pv_support_ratio_state,
        use_price_obs_state=use_price_obs_state,
        use_coral=use_coral,
        coral_delta=args.coral_delta,
        coral_buffer=args.coral_buffer,
        coral_window=args.coral_window,
    )


def rebuild_safety_net(context: ReplayContext) -> Optional[SafetyNet]:
    if not context.use_coral:
        return None
    set_conformal_params(window=context.coral_window, delta=context.coral_delta)
    clear_residual_buffer()
    return SafetyNet(
        battery_capacity_kwh=BATTERY_CAPACITY_KWH,
        battery_power_kw=max(BATTERY_CHARGE_PMAX_KW, BATTERY_DISCHARGE_PMAX_KW),
        battery_efficiency=BATTERY_EFFICIENCY,
        soc_min=0.10,
        soc_max=0.90,
        initial_buffer_ratio=context.coral_buffer,
        min_buffer_ratio=0.01,
        boundary_epsilon=0.005,
        time_step=0.25,
        n_step_preview=2,
        enable_n_step_preview=True,
    )


def detect_restart_steps(deployment_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if deployment_df is None or "step" not in deployment_df.columns:
        return {}
    df = deployment_df.sort_values("timestamp").reset_index(drop=True).copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["soc"] = pd.to_numeric(df.get("soc"), errors="coerce")
    restarts: Dict[str, float] = {}
    prev_step = None
    for _, row in df.iterrows():
        step = row.get("step")
        if pd.isna(step):
            continue
        step = int(step)
        ts = pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        if prev_step is not None and step <= 1 and prev_step > 1:
            restarts[ts] = float(row.get("soc", np.nan))
        prev_step = step
    return restarts


def apply_action_guards(
    action_kw_raw: float,
    soc: float,
    pv_kw: float,
    latest_batt_v: float,
    context: ReplayContext,
) -> Dict[str, Any]:
    action_kw = float(action_kw_raw)
    coral_clipped = False
    coral_delta_kw = 0.0

    if context.safety_net is not None:
        state_soc = np.array([soc], dtype=np.float32)
        action_arr = np.array([action_kw_raw], dtype=np.float32)
        proj_result = context.safety_net.project(state_soc, action_arr)
        action_kw_safe = float(proj_result[0]) if isinstance(proj_result[0], (int, float)) else float(proj_result[0][0])
        info_proj = proj_result[1] if len(proj_result) > 1 else {}
        coral_clipped = info_proj.get("clipped", abs(action_kw_safe - action_kw_raw) > 1e-8)
        coral_delta_kw = abs(action_kw_safe - action_kw_raw)
        action_kw = action_kw_safe
        update_conformal_residual(coral_delta_kw)

    block_force_charge = False
    block_low_soc_discharge = False
    block_high_soc_charge = False
    block_pv_active_discharge = False
    block_voltage_cutoff = False

    if soc <= 0.05 and action_kw < 0:
        action_kw = BATTERY_CHARGE_PMAX_KW * 0.5
        block_force_charge = True
    elif soc <= 0.10 and action_kw < 0:
        action_kw = 0.0
        block_low_soc_discharge = True
    elif soc >= 0.90 and action_kw > 0:
        action_kw = 0.0
        block_high_soc_charge = True

    pv_active = float(pv_kw > PV_PRESENT_THRESHOLD_KW)
    if pv_active > 0.5 and action_kw < 0:
        action_kw = 0.0
        block_pv_active_discharge = True

    if context.voltage_cutoff_active:
        if action_kw < 0:
            action_kw = 0.0
            block_voltage_cutoff = True
    elif latest_batt_v > 0 and latest_batt_v < BATTERY_CUTOFF_V and action_kw < 0:
        action_kw = 0.0
        context.voltage_cutoff_active = True
        block_voltage_cutoff = True

    return {
        "action_kw_final": action_kw,
        "coral_clipped": int(coral_clipped),
        "coral_delta_kw": coral_delta_kw,
        "pv_active": pv_active,
        "guard_force_charge_low_soc": int(block_force_charge),
        "guard_block_low_soc_discharge": int(block_low_soc_discharge),
        "guard_block_high_soc_charge": int(block_high_soc_charge),
        "guard_block_pv_active_discharge": int(block_pv_active_discharge),
        "guard_block_voltage_cutoff": int(block_voltage_cutoff),
        "coral_residual_count": int(get_residual_count()) if context.use_coral else 0,
    }


def replay_windows(
    raw_df: pd.DataFrame,
    context: ReplayContext,
    deployment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    dep_lookup = None
    if deployment_df is not None:
        dep_lookup = deployment_df.copy()
        dep_lookup["timestamp_key"] = dep_lookup["timestamp"].astype(str)
        dep_lookup = dep_lookup.set_index("timestamp_key")
    restart_steps = detect_restart_steps(deployment_df)

    records: list[dict[str, Any]] = []

    for _, row in raw_df.iterrows():
        reading = row_to_reading(row)
        context.buffer.add(reading)
        context.soc_tracker.update(reading.timestamp, reading.batt_i_ma, reading.batt_v)

        if context.buffer.is_window_complete(reading.timestamp) and context.buffer.count > 0:
            agg = context.buffer.aggregate()
            soc = float(context.soc_tracker.get_soc())
            now = reading.timestamp
            window_start_dt = context.buffer.window_start
            if window_start_dt is None:
                window_start_dt = now
            window_end_dt = window_start_dt + pd.Timedelta(seconds=context.buffer.window_sec)
            if isinstance(window_end_dt, pd.Timestamp):
                window_end_dt = window_end_dt.to_pydatetime()
            window_end_key = window_end_dt.strftime("%Y-%m-%d %H:%M:%S")

            restart_applied = 0
            restart_soc = np.nan
            if window_end_key in restart_steps:
                restart_soc = restart_steps[window_end_key]
                if not np.isnan(restart_soc):
                    context.soc_tracker = SoCTracker(initial_soc=float(restart_soc))
                    context.safety_net = rebuild_safety_net(context)
                    context.voltage_cutoff_active = False
                    restart_applied = 1

            state = build_state_from_aggregation(
                agg,
                float(context.soc_tracker.get_soc()),
                now,
                include_pv_support_ratio=context.use_pv_support_ratio_state,
                include_price_obs=context.use_price_obs_state,
            )
            soc = float(context.soc_tracker.get_soc())

            load_kw = float(state[1])
            if context.use_pv_support_ratio_state:
                pv_support_ratio = float(state[2])
                pv_bool = float(state[3])
                state_hour = float(state[5] if context.use_price_obs_state else state[4])
                state_dow = float(state[6] if context.use_price_obs_state else state[5])
            else:
                pv_support_ratio = float(np.clip(max(0.0, agg.get("bus_p_mean_mW", 0.0) / 1e6) / max(load_kw, 1e-9), 0.0, 1.5))
                pv_bool = float(state[2])
                state_hour = float(state[4] if context.use_price_obs_state else state[3])
                state_dow = float(state[5] if context.use_price_obs_state else state[4])

            load_kw_derived, load_source, load_groups = derive_load_kw_from_aggregation(agg, now)
            pv_kw, pv_support_ratio_derived, pv_bool_derived = derive_pv_features_from_aggregation(agg, load_kw_derived, now)
            price = float(get_tou_price(now.hour, now.weekday()))
            latest_batt_v = float(reading.batt_v)

            out: Dict[str, Any] = {
                "timestamp": window_end_key,
                "window_start": window_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "window_end": window_end_key,
                "restart_applied": restart_applied,
                "restart_soc": restart_soc,
                "soc_replay": soc,
                "soc_unclamped_replay": float(context.soc_tracker.get_soc_unclamped()),
                "charge_mah_replay": float(context.soc_tracker.get_stats()["total_charge_mah"]),
                "discharge_mah_replay": float(context.soc_tracker.get_stats()["total_discharge_mah"]),
                "n_samples": int(agg["n_samples"]),
                "completeness": float(agg["completeness"]),
                "mppt_mean_mW": float(agg["mppt_p_mean_mW"]),
                "mppt_max_mW": float(agg["mppt_p_max_mW"]),
                "mppt_std_mW": float(agg["mppt_p_std_mW"]),
                "bus_p_mean_mW": float(agg["bus_p_mean_mW"]),
                "load_p_mean_mW": float(agg["load_p_mean_mW"]),
                "batt_p_mean_mW": float(agg["batt_p_mean_mW"]),
                "batt_v_mean": float(agg["batt_v_mean"]),
                "batt_i_mean_ma": float(agg["batt_i_mean_ma"]),
                "bus_v_mean": float(agg.get("bus_v_mean", 0.0)),
                "grid_v_mean": float(agg.get("grid_v_mean", 0.0)),
                "state_dim": int(state.shape[0]),
                "state_soc": float(state[0]),
                "state_load_kw": load_kw,
                "state_pv_support_ratio": pv_support_ratio,
                "state_pv_bool": pv_bool,
                "state_price": price,
                "state_hour": state_hour,
                "state_dow": state_dow,
                "pv_kw_bus": pv_kw,
                "load_kw_derived": float(load_kw_derived),
                "pv_support_ratio_derived": float(pv_support_ratio_derived),
                "pv_bool_derived": float(pv_bool_derived),
                "load_source": load_source,
                "load_fallback_used_replay": int(load_source != "measured"),
                "load_groups": int(load_groups),
                "latest_batt_v": latest_batt_v,
                "voltage_cutoff_active_before_action": int(context.voltage_cutoff_active),
            }

            if context.agent is not None:
                with torch.no_grad():
                    action_norm = context.agent.select_action(state, evaluate=True)
                power_norm = float(action_norm[0])
                action_kw_raw = float(norm_to_power_kw(power_norm))
                guarded = apply_action_guards(
                    action_kw_raw=action_kw_raw,
                    soc=soc,
                    pv_kw=pv_kw,
                    latest_batt_v=latest_batt_v,
                    context=context,
                )
                final_action_kw = float(guarded["action_kw_final"])
                out.update(
                    {
                        "action_raw_kw_replay": action_kw_raw,
                        "action_final_kw_replay": final_action_kw,
                        "situation_code_replay": int(determine_situation(final_action_kw, load_kw, pv_kw)),
                        "coral_clipped_replay": int(guarded["coral_clipped"]),
                        "coral_delta_mW_replay": float(guarded["coral_delta_kw"] * 1e6),
                        "pv_active_replay": float(guarded["pv_active"]),
                        "guard_force_charge_low_soc": int(guarded["guard_force_charge_low_soc"]),
                        "guard_block_low_soc_discharge": int(guarded["guard_block_low_soc_discharge"]),
                        "guard_block_high_soc_charge": int(guarded["guard_block_high_soc_charge"]),
                        "guard_block_pv_active_discharge": int(guarded["guard_block_pv_active_discharge"]),
                        "guard_block_voltage_cutoff": int(guarded["guard_block_voltage_cutoff"]),
                        "coral_residual_count_replay": int(guarded["coral_residual_count"]),
                    }
                )

            if dep_lookup is not None:
                key = out["timestamp"]
                if key in dep_lookup.index:
                    dep_row = dep_lookup.loc[key]
                    out.update(
                        {
                            "recorded_price_norm": float(dep_row.get("price_norm", np.nan)),
                            "recorded_pv_support_ratio": float(dep_row.get("pv_support_ratio", np.nan)),
                            "recorded_pv_bool": float(dep_row.get("pv_bool", np.nan)),
                            "recorded_pv_active": float(dep_row.get("pv_active", np.nan)),
                            "recorded_load_fallback_used": float(dep_row.get("load_fallback_used", np.nan)),
                            "recorded_soc": float(dep_row.get("soc", np.nan)),
                            "recorded_action_power_kw": float(dep_row.get("action_power_kw", np.nan)),
                            "recorded_action_raw_kw": float(dep_row.get("action_raw_kw", np.nan)),
                            "recorded_situation_code": float(dep_row.get("situation_code", np.nan)),
                            "recorded_guard_force_charge_low_soc": float(dep_row.get("guard_force_charge_low_soc", np.nan)),
                            "recorded_guard_block_low_soc_discharge": float(dep_row.get("guard_block_low_soc_discharge", np.nan)),
                            "recorded_guard_block_high_soc_charge": float(dep_row.get("guard_block_high_soc_charge", np.nan)),
                            "recorded_guard_block_pv_active_discharge": float(dep_row.get("guard_block_pv_active_discharge", np.nan)),
                            "recorded_guard_block_voltage_cutoff": float(dep_row.get("guard_block_voltage_cutoff", np.nan)),
                            "recorded_voltage_cutoff_active": float(dep_row.get("voltage_cutoff_active", np.nan)),
                            "recorded_coral_clipped": float(dep_row.get("coral_clipped", np.nan)),
                            "recorded_coral_delta_mW": float(dep_row.get("coral_delta_mW", np.nan)),
                            "recorded_pv_kw": float(dep_row.get("pv_kw", np.nan)),
                            "recorded_load_kw": float(dep_row.get("load_kw", np.nan)),
                        }
                    )
                    if "action_final_kw_replay" in out and pd.notna(out["recorded_action_power_kw"]):
                        out["action_kw_diff_vs_recorded"] = float(out["action_final_kw_replay"] - out["recorded_action_power_kw"])

            records.append(out)

            curr_aligned_min = (now.minute // int(context.buffer.window_sec / 60)) * int(context.buffer.window_sec / 60)
            next_start = now.replace(minute=curr_aligned_min, second=0, microsecond=0)
            if context.buffer.window_start is not None and next_start <= context.buffer.window_start:
                next_min = curr_aligned_min + int(context.buffer.window_sec / 60)
                if next_min >= 60:
                    next_start = now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
                    next_start = pd.Timestamp(next_start).to_pydatetime()
                else:
                    next_start = now.replace(minute=next_min, second=0, microsecond=0)
            context.buffer.reset(new_start=next_start)

    return pd.DataFrame.from_records(records)


def default_output_path(raw_csv: Path) -> Path:
    return raw_csv.parent / "replay" / f"{raw_csv.stem}_window_replay.csv"


def _boolish_equal(a: Any, b: Any) -> bool:
    if pd.isna(a) or pd.isna(b):
        return True
    return int(float(a) > 0.5) == int(float(b) > 0.5)


def classify_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    obs_load_mismatch = (
        out["recorded_load_kw"].notna()
        & ((out["state_load_kw"] - out["recorded_load_kw"]).abs() > OBS_LOAD_EPS_KW)
    ) if "recorded_load_kw" in out.columns else pd.Series(False, index=out.index)
    obs_pv_mismatch = (
        out["recorded_pv_kw"].notna()
        & ((out["pv_kw_bus"] - out["recorded_pv_kw"]).abs() > OBS_PV_EPS_KW)
    ) if "recorded_pv_kw" in out.columns else pd.Series(False, index=out.index)
    obs_soc_mismatch = (
        out["recorded_soc"].notna()
        & ((out["soc_replay"] - out["recorded_soc"]).abs() > OBS_SOC_EPS)
    ) if "recorded_soc" in out.columns else pd.Series(False, index=out.index)
    obs_pv_bool_mismatch = (
        out["recorded_pv_bool"].notna()
        & ~out.apply(lambda r: _boolish_equal(r.get("state_pv_bool"), r.get("recorded_pv_bool")), axis=1)
    ) if "recorded_pv_bool" in out.columns else pd.Series(False, index=out.index)
    obs_load_source_mismatch = (
        out["recorded_load_fallback_used"].notna()
        & ~out.apply(lambda r: _boolish_equal(r.get("load_source") == "schedule_fallback", r.get("recorded_load_fallback_used")), axis=1)
    ) if "recorded_load_fallback_used" in out.columns else pd.Series(False, index=out.index)

    raw_action_mismatch = (
        "recorded_action_raw_kw" in out.columns
        and "action_raw_kw_replay" in out.columns
        and out["recorded_action_raw_kw"].notna()
        & ((out["action_raw_kw_replay"] - out["recorded_action_raw_kw"]).abs() > ACTION_EPS_KW)
    ) if ("recorded_action_raw_kw" in out.columns and "action_raw_kw_replay" in out.columns) else pd.Series(False, index=out.index)

    guard_coral_mismatch = (
        "recorded_coral_clipped" in out.columns
        and "coral_clipped_replay" in out.columns
        and out["recorded_coral_clipped"].notna()
        & ~out.apply(lambda r: _boolish_equal(r.get("coral_clipped_replay"), r.get("recorded_coral_clipped")), axis=1)
    ) if ("recorded_coral_clipped" in out.columns and "coral_clipped_replay" in out.columns) else pd.Series(False, index=out.index)
    guard_high_soc_mismatch = (
        "recorded_guard_block_high_soc_charge" in out.columns
        and "guard_block_high_soc_charge" in out.columns
        and out["recorded_guard_block_high_soc_charge"].notna()
        & ~out.apply(lambda r: _boolish_equal(r.get("guard_block_high_soc_charge"), r.get("recorded_guard_block_high_soc_charge")), axis=1)
    ) if ("recorded_guard_block_high_soc_charge" in out.columns and "guard_block_high_soc_charge" in out.columns) else pd.Series(False, index=out.index)
    guard_pv_active_mismatch = (
        "recorded_guard_block_pv_active_discharge" in out.columns
        and "guard_block_pv_active_discharge" in out.columns
        and out["recorded_guard_block_pv_active_discharge"].notna()
        & ~out.apply(lambda r: _boolish_equal(r.get("guard_block_pv_active_discharge"), r.get("recorded_guard_block_pv_active_discharge")), axis=1)
    ) if ("recorded_guard_block_pv_active_discharge" in out.columns and "guard_block_pv_active_discharge" in out.columns) else pd.Series(False, index=out.index)
    guard_voltage_mismatch = (
        "recorded_guard_block_voltage_cutoff" in out.columns
        and "guard_block_voltage_cutoff" in out.columns
        and out["recorded_guard_block_voltage_cutoff"].notna()
        & ~out.apply(lambda r: _boolish_equal(r.get("guard_block_voltage_cutoff"), r.get("recorded_guard_block_voltage_cutoff")), axis=1)
    ) if ("recorded_guard_block_voltage_cutoff" in out.columns and "guard_block_voltage_cutoff" in out.columns) else pd.Series(False, index=out.index)

    final_action_mismatch = (
        "recorded_action_power_kw" in out.columns
        and "action_final_kw_replay" in out.columns
        and out["recorded_action_power_kw"].notna()
        & ((out["action_final_kw_replay"] - out["recorded_action_power_kw"]).abs() > ACTION_EPS_KW)
    ) if ("recorded_action_power_kw" in out.columns and "action_final_kw_replay" in out.columns) else pd.Series(False, index=out.index)

    out["obs_load_mismatch"] = obs_load_mismatch.astype(int)
    out["obs_pv_mismatch"] = obs_pv_mismatch.astype(int)
    out["obs_soc_mismatch"] = obs_soc_mismatch.astype(int)
    out["obs_pv_bool_mismatch"] = obs_pv_bool_mismatch.astype(int)
    out["obs_load_source_mismatch"] = obs_load_source_mismatch.astype(int)
    out["raw_action_mismatch"] = raw_action_mismatch.astype(int)
    out["guard_coral_mismatch"] = guard_coral_mismatch.astype(int)
    out["guard_high_soc_mismatch"] = guard_high_soc_mismatch.astype(int)
    out["guard_pv_active_mismatch"] = guard_pv_active_mismatch.astype(int)
    out["guard_voltage_mismatch"] = guard_voltage_mismatch.astype(int)
    out["final_action_mismatch"] = final_action_mismatch.astype(int)

    categories: list[str] = []
    for _, row in out.iterrows():
        row_categories = []
        if any(int(row[c]) for c in ["obs_load_mismatch", "obs_pv_mismatch", "obs_soc_mismatch", "obs_pv_bool_mismatch", "obs_load_source_mismatch"]):
            row_categories.append("observation")
        if int(row["raw_action_mismatch"]):
            row_categories.append("model_raw_action")
        if any(int(row[c]) for c in ["guard_coral_mismatch", "guard_high_soc_mismatch", "guard_pv_active_mismatch", "guard_voltage_mismatch"]):
            row_categories.append("guard")
        if int(row["final_action_mismatch"]):
            row_categories.append("final_action")
        categories.append(",".join(row_categories) if row_categories else "aligned")
    out["mismatch_category"] = categories
    return out


def build_summary_markdown(df: pd.DataFrame) -> str:
    lines = ["# Replay Mismatch Summary", ""]
    lines.append(f"- Windows: `{len(df)}`")
    availability_checks = {
        "observation": ["recorded_load_kw", "recorded_pv_kw", "recorded_soc", "recorded_pv_bool", "recorded_load_fallback_used"],
        "guard": ["recorded_coral_clipped", "recorded_guard_block_high_soc_charge", "recorded_guard_block_pv_active_discharge", "recorded_guard_block_voltage_cutoff"],
        "action": ["recorded_action_raw_kw", "recorded_action_power_kw"],
    }
    lines.append("- Diagnostic coverage:")
    for label, cols in availability_checks.items():
        available = [col for col in cols if col in df.columns and df[col].notna().any()]
        missing = [col for col in cols if col not in available]
        lines.append(f"  - `{label}`: {len(available)}/{len(cols)} recorded fields available")
        if missing:
            lines.append(f"    - missing: {', '.join(f'`{col}`' for col in missing)}")
    if "mismatch_category" in df.columns:
        counts = df["mismatch_category"].value_counts(dropna=False)
        lines.append("- Category counts:")
        for key, value in counts.items():
            lines.append(f"  - `{key}`: {int(value)}")

    top_cols = [
        "timestamp",
        "mismatch_category",
        "soc_replay",
        "recorded_soc",
        "action_raw_kw_replay",
        "recorded_action_raw_kw",
        "action_final_kw_replay",
        "recorded_action_power_kw",
    ]
    subset_cols = [c for c in top_cols if c in df.columns]
    interesting = df[df["mismatch_category"] != "aligned"] if "mismatch_category" in df.columns else df
    if not interesting.empty:
        lines.append("")
        lines.append("## Top mismatched windows")
        top = interesting.copy()
        if "action_kw_diff_vs_recorded" in top.columns:
            top = top.assign(_rank=top["action_kw_diff_vs_recorded"].abs())
            top = top.sort_values("_rank", ascending=False).drop(columns="_rank")
        top = top.head(12)
        lines.append("```text")
        lines.append(top[subset_cols].to_string(index=False))
        lines.append("```")
    return "\n".join(lines) + "\n"


def summarize(df: pd.DataFrame) -> str:
    lines = [f"windows: {len(df)}"]
    if "action_kw_diff_vs_recorded" in df.columns:
        diffs = df["action_kw_diff_vs_recorded"].dropna()
        if not diffs.empty:
            lines.append(f"mean |action diff|: {diffs.abs().mean() * 1e6:.1f} mW")
            lines.append(f"max  |action diff|: {diffs.abs().max() * 1e6:.1f} mW")
    if "guard_block_voltage_cutoff" in df.columns:
        guard_cols = [
            "guard_force_charge_low_soc",
            "guard_block_low_soc_discharge",
            "guard_block_high_soc_charge",
            "guard_block_pv_active_discharge",
            "guard_block_voltage_cutoff",
        ]
        for col in guard_cols:
            if col in df.columns:
                lines.append(f"{col}: {int(df[col].sum())}")
    if "mismatch_category" in df.columns:
        counts = df["mismatch_category"].value_counts(dropna=False)
        for key, value in counts.items():
            lines.append(f"mismatch[{key}]: {int(value)}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    deployment_csv = Path(args.deployment_csv) if args.deployment_csv else None
    model_path = Path(args.model_path) if args.model_path else None
    if model_path is None and DEFAULT_MODEL.exists():
        model_path = DEFAULT_MODEL
        args.model_path = str(model_path)

    raw_df = load_raw_csv(raw_csv)
    deployment_df = load_deployment_csv(deployment_csv) if deployment_csv and deployment_csv.exists() else None
    context = build_context(args, raw_df)
    replay_df = replay_windows(raw_df, context, deployment_df=deployment_df)
    replay_df = classify_mismatches(replay_df)

    output_csv = Path(args.output_csv) if args.output_csv else default_output_path(raw_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    replay_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary_md = output_csv.with_name(output_csv.stem + "_summary.md")
    summary_md.write_text(build_summary_markdown(replay_df), encoding="utf-8")

    print(f"Saved replay CSV: {output_csv}")
    print(f"Saved summary MD: {summary_md}")
    print(summarize(replay_df))


if __name__ == "__main__":
    main()

"""Load daily deployment/raw CSV pairs and convert them to plotting columns."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd


def _read_daily_files(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "timestamp" not in frame:
            raise RuntimeError(f"Missing timestamp column: {path}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if frame["timestamp"].isna().any():
            raise RuntimeError(f"Invalid timestamp found in: {path}")
        frame["_source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp", "_source_file"])
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def _date_time_mask(
    timestamps: pd.Series,
    start_date: date,
    end_date: date,
    start_time: time,
    end_time: time,
) -> pd.Series:
    start = pd.Timestamp.combine(start_date, start_time)
    end = pd.Timestamp.combine(end_date, end_time)
    if start > end:
        raise ValueError("The requested start datetime must not be after the end datetime")
    return timestamps.between(start, end, inclusive="both")


def _first_available(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    raise RuntimeError(f"None of the required source columns exist: {list(names)}")


def _signed_power_w(deployment: pd.DataFrame) -> pd.Series:
    if "action_power_kw" in deployment:
        return pd.to_numeric(deployment["action_power_kw"], errors="coerce") * 1000.0
    power = _first_available(deployment, ("power_mw_cmd",)) / 1000.0
    if "situation_code" in deployment:
        situation = pd.to_numeric(deployment["situation_code"], errors="coerce")
        power = power.where(situation != 1, -power.abs())
    return power


def _raw_aggregates(
    data_dir: Path,
    selected_dates: set[date],
    start_date: date,
    end_date: date,
    start_time: time,
    end_time: time,
) -> pd.DataFrame:
    paths = sorted(data_dir.glob("raw_data_v2_*.csv"))
    raw = _read_daily_files(paths)
    if raw.empty:
        return raw
    raw = raw[raw["timestamp"].dt.date.isin(selected_dates)].copy()
    raw = raw[
        _date_time_mask(
            raw["timestamp"], start_date, end_date, start_time, end_time
        )
    ].copy()
    if raw.empty:
        return raw

    raw["_bin"] = raw["timestamp"].dt.floor("15min")
    columns = [name for name in ("voltage_v", "current_ma") if name in raw]
    if not columns:
        return pd.DataFrame()
    for column in columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    grouped = raw.groupby("_bin", as_index=False)[columns].mean()
    return grouped.rename(columns={"_bin": "timestamp"})


def load_outdata_directory(
    data_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    start_time: time = time(0, 0),
    end_time: time = time(23, 59, 59),
    apply_causal_lag: bool = True,
) -> pd.DataFrame:
    """Convert a folder of deployment/raw daily files into plotting columns."""
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory does not exist: {data_dir}")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date")

    deployment_paths = sorted(data_dir.glob("deployment_v2_*.csv"))
    deployment = _read_daily_files(deployment_paths)
    if deployment.empty:
        raise RuntimeError(f"No deployment_v2_*.csv files found in: {data_dir}")

    available_start = deployment["timestamp"].min().date()
    available_end = deployment["timestamp"].max().date()
    effective_start_date = start_date or available_start
    effective_end_date = end_date or available_end
    mask = _date_time_mask(
        deployment["timestamp"],
        effective_start_date,
        effective_end_date,
        start_time,
        end_time,
    )
    deployment = deployment[mask].copy()
    if deployment.empty:
        raise RuntimeError("No deployment rows match the requested date/time range")

    selected_dates = set(deployment["timestamp"].dt.date)
    raw = _raw_aggregates(
        data_dir,
        selected_dates,
        effective_start_date,
        effective_end_date,
        start_time,
        end_time,
    )
    if not raw.empty:
        deployment = deployment.merge(
            raw.rename(
                columns={
                    "voltage_v": "_raw_voltage_v",
                    "current_ma": "_raw_current_ma",
                }
            ),
            on="timestamp",
            how="left",
        )

    result = pd.DataFrame({"timestamp": deployment["timestamp"]})
    result["load_w"] = _first_available(
        deployment, ("load_kw", "load_p_mean_mW")
    )
    if "load_kw" in deployment:
        result["load_w"] *= 1000.0
    else:
        result["load_w"] /= 1000.0

    result["pv_w"] = _first_available(
        deployment, ("pv_kw", "mppt_mean_mW")
    )
    if "pv_kw" in deployment:
        result["pv_w"] *= 1000.0
    else:
        result["pv_w"] /= 1000.0

    result["soc_pct"] = _first_available(
        deployment, ("soc", "soc_coulomb")
    )
    if result["soc_pct"].dropna().max() <= 1.5:
        result["soc_pct"] *= 100.0

    result["flow_pct"] = _first_available(
        deployment, ("flow_pct_cmd", "action_flow_pct")
    )
    result["battery_power_w"] = _signed_power_w(deployment)

    deployment_voltage = _first_available(deployment, ("batt_v_mean",))
    deployment_current = _first_available(deployment, ("batt_i_mean_ma",))
    if "_raw_voltage_v" in deployment:
        result["voltage_v"] = pd.to_numeric(
            deployment["_raw_voltage_v"], errors="coerce"
        ).fillna(deployment_voltage)
    else:
        result["voltage_v"] = deployment_voltage
    if "_raw_current_ma" in deployment:
        result["current_ma"] = pd.to_numeric(
            deployment["_raw_current_ma"], errors="coerce"
        ).fillna(deployment_current)
    else:
        result["current_ma"] = deployment_current

    unique_dates = sorted(selected_dates)
    day_map = {value: index + 1 for index, value in enumerate(unique_dates)}
    result["day_number"] = result["timestamp"].dt.date.map(day_map)
    seconds = (
        result["timestamp"].dt.hour * 3600
        + result["timestamp"].dt.minute * 60
        + result["timestamp"].dt.second
    )
    result["x_day"] = result["day_number"] - 1 + seconds / 86400.0
    result["date_label"] = result["timestamp"].dt.strftime("%Y-%m-%d")

    gap = result["timestamp"].diff().dt.total_seconds().fillna(np.inf)
    reset = gap > 30 * 60
    if apply_causal_lag:
        source_power = result["battery_power_w"].to_numpy(float).copy()
        source_flow = result["flow_pct"].to_numpy(float).copy()
        lagged_power = np.r_[0.0, source_power[:-1]]
        lagged_flow = np.r_[0.0, source_flow[:-1]]
        lagged_power[reset.to_numpy()] = 0.0
        lagged_flow[reset.to_numpy()] = 0.0
        result["battery_power_w"] = lagged_power
        result["flow_pct"] = lagged_flow
    result["causal_lag_applied"] = bool(apply_causal_lag)
    result["segment_id"] = reset.cumsum().astype(int)

    numeric = [
        "load_w",
        "pv_w",
        "soc_pct",
        "flow_pct",
        "voltage_v",
        "current_ma",
        "battery_power_w",
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numeric].isna().any().any():
        bad = result[numeric].columns[result[numeric].isna().any()].tolist()
        raise RuntimeError(f"Selected deployment/raw data contain missing values in: {bad}")

    result.attrs["source_dir"] = str(data_dir)
    result.attrs["selected_dates"] = unique_dates
    result.attrs["raw_files_used"] = not raw.empty
    return result.sort_values("timestamp").reset_index(drop=True)

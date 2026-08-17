from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "control"))
sys.path.insert(0, str(ROOT / "data" / "scripts" / "newHW"))

from analyze_energy_bound_newHW import calculate_energy_bounds  # noqa: E402
from io_protocol_newHW import (  # noqa: E402
    NewHWProtocolUnavailable,
    encode_command_newHW,
    parse_measurement_newHW,
)
from microgrid_env_newHW import NewHWMicrogridEnvironment  # noqa: E402


def make_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_newHW.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-14", periods=4, freq="15min"),
            "Solar": [0.0, 0.0, 0.050, 0.100],
            "Consumption": [0.0282] * 4,
        }
    ).to_csv(path, index=False)
    return path


def make_env(tmp_path: Path) -> NewHWMicrogridEnvironment:
    return NewHWMicrogridEnvironment(
        dataset_csv_path=str(make_dataset(tmp_path)),
        episode_length=4,
        time_step=0.25,
        battery_capacity_kwh=0.20,
        battery_charge_power_kw=0.129,
        battery_discharge_power_kw=0.0357,
        battery_efficiency=0.95,
        soc_min=0.10,
        soc_max=0.90,
        initial_soc=0.80,
        reward={"unmet_load": 12.0},
    )


def test_newhw_action_space_is_one_dimensional(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert env.action_space.shape == (1,)
    assert env.use_flow_rate_action is False


def test_newhw_never_reports_grid_flow(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    env.reset()
    _, _, _, _, info = env.step([-0.0282])
    assert info["grid_kw"] == 0.0
    assert info["situation_code"] == 1
    assert info["unmet_load_kw"] == pytest.approx(0.0)


def test_newhw_standby_records_unmet_load_at_night(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    env.reset()
    _, reward, _, _, info = env.step([0.0])
    assert info["situation_code"] == 4
    assert info["unmet_load_kw"] == pytest.approx(0.0282)
    assert reward < 0.0


def test_newhw_charge_is_limited_to_pv_surplus(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    env.fixed_start_idx = 0
    env.reset()
    env.current_step = 2
    _, _, _, _, info = env.step([0.129])
    expected_surplus = 0.050 - 0.0282
    assert info["applied_action_kw"] == pytest.approx(expected_surplus)
    assert info["grid_kw"] == 0.0


def test_newhw_protocol_is_intentionally_unimplemented() -> None:
    with pytest.raises(NewHWProtocolUnavailable):
        parse_measurement_newHW("unknown")
    with pytest.raises(NewHWProtocolUnavailable):
        encode_command_newHW({"power_kw": 0.0})


def test_newhw_energy_bound_uses_dataset_load_without_extra_standby(
    tmp_path: Path,
) -> None:
    dataset = pd.read_csv(make_dataset(tmp_path))
    config = {
        "env": {
            "time_step": 0.25,
            "battery_capacity_kwh": 0.20,
            "battery_charge_power_kw": 0.129,
            "battery_discharge_power_kw": 0.0357,
            "battery_efficiency": 0.95,
            "soc_min": 0.10,
            "soc_max": 0.90,
            "initial_soc": 0.80,
        }
    }
    summary, _ = calculate_energy_bounds(dataset, config)
    expected_load_kwh = 4 * 0.0282 * 0.25
    assert summary["total_load_kwh"] == pytest.approx(expected_load_kwh)
    assert summary["assumptions"]["standby_load_added_separately"] is False


def test_newhw_chronological_oracle_is_bounded_by_energy_only_limit(
    tmp_path: Path,
) -> None:
    dataset = pd.read_csv(make_dataset(tmp_path))
    config = {
        "env": {
            "time_step": 0.25,
            "battery_capacity_kwh": 0.20,
            "battery_charge_power_kw": 0.129,
            "battery_discharge_power_kw": 0.0357,
            "battery_efficiency": 0.95,
            "soc_min": 0.10,
            "soc_max": 0.90,
            "initial_soc": 0.80,
        }
    }
    summary, _ = calculate_energy_bounds(dataset, config)
    assert (
        summary["chronological_oracle_served_kwh"]
        <= summary["energy_only_upper_bound_kwh"] + 1e-12
    )
    assert (
        summary["terminal_soc_neutral_oracle_served_kwh"]
        <= summary["chronological_oracle_served_kwh"] + 1e-12
    )
    assert summary["chronological_oracle_served_kwh"] <= summary["total_load_kwh"]

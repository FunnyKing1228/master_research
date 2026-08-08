import os
import sys

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data", "scripts", "diagnostics"))

from replay_deployment_window import build_summary_markdown, classify_mismatches


def test_classify_mismatches_separates_observation_guard_and_final_action():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-20 00:15:00",
                "state_load_kw": 0.0030,
                "recorded_load_kw": 0.0020,
                "pv_kw_bus": 0.0010,
                "recorded_pv_kw": 0.0010,
                "soc_replay": 0.50,
                "recorded_soc": 0.50,
                "state_pv_bool": 1.0,
                "recorded_pv_bool": 0.0,
                "load_source": "schedule_fallback",
                "recorded_load_fallback_used": 0.0,
                "action_raw_kw_replay": 0.0010,
                "recorded_action_raw_kw": 0.0010,
                "coral_clipped_replay": 0.0,
                "recorded_coral_clipped": 0.0,
                "guard_block_high_soc_charge": 1.0,
                "recorded_guard_block_high_soc_charge": 0.0,
                "guard_block_pv_active_discharge": 0.0,
                "recorded_guard_block_pv_active_discharge": 0.0,
                "guard_block_voltage_cutoff": 0.0,
                "recorded_guard_block_voltage_cutoff": 0.0,
                "action_final_kw_replay": 0.0,
                "recorded_action_power_kw": 0.0010,
            },
            {
                "timestamp": "2026-04-20 00:30:00",
                "state_load_kw": 0.0020,
                "recorded_load_kw": 0.0020,
                "pv_kw_bus": 0.0010,
                "recorded_pv_kw": 0.0010,
                "soc_replay": 0.50,
                "recorded_soc": 0.50,
                "state_pv_bool": 0.0,
                "recorded_pv_bool": 0.0,
                "load_source": "measured",
                "recorded_load_fallback_used": 0.0,
                "action_raw_kw_replay": -0.0010,
                "recorded_action_raw_kw": 0.0010,
                "coral_clipped_replay": 0.0,
                "recorded_coral_clipped": 0.0,
                "guard_block_high_soc_charge": 0.0,
                "recorded_guard_block_high_soc_charge": 0.0,
                "guard_block_pv_active_discharge": 0.0,
                "recorded_guard_block_pv_active_discharge": 0.0,
                "guard_block_voltage_cutoff": 0.0,
                "recorded_guard_block_voltage_cutoff": 0.0,
                "action_final_kw_replay": -0.0010,
                "recorded_action_power_kw": -0.0010,
            },
        ]
    )

    result = classify_mismatches(df)

    assert result.loc[0, "obs_load_mismatch"] == 1
    assert result.loc[0, "obs_pv_bool_mismatch"] == 1
    assert result.loc[0, "obs_load_source_mismatch"] == 1
    assert result.loc[0, "guard_high_soc_mismatch"] == 1
    assert result.loc[0, "final_action_mismatch"] == 1
    assert result.loc[0, "mismatch_category"] == "observation,guard,final_action"

    assert result.loc[1, "raw_action_mismatch"] == 1
    assert result.loc[1, "mismatch_category"] == "model_raw_action"


def test_build_summary_markdown_lists_category_counts():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-20 00:15:00",
                "mismatch_category": "aligned",
                "soc_replay": 0.5,
                "recorded_soc": 0.5,
                "action_raw_kw_replay": 0.0,
                "recorded_action_raw_kw": 0.0,
                "action_final_kw_replay": 0.0,
                "recorded_action_power_kw": 0.0,
            },
            {
                "timestamp": "2026-04-20 00:30:00",
                "mismatch_category": "model_raw_action",
                "soc_replay": 0.5,
                "recorded_soc": 0.5,
                "action_raw_kw_replay": 0.001,
                "recorded_action_raw_kw": 0.0,
                "action_final_kw_replay": 0.001,
                "recorded_action_power_kw": 0.001,
                "action_kw_diff_vs_recorded": 0.0,
            },
        ]
    )

    summary = build_summary_markdown(df)

    assert "# Replay Mismatch Summary" in summary
    assert "`aligned`: 1" in summary
    assert "`model_raw_action`: 1" in summary
    assert "Top mismatched windows" in summary

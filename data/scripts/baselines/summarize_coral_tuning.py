"""Summarize CORAL tuning runs into a ranked table."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
TUNE_DIR = ROOT / "configs" / "baselines" / "research" / "coral_tuning"
OUT_DIR = ROOT / "experiments" / "seminar_baseline_results"
MANIFEST = TUNE_DIR / "manifest.yaml"


def load_manifest() -> List[Dict[str, object]]:
    with open(MANIFEST, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def summarize_log(exp_name: str, last_n: int = 20) -> Dict[str, float]:
    log_path = ROOT / "experiments" / exp_name / "logs" / "episode_log.csv"
    if not log_path.exists():
        return {"status": "missing"}
    df = pd.read_csv(log_path)
    tail = df.tail(min(last_n, len(df)))
    row = {
        "status": "done",
        "episodes": float(len(df)),
        "violations_realized": float(tail["violations_realized"].mean()),
        "violations_attempted": float(tail["violations_attempted"].mean()),
        "safety_projected_meaningful": float(tail["safety_projected_meaningful"].mean()),
        "projection_delta_mean_w": float(tail["projection_delta_mean_w"].mean()),
        "projection_delta_max_w": float(tail["projection_delta_max_w"].max()),
        "net_profit": float(tail["net_profit"].mean()),
        "ep_reward": float(tail["ep_reward"].mean()),
        "soc_min": float(tail["soc_min"].mean()),
        "soc_max": float(tail["soc_max"].mean()),
    }
    # Lexicographic seminar score: hard safety first, then raw/intervention reliance, then profit.
    row["selection_score"] = (
        -1000.0 * row["violations_realized"]
        -10.0 * row["violations_attempted"]
        -5.0 * row["safety_projected_meaningful"]
        +100.0 * row["net_profit"]
    )
    return row


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "experiment",
        "beta_occ",
        "warmup",
        "attempted_penalty",
        "projection_penalty",
        "violations_realized",
        "violations_attempted",
        "safety_projected_meaningful",
        "projection_delta_mean_w",
        "projection_delta_max_w",
        "net_profit",
        "selection_score",
        "reason",
    ]
    existing = [c for c in cols if c in df.columns]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# CORAL Tuning Summary\n\n")
        f.write("Ranked by safety-first selection score. Use this to choose the seminar CORAL setting.\n\n")
        f.write("| " + " | ".join(existing) + " |\n")
        f.write("| " + " | ".join(["---"] * len(existing)) + " |\n")
        for _, row in df[existing].iterrows():
            values = []
            for value in row:
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.4f}")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    rows = []
    for item in load_manifest():
        exp_name = str(item["experiment"])
        row = dict(item)
        row.update(summarize_log(exp_name))
        rows.append(row)
    df = pd.DataFrame(rows)
    if "selection_score" in df.columns:
        df = df.sort_values(["status", "selection_score"], ascending=[True, False])
    csv_path = OUT_DIR / "coral_tuning_summary.csv"
    md_path = OUT_DIR / "coral_tuning_summary.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()


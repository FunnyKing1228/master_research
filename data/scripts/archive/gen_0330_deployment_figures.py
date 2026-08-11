"""
Generate day-by-day deployment figures with clear captions.

Input folder (default):
  data/raw/deployment_0330

Output folder (default):
  experiments/deployment_0330_report
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


@dataclass
class DayStats:
    date: str
    n_steps: int
    n_raw: int
    soc_start: float
    soc_end: float
    soc_min: float
    soc_max: float
    action_min_mw: float
    action_max_mw: float
    discharge_steps: int
    discharge_minutes: float
    batt_v_min: float
    batt_v_max: float
    pv_peak_w: float
    load_mean_w: float
    grid_mean_w: float


def load_pair(input_dir: Path, date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dep_path = input_dir / f"deployment_v2_{date}.csv"
    raw_path = input_dir / f"raw_data_v2_{date}.csv"
    if not dep_path.exists():
        raise FileNotFoundError(f"Missing file: {dep_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing file: {raw_path}")

    dep = pd.read_csv(dep_path)
    raw = pd.read_csv(raw_path)
    dep["timestamp"] = pd.to_datetime(dep["timestamp"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    dep = dep.sort_values("timestamp").reset_index(drop=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    return dep, raw


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def prep_data(dep: pd.DataFrame, raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dep_cols = [
        "soc",
        "action_power_kw",
        "pv_kw",
        "load_kw",
        "situation_code",
        "coral_clipped",
        "coral_delta_mW",
        "mppt_mean_mW",
        "batt_v_mean",
    ]
    raw_cols = [
        "voltage_v",
        "current_ma",
        "mppt_p_mw",
        "load_p_mw",
        "grid_p_mw",
    ]
    _to_numeric(dep, dep_cols)
    _to_numeric(raw, raw_cols)

    dep["action_mw"] = dep["action_power_kw"] * 1e6
    dep["soc_pct"] = dep["soc"] * 100.0
    dep["pv_w"] = dep["pv_kw"] * 1000.0
    dep["load_w"] = dep["load_kw"] * 1000.0

    raw["batt_p_w"] = (raw["voltage_v"] * raw["current_ma"]) / 1000.0
    raw["pv_w"] = raw["mppt_p_mw"] / 1000.0
    raw["load_w"] = raw["load_p_mw"] / 1000.0
    raw["grid_w"] = raw["grid_p_mw"] / 1000.0
    return dep, raw


def plot_1_action(date: str, dep: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    x = dep["timestamp"]
    y = dep["action_mw"]
    ax.plot(x, y, color="#1f77b4", linewidth=1.8, label="Action power (mW)")
    ax.fill_between(x, 0, y, where=(y < 0), color="#d62728", alpha=0.25, label="Discharge")
    ax.fill_between(x, 0, y, where=(y > 0), color="#2ca02c", alpha=0.18, label="Charge")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{date} - Policy Action Power (positive=charge, negative=discharge)")
    ax.set_ylabel("Power command (mW)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    out = out_dir / f"fig01_action_power_{date}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_2_soc_voltage(date: str, dep: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()

    ax1.plot(dep["timestamp"], dep["soc_pct"], color="#1f77b4", linewidth=2, label="SoC (%)")
    ax2.plot(raw["timestamp"], raw["voltage_v"], color="#ff7f0e", linewidth=1.2, alpha=0.9, label="Battery V")

    ax1.set_title(f"{date} - SoC and Battery Voltage")
    ax1.set_ylabel("SoC (%)", color="#1f77b4")
    ax2.set_ylabel("Battery voltage (V)", color="#ff7f0e")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.autofmt_xdate()
    plt.tight_layout()
    out = out_dir / f"fig02_soc_voltage_{date}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_3_flow(date: str, raw: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw["timestamp"], raw["pv_w"], label="PV power (W)", linewidth=1.3)
    ax.plot(raw["timestamp"], raw["load_w"], label="Load power (W)", linewidth=1.3)
    ax.plot(raw["timestamp"], raw["grid_w"], label="Grid power (W)", linewidth=1.3)
    ax.plot(raw["timestamp"], raw["batt_p_w"], label="Battery power (W, +charge/-discharge)", linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{date} - Power Flow (PV / Load / Grid / Battery)")
    ax.set_ylabel("Power (W)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    out = out_dir / f"fig03_power_flow_{date}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_4_scenario(date: str, dep: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.step(dep["timestamp"], dep["situation_code"], where="post", linewidth=1.8, color="#9467bd", label="Situation code")
    if "coral_clipped" in dep.columns:
        clipped = dep[dep["coral_clipped"] > 0.5]
        if len(clipped) > 0:
            ax.scatter(
                clipped["timestamp"],
                clipped["situation_code"],
                s=18,
                color="#d62728",
                alpha=0.9,
                label="CORAL clipped",
                zorder=3,
            )
    ax.set_title(f"{date} - Situation Code Timeline")
    ax.set_ylabel("Code")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.5, 4.5)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    out = out_dir / f"fig04_situation_{date}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def calc_day_stats(date: str, dep: pd.DataFrame, raw: pd.DataFrame) -> DayStats:
    discharge_steps = int((dep["action_mw"] < -1.0).sum())
    discharge_minutes = float(discharge_steps * 15.0)
    return DayStats(
        date=date,
        n_steps=len(dep),
        n_raw=len(raw),
        soc_start=float(dep["soc_pct"].iloc[0]),
        soc_end=float(dep["soc_pct"].iloc[-1]),
        soc_min=float(dep["soc_pct"].min()),
        soc_max=float(dep["soc_pct"].max()),
        action_min_mw=float(dep["action_mw"].min()),
        action_max_mw=float(dep["action_mw"].max()),
        discharge_steps=discharge_steps,
        discharge_minutes=discharge_minutes,
        batt_v_min=float(raw["voltage_v"].min()),
        batt_v_max=float(raw["voltage_v"].max()),
        pv_peak_w=float(raw["pv_w"].max()),
        load_mean_w=float(raw["load_w"].mean()),
        grid_mean_w=float(raw["grid_w"].mean()),
    )


def write_caption_md(out_dir: Path, all_stats: List[DayStats], all_figs: Dict[str, List[Path]]) -> Path:
    md = []
    md.append("# Deployment v2 (2026-03-27 ~ 2026-03-30) 圖說")
    md.append("")
    md.append("以下每一張圖都可獨立放進簡報，caption 已整理為可直接貼上的版本。")
    md.append("")

    for st in all_stats:
        md.append(f"## {st.date}")
        md.append("")
        fig_paths = all_figs[st.date]
        md.append(f"- `fig01_action_power_{st.date}.png`")
        md.append(
            f"  - Caption: 模型在 {st.date} 的每 15 分鐘功率指令。正值代表充電、負值代表放電。"
            f"當日指令範圍為 {st.action_min_mw:.1f} ~ {st.action_max_mw:.1f} mW；"
            f"可觀察到放電步數 {st.discharge_steps}（約 {st.discharge_minutes:.0f} 分鐘）。"
        )
        md.append(f"- `fig02_soc_voltage_{st.date}.png`")
        md.append(
            f"  - Caption: {st.date} 的 SoC 與電池端電壓變化。"
            f"SoC 由 {st.soc_start:.2f}% 變化至 {st.soc_end:.2f}%（範圍 {st.soc_min:.2f}%~{st.soc_max:.2f}%），"
            f"電壓範圍 {st.batt_v_min:.2f}~{st.batt_v_max:.2f} V。"
        )
        md.append(f"- `fig03_power_flow_{st.date}.png`")
        md.append(
            f"  - Caption: {st.date} 的功率流向圖（PV、負載、市電、電池）。"
            f"PV 峰值 {st.pv_peak_w:.2f} W，平均負載 {st.load_mean_w:.2f} W，平均市電 {st.grid_mean_w:.2f} W；"
            f"電池功率正負號可直接辨識充/放電切換。"
        )
        md.append(f"- `fig04_situation_{st.date}.png`")
        md.append(
            f"  - Caption: {st.date} 的情境碼（Scenario 1~4）時間軸；"
            f"紅點表示 CORAL 有介入裁切。可對照 `fig01` 檢查「策略輸出」與「安全投影」的差異。"
        )
        md.append("")

    md.append("## 總結建議（可口頭補充）")
    md.append("")
    md.append("- 先講 `fig01`：直接證明模型有輸出放電（負功率）。")
    md.append("- 再講 `fig02`：放電段落對應 SoC 下降，行為方向一致。")
    md.append("- 最後用 `fig03` / `fig04`：說明能流分配與安全機制介入位置。")
    md.append("")
    md_path = out_dir / "captions.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 0330 deployment figures and captions.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/deployment_0330",
        help="Folder containing deployment_v2_*.csv and raw_data_v2_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/deployment_0330_report",
        help="Output folder for figures and captions",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        default=["2026-03-27", "2026-03-28", "2026-03-29", "2026-03-30"],
        help="Date list in YYYY-MM-DD",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[DayStats] = []
    all_figs: Dict[str, List[Path]] = {}

    for date in args.dates:
        dep, raw = load_pair(input_dir, date)
        dep, raw = prep_data(dep, raw)

        figs = [
            plot_1_action(date, dep, out_dir),
            plot_2_soc_voltage(date, dep, raw, out_dir),
            plot_3_flow(date, raw, out_dir),
            plot_4_scenario(date, dep, out_dir),
        ]
        st = calc_day_stats(date, dep, raw)
        all_figs[date] = figs
        all_stats.append(st)

        print(
            f"[{date}] done: steps={st.n_steps}, raw={st.n_raw}, "
            f"discharge={st.discharge_steps} steps ({st.discharge_minutes:.0f} min), "
            f"action range=({st.action_min_mw:.1f}, {st.action_max_mw:.1f}) mW"
        )

    md_path = write_caption_md(out_dir, all_stats, all_figs)
    print(f"Captions written: {md_path}")
    print(f"Output folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

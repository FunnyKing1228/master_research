"""
Generate easy-to-read story figures for deployment_v2 data.

Input (default):
  data/raw/deployment_0330

Output (default):
  experiments/deployment_0330_story
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


DATES = ["2026-03-27", "2026-03-28", "2026-03-29", "2026-03-30"]

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class DailyKPI:
    date: str
    discharge_minutes: float
    soc_start_pct: float
    soc_end_pct: float
    soc_delta_pct: float
    load_wh: float
    pv_wh: float
    grid_wh: float
    batt_discharge_wh: float
    batt_charge_wh: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate story figures for 0330 deployment data.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/deployment_0330",
        help="Folder containing deployment_v2_*.csv and raw_data_v2_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/deployment_0330_story",
        help="Output folder for story figures",
    )
    return parser.parse_args()


def _load_day(input_dir: Path, date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dep = pd.read_csv(input_dir / f"deployment_v2_{date}.csv")
    raw = pd.read_csv(input_dir / f"raw_data_v2_{date}.csv")
    dep["timestamp"] = pd.to_datetime(dep["timestamp"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    dep = dep.sort_values("timestamp").reset_index(drop=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    for c in ["soc", "action_power_kw", "situation_code", "coral_clipped"]:
        if c in dep.columns:
            dep[c] = pd.to_numeric(dep[c], errors="coerce")
    for c in ["voltage_v", "current_ma", "mppt_p_mw", "load_p_mw", "grid_p_mw"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    dep["action_mw"] = dep["action_power_kw"] * 1e6
    dep["soc_pct"] = dep["soc"] * 100.0

    raw["pv_w"] = raw["mppt_p_mw"] / 1000.0
    raw["load_w"] = raw["load_p_mw"] / 1000.0
    raw["grid_w"] = raw["grid_p_mw"] / 1000.0
    raw["batt_w"] = (raw["voltage_v"] * raw["current_ma"]) / 1000.0  # +charge / -discharge
    raw["batt_discharge_w"] = (-raw["batt_w"]).clip(lower=0.0)  # discharge only
    raw["date"] = date
    dep["date"] = date
    return dep, raw


def _energy_wh(series_w: pd.Series, dt_h: float) -> float:
    return float(np.nansum(series_w.values) * dt_h)


def _daily_kpi(date: str, dep: pd.DataFrame, raw: pd.DataFrame) -> DailyKPI:
    dt_sec = raw["timestamp"].diff().dt.total_seconds().median()
    if pd.isna(dt_sec) or dt_sec <= 0:
        dt_sec = 10.0
    dt_h = float(dt_sec) / 3600.0

    load_wh = _energy_wh(raw["load_w"].clip(lower=0), dt_h)
    pv_wh = _energy_wh(raw["pv_w"].clip(lower=0), dt_h)
    grid_wh = _energy_wh(raw["grid_w"].clip(lower=0), dt_h)
    batt_discharge_wh = _energy_wh((-raw["batt_w"]).clip(lower=0), dt_h)
    batt_charge_wh = _energy_wh((raw["batt_w"]).clip(lower=0), dt_h)

    discharge_steps = int((dep["action_mw"] < -1.0).sum())
    discharge_minutes = discharge_steps * 15.0
    soc_start = float(dep["soc_pct"].iloc[0])
    soc_end = float(dep["soc_pct"].iloc[-1])
    return DailyKPI(
        date=date,
        discharge_minutes=discharge_minutes,
        soc_start_pct=soc_start,
        soc_end_pct=soc_end,
        soc_delta_pct=soc_end - soc_start,
        load_wh=load_wh,
        pv_wh=pv_wh,
        grid_wh=grid_wh,
        batt_discharge_wh=batt_discharge_wh,
        batt_charge_wh=batt_charge_wh,
    )


def _first_discharge_segment(dep_0330: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    neg = dep_0330["action_mw"] < -1.0
    if not neg.any():
        t = dep_0330["timestamp"].iloc[0]
        return t, t
    idx = np.where(neg.values)[0]
    groups: List[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        groups.append((start, prev))
        start = i
        prev = i
    groups.append((start, prev))
    longest = max(groups, key=lambda g: (g[1] - g[0] + 1))
    return dep_0330["timestamp"].iloc[longest[0]], dep_0330["timestamp"].iloc[longest[1]]


def fig01_clear_takeaway(kpis: List[DailyKPI], out_dir: Path) -> Path:
    dates = [k.date for k in kpis]
    x = np.arange(len(dates))
    discharge_hours = [k.discharge_minutes / 60.0 for k in kpis]
    soc_delta = [k.soc_delta_pct for k in kpis]

    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    ax2 = ax1.twinx()

    bars = ax1.bar(x - 0.2, discharge_hours, width=0.4, color="#d62728", alpha=0.8, label="放電時數 (小時)")
    ax2.plot(x + 0.2, soc_delta, "o-", color="#1f77b4", linewidth=2.2, label="SoC 變化 (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, rotation=0)
    ax1.set_ylabel("放電時數 (小時)")
    ax2.set_ylabel("SoC 結束 - 開始 (%)")
    ax1.set_title("圖 1（先講這張）：只有 03-30 出現長時間放電", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.25, axis="y")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    for i, b in enumerate(bars):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{discharge_hours[i]:.1f}h", ha="center", fontsize=10)
        y_off = 0.8 if soc_delta[i] >= 0 else -1.4
        ax2.text(i + 0.2, soc_delta[i] + y_off, f"{soc_delta[i]:.1f}%", ha="center", fontsize=10, color="#1f77b4")

    i_0330 = dates.index("2026-03-30")
    ax1.annotate(
        "關鍵日：策略開始明顯放電",
        xy=(i_0330 - 0.2, discharge_hours[i_0330]),
        xytext=(i_0330 - 1.15, max(discharge_hours) * 0.9),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.6},
        fontsize=10,
        color="#d62728",
    )

    plt.tight_layout()
    out = out_dir / "story01_clear_takeaway.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig02_focus_0330(dep_0330: pd.DataFrame, raw_0330: pd.DataFrame, out_dir: Path) -> Path:
    t0, t1 = _first_discharge_segment(dep_0330)
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.0), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(dep_0330["timestamp"], dep_0330["action_mw"], color="#d62728", linewidth=1.8)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("策略功率 (mW)")
    ax1.set_title("圖 2：03-30 放電證據（命令 / SoC / 電池功率對齊）", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.25)

    ax2.plot(dep_0330["timestamp"], dep_0330["soc_pct"], color="#1f77b4", linewidth=1.8)
    ax2.set_ylabel("SoC (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.25)

    ax3.plot(raw_0330["timestamp"], raw_0330["batt_w"], color="#2ca02c", linewidth=1.2)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_ylabel("電池功率 (W)\n(+充電 / -放電)")
    ax3.grid(alpha=0.25)

    for ax in axes:
        ax.axvspan(t0, t1, color="#ff9896", alpha=0.22)

    ax1.annotate(
        "這段開始連續負功率",
        xy=(t0, float(dep_0330["action_mw"].min()) * 0.4),
        xytext=(t0, float(dep_0330["action_mw"].max()) * 0.55),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.5},
        color="#d62728",
        fontsize=10,
    )
    ax2.text(t0, 92, "放電期間 SoC 持續下降", fontsize=10, color="#1f77b4")
    ax3.text(t0, min(-0.05, raw_0330["batt_w"].min() * 0.8), "電池功率轉負 = 實際放電", fontsize=10, color="#2ca02c")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    out = out_dir / "story02_focus_0330_evidence.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig03_powerflow_simple(raw_0330: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.plot(raw_0330["timestamp"], raw_0330["load_w"], label="負載", linewidth=1.3, color="#ff7f0e")
    ax.plot(raw_0330["timestamp"], raw_0330["grid_w"], label="市電", linewidth=1.2, color="#1f77b4")
    ax.plot(raw_0330["timestamp"], raw_0330["pv_w"], label="太陽能", linewidth=1.2, color="#9467bd")
    ax.plot(raw_0330["timestamp"], raw_0330["batt_discharge_w"], label="電池放電", linewidth=1.4, color="#2ca02c")
    ax.set_title("03-30 能流圖（僅保留電池放電）", fontsize=13, fontweight="bold")
    ax.set_ylabel("功率 (W)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=10)

    dis_mask = raw_0330["batt_discharge_w"] > 0
    if dis_mask.any():
        t_first = raw_0330.loc[dis_mask.idxmax(), "timestamp"]
        ax.axvline(t_first, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(t_first, ax.get_ylim()[1] * 0.86, "電池放電開始", color="#2ca02c", fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    out = out_dir / "story03_powerflow_simple_0330.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig04_energy_mix_simple(kpis: List[DailyKPI], out_dir: Path) -> Path:
    df = pd.DataFrame([k.__dict__ for k in kpis])
    before = df[df["date"] != "2026-03-30"]
    after = df[df["date"] == "2026-03-30"]

    before_mean = {"太陽能": before["pv_wh"].mean(), "市電": before["grid_wh"].mean(), "電池放電": before["batt_discharge_wh"].mean()}
    after_val = {
        "太陽能": float(after["pv_wh"].iloc[0]),
        "市電": float(after["grid_wh"].iloc[0]),
        "電池放電": float(after["batt_discharge_wh"].iloc[0]),
    }

    labels = ["太陽能", "市電", "電池放電"]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    b1 = ax.bar(
        x - width / 2,
        [before_mean[k] for k in labels],
        width=width,
        color="#7f7f7f",
        alpha=0.85,
        label="03-27~03-29 平均",
    )
    b2 = ax.bar(x + width / 2, [after_val[k] for k in labels], width=width, color="#1f77b4", alpha=0.9, label="03-30")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("日能量 (Wh/day)")
    ax.set_title("圖 4：03-30 供能來源有改變（對照前 3 日平均）", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")

    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}", ha="center", fontsize=9)

    delta_batt = after_val["電池放電"] - before_mean["電池放電"]
    ax.text(1.95, max(after_val.values()) * 0.88, f"電池放電提升 +{delta_batt:.1f} Wh", color="#1f77b4", fontsize=10)

    plt.tight_layout()
    out = out_dir / "story04_energy_mix_simple.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def write_captions(out_dir: Path, kpis: List[DailyKPI]) -> Path:
    df = pd.DataFrame([k.__dict__ for k in kpis]).sort_values("date")
    d30 = df[df["date"] == "2026-03-30"].iloc[0]
    before = df[df["date"] != "2026-03-30"]

    msg = []
    msg.append("# 0330 圖說（易懂版）")
    msg.append("")
    msg.append("每張圖只對應一個清楚結論。")
    msg.append("")

    msg.append("## `story01_clear_takeaway.png`")
    msg.append(
        f"- 只有 03-30 有明顯放電（{d30['discharge_minutes']/60.0:.1f} 小時），"
        f"且 SoC 同步下降 {d30['soc_delta_pct']:.1f}%。前 3 天幾乎沒放電。"
    )
    msg.append("")

    msg.append("## `story02_focus_0330_evidence.png`")
    msg.append(
        "- 紅色陰影區是連續放電時段；上圖命令轉負、中圖 SoC 下降、下圖電池功率轉負，三個證據一致。"
    )
    msg.append("")

    msg.append("## `story03_powerflow_simple_0330.png`")
    msg.append(
        "- 03-30 當天能流分配中，電池只保留放電功率（不含充電），且方向與負載/市電/太陽能一致，方便直接比較。"
    )
    msg.append("")

    msg.append("## `story04_energy_mix_simple.png`")
    msg.append(
        f"- 03-30 的電池放電能量為 {d30['batt_discharge_wh']:.1f} Wh，"
        f"相比前 3 日平均 {before['batt_discharge_wh'].mean():.1f} Wh 明顯增加。"
    )
    msg.append("")

    msg.append("## 30 秒口頭結論")
    msg.append("")
    msg.append("- 模型在 03-30 才進入可持續放電。")
    msg.append("- 放電命令、SoC 下降、電池功率轉負三者一致。")
    msg.append("- 可解讀為：模型有放電能力，但觸發時機偏晚。")
    msg.append("")

    out = out_dir / "story_captions.md"
    out.write_text("\n".join(msg), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_by_date: Dict[str, pd.DataFrame] = {}
    dep_by_date: Dict[str, pd.DataFrame] = {}
    kpis: List[DailyKPI] = []

    for d in DATES:
        dep, raw = _load_day(input_dir, d)
        dep_by_date[d] = dep
        raw_by_date[d] = raw
        kpis.append(_daily_kpi(d, dep, raw))

    f1 = fig01_clear_takeaway(kpis, out_dir)
    f2 = fig02_focus_0330(dep_by_date["2026-03-30"], raw_by_date["2026-03-30"], out_dir)
    f3 = fig03_powerflow_simple(raw_by_date["2026-03-30"], out_dir)
    f4 = fig04_energy_mix_simple(kpis, out_dir)
    cap = write_captions(out_dir, kpis)

    print("Generated:")
    for p in [f1, f2, f3, f4, cap]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()

# 四面板繪圖操作指南

本工具把每日 `deployment_v2_*.csv` 與 `raw_data_v2_*.csv` 轉成四面板 PNG。技術細節見 [README_AI.md](README_AI.md)，全專案資料交接見 [data_and_plotting.md](../../docs/handover/data_and_plotting.md)。以下命令都從 `conformal-microgrid-rl` repository root 執行。

## 兩條 pipeline

- `data_verification`：直接畫量測／部署資料，不執行 Monte Carlo（MC）。
- `mc_replay`：以簡化 MC replay 顯示參考 SoC、MC 中位數與 5–95% 區間。

## 安裝

需要 Python 3.10 以上版本。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r tools/plotting_handoff/requirements.txt
```

macOS／Linux 將前兩行改為 `python3 -m venv .venv`、`source .venv/bin/activate`。

## 最常用命令

資料驗證（`--view`：`both`、`power`、`voltage-current`）：

```powershell
python tools/plotting_handoff/data_verification/dataset_to_figures.py --data-dir tools/plotting_handoff/dataset --start-date 2026-07-17 --end-date 2026-07-19 --start-time 08:00 --end-time 17:00 --output-dir tools/plotting_handoff/data_verification/example_output
```

MC replay（`--view`：`both`、`command`、`voltage-current`）：

```powershell
python tools/plotting_handoff/mc_replay/dataset_to_figures.py --data-dir tools/plotting_handoff/dataset --start-date 2026-07-17 --end-date 2026-07-19 --start-time 08:00 --end-time 17:00 --output-dir tools/plotting_handoff/mc_replay/example_output
```

## 輸入與輸出

- 輸入目錄須含同日配對的 `deployment_v2_YYYY-MM-DD.csv` 與 `raw_data_v2_YYYY-MM-DD.csv`；內附範例在 `tools/plotting_handoff/dataset/`。MC 也可用 `--input-csv` 讀取已轉換的單一 CSV。
- 日期與時間合成一段連續區間，例如 `2026-07-17 08:00` 至 `2026-07-19 17:00`，不是每天只取 08:00–17:00。
- 預設各產生兩張 300 dpi PNG，位置分別為 `data_verification/example_output/` 與 `mc_replay/example_output/`；可用 `--output-dir`、`--name` 改位置與檔名前綴。

## 下一步

先用內附資料執行對應 pipeline；確認圖面與時間範圍後，再把 `--data-dir` 和 `--output-dir` 換成自己的路徑。若要解讀欄位、MC 假設或物理限制，先讀 [README_AI.md](README_AI.md)。

## 三個禁止事項

1. 禁止把 PV 與 grid 解讀成嚴格二選一，或由 `grid == 0` 宣稱全太陽能供電。
2. 禁止把電池放電解讀成與 PV／grid 並聯的部分助力來源。
3. 禁止把簡化 MC 區間當成完整物理可信區間，或以單日好圖宣稱跨日穩定／論文就緒。

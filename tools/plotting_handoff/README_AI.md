# 四面板繪圖技術交接

本文件供 AI／維護者理解實作與限制；人類日常操作見 [README.md](README.md)，全專案資料與繪圖交接見 [data_and_plotting.md](../../docs/handover/data_and_plotting.md)。所有命令皆假設目前目錄為 `conformal-microgrid-rl` repository root。

## Pipeline 與 source map

兩條 pipeline 彼此獨立：

- `data_verification/`：不執行 MC，直接繪製量測／部署資料。
  - `dataset_to_figures.py`：CLI、連續時間篩選、四面板繪圖與 300 dpi PNG 輸出。
  - `outdata_folder_loader.py`：讀取每日 deployment/raw 配對、欄位正規化、15 分鐘 raw aggregation、causal lag 與 segment 建立。
- `mc_replay/`：執行簡化的 1000-run MC replay。
  - `dataset_to_figures.py`：資料夾或已轉換單一 CSV 的 CLI 入口，選擇 command／voltage-current 圖。
  - `outdata_folder_loader.py`：與資料驗證 pipeline 同目的的每日檔案轉換器。
  - `common_4panel_plotting.py`：canonical schema 驗證、MC、邊界判定、等效命令、前三面板及共用排版。
  - `plot_4panel_mc_command.py`：第四面板為參考／MC 電池功率命令。
  - `plot_4panel_voltage_current.py`：第四面板為電池電壓／電流。
- `dataset/`：可執行的三日範例輸入。
- `*/example_output/`：參考 PNG，不是執行時必要資料。
- `requirements.txt`：Python 相依套件。

## 輸入、欄位與資料流

資料夾輸入依日期讀取：

```text
deployment_v2_YYYY-MM-DD.csv ─┐
                              ├─ loader → canonical frame → plot／MC → PNG
raw_data_v2_YYYY-MM-DD.csv ───┘
```

每日檔案先依 `timestamp` 排序；重複 timestamp 保留排序後最後一筆。日期與時間參數定義一個含端點的連續區間。deployment 是主要時間軸；raw 的 `voltage_v`、`current_ma` 先按 15 分鐘 floor bin 取平均，再以 timestamp left join，缺值才回退到 deployment aggregation。

來源欄位轉為 canonical frame：

- `load_w`：優先 `load_kw × 1000`，否則 `load_p_mean_mW ÷ 1000`。
- `pv_w`：優先 `pv_kw × 1000`，否則 `mppt_mean_mW ÷ 1000`。
- `soc_pct`：`soc` 或 `soc_coulomb`；若最大值不超過 1.5，視為 0–1 比例並乘 100。
- `flow_pct`：`flow_pct_cmd` 或 `action_flow_pct`。
- `battery_power_w`：優先 `action_power_kw × 1000`；否則 `power_mw_cmd ÷ 1000`，有 `situation_code` 時 code 1 轉為負號。
- `voltage_v`、`current_ma`：優先同 timestamp 的 raw 15 分鐘平均，否則 `batt_v_mean`、`batt_i_mean_ma`。
- 輔助欄位：`day_number`、`x_day`、`date_label`、`segment_id`、`causal_lag_applied`。

若 canonical 數值欄位含缺值，loader 直接報錯。MC 的 `--input-csv` 可跳過每日檔案 loader，但至少必須包含 `timestamp`、`load_w`、`pv_w`、`soc_pct`、`flow_pct`、`voltage_v`、`current_ma`、`battery_power_w`；timestamp 不可重複，且至少兩列。`day_number`、`x_day`、`segment_id` 可由程式補建。

## Causal lag

每日檔案轉換預設對 `battery_power_w` 與 `flow_pct` 套用一個取樣步的 causal lag，使命令與下一筆觀察到的後果依因果順序對齊。首筆及超過 30 分鐘資料缺口後的首筆重設為 0，並以 `causal_lag_applied=True` 標記。

這是資料時間對齊規則，不代表硬體具有固定一取樣步的純延遲。MC 讀到該標記時使用同列命令；未標記時使用前一列命令，避免重複位移。

## 11.2 Wh 與等效功率

資料驗證的 SoC 等效電池功率，以及 MC replay，均採用 `11.2 Wh` 名目容量。資料驗證以 SoC 差分除以實際時間差換算，僅接受 `0 < Δt ≤ 0.5 h`，結果裁切至 `[-6, 9] W`，小於 `0.10 W` 的值設為 0。這是推導量，不是獨立量測。

## 1000-run MC 簡化假設

MC 固定執行 1000 次，亂數種子為 `20260722`。每次抽樣會擾動：

- 容量：`11.2 ± 0.35 Wh`，裁切至 `[9.8, 12.8] Wh`。
- 充電／放電 transfer gain：中心分別為 `0.93`、`1.175`，標準差 `0.030`，各自裁切。
- command gain：中心 `1.0`、標準差 `0.030`，裁切至 `[0.85, 1.15]`。
- SoC 下／上界：中心 `0.20`、`0.80`，標準差 `0.004`，各自裁切。
- 初始及重設 SoC：參考值加標準差 `0.006` 的雜訊後裁切。
- 每列命令雜訊：標準差 `0.025 W`；絕對值低於 `0.015 W` 時設為 0。

一般步進按實際 `Δt` 更新；非正時間差回退為 0.25 小時。遇到超過 0.5 小時的真實缺口會重設。若相鄰 SoC 跳變至少 0.05，且跳變前後命令絕對值都低於 0.015 W，也視為無法解釋的 seam 並重設。每次步進均約束在該 run 抽樣的 SoC 上下界。

SoC 面板畫參考軌跡、1000-run 中位數及 5–95% 區間；command 圖另將各 run 的 SoC 反推成等效命令並畫中位數與 5–95% 區間。這是供繪圖與不確定性呈現的簡化 replay，不是完整電化學模型、硬體數位分身、來源辨識器，也不是經校準後可泛化的機率可信區間。

## 連續時間區間

`--start-date`／`--start-time` 與 `--end-date`／`--end-time` 合成單一連續區間，例如：

```text
2026-07-17 08:00 → 2026-07-19 17:00
```

它不是「每天只取 08:00–17:00」。省略日期時採用 deployment 可用首末日；省略時間時使用 `00:00` 至 `23:59:59`。繪圖線只在超過 30 分鐘的真實 timestamp gap 斷開，不因換日而中斷。

## 供能物理語意

- **PV–grid mixed supply**：真實平台允許 PV 與 grid 同時支援負載，不可解讀為嚴格的 solar／grid 二選一。
- **Battery discharge solo-only**：電池不是第三個並聯部分助力來源；若允許放電，電池必須具備獨自供應負載的能力，不可解讀為與 PV／grid 同時分攤負載。
- `PV power`、`load`、`grid draw`、瞬時電壓比較或 `grid == 0` 都不能單獨證明某來源完全供電。
- 圖中 PV 與負載曲線代表支援程度與需求的關係，不是精確來源歸因。論文與圖說應使用「PV 支援增加」「grid 需求降低」等可支持的措辭。

## 安裝與驗證命令

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r tools/plotting_handoff/requirements.txt
```

兩條 pipeline 的最小全資料驗證：

```powershell
python tools/plotting_handoff/data_verification/dataset_to_figures.py --data-dir tools/plotting_handoff/dataset --output-dir tools/plotting_handoff/data_verification/example_output
python tools/plotting_handoff/mc_replay/dataset_to_figures.py --data-dir tools/plotting_handoff/dataset --output-dir tools/plotting_handoff/mc_replay/example_output
```

指定連續區間與單一 view：

```powershell
python tools/plotting_handoff/data_verification/dataset_to_figures.py --data-dir tools/plotting_handoff/dataset --start-date 2026-07-17 --end-date 2026-07-19 --start-time 08:00 --end-time 17:00 --view voltage-current --output-dir tools/plotting_handoff/data_verification/example_output
python tools/plotting_handoff/mc_replay/dataset_to_figures.py --data-dir tools/plotting_handoff/dataset --start-date 2026-07-17 --end-date 2026-07-19 --start-time 08:00 --end-time 17:00 --view command --output-dir tools/plotting_handoff/mc_replay/example_output
```

預設 `both` 各產生兩張 300 dpi PNG。`--name` 可指定檔名前綴；MC 另支援 `--title-prefix`，以及 `--input-csv`／`--input` 載入 canonical 單一 CSV。macOS／Linux 使用對應的虛擬環境啟用命令，Python 執行命令不變。

## 已知限制

- `example_output/` 是參考成果；一般驗證不應覆寫，除非明確要更新範例。
- raw 電壓／電流只按 15 分鐘 floor bin 平均並以完全相同 timestamp 合併；不是一般 nearest/as-of join。
- 缺少任一 canonical 數值欄位時會失敗；不同 runtime／vendor schema 仍須先確認欄位語意。
- `11.2 Wh`、增益、邊界、雜訊與 seam 規則都是此繪圖 replay 的固定簡化假設，不能外推為電池真實參數。
- MC 區間不能宣稱為完整物理模型可信區間，圖也不能作精確來源歸因。
- 單日圖只能作局部檢查；跨日控制穩定與論文品質必須另以連續多日模擬／實驗驗證。
- 任何論文結論都須區分量測量、推導量、命令與模型輸出，不得因圖形看似合理而過度宣稱。

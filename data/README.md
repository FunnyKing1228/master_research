# 資料入口

本頁是人員操作簡版；AI／維護者請讀 [`README_AI.md`](README_AI.md)。

## 這裡做什麼

`data/` 保存資料處理程式與本機資料工作區：`scripts/preprocessing/` 放不同來源、不同目的的前處理腳本，`processed/` 放訓練可讀的衍生 CSV。實際訓練資料永遠以所用 YAML 的 `env.dataset_csv_path` 為準。

## 現場 deployment 資料怎麼放

從實驗電腦取得資料後，將**同一天的兩個 CSV 成對放入 `data/raw/`**：

```text
data/raw/
  deployment_v2_2026-07-17.csv
  raw_data_v2_2026-07-17.csv
  deployment_v2_2026-07-18.csv
  raw_data_v2_2026-07-18.csv
```

- `deployment_v2_YYYY-MM-DD.csv` 與 `raw_data_v2_YYYY-MM-DD.csv` 的日期必須相同。
- 每一天都應有一組；只有其中一個檔案代表資料不完整。
- 保留原始檔名與內容，不要覆寫或先手動合併。
- 多數現有診斷、繪圖與歷史 preprocessing script 預設直接讀 `data/raw/`，因此最安全的做法是把成對檔案放在這一層。
- `data/raw/` 已被 Git 忽略；這些現場 CSV 只留在本機，不提交到 repository。

## 開始命令

先從 repository 根目錄確認目前 v22 指向的資料存在：

```powershell
$cfg = Get-Content configs\experiments\p302\config_p302_v22_flow_power_limited.yaml -Raw
Test-Path data\processed\training_v20_0511_strict_clean_full_days_raw_only.csv
```

前處理沒有單一通用一鍵命令；先依來源 schema 與目的選腳本。若原檔符合通用欄位別名，可明示輸入、輸出執行：

```powershell
py data\scripts\preprocessing\preprocess_raw_to_15min.py `
  --input <本機原始CSV> `
  --output data\processed\<新資料集>.csv `
  --window_min 15
```

## 輸入

- 外部／本機保存的現場原始 CSV（不可變、不可提交）。
- `data/raw/` 中依日期成對的 `deployment_v2_*.csv` 與 `raw_data_v2_*.csv`。
- 個別腳本指定的既有 `data/processed/*.csv` 或日期集合。
- 欄位、單位、時間戳與 vendor controller 版本紀錄。

## 輸出

- `data/processed/*.csv`：清理、聚合或合成後的訓練資料。
- `data/soh_segments*/`：SoH 趨勢檢查用片段，不是真實 SoH 標籤。
- 執行訓練後的模型、log 與圖不在此處，而在 `experiments/<name>/`。

## 下一步

確認輸出 CSV 的時間連續性、單位、缺值、PV/load 欄位與切分方式，再把路徑寫入可攜 YAML 的 `env.dataset_csv_path`；訓練與跨日驗證流程見 [`../docs/handover/data_and_plotting.md`](../docs/handover/data_and_plotting.md) 與 [`../docs/handover/training.md`](../docs/handover/training.md)。

## 三個禁止事項

1. 禁止提交現場原始檔、含個資／機器路徑的 CSV 或大量衍生資料。
2. 禁止假裝 `data/scripts/preprocessing/` 有適用所有 schema 的單一一鍵流程。
3. 禁止只看檔名猜訓練輸入，或覆蓋既有原始檔；以 `env.dataset_csv_path` 與實際欄位查核為準。

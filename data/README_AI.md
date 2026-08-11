# Data AI Map

本頁供 AI／自動化工具定位資料管線；先讀 [`../AGENTS.md`](../AGENTS.md)，人員操作簡版見 [`README.md`](README.md)。深入 SSOT 是 [`../docs/HANDOVER_zh.md`](../docs/HANDOVER_zh.md) 與 [`../docs/handover/data_and_plotting.md`](../docs/handover/data_and_plotting.md)，衝突時以可執行 source、實驗封存 config 與交接文件的最新可驗證證據為準。

## 真實 source map

- `data/scripts/preprocessing/preprocess_raw_to_15min.py`：具 CLI 的通用欄位別名解析與固定時間窗聚合；只適用能對上其 timestamp/PV/load aliases 的 CSV。
- `data/scripts/preprocessing/build_training_dataset.py`：讀 `data/raw/collected_data_*.csv`，做欄位正規化、日期挑選與訓練資料建構；內含特定 P302 常數與負載排程。
- `data/scripts/preprocessing/generate_hybrid_training_dataset.py`：以 `data/processed/training_v16.csv` 的完整日模板產生可重現 hybrid synthetic dataset。
- `data/scripts/preprocessing/build_0504_curated_training_dataset.py`：合併既有 hybrid dataset 與指定 deployment 日期；日期與排除 freeze 日寫死，屬歷史專用流程。
- `data/scripts/preprocessing/gen_v14_data.py`：歷史 v14、固定 seed 的 14 日 synthetic generator，不是現行現場資料清理器。
- `data/scripts/preprocessing/export_soh_segments.py`：從 `raw_data_v2_*.csv` 匯出保守的充放電片段，只供 SoH trend check，不產生 ground-truth SoH。
- `data/processed/`：本機衍生訓練 CSV；被 `.gitignore` 排除。
- `core/microgrid_env.py`：`dataset_csv_path` 的最終消費者，載入時間、PV、load、price 並形成 observation／episode。
- `core/train_sac_microgrid.py::load_config()`：解析相對 `env.dataset_csv_path`；依目前工作目錄、YAML 目錄、YAML 上一層嘗試存在的候選。

## 現場 CSV 放置契約

同一實驗日的 `deployment_v2_YYYY-MM-DD.csv` 與 `raw_data_v2_YYYY-MM-DD.csv` 必須保留原檔名，成對放在 `data/raw/`。多日資料就是多組同日期配對；缺少任一檔案時應標記該日資料不完整，不可靜默假設另一份可替代。

多數歷史 diagnostics、figures 與部分 preprocessing scripts 直接以 `data/raw/` 為 `RAW_DIR`，不會遞迴搜尋任意子目錄。除非使用者明確傳入支援的 `--data-dir`／input path，否則不要自行改成其他層級。

## 資料／config 流

```text
實驗電腦 results/deployment/
  deployment_v2_YYYY-MM-DD.csv
  raw_data_v2_YYYY-MM-DD.csv
  -> 成對複製至 data/raw/，保留檔名與內容
  -> 選定且人工查核過的 preprocessing script
  -> data/processed/<dataset>.csv
  -> configs/experiments/p302/<source-config>.yaml
       env.dataset_csv_path
  -> core/train_sac_microgrid.py
  -> core/microgrid_env.py
  -> experiments/<name>/configs/experiment_config.yaml
     + logs/ + models/ + results/
```

現行 v22 可攜 config 是 `configs/experiments/p302/config_p302_v22_flow_power_limited.yaml`，其 `env.dataset_csv_path` 指向 `data/processed/training_v20_0511_strict_clean_full_days_raw_only.csv`。每次實驗的實際解析結果應回查 `experiments/<name>/configs/experiment_config.yaml`；它可能含本機絕對路徑，不能當跨機器可攜來源。

## 不可破壞 invariants

- 原始現場檔不可變、不可提交；清理結果另寫新檔，保留來源、vendor 版本、schema、單位與時間範圍。
- 不存在單一可信的一鍵 preprocessing。先核對腳本的硬編碼來源、欄位、日期、單位與輸出，再執行。
- `env.dataset_csv_path` 是訓練輸入權威；不可從檔名、最新修改時間或某張圖倒推。
- PV 與 grid 可 mixed supply；不可由 `grid == 0`、bus/grid 電壓或單一 boolean 宣稱來源互斥。
- 觀測保留連續 `pv_support_ratio`；PV availability、support level 與 stateful/hysteretic discharge blocking 不可混成同一瞬時真值。
- battery discharge 是 solo-only，不可把電池建模或繪圖成與 PV/grid 並聯的 partial assist。
- 資料切分不得洩漏未來資訊；跨日驗證須保持連續 SoC，不得每日 reset。

## 已知技術債

- 多支 preprocessing script 含歷史硬編碼日期、常數、來源檔名與不同 schema 假設，尚無統一 manifest／provenance sidecar。
- `build_training_dataset.py` 的 `DATA_ROOT` 由 script 目錄上一層推導，歷史路徑假設需在使用前實測；不得假設它代表 repo 的 `data/`。
- 部分歷史資料含瞬時 `PV_bool`；它不能取代連續 PV support 與 hysteretic blocking semantics。
- `data/processed/` 被 Git 忽略，clone 後不保證資料存在；experiment config 也可能封存機器專屬絕對路徑。
- vendor `Data.txt`／deployment CSV schema 隨控制器版本演進，parser 能讀多種格式不代表語意完全相容。

## 驗證命令

從 repo 根目錄執行：

```powershell
Test-Path data\scripts\preprocessing
Test-Path data\processed
Test-Path data\processed\training_v20_0511_strict_clean_full_days_raw_only.csv
py data\scripts\preprocessing\preprocess_raw_to_15min.py --help
py core\train_sac_microgrid.py --help
py -m pytest tests\test_microgrid_env.py
git status --short --untracked-files=all
```

資料與圖表的完整 schema、vendor protocol、驗證窗口與輸出規則續見 [`../docs/handover/data_and_plotting.md`](../docs/handover/data_and_plotting.md)；實驗選擇見 [`../docs/handover/experiments.md`](../docs/handover/experiments.md)，release 證據見 [`../docs/handover/release_manifest.md`](../docs/handover/release_manifest.md)。

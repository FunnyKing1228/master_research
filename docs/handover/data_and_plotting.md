# 資料與繪圖交接

本文件區分訓練資料、真機交換檔、部署日誌、SoH 資料與繪圖產物。原始資料、checkpoint、release、產生圖與本機絕對路徑通常不應 commit。

## 訓練 CSV

訓練資料位於 `data/processed/`。每個 experiment 的實際輸入以其：

```text
experiments/<experiment>/configs/experiment_config.yaml
  env.dataset_csv_path
```

為準，不要只依檔名猜測。`v22_flow_power_limited_gpu300` 當時設定的輸入為：

```text
data/processed/training_v20_0511_strict_clean_full_days_raw_only.csv
```

config 內可能保存建立機器的絕對 Windows 路徑；換機時應解析為 repository 內對應相對路徑，不要把個人路徑寫回公共文件。訓練輸出主要在：

```text
experiments/<experiment>/logs/episode_log.csv
experiments/<experiment>/models/best_sac_model.pth
experiments/<experiment>/models/final_sac_model.pth
experiments/<experiment>/results/training_results.png
experiments/<experiment>/results/training_results.json
experiments/<experiment>/results/sac_training_metrics.npz
```

其中 `episode_log.csv` 是每 episode 的訓練統計；它不是硬體 raw log。`best` 與 `final` 必須依 manifest/驗證結果明確區分。

## `Data.txt`：vendor → AI

真機執行時，vendor controller 寫 `Data.txt`，AI 由 `control/io_protocol.py` / `control/run_deployment.py` 讀取。流程為：

```text
vendor controller -> Data.txt -> raw sample buffer
  -> 15 分鐘 aggregation/state -> policy -> guards
```

`P302V2.4`、`P302_AI_v2.5`、`P302_AI_V4.0` 是廠商軟體版本名稱，其 `Data.txt`／`Command.txt` protocol 有差異。正式資料來源是實驗用電腦上的實際版本；開發／打包電腦保留的廠商軟體副本只供 GUI 打包後測試。保存 CSV 時必須一併記錄廠商版本，不能只記「P302」。

目前 parser 可處理的重點格式如下，但這不是所有廠商版本完全相容的保證：

- 第一行：`YYYYMMDDhhmmss,{load_groups}`，其中負載組數記為 `vendor_load_count`；
- MPPT：Solar 與 MPPT 的 V/I/P，新格式可另含 bus V/I/P；
- load：舊格式可只有 load V/I/P，新格式可再含 grid V/I/P；
- battery：支援既有六欄與含 charge voltage 的七欄資料列。

讀取函式可在成功／嘗試讀取後清空檔案，因此 `Data.txt` 是交換緩衝，不是長期資料庫。需要保留的量測必須寫入 raw CSV。

## `Command.txt`：AI → vendor

`Command.txt` 的現行 vendor 格式為：

```text
{situation_code}
YYYYMMDDhhmmss,{load_groups}
PP,power_mW,flow_percent,
```

重要語意：

- power 欄為 mW；程式內 policy/action 常使用 kW，寫檔前需正確換算；
- flow 欄為百分比；
- rest 與 pre-measure 使用 mode 3；
- 零功率仍保留實體 battery PP（例如 `01`），不可改成 `00`，否則 flow command 可能失去目標；
- mode 4 是明確停 motor/battery 的情境，不是一般 standby；
- 最終寫入 command 可因 CORAL、電壓、PV、SoC、flow-power 與硬體 guard 而不同於 raw policy action。

## deployment CSV 與 raw CSV

預設 log 目錄為 repository／release 下的：

```text
results/deployment/
```

GUI 可透過 `config_gui.json` 的 `log_dir` 指定其他本機目錄。每次執行會按日期產生：

```text
results/deployment/deployment_v2_YYYY-MM-DD.csv
results/deployment/raw_data_v2_YYYY-MM-DD.csv
```

- `raw_data_v2_*`：約每次輪詢的原始／解析後量測，適合檢查 freeze、時間戳、電壓、電流、MPPT、load/grid 與 vendor load count。
- `deployment_v2_*`：每個模型決策窗的 aggregation、SoC、raw/safe/final action、guard、命令與狀態，適合 closed-loop 行為、阻擋原因與 replay 分析。
- 目前兩類 CSV 都會記錄／推導 `vendor_load_count`、`load_power_per_unit_w`、`load_power_est_w`；估算值是組數乘每組功率，不可誤當 load power 的獨立直接量測。

分析時先確認 CSV schema、產生該 CSV 的 AI runtime 版本，以及實驗電腦上的廠商軟體版本。Stable pre-measure／probe 欄位只存在於較新的 runtime CSV；舊 CSV 不會因 source 已更新而自動補欄位。

## SoH 輸入與輸出

SoH 模型 artifacts 可位於 `soh_models/` 或 GUI／CLI 指定目錄，常見副檔名為 `.pth`、`.pkl`、`.npz`。啟用 online SoH 後輸出：

```text
<log_dir>/soh_online_segments/
  soh_online_predictions.csv
  cycles/cycle_soh_online_YYYYMMDD_HHMMSS_NNNN.csv
```

cycle CSV 欄位為 time、voltage、current，供 feature extraction 與預測；prediction CSV 保存每次 SoH 結果與狀態。SoH 目前應視為整合與蒐集管線，未經電池專屬老化資料驗證前，不應啟用 `--soh-use-for-capacity` 或把預測當成真實容量。

## 繪圖輸入與輸出

繪圖程式分為：

```text
data/processed/plot_*.py          訓練資料概覽
experiments/plot_experiment_behavior.py
data/scripts/figures/             驗證、部署、報告與論文圖
data/scripts/diagnostics/         freeze、cutoff、replay、PV/load 等診斷
```

常見輸入與輸出：

- dataset 圖：輸入 `data/processed/training_*.csv`，輸出至 experiment `results/` 或指定 `--output`；
- experiment behavior：輸入 experiment config、checkpoint 與其 dataset CSV，輸出至 experiment `results/`；
- deployment 圖：輸入 `raw_data_v2_*.csv` 和／或 `deployment_v2_*.csv`，輸出 PNG/PDF、摘要 CSV/JSON/Markdown 至命令列指定目錄、`outputs/` 或 `data/raw/figures/`；
- SoH 圖：輸入 cycle／prediction CSV，輸出 report PNG；
- script 若有 `--output-dir`、`--output`、日期或 dataset override，應優先明確傳入，避免沿用程式內歷史絕對路徑或覆蓋舊圖。

圖產生後要保存「script、輸入 CSV、checkpoint、config、時間窗與輸出路徑」的對應關係。單日圖只能作局部檢查；模型要作論文或 release 證據，必須另做連續多日驗證。

## 物理與圖表解讀守則

- **PV 與 grid 是 mixed supply**：兩者可同時支援負載，不能畫成嚴格「太陽能或市電」二選一。
- 使用連續 `pv_support_ratio` 描述 PV 支援程度；需要阻擋放電時，應使用具狀態與 hysteresis 的 sufficiency/blocking 狀態，不用抖動的瞬時 boolean 充當全部物理真相。
- `grid == 0`、bus/grid 電壓比較或單一瞬時門檻，都不能證明「負載完全由 PV 供應」。
- **battery discharge 是 solo-only**：電池不能被畫成與 PV/grid 並聯的第三個部分助力來源。有效放電必須能獨立接管負載；否則應阻擋／待機。
- 圖標與 caption 優先使用「PV 支援增加」「grid demand 降低」「battery power／SoC」等可量測敘述，不要過度宣稱來源排他性。

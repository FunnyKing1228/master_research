# newHW 重現流程：從原始 CSV 到 in-sample rollout

> 本頁供人員與 AI 直接執行；先讀 [`../../AGENTS.md`](../../AGENTS.md)。
> 背景與各站狀態見 [`newHW_lifecycle_mapping.md`](newHW_lifecycle_mapping.md)；
> 缺口清單見 [`newHW_pending_data.md`](newHW_pending_data.md)。
>
> **本頁只涵蓋站①②③（in-sample）。站④部署與站⑤現場繪圖被 I/O 規格缺口阻擋，不在範圍內。**

## 給 AI 助手的執行摘要

> 若你是 AI 助手（Cursor、Claude Code 等）被要求復刻本流程，先讀本節，可省去大量探索。

**環境前提**

- 本流程在使用者本機執行，直接讀寫 repository。若你在無法存取本機檔案的沙盒環境，先向使用者索取 `Data140826.csv`。
- 安裝 `pandas<3`。pandas 3.x 起 `to_datetime` 預設單位由奈秒改為微秒，會使 `prepare_data_newHW.py` 的 `first_night_independent_discharge_wh` 少算 1000 倍。此欄位僅供人工查閱，不影響資料集、訓練與 rollout，但會讓 `git status` 顯示該 JSON 被改動。

**一次跑完的最短路徑**（Windows PowerShell；Linux／macOS 把 `py` 換 `python3`、路徑分隔符換 `/`、移除行尾反引號）

```powershell
py -m pip install torch numpy "pandas<3" pyyaml matplotlib gymnasium scipy pytest
New-Item -ItemType Directory -Force data\newHW\raw
copy <來源>\Data140826.csv data\newHW\raw\Data140826.csv

py data\scripts\newHW\prepare_data_newHW.py --input data\newHW\raw\Data140826.csv --output data\newHW\processed\training_newHW_15min.csv --summary data\newHW\processed\newHW_data_quality_summary.json
py core\train_sac_microgrid_newHW.py --config configs\config_newHW_soc20_80_sim.yaml --name newHW_repro_s42 --episodes 5
py core\train_sac_microgrid_newHW.py --config configs\config_newHW_soc20_80_sim.yaml --name newHW_repro_s42_full
py data\scripts\newHW\rollout_newHW.py --experiment newHW_repro_s42_full --model best_sac_model.pth
py data\scripts\newHW\rollout_newHW.py --experiment newHW_repro_s42_full --model final_sac_model.pth
```

第一次訓練刻意只跑 5 episodes 做 smoke，約 6 秒即產出結構相同的 best／final，可在投入完整 300 episodes（約 3 分鐘）前先確認環境無誤。

**產出位置**（供 AI 快速定位）

```text
experiments/<name>/results/in_sample_rollout_newHW/
  best_sac_model_rollout_newHW.png     ← 四面板 rollout 圖，通常是使用者要看的東西
  best_sac_model_summary_newHW.json    ← served_energy_fraction 等指標
  best_sac_model_audit_newHW.csv       ← 逐步稽核紀錄
  final_sac_model_*                    ← final checkpoint 的同組三個檔
```

**不需要資料也能先做的事**

`tests/test_newHW.py`（7 passed）與三個 P302 regression 檔（175 passed）在乾淨 clone 上**不需要任何 newHW 資料即可執行**。等待原始 CSV 期間可先完成安裝與這兩組測試。

**禁止事項**

- 不得修改 `core/microgrid_env.py`、`core/train_sac_microgrid.py`、`configs/experiments/p302/**`、`control/io_protocol.py`。
- 不得調整 newHW config 的物理參數或 reward 權重；那是研究決策，非實作選擇。
- 不得為湊既有數字而重跑挑結果或改 seed。
- 未獲明確要求不得 `git commit`。

---

## 0. 前置需求

| 項目 | 需求 |
|---|---|
| Python | 3.10 以上 |
| 必要套件 | `torch numpy pandas<3 pyyaml matplotlib gymnasium scipy pytest` |
| 原始資料 | `Data140826.csv`，SHA256 `9ada734a86a8aec822589d402b3ee639b62aa5fa3b7ec16025edf4af55e98881` |
| 工作目錄 | **必須在 repository root 執行**，所有相對路徑以此為基準 |

不要直接 `pip install -r requirements.txt`。該檔含 `pyinstaller`、`pvlib`、`openmeteo-requests`、`stable-baselines3`、`python-microgrid`，全部是 P302 部署或歷史流程用的，newHW 路徑一律不需要。

```powershell
py -m pip install torch numpy "pandas<3" pyyaml matplotlib gymnasium scipy pytest
```

`python-microgrid` 未安裝時，執行會印出一行 `Info: python-microgrid not installed...`。**這是預期行為，不是錯誤**，newHW 環境不使用該套件。

Windows 請使用 `py`；Linux／macOS 通常改用 `python3`。下列命令是 PowerShell
語法，其他 shell 請把路徑分隔符改為 `/`，並移除行尾反引號。

---

## 1. 放置原始資料

`data/newHW/raw/` 被 `.gitignore` 排除，clone 後**不存在**，必須手動建立。

```powershell
New-Item -ItemType Directory -Force data\newHW\raw
copy <來源路徑>\Data140826.csv data\newHW\raw\Data140826.csv
```

放置後先驗證雜湊，不符就停止，不要繼續往下跑：

```powershell
certutil -hashfile data\newHW\raw\Data140826.csv SHA256
```

預期值：`9ada734a86a8aec822589d402b3ee639b62aa5fa3b7ec16025edf4af55e98881`

建議把該檔設為唯讀，避免後續步驟意外覆寫。

---

## 2. 站①：資料準備

`data/newHW/processed/training_newHW_15min.csv` 未進版控，必須重新產生，否則訓練會找不到資料。

```powershell
py data\scripts\newHW\prepare_data_newHW.py `
  --input data\newHW\raw\Data140826.csv `
  --output data\newHW\processed\training_newHW_15min.csv `
  --summary data\newHW\processed\newHW_data_quality_summary.json
```

**驗收檢查點**

- `training_newHW_15min.csv` 共 189 行（1 行 header + 188 個 15 分鐘 bins）。
- summary JSON 的 `source_sha256` 與步驟 1 的雜湊一致。
- summary JSON 的 `notes` 應包含 `Load_W was replaced by the supplied 28.2 W regression-derived baseline.`

**已知瑕疵**：`prepare_data_newHW.py` 換算時間時假設 pandas 的 datetime 單位為奈秒。pandas 3.x 起預設改為微秒，會使 `first_night_independent_discharge_wh` 少算 1000 倍（`0.199` 而非 `199.07`），並讓 `git status` 顯示該 JSON 被改動。**請安裝 `pandas<3`**。此欄位僅供人工查閱，不影響資料集、訓練或 rollout。

---

## 3. 站②：模型訓練

有兩個 config，差別只在 SoC 操作範圍與預設 episode 數：

| Config | SoC 範圍 | 初始 SoC | 預設 episodes | 用途 |
|---|---|---|---|---|
| `configs/config_newHW_sim.yaml` | 0.10–0.90 | 0.90 | 50 | 原始 smoke |
| `configs/config_newHW_soc20_80_sim.yaml` | 0.20–0.80 | 0.80 | 300 | **最新診斷，建議用這個** |

```powershell
py core\train_sac_microgrid_newHW.py `
  --config configs\config_newHW_soc20_80_sim.yaml `
  --name newHW_repro_soc20_80_s42
```

**實驗名稱必須以 `newHW_` 開頭**，否則程式直接 `raise ValueError`。這是刻意的隔離保護，不要繞過。

需要縮短時間先確認流程可跑時，加 `--episodes 5` 做 smoke（約數秒）。正式重現請用 config 預設的 300。

若 `experiments\newHW_repro_soc20_80_s42\` 已存在，請換一個仍以 `newHW_`
開頭的新名稱，並在後續兩次 rollout 使用相同名稱，避免混淆不同次執行的產物。

**驗收檢查點**

執行結束後 `experiments/<name>/` 應含：

```
configs/experiment_config.yaml
logs/episode_log.csv
models/best_sac_model.pth
models/final_sac_model.pth
results/training_results.json
results/training_results_newHW.png
results/VALIDATION_STATUS_newHW.md
```

此外會產生 `models/sac_checkpoint_ep*.pth` 與
`results/sac_training_metrics_newHW.npz`；它們也屬 git-ignored 實驗產物。

啟動訊息應顯示 `State dim: 6`、`Action dim: 1`、`Variant: sac_sn`。動作維度若不是 1，代表載到了 P302 環境，停止並檢查 config。

---

## 4. 站③：in-sample rollout

```powershell
py data\scripts\newHW\rollout_newHW.py `
  --experiment newHW_repro_soc20_80_s42 `
  --model best_sac_model.pth
```

**兩個參數都不是路徑**：`--experiment` 傳實驗**名稱**（程式自行組出 `experiments/<name>/`），`--model` 傳 `models/` 底下的**檔名**。傳路徑會失敗。

`best` 與 `final` 都要各跑一次並保留比較。final 的供電與 SafetyNet 依賴率可能隨執行環境變動，因此不能只憑 checkpoint 名稱或單次 SafetyNet 數字選模。

產物位於：

```text
experiments/<name>/results/in_sample_rollout_newHW/
  best_sac_model_rollout_newHW.png      ← 教授要看的圖
  best_sac_model_summary_newHW.json     ← 數字在這
  best_sac_model_audit_newHW.csv
```

執行 final rollout 時，同一目錄會產生對應的 `final_sac_model_*` 三個檔案。

執行時可能先出現 `Gym has been unmaintained since 2022...`。這來自共用訓練
模組的相依 import；本次 Windows 實測仍可正常完成 rollout。它不是
`Action dim: 2` 或載入 P302 環境的證據。

**驗收檢查點**

輸出 JSON 應包含 `caveats` 區塊，明列 in-sample、無 held-out、無法跑 3 日／5 日等限制。若缺少此區塊，代表跑到錯的腳本。

---

## 5. 站③補充：能量上界與資料品質圖（選用）

```powershell
py data\scripts\newHW\figures\plot_data_newHW.py `
  --input data\newHW\processed\training_newHW_15min.csv `
  --output data\newHW\processed\data_quality_newHW.png

py data\scripts\newHW\analyze_energy_bound_newHW.py `
  --dataset data\newHW\processed\training_newHW_15min.csv `
  --config configs\config_newHW_soc20_80_sim.yaml `
  --output-json data\newHW\processed\energy_upper_bound_newHW.json `
  --output-trace data\newHW\processed\energy_upper_bound_trace_newHW.csv `
  --output-plot data\newHW\processed\energy_upper_bound_newHW.png
```

---

## 6. 迴歸測試

```powershell
py -m pytest tests\test_newHW.py -q
```

預期 7 passed。

確認未污染既有 P302 系統：

```powershell
py -m pytest tests\test_deployment.py tests\test_io_protocol.py tests\test_microgrid_env.py -q
```

本次實測為 `175 passed, 44 warnings`；44 個 warning 都是 Gymnasium 將
Box bounds 由 float64 降為 float32 的精度提醒，不是測試失敗。

---

## 7. 結果如何解讀

跑完之後**不要**把任何數字描述成模型驗證結果。目前已知的限制：

- 訓練與 rollout 使用同一份 47 小時資料，屬 in-sample。
- 沒有 held-out 日期，無法判斷泛化。
- 資料不足 3 日，既有的 3 日／5 日跨日驗證無法執行。
- reward 權重、BMS 限制、SoC 範圍、功率上限、初始 SoC 全部是暫定值。
- 系統為離網屬**推論**，未經硬體端確認。
- best raw policy 仍有高比例 steps 需要 SafetyNet 投影。

同一 seed 跨機器不保證位元級重現；best 指標對上既有紀錄可能只是策略已飽和於 finite-window oracle 天花板，不構成重現證據；final 的 SafetyNet 依賴率可能隨執行環境變動，不應單獨作為選模依據。

正確的描述是「隔離遷移骨架可執行，且在同一窗口達到 finite-window oracle」，而不是「模型可部署」或「可作為論文證據」。

---

## 8. 常見錯誤對照

| 訊息 | 原因 | 處理 |
|---|---|---|
| `newHW experiment names must start with 'newHW_'` | 實驗名未加前綴 | 改名，不要改程式 |
| `Only newHW_* experiments are accepted` | 同上，或 `--experiment` 誤傳路徑 | 只傳名稱 |
| `FileNotFoundError: training_newHW_15min.csv` | 跳過步驟 2 | 先跑 prepare |
| 整份 CSV 被解析成單一欄位 | 直接讀原始 CSV 未經 prepare | 原始 header 被包成單一 quoted field，必須經 prepare 處理 |
| `Info: python-microgrid not installed` | 選用套件未裝 | 預期行為，忽略 |
| `Gym has been unmaintained since 2022...` | 共用模組仍觸發舊 Gym 的相依 import | 本次實測不阻擋 newHW；先記錄，不要為此修改 P302 共用檔 |
| `first_night_independent_discharge_wh` 為 `0.199` | 使用 pandas 3.x，datetime 單位由奈秒改為微秒 | 改裝 `pandas<3` 後重新執行資料準備 |
| `Action dim: 2` | 載到 P302 環境 | 檢查 `--config` 是否指向 newHW config |

---

## 9. 2026-08-18 乾淨 publish clone 實測紀錄

### 環境

- OS：Windows 10 `10.0.22621`
- Python：3.11.9
- PyTorch：2.6.0+cu124；CUDA 12.4，訓練使用 `cuda`
- NumPy 2.3.2、pandas 2.3.1、PyYAML 6.0.2、matplotlib 3.10.5
- Gymnasium 1.2.0、SciPy 1.16.1、pytest 9.0.2

### 實測結果

| 步驟 | 實際指令摘要 | 結果 | wall-clock 耗時 |
|---|---|---|---:|
| raw 驗證 | `Get-FileHash -Algorithm SHA256 ...` | hash 完全相符 | < 1 秒 |
| 資料準備 | `py -3 data\scripts\newHW\prepare_data_newHW.py ...` | 15,780 source rows、188 bins；summary source hash 相符 | 0.465 秒 |
| 模型訓練 | `py -3 core\train_sac_microgrid_newHW.py --config configs\config_newHW_soc20_80_sim.yaml --name newHW_repro_soc20_80_s42` | 300 episodes；state 6、action 1、`sac_sn`；成功產生 best／final | 192.293 秒 |
| best rollout | `py -3 data\scripts\newHW\rollout_newHW.py --experiment newHW_repro_soc20_80_s42 --model best_sac_model.pth` | 成功，188 steps／47 小時 | 4.986 秒 |
| final rollout | 同上，model 改為 `final_sac_model.pth` | 成功，188 steps／47 小時 | 4.878 秒 |
| newHW tests | `py -3 -m pytest tests\test_newHW.py -q` | 7 passed | 1.355 秒 |
| P302 regression | `py -3 -m pytest tests\test_deployment.py tests\test_io_protocol.py tests\test_microgrid_env.py -q` | 175 passed、44 warnings | 5.098 秒 |

### 與 lifecycle 站 3 的數值對照

| checkpoint | 指標 | lifecycle 記錄 | 本次實測 | 判定 |
|---|---|---:|---:|---|
| best | served energy fraction | 52.9131% | 52.9131% | 一致 |
| best | SafetyNet projection fraction | 55.85%（105/188） | 55.85%（105/188） | 一致 |
| best | realized violations | 0 | 0 | 一致 |
| final | served energy fraction | 52.8675% | 52.3663% | 不一致，少 0.5012 個百分點 |
| final | SafetyNet projection fraction | 25.53%（48/188） | 61.17%（115/188） | 不一致 |
| final | realized violations | 0 | 0 | 一致 |

本次 best 仍精確達到目前假設下的 finite-window chronological oracle
（52.9131%）；final 為 oracle 的 98.97%。但本次 final 並未重現 lifecycle
所記錄的「接近 best 且 SafetyNet 較少」結果。訓練使用 CUDA；目前流程沒有承諾
跨硬體或重跑時逐 bit deterministic。**不得調參或重跑挑結果來湊舊數字**；
每次執行都必須保留 best／final 並如實記錄。

### 乾淨機器最可能卡住的地方

1. `data/newHW/raw/`、processed CSV、checkpoint 與完整 experiment 都被
   `.gitignore` 排除；clone 後必須先私下取得 raw CSV，其他產物則依本頁重建。
2. 原先套件清單漏列 rollout／上界分析所需的 `scipy`，以及步驟 6 所需的
   `pytest`；本頁已補上。
3. `py` 是 Windows launcher；Linux／macOS 通常只有 `python3`，PowerShell
   的反引號續行與反斜線路徑也不能原樣複製。
4. 同名 experiment 目錄可能混入前一次結果；重跑時應使用新的 `newHW_*`
   名稱，並同步替換兩個 rollout 指令的 `--experiment`。
5. Windows console 若未使用 UTF-8，訓練輸出的符號可能顯示為亂碼，但 JSON、
   CSV 與 checkpoint 不受影響。
6. GPU、PyTorch／CUDA 版本不同時，final checkpoint 指標可能不與既有紀錄
   完全相同；判讀重點是完整記錄，不是調整 seed 或參數追數字。

### 本次落差結論

- 原始資料 hash、188 bins、best rollout、7 個 newHW tests 與 175 個指定
  P302 regression 均符合文件／lifecycle。
- 訓練產物比原驗收清單多出定期 checkpoint 與 NPZ metrics，屬正常產物。
- final rollout 對 SafetyNet 的依賴與 lifecycle 舊 run 有明顯差異；這是
  必須保留的重現落差，不構成修改物理參數或挑選結果的理由。

---

## 10. 跨環境重現的已知落差

同一 seed 在不同作業系統、PyTorch 版本或 CPU／GPU 上不保證位元級重現。三次已知執行的對照：

| 指標 | 原始紀錄 | Windows／CUDA | Linux／CPU |
|---|---:|---:|---:|
| best served energy fraction | 52.9131% | 52.9131% | 52.9131% |
| best SafetyNet projection | 55.85% | 55.85% | 55.85% |
| final served energy fraction | 52.8675% | 52.3663% | 52.8675% |
| final SafetyNet projection | 25.53% | 61.17% | 55.32% |
| realized violations（全部） | 0 | 0 | 0 |

**判讀方式**

- served energy fraction 貼近 finite-window oracle 天花板，三次皆穩定；但正因為飽和，它**對上既有紀錄不構成重現證據**。
- SafetyNet projection fraction 三次三個數字，跨環境明顯不穩定，**不應單獨作為選模依據**。
- 出現第四個不同的 final 數字屬預期行為。如實記錄，不要調參或重跑挑結果。

---

## 11. 站④打包的現況

本節僅供說明，**不是可執行步驟**。

**P302 打包鏈**

`packaging/build_release.spec` 與 `_deploy.ps1` 的 source 完整，所引用的 control、core、gui、soh_predictor、config、load_pattern 路徑全部存在。另一次 Linux 代用 checkpoint 的 PyInstaller 機制 smoke 已完成 Analysis、PYZ、EXE 三階段，hidden imports 可解析，關鍵檔案可收入 `_internal/`。這不等同於使用真實 checkpoint 的正式 release 重建或實驗電腦驗收。

已確認的設計行為：spec 的 `excludes` 排除 `gymnasium`，而 `core/microgrid_env.py` 需要它；但部署執行路徑（`control/`、`gui/`）沒有檔案 import `microgrid_env`，該模組僅以資料檔形式隨附，故不會在目前執行路徑觸發缺少 `gymnasium`。此為刻意設計，非已知缺陷。

**仍缺的 artifacts**（皆被 `.gitignore` 排除，需另行取得）

- `experiments/v22_flow_power_limited_gpu300/models/best_sac_model.pth`
- `experiments/v22_flow_power_limited_gpu300/configs/experiment_config.yaml`
- `soh_models/` 或 `core/soh_predictor/model/` 之下的 SoH artifacts

實驗名稱 `v22_flow_power_limited_gpu300` 寫死於 spec 與 `_deploy.ps1`；使用其他實驗需同步修改兩處。

**newHW 沒有打包路徑**

`packaging/` 與 `_deploy.ps1` 不含任何 newHW 內容，`control/io_protocol_newHW.py` 仍拋 `NotImplementedError`。newHW 產出的 checkpoint **無法**餵入 P302 打包鏈。站④對 newHW 而言不是「尚未完成」，而是「在取得硬體端 I/O 規格前無法開始」。

**尚未驗證**

從目前 source 以真實 checkpoint 重建 GUI release，並在實驗電腦完成硬體驗收——此事尚未有人執行。不得將「spec 機制已驗證」描述為「release 已可交付」。正式狀態仍以 [`release_manifest.md`](release_manifest.md) 為準。

---

## 12. 不在本流程範圍內

- 站④部署與 GUI 打包：`control/io_protocol_newHW.py` 只是會拋 `NotImplementedError` 的骨架，等硬體端／廠商提供 I/O 規格。
- 站⑤現場繪圖：現場 CSV 格式因站④未開始而尚不存在。
- 3 日／5 日跨日驗證：等長期連續資料蒐集。
- 目標函數定案與通過標準：等計畫主持人決定。
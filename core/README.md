# 核心訓練入口

本頁是人員操作簡版；AI／維護者請讀 [`README_AI.md`](README_AI.md)。

## 這裡做什麼

`core/` 提供 P302 SAC 訓練、微電網環境、代理、SafetyNet 與實驗輸出管理。唯一主訓練入口是 `core/train_sac_microgrid.py`；不要另建或改用第二套入口。

## 訓練指令

下面這一個指令才會訓練模型、更新模型權重，並建立一個新的 `experiments/<實驗名稱>/`：

```powershell
# 訓練：執行一次，就是建立一次新的訓練實驗
py core\train_sac_microgrid.py `
  --config configs\experiments\p302\config_p302_v22_flow_power_limited.yaml `
  --name <新且唯一的-v22-實驗名稱>
```

需要延續相容 checkpoint 時可加：

```powershell
  --actor-warmstart experiments\<來源實驗>\models\best_sac_model.pth `
  --warmstart-mode actor_critics
```

Warm-start 只代表初始化，不代表新資料、新 config 或跨日行為已驗證。

## 輸入

- 明示的 YAML；現行 v22：`configs/experiments/p302/config_p302_v22_flow_power_limited.yaml`。
- YAML `env.dataset_csv_path` 指向的 `data/processed/*.csv`。
- 可選的相容 checkpoint 與 `actor_only`、`actor_critics` 或 `full_agent` warm-start 模式。

## YAML 設定檔裡有什麼？

可以把 YAML 理解成「這次實驗的完整配方」。程式碼定義怎麼訓練，YAML 決定這次實際使用哪些數值。常見區塊如下：

| 區塊 | 簡單說明 |
|---|---|
| `random_seed`、`device` | 隨機種子，以及使用 CPU／GPU／自動選擇 |
| `env` | 資料集路徑、時間步、episode 長度、電池容量與功率、SoC 範圍、電網／PV／flow 等物理假設 |
| `sac` | actor／critic 學習率、網路大小、batch、replay buffer、折扣率等 SAC 超參數 |
| `training` | 總 episodes、每回合步數、多久 evaluation、多久存 checkpoint、是否啟用 SafetyNet |
| `safetynet`、`conformal` | 動作投影、安全邊界與 conformal residual window |
| `reward` 或 `reward_newHW` | reward 各項權重；這些數值代表研究目標，不是單純程式設定 |
| `logging` | log、模型、metrics、圖與逐 episode CSV 是否輸出 |
| `guided_teacher`、`stress` | 特殊訓練或壓力測試開關；一般重現不要自行開啟 |

最重要的原則：

- `configs/.../*.yaml` 是訓練前選用的配方；`experiments/<name>/configs/experiment_config.yaml` 是該次訓練封存的實際配方。
- 重跑舊實驗時先看封存 config，不要只靠檔名猜參數。
- 電池容量、功率、SoC、reward 權重與硬體限制都是研究／物理決策，不應為了讓結果好看而臨時修改。

## `core/` 主要檔案分工

| 檔案 | 內容與用途 |
|---|---|
| `train_sac_microgrid.py` | P302 主訓練入口；讀 YAML、建立環境與 agent、執行訓練並保存實驗 |
| `microgrid_env.py` | P302 的物理、觀測、動作、reward 與情境碼 |
| `train_sac_microgrid_newHW.py` | newHW 隔離訓練入口；只能搭配 `newHW_*` 實驗與 newHW config |
| `microgrid_env_newHW.py` | newHW 的 1D battery action、LFP／PV／load 能量模型與暫定 reward |
| `sac_agent.py` | SAC actor／critic、更新流程與 checkpoint 儲存／載入 |
| `safety_net.py` | 在動作送進環境前做安全投影，並記錄 attempted／realized violations |
| `experiment_manager.py` | 建立 `experiments/<name>/`，封存 config 並管理 log、models、results |
| `compute_resources.py` | 記錄 CPU、GPU、記憶體、模型參數量與訓練耗時 |

P302 與 newHW 的環境和訓練入口刻意分開；不要把 newHW checkpoint、config 或物理參數放進 P302 流程。

## 輸出

`experiments/<name>/` 下會依設定產生：

- `configs/experiment_config.yaml`
- `logs/episode_log.csv`
- `models/best_sac_model.pth`、`final_sac_model.pth` 與週期 checkpoint
- `results/training_results.json`、metrics 與訓練圖

`best` 是內建 evaluation reward 創新高的 checkpoint；`final` 是訓練終點。兩者都不是自動通過物理、安全或跨日門檻。

## 下一步

比較同條件下的 best/final，依序跑單日、連續 3 日與連續 5 日 rollout，且跨日不可每日重設 SoC。完整命令與判讀見 [`../docs/handover/training.md`](../docs/handover/training.md)、[`../docs/handover/experiments.md`](../docs/handover/experiments.md)。

## 三個禁止事項

1. 禁止建立第二個主訓練入口，或把 baseline、繪圖、部署腳本當成正式訓練入口。
2. 禁止把 PV/grid 畫成二選一，或把 battery 當成並聯 partial assist；PV/grid 可 mixed supply，放電必須符合 solo-only。
3. 禁止只憑單日、`best`／`final` 檔名或 warm-start 成功宣稱模型 thesis-ready；必須保留多日驗證證據。

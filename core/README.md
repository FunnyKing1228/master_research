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

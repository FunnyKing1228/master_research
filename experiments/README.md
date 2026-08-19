# 實驗資料夾

`experiments/` 用來保存每次模型訓練與後續驗證產生的本機資料。

## 什麼是一個實驗？

執行一次模型訓練，並指定一個新的實驗名稱，就會建立：

```text
experiments/<實驗名稱>/
```

這個資料夾代表**一次訓練實驗**，裡面通常包含：

```text
configs/   該次訓練實際使用的設定
logs/      訓練過程紀錄
models/    best、final 與其他 checkpoint
results/   訓練圖、摘要與後續驗證結果
```

訓練方式見[模型訓練](../core/README.md)。一般使用者只需要知道：**訓練一次就是一個實驗，不是四個實驗。**

## 實驗裡每個檔案看什麼？

| 路徑或檔名 | 主要內容 |
|---|---|
| `configs/experiment_config.yaml` | 該次訓練實際使用的完整 YAML；重現時最先核對 |
| `logs/episode_log.csv` | 每個 episode 的 reward、SoC、violations、SafetyNet 與動作統計 |
| `models/best_sac_model.pth` | evaluation objective 創新高時保存的 checkpoint |
| `models/final_sac_model.pth` | 最後一個 episode 結束時的 checkpoint |
| `models/sac_checkpoint_ep*.pth` | 依 `save_every` 定期保存的中間 checkpoint |
| `results/training_results.json` | 訓練摘要、最佳分數、耗時與運算資源等機器可讀結果 |
| `results/training_results*.png` | objective、violations、SafetyNet、SoC 等訓練趨勢圖 |
| `results/VALIDATION_STATUS*.md` | 這次結果可聲稱到什麼程度，以及尚缺哪些驗證 |

`.pth` 是模型參數與 optimizer 狀態，不是可以直接閱讀的報告；要看行為，必須用同一實驗的 config 做 rollout。

## 後面的單日、3 日、5 日是什麼？

它們是對同一個實驗內的模型做驗證，不會重新訓練模型：

- 單日：快速檢查模型行為。
- 連續 3 日：檢查跨日 SoC 與控制是否開始退化。
- 連續 5 日：確認較長時間的連續穩定性。

只有當模型準備成為候選、論文證據或部署候選時，才需要做完整驗證。驗證結果仍放回同一個 `experiments/<實驗名稱>/results/`。

## newHW 的實驗有什麼不同？

newHW 實驗名稱必須以 `newHW_` 開頭，使用獨立的 newHW 訓練入口與環境。rollout 後通常會多出：

```text
results/in_sample_rollout_newHW/
  best_sac_model_rollout_newHW.png
  best_sac_model_summary_newHW.json
  best_sac_model_audit_newHW.csv
  final_sac_model_*
```

- PNG：四面板行為圖。
- summary JSON：served energy、unmet load、SafetyNet、violations、SoC 與能量上界摘要。
- audit CSV：每個 15 分鐘 step 的 observation、raw／safe／applied action 與供電結果。

newHW 目前只有約 47 小時資料，這些輸出是**同一資料上的 in-sample rollout**，不能改稱 3 日／5 日驗證、泛化驗證或部署通過。實際重現指令見
[`../docs/handover/newHW_reproduce.md`](../docs/handover/newHW_reproduce.md)。

## 注意

- `best_sac_model.pth` 與 `final_sac_model.pth` 都只是該次訓練產生的模型，不能只看檔名決定哪個較好。
- 單日結果正常不代表跨日穩定。
- `experiments/` 內的 checkpoint、log、CSV 與圖片只保存在本機，不提交到公開 repository。

驗證指令、判讀標準與淘汰紀錄交由維護者或 AI 處理；技術細節見 [`README_AI.md`](README_AI.md) 與[實驗交接](../docs/handover/experiments.md)。

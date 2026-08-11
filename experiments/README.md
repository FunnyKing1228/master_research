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

## 後面的單日、3 日、5 日是什麼？

它們是對同一個實驗內的模型做驗證，不會重新訓練模型：

- 單日：快速檢查模型行為。
- 連續 3 日：檢查跨日 SoC 與控制是否開始退化。
- 連續 5 日：確認較長時間的連續穩定性。

只有當模型準備成為候選、論文證據或部署候選時，才需要做完整驗證。驗證結果仍放回同一個 `experiments/<實驗名稱>/results/`。

## 注意

- `best_sac_model.pth` 與 `final_sac_model.pth` 都只是該次訓練產生的模型，不能只看檔名決定哪個較好。
- 單日結果正常不代表跨日穩定。
- `experiments/` 內的 checkpoint、log、CSV 與圖片只保存在本機，不提交到公開 repository。

驗證指令、判讀標準與淘汰紀錄交由維護者或 AI 處理；技術細節見 [`README_AI.md`](README_AI.md) 與[實驗交接](../docs/handover/experiments.md)。

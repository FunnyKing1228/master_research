# 研討會模型差距表

資料來源：

- 主比較：`experiments/seminar_baseline_results/main/baseline_summary.csv`
- Ablation：`experiments/seminar_baseline_results/ablation/baseline_summary.csv`
- 訓練長度：主要 SAC 系列皆為 `120 episodes`
- 評估資料：`data/processed/training_v20_0511_strict_clean_full_days_raw_only.csv`
- episode 長度：`96 steps = 1 day`

## 主比較表

| 方法 | Raw 嘗試違規 / ep | 實際 SoC 違規 / ep | SafetyNet 有意義介入 / ep | 平均修正幅度 W | 最大修正幅度 W | Net profit TWD / ep | 相對 SAC raw 的安全改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Rule-based heuristic | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.2134 | 非學習式工程基準 |
| SAC | 48.90 | 8.00 | 0.00 | 0.00 | 0.00 | -0.1844 | 基準 |
| SAC + reward safety penalty | 48.40 | 7.65 | 0.00 | 0.00 | 0.00 | -0.1856 | 嘗試違規僅下降 `1.0%` |
| SAC + SafetyNet projection | 28.60 | 0.00 | 48.05 | 5.29 | 9.05 | -0.1950 | 嘗試違規下降 `41.5%`，實際違規歸零 |
| OURS | 28.40 | 0.00 | 48.55 | 5.29 | 9.26 | -0.1887 | 嘗試違規下降 `41.9%`，實際違規歸零 |

## Ablation 表

| 方法 | Raw 嘗試違規 / ep | 實際 SoC 違規 / ep | SafetyNet 有意義介入 / ep | 平均修正幅度 W | 最大修正幅度 W | Net profit TWD / ep | 解讀 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SAC | 48.90 | 8.05 | 0.00 | 0.00 | 0.00 | -0.1842 | 純 RL 會頻繁提出危險 action |
| SAC + reward safety penalty | 48.40 | 7.65 | 0.00 | 0.00 | 0.00 | -0.1856 | 單靠 reward penalty 幾乎沒有 internalize safety |
| SAC + SafetyNet projection | 29.95 | 0.00 | 49.20 | 5.33 | 9.14 | -0.1704 | Shield 可保安全，但仍高度依賴修正 |
| SAC + SafetyNet + OCC | 28.60 | 0.00 | 47.90 | 5.30 | 8.88 | -0.1878 | OCC 讓 raw 嘗試違規與介入次數略降 |
| OURS | 28.50 | 0.05 | 47.35 | 5.35 | 9.49 | -0.1896 | 完整方法維持低 raw violation 與低修正依賴 |

## 投影片建議講法

這組表不要只說「最後安全」，而是說：

> 在同一份真實 P302 clean raw-data、同一個 SoC 0.2-0.8 邊界與同一個 1-day episode 設定下，純 SAC 每天平均約 `49` 次嘗試越界，且實際 SoC 違規約 `8` 次。Reward penalty 幾乎沒有改善 raw policy。加入 SafetyNet 後實際違規歸零，而我們的方法同時把 raw 嘗試違規降到約 `28` 次，代表 policy 本身比較少提出危險動作，不只是靠最後 hard guard 修正。

## 圖片位置

如果要直接拿圖，主比較圖在：

- `experiments/seminar_baseline_results/main/raw_policy_safety.png`
- `experiments/seminar_baseline_results/main/deployment_profit.png`
- `experiments/seminar_baseline_results/main/deployment_energy_use.png`

Ablation 圖在：

- `experiments/seminar_baseline_results/ablation/raw_policy_safety.png`
- `experiments/seminar_baseline_results/ablation/deployment_profit.png`
- `experiments/seminar_baseline_results/ablation/deployment_energy_use.png`

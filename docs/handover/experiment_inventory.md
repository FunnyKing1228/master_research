# 代表實驗清冊

本清冊刻意只列代表性、候選、baseline 與淘汰項目，不是 `experiments/` 全目錄抄錄。可攜設定皆使用 repo 相對連結；「本機存在」只表示目前工作樹的 `experiments/` 實際看得到該資料夾或檔案，不代表 Git clone 後仍存在。

## 代表性／歷史參考

### v16：`sac_v16_profit`

- 類別：代表性歷史實驗。
- 可攜設定：[`config_p302_v16.yaml`](../../configs/experiments/p302/config_p302_v16.yaml)。
- 本機存在：`experiments/sac_v16_profit/`。
- 用途：追溯 2D power/flow、3 日 episode 與 v16 尺度設定。
- 限制：使用舊式 PV boolean 與早期 flow 假設，不是目前 thesis/deployment 候選。

### v16sp：`v16sp_no_teacher_v14_0511_clean_v20_solo_intent`

- 類別：代表性且仍可驗證的候選。
- 可攜設定：[`config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml`](../../configs/experiments/p302/config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml)。
- 本機存在：`configs/experiment_config.yaml`、`logs/episode_log.csv`、`models/`、`results/`，並看得到多日期 selected-day 驗證產物。
- 物理語意：`continuous_operation_mode: true`、`deployment_guard_style: true`、`discharge_mode: solo_only`、SoC 0.20–0.80、`pv_support_ratio_obs: true`。
- 狀態：候選，不是最終定案。必須重新確認連續 3 日與 5 日，且 PV hard-blocking 不可由抖動的瞬時 boolean 直接充當完整物理真值。

### v22 deployment package：`v22_flow_power_limited_gpu300`

- 類別：目前 Windows release 綁定的模型來源。
- 可攜設定：[`config_p302_v22_flow_power_limited.yaml`](../../configs/experiments/p302/config_p302_v22_flow_power_limited.yaml)。
- 本機存在：封存設定、episode log、`best_sac_model.pth`、`final_sac_model.pth`、週期 checkpoint 與訓練結果。
- 目前打包選擇：`best_sac_model.pth`；模型與封存 config 的 hash 見 [`release_manifest.md`](release_manifest.md)。
- 狀態：目前 release 候選，不等於 thesis-ready。打包使用 v22 best 的事實，不能取代單日、連續 3 日與 5 日驗證。

### v22 flow/CORAL comparison：`v22_flow_coral_long1000_s42`

- 類別：代表性 flow/CORAL 研究候選。
- 可攜設定：[`main_flow_v22_long1000_s42_ours_full.yaml`](../../configs/baselines/research/main_flow_v22_long1000_s42_ours_full.yaml)。
- 本機存在：封存設定、episode log、`best_sac_model.pth`、`final_sac_model.pth`、週期 checkpoint 與 `training_results.json`/NPZ。
- 物理語意：沿用 `solo_only`，加入 flow action、flow available-power limit 與 PV-surplus charge limit。
- 狀態：可做模擬方法比較；不是「P302 真機 flow-profit 最佳化已完成」的證據。

## Baseline

### Flow v22 Long1000 baseline suite

- 類別：baseline 比較集合。
- 可攜設定：[`configs/baselines/research/main_flow_v22_long1000_s42_*.yaml`](../../configs/baselines/research/)；實際檔案包含 `ours_full`、`sac_raw`、`sac_penalty`、`sac_sn_occ`、`sac_train_safetynet`。
- 本機彙整：`experiments/seminar_baseline_results_flow_v22_long1000/`，確認有 SAC/CORAL 比較 CSV/Markdown 及 `ppo_baselines_v2/` 的 PPO、PPO + SafetyNet 模型與評估 log。
- 代表方法：Heuristic Rule、Standard SAC、SAC + Penalty、SAC + SafetyNet、SAC + S + OCC、CORAL、Standard PPO、PPO + SafetyNet。
- 使用限制：只能在資料、seed、horizon、物理語意與評估窗一致時比較。現有彙整同時含 last-20 train average 與 PPO eval-last-50，寫論文前應先做公平的共同 rollout，不能直接把異質統計當最終排名。

### Heuristic baseline

- 類別：baseline 定義。
- repo 依據：[`data/scripts/baselines/`](../../data/scripts/baselines/) 內的執行腳本與對應 config。歷史 `greedy_heuristic_baselines.md`、`human_knowledge_heuristic_rule.md` 已不在目前工作樹，不可保留失效連結或假設另一台機器仍有。
- 狀態：保留 safety-first、profit-first、balanced safety-profit 等可解釋比較；不得宣稱為完美 expert。
- 物理限制：PV/grid 可共同支援；battery 放電仍須 `solo_only`，不是第三個 partial-assist source。

## 明確淘汰

### `v16s_aggr1000`

- 類別：淘汰。
- 本機存在：完整訓練 config/log、best/final/checkpoints、單日圖，以及 `results/thesis/` 下的 3 日與 5 日圖。
- 淘汰理由：單日表現看似合理，但 3 日、5 日連續模擬暴露明顯跨日失效／退化。
- 禁止用途：不得作最終論文證據；[`generate_thesis_behavior_figures.py`](../../data/scripts/figures/generate_thesis_behavior_figures.py) 仍把它設為預設實驗，執行時必須覆寫，舊圖不得重用。

### `v16s_crossday3_warm200_v7`

- 類別：淘汰。
- 可攜設定來源：[`config_p302_v16s_crossday3_warm.yaml`](../../configs/experiments/p302/config_p302_v16s_crossday3_warm.yaml)；本機實驗名稱帶 `_v7`。
- 本機存在：封存設定、episode log、best/final 與結果目錄。
- 淘汰理由：相較 `v16s_aggr1000` 改善部分跨日穩定性，但仍未達 final-quality。
- 禁止用途：不得因是 cross-day retraining 就稱為 thesis-ready。

### Partial-assist 全線

- 類別：整族淘汰。
- repo 中仍存在的歷史設定例：[`config_p302_v16s_partialassist_quick.yaml`](../../configs/experiments/p302/config_p302_v16s_partialassist_quick.yaml)、[`config_p302_v16sp_partialassist_antidrift.yaml`](../../configs/experiments/p302/config_p302_v16sp_partialassist_antidrift.yaml)、[`config_p302_v16sp_partialassist_antidrift_v2.yaml`](../../configs/experiments/p302/config_p302_v16sp_partialassist_antidrift_v2.yaml)。
- 本機可見的歷史名稱包括 `v16s_partialassist_quick_warm_from_v16s200`、`v16sp_partialassist_quick_warm_from_v16s200`、`v16sp_pa_antidrift_*` 與 `v16sp_pa_antidrift_v2_from_best500`。
- 淘汰理由：真機允許 PV/grid 互補，但 battery 不能成為第三個並聯部分供能來源；此線與硬體拓撲不符。
- 禁止用途：即使舊結果較平滑，也不得當部署或論文候選。新實驗必須使用 `solo_only`。

## 共同禁令

1. 不得把系統寫成 binary solar/grid source selector。
2. 不得由 `grid == 0`、bus/grid 電壓或瞬時 PV boolean 宣稱獨占供能。
3. 不得允許或美化 battery partial assist。
4. 不得只用單日結果選模型；候選至少需單日、連續 3 日、連續 5 日驗證。
5. 不得假設本清冊未明列的 checkpoint、CSV、圖或外部資料夾存在；使用前先檢查實體檔案。

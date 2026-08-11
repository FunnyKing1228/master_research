# 實驗與驗證交接

## 實驗線定位

以下分類以目前 repo 的 code/config 與本機 `experiments/` 實際內容為準，不以名稱推測不存在的結果。

- **v16**：早期 P302 flow-scaled 線；[`config_p302_v16.yaml`](../../configs/experiments/p302/config_p302_v16.yaml) 為 2D power/flow、3 日 episode、舊式 PV boolean 的歷史設定。可用來理解演進，不是目前物理語意的最終候選。
- **v16s**：關閉 flow、加入 PV support 觀測與較短訓練的過渡線。`v16s_aggr1000` 曾有看似合理的單日圖，但連續 3 日與 5 日暴露跨日失效，已淘汰。`v16s_crossday3_warm200_v7` 雖改善部分跨日穩定性，仍未達最終品質，也已淘汰。
- **v16sp**：deployment-aligned、`solo_only`、連續 operation 與較嚴格 SoC/guard 的主線家族。代表可攜設定是 [`config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml`](../../configs/experiments/p302/config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml)。本機有同名 1000-episode 實驗與 selected-day 結果，故可列候選；仍須以目前 PV-state 規則重新確認連續 3/5 日，不能僅憑單日圖定案。
- **v22**：在 v16sp 物理語意上重新啟用 flow action、PV-surplus charge limit 與 power-limited flow 的研究線。代表設定為 [`main_flow_v22_long1000_s42_ours_full.yaml`](../../configs/baselines/research/main_flow_v22_long1000_s42_ours_full.yaml)，本機有 `v22_flow_coral_long1000_s42` 與對照產物。它適合 flow 模擬與方法比較；P302 真機目前仍建議固定 flow，不可把 flow-profit 最佳化寫成已完成部署成果。
- **baseline**：公平比較族群，不是另一個主訓練入口。SAC/CORAL 變體由 [`data/scripts/baselines/run_seminar_baseline_suite.py`](../../data/scripts/baselines/run_seminar_baseline_suite.py) 建立與呼叫相同訓練入口；PPO 與 heuristic 相關實作亦集中於 [`data/scripts/baselines/`](../../data/scripts/baselines/)。比較時必須固定資料、seed、episode、物理限制與評估窗。

所有名稱含 `partialassist`、`partial_assist` 或 `pa_antidrift` 的實驗線一律列為**物理不對齊、已淘汰**。目前環境即使收到 legacy `partial_assist` 也會回退至 `solo_only`；舊結果不得當部署正確性的證據。

## 標準工作流

1. 從可攜 YAML 複製出新設定，先確認資料檔確實存在。
2. 固定 PV-state / blocking 邏輯；初期不要同時大改 reward、horizon 與繪圖解釋。
3. 用 [`core/train_sac_microgrid.py`](../../core/train_sac_microgrid.py) 建立新且唯一的 `--name`。
4. 檢查 `configs/experiment_config.yaml`、`logs/episode_log.csv`、`results/training_results.json` 是否完整。
5. 同時檢查 `best_sac_model.pth` 與 `final_sac_model.pth`，不要依檔名直接定案。
6. 依序做單日、連續 3 日、連續 5 日；任一跨日失效就退回模型/PV-state 設計，不得只挑好看的日期。
7. 與同條件 baseline 比較 safety、raw attempts、SafetyNet intervention、profit、SoC 與 battery/PV/grid 行為。

## 單日驗證

多日期單日驗證可使用 [`generate_selected_day_validation.py`](../../data/scripts/figures/generate_selected_day_validation.py)：

```powershell
py data\scripts\figures\generate_selected_day_validation.py `
  --experiment <實驗名稱> `
  --model best_sac_model.pth `
  --dates "2026-05-02 00:00:00" "2026-05-03 00:00:00" `
  --output-subdir selected_day_validation_best
```

日期必須確實存在於該實驗設定所指的 dataset。可用 `--dataset-override <CSV>` 明示替代資料，但報告中必須記錄替代檔。輸出包含各日期 PNG/PDF、overlay 與終端摘要；部分既有本機結果另有 `validation_summary.csv`，但腳本目前不保證每次都建立該 CSV，所以不得假設必然存在。

## 連續 3 日與 5 日驗證

通用窗口可使用 [`plot_hybrid_model_window.py`](../../data/scripts/figures/plot_hybrid_model_window.py)，分別執行：

```powershell
py data\scripts\figures\plot_hybrid_model_window.py `
  --experiment <實驗名稱> --model best_sac_model.pth `
  --dataset data\processed\<實際資料檔>.csv `
  --start-date "YYYY-MM-DD 00:00:00" --days 3 `
  --output-subdir crossday_3_best --style single_day_thesis

py data\scripts\figures\plot_hybrid_model_window.py `
  --experiment <實驗名稱> --model best_sac_model.pth `
  --dataset data\processed\<實際資料檔>.csv `
  --start-date "YYYY-MM-DD 00:00:00" --days 5 `
  --output-subdir crossday_5_best --style single_day_thesis
```

該腳本以單次 rollout 跑完整天數，因此不應每日重設 SoC。開始時間與資料檔必須有足夠連續資料；不要直接照抄範例日期。

[`generate_thesis_behavior_figures.py`](../../data/scripts/figures/generate_thesis_behavior_figures.py) 也會一次產生單日、日期/初始 SoC 比較、3 日與 5 日圖，但程式內日期固定於 2026-03-23 至 2026-04-09，且預設實驗仍是已淘汰的 `v16s_aggr1000`。只有在候選 dataset 含那些時間戳時才能明示 `--experiment` 使用；不得直接沿用其預設值或舊圖。

## 驗證判讀

至少檢查：

- SoC 是否全程維持指定範圍，且日與日之間沒有不合理漂移或人為 reset。
- raw action、SafetyNet 後 action 與實際 battery power 是否分開記錄；不能只看最終安全動作。
- `pv_support_ratio`、PV blocking/sufficiency state、load、grid draw、battery power 與 SoC 的關係。
- battery 放電時是否能獨供負載；不得出現 battery 與 PV/grid 並聯 partial assist 的解釋。
- PV 支援增加時可描述為「grid demand 降低」，不可由 `grid == 0` 或二元標籤宣稱「負載完全由太陽能供應」。
- best/final 在相同日期、seed、資料與 safety 設定下的結果。

論文圖說必須單獨就能表達這些限制。單日成功只能當 behavior check；沒有 3 日與 5 日連續驗證，就不是 thesis-ready。

## 本機產物與可攜性

`experiments/<name>/`、checkpoint、訓練 log 與產生的圖片只是目前工作樹下的本機參考，不由 Git 保證存在，也不具可攜性。部分歷史繪圖腳本仍可能嘗試加入 repo 外的舊研究 source 路徑；其他電腦不可假設該路徑存在。核心訓練與驗證應以 repo 內 [`core/safety_net.py`](../../core/safety_net.py) 為準。

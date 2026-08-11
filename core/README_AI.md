# Core AI Map

本頁供 AI／自動化工具定位訓練核心；先讀 [`../AGENTS.md`](../AGENTS.md)，人員操作簡版見 [`README.md`](README.md)。唯一深入 SSOT 入口是 [`../docs/HANDOVER_zh.md`](../docs/HANDOVER_zh.md)，訓練細節見 [`../docs/handover/training.md`](../docs/handover/training.md)，實驗與跨日驗證見 [`../docs/handover/experiments.md`](../docs/handover/experiments.md)，release 候選與 hash 見 [`../docs/handover/release_manifest.md`](../docs/handover/release_manifest.md)。

## 真實 source map

- `core/train_sac_microgrid.py`：P302 SAC 唯一主訓練 CLI；載入 YAML、建立 env/agent/experiment、warm-start、訓練、evaluation、best/final 儲存與結果繪圖。
- `core/microgrid_env.py`：`MicrogridEnvironment` 與 `create_microgrid_env()`；消費 dataset，定義 observation/action、SoC、PV/load/grid、flow、reward 與硬體對齊 guard semantics。
- `core/sac_agent.py`：actor、critics、target critics、replay buffer、entropy 與 checkpoint save/load。
- `core/safety_net.py`：SafetyNet projection、conformal residual buffer 與參數；raw action、projected action、realized behavior 不可混稱。
- `core/experiment_manager.py`：建立 `experiments/<name>/{configs,logs,models,results}`，封存 config 與輸出。
- `core/compute_resources.py`：訓練後計算資源 metadata。
- `core/evidential_head.py`、`core/lagrangian_constraints.py`、`core/sim2real_metrics.py`：研究擴充；不是另一個主訓練入口。
- `core/pretrain_rule_actor.py`、`core/rule_expert.py`：規則／預訓練研究工具；不可取代 `train_sac_microgrid.py`。
- `core/verify_deployment.py`：deployment-alignment 檢查工具，不執行主訓練。

## config／訓練／產物流

```text
configs/experiments/p302/<portable-source>.yaml
  env.dataset_csv_path -> data/processed/<dataset>.csv
  training/sac/reward/safetynet/conformal/logging
        |
        v
core/train_sac_microgrid.py
  -> load_config()
  -> ExperimentManager + saved experiment config
  -> create_environment() -> core/microgrid_env.py
  -> create_agent()       -> core/sac_agent.py
  -> optional warm-start
  -> train/evaluate + safety projection
        |
        v
experiments/<name>/
  configs/experiment_config.yaml
  logs/episode_log.csv
  models/{best_sac_model,final_sac_model,sac_checkpoint_ep<N>}.pth
  results/{training_results.json,sac_training_metrics.npz,training_results.png}
```

現行 v22 可攜 config 是 `configs/experiments/p302/config_p302_v22_flow_power_limited.yaml`。它使用 flow action、power-limited flow、PV-surplus charge limit、`pv_support_ratio_obs` 與 `discharge_mode: solo_only`。2026-08-11 manifest 的現行 release 候選是 `experiments/v22_flow_power_limited_gpu300/models/best_sac_model.pth`；這是有日期的交接快照，不是永遠自動更新的真相。

## Warm-start 與 best/final

- `--actor-warmstart <checkpoint>` 搭配 `--warmstart-mode actor_only|actor_critics|full_agent`。
- `actor_only` 可對最後一維做相容部分載入；1D actor 可初始化 2D power/flow actor。
- `actor_critics` 另載 critic、target critic、OCC head、`log_alpha`（checkpoint 有對應欄位時）；架構不相容會失敗。
- `full_agent` 要求完整 checkpoint 與 state/action/network 結構相容。
- `best_sac_model.pth` 只代表訓練內 evaluation reward 嚴格創新高；`final_sac_model.pth` 只代表最後 episode。候選選擇必須同窗比較並留下理由。

## 不可破壞 invariants

- 唯一主訓練入口維持 `core/train_sac_microgrid.py`；baseline／pretrain／plot／deployment 不得形成平行主線。
- PV 與 grid 是 mixed supply，可同時支援負載；禁止 binary source selector 敘事。
- observation 應保留連續 `pv_support_ratio`；PV availability、support level、stateful/hysteretic blocking state 必須分離。
- 不可從 `grid == 0`、bus/grid 電壓比較或抖動瞬時 PV boolean 推論「全 PV」。
- battery discharge 是 solo-only：放電有效時須能獨立承擔當下負載；不可回復 legacy partial-assist semantics。
- raw policy action、SafetyNet/projected action、environment guard 後的 realized action 必須可區分。
- best/final、warm-start 與單日成功均不等於 thesis-ready；連續 3 日與 5 日 rollout 不得每日 reset SoC。
- 初次修正 PV-state/blocking 時保持 reward 與其他變因盡量固定，避免無法歸因。

## 已知技術債

- `train_sac_microgrid.py` 的 `--config` 預設值依工作目錄且不對應現行主設定；操作命令必須明示 YAML。
- 檔首仍有歷史 `sys.path` 注入；核心目前依靠從 `core/` 執行時的 import 行為，尚未完整 package 化。
- `experiment_manager.py` 可在既有同名目錄上建立／寫入；呼叫者必須提供新且唯一的 `--name`，避免混合產物。
- `experiments/` 與 checkpoint 被 Git 忽略，不隨 clone 提供；封存 config 可能含本機絕對 dataset 路徑。
- 現行 v22 仍保留 `pv_obs_boolean: true`；不得讓該欄位重新成為 observation 與 blocking 共用的瞬時物理真值。
- stable pre-measure／probe 已回寫 deployment source 並通過測試；正式 package 重建與實驗電腦驗收仍待完成，訓練結果不可被誤寫成已完成硬體 release 驗收。
- flow 模擬研究不等於真機 flow-profit 最佳化完成；目前硬體部署仍以固定 flow 原則解讀。

## 驗證命令

從 repo 根目錄執行最小靜態／單元檢查：

```powershell
Test-Path core\train_sac_microgrid.py
Test-Path configs\experiments\p302\config_p302_v22_flow_power_limited.yaml
Test-Path data\processed\training_v20_0511_strict_clean_full_days_raw_only.csv
py core\train_sac_microgrid.py --help
py -m pytest tests\test_microgrid_env.py tests\test_deployment.py tests\test_io_protocol.py
git diff --check
git status --short --untracked-files=all
```

模型行為驗證不可只跑 unit test；應依 [`../docs/handover/experiments.md`](../docs/handover/experiments.md) 對 best 與 final 執行 selected-day、連續 3 日及連續 5 日 rollout。論文敘事與資料欄位限制續見 [`../docs/handover/data_and_plotting.md`](../docs/handover/data_and_plotting.md)。

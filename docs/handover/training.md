# 訓練交接

## 唯一訓練入口

P302 SAC 系列統一由 [`core/train_sac_microgrid.py`](../../core/train_sac_microgrid.py) 啟動。不要另建第二套訓練入口，也不要把 baseline、繪圖或部署腳本誤當成主訓練程式。環境、代理與實驗目錄分別由 [`core/microgrid_env.py`](../../core/microgrid_env.py)、[`core/sac_agent.py`](../../core/sac_agent.py) 與 [`core/experiment_manager.py`](../../core/experiment_manager.py) 提供。

建議固定從 repo 根目錄執行，並明示 YAML。**以下命令會真正訓練模型、更新權重，並建立一個新的 experiment**。現行 v22 release 所用實驗的可攜來源設定是 [`config_p302_v22_flow_power_limited.yaml`](../../configs/experiments/p302/config_p302_v22_flow_power_limited.yaml)：

```powershell
# 訓練命令：執行一次會建立一個 experiments/<name>/
py core\train_sac_microgrid.py `
  --config configs\experiments\p302\config_p302_v22_flow_power_limited.yaml `
  --name <新且唯一的-v22-實驗名稱>
```

另一個仍值得做跨日驗證的 v16sp `solo_only` 範例為：

```powershell
py core\train_sac_microgrid.py `
  --config configs\experiments\p302\config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml `
  --name <新且唯一的實驗名稱>
```

可用 CLI 覆寫：

- `--episodes N`：覆寫 `training.total_episodes`。
- `--variant sac|sac_penalty|sac_sn|sac_sn_evi`：覆寫 `training.variant`。
- `--name NAME`：輸出至 `experiments/NAME/`；若省略，名稱為時間戳。
- `--actor-warmstart CHECKPOINT`：載入 actor-only checkpoint 或完整 SAC checkpoint。
- `--warmstart-mode actor_only|actor_critics|full_agent`：控制載入範圍。

## YAML 與路徑解析

`load_config()` 以 UTF-8 讀 YAML。對相對的 `env.dataset_csv_path`，依序嘗試：

1. 相對於目前工作目錄；
2. 相對於 YAML 所在目錄；
3. 相對於 YAML 上一層。

因此從 repo 根目錄執行、在 YAML 內使用 `data/processed/...` 最清楚。程式只會把第一個「確實存在」的候選寫回設定；三處都不存在時不會替你建立檔案，之後建立環境時才會失敗。實驗啟動後，實際解析結果會封存於 `experiments/<name>/configs/experiment_config.yaml`，這份封存設定可能含本機絕對路徑，不具跨電腦可攜性；可攜來源仍是 [`configs/experiments/p302/`](../../configs/experiments/p302/) 或 [`configs/baselines/research/`](../../configs/baselines/research/) 內的 YAML。

不要依賴 `--config` 的預設值 `../configs/config_microgrid.yaml`：它取決於目前工作目錄，而且 repo 目前的主要設定命名並非該預設。交接命令一律明示 `--config`。

## Warm-start

代表命令：

```powershell
py core\train_sac_microgrid.py `
  --config configs\experiments\p302\config_p302_v16sp_no_teacher_v14_0511_clean_v20_solo_intent.yaml `
  --name <新實驗名稱> `
  --actor-warmstart experiments\<來源實驗>\models\best_sac_model.pth `
  --warmstart-mode actor_critics
```

- `actor_only` 只載 actor，且允許最後一維不同時做部分載入；可用 1D power actor 初始化 2D power/flow actor。
- `actor_critics` 另載入 critic、target critic、OCC head 與 `log_alpha`（checkpoint 內有該欄位時）。過往經驗顯示只載 actor、critics 隨機初始化容易把原策略拉壞，延續同架構時優先評估此模式。
- `full_agent` 呼叫代理的完整 `load()`；要求 checkpoint 與目前 state/action 維度及網路結構相容。
- Warm-start 只是初始化，不代表來源模型已通過新設定、新資料與跨日驗證。

## 每次訓練的產物

[`core/experiment_manager.py`](../../core/experiment_manager.py) 會建立：

```text
experiments/<name>/
├─ configs/
│  └─ experiment_config.yaml
├─ logs/
│  └─ episode_log.csv
├─ models/
│  ├─ best_sac_model.pth
│  ├─ final_sac_model.pth
│  └─ sac_checkpoint_ep<N>.pth
└─ results/
   ├─ training_results.json
   ├─ sac_training_metrics.npz
   └─ training_results.png
```

實際是否有模型、NPZ 與圖取決於 YAML 的 `logging.save_models`、`save_metrics`、`plot_results`。啟用 guided teacher 且允許儲存時，`models/guided_teacher_actor.pth` 也是可能產物。後續驗證腳本還會在 `results/` 下新增自訂子目錄；不要假設每個舊實驗都有所有可選檔案。

repo 目前的 `experiments/` 是此電腦上的本機實驗資料，未由 Git 追蹤；複製 repo 不會自動取得 checkpoint、log 或圖。

## best 與 final 的選擇

- `best_sac_model.pth`：訓練中每逢 `eval_every`，以平均 evaluation reward 嚴格創新高時覆寫。它是「該次內建評估 reward 最佳」，不是自動通過物理、安全或跨日門檻。
- `final_sac_model.pth`：最後一個 episode 完成後儲存，代表訓練終點，不代表最佳。
- 一般驗證先以 `best` 為主，並用相同資料窗與指標比較 `final`；若 `final` 較佳，必須保留比較證據與選擇理由，不能只因檔名為 final 就採用。
- 任一候選都必須再做單日、連續 3 日、連續 5 日驗證；跨日過程不可每日重設 SoC。未通過前不得稱為 thesis-ready。

## 不可退讓的物理限制

- 禁止把來源建模為 binary「solar 或 grid」；PV 與 grid 可同時支援負載。
- 觀測應保留連續 `pv_support_ratio`。若要用 PV 放電阻擋狀態，應採有狀態、具遲滯的判定，不可把抖動的瞬時 boolean 同時當物理真值與硬阻擋真值。
- 禁止 battery partial assist。電池不是 PV/grid 之外的第三個並聯支援來源；有效放電必須符合 `solo_only`、能獨自供應當下負載。
- 不可由 `grid == 0`、bus/grid 電壓比較或單一瞬時門檻宣稱「全太陽能供電」。

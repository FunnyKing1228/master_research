# Experiments AI Handover

本頁供後續 AI／自動化代理快速建立實驗上下文；先讀 [`../AGENTS.md`](../AGENTS.md)，人員操作簡版見 [`README.md`](README.md)。`experiments/` 是本機衍生產物區，不是可攜 source；除 `README.md` 與本檔外，內容一律維持 Git ignored。

## Source map

- 唯一會更新模型權重的訓練入口：`core/train_sac_microgrid.py`
- 可攜實驗設定：`configs/experiments/p302/`
- baseline 設定：`configs/baselines/research/`
- 環境與物理語意：`core/microgrid_env.py`
- SafetyNet：`core/safety_net.py`
- 單日驗證（只讀 checkpoint，不訓練）：`data/scripts/figures/generate_selected_day_validation.py`
- 連續 3/5 日 rollout（只讀 checkpoint，不訓練）：`data/scripts/figures/plot_hybrid_model_window.py`
- 實驗規範：[`docs/handover/experiments.md`](../docs/handover/experiments.md)
- inventory／狀態權威：[`docs/handover/experiment_inventory.md`](../docs/handover/experiment_inventory.md)
- 部署 checkpoint 與 hash：[`docs/handover/release_manifest.md`](../docs/handover/release_manifest.md)
- 全域交接入口：[`docs/HANDOVER_zh.md`](../docs/HANDOVER_zh.md)

典型本機產物為 `experiments/<name>/{configs,logs,models,results}/`。使用前必須檢查實體檔案，不能因 inventory 提到名稱就假設另一台機器也有 checkpoint 或結果。

## Invariants

1. 真實平台允許 PV 與 grid 同時支援負載，不是 binary source selector。
2. Battery 不可作為與 PV/grid 並聯的 partial-assist source；放電必須維持 `solo_only`，且能獨立承擔負載。
3. `PV availability`、連續 `pv_support_ratio`、stateful/hysteretic blocking state 必須分離；不得用 `grid == 0`、bus/grid 電壓或抖動瞬時 boolean 推論「完全由 PV 供應」。
4. 先固定 PV-state/blocking，再改 reward、horizon 或圖表敘事，避免同輪混入多個變因。
5. 候選必須依相同 dataset、日期、seed 與 safety 設定比較 `best_sac_model.pth` 和 `final_sac_model.pth`，並完成單日、連續 3 日、連續 5 日 rollout；跨日不得重設 SoC。
6. `best` 不保證優於 `final`：v16sp 歷史候選曾人工選 final；目前 v22 package 明確選 best。
7. 所有 checkpoint、logs、results、CSV、圖與 experiment config snapshot 都是衍生／本機資料，不得因新增 README 而放行追蹤。

## 已淘汰與 inventory

- `v16s_aggr1000`：單日合理但 3/5 日跨日失效，禁止作論文最終證據。
- `v16s_crossday3_warm200_v7`：只有部分改善，仍非 final-quality。
- 名稱含 `partialassist`、`partial_assist`、`pa_antidrift`：硬體拓撲不對齊，全線淘汰。
- v16：只供演進追溯；舊 PV boolean／flow 假設不是最終語意。
- v16sp：可驗證候選，仍須按現行 PV-state 規則重驗 3/5 日。
- v22：目前 deployment package 來源及 flow/CORAL 研究線；package 候選不自動等於 thesis-ready。

任何新增、選用或淘汰決策都更新 [`experiment_inventory.md`](../docs/handover/experiment_inventory.md)，至少寫明 config、dataset、seed、checkpoint 類型、單日／3 日／5 日結果、物理語意、實體檔案狀態與淘汰理由。

## 技術債

- PV sufficiency／hard-blocking 邏輯仍需以連續 support 加 stateful hysteresis 系統化驗證。
- 部分舊繪圖腳本仍預設已淘汰的 `v16s_aggr1000`，呼叫時必須覆寫 experiment，且不得重用舊圖。
- `generate_selected_day_validation.py` 不保證每次產生 `validation_summary.csv`；consumer 不得硬依賴。
- 現有 baseline 彙整混有不同統計窗口；論文排名前需共同 rollout。
- 本機 experiment inventory 不是機器可攜 artifact index；未來可補 manifest/hash，但不可因此提交大型產物。

## 驗證順序

1. 檢查 config、dataset 時間範圍、checkpoint 與 worktree 狀態。
2. 跑 smoke／單日；檢查 SoC、load、grid draw、battery power、`pv_support_ratio`、blocking state，以及 raw／SafetyNet／final action。
3. 對 best/final 跑同一連續 3 日與 5 日窗口；不得逐日 reset。
4. 確認 battery 放電無 partial assist，且論述只使用可量測的「PV support／grid demand」。
5. 將結果與決策寫回 inventory；若用於部署，再同步 release manifest。

詳細 handover：[`experiments.md`](../docs/handover/experiments.md) · [`experiment_inventory.md`](../docs/handover/experiment_inventory.md) · [`release_manifest.md`](../docs/handover/release_manifest.md)。

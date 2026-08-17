# newHW 新增檔案對照

> 本遷移不修改任何既有 P302 檔案。表格在工作進行中持續補充。

| 新檔案 | 複製自／依據 | 修改內容 | 修改原因 |
|---|---|---|---|
| `docs/handover/newHW_migration_log.md` | 任務規格 | 時序記錄決定、假設、執行與結果 | 保留完整遷移證據 |
| `docs/handover/newHW_pending_data.md` | 任務第 2、4 節 | 依阻擋階段整理未知資料 | 禁止用推估值掩蓋缺口 |
| `docs/handover/newHW_file_changes.md` | 任務第 4.3 節 | 記錄所有新增檔案來源與差異 | 讓接手者快速稽核 |
| `data/newHW/raw/Data140826.csv` | 外部 `C:\Users\Administrator\Downloads\Data140826.csv` | 位元不變副本，SHA256 `9ada…881` | 隔離新硬體原始來源，不混入 P302 raw |
| `data/scripts/newHW/prepare_data_newHW.py` | 任務資料規則與 `data/scripts/preprocessing/` 慣例 | 處理 malformed header、PV 反號、電壓遮罩、負載代入、SoC 暫定積分與 15 分鐘重採樣 | 建立可稽核的 newHW 專用資料流程 |
| `data/newHW/processed/training_newHW_15min.csv` | `Data140826.csv` | 188 個 15 分鐘 bins，加入品質與暫定 SoC 欄位 | 供隔離模擬 smoke 使用 |
| `data/newHW/processed/newHW_data_quality_summary.json` | 資料處理 script | 保存來源 hash、斷點、invalid voltage、假設與 SoC 反證 | 機器可讀 provenance |
| `data/scripts/newHW/figures/plot_data_newHW.py` | newHW processed schema | newHW 百瓦尺度資料品質圖 | 不沿用 P302 瓦以下座標與欄位 |
| `data/newHW/processed/data_quality_newHW.png` | `plot_data_newHW.py` | PV/load、暫定 SoC、電壓有效率與 gaps | 視覺化資料缺口 |
| `data/scripts/newHW/analyze_energy_bound_newHW.py` | 能量守恆與同假設 greedy oracle | 計算 PV/load 能量帳、energy-only bound、chronological dispatch bound | 避免把硬體不足誤讀為純模型問題 |
| `data/newHW/processed/energy_upper_bound_newHW.json` | `analyze_energy_bound_newHW.py` | 保存 62.9% PV/load、74.4% energy-only、58.6% finite-window oracle、54.0% terminal-SoC-neutral oracle 與假設 | 機器可讀物理基線 |
| `data/newHW/processed/energy_oracle_trace_newHW.csv` | `analyze_energy_bound_newHW.py` | 每 15 分鐘 oracle SoC、充放電、served/unmet/curtailment | 與 agent 逐步公平比較 |
| `data/newHW/processed/energy_upper_bound_newHW.png` | `analyze_energy_bound_newHW.py` | 47 小時 load、PV 與兩種上界圖 | 報告中呈現硬體能量限制 |
| `configs/config_newHW_sim.yaml` | `configs/config_p302_sim.yaml` 結構參考 | 0.20 kWh、離網、1D action、無 flow、無 TOU、暫定 reliability reward | 新硬體物理系統不同 |
| `core/microgrid_env_newHW.py` | `core/microgrid_env.py` 介面需求，非內容覆寫 | 獨立 PV/battery/load/curtailment/unmet-load environment；grid 固定 0 | 避免 P302 grid/flow/reward 殘留 |
| `core/train_sac_microgrid_newHW.py` | `core/train_sac_microgrid.py` 的共享 agent/training APIs | newHW factory、名稱保護、暫定 metadata 與專用訓練圖 | 不修改既有主訓練入口 |
| `control/io_protocol_newHW.py` | 任務第 3.2/階段 4 | 只有需求清單與 `NotImplementedError` | I/O 未知，禁止臆造 parser/writer |
| `data/scripts/newHW/rollout_newHW.py` | 既有 rollout 的 raw/safe/applied 審計概念 | newHW 47 小時 in-sample audit、reward 定義、物理 oracle、loss-of-load 與百瓦尺度圖 | 同時顯示硬體上限與 agent 距離，不只畫 applied action |
| `tests/test_newHW.py` | newHW environment/protocol contract | 驗證 1D、無 grid、unmet load、PV charge limit、I/O 明確未實作 | 保護隔離語意 |
| `experiments/newHW_lfp_provisional_50ep_s42/` | newHW config、environment、shared SAC components | 50-episode 暫定訓練、best/final/checkpoints、logs、metrics 與 in-sample rollout | 證明流程可執行；不代表模型通過 |

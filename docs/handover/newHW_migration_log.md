# newHW 遷移紀錄

> 建議先讀 [`newHW_lifecycle_mapping.md`](newHW_lifecycle_mapping.md) 了解整體架構。

> 狀態：**交接完成，不代表 newHW 實作完成。** 現況、可重現入口、暫定 smoke、異常與阻擋資料均已留下；模型優化與硬體部署由接手者續辦。此工作只新增 `_newHW` 或 `data/newHW`／`experiments/newHW_*` 內容，不修改既有 P302 檔案。

## 2026-08-17：啟動與邊界確認

### 工作樹保護

- 執行 `git status --short --untracked-files=all`。
- 發現工作樹在本任務開始前已有大量修改、刪除與本機 experiment 產物。
- 決定：全部視為既有內容，不還原、不覆寫、不納入 newHW 修改。
- 既有 release、P302 source、config、tests、README 與 manifest 均保持不動。

### 已閱讀的基線

- `AGENTS.md`、`.cursorrules`
- `data/README_AI.md`
- `core/README_AI.md`
- `experiments/README_AI.md`
- `packaging/README_AI.md`
- `docs/handover/release_manifest.md`
- `configs/config_p302_sim.yaml`
- `core/microgrid_env.py`
- `core/train_sac_microgrid.py`
- `core/safety_net.py`

### 初始技術決定

1. newHW 是離網 LFP 系統，不沿用 P302 的 grid/TOU、flow/pump 或四情境語意。
2. 新增獨立的 `core/microgrid_env_newHW.py`，不就地修改 P302 environment。
3. 新增獨立的 `core/train_sac_microgrid_newHW.py`，讓既有 P302 訓練入口維持不變。
4. newHW 動作空間固定為 1D battery power；正值充電、負值放電。
5. 情境碼暫定只輸出 `1=Battery Solo` 與 `4=Standby`；不產生舊系統的 grid 情境 2/3。
6. reward 只能是明確標註 `TODO(newHW)` 的暫定版本；不得宣稱為最終目標函數。
7. 資料僅約 47 小時，因此訓練與 rollout 都只能是 in-sample smoke，不構成泛化驗證。
8. 新硬體 I/O 未知，因此只新增會拋出 `NotImplementedError` 的 protocol 骨架，不進行 GUI 打包。

### 附件狀態

- 已找到：`C:\Users\Administrator\Downloads\Data140826.csv`
- 未在 `C:\Users\Administrator\Downloads` 找到：`diagnose.py`
- 未在 `C:\Users\Administrator\Downloads` 找到：`data_quality_diagnosis.png`
- 決定：不臆造遺失附件；資料處理依任務中已明示且標記「已驗證」的規則實作，附件缺失列入 pending。

### 已知假設與依據

- `battery_capacity_kwh = 0.20`：依 198.5 Wh 實測可用能量取保守值；仍需 BMS／單體資料確認。
- 固定負載基準 `28.2 W`：依任務提供的 ACS712 迴歸與夜間狀態 B；不是有效 `Load_W` 量測。
- PV 最大功率暫定 `0.129 kW`：依 `MPPT_W_PV` 實測峰值；不是 PV 板或 MPPT 額定值。
- 電池充電功率上限暫定 `0.129 kW`：只為避免模擬吸收超過已觀測 PV 峰值；不是 BMS／MPPT 驗收值。
- 電池放電功率上限暫定 `0.0357 kW`：依已觀測最高穩態負載 A，僅供目前固定負載 smoke；不是電池或 BMS 額定值。
- SoC 操作範圍暫定 `0.10–0.90`：缺少單體電壓與 BMS 門檻，不代表已校正安全範圍。

上述每項假設都必須同時出現在 config 註解與 `newHW_pending_data.md`。

## 2026-08-17：資料準備

### 原始資料隔離

- 建立 `data/newHW/raw/`、`data/newHW/processed/`、`data/scripts/newHW/figures/`。
- 將外部 `Data140826.csv` 複製為 `data/newHW/raw/Data140826.csv`。
- 原始副本 SHA256：`9ada734a86a8aec822589d402b3ee639b62aa5fa3b7ec16025edf4af55e98881`。
- 沒有改寫外部附件或 repository 內副本。

### CSV 格式發現

- 原始 CSV 的整條 header 被包成單一 quoted field；直接 `pandas.read_csv()` 會誤解析為一欄。
- 處理方式：`prepare_data_newHW.py` 明示八個欄位名稱並 `skiprows=1`。
- 發現 26 筆重複 timestamp rows（13 個重複時間點），先按 timestamp 對數值欄取平均；原始檔保持不動。
- 去重後為 15,780 個時間點，範圍 `2026-08-14 16:57:34` 至 `2026-08-16 16:01:57`。

### 修正與重採樣

- `ACS712_PV` 乘以 -1。
- `MPPT_V_batt < 20 V` 或 `> 31 V` 遮罩，共 1,307 rows；暫以時間插值建立積分用電壓，同時保留 valid flag。
- `Load_W` 非零筆數為 0；改用任務提供的固定 28.2 W 推導負載。
- 固定 28.2 W 是刻意的交接簡化，不是宣稱實際負載恆定。實測在 `2026-08-15 14:08` 後轉為約 35.7 W；若把該時段按 35.7 W 計入，整個 47 小時窗口的負載能量約增加 0.19 kWh，固定模型會低估整體需求約一成（約 10–15%）。因此目前 53.97% terminal-SoC-neutral 上界偏樂觀，接手者重建負載狀態後必須重算。
- raw `SoC == 0` 有 8,368 rows，不作訓練真值。
- 以 `ACS712_Batt × 暫定電壓`、0.20 kWh 與 0.95 round-trip efficiency 做暫定能量積分。
- SoC 初值未知，暫以 1.0 作相對能量 anchor，並同時輸出 unclipped 與 clipped 結果。
- 對不規則 timestamp 以實際時間差作梯形積分；這等同假設最長 163 秒缺口間呈線性變化，已標記 `TODO(newHW)`。
- 產生 188 個完整 15 分鐘 bins，範圍 `2026-08-14 17:00` 至 `2026-08-16 15:45`。

### 重要反證

- 以 0.20 kWh 與初始 SoC=1.0 連續積分時，unclipped SoC 最低達 `-1.605`，共有 8,791 raw rows 需要顯示裁切。
- 這表示「0.20 kWh 可用容量 + 單一連續 SoC + 現有電流／電壓資料」彼此不能構成可信的 47 小時 SoC 軌跡；可能涉及 BMS 斷開／重接、容量異常、量測語意或未知充放電狀態。
- 決定：不調整容量或重設 SoC 來美化曲線。重建 SoC 只保留為品質診斷，環境訓練只消費 PV／固定負載並自行模擬 SoC。

### 產物

- `data/scripts/newHW/prepare_data_newHW.py`
- `data/newHW/processed/training_newHW_15min.csv`
- `data/newHW/processed/newHW_data_quality_summary.json`
- `data/scripts/newHW/figures/plot_data_newHW.py`
- `data/newHW/processed/data_quality_newHW.png`

## 2026-08-17：暫定環境與訓練入口

- 新增 `core/microgrid_env_newHW.py`：獨立 off-grid environment。
- 能量流只有 PV、battery、load、curtailment、unmet load；`grid_kw` 永遠為 0。
- 動作為 1D battery kW；無 flow/pump。
- 情境碼只使用 1（battery active）與 4（non-discharge）。
- 新增 `configs/config_newHW_sim.yaml`，所有容量、功率、SoC 與 reward 假設均有 `TODO(newHW)`。
- reward 暫以 served load、unmet load、low-SoC reserve、throughput、PV curtailment 組成；明確標記為人類尚未定案。
- 新增 `core/train_sac_microgrid_newHW.py`；重用既有 SAC agent／SafetyNet／experiment manager，但不修改 P302 訓練入口。
- 新增 `control/io_protocol_newHW.py`；量測與命令函式一律拋出 `NotImplementedError`，未臆造通訊協定。
- 新增 `tests/test_newHW.py`；初始 5 項隔離測試通過，補入 energy-bound tests 後為 7 項。
- newHW Python compileall 通過，IDE linter 無錯誤。

## 2026-08-17：暫定訓練

### 執行

- 命令：
  `py -3 core\train_sac_microgrid_newHW.py --config configs\config_newHW_sim.yaml --name newHW_lfp_provisional_50ep_s42`
- 50 episodes、每 episode 188 steps，共 9,400 steps。
- seed 42、SAC + SafetyNet curriculum；前 10 episodes 不啟用 SafetyNet。
- 使用目前可用 CUDA；耗時約 31.5 秒。
- 產物位於 `experiments/newHW_lfp_provisional_50ep_s42/`。

### 訓練判讀

- 訓練成功建立 best、final、週期 checkpoint、episode log、metrics 與 newHW 專用圖。
- shared trainer 的終端標題仍顯示歷史字串 `P302 Microgrid Simulation`，且列出 revenue/cost=0；這只是未修改既有 trainer 的輸出文字，不代表 newHW 使用 P302 grid/TOU reward。
- newHW environment 的 `grid_kw` 固定為 0，訓練 episode 的 situation 2/3 計數均為 0。
- evaluation reward 沒有形成可信改善；best checkpoint 在第一次 evaluation 即被保存，後續 evaluation 多次維持約 `-1385.44`。
- 訓練後段仍有大量 SafetyNet projection，故不得把完成訓練描述為策略合格。

## 2026-08-17：in-sample rollout

### 程式與修正

- 新增 `data/scripts/newHW/rollout_newHW.py`，逐步保存 raw、SafetyNet、applied action、served/unmet load、PV curtailment、SoC 與 situation code。
- 第一次執行失敗：experiment manager 封存 YAML 使用頂層 `config:` wrapper，rollout 初版直接讀取造成 `KeyError: 'env'`。
- 修正只發生在新檔 `rollout_newHW.py`：若存在 `config:` wrapper 就解包；P302 檔案未修改。
- 修正後 best/final 兩個 rollout 均完成 188 steps（47 小時）。

### Best checkpoint

- served energy fraction：`42.50%`
- unmet load：`0.7620 kWh`
- loss-of-load step fraction：`74.47%`
- SafetyNet projection：`9/188 = 4.79%`
- environment SoC violations：0
- situation 2/3：0

### Final checkpoint

- served energy fraction：`42.50%`
- unmet load：`0.7620 kWh`
- loss-of-load step fraction：`70.21%`
- SafetyNet projection：`166/188 = 88.30%`
- environment SoC violations：0
- situation 2/3：0

### 結論

- best 與 final 都有明顯 loss-of-load，但必須先和相同假設下的物理上界比較，不能單獨把 42.5% 解讀為「AI 無效」。
- final 幾乎全程依賴 SafetyNet，且其 88.3% 不可和 best 的 42.5% 供電率混成同一 checkpoint 敘事。
- best 的 SafetyNet projection 是 4.8%，final 才是 88.3%。
- checkpoint 僅證明 newHW 隔離訓練／儲存／載入／審計流程可執行；不構成模型驗證、部署候選或 reward 正確性的證據。
- 資料不足 3 日，沒有執行 3 日／5 日 rollout；GUI 打包依規格停在未知 I/O。

## 2026-08-17：能量守恆上界補算與判讀修正

### 為何補算

- 初版只報 agent served/unmet，沒有同時報物理基線，可能把硬體供能不足誤讀為純模型問題。
- 新增 `data/scripts/newHW/analyze_energy_bound_newHW.py`，並把相同計算嵌入 best/final rollout JSON、audit CSV 與 PNG。
- 上界使用 chronological greedy oracle：每一步 PV 先供負載，有剩餘就充電；PV 不足時在電池能量與功率允許下盡量供電。
- 另以線性規劃計算 terminal-SoC-neutral oracle，要求結束 SoC 不低於開始 SoC，避免把窗前已儲存的電池能量當成系統在窗內創造的能量。

### 目前資料算出的能量帳

- 固定負載需求：`1.3254 kWh`
- MPPT PV 能量：`0.8341 kWh`
- PV／load 能量比：`62.93%`
- 直接用 `load - PV` 得到的總能量缺口：`0.4913 kWh`
- 不考慮時序、容量與功率限制，並加入初始 SoC 0.90 可交付能量後的 energy-only bound：`74.40%`
- 按時間順序、0.20 kWh、0.129/0.0357 kW、效率 0.95、SoC 0.10–0.90、初始 SoC 0.90 計算的 oracle bound：`58.65%`
- oracle unmet load：`0.5481 kWh`
- oracle loss-of-load steps：`43.62%`
- 若要求 `end SoC >= start SoC = 0.90`，可持續／循環上界為 `53.97%`
- terminal-SoC-neutral unmet load：`0.6100 kWh`
- terminal-SoC-neutral loss-of-load steps：`57.98%`

因此，在目前 processed dataset 與假設下，`42.5%` **不是物理上限**：

- agent 達到 provisional oracle 的 `72.48%`
- agent 比 oracle 少供 `0.2140 kWh`
- best 的 loss-of-load steps 比 oracle 高 `30.85` percentage points
- 對較符合「電池只能搬移窗內能量」的 terminal-SoC-neutral oracle，agent 達到 `78.75%`，仍少供 `0.1520 kWh`

同時，finite-window oracle 只能供應 58.65%，要求期末 SoC 回復後更只有 53.97%，所以硬體／資料假設確實存在不可由 agent 消除的能量不足。正確敘事是：

> 系統本身無法達成 100% 供電；目前 agent 又低於相同假設下的 provisional oracle。兩個問題同時存在。

使用者提出的「PV 約 55%、缺口約 650 Wh」與目前 processed CSV 的 `62.93%／491 Wh` 不一致，不能擇一宣稱為真。差異可能來自負載尺度、待機／寄生負載、資料窗口或重採樣方式，已列 `TODO(newHW)`。

### 4.8 W 待機是否重複計算

- 沒有重複計算。
- preprocessing 與 environment 在每一個 15 分鐘 step 都只使用固定 `28.2 W`。
- `4.8 W` 沒有額外加到 load，也沒有再從 battery 扣除。
- 但 28.2 W 迴歸截距是否已包含 MPPT、logger、DC-DC 或其他寄生負載仍未知；這會改變物理上界。

### 暫定 reward 的精確定義

每 step：

```text
reward =
  1.00 × served_fraction
  - 12.00 × unmet_fraction
  - 2.00 × low_soc_reserve_deficit
  - 0.05 × battery_throughput_fraction
  - 0.05 × pv_curtailment_fraction
```

- reserve SoC 暫定 0.20。
- 選擇理由：優先讓 smoke 學習供電可靠度，同時保留低 SoC、循環與棄光的小型代價。
- 這不是人類核准的最終目標，也沒有經 sensitivity／baseline 比較。
- reward 定義與全部 weights 已寫入每份 rollout summary JSON。

### Episode 與 in-sample 程度

- `episode_length = 188`，就是整份 47 小時 processed dataset。
- `fixed_start_idx = 0`；每個 episode 都從相同資料第一列開始。
- 每次 reset 都把模擬 SoC 設回暫定 `0.90`。
- 50 episodes 等於同一條 47 小時軌跡重跑 50 次；沒有不同日期、不同起點或 held-out window。

### 測試數量拆分

- newHW 新增測試：`7`
- 既有 P302 regression tests：`175`
- 合計：`182 passed`

七個 newHW tests 涵蓋：1D action、無 grid、夜間 standby 產生 unmet load、充電受 PV surplus 限制、未知 I/O 必須拋出 `NotImplementedError`、energy bound 不額外加入 standby、chronological oracle 不超過 energy-only bound。它們仍不證明 reward 正確、硬體參數正確或可部署。

### 與先前資料診斷的對照

- `28.2 W`：直接採用任務提供的迴歸結果，沒有假裝重新完成迴歸驗證。
- 首夜 `2026-08-14 17:45` 至 `2026-08-15 00:51:12` 依有效 pack voltage 與 ACS712_Batt 梯形積分得到 `199.07 Wh`，與先前 `198.5 Wh` 相差約 0.3%，一致。
- 該窗口最後有效值為 `25.42 V`，資料中 00:49–00:50 多筆為 `25.43 V`，與先前 BMS 跳脫電壓描述一致。
- 0.20 kWh 尚未因此成為通用容量真值；第二夜與完整 47 小時積分仍出現無法由單一 0.20 kWh SoC 軌跡解釋的矛盾。

## 2026-08-17：最終驗證與交付狀態

- 將 `data/newHW/raw/Data140826.csv` 設為 Windows 唯讀；SHA256 再驗仍為 `9ada734a…e98881`。
- 重新在 `%TEMP%` 產生 processed CSV 與 summary JSON，兩者 hash 均與 repository 產物一致。
- 重新在 `%TEMP%` 求解 energy bounds；bound JSON 與 oracle trace CSV hash 均可重現。
- `tests/test_newHW.py` 7 項加既有 P302 environment/deployment/I/O 175 項 regression：`182 passed`，只有既有 Gymnasium Box precision warnings。
- newHW 全部 Python compileall 通過；IDE linter 無錯誤。
- 任務前後 `git status` set 差異只有 16 條 newHW 新檔；沒有新增任何既有檔案的 `M`／`D` 狀態。
- `git diff --check` 仍因任務開始前已修改的 `docs/repository_structure.md` 兩處 trailing whitespace 失敗；依「不得修改既有檔案」規則保留，不修正。
- raw CSV、processed training CSV 與整個 experiment 受既有 `.gitignore` 的 `*.csv`／`experiments/*` 規則排除；檔案存在於本機但不會隨 Git clone 提供。
- best checkpoint SHA256：`e95f3b6ae8e1919b8f99034e9c9018c9ea5c69dd83a829f1b673f681a051ae4a`。
- final checkpoint SHA256：`11f68fc32a76171771f8b6cdf5be90f7e150769656dff4d1d841fdaa67c827d7`。
- 以上兩個 checkpoint 都是未驗證且低於 provisional oracle 的 in-sample smoke artifact；hash 只供辨識，不代表核准。

### 完成標準對照

1. 資料處理、獨立環境與設定已完成；所有非實測假設已標記 `TODO(newHW)`。
2. 訓練可執行並產生暫定模型；結果低於 provisional oracle，且硬體本身也無法達成 100% 供電。
3. rollout 明確標記 in-sample，沒有宣稱通過，也沒有臆造 3 日／5 日結果。
4. 部署停在 I/O 缺失；protocol 骨架只會拋出 `NotImplementedError`，沒有進行 GUI 打包。
5. migration log、pending data、file changes 三份文件已完成。

> 最終狀態：**newHW 作為新硬體／新資料切入點的交接已完成。** 原 `42.50%` 異常已於後續診斷確認為 50 episodes 訓練不足；模型定案、長期資料、泛化驗證與部署 I/O 仍是後續工作。

## 2026-08-17：白名單發布準備

- 為避免碰觸原 repository 的既有 dirty working tree，發布暫存只在獨立 clone `C:\Users\Administrator\Downloads\master_research_publish` 進行。
- 逐項複製與暫存 16 個白名單檔案：newHW core/config/control、4 個 newHW Python scripts、7 項專屬測試、三份交接文件，以及 4 個小型 summary JSON/PNG。
- 明確排除所有 CSV、checkpoint、完整 experiment 與既有檔案修改；4 個 summary 檔最大為 250,579 bytes，皆小於 1 MB。
- 在發布 clone 執行 `tests/test_newHW.py`，結果為 `7 passed`；README 未修改，入口連結等待人類決定並應另開獨立 commit。
- 本紀錄完成時尚未 commit 或 push；依要求停在 staged review，待人類確認 `git status` 與 `git diff --cached --stat`。

## 2026-08-18：固定策略診斷與 300-episode 對照

### 固定策略結果

- 保持 `Data140826.csv` 的 188 bins、初始 SoC=0.90、暫定 reward、SafetyNet 與所有物理參數不變。
- action 恆 0：served energy `31.0366%`、unmet `0.914040 kWh`、期末 SoC `0.90`。
- action 恆最大充電：初始 SoC 已在 0.90 上限，因此 188 steps 均被 SafetyNet 投影為 0，結果與恆 0 相同；另以 SoC=0.10 做充電路徑控制測試，可實際充電 `0.168421 kWh` 並回到 SoC=0.90。
- action 恆最大放電：served energy `42.5049%`、unmet `0.762040 kWh`、期末 SoC `0.10`，精確重現原 final checkpoint。
- 固定規則「PV surplus 時最大充電、缺電時最大放電」：served energy `58.6472%`、unmet `0.548090 kWh`、期末 SoC `0.573946`，等於目前 finite-window chronological oracle。
- 結論：正、負與零 action 均會改變 applied action、SoC 與 served energy；action→SafetyNet→environment 路徑沒有斷裂。原 50-episode checkpoint 未在白天充電，是訓練不足。

### 隔離延長訓練

- 不修改 source config，以 `--episodes 300` 建立新 experiment：`experiments/newHW_lfp_provisional_diag300_s42/`。
- 只改訓練長度 50→300；reward、seed、資料、SafetyNet 與硬體假設全部保持不變，避免混入第二個變因。
- best 在 episode 180 evaluation 達 `-869.85` 後保存；final 並非最佳。
- best in-sample rollout：served energy `56.9260%`、unmet `0.570903 kWh`、finite-window oracle 達成率 `97.07%`、SafetyNet 介入 `4.79%`、期末 SoC `0.573946`。
- best 有 29 個正 applied-action steps，實際充電 `0.311564 kWh`；SoC 會在兩段 PV surplus 期間上升，證明已學到充電行為。
- final in-sample rollout：served energy `52.9769%`、unmet `0.623244 kWh`、SafetyNet 介入 `6.91%`；低於 best，顯示訓練後段仍有退化／不穩定。
- best checkpoint SHA256：`90e17537877ed361741c1a4852072bd18993d048c48e9868661470d9469783d7`。
- final checkpoint SHA256：`f425e4fb29e3594522d2eb7ab58ed32f8e9ff1b7c63f433202e220a9cc4339da`。

### 解讀限制

- 此結果只回答「充電路徑是否有效」及「50 episodes 是否不足」，不改變交接狀態。
- 300 episodes 仍重複使用同一段 47 小時訓練資料，屬高度 in-sample；不構成泛化、3 日／5 日、硬體或部署驗證。
- 56.93% 高於 terminal-SoC-neutral 53.97%，是因 rollout 從 SoC=0.90 結束於 0.573946，消耗了窗口外帶入的初始電量；不可用來宣稱可持續供電率。
- reward 權重、離網架構、BMS／MPPT 限制、SoC 範圍與固定 28.2 W 負載仍為 `TODO(newHW)`，不得將此 checkpoint 升格為 deployment candidate。

## 2026-08-18：rollout 圖可讀性重畫

- 修改 `data/scripts/newHW/rollout_newHW.py` 的圖表呈現，不改 rollout 數據、環境、checkpoint 或指標計算。
- panel 1 改為 PV generation、load demand、power delivered to load 與紅色 unserved gap；紅色只填在「已供應」與「需求」之間，並把 legend 移到 panel 外，避免遮住資料。
- panel 2 保留 SoC，補上明確標題；panel 3 補上「正值=充電、負值=放電」與 0 W 基準線。
- panel 4 移除難以直讀的四條 unmet／curtailment 累積線，改為總負載需求、agent 已供應、未供應及 finite-window oracle 已供應的累積能源帳，並標出期末 kWh。
- 全圖字體相對原版增加 2 pt，畫布同步由 14×13 放大為 16×15 inches；panel 3／4 legend 另行移位，避免放大後遮住曲線或摘要框。
- 重新產生 50-episode 與 300-episode experiment 的 best／final 四張 rollout 圖；數值 summary 未改變。

## 2026-08-18：SoC 20–80% 隔離試驗與結果欄位

- 新增 `configs/config_newHW_soc20_80_sim.yaml`，不覆寫原 10–90% config；只把 SoC operating range 改為 0.20–0.80、initial SoC 改為 0.80，並固定 300 episodes。
- 20–80% 是使用者指定的操作範圍，不是已由 BMS／cell 規格驗證的保護門檻，仍標記 `TODO(newHW)`。
- 新 experiment：`experiments/newHW_lfp_soc20_80_diag300_s42/`；資料、reward、seed、功率限制與 SafetyNet 其餘設定維持不變。
- best in-sample：served `52.9131%`，等於同範圍 finite-window oracle；unmet `0.624090 kWh`、realized violations `0`、attempted violations／SafetyNet projections `105/188`（55.85%）、期末 SoC `0.473946`。
- final in-sample：served `52.8675%`，比 oracle 少 `0.000604 kWh`；unmet `0.624694 kWh`、realized violations `0`、attempted violations／SafetyNet projections `48/188`（25.53%）、期末 SoC `0.477123`。
- best 在 episode 30 保存；其供電略高但高度依賴 SafetyNet。final 幾乎維持相同供電，attempted/projection 較低，因此兩者都需保留，不可只因檔名 `best` 就宣稱行為較佳。
- best SHA256：`92c6ba15228ededf8eec9278e802d8fd544803a4ee01b81b471003255a0016f0`。
- final SHA256：`b11e3a892ae3a55af3338136102d8e46354d95cb7bdc3fae4908f5442e8bedd5`。
- rollout JSON／圖新增 operating SoC bounds、attempted／realized violations 與 SafetyNet 介入；SoC guide 不再硬編碼 10–90%。
- Profit 明列為 `N/A (off-grid; no tariff/revenue model)`，JSON 使用 `profit: null` 與明確 status；不以數值 0 假裝已完成經濟模型。
- training result 重畫為 provisional objective 原始值／20-episode mean／deterministic evaluation、attempted／realized out-of-bounds、SafetyNet intervention rate 與 episode SoC min／mean／end；舊的 action-magnitude panel 移除。
- 圖上明確註記負值是 objective score 而非金錢。20–80% 試驗的 evaluation 約在 episode 20–30 提升至 `-964` 後停滯；attempted violations 的 20-episode mean 從約 137 降至約 25，realized violations 維持 0。
- 此結果仍是相同 47 小時資料的 in-sample trial，不能視為 3 日／5 日、泛化或部署驗證。

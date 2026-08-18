# newHW 生命週期對照：舊硬體到新硬體，每一站改了什麼

> **這份文件是 newHW 交接的第一份閱讀材料。**
> 時序過程請看 [`newHW_migration_log.md`](newHW_migration_log.md)；
> 逐檔異動請看 [`newHW_file_changes.md`](newHW_file_changes.md)；
> 完整缺口清單請看 [`newHW_pending_data.md`](newHW_pending_data.md)。

---

## 第一部分：這個專案的生命週期是什麼

本專案的工作流是一個閉環，從資料進來開始，到現場繪圖結束，共五站。
README 的「依工作選入口」就是這五站。理解這一圈，才能理解換硬體時哪些環節會被影響。

```
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  ▼                                                         │
① 資料準備  →  ② 模型訓練  →  ③ 實驗與跨日驗證  →  ④ 部署與 GUI 打包  →  ⑤ 現場 CSV 與繪圖
  │              │                │                    │                    │
data/         core/           experiments/         packaging/         tools/plotting_handoff/
                                                    control/
```

### ① 資料準備

把原始量測資料整理成訓練可用的格式。
輸入是感測器原始輸出，輸出是符合設定檔 schema 的 processed CSV。
這一站決定後面所有環節的資料品質上限——**這裡缺的欄位，後面補不回來**。

### ② 模型訓練

以 processed CSV 為輸入，透過唯一訓練入口跑 SAC，產出 checkpoint。
環境（`microgrid_env`）定義物理模型與 reward，設定檔（YAML）定義所有參數。
每次訓練產生一個 `experiments/<name>/`，封存 config、log、best／final checkpoint 與訓練結果。

### ③ 實驗與跨日驗證

讀取既有 checkpoint 做 rollout，不訓練。
標準流程是單日 → 連續 3 日 → 連續 5 日，跨日不重設 SoC。
目的是確認策略在時間軸上的行為合理，而不只是 reward 數字好看。
**這一站是模型能否進入部署的門檻。**

### ④ 部署與 GUI 打包

把通過驗證的 checkpoint 包成可在實驗電腦上執行的 GUI release。
關鍵在 `control/io_protocol.py`——它定義與硬體端軟體的通訊格式
（P302 是 `Data.txt` 讀量測、`Command.txt` 寫命令）。
**沒有 I/O 規格就沒有這一站**，因為程式無法與硬體對話。

### ⑤ 現場 CSV 與繪圖

部署跑起來之後，現場會產出 CSV，用可攜式繪圖工具畫成四面板圖，
用於檢查實際運作行為與後續分析。
這一站的輸出可以回饋到 ① 成為新的訓練資料，形成閉環。

### 換硬體時，這一圈會發生什麼

換硬體不是換一個數字，而是換一個物理系統。影響會沿著這一圈往下傳：

- ① 感測器不同、欄位不同、取樣率不同 → 前處理要重寫
- ② 電池化學、容量、功率、有無電網都不同 → 環境與 reward 要重寫
- ③ 資料長度不足就無法跑既有的多日驗證 → 驗證標準要重新定義
- ④ 通訊協定不同 → I/O 層要重寫
- ⑤ 資料尺度不同 → 繪圖座標與單位要重設

**這也是為什麼 newHW 不能沿用 P302 的任何一站，只能整圈重來。**

---

## 第二部分：兩套硬體的差異

| 項目 | P302 | newHW |
|---|---|---|
| 電池 | 鋅空氣液流電池（SLFB） | 磷酸鋰鐵 LFP，8S1P |
| 標稱容量 | 0.0099 kWh | 0.449 kWh（實測可用約 0.20 kWh） |
| 額定功率 | 0.000825 kW | PV 實測峰值 0.129 kW |
| 電網 | 併網（import-only），TOU 電價套利 | **完全離網，沒有電網** |
| 液流動力學 | 有 pump、flow fraction | 無 |
| 動作空間 | 2D：[power_kw, flow_fraction] | 1D：[power_kw] |
| 情境碼 | 1／2／3／4 四種 | 只有 1（Battery Solo）與 4（Standby） |
| 原始取樣 | 15 分鐘 | 5 秒（需重採樣） |
| 資料長度 | 4 天以上 | 約 47 小時 |

**最關鍵的一條**：沒有電網就沒有電價，P302 的 reward 核心
`(Baseline_Cost − Agent_Cost) × tou_reward_scale` 在 newHW 完全失效。
這不是調權重，是要重新定義目標函數。

---

## 第三部分：逐站對照

每一站分三段：**舊硬體怎麼做 → 新硬體改成什麼 → 阻擋在哪**。

> **關於「阻擋」的說明**：
> 以下標記為阻擋的項目，都是**因為缺少外部輸入而無法在程式端解決**的事項。
> 每一項都註明了阻擋來源與應由誰提供。
> 這些不是未完成的工作，而是**在取得對應資訊之前，任何人都無法推進的節點**。
> 在資料補齊前強行填補，只會產生看似完成、實則不可信的結果。

### 站 1：資料準備

| | P302 | newHW |
|---|---|---|
| 入口 | [`data/README.md`](../../data/README.md) | `data/scripts/newHW/prepare_data_newHW.py` |
| 原始資料 | `data/raw/` | `data/newHW/raw/Data140826.csv` |
| 產物 | `data/processed/*.csv` | `data/newHW/processed/training_newHW_15min.csv`（188 bins） |

**改了什麼**

- 新資料**既不是既有的 raw 也不是既有的 deployment**，來源與硬體都不同，
  因此放在獨立的 `data/newHW/`，不混入既有資料目錄。
- 5 秒 → 15 分鐘重採樣，與既有 `time_step: 0.25` 對齊。
- 原始 CSV 的 header 被包成單一 quoted field，需明示欄位名並 `skiprows=1`。
- 逐項修正：`ACS712_PV` 乘以 −1（接線反接）、`MPPT_V_batt` 在 20–31 V 之外遮罩、
  26 筆重複 timestamp 取平均。
- `Load_W` 整欄為 0，改用 ACS712 迴歸推導的固定 28.2 W。

**阻擋項**

| 缺什麼 | 為何無法在程式端解決 | 應由誰提供 |
|---|---|---|
| `Load_W` 感測通道失效 | 負載是能量平衡的必要輸入。目前的 28.2 W 是由 ACS712 迴歸推導，不是量測值，無法驗證 | 硬體端修復感測器 |
| SoC 未校正 | 原始 SoC 有 53% 時間為 0，且提供者已聲明不可用；無法從現有資料重建可信軌跡 | 硬體端提供經驗校正或 BMS coulomb counting |
| `MPPT_V_batt` 夜間固定 9.05 V 假值 | 屬感測或韌體行為，需在硬體端確認成因 | 硬體端 |
| 資料僅 47 小時 | 無法切分 train／validation／test，任何泛化宣稱都不成立 | 需長期連續蒐集 |

---

### 站 2：模型訓練

| | P302 | newHW |
|---|---|---|
| 入口 | [`core/README.md`](../../core/README.md) | `core/train_sac_microgrid_newHW.py` |
| 環境 | `core/microgrid_env.py` | `core/microgrid_env_newHW.py` |
| 設定 | `configs/config_p302_*.yaml` | `configs/config_newHW_sim.yaml` |

**改了什麼**

- 環境與訓練入口**都是新檔案，不是就地修改**，P302 的唯一訓練入口維持不變。
- 能量流只剩 PV、battery、load、curtailment、unmet load；`grid_kw` 恆為 0。
- 移除全部 flow／pump 相關參數與動作維度。
- 情境碼只輸出 1 與 4；訓練中情境 2／3 計數確認為 0。
- reward 改為 served load、unmet load、low-SoC reserve、throughput、PV curtailment 五項。
- 主要參數：容量 0.20 kWh、充電上限 0.129 kW、放電上限 0.0357 kW、SoC 範圍 0.10–0.90。

**阻擋項**

| 缺什麼 | 為何無法在程式端解決 | 應由誰提供 |
|---|---|---|
| 系統是否確為離網未經確認 | `grid_kw` 恆為 0、情境碼僅 1 與 4、reward 無 TOU，全部建立在此推論上；若有市電則環境設計需重做 | 硬體端確認 |
| 目標函數未定案 | 離網系統的優先順序（供電可靠度／夜間存活／BMS 保護／深度循環／curtailment）是研究方向決策，不是實作選擇 | 計畫主持人決定 |
| BMS 型號與電流設定值 | 充放電功率上限目前以 PV 實測峰值與觀測負載暫代，非驗收值 | 硬體端提供規格書 |
| 電芯單體電壓 | 實測可用容量僅標稱 44%，且 BMS 在 3.18 V/cell 即跳脫，成因無法從 pack 電壓判斷 | 硬體端逐顆量測 |
| PV 板與 MPPT 額定 | 目前 0.129 kW 是觀測峰值，不是額定值，無法作為模擬邊界 | 硬體端提供規格 |
| 是否允許可控卸載 | 目前動作只有 battery power，環境無法主動關閉負載；是否開放屬系統設計決策 | 計畫主持人決定 |

---

### 站 3：實驗與跨日驗證

| | P302 | newHW |
|---|---|---|
| 入口 | [`experiments/README.md`](../../experiments/README.md) | `data/scripts/newHW/rollout_newHW.py` |
| 標準流程 | 單日 → 連續 3 日 → 連續 5 日，跨日不重設 SoC | **無法執行** |
| 產物 | `experiments/<name>/` | 歷史：`newHW_lfp_provisional_50ep_s42/`；最新診斷：`newHW_lfp_soc20_80_diag300_s42/` |

**改了什麼**

- 資料僅 47 小時，**不足 3 日**，既有的 3 日／5 日驗證無法進行。
- 改以訓練資料本身做 in-sample rollout，並在所有輸出明確標註
  「此為 in-sample，不構成泛化驗證」。
- 未臆造 3 日／5 日結果。

**歷史 50-episode 結果**

- best 與 final 的 served energy fraction 皆為 42.50%、unmet load 皆為 0.7620 kWh。
- 可持續上界為 53.97%，仍有約 152 Wh 空間。
- 兩者 SafetyNet 介入率相差 18 倍（4.8% vs 88.3%），卻產出相同結果。
- evaluation reward 全程停留約 −1385.44，best 在第一次評估即保存後未再被超越。
- 固定策略對照已確認 action→SafetyNet→environment 路徑有效；此結果是訓練不足的歷史失敗基線，不再代表最新模型狀態。

**最新 300-episode／SoC 20–80% 診斷**

- best served energy fraction 為 52.9131%，等於同範圍 finite-window oracle；SoC 實際使用完整 20–80% 操作範圍。
- best realized violations 為 0，但 attempted violations／SafetyNet projections 為 105/188（55.85%），表示 raw policy 仍高度依賴安全層。
- final served energy fraction 為 52.8675%，只比 oracle 少 0.000604 kWh；realized violations 為 0，attempted／projected 為 48/188（25.53%）。
- final 幾乎維持相同供電且較少依賴 SafetyNet；因此 `best` 與 `final` 都需保留比較，不能只依 checkpoint 名稱決定。
- terminal-SoC-neutral 上界為 48.2390%；best 從 SoC 80% 結束於約 47.4%，因此 52.9131% 包含窗口開始時帶入的電量，不是可持續供電率。
- Economic profit 為 N/A：目前推論為離網，沒有 tariff／revenue model；圖中的負值是 provisional objective score，不是金錢。
- 以上仍只是在同一份 47 小時訓練資料上的 in-sample 診斷，不構成泛化、3 日／5 日或部署驗證。

**阻擋項**

| 缺什麼 | 為何無法在程式端解決 | 應由誰提供 |
|---|---|---|
| 至少 5 個完整連續日 | 既有驗證流程的硬性前提；47 小時無法執行 | 需長期連續蒐集 |
| 未參與訓練的獨立日期 | 沒有 held-out 資料就無法判斷泛化 | 需長期連續蒐集 |
| newHW 的通過標準 | unmet load、BMS event、SoC、curtailment、SafetyNet 介入的門檻值需人為訂定 | 計畫主持人決定 |

**已完成的技術診斷**

> 固定策略與 300-episode 對照已排除動作路徑斷裂。
> 最新模型已能在目前假設下接近 finite-window oracle；
> 剩餘問題是 SafetyNet 依賴、目標函數未定案、資料不足與缺少跨日泛化驗證。

---

### 站 4：部署與 GUI 打包

| | P302 | newHW |
|---|---|---|
| 入口 | [`packaging/README.md`](../../packaging/README.md) | **未進行** |
| I/O protocol | `control/io_protocol.py`（Data.txt／Command.txt） | `control/io_protocol_newHW.py`（`NotImplementedError`） |
| 打包 | PyInstaller → GUI release | 未執行 |

**改了什麼**

- 新硬體的通訊介面**完全未知**：是否存在、什麼格式、由誰提供，全部沒有資訊。
- 因此只建立會拋出 `NotImplementedError` 的骨架，
  **未建立 parser、未臆造協定、未進行 GUI 打包**。

**阻擋項**

這一站的阻擋最徹底：**在取得 I/O 規格前，這一站無法開始，不是無法完成。**
臆造協定格式會產生看似可用、實際上無法與硬體通訊的程式碼，風險高於留白。

| 缺什麼 | 為何無法在程式端解決 | 應由誰提供 |
|---|---|---|
| I/O 介面是否存在、協定格式、傳輸方式 | 通訊格式由硬體端定義，無法推測 | 硬體端／廠商 |
| 量測與命令欄位、單位、更新頻率 | 同上 | 硬體端／廠商 |
| ack／timeout／fail-safe 行為 | 涉及安全，不可假設 | 硬體端／廠商 |
| controller／BMS／MPPT 的控制權限 | 需確認哪一層可被程式控制 | 硬體端／廠商 |
| 緊急停止、失聯、低壓、過壓、過流的責任層級 | 安全設計決策 | 硬體端與計畫主持人共同決定 |
| 硬體驗收程序、場地、負責人、簽核方式 | 流程決策 | 計畫主持人 |

---

### 站 5：現場 CSV 與繪圖

| | P302 | newHW |
|---|---|---|
| 入口 | [`tools/plotting_handoff/README.md`](../../tools/plotting_handoff/README.md) | `data/scripts/newHW/figures/plot_data_newHW.py` |
| 產物 | 四面板現場圖 | `data_quality_newHW.png`、`energy_upper_bound_newHW.png` |

**改了什麼**

- 既有繪圖綁定 P302 的欄位名與功率尺度（瓦以下等級）；
  newHW 是百瓦等級，座標軸範圍、單位與標註全部重設。
- 新增能量上界圖，用於說明供需限制。

**阻擋項**

| 缺什麼 | 為何無法在程式端解決 | 應由誰提供 |
|---|---|---|
| 現場 CSV 格式 | 部署未進行，現場輸出格式尚未存在 | 待站 4 解除阻擋後自然產生 |

---

## 第四部分：整圈狀態總覽

| 站 | 狀態 | 阻擋來源 |
|---|---|---|
| ① 資料準備 | 可執行 | 硬體端（感測器修復、SoC 校正）＋ 長期蒐集 |
| ② 模型訓練 | 可執行 | 計畫決策（目標函數）＋ 硬體端（BMS／PV 規格） |
| ③ 實驗驗證 | 部分執行 | 長期蒐集（資料不足 3 日）＋ 計畫決策（通過標準） |
| ④ 部署打包 | **未開始** | **硬體端／廠商（I/O 規格完全未知）** |
| ⑤ 現場繪圖 | 僅資料診斷 | 待站 ④ 解除 |

### 結論

newHW 目前完成的是**隔離的遷移骨架與有限的 in-sample 診斷**。
50-episode 歷史模型沒有實質學習；後續 300-episode／SoC 20–80% 試驗已在同一窗口達到
finite-window oracle，證明資料、訓練、充放電與 SafetyNet 管線可執行，且未污染既有 P302 系統
（175 項 P302 regression 全數通過）。

**但它仍不構成模型驗證，也不是部署候選。**
最新 best raw policy 有 55.85% steps 需要 SafetyNet 投影；沒有 held-out 日期、3 日／5 日資料或硬體端確認的限制。

整圈五站的主要阻擋仍是硬體規格、感測器修復、長期資料蒐集、I/O 規格與研究方向決策。
在這些外部輸入取得前，可以改善訓練穩定性與降低 SafetyNet 依賴，但無法證明泛化或部署安全。

接手者可立即比較 best／final、多 seed 與 SafetyNet 依賴；其餘各站仍需依上表向對應負責方取得資訊後才能推進。

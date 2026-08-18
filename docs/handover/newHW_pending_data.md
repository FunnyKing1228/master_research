# newHW 待補資料與阻擋項目

> 建議先讀 [`newHW_lifecycle_mapping.md`](newHW_lifecycle_mapping.md) 了解整體架構。

> 所有未確認值都標記為 `TODO(newHW)`。本頁不得用推估值冒充硬體規格或驗收結果。

## 阻擋環境定案

- [ ] `TODO(newHW)` **確認系統架構是否確為離網。** 目前 `core/microgrid_env_newHW.py` 的 `grid_kw` 恆為 0、情境碼僅輸出 1 與 4、reward 移除全部 TOU 項目，皆建立在「無市電接入」的推論上。依據為：(a) 原始 CSV 無任何 grid 電流或功率欄位；(b) 資料提供者描述的系統組成僅含 PV、電池、MPPT、BMS 與單一負載；(c) 2026-08-15 01:00–05:00 觀測到 BMS 跳脫後負載實際斷電，若有市電則不應發生。此推論尚未經硬體端確認。若實際存在市電（含未被量測的接入），環境設計需重做。應由硬體端確認。
- [ ] `TODO(newHW)` BMS 型號、規格書、continuous／charge／cutoff 電流設定值。
- [ ] `TODO(newHW)` BMS 實際單體低壓保護門檻；目前無法解釋為何 pack 約 25.43 V（3.18 V/cell）即跳脫。
- [ ] `TODO(newHW)` 8 顆電芯逐顆電壓；用於確認是否有弱電芯或不平衡。
- [ ] `TODO(newHW)` MPPT 控制器型號、額定功率與最大充電電流。
- [ ] `TODO(newHW)` PV 板額定瓦數、片數、串並聯方式與控制器接法。
- [ ] `TODO(newHW)` 主線線徑、保險絲／斷路器額定與其他電流限制。
- [ ] `TODO(newHW)` 真實負載規格；釐清「13 W，1 A」與約 26–28 W 實測的矛盾。
- [ ] `TODO(newHW)` 確認 28.2 W 迴歸負載是否已包含 MPPT、logger、DC-DC 與 4.8 W standby 寄生負載；目前不額外加 4.8 W。
- [ ] `TODO(newHW)` 修復 `Load_W` 感測通道並提供校正資料。
- [ ] `TODO(newHW)` SoC 經驗校正或 BMS coulomb-counting 依據。
- [ ] `TODO(newHW)` `MPPT_V_batt` 夜間固定 9.05 V 假值的成因與修復。
- [ ] `TODO(newHW)` 30.29 V（3.79 V/cell）三筆資料是量測突波或真實過充。
- [ ] `TODO(newHW)` 確認 pack voltage 缺值期間的替代方式；目前資料處理使用時間插值。
- [ ] `TODO(newHW)` 確認超過 60 秒斷點期間的電流／功率行為；目前能量積分暫按相鄰點線性變化。
- [ ] `TODO(newHW)` 確認 reconstructed SoC 初始 anchor；資料診斷暫用 1.0，模擬 reset 暫用 0.90。
- [ ] `TODO(newHW)` 確認 round-trip efficiency；目前暫用 0.95。
- [ ] `TODO(newHW)` 重新量測實際可用容量；0.20 kWh 假設下連續積分最低達 -1.605，與完整 47 小時軌跡不相容。
- [ ] `TODO(newHW)` 決定正式 SoC 上下限與 voltage cutoff；目前 `0.10–0.90` 只是保守 smoke 假設。
- [ ] `TODO(newHW)` 決定正式 battery charge/discharge power；目前分別以 0.129 kW PV 實測峰值與 35.7 W 負載狀態作暫定上限。
- [ ] `TODO(newHW)` 決定 PV availability 定義；processed CSV 暫以 1 W 作資料標記，但環境使用連續 PV power。
- [ ] `TODO(newHW)` 釐清不同能量帳版本：目前 processed CSV 為 load `1.3254 kWh`、PV `0.8341 kWh`、比例 `62.93%`、raw gap `0.4913 kWh`，與先前「PV 約 55%、gap 約 0.650 kWh」不一致。

## 阻擋 reward／訓練定案

- [ ] `TODO(newHW)` 由人類決定優先目標與權重：供電可靠度、夜間存活、BMS 保護、深度循環與 PV curtailment。
- [ ] `TODO(newHW)` `42.50%` served-energy 結果尚未釐清是訓練不足、reward／action 語意或環境限制；接手者應先在相同資料、初始 SoC 與限制下跑 PV-only、固定規則／greedy oracle、always-discharge 等非學習策略，逐步比較 raw／safe／applied action、served energy、unmet load 與期末 SoC，再決定是否重訓。
- [ ] `TODO(newHW)` best 與 final 兩個 checkpoint 的 SafetyNet 介入率相差約 18 倍（4.8% vs 88.3%），但 served energy fraction 與 unmet load 完全相同（42.50%／0.7620 kWh）。在 terminal-SoC-neutral 可持續上界 53.97% 下仍有約 152 Wh 空間，兩個差異極大的策略不應未經解釋地產生相同能源結果。輔證是 evaluation reward 長期停在約 `-1385.44`，且 best checkpoint 在第一次評估即保存、後續未被超越，顯示 50 episodes 沒有可確認的實質學習。此異常可能是：(a) action 沒有實際影響環境的動作路徑斷裂；或 (b) 訓練不足。接手者第一步應在完全相同資料、初始 SoC 與限制下，分別以 action 恆為 0、恆為最大充電、恆為最大放電跑 rollout；若三者 served energy fraction 都是 42.50%，優先查 action→SafetyNet→environment→applied action 路徑；若結果明顯不同，再歸入訓練／reward 問題。
- [ ] `TODO(newHW)` 定義是否允許可控卸載（load shedding）及其優先級；目前動作只有 battery power，環境不能主動關閉負載。
- [ ] `TODO(newHW)` 確認白天 PV 對負載、充電及 curtailment 的真實控制拓撲。
- [ ] `TODO(newHW)` 提供至少涵蓋不同天氣與負載狀態的長期連續資料。
- [ ] `TODO(newHW)` 建立彼此不重疊的 training／validation／held-out test 日期。

## 阻擋 3 日／5 日驗證

- [ ] `TODO(newHW)` 目前只有約 47 小時，無法執行既有連續 3 日與 5 日 rollout。
- [ ] `TODO(newHW)` 需要至少 5 個完整連續日；若要判斷泛化，還需額外未參與訓練的日期。
- [ ] `TODO(newHW)` 定義 newHW rollout 的硬性通過標準，包括 unmet load、BMS event、SoC、curtailment 與 SafetyNet intervention。

## 阻擋部署與打包

- [ ] `TODO(newHW)` 新硬體 I/O 通訊介面是否存在、協定格式、傳輸方式與提供者。
- [ ] `TODO(newHW)` 新硬體量測欄位、命令欄位、單位、更新頻率、ack／timeout／fail-safe 行為。
- [ ] `TODO(newHW)` 新硬體 controller／BMS／MPPT 的實際控制權限。
- [ ] `TODO(newHW)` 緊急停止、失聯、低壓、過壓與過流時由哪一層負責。
- [ ] `TODO(newHW)` 硬體驗收程序、場地、負責人與簽核方式。

在以上項目補齊前，`control/io_protocol_newHW.py` 必須維持 `NotImplementedError`，不得建立 parser、GUI release 或假協定。

## 阻擋附件重現

- [ ] `TODO(newHW)` 提供任務提到但目前未找到的 `diagnose.py`。
- [ ] `TODO(newHW)` 提供任務提到但目前未找到的 `data_quality_diagnosis.png`。

## Git clone 不包含的本機 artifacts

既有 `.gitignore` 會排除 `*.csv` 與 `experiments/*`。下一位接手者只 clone source 時，不會取得以下內容：

- 原始資料：`C:\Users\Administrator\Downloads\conformal-microgrid-rl\data\newHW\raw\Data140826.csv`
- processed dataset：`C:\Users\Administrator\Downloads\conformal-microgrid-rl\data\newHW\processed\training_newHW_15min.csv`
- oracle trace：`C:\Users\Administrator\Downloads\conformal-microgrid-rl\data\newHW\processed\energy_oracle_trace_newHW.csv`
- best checkpoint：`C:\Users\Administrator\Downloads\conformal-microgrid-rl\experiments\newHW_lfp_provisional_50ep_s42\models\best_sac_model.pth`
- final checkpoint：`C:\Users\Administrator\Downloads\conformal-microgrid-rl\experiments\newHW_lfp_provisional_50ep_s42\models\final_sac_model.pth`
- 完整 experiment：`C:\Users\Administrator\Downloads\conformal-microgrid-rl\experiments\newHW_lfp_provisional_50ep_s42\`

目前取得方式：

1. 向目前專案持有人／本工作站管理者索取，從上述絕對路徑做**私下 artifact 交接**，不可依賴 GitHub；目前 Windows 帳號為 `Administrator`，長期保管人姓名仍待指定。
2. 原始 CSV 傳輸後核對 SHA256：
   `9ada734a86a8aec822589d402b3ee639b62aa5fa3b7ec16025edf4af55e98881`。
3. best checkpoint SHA256：
   `e95f3b6ae8e1919b8f99034e9c9018c9ea5c69dd83a829f1b673f681a051ae4a`。
4. final checkpoint SHA256：
   `11f68fc32a76171771f8b6cdf5be90f7e150769656dff4d1d841fdaa67c827d7`。
5. 取得 source 後可重新執行 `prepare_data_newHW.py` 核對 processed CSV。

- [ ] `TODO(newHW)` 指定長期、具權限控管的 private artifact 保存位置與交接負責人；目前唯一明確位置是本工作站。

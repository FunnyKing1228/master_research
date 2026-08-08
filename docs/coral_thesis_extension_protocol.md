# CORAL 論文延伸實驗 Protocol

## 實驗動機

seminar 版模型差距表顯示，CORAL/OURS 的主要價值不應只看最終 net profit，而是看 learning-based EMS 是否在部署安全層之前就比較少提出危險動作。論文版延伸實驗因此把問題改寫為：

> 在與實體平台對齊的 P302 微電網模擬器中，CORAL 是否能比一般 SAC baseline 更少依賴 SafetyNet 修正，並在 held-out policy-in-loop、recovery-aware 與延伸情境中維持可部署性？

這個實驗不應被描述成「CORAL 一定打敗 expert heuristic」。它應該支援較穩健的主張：SafetyNet/CORAL 可作為安全診斷與部署投影機制，並檢查 learned policy 是否逐步 internalize hardware safety constraints。

## 核心設定

- 資料切分：`2026-06-17 12:00` 到 `2026-06-21 12:00` 保留作為 held-out policy-in-loop validation，不用於訓練。
- Episode 長度：主要訓練彙整使用 `96 steps = 1 day`。
- 主要觀察：保留 `pv_support_ratio` 與 PV blocking state 的區分，不把 PV/grid 寫成二元來源切換。
- 電池行為：battery discharge 不作為 PV/grid 的 parallel partial assist；若放電有效，需符合 solo load supply 的硬體解釋。
- 主要安全指標：raw attempted unsafe actions、realized SoC violations、SafetyNet meaningful interventions、projection magnitude。
- 主要部署指標：held-out net profit、SoC violations、final SoC、low-SoC duration、recovery-adjusted profit。

## No-Flow 與 Flow-Control 的公平性

no-flow 與 flow-control 不應混在同一張「誰比較強」總表中直接比較，因為它們的 action space 與 actuator setting 不同。

- no-flow：固定或不使用 flow-rate action，適合作為 seminar-style thesis rerun 的主比較設定。
- flow-control：多一個流速控制自由度，適合回答「額外 actuator 是否改善部署可行性」。
- 如果正文要放 flow/no-flow，建議明確標為 actuator-extension analysis。
- 若只有 CORAL flow-control 結果，不宜宣稱 flow-control 讓 CORAL 全面優於所有方法；至少需要 SAC+SafetyNet 或 SAC+SafetyNet+OCC 作為控制組。

## Training-Stage Internalization

正文可使用 last-20 episode mean 作為訓練末期行為摘要，但 appendix 應同時保留 last-50 與 last-100，以避免 cherry-picking。

建議欄位：

- method
- setting
- last-N net profit
- realized violations
- raw attempted unsafe actions
- SafetyNet interventions
- projection mean/max magnitude
- flow active ratio / pump Wh（只對 flow-control 有意義）

可支持的論文語氣：

> CORAL 在訓練末期是否比 baseline 更少提出 unsafe action，代表 learned actor 是否較少依賴部署安全層。

不建議的語氣：

> last-20 training net profit 證明 CORAL 已在部署上勝過 expert heuristic。

## Held-Out Deployed Validation

Training table 之後必須接 held-out policy-in-loop validation。這一步使用同一段外部條件，但讓 controller/policy 在模擬器內決策。

建議欄位：

- held-out net profit
- held-out SoC violations
- SoC min / final SoC
- grid demand
- PV blocks
- battery charge/discharge energy
- projection events
- low-SoC exposure

Recorded-command replay 仍只作為 energy-accounting sanity check，不能替代 policy-in-loop validation。

## 延伸情境

### Sell-Back / Export

Sell-back 是 extended economic scenario，不取代 base self-consumption 結果。它可用來測試當系統允許售電或 feed-in tariff 時，RL/CORAL 是否比 greedy 有更多策略空間。

建議敏感度：

- feed-in tariff ratio `0.3`
- feed-in tariff ratio `0.5`
- feed-in tariff ratio `0.8`

若只有 smoke run，應放 appendix 或 discussion，標註 pending。

### Component Specification / Capacity Stress

改變 battery capacity、battery power、pump power 或 load scale 可以作為 hypothetical design-space analysis。除非實體平台真的更換元件，否則不應稱為 sim-to-real validation。

目前最穩的是 capacity scale：

- `100%`
- `80%`
- `60%`
- `40%`

每個 capacity row 都應報告 recovery-aware 指標，避免低 SoC 策略造成短期 profit 假優勢。

## 建議章節位置

- 第 3 章：說明 seminar-style experiment 為何被升級成 training-stage internalization + held-out deployment protocol。
- 第 4 章：先放 base no-flow 主表，再放 flow/no-flow actuator extension，再放 held-out/recovery interpretation。
- 第 5 章：回扣 strong expert/greedy baseline 與 CORAL/SafetyNet 的部署價值和限制。
- Appendix：last-50/last-100 sensitivity、sell-back smoke、capacity stress、完整 flow-control 表。

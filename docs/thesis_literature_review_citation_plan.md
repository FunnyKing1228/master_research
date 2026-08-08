# Thesis Literature Review Citation Plan

本文件依據 `docs/refPaper/reference_paper_citation_notes.md` 的 40 篇文獻，以及目前 `NTU_Thesis_HaoYu_working/contents/chapter02.tex` 的文獻回顧現況，規劃第二章可逐步補強的 citation 佈局。目標是讓 literature review 能放入較多但不冗的引用，並避免把不同場景的假設誤套到本 thesis 的實體微電網平台。

## Chapter02 現況

目前 `chapter02.tex` 已有三個主要段落：

- `微電網能源管理中的強化學習`：目前引用 `Hadi2025`、`Kiasari2024`、`Haarnoja2018`、`Sharma2025`，可讀性順，但對 refPaper 新整理出的 microgrid EMS / simulator / PPO / DDPG 文獻使用偏少。
- `安全約束與可部署控制方法`：目前引用 `Yu2024`、`Kiasari2024`、`Vitelli2022`，方向正確，但可加入 safety layer、runtime shielding、conformal safety filter、viability shielding 等文獻，讓外掛式安全保護的論證更紮實。
- `模擬至實體遷移與現實落差問題`：目前引用 `Debnath2020`、`Angelopoulos2021`、`Alghumayjan2025`，但尚未充分連到 battery/BMS、hardware-aligned simulation、execution semantics 與 time delay。

目前 thesis `back/references.bib` 已有 `Haarnoja2018`、`Yu2024`、`Angelopoulos2021`、`Hadi2025`、`Kiasari2024`、`Sharma2025`、`Vitelli2022`、`Debnath2020`、`Alghumayjan2025`。因此，不建議在尚未補 `.bib` 前直接把 refPaper draft keys 寫進 LaTeX 主文。

## 建議 Chapter02 Structure

建議將第二章從目前三節擴為六個聚焦 subsection。可以保留既有文字作骨架，但在後續改寫時依此順序重組：

- `2.1 Microgrid EMS and Reinforcement Learning`
- `2.2 Greedy and Rule-Based Baselines`
- `2.3 Safe Reinforcement Learning and Runtime Intervention`
- `2.4 Conformal Prediction and Uncertainty-Aware Control`
- `2.5 Hardware-Aligned Simulation and Battery Constraints`
- `2.6 Safety-Margin Evaluation and Certifiability`
- `2.7 Research Gap and Thesis Positioning`

此結構能把「RL 可行性」、「baseline 為何合理」、「safe runtime intervention」、「conformal uncertainty」、「硬體限制」與「研究缺口」拆清楚，也能避免把 citation 全部堆在同一段。

## 40 篇 Citation Key 與最適用途

- `Haarnoja2018`：SAC foundational RL algorithm；用於 Method/RL algorithm 背景。已在 thesis `.bib`。
- `Schulman2017PPO`：PPO foundational algorithm；用於 PPO baseline 與穩定 policy update 背景。
- `Dalal2018SafetyLayer`：continuous action safety layer；用於 runtime safety intervention。
- `Alshiekh2018Shielding`：safe RL via shielding；用於 runtime shield / 最小干預安全修正。
- `Krasowski2023ProvablySafeRL`：provably safe RL survey / benchmarking；用於 action replacement、action projection、action masking 分類，以及 projection-based safety filters。
- `SafeActionProjection2025`：projection-based safe RL 中 safe-environment vs safe-policy、action aliasing；用於說明過度保守 projection 可能改變 closed-loop behavior。
- `Spoor2025LagrangianSafeRL`：Lagrangian safe RL empirical study；用於 safety-related hyperparameter profile / sensitivity analysis 類比。
- `Angelopoulos2024ConformalRiskControl`：conformal risk control；用於 conformal risk / safety threshold 理論。
- `Henri2021Pymgrid`：microgrid simulator benchmark；用於 microgrid simulation / benchmark 背景。
- `CommunityMicrogrid2024PPO`：community microgrid PPO EMS；用於 microgrid EMS/RL application。
- `XDRL2024BatteryAwareMicrogrid`：explainable battery-aware microgrid DRL；用於 battery-aware RL 與 explainability discussion。
- `HierarchicalShielding2024PowerGrid`：runtime shielding for power-grid operation；用於 safe RL / runtime shield。
- `Strawn2023ConformalSafetyFilter`：conformal safety filter for RL controllers；用於 conformal + safety filter 結合。
- `Tonkens2023SafePlanningCP`：safe planning with conformal prediction；用於 uncertainty-aware planning。
- `Lindemann2023SafePlanningCRC`：CRC + CBF for probabilistic safety；用於 risk control + control-layer protection。
- `SelectiveCRC2025`：selective conformal risk control；用於 abstention / conservative operation discussion。
- `SIOCP2025SafeDynamics`：online conformal prediction with multi-step coverage；用於 cross-day adaptation discussion。
- `AdaptNC2025AdaptiveNonconformity`：adaptive nonconformity under dynamic environments；用於 distribution shift discussion。
- `DualAdaptiveCP2025TrustworthyUQ`：distributionally robust adaptive CP；用於 temporal dependence / shift-aware UQ。
- `SPICE2025WeakestPreconditions`：weakest-precondition safe exploration；用於 training-time shield/future work。
- `ViabilityShielding2025Hypersonic`：viability-based action shielding；用於 action admissibility / safe set concept。
- `Li2025HumanAlignedSafeRL`：CMDP + MPC shielding；用於 reward-only safety 不足。
- `Tetteh2026VLMSafeRL`：anticipatory safe RL；用於 anticipatory safety concept，較適合 discussion。
- `STREAMRL2025SafeTraffic`：safe urban traffic via uncertainty-aware conformal methods；用於 dynamic-demand safety analogy，較適合 discussion。
- `MeasurementInformed2025QuantumBatterySafeRL`：BMS constraints + diagnostics；用於高層 battery safety analogy，不作核心。
- `Che2025EnhancedSOC`：LFP SOC estimation uncertainty；用於 SOC/BMS state uncertainty。
- `Kulkarni2026SOCTrapezoidal`：long-horizon SOC drift reduction；用於 cross-day SOC drift discussion。
- `Lanubile2024SOHDomainKnowledge`：SOH estimation with domain knowledge；用於 battery health / degradation discussion。
- `Plett2004EKFBMSPart2`：EKF BMS modeling and identification；用於 battery state modeling foundational support。
- `Wang2018VRBFlowRate`：VRB flow-rate efficiency/pump tradeoff；用於 hardware auxiliary-loss analogy。
- `Kolli2019RFBFlowRateOptimization`：heuristic/ML RFB flow-rate optimization；用於 heuristic baseline analogy，非核心。
- `Xie2023VRFBFlowRates`：VRFB flow-rate experiment and pump loss；用於 storage hardware constraints discussion。
- `Dong2026MicrogridDDPG`：DDPG microgrid real-time dispatch；用於 microgrid EMS/RL。
- `Liu2023MicrogridDDPG`：model-free DRL under RES/load/price uncertainty；用於 microgrid EMS/RL。
- `Ali2026ForecastAwarePPO`：forecast-aware PPO for load scheduling；用於 forecast feature / energy scheduling discussion。
- `Vovk2005RandomWorld`：conformal prediction foundational book；用於 conformal theory。
- `Sun2023PlanCP`：CP for uncertainty-aware planning with diffusion dynamics；用於 CP + planning。
- `Bedi2026CARE`：CRC as post-hoc safety layer；用於 model-agnostic safety layer analogy，discussion。
- `Wang2025FeedbackCPPoster`：feedback CP in trajectory optimization loop；poster-only，appendix/future work only unless paper found。
- `Chehadeh2022TimeDelaySim2Real`：time-delay sim-to-real gap in CPS RL；用於 deployment limitations。
- `Hoss2026ExecutionSemantics`：execution semantics for sim-to-real dispatching；用於 action validity / execution outcome attribution。
- `Almaghrabi2026HybridMPCDRLMicrogrids`：hybrid MPC-DRL microgrid control；用於 safety-aware microgrid control。
- `Badakhshan2026HeuristicsRL`：heuristics-to-RL comparison；用於 greedy baseline rationale，但不是 microgrid evidence。
- `Yu2024SafeRLPowerSystemReview`：safe RL for power-system control review；用於 safety/RL/power systems 主軸。thesis 目前用 `Yu2024`，可視情況沿用既有 key 或改為 draft key。

## 2.1 Microgrid EMS and Reinforcement Learning

核心引用：`Hadi2025`、`Kiasari2024`、`Liu2023MicrogridDDPG`、`Dong2026MicrogridDDPG`、`CommunityMicrogrid2024PPO`、`Almaghrabi2026HybridMPCDRLMicrogrids`、`Henri2021Pymgrid`。

可合併引用：`Haarnoja2018`、`Schulman2017PPO`、`Sharma2025`、`Ali2026ForecastAwarePPO`、`XDRL2024BatteryAwareMicrogrid`。

示範段落：

微電網 EMS 通常需在負載、再生能源輸出、儲能狀態與電價訊號變動下進行連續決策，因此近年研究大量採用 DRL 處理不確定性與即時調度問題。既有 review 與 microgrid EMS 研究已指出，DDPG、PPO、SAC 等方法可用於經濟調度、renewable utilization 與儲能管理，但多數設定仍以模擬或一般化 ESS 模型為主，而非特定硬體平台的動作可行性驗證 `\citep{Hadi2025,Kiasari2024,Liu2023MicrogridDDPG,Dong2026MicrogridDDPG,CommunityMicrogrid2024PPO}`。

在方法層面，SAC 與 PPO 分別代表 off-policy maximum-entropy continuous control 與 clipped policy update 的常用選擇，可作為本研究 learning-based EMS 的演算法背景 `\citep{Haarnoja2018,Schulman2017PPO}`。同時，pymgrid 等模擬平台文獻凸顯 microgrid RL benchmark 的重要性，但也提醒 thesis 必須清楚區分 benchmark-level energy balance 與本平台 PV/grid co-support、battery discharge blocking 等硬體條件 `\citep{Henri2021Pymgrid,Almaghrabi2026HybridMPCDRLMicrogrids}`。

寫作注意：

- 避免使用「solar/grid 二選一」語句。
- 建議用「PV support reduces grid demand」與「simulation assumptions may not encode hardware action admissibility」。
- `XDRL2024BatteryAwareMicrogrid` 適合放一兩句補充 battery-aware / explainable policy，不必成為核心段落。

## 2.2 Greedy and Rule-Based Baselines

核心引用：`Badakhshan2026HeuristicsRL`、`CommunityMicrogrid2024PPO`、`Kolli2019RFBFlowRateOptimization`。

可合併引用：`Hadi2025`、`Henri2021Pymgrid`、`Ali2026ForecastAwarePPO`。

示範段落：

除了學習式控制器，rule-based 或 greedy heuristic 仍常被用作能源管理研究中的可解釋比較基準。這類方法的價值不在於提供完美專家策略，而在於以透明規則呈現安全優先、成本優先或折衷型調度邏輯，使學習式策略的改善幅度與失敗模式更容易被檢視 `\citep{CommunityMicrogrid2024PPO,Badakhshan2026HeuristicsRL}`。

因此，本 thesis 可將 `safety-greedy`、`profit-greedy` 與 `balanced-greedy` 定位為 practical baseline，而非人工最優控制器。若要補充 energy-storage control 中 heuristic/ML 的常見性，可簡短引用 RFB flow-rate optimization 文獻，但應明確說明其場景只提供方法類比，不能用來支持本平台電池化學或充放電限制 `\citep{Kolli2019RFBFlowRateOptimization}`。

寫作注意：

- 不要稱 greedy baseline 為 perfect expert。
- 若主文篇幅有限，可把 `Badakhshan2026HeuristicsRL` 放主文，`Kolli2019RFBFlowRateOptimization` 放 discussion 或 appendix。

## 2.3 Safe Reinforcement Learning and Runtime Intervention

核心引用：`Yu2024SafeRLPowerSystemReview`、`Dalal2018SafetyLayer`、`HierarchicalShielding2024PowerGrid`、`Strawn2023ConformalSafetyFilter`、`ViabilityShielding2025Hypersonic`、`Li2025HumanAlignedSafeRL`。

可合併引用：`Vitelli2022`、`SPICE2025WeakestPreconditions`、`Tetteh2026VLMSafeRL`、`STREAMRL2025SafeTraffic`、`Almaghrabi2026HybridMPCDRLMicrogrids`。

示範段落：

Safe RL 文獻普遍指出，僅透過 reward penalty 引導策略避開違規狀態，並不等同於部署時的安全保證。Power-system safe RL review 將 action projection、safety layer、shielding、CMDP 與 CBF 等方法整理為降低 critical constraint violation 的主要方向；對本 thesis 而言，這些文獻可支撐「policy output 與實體執行之間需要 runtime intervention」的研究定位 `\citep{Yu2024SafeRLPowerSystemReview,Dalal2018SafetyLayer,HierarchicalShielding2024PowerGrid}`。

在外掛式安全保護中，safety layer 與 shielding 的共通思想是：策略可以提出候選動作，但執行前必須經過 action admissibility 檢查或最小修正。此觀點與本研究的 hardware-aligned blocking 相容，因為 battery discharge 是否可執行不應只由 reward 學得，而應由 PV support、load demand、SoC 與平台供能邏輯共同決定 `\citep{Dalal2018SafetyLayer,ViabilityShielding2025Hypersonic,Li2025HumanAlignedSafeRL}`。

寫作注意：

- `Yu2024SafeRLPowerSystemReview` 與現有 `.bib` 的 `Yu2024` 可能是同一篇；若沿用 thesis 既有 key，plan 內對應到 `Yu2024` 即可。
- `SPICE2025WeakestPreconditions`、`Tetteh2026VLMSafeRL`、`STREAMRL2025SafeTraffic` 比較適合 discussion/future work，主文不用全部放。

## 2.4 Conformal Prediction and Uncertainty-Aware Control

核心引用：`Angelopoulos2021`、`Vovk2005RandomWorld`、`Angelopoulos2024ConformalRiskControl`、`Strawn2023ConformalSafetyFilter`、`Tonkens2023SafePlanningCP`、`Sun2023PlanCP`。

可合併引用：`Lindemann2023SafePlanningCRC`、`SelectiveCRC2025`、`SIOCP2025SafeDynamics`、`AdaptNC2025AdaptiveNonconformity`、`DualAdaptiveCP2025TrustworthyUQ`、`Alghumayjan2025`、`Bedi2026CARE`、`Wang2025FeedbackCPPoster`。

示範段落：

Conformal prediction 提供 distribution-free uncertainty quantification 的基礎工具，可將模型預測誤差轉化為具有有限樣本意義的 prediction set 或 risk control threshold `\citep{Vovk2005RandomWorld,Angelopoulos2021,Angelopoulos2024ConformalRiskControl}`。在能源管理場景中，此類方法可用於處理 price、load 或 renewable forecast 的不確定性；但在本 thesis 中，更重要的是將 uncertainty 與 runtime safety decision 連結，而不是只把 conformal interval 作為事後分析指標 `\citep{Alghumayjan2025,Strawn2023ConformalSafetyFilter}`。

近期 CP planning 與 conformal safety filter 文獻顯示，校準後的不確定性可以被放入 planning 或 controller safety layer，用以在高風險狀態下採取保守行為 `\citep{Tonkens2023SafePlanningCP,Sun2023PlanCP,Lindemann2023SafePlanningCRC}`。不過，這類保證通常依賴 calibration data 與 deployment data 的假設條件；若本研究面對跨日 PV/load drift，則需要把 online/adaptive conformal 方法作為限制與 future work 討論，而非宣稱已有完整跨日保證 `\citep{SIOCP2025SafeDynamics,AdaptNC2025AdaptiveNonconformity,DualAdaptiveCP2025TrustworthyUQ}`。

寫作注意：

- 主文核心可放 5-7 篇，adaptive/online CP 可集中到 discussion。
- `Wang2025FeedbackCPPoster` 是 poster-only，除非找到正式 paper，建議只放 appendix/future work 或不進主文。

## 2.5 Hardware-Aligned Simulation and Battery Constraints

核心引用：`Che2025EnhancedSOC`、`Kulkarni2026SOCTrapezoidal`、`Lanubile2024SOHDomainKnowledge`、`Plett2004EKFBMSPart2`、`Chehadeh2022TimeDelaySim2Real`、`Hoss2026ExecutionSemantics`。

可合併引用：`Wang2018VRBFlowRate`、`Xie2023VRFBFlowRates`、`MeasurementInformed2025QuantumBatterySafeRL`、`XDRL2024BatteryAwareMicrogrid`、`Debnath2020`。

示範段落：

實體微電網中的 battery state 並非可由單一瞬時訊號完全決定。BMS 與 SOC/SOH 文獻指出，電池狀態估測會受到 OCV 平坦區、hysteresis、path dependence、sensor error、capacity fade 與 long-horizon drift 影響，因此 thesis 在描述 SoC 或 battery availability 時，應避免把量測值寫成無誤差的物理真值 `\citep{Plett2004EKFBMSPart2,Che2025EnhancedSOC,Kulkarni2026SOCTrapezoidal,Lanubile2024SOHDomainKnowledge}`。

此外，hardware-aligned simulation 不只需要考慮名目 power balance，也需要定義 action admissibility、execution delay、感測/通訊延遲與實際執行結果的 attribution。Time-delay sim-to-real 與 execution semantics 文獻可支撐本研究的部署導向觀點：即使 policy 在 simulator 中得到高 reward，若執行層沒有記錄 action 是否可執行、是否被 blocking、以及 blocking 的物理原因，就難以判斷策略是否真正可部署 `\citep{Chehadeh2022TimeDelaySim2Real,Hoss2026ExecutionSemantics,Debnath2020}`。

寫作注意：

- VRFB flow-rate 文獻只用於「儲能系統存在內部效率與 auxiliary loss」的類比，不可拿來描述本平台 battery chemistry。
- `MeasurementInformed2025QuantumBatterySafeRL` 場景差異很大，建議不進主文，最多 discussion 一句。

## 2.6 Safety-Margin Evaluation and Certifiability

核心引用：`Dalal2018SafetyLayer`、`Alshiekh2018Shielding`、`Krasowski2023ProvablySafeRL`、`Spoor2025LagrangianSafeRL`、`SafeActionProjection2025`、`Angelopoulos2024ConformalRiskControl`。

可合併引用：`Yu2024SafeRLPowerSystemReview`、`Strawn2023ConformalSafetyFilter`、`Li2025HumanAlignedSafeRL`、`ViabilityShielding2025Hypersonic`。

示範段落：

本 thesis 的 `soc_margin` sweep 不應被描述為一般 hyperparameter tuning，而應定位為 deployment safety layer 的 conservativeness / certifiability analysis。Safe RL 文獻中，runtime safety layer 或 shielding 常被用來在 policy action 與 physical execution 之間執行 action correction、projection 或 masking，使原始策略輸出的候選動作在執行前被轉換為可接受動作 `\citep{Dalal2018SafetyLayer,Alshiekh2018Shielding,Krasowski2023ProvablySafeRL}`。在此脈絡下，margin 代表 safety filter 的保守程度：margin 越大，internal projection set 越小；margin 越小，policy 可使用的操作空間越接近真實安全邊界。

因此，本研究的 margin sweep 回答的不是「哪個 margin 讓分數最好」，而是「在多大的保守部署緩衝下，各 policy 可以通過 zero strict violation gate」。這與 constrained / safe RL 文獻中常見的 cost-limit、constraint-threshold 或 Lagrange multiplier profile 類似，目的在於呈現 reward-safety trade-off 與方法對 safety threshold 的敏感度，而不是單點調參 `\citep{Spoor2025LagrangianSafeRL}`。

此外，projection-based safety layer 並不保證 margin 越大越好。當 internal projection set 過窄時，許多不同 unsafe actions 可能被投影到相同或相近的 safe action，造成 action aliasing；在 flow-rate control 這類有 pump auxiliary loss、effective power limits 與 multi-dimensional action 的設定中，過度保守 projection 也可能改變 closed-loop trajectory，使 recovery capability 變差 `\citep{SafeActionProjection2025}`。因此，本 thesis 觀察到 flow-control 中 margin 過大後 violation 重新出現，應解釋為 safety layer 與 actuator dynamics 的 non-monotonic interaction，而非單純實驗錯誤。

寫作注意：

- `soc_margin` 不是 true safety boundary。True SoC bounds 仍是 20-80%；margin 只是在 deployment projection 中把 internal bounds tighten 成 20+margin 到 80-margin。
- 不要說「margin sweep 找最佳 hyperparameter」。建議說「margin sweep evaluates the minimum conservative buffer needed for strict safety certification」。
- 若結果有 non-monotonic behavior，應寫成 limitation / diagnostic：current SafetyNet is a practical one-step projection layer, not a formal invariant controller for the full flow-action dynamics。
- Flow-control 結果應作 actuator-space stress test；sell-back/grid-export 版本應作 future extension，不能和目前 no-flow/flow strict safety gate 結論混在一起。

## 2.7 Research Gap and Thesis Positioning

核心引用：`Hadi2025`、`Yu2024SafeRLPowerSystemReview`、`Liu2023MicrogridDDPG`、`Dong2026MicrogridDDPG`、`Angelopoulos2024ConformalRiskControl`、`Chehadeh2022TimeDelaySim2Real`、`Hoss2026ExecutionSemantics`。

可合併引用：`Henri2021Pymgrid`、`Almaghrabi2026HybridMPCDRLMicrogrids`、`Che2025EnhancedSOC`、`Plett2004EKFBMSPart2`、`Strawn2023ConformalSafetyFilter`。

示範段落：

綜合上述文獻，既有 microgrid RL 研究已證明 learning-based EMS 對不確定能源調度具有潛力，但多數工作仍以模擬績效、經濟成本或一般化 ESS dispatch 為主要評估焦點 `\citep{Hadi2025,Liu2023MicrogridDDPG,Dong2026MicrogridDDPG,Henri2021Pymgrid}`。相對地，本 thesis 的缺口設定不在於單純替換更強的 RL algorithm，而在於把 PV support、battery discharge admissibility、safety intervention 與 cross-day validation 整合到一個可部署的控制流程中。

另一方面，safe RL 與 conformal uncertainty 文獻提供了 safety layer、shielding 與 calibrated risk control 的方法背景，但這些概念仍需轉譯到本平台的物理限制與執行邏輯 `\citep{Yu2024SafeRLPowerSystemReview,Angelopoulos2024ConformalRiskControl,Strawn2023ConformalSafetyFilter}`。因此，本研究可定位為：在實體 microgrid 條件下，重新定義 observation、action blocking、evaluation metrics 與 deployment records，使 RL policy 的名目表現、硬體可行性與不確定性安全邏輯能被同時檢驗。

寫作注意：

- Research gap 盡量少列舉文獻，多用「既有文獻共同留下的缺口」收束。
- 必須強調 multi-day / cross-day validation，不要把 single-day result 寫成 thesis-ready evidence。

## 建議只放 Appendix 或 Discussion 的文獻

以下文獻可以保留在 plan 中，但主文若篇幅有限不建議硬塞：

- `Tetteh2026VLMSafeRL`：VLM/racing 場景差太大，只保留 anticipatory safety 概念。
- `STREAMRL2025SafeTraffic`：traffic control 類比 dynamic demand + conformal safe RL，適合 discussion。
- `MeasurementInformed2025QuantumBatterySafeRL`：quantum battery 與本平台差異過大，只能支持 BMS constraints 高層概念。
- `Kolli2019RFBFlowRateOptimization`：文獻品質與 metadata 仍需確認，可支援 heuristic/ML storage control 類比。
- `Wang2025FeedbackCPPoster`：poster-only，正式 thesis 引用前需找到 paper 或正式頁面。
- `Bedi2026CARE`：medical summarization 場景不同，適合說明 CRC 可作 post-hoc safety layer。
- `SPICE2025WeakestPreconditions`：若 thesis 未做 formal weakest precondition，可放 future work。
- `SelectiveCRC2025`、`SIOCP2025SafeDynamics`、`AdaptNC2025AdaptiveNonconformity`、`DualAdaptiveCP2025TrustworthyUQ`：可在 discussion 說明 online/adaptive calibration 是處理 cross-day drift 的後續方向。
- `Wang2018VRBFlowRate`、`Xie2023VRFBFlowRates`：若本文電池不是 VRFB，主文最多作 hardware auxiliary loss 類比，細節可放 discussion。
- `SafeActionProjection2025`：若正式出版資訊未確認，可先用於 thesis discussion 或 internal report，正式論文引用前需確認版本。

## `.bib` 優先補入順序

第一優先：最能直接支撐 chapter02 主線，建議先補進 `NTU_Thesis_HaoYu_working/back/references.bib`。

- `Schulman2017PPO`
- `Liu2023MicrogridDDPG`
- `Dong2026MicrogridDDPG`
- `CommunityMicrogrid2024PPO`
- `Henri2021Pymgrid`
- `Almaghrabi2026HybridMPCDRLMicrogrids`
- `Dalal2018SafetyLayer`
- `Alshiekh2018Shielding`
- `Krasowski2023ProvablySafeRL`
- `HierarchicalShielding2024PowerGrid`
- `Strawn2023ConformalSafetyFilter`
- `Vovk2005RandomWorld`
- `Angelopoulos2024ConformalRiskControl`
- `Che2025EnhancedSOC`
- `Plett2004EKFBMSPart2`
- `Chehadeh2022TimeDelaySim2Real`
- `Hoss2026ExecutionSemantics`
- `Badakhshan2026HeuristicsRL`

第二優先：適合補強 discussion、limitations 或更完整文獻回顧。

- `Tonkens2023SafePlanningCP`
- `Lindemann2023SafePlanningCRC`
- `Sun2023PlanCP`
- `ViabilityShielding2025Hypersonic`
- `Li2025HumanAlignedSafeRL`
- `Spoor2025LagrangianSafeRL`
- `SafeActionProjection2025`
- `Ali2026ForecastAwarePPO`
- `XDRL2024BatteryAwareMicrogrid`
- `Kulkarni2026SOCTrapezoidal`
- `Lanubile2024SOHDomainKnowledge`
- `Wang2018VRBFlowRate`
- `Xie2023VRFBFlowRates`

第三優先：建議先查 metadata 或只放 appendix/future work。

- `SelectiveCRC2025`
- `SIOCP2025SafeDynamics`
- `AdaptNC2025AdaptiveNonconformity`
- `DualAdaptiveCP2025TrustworthyUQ`
- `SPICE2025WeakestPreconditions`
- `Tetteh2026VLMSafeRL`
- `STREAMRL2025SafeTraffic`
- `MeasurementInformed2025QuantumBatterySafeRL`
- `Kolli2019RFBFlowRateOptimization`
- `Bedi2026CARE`
- `Wang2025FeedbackCPPoster`

特殊處理：

- `Yu2024SafeRLPowerSystemReview` 已在 thesis `.bib` 以 `Yu2024` 存在，建議不要重複新增；可選擇保留 `Yu2024` 作 thesis key，或統一改 key 但需全專案同步。
- `Haarnoja2018` 已存在於 thesis `.bib`，不需重複新增。
- `Angelopoulos2021` 已存在，可繼續作 conformal prediction 入門引用；若要寫 CRC，仍需補 `Angelopoulos2024ConformalRiskControl`。

## 建議改寫順序

1. 先把第一優先 `.bib` entries 補齊，至少補完 microgrid EMS、safe RL、conformal theory、battery/BMS 與 sim-to-real 五類核心文獻。
2. 再重組 `chapter02.tex` 為六個 subsection。這一步可以沿用目前三節文字，但把 citation 插入自然段落，而不是在句末堆 8-10 篇。
3. 最後才視篇幅加入 discussion-only 文獻。若第二章已過長，第三優先文獻可放到 limitations 或 appendix 的 extended related work。

## LaTeX 插入建議

目前不建議直接在 `chapter02.tex` 加入尚未進 `.bib` 的 `\citep{...}`。後續若要小步修改，可先在相關段落加入 LaTeX comments，而非正式 citation，例如：

```tex
% TODO(citations): after adding bib entries, expand with Liu2023MicrogridDDPG, Dong2026MicrogridDDPG, CommunityMicrogrid2024PPO.
% TODO(citations): safe RL paragraph can cite Dalal2018SafetyLayer, HierarchicalShielding2024PowerGrid, Strawn2023ConformalSafetyFilter.
% TODO(citations): hardware-aligned simulation paragraph can cite Che2025EnhancedSOC, Plett2004EKFBMSPart2, Chehadeh2022TimeDelaySim2Real, Hoss2026ExecutionSemantics.
```

若下一步要直接改 `chapter02.tex`，建議同步進行 `.bib` 補齊，避免 LaTeX 編譯時出現 unresolved citations。

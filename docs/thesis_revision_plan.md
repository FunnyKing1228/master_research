# Thesis Revision Plan

This document is the working plan for revising the thesis after the later deployment, flow-rate, SoH, SoC calibration, and baseline/ablation work. It should be treated as the main coordination note so the thesis revision does not become scattered.

## Current Problem With the Existing Thesis

The current thesis version still ends mainly at deployment-oriented virtual validation. It presents a candidate policy validated by single-day and three-day simulation-style rollouts, and Appendix B gives too much central importance to teacher-guided training and behavior cloning.

This no longer matches the current project status. The project has since expanded into a broader deployment-safe RL framework with:

- CORAL and ablation/baseline comparisons.
- Real deployment preparation and raw-data diagnostics.
- SoC calibration and sensor-freeze analysis.
- Flow-rate-aware action and flow-power-limited modeling.
- SoH monitoring and partial online SoH prediction integration.
- Deployment packaging and GUI integration.

## New Thesis Main Narrative

Old narrative:

> I trained a teacher-guided candidate policy and verified it in virtual single-day and three-day rollouts.

New narrative:

> I developed a deployment-safe reinforcement learning workflow for a real microgrid platform. The work aligns simulation, physical constraints, safety mechanisms, flow-rate control, SoC/SoH monitoring, and deployment diagnostics. Real deployment findings were used to identify remaining sim-to-real gaps and motivate further safety-aware improvements.

The thesis should shift from a candidate-model story to a system-level deployment-safety story.

### 2026-06-22 Thesis Direction Update

The thesis title and main theme should remain close to:

> Safety- and profit-oriented microgrid energy management system.

However, the narrative order should move the physical-platform alignment earlier. The thesis should first establish a simulation environment that is explicitly aligned with the real P302 platform, then show a simulation-vs-hardware validation plan / placeholder, and only after that use the trusted aligned simulation to compare control strategies. This ordering matches the advisor's current priority: physical validation should support the credibility of later simulation scaling.

Recommended revised evidence logic:

1. Define the real microgrid platform, EMS scope, hardware constraints, and source semantics.
2. Build the aligned simulation environment and explain why it reflects the deployable control problem.
3. Reserve a simulation-vs-hardware validation section to compare simulated and physical behavior once enough data is available.
4. Compare safety/profit/balanced greedy baselines, RL baselines, and CORAL within the trusted simulation setting.
5. Move complex implementation details, parameter tables, rule variants, and non-core diagnostics to appendix.

Important writing decision:

- Do not put substantive simulation-vs-hardware result claims into the thesis yet. Keep only the section structure, TODO placeholder, validation protocol, and expected evidence type until enough measured data is available.
- The main baseline should be the compact greedy-family framing: safety-first greedy, profit-first greedy, and balanced safety-profit greedy. Do not return to presenting a perfect expert rule as the main baseline.
- Use "PV support" / "grid demand reduction" wording. Avoid binary solar-vs-grid source claims unless directly measured.

### 2026-06-25 Confirmed Baseline / Ablation Chapter Spec

This is now the working specification for future thesis writing and experiment planning. Avoid adding duplicate planning documents unless the scope changes substantially.

Chapter 3 is responsible for experimental logic and design: why the comparison is made and what is being compared. Section 3.7 should define baseline methods, ablation variants, and fairness design. Baselines should be rule-based / greedy controllers used for engineering and academic comparison, not a perfect expert. Ablations should remove or disable core mechanisms such as OCC or adaptive rules to test the corresponding hypotheses. The fairness statement must require the same training data, decision step, physical boundaries, action limits, and evaluation metrics wherever applicable. Section 3.9 should define the evaluation protocol and metrics, including total cost or net return, violations, safety interventions, unsafe attempts, SoH-related indicators if used, multi-day rollout, and random-start evaluation. Detailed hyperparameters and rule thresholds belong in Appendix B, not in the main method text.

Chapter 4 is responsible for scenarios, data, and analysis: what test data was used, what results came out, and what each result proves. Section 4.2 should define the strategy-comparison setting, including test scenario, duration, typical weather/day selection where available, held-out interval, and random-start range. Section 4.3 should present the baseline and safety-mechanism ablation results through a main metric table, convergence or learning behavior only when generated from the same experimental line, and qualitative phenomenon analysis. Do not mix training, evaluation, and held-out numbers in one result table. If a table cell is not available from the same output protocol, leave it as TODO rather than filling inferred values.

Appendix B is responsible for reproducibility details. B.1 should contain mainline training and deployment settings such as learning rate, batch size, network architecture, discount factor, safety parameters, and shared environment assumptions. B.2 should contain the greedy / rule-based controller logic, including if-else thresholds and concrete execution values. B.3 may contain flow-rate and deployment settings, including pre-measure flow, standby / active flow assumptions, flow/no-flow scope, and any deployment command conversion details.

Important interpretation rules:

- Separate the raw actor policy from the shielded deployed policy. Results should report raw unsafe attempts and safety-layer intervention separately from final executed actions.
- Old experiments can remain, but they must be downgraded to early pre-deployment behavior checks, historical comparisons, or supplemental evidence. They should not be presented as the main thesis result.
- Flow/no-flow results should be explicitly labeled preliminary unless the comparison is complete and uses the same held-out protocol.
- Use wording such as `太陽能占比`, `太陽能功率`, and `市電補足量`. Avoid binary source-switching terminology.

## Seminar Material Mapping

Two later presentation files should be used differently during thesis revision.

### 2026-05-26 ECS Presentation

This presentation is close to the revised thesis main line. It should be treated as a core source for:

- Deployment-safe RL motivation.
- Deployment gap framing:
  - noisy and censored sensors;
  - Coulomb-counting SoC drift;
  - demand-censored PV measurement;
  - hardware latency;
  - messy and incomplete real-world data.
- CORAL explanation:
  - CRTSN as guardrail;
  - OCC as proactive opportunity-cost awareness;
  - Adaptive Loop as long-term calibrator.
- Baseline and ablation comparison framing.
- Real-world deployment case study framing.
- Limitations of CORAL as a deployment-aware framework rather than a plug-and-play controller.

Use this presentation to rebuild Chapter 1, Chapter 3, and the first half of Chapter 4.

### 2026-06-16 Flow-Rate Presentation

This presentation should be treated as a hardware-aware extension, not the main thesis line.

It can support a Chapter 4 or Chapter 5 subsection on flow-rate-aware control:

- Motivation for flow-rate control in a flow battery.
- 2D action space: battery power and flow rate.
- Conservative linear flow-power coupling.
- Minimum 60% active flow constraint.
- Idealized pump motor assumption using 10% of measured motor power.
- Cubic pump power relationship from pump affinity laws.
- 1D power-only vs 2D flow-aware comparison.
- Explanation that flow constraints reduce feasible action space and profit but improve safety.

The flow presentation title is broader than the thesis should become. Do not let it force the whole thesis into a "power-flow co-optimization" thesis unless the entire method, experiments, and title are rewritten around flow.

## Flow-Rate Positioning Decision

Flow rate should be positioned as an extension or case study, not the central thesis contribution.

Recommended thesis positioning:

> Flow-rate control is treated as a hardware-aware extension rather than the central thesis objective. The main contribution of this thesis is the deployment-safe RL framework for microgrid energy management under real physical constraints. Flow rate is introduced as an additional case study showing how the framework can be expanded when deployment reveals new hardware-dependent operating variables.

Reasoning:

- The main thesis already has a complete contribution line around deployment-safe RL, CORAL, safety, sim-to-real alignment, and real deployment diagnosis.
- Making flow rate the main topic would require rewriting the title, objective, method, baseline protocol, and all results around 2D power-flow control.
- Current flow-rate results rely on idealized assumptions, especially scaled pump motor power and conservative flow-power coupling.
- The real hardware flow-power relationship and pump cost need more calibration before flow rate can carry the entire thesis.

Suggested placement:

- Chapter 3: describe flow rate as an optional hardware-aware action-space extension.
- Chapter 4: include a section such as "Hardware-Aware Extension: Flow-Rate Control".
- Chapter 5: discuss flow rate as future deployment direction and platform-specific extension.

## Reposition Teacher Guidance

Teacher guidance should no longer be described as the core final method.

Recommended positioning:

- Teacher guidance was an early-stage stabilization method.
- It can be discussed as one optional training aid or historical baseline.
- It should not be presented as the main contribution.
- The final thesis should emphasize CORAL, deployment alignment, safety mechanisms, and hardware-aware modeling.

Suggested wording:

> Teacher-guided learning was used in early development to stabilize exploration and provide physically feasible examples, but it is not the central mechanism of the proposed framework. The final research focus is on deployment-safe control, CORAL-based safety adaptation, and real-platform alignment.

## Chapter-Level Revision Plan

## Proposed Thesis Structure Revision

Do not start by polishing individual paragraphs. The current thesis problem is structural: Chapter 4 and Appendix B no longer match the current project status.

Recommended revision order:

1. Redesign Chapter 4 first, because the result chapter determines what the thesis is actually proving.
2. Revise Chapter 3 methods based on the final Chapter 4 result structure.
3. Rewrite Chapter 1 contributions and abstract after the method/result story is stable.
4. Update appendices last.
5. Expand the literature review near the end, once the final thesis scope is fixed.

### Proposed Main-Text Structure

Suggested revised structure:

```text
Chapter 1 前言
1.1 研究背景與動機
1.2 研究目標
1.3 研究貢獻

Chapter 2 文獻探討
2.1 微電網能源管理中的強化學習
2.2 安全約束與可部署控制方法
2.3 模擬至實體遷移與現實落差問題
2.4 Optional: data augmentation, robustness validation, SoC/SoH, or hardware-aware battery control

Chapter 3 材料與方法
3.1 系統架構與研究定位
3.2 實體微電網平台與部署限制
3.3 對齊式決策環境設計
3.4 Simulation-vs-hardware validation protocol / placeholder
3.5 CORAL 安全導向控制機制
3.6 Baseline 與消融方法設計
3.7 部署導向控制流程與安全檢查
3.8 硬體感知延伸：流速控制與 SoH 監測
3.9 訓練、驗證與評估設計

Chapter 4 結果與討論
4.1 對齊式模擬環境與實體平台驗證計畫
4.2 Baseline 與消融實驗結果
4.3 CORAL 之安全性與收益權衡分析
4.4 長時長部署前驗證與跨日行為檢查
4.5 硬體感知延伸：流速控制結果（若保留）
4.6 部署限制與討論

Chapter 5 結論
```

Note for `4.1`: for now this should be a placeholder / writing plan, not a result-heavy section. It should state what will be compared between simulation and hardware, such as load/PV support profile, grid demand trend, battery power command, SoC trajectory, and command feasibility. Do not invent measured agreement results before the data is ready.

### Proposed Appendix Structure

```text
Appendix A — CORAL 方法補充說明
A.1 研究動機與現實落差建模
A.2 CRTSN 風險管路與最小投影
A.3 OCC 機會成本評論器
A.4 Adaptive Loop 自調適規則
A.5 與主文部署流程及消融結果之對應關係

Appendix B — 實驗設定與補充方法
B.1 主要模型與 baseline 訓練設定
B.2 Reward 與部署對齊邏輯
B.3 Baseline fairness protocol
B.4 Heuristic rule definition
B.5 Flow-rate-aware environment settings
B.6 SoH predictor and deployment configuration
B.7 Historical teacher-guided training variant
```

Teacher guidance should be moved to the end of Appendix B as a historical or auxiliary training variant.

### Literature Review Strategy

The literature review is currently too short for the expanded thesis scope, but it should be expanded later rather than first. The literature review should follow the final scope instead of deciding it.

Potential literature areas to add near the end:

- Safe RL for power systems and microgrids.
- Rule-based and optimization-based EMS baselines.
- Sim-to-real gaps and domain randomization / data augmentation.
- SoC estimation, Coulomb counting drift, and voltage-based correction.
- Battery SoH estimation and health-aware control.
- Flow battery operation, pump/flow constraints, and flow-rate effects.
- Sensor anomaly, missing data, and deployment diagnostics.

### Chapter 1: Introduction

Keep the sim-to-real and deployability motivation, but update the contributions.

New contribution points should include:

- A deployment-safe RL workflow for a real microgrid platform.
- CORAL-based safety control and comparison against baseline/ablation methods.
- Hardware-aware environment alignment, including mode-1 full-load discharge semantics.
- Flow-rate-aware extension and flow-power-limited modeling.
- SoC calibration and real deployment diagnostic analysis.
- SoH monitoring and partial online SoH integration.

### Chapter 2: Literature Review

Keep the existing sections, but consider adding or strengthening:

- Data augmentation / domain randomization for microgrid RL.
- Battery SoC/SoH estimation and deployment uncertainty.
- Rule-based EMS and heuristic control baselines.
- Flow-rate or battery-operating-condition-aware control if useful.

### Chapter 3: Materials and Methods

This chapter needs major restructuring.

Recommended method sections:

- Physical microgrid platform and deployment constraints.
- State, action, reward, and physical feasibility design.
- CORAL safety framework.
- Baseline and ablation methods:
  - Rule-based heuristic.
  - Standard SAC.
  - SAC + safety penalty.
  - SAC + SafetyNet.
  - SAC + SafetyNet + OCC.
  - CORAL.
  - PPO variants if included.
- Flow-rate-aware environment:
  - 2D action: battery power and flow rate.
  - Minimum active flow.
  - Flow-power-limited battery capability.
  - Pump/motor power modeling.
- SoC tracking and calibration:
  - Coulomb counting.
  - Voltage-based correction / cutoff logic.
  - 15-minute decision interval and boundary-distance uncertainty.
- SoH extension:
  - Offline/online SoH predictor status.
  - SoH-adjusted effective capacity as deployment extension.

Teacher-guided training should be moved to a secondary subsection or appendix.

### Existing Chapter 3 Assessment

The current Chapter 3 has a strong foundation and should not be rewritten from scratch. It already covers:

- system architecture and EMS positioning;
- physical microgrid platform constraints;
- aligned state/action/reward design;
- CORAL concepts;
- deployment-oriented command generation;
- training, validation, and evaluation design.

However, it is missing several items required by the revised Chapter 4:

- baseline and ablation method definitions;
- baseline fairness protocol;
- rule-based heuristic definition;
- updated wording that Chapter 4 now includes actual ablation comparison;
- flow-rate-aware extension;
- SoC calibration and sensor/data-quality considerations;
- SoH predictor and deployment-support extension;
- teacher guidance repositioned as historical/auxiliary, not central.

Recommended Chapter 3 restructuring:

```text
3.1 系統架構與研究定位
3.2 實體微電網平台與操作限制
3.3 對齊式決策環境設計
3.4 CORAL 安全導向控制機制
3.5 部署導向控制流程與命令轉換
3.6 基準方法、消融實驗與公平性設計
3.7 硬體感知延伸：流速控制與 SoH 監測
3.8 訓練、驗證與評估設計
```

Specific changes:

- Keep `3.1`, `3.2`, and `3.3`, but reduce repeated wording where possible.
- Keep `3.4`, but update the final paragraph: Chapter 4 now does include baseline/ablation comparison, so do not say the thesis does not independently compare safety modules.
- Keep deployment flow as `3.5`, but avoid internal hardware labels such as `mode-1`.
- Add new `3.6` for baseline/fairness/heuristic. This is required before Chapter 4 table.
- Add new `3.7` for flow rate and SoH as extensions, not central method.
- Rewrite `3.8` so it reflects both old behavior validation and new baseline/ablation evaluation.

### Chapter 3 Rewrite Strategy

Use a whole-chapter structural rewrite rather than editing isolated paragraphs one by one.

Recommended workflow:

1. First pass: rewrite the entire Chapter 3 structure so all required sections exist in the correct order.
2. Second pass: polish wording, reduce repeated explanations, and standardize Chinese/English terminology.
3. Third pass: check consistency with Chapter 4 tables, figures, and evaluation metrics.

Do not try to make Chapter 3 final before Chapter 4 is stable. The purpose of the first rewrite is to make Chapter 3 support the new result chapter:

- baseline and ablation comparison;
- fairness protocol;
- rule-based heuristic;
- CORAL as the central deployment-safe framework;
- flow-rate control as hardware-aware extension;
- SoC calibration and sensor reliability;
- SoH as deployment-support extension;
- teacher guidance as historical/auxiliary, not central.

### Chapter 4: Results and Discussion

The current Chapter 4 should be mostly rewritten.

Recommended new result flow:

1. **Baseline and Ablation Comparison**
   - Compare heuristic, SAC, penalty, SafetyNet, OCC, CORAL.
   - Use profit, realized violations, raw unsafe attempts, SafetyNet interventions, and stability metrics.

2. **Flow-Rate-Aware Control**
   - Explain why flow-rate action was added.
   - Compare 1D power-only model vs 2D flow-rate-aware model.
   - Discuss flow-power-limited behavior and motor-power assumptions.

3. **Deployment Validation and Diagnostics**
   - Show deployment/logging flow.
   - Confirm command output includes both power and flow rate.
   - Discuss raw data from 6/13 to 6/15 as a real deployment case.

4. **SoC Calibration and Sensor Freeze Case Study**
   - Explain that SoC is software-estimated via Coulomb counting.
   - Discuss 15-minute decision delay and load uncertainty.
   - Explain why apparent over-discharge can arise from stale/frozen sensor feedback and boundary-distance estimation error.

5. **SoH and Deployment Extensions**
   - Present SoH predictor as partial but implemented deployment support.
   - Position it as a future-facing extension for capacity-aware control.

6. **Discussion of Practical Limitations**
   - Small battery capacity.
   - Hardware pump/motor power.
   - Sensor freeze and raw data reliability.
   - Limited long-term deployment data.
   - Data augmentation not yet fully completed.

### Existing Chapter 4 Assessment

The current Chapter 4 is well written, but it reflects an older project stage. It mainly shows simulation-style deployment-prevalidation:

- Single-day behavior validation.
- Different date and low-initial-SoC validation.
- Three-day continuous rollout without SoC reset.
- Discussion of deployability based on those virtual rollouts.
- Limitations and future work.

These sections should not be deleted blindly. They can be reused, but their role should change.

Recommended treatment:

- Keep the figures if they are visually clear and still technically correct.
- Rename the section from final results to **deployment-prevalidation behavior checks**.
- Do not let these old virtual rollouts remain the main thesis evidence.
- Move some older text to Appendix if Chapter 4 becomes too long.
- Use these results after the baseline/ablation comparison, not before, because the thesis now needs to prove CORAL and deployment-safe control first.

Suggested repositioning:

```text
4.1 Baseline and ablation comparison
4.2 CORAL profit-safety trade-off
4.3 Deployment-prevalidation behavior checks
    4.3.1 Representative single-day behavior
    4.3.2 Different date and initial SoC
    4.3.3 Three-day no-reset rollout
4.4 Flow-rate-aware extension
4.5 Real deployment diagnostics and SoC calibration
4.6 Limitations and discussion
```

Old Chapter 4 text that is still useful:

- The explanation of conservative gradual charging.
- The interpretation that the policy uses PV when available and discharges when PV weakens and price becomes high.
- The emphasis that behavior validation is not only about cumulative reward.
- The warning that PV surplus is estimated, not directly measured.
- The limitations about limited data coverage, platform dependence, and longer deployment needs.

Old Chapter 4 text that must be updated:

- Claims that the candidate policy is the main final result.
- Claims that SoC stayed safe without violations as the main evidence, because later deployment diagnostics reveal sensor and SoC-estimation issues.
- Any wording that implies virtual validation alone is enough for deployability.
- The future-work section, because SoH, flow rate, deployment packaging, and sensor-freeze diagnostics are no longer only future work.

### Chapter 5: Conclusion

The conclusion should no longer claim only that a candidate model was validated in virtual rollouts.

New conclusion direction:

> This thesis demonstrates a deployment-safe RL workflow for microgrid energy management, including physical constraint alignment, safety-aware learning, baseline comparison, flow-rate extension, SoC/SoH monitoring, and real deployment diagnostics. The real platform experiments reveal that deployability depends not only on policy performance, but also on sensor reliability, SoC calibration, hardware semantics, and safe command translation.

## Appendix Revision

### Appendix A: CORAL

Keep and update. It is still valuable.

Add links to new ablation/baseline results, so CORAL is not only conceptual.

### Appendix B: Training Settings

Rewrite this appendix.

The current version overstates teacher-guided training as the final candidate model. Instead:

- Separate historical teacher-guided configuration from final evaluation configurations.
- Add baseline/ablation protocol.
- Add flow-rate-aware configuration.
- Add deployment model packaging notes if relevant.

## Figures and Tables to Add

Candidate figures:

- Baseline/ablation comparison table, preferably as a LaTeX table rather than a figure.
- CORAL vs other methods profit/safety plot.
- 1D vs 2D flow-rate-aware behavior plot only if a clear and honest comparison figure is generated later.
- Flow motor power / flow-power-limited conceptual plot.
- Deployment raw-data diagnostic plot for 6/13-6/15 only as a limitation or diagnostic figure, not as a main success result.
- SoC calibration / sensor freeze evidence plot only if clearly framed as a deployment failure/diagnostic case.
- SoH online predictor pipeline diagram if needed.

Candidate tables:

- Baseline fairness protocol.
- Heuristic rule definition.
- Flow-rate environment parameters.
- Deployment limitations and mitigation strategies.
- Data augmentation / robustness validation plan.

## Figure and Evidence Caution Notes

Review these notes before writing or revising Chapter 4.

## Thesis Language Style Notes

Use Chinese as the main thesis language. English terms should be retained only when they are established method names, abbreviations, or table labels that are clearer in English.

Recommended style:

- Keep: `CORAL`, `CRTSN`, `OCC`, `SafetyNet`, `SAC`, `PPO`, `SoC`, `SoH`.
- Translate repeated generic terms:
  - `baseline` -> `基準方法` or `基準模型`
  - `ablation` -> `消融實驗`
  - `policy` -> `控制策略`
  - `reward` -> `回饋函數` or `回饋訊號`
  - `raw attempts` -> `原始不安全嘗試`
  - `realized violations` -> `實際違規`
  - `SafetyNet interventions` -> `安全層介入次數`
  - `demand-censored PV` -> `受負載需求截斷之太陽能量測`
  - `action mapping` -> `動作映射` or `命令轉換`
  - `flow-rate-aware extension` -> `流速感知延伸`
  - `sensor freeze` -> `感測資料停滯`
  - `counterfactual` -> `反事實示意` only if used.
- Avoid internal deployment labels such as `mode-1` in the thesis main text. Use hardware-level descriptions instead.
- Prefer Chinese table headers unless the table is copied from an English conference figure.

The goal is not to remove all English, but to prevent the text from reading like a meeting slide or code note.

### Existing vs. New Figures

Use this rule during thesis writing:

- Existing figures from the old thesis can be reused directly or moved to a different section if they remain technically correct.
- Do not write non-existing figures into the main LaTeX as if they already exist.
- For figures that do not yet exist, first record where they should appear, what they should show, and what file name they should use. Generate them later only if they are truly needed.
- Prefer tables over new figures when the evidence is numerical, especially for baseline/ablation comparisons.

Current existing figures that can be reused:

- `BehaviorValidate.png`: old representative single-day behavior validation.
- `validate0409.png`: old different-date / low-initial-SoC validation.
- `MutliDaysValidate.png`: old three-day no-reset validation. Confirm whether the filename is intentionally `Mutli` rather than `Multi`.

Figures not currently finalized and should not be inserted as required thesis figures yet:

- 1D vs 2D flow behavior figure.
- CORAL profit-safety trade-off plot if the baseline table is sufficient.
- Freeze diagnosis figure, unless used only in limitations/deployment diagnostics.
- Shifted discharge counterfactual figure, unless placed in appendix and clearly labeled illustrative.

### Chapter 4 Figure/Table Checklist

Required and already available:

- `BehaviorValidate.png`
  - Section: deployment-prevalidation behavior checks.
  - Purpose: representative single-day control behavior.
- `validate0409.png`
  - Section: deployment-prevalidation behavior checks.
  - Purpose: different date and low-initial-SoC behavior check.
- `MutliDaysValidate.png`
  - Section: deployment-prevalidation behavior checks.
  - Purpose: three-day no-reset rollout.
  - Check filename spelling before compiling (`Mutli` vs `Multi`).

Required but should be a LaTeX table, not an image:

- `tab:baseline_ablation_metrics`
  - Section: baseline and ablation comparison.
  - Purpose: main quantitative result comparing heuristic, SAC, PPO, SafetyNet, OCC, and CORAL.
  - Current source: seminar benchmark table.

Optional figures to generate only if needed:

- `ch4_flow_rate_extension.png`
  - Section: hardware-aware extension: flow-rate control.
  - Status: not an existing figure. This is only a placeholder name.
  - Purpose option A: show a 2D flow-aware rollout with PV/load ratio, SoC, power command, and flow command.
  - Purpose option B: show a flow-power relationship figure, such as minimum active flow, flow-limited available power, and pump power curve.
  - Use only one clear figure if needed. Do not include both unless Chapter 4 needs a deeper flow-control section.
  - Include only if the figure is clean and assumptions are clearly stated, especially the idealized 10% pump motor power assumption.
- `ch4_deployment_freeze_diagnosis.png`
  - Section: real deployment diagnostics or appendix.
  - Purpose: show 6/13--6/15 sensor freeze / SoC estimation issue.
  - Use only as limitation/diagnostic evidence, not as main success result.

Not recommended for main text:

- `ch4_shifted_discharge_counterfactual.png`
  - Reason: illustrative/counterfactual and partly synthetic.
  - If used, place in appendix only and clearly label as not measured data.

### Rule-Based Heuristic and Fairness Notes

The rule-based heuristic and baseline fairness protocol should be included before or immediately after the baseline/ablation table in Chapter 4, or described in Chapter 3 and referenced in Chapter 4.

Why it is needed:

- The heuristic provides a human-engineered if-else reference.
- It helps show that CORAL is not only better than weak learning baselines, but also competitive against a safe conventional controller.
- The fairness protocol prevents reviewers from asking whether CORAL received more favorable reward shaping, constraints, or evaluation conditions.

Recommended placement:

```text
Chapter 3:
3.x Baseline and fairness protocol
3.x Rule-based heuristic controller

Chapter 4:
4.1 Baseline and ablation comparison
```

Minimal text to include near the baseline table:

> 為確保比較公平，本研究所有方法皆於相同微電網環境、相同資料來源、相同決策週期、相同 SoC 安全範圍、相同功率限制與相同評估指標下進行比較。學習式方法共用相同之核心經濟回饋訊號；各方法僅保留其自身定義所需之安全機制，例如安全懲罰、SafetyNet 投影、OCC 機會成本項或 CORAL 自調適安全流程。

Rule-based heuristic summary:

```text
If SoC <= 22%:
    Charge only during off-peak periods; otherwise idle.
Else if PV / Load >= 0.95 and SoC < 74%:
    Charge using PV surplus.
Else if peak price and SoC > 30% and load is within discharge limit:
    Discharge to cover the load.
Else:
    Idle.
```

In Chinese thesis wording:

> 規則式控制器代表人工設計之條件判斷策略，其目標並非追求最佳化，而是提供一個安全、透明且符合工程直覺的比較基準。該控制器在低 SoC 時避免放電，在太陽能相對充足時充電，並僅於高電價且電池具足夠能量與放電能力時供應負載。

### Baseline/Ablation Result

The baseline/ablation summary was originally a table, not necessarily a figure. It is acceptable, and probably cleaner, to present it as a LaTeX table in Chapter 4.

Use table-first framing:

> The main quantitative evidence is the baseline/ablation metric table.

Optional figures can be added only if they improve readability.

### 1D vs 2D Flow Behavior Figure

There is currently no finalized 1D-vs-2D behavior figure.

Do not invent this figure in the thesis. Only include it if a clean comparison plot is later generated from valid rollouts.

Purpose if used:

- To show that 2D flow-aware control changes the feasible action space.
- To show that flow control tends to keep operation near the minimum active flow due to pump cost.
- To explain why safety improves while profit may decrease slightly.

If no good figure exists, use a table or short discussion instead.

### Freeze Diagnosis Figure

The 6/13--6/15 freeze diagnosis is not a success result. It is a deployment diagnostic / limitation case.

Frame it as:

> Real deployment revealed sensor-freeze and SoC-estimation reliability issues. This supports the need for deployment diagnostics, SoC calibration, and robust data-quality monitoring.

Do not frame it as:

> The model successfully controlled the system during this period.

This case may be included in limitations, deployment discussion, or appendix. It should not dominate the main result chapter.

### Shifted Discharge Counterfactual Figure

The shifted-discharge figure is illustrative and partly synthetic. It should not be used as primary thesis evidence.

Default decision:

- Do not include it in the main thesis unless absolutely needed for an explanatory note.
- If included, place it in appendix or discussion only.
- Clearly label it as illustrative/counterfactual, not measured data.

### General Evidence Rule

Main results should rely on:

- baseline/ablation metrics;
- valid simulation rollouts;
- clean deployment-prevalidation behavior checks;
- flow-rate simulation results with assumptions clearly stated.

Problematic deployment raw data should support limitations and lessons learned, not primary performance claims.

## Baseline Fairness Statement

Use this principle:

> Fairness does not require all methods to have the same internal mechanism. It requires the same task, same environment, same dataset, same action limits, same safety constraints, same training budget where applicable, and same evaluation metrics.

Recommended wording:

> The base economic reward is kept consistent across learning-based methods, while method-specific safety or opportunity terms are enabled only when they define the corresponding baseline. For example, the penalty baseline includes an explicit safety penalty, SafetyNet variants include action projection, and CORAL includes the conformal risk boundary, OCC, and adaptive loop.

## Heuristic Rule Explanation

The heuristic rule should be presented as a human-engineered if-else baseline, not as an optimal controller.

Core rules:

```text
If SoC <= 22%:
    Charge only during off-peak price periods.
    Otherwise stay idle.

Else if PV / Load >= 0.95 and SoC < 74%:
    Charge using available PV surplus.

Else if price is peak and SoC > 30% and load is within discharge limit:
    Discharge to cover the load.

Else:
    Stay idle.
```

Purpose:

> The heuristic provides a transparent reference for how a conventional human-designed controller would behave. It protects low SoC, uses PV surplus for charging, discharges only when economically useful, and remains idle otherwise.

## Data Augmentation and Robustness Ideas

These are planned or future validation ideas unless formal experiments are later added:

- PV scaling.
- Measurement noise.
- Time shifting.
- Initial SoC randomization.
- Optional load scaling.

These should be described as robustness validation or future work unless results are generated.

## Current Thesis Working Copy

The editable thesis source has been extracted from:

```text
C:\Users\Administrator\Downloads\National_Taiwan_University_Thesis_Hao_Yu_Wen.zip
```

Current working copy:

```text
C:\Users\Administrator\Downloads\NTU_Thesis_HaoYu_working
```

Use this folder for direct LaTeX edits. The active Chapter 3 file is:

```text
C:\Users\Administrator\Downloads\NTU_Thesis_HaoYu_working\contents\chapter03.tex
```

Status:

- Chapter 3 has been moved toward the revised deployment-safe RL structure.
- Chapter 1 research objectives and contributions have been updated to reflect CORAL, baseline/ablation comparison, and deployment diagnostics.
- Chapter 5 has been rewritten to conclude around deployment-safe RL, CORAL, limitations, SoC/SoH, and flow-rate extension.
- The Chinese and English abstracts have been updated to match the revised thesis narrative.
- Flow rate is described as a hardware-aware extension, not the main thesis contribution.
- SoH is described as deployment support and future closed-loop extension.
- Baseline fairness and heuristic-rule framing are now reflected in the Chapter 3 method structure.
- Local PDF compilation was not performed because `xelatex` / `latexmk` is not currently available in the shell environment.
- Appendix 2 has been restored to the longer detailed version and updated to match the current main settings: 1000 episodes, 96 steps, SoC range `[0.20, 0.80]`, teacher/BC disabled in the final main line, and OCC weight `0.8`.

Remaining alignment notes:

- Avoid framing the thesis as a single "candidate model" study. Use "deployment-oriented control workflow", "safety-aligned RL framework", or "control architecture" instead.
- In Chapter 4, describe real deployment carefully: deployment currently supports SafetyNet-style action projection and conformal diagnostics, but the fully residual-driven adaptive conformal loop should be treated as incomplete / future work.
- Keep flow rate as a hardware-aware extension. Avoid making it sound like the main thesis contribution or a fully calibrated hardware result.
- Remove LaTeX TODO comments about future figures from thesis source before final compilation. Store figure plans here instead.
- Watch for possible figure path issues caused by Chinese filenames after zip extraction. If compilation fails, rename affected files to English names and update `\includegraphics`.
- Do not include the problematic deployment data freeze / sensor-freeze narrative in the thesis main text for now. If it is needed later, keep it as internal diagnosis or a carefully framed appendix limitation, not as main evidence.
- Chapter 1 should emphasize the ECS-style sim-to-real motivation: good simulation reward does not imply deployability because real platforms impose measurement, state-estimation, command-conversion, and hardware-feasibility constraints.

Tone and terminology consistency rules:

- Use "safety" only in the EMS / device-level sense: SoC bounds, power limits, supply-path mutual exclusion, and command feasibility. Do not imply voltage/frequency transient stability.
- Avoid overclaiming. Prefer "reduce", "mitigate", "align", "improve operational feasibility", and "trade-off" over "guarantee", "solve completely", or "eliminate".
- When discussing result figures that are not physical closed-loop runs, use "deployment-prevalidation", "pre-deployment validation", or "aligned decision environment" rather than generic "simulation" where possible.
- Use "net return", "grid cost reduction", or "economic benefit" consistently. Avoid mixing "profit", "reward", and "return" unless the distinction is intentional.
- Describe flow rate and SoH as preliminary hardware-aware extensions and evidence of modular extensibility, not as complete calibrated hardware studies.

Potential figures to add later:

- CORAL safety metric bar chart based on Chapter 4 baseline/ablation table: actual violations, raw unsafe attempts, and safety-layer interventions.
- Flow-rate extension figure: flow fraction versus available battery power / pump power, clearly labeled as assumed or conservative model.
- Deployment diagnostic figures involving sensor freeze or problematic raw data are deferred for now and should not be added to the main thesis unless explicitly needed.
- Heuristic rule flowchart for Appendix 2 if a visual explanation is needed beyond the pseudo code.

## Immediate Next Steps

1. Start thesis editing from the structure, not paragraph polishing: move physical platform alignment and simulation-vs-hardware validation before strategy comparison.
2. Add a simulation-vs-hardware validation placeholder/protocol only; do not add result claims until measured data is sufficient.
3. In `thesis_sim`, first lengthen validation/evaluation windows and check whether safety, profit, greedy baselines, and RL results separate more clearly.
4. If longer validation still does not separate the story, then consider export/sell-back simulation as a later extension.
5. Keep safety/profit/balanced greedy as the main heuristic baseline family, not a perfect expert rule.
6. Move complex implementation details, rule variants, and preliminary flow/SoH notes into appendices unless they directly support the main claim.

# Margin Sweep / Safety Certifiability Survey Memo

## 1. 核心回答

本研究的 `soc_margin` test 不應稱為一般調參，而應稱為：

```text
safety-layer conservativeness analysis
margin sensitivity analysis
strict-safety certifiability analysis
```

它的目的不是找 reward 最高的 margin，而是回答：

> 一個 policy 需要多大的保守 deployment buffer，才能在 strict SoC accounting 下達到 0 violation？

因此它比較像 safe RL / constrained RL 中的 safety threshold、cost limit、shield conservativeness 或 Lagrange multiplier sensitivity analysis。

本 memo 相關文獻已整理成 BibTeX：

```text
docs/safe_rl_margin_related_references.bib
```

## 2. Boundaries vs Margin

真實安全邊界：

```text
true SoC bounds = 20-80%
```

這是 violation 的唯一判定標準。無論 margin 是多少，只要 SoC 超出 20-80%，就算 strict violation。

Safety margin：

```text
projection lower bound = 20% + margin
projection upper bound = 80% - margin
```

例如：

```text
soc_margin = 0.04
projection bounds = 24-76%
```

這代表 SafetyNet 在執行前更保守地修正 action，盡量不讓下一步 SoC 靠近真實邊界。它不是把真實安全範圍改成 24-76%。

## 3. 可引用文獻方向

### 3.0 使用者補充的四篇文獻評估

| Paper | 推薦程度 | 對本 thesis 的用途 | 注意事項 |
|---|---|---|---|
| `A Review On Safe Reinforcement Learning Using Lyapunov and Barrier Functions` | 高 | 支撐「safe RL 需要 constraint satisfaction / stability certificate，不只是 reward penalty」；可放文獻回顧。 | 它支撐的是 barrier/Lyapunov 安全概念，不直接等於我們的 empirical margin sweep。 |
| `SafeCityLearn: A Benchmark for Safety-Constrained Reinforcement Learning in Distributed Energy Systems` | 很高 | 最直接支撐「能源系統也需要 CMDP、安全成本、SoC bounds、constraint violation 評估」；可放 related work 和 evaluation rationale。 | 它的 benchmark 是 CityLearn / distributed building energy，不是我們的硬體 microgrid；不能直接比較數值。 |
| `A Safety-Aware Direct Control Optimization Method for Industrial Data Center Cooling Systems` / related safe data-center cooling papers | 中高 | 支撐「能源/熱系統中，reward shaping 不夠，常需要 post-hoc rectification / safety-aware direct action correction」。 | data center cooling 不是微電網；適合作方法類比，不適合作主 evidence。若引用 exact title，需確認正式 DOI/出版狀態。 |
| `Energy-Efficient Resilience Scheduling for Elevator Group Control via Queueing-Based Planning and Safe Reinforcement Learning` | 中 | 支撐「安全約束下做 sensitivity / capacity margin / CVaR / action masking/projection」這種評估形式。 | elevator 場景差異大，建議只放 discussion 或用一句話類比，不要當核心文獻。 |

結論：如果只挑 2 篇進主文，優先選 `SafeCityLearn` 和 `A Review On Safe RL Using Lyapunov and Barrier Functions`。如果要補強「reward shaping 不夠，需要 action correction」，可再放 data-center safe cooling。Elevator 那篇可保留在 discussion / appendix。

### 3.1 Safety Layer / Action Projection

Dalal et al., `Safe Exploration in Continuous Action Spaces`, arXiv 2018.

可支持：

- 在 continuous action RL 中加入 safety layer。
- Safety layer 在 action 執行前修正 raw action。
- 對物理系統而言，不能只靠 reward penalty 期待 agent 自己學會安全。

可用在 thesis 的句子：

> The proposed margin sweep follows the safety-layer view in safe RL, where the policy action is treated as a candidate command and a deployment layer projects it into a conservative admissible set before execution.

### 3.2 Runtime Shielding

Alshiekh et al., `Safe Reinforcement Learning via Shielding`, AAAI 2018.

可支持：

- Shield 監控 learner action。
- 當 action 會造成 specification violation 時，shield 進行 correction。
- Safety layer / shield 可以被視為 deployment environment 的一部分，而不是某個 policy 的私有作弊工具。

可用在 thesis 的句子：

> Applying the same safety margin to all learned controllers is analogous to evaluating policies under a shared runtime shield.

### 3.3 Provably Safe RL / Projection-Based Filters

Krasowski et al., `Provably Safe Reinforcement Learning: Conceptual Analysis, Survey, and Benchmarking`.

可支持：

- Safe RL 中常見 action replacement、action projection、action masking。
- Projection-based safety filters 是一類標準 safe RL intervention。
- 不同 safety intervention 會影響 performance，因此要評估 reward-safety trade-off。

可用在 thesis 的句子：

> We report both raw-policy diagnostics and safety-filtered performance because projection-based safeguards can change the executed action distribution and should be evaluated explicitly.

### 3.4 Constraint Threshold / Lagrange Multiplier Sensitivity

Spoor et al., `An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning` / `Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning`.

可支持：

- Safe RL 對 safety-related parameters 很敏感。
- 文獻會用 λ-profile 或 threshold sweep 呈現 reward-safety trade-off。
- 掃描 safety threshold / penalty multiplier 是理解 constrained policy behavior 的合理做法。

可用在 thesis 的句子：

> Similar to cost-limit or Lagrange-multiplier profiles in constrained RL, the margin sweep visualizes how strict safety feasibility and economic return change as the deployment safety layer becomes more conservative.

### 3.5 Projection Conservativeness and Action Aliasing

Recent projection-based safe RL work discusses action aliasing and the difference between safeguarding the policy and safeguarding the environment.

可支持：

- Projection 不一定越強越好。
- 過度保守 projection 可能讓多個 raw actions 對應到同一個 safe action，造成 action aliasing。
- 在多維 action / flow-control setting 中，projection 可能改變 closed-loop trajectory。

可用在 thesis 的句子：

> The non-monotonic margin behavior in the flow-rate setting is consistent with known limitations of projection-based safety filters: overly restrictive projection can alter the closed-loop trajectory and reduce the controller's recovery capability.

## 4. 論文中建議怎麼放

### Method / Evaluation Protocol

可放：

> To avoid treating the safety margin as a method-specific advantage, we evaluate learned policies under two complementary protocols. First, a common-margin protocol applies the same deployment safety layer to all learned policies. Second, a per-method certified-margin protocol records the smallest tested margin required for each policy to satisfy the zero strict-violation gate. This follows the safety-layer and shielding perspective in safe RL, where the runtime filter is part of the deployment environment and its conservativeness should be evaluated explicitly.

### Results

可放：

> The margin sweep should not be interpreted as reward tuning. Instead, it estimates the conservative buffer required for strict-safety certification. A method that reaches zero violation at a smaller margin is easier to certify, while a method that requires a larger margin depends more strongly on deployment-side correction.

### Discussion / Limitation

可放：

> The relationship between margin and safety is not necessarily monotonic. In the flow-rate setting, large margins narrow the internal feasible set and interact with pump losses and flow-dependent power limits. This can reduce recovery capability and cause violations to reappear. Therefore, the current SafetyNet should be interpreted as a practical one-step projection layer rather than a formal invariant controller for the full flow-action dynamics.

## 5. Sell-Back / Grid Export Extension

可以先放進 future work，但不要和現在主結果混在一起。

建議寫法：

> A grid-export extension is a natural next step because it changes the economic objective from cost reduction alone to a combination of self-consumption, storage arbitrage, and feed-in revenue. However, it requires a separate reward model and evaluation protocol, since the action semantics and profit attribution differ from the no-export behind-the-meter setting studied in the main experiments.

原因：

- 目前 `allow_grid_export: false`、`allow_grid_trading: false`。
- 賣電回去會改變 reward/profit definition。
- 可能需要新增 PV curtailment、export revenue、feed-in tariff、battery-to-grid constraints。
- 仍要保留 20-80% strict SoC safety gate。

## 6. 答辯短答

如果被問「margin test 會不會很像隨便做的？」可以答：

> 不是。Margin sweep 在這裡不是為了調最高分，而是為了評估 deployment safety layer 的保守程度。真實安全邊界固定是 20-80%，margin 只是把 SafetyNet 的 internal projection bounds 收緊。這類分析對應 safe RL 中的 safety layer、runtime shield 與 constraint-threshold sensitivity analysis。我們報告 common margin 和 per-method certified margin，是為了區分同一部署環境下的公平比較，以及各方法被 strict-safety certified 所需的最小保守 buffer。

若要接上新找的文獻，可以加一句：

> Similar benchmark and safe-control literature, such as SafeCityLearn and Lyapunov/barrier-function safe RL surveys, also treats safety as explicit constraint satisfaction rather than a reward-only objective. Our margin sweep follows this spirit by evaluating the conservativeness needed for deployment-time safety certification.

# Safe RL / Teacher / CORAL 文獻與使用筆記

本檔整理目前這個專案最適合補進論文、報告或方法章的文獻脈絡，並說明每篇文獻在本專案中「能拿來背書什麼」，避免把不同概念混在一起。

---

## 1. 建議把方法拆成三層

目前專案其實不是單一技術，而是三層混合：

1. `Teacher / imitation warm-start`
   - 用規則型導師先提供可行策略
   - 再用 BC / imitation 幫 policy 起步

2. `RL policy improvement`
   - 在 teacher 的基礎上學更好的經濟行為、跨時段規劃與 deployment-aligned behavior

3. `Runtime safety layer / guard / CORAL`
   - 在部署時對 raw action 做最後一道安全修正
   - 目標是保證 hard constraint 不被直接打穿

論文或報告若要寫清楚，最好把這三層分開敘述，不要只說「我們用了 RL」。

---

## 2. Teacher / Imitation Learning 可補的文獻

### 2.1 Ross, Gordon, Bagnell (2011)

- 題目：
  - `A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning`
- 來源：
  - PMLR 2011
  - <https://proceedings.mlr.press/v15/ross11a.html>
- 核心概念：
  - 單純 behavior cloning 容易有 distribution mismatch 問題
  - learner 自己 rollout 後看到的狀態分布，會和只看 expert demonstration 時不同
  - DAgger 這類作法就是在處理 sequential decision 下的 covariate shift

### 2.2 Battery charging / constrained charging 上的 imitation 類工作

- 可用方向：
  - `Imitation Learning for Agnostic Battery Charging: A DAGGER-Based Approach`
- 用途：
  - 雖然不是微電網完全同一題，但可用來支持
    - 導師示範不是亂加的技巧
    - 在有安全與物理約束的能源系統中，先用 expert / MPC / rule policy 帶 learner 起步是合理做法

### 2.3 在本專案中怎麼用

這組文獻可以用來支持：

- 為什麼需要 heuristic teacher
  - 因為純 RL 初期探索很容易踩到不合理或危險行為
- 為什麼不是只做純 BC
  - 因為部署時 state distribution 會和 demonstration 不同
- 為什麼可以先 BC，再接 RL
  - 因為這是「先給可行策略，再做 policy improvement」的標準套路

### 2.4 論文可用的說法

可寫成：

> 本研究中的 rule-based teacher 主要用來提供安全且可行的初始行為分布，降低純隨機探索在能源管理問題中的低效率與高風險。此做法與 imitation learning / dataset aggregation 文獻的精神一致，即先利用 expert-like policy 緩解 sequential decision 下的 distribution mismatch，再透過後續 RL fine-tuning 學習更佳的長期策略。

---

## 3. Safe RL / CMDP 可補的文獻

### 3.1 Achiam et al. (2017)

- 題目：
  - `Constrained Policy Optimization`
- 來源：
  - ICML 2017, PMLR
  - <https://proceedings.mlr.press/v70/achiam17a.html>
- 核心概念：
  - 不只是設 reward，而是把 safety constraint 明確當成 constraint 來處理
  - 能支持「安全不是單純 penalty，而是約束條件」這個觀點

### 3.2 Chow et al. (2018)

- 題目：
  - `A Lyapunov-Based Approach to Safe Reinforcement Learning`
  - 或相關後續的 `Lyapunov-based Safe Policy Optimization`
- 來源：
  - NeurIPS / OpenReview / arXiv 系列版本
  - 可從 <https://openreview.net/forum?id=Syxgbh05tQ> 開始查
- 核心概念：
  - 用 Lyapunov / state-dependent constraints 來保證政策更新時仍盡量滿足安全性
  - 適合支持「safe RL 不只是部署端擋，訓練端也應該把安全性內化」

### 3.3 Power systems safe RL review (2024)

- 可用 review：
  - `Safe Reinforcement Learning for Power System Control: A Review`
  - `A Review of Safe Reinforcement Learning Methods for Modern Power Systems`
- 用途：
  - 幫你把題目拉回 power / energy domain
  - 支持在電力系統、儲能與微電網中，safe RL 是合理而且必要的研究方向

### 3.4 在本專案中怎麼用

這組文獻可以用來支持：

- 為什麼不能只看 final action 安不安全
  - 若 raw policy 一直提危險動作，只靠 runtime shield 擋，並不代表學到了安全行為
- 為什麼應把 SoC 過充/過放傾向內化到 policy
  - 因為 deployment guard 只能當最後一道防線
- 為什麼該把 `raw-final delta` 視為訓練信號
  - 因為這本質上就是「約束被違反後的投影修正量」

### 3.5 論文可用的說法

可寫成：

> 本研究將 SoC 上下限、禁止過度充放電與部署期可接受動作集合視為安全約束，而非僅以 reward penalty 間接塑形。此觀點與 constrained RL / safe RL 文獻一致，即安全要求應被視為獨立於經濟目標之外的 constraint，而非完全交由 reward 自行內化。

---

## 4. Runtime Safety Layer / Shield 可補的文獻

### 4.1 Dalal et al. (2018)

- 題目：
  - `Safe Exploration in Continuous Action Spaces`
- 來源：
  - arXiv 2018
  - <https://arxiv.org/abs/1801.08757>
- 核心概念：
  - 在 continuous action space 上加入 safety layer
  - 對原始動作做解析或近似修正，確保 constraint 不被違反

### 4.2 Shield / runtime projection 類方法

- 用途：
  - 支持「在部署期最後做 action projection / shield」這件事是合理的
  - 也能說明為什麼 runtime guard 不該完全消失

可補文獻：

- Alshiekh et al. (2018), `Safe Reinforcement Learning via Shielding`
  - shield 監控 learner action，並在可能違反 specification 時修正
  - 可用來支持「安全層是部署執行語意的一部分」
- Krasowski et al. / related survey, `Provably Safe Reinforcement Learning: Conceptual Analysis, Survey, and Benchmarking`
  - 將 provably safe RL 的 action adaptation 分成 action replacement、action projection、action masking
  - 可用來支持本專案的 SafetyNet 屬於 projection / shielding 類 runtime intervention
- Safe RL action projection / safeguard-policy-vs-environment 類近期文獻
  - 討論 projection-based safety filters、safe environment RL、safe policy RL 與 action aliasing
  - 可用來解釋 margin 過大時，projection 可能過度干預 closed-loop behavior

### 4.3 在本專案中怎麼用

這組文獻可以用來支持：

- 為什麼部署端需要 CORAL / SafetyNet
  - 因為真機不能接受 exploration 直接打穿硬限制
- 為什麼 `SoC <= 0.10` 禁止放電、`SoC >= 0.90` 禁止充電這種 guard 合理
  - 因為它們屬於 runtime shield 的 hard safety constraint

### 4.4 但要避免誤寫的地方

不能寫成：

> 只要有 runtime shield，policy 就是安全的。

比較準確的寫法應是：

> runtime shield 能保護系統不被 raw action 直接打穿硬限制，但若 raw policy 長期依賴 shield 修正，仍表示安全性尚未內化，訓練端仍需進一步對齊。

---

## 4.5 Margin Sweep 的正當性

`soc_margin` 不應寫成隨便調出來的 hyperparameter。比較好的定位是：

- `true bounds`：真實安全邊界，例如 SoC 20-80%，所有 violation 都用這個範圍算。
- `projection bounds`：SafetyNet 內部使用的保守範圍，例如 margin 0.04 對應 24-76%。
- `margin sweep`：測試 safety layer 的保守程度，觀察 policy 在多大的 buffer 下可以通過 strict safety gate。

因此 margin sweep 的問題不是：

> 哪個 margin 讓 reward 最高？

而是：

> 哪個 policy 需要多大的 conservative buffer 才能被 strict safety-certified？

這和 safe / constrained RL 中常見的 safety threshold、cost limit、Lagrange multiplier profile 類似。文獻上常透過改變 constraint limit 或 penalty/multiplier，呈現 reward-safety trade-off 與方法對安全門檻的敏感度。

### 4.5.1 為什麼 margin 太大可能反而不好

直覺上 margin 越大越安全，但 projection-based safety layer 不一定單調。原因：

- margin 變大會縮小 policy 可使用的 internal feasible set。
- 眾多 raw actions 可能被投影到相同/相近 safe action，產生 action aliasing。
- projection 改變 closed-loop trajectory，policy 可能進入訓練時不熟悉的 state distribution。
- 在 flow-rate control 中，pump loss、flow-dependent power limit 與 battery projection 互相作用；若 battery action 被限制過頭，SoC recovery 可能反而變差。

可用答辯說法：

> Margin sweep is not arbitrary tuning. It evaluates the conservativeness of the deployment safety layer and estimates the minimum buffer required for strict safety certification. The non-monotonic behavior observed in the flow-rate setting indicates that the current SafetyNet is a practical one-step projection layer, not a formal invariant controller for the full flow-action dynamics.

---

## 5. 目前最適合你這個專案的寫法

### 5.1 方法定位

目前這個專案最合理的定位是：

- `Heuristic teacher + BC warm-start`
- `deployment-aligned RL fine-tuning`
- `runtime SafetyNet / CORAL / deployment guard`

也就是：

> 先用 teacher 讓 policy 站上可行區，再用 RL 學經濟性與跨時段行為，最後在部署端保留 runtime safety layer 作為 hard constraint shield。

### 5.2 目前最值得補的論文引用用途

#### A. 用 Ross et al. 背 teacher / BC

- 支持：
  - sequential decision 下，純 BC 會有 distribution mismatch
  - teacher 不只是為了快，而是為了讓初期行為分布更合理

#### B. 用 Achiam / Chow 背 safe RL

- 支持：
  - 過充、過放、低 SoC 放電這些應視為 constraint
  - 不應只把它們藏在 reward 裡

#### C. 用 Dalal 背 runtime action correction

- 支持：
  - deployment 端對 raw action 做投影 / 修正
  - 在真機系統中這是合理且必要的

#### D. 用 power systems review 背研究場景

- 支持：
  - 微電網 / 儲能控制中的 safe RL 是重要而合理的研究方向

---

## 6. 對你目前方法的具體建議

### 6.1 Teacher 部分

建議不要把 teacher 寫成「人工瞎猜規則」，而是寫成：

- 一個保守、可解釋、符合硬體常識的 heuristic expert
- 主要目的是提供安全可行的初始行為分布
- 不宣稱它本身最優，只宣稱它能降低早期探索風險

### 6.2 Safe RL 部分

下一步比較合理的是：

- 將 `raw_action` 被 `CORAL/guard` 修正的幅度納入 penalty
- 把低 SoC 想放電、高 SoC 想充電，直接當作 unsafe tendency
- 增加靠近邊界的訓練覆蓋，不要只在中間 SoC 區域學

### 6.3 SoC 邊界寫法

論文上建議分成：

- hard bounds：
  - `soc_min = 0.10`
  - `soc_max = 0.90`
- soft / tightened bounds：
  - `soc_soft_min = soc_min + δ_low`
  - `soc_soft_max = soc_max - δ_high`

其中 `δ_low, δ_high` 最好寫成由 replay / deployment uncertainty 去估，而不是直接拍腦袋定 `0.12 / 0.88`。

---

## 7. 對目前 CORAL / SafetyNet 的診斷

這一段很重要，因為它解釋了為什麼目前看起來像「CORAL 的動態邊界消失了」。

### 7.1 目前有在動的部分

目前 deployment 端確實有：

- 啟用 `SafetyNet`
- 記錄 `coral_clipped`
- 記錄 `coral_delta_mW`
- 呼叫 `update_conformal_residual(coral_delta_kw)`
- 累積 `coral_residual_count`

也就是說，**殘差池本身有在長**，不是完全沒資料。

### 7.2 目前看起來沒在動的部分

雖然 `core/safety_net.py` 裡有：

- `_conformal_tube()`
- `current_buffer_ratio`
- `update_buffer_after_episode()`
- `buffer_decay_episodes`
- `buffer_decay_rate`

但在目前 deployment 實際用到的 `SafetyNet.project()` 裡：

- 投影邊界是由 `bounds()` 算出來
- `bounds()` 只用到
  - `safe_soc_min / safe_soc_max`
  - `current_buffer_ratio`
  - `n_step_preview`
- **沒有把 `_conformal_tube()` 接進邊界計算**
- deployment loop 也**沒有呼叫** `update_buffer_after_episode()`

換句話說：

> 現在的 `conformal residual buffer` 有在累積，但沒有真正回饋到 `project()` 的可行邊界裡。  
> 所以你看到的是「CORAL clipping 很多、residual_count 在長」，但不是「邊界會依資料自動變寬/變窄」。

### 7.3 為什麼會有這種落差

這表示目前的程式比較像：

- `Conformal logging + static SafetyNet`

而不是：

- `Conformal-adaptive SafetyNet`

也就是：

- 有 conformal 的殘差概念
- 但尚未真正完成「用殘差分位數去動態 tighten / relax safety bounds」

### 7.4 這件事對分析的影響

因此在解讀目前部署資料時，應該說：

- `CORAL` 現在主要扮演的是 `static projection shield`
- `coral_delta` 可以反映 raw policy 與 safe action 的差距
- 但不能說它已經完成「依近期風險自動調整安全邊界」

### 7.5 如果未來要把它補完整

可考慮的方向是：

1. 將 `_conformal_tube()` 明確接進 `bounds()`  
   - 例如把近期 residual 的高分位數轉成 SoC margin 或 action margin

2. 定義 tube 到邊界的映射  
   - action-space tightening
   - 或 SoC-space tightening

3. 明確決定更新頻率  
   - 每步
   - 每窗格
   - 每 episode / 每日

4. 區分兩種邏輯  
   - `static hard guard`
   - `adaptive conformal margin`

---

## 8. 短期最實用的建議

如果明天之後要繼續改，我建議順序如下：

1. 先補論文定位
   - teacher / imitation
   - safe RL / CMDP
   - runtime shield

2. 在文件中誠實描述目前 CORAL 狀態
   - 目前是 static projection + residual logging
   - 尚未 fully adaptive

3. 之後再決定要不要升級成真正的 conformal-adaptive safety layer

4. 若要優先提升真機穩定度
   - 先讓 policy 少提出會被 guard / CORAL 修正的 raw action
   - 這比先把 CORAL 做得更花俏更重要

---

## 9. 可直接引用的文獻清單

1. Ross, S., Gordon, G., and Bagnell, D. (2011).  
   `A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.`  
   PMLR 15:627-635.  
   <https://proceedings.mlr.press/v15/ross11a.html>

2. Achiam, J., Held, D., Tamar, A., and Abbeel, P. (2017).  
   `Constrained Policy Optimization.`  
   PMLR 70:22-31.  
   <https://proceedings.mlr.press/v70/achiam17a.html>

3. Dalal, G., Gilboa, E., Mannor, S., and Shie Mannor related coauthors (2018).  
   `Safe Exploration in Continuous Action Spaces.`  
   arXiv:1801.08757.  
   <https://arxiv.org/abs/1801.08757>

4. Chow, Y. et al. (2018 / 2019 related versions).  
   `A Lyapunov-Based Approach to Safe Reinforcement Learning` /  
   `Lyapunov-based Safe Policy Optimization for Continuous Control.`  
   OpenReview / arXiv.  
   <https://openreview.net/forum?id=Syxgbh05tQ>

5. Safe RL reviews for power systems (2024)
   - `Safe Reinforcement Learning for Power System Control: A Review`
   - `A Review of Safe Reinforcement Learning Methods for Modern Power Systems`


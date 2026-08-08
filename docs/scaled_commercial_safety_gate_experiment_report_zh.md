# 商用尺度微電網 Strict Safety Gate 實驗報告

## 1. 實驗目的

本實驗的目的，是把原本 P302 實驗中的 PV/load 時序資料放大到較合理的小型商用微電網尺度，讓 EMS 控制策略在收益、安全性、SoC recovery 與部署保護機制上的差異更容易被觀察。

這個實驗不是宣稱 P302 硬體本身就是商用微電網，也不是直接的商用硬體驗證；它是一個 **physically motivated scale-up / design-space analysis**。也就是說，我們保留原始實測資料中的太陽能與負載時間型態，以及每個時間點的 PV/load ratio，但把功率與能量尺度放大到商用情境。

核心問題是：

> 在嚴格的電池 SoC 安全限制下，learning-based EMS 是否仍能相對 expert/greedy baseline 保留經濟效益？

最後的實驗主軸不是「某個 RL 方法無限制地賺最多」，而是：

1. Raw learned policies 可以當作 profit-safety trade-off 的診斷。
2. Profit ranking 必須先通過 strict safety gate 才有意義。
3. 共同 deployment safety layer 應該視為環境 / 部署設定，而不是某個方法的私有優勢。
4. 共同 margin 與 per-method minimum certified margin 回答不同公平性問題，兩者都應報告。
5. 在共同安全層下，learned controllers 可以達到 0 strict SoC violation，且 profit 優於 0-violation greedy baselines。

## 2. Dataset 如何 Scale Up

### 2.1 來源資料

商用尺度資料集由以下 clean-window 資料產生：

```text
data/processed/thesis_coral_stage1_training_clean_windows.csv
```

輸出檔案為：

```text
data/processed/thesis_scaled_commercial_60kw_clean_windows.csv
data/processed/thesis_scaled_commercial_60kw_clean_windows_meta.json
```

來源資料包含原本論文實驗中整理好的實測 PV/load 時序。這裡最重要的設計是：**不改變時間型態，也不改變 PV/load ratio**。

### 2.2 縮放規則

目標 peak load 設定為：

```text
60 kW
```

令：

```text
L_source_peak = 原始資料 peak load
L_target_peak = 60 kW
s = L_target_peak / L_source_peak
```

metadata 中記錄：

```text
L_source_peak = 0.009423 kW
s = 6367.398917542184
```

接著同時放大負載與太陽能：

```text
Consumption_scaled = Consumption_source × s
Solar_scaled       = Solar_source × s
```

因為 `Solar` 與 `Consumption` 使用同一個 scale factor，所以：

```text
Solar_scaled / Consumption_scaled = Solar_source / Consumption_source
```

也就是每個時間點的 PV/load ratio 都被保留。這樣做的好處是，我們沒有任意創造新的太陽能比例，而是保留實測資料中的 PV support pattern，只改變元件額定尺度。

### 2.3 Dataset 統計

| 指標 | 原始 P302 clean-window | scaled commercial |
|---|---:|---:|
| rows | 1556 | 1556 |
| peak load | 0.009423 kW | 60.000 kW |
| peak PV | 0.012612 kW | 80.308 kW |
| mean load | 0.005551 kW | 35.349 kW |
| mean PV | 0.002179 kW | 13.875 kW |
| peak PV/load ratio | 1.338 | 1.338 |
| energy PV/load ratio | 0.393 | 0.393 |

因此 scaled commercial scenario 的 PV peak `80.308 kW` 不是另外指定的，而是因為 peak load 被放大到 `60 kW` 後，保留 PV/load ratio 自然得到的結果。

### 2.4 PV Boolean / PV Sufficiency

若資料中存在 `PV_bool`，會在 scale 後重新計算：

```text
PV_bool = 1 if Solar / Consumption >= 0.8 else 0
```

這與目前模擬中的 `pv_sufficient_ratio_threshold = 0.8` 一致。不過論文寫法仍應避免把 PV/grid 說成二元切換。建議使用：

- PV support
- grid demand
- PV support ratio

而不是直接說「完全由太陽能供應」。

## 3. 商用微電網元件規格

### 3.1 負載與 PV

| 元件 | 規格 | 說明 |
|---|---:|---|
| Peak load | 60 kW | 小型商用 / light commercial 尺度 |
| Mean load | 35.349 kW | 由 scaled dataset 得到 |
| Peak PV | 80.308 kW | 保留 PV/load ratio 後自然得到 |
| Mean PV | 13.875 kW | 由 scaled dataset 得到 |
| PV/load energy ratio | 0.393 | 與原始資料一致 |

### 3.2 Battery Energy Storage System

| 參數 | 數值 |
|---|---:|
| Battery energy capacity | 240 kWh |
| Battery charge power | 60 kW |
| Battery discharge power | 60 kW |
| Duration at rated power | 4 h |
| Battery efficiency | 0.90 |
| True SoC safety bounds | 20-80% |
| Post-step SoC clipping | disabled |

`60 kW / 240 kWh` 對應 4 小時電池，符合 C&I storage 常見的 2-4 小時使用情境。SoC 真實安全邊界修正為 `20-80%`，這是所有 strict violation 的評分標準。

### 3.3 Flow Battery / Pump 規格

目前主結果採用 no-flow action，但 scenario 保留 flow-related 參數供後續延伸。

| 參數 | 數值 |
|---|---:|
| Pump max power | 4.8 kW |
| Pump power fraction | 60 kW 的 8% |
| Pump curve | `P_pump(Q) = P_max × Q^3` |
| `flow_R_base_ohm` | 10.0 |
| `flow_k_R` | 0.5 |
| `flow_V_OCV_charge` | 8.5 |
| `flow_V_OCV_discharge` | 5.6 |
| `flow_min_active_fraction` | 0.15 |
| `flow_power_min_fraction` | 0.15 |

主實驗設定：

```text
use_flow_rate_action: false
```

因此 flow-control 結果應視為後續延伸，不應與本次 no-flow strict safety gate 結果混在同一個主結論中。

### 3.4 Battery Discharge 的硬體對齊規則

目前設定避免把 battery 當成 PV/grid 之外的第三個 partial-assist source：

```text
discharge_auto: true
discharge_mode: solo_only
enforce_solo_discharge_load_limit: true
```

其含意是：

- PV 與 grid 可以同時支援負載。
- Battery discharge 不被視為與 PV/grid 並聯 partial assist。
- 若 battery discharge 有效，它必須在模擬邏輯上能夠獨立服務對應負載。

這符合目前專案 guardrails，也避免論文中錯誤描述「太陽能或市電二選一」。

## 4. 環境與安全定義

### 4.1 Episode / Rollout 設定

| 參數 | 數值 |
|---|---:|
| Time step | 0.25 h |
| Steps per day | 96 |
| Training episodes | 1000 |
| Fair rollout days | 16 |
| Evaluation data | 所有 policy 使用相同 16 rollout days |

使用相同 rollout days 是為了公平比較。不應把 training last-N、不同 held-out 片段、或不同 episode 起點的結果直接混在一起當主表。

### 4.2 Strict SoC Accounting

舊版環境在 action 導致 SoC 越界時，會把 SoC clip 回邊界。這會低估實際部署風險，因為模型一旦把 SoC 推到 81%，若後續因 PV support 無法放電，它可能會在界外停留一段時間。

因此 strict 版本設定：

```text
clip_soc_to_bounds: false
```

若 SoC 超過真實 `20-80%` 邊界，系統會保留這個越界狀態，直到後續 action 將其帶回安全範圍。每一個越界 timestep 都會被計入 strict violation。

### 4.3 Safety Metrics

| 指標 | 意義 |
|---|---|
| `strict_soc_violation_steps` | SoC 位於 20-80% 外的 timestep 數；summary CSV 中為 16 天平均，報告表格改用總數 `x/1536` 呈現 |
| `strict_soc_violation_hours` | SoC 位於 20-80% 外的總時間 |
| `strict_soc_violation_kwh` | 越界幅度的能量積分 |
| `strict_soc_violation_max_kwh` | 單步最嚴重越界幅度 |
| `violations_realized` | 舊式 boundary-hit / event count |
| `violations_attempted` | raw action 在投影前是否會越界 |

主要安全 gate 定義為：

```text
strict_soc_violation_steps = 0
strict_soc_violation_hours = 0
strict_soc_violation_kwh = 0
```

只有通過此 gate 的方法，才適合進入 profit ranking。

## 5. 控制器與 Baselines

### 5.1 Heuristic Baselines

包含：

- Safety-first greedy
- Balanced greedy
- Profit-first greedy

這些是手寫 expert-style baseline。在 20/80 strict evaluation 中，它們皆為 0 strict violation。

### 5.2 Learned Controllers

包含：

- SAC
- SAC + reward safety penalty
- SAC + SafetyNet projection
- SAC + SafetyNet + OCC
- CORAL
- PPO
- PPO + SafetyNet

### 5.3 OCC 的意義

OCC 是 Opportunity Cost Critic。它的意義是讓模型知道「靠近 SoC 邊界」本身會造成未來操作彈性的損失。也就是說，OCC 不只是單純懲罰違規，而是讓 critic 學到邊界附近的 opportunity cost。

論文可以這樣解釋：

> OCC helps the controller internalize the future opportunity cost of consuming SoC safety margin.

以目前結果來看，OCC 的特色不是 raw policy 完全不出界，而是：

- 出界幅度相對小。
- profit 相對穩定。
- 在共同 safety layer 下 profit 表現最好。

這是 OCC 很值得討論的地方。

### 5.4 Common Deployment Safety Layer

為了避免「只有我們的方法有 safety margin」的不公平問題，最後主比較把 SafetyNet margin 視為共同部署環境設定。

主 fair comparison 使用：

```text
true SoC bounds: 20-80%
common deployment margin: 0.04
internal projection bounds: 24-76%
```

所有 learned controllers 都在同一個 deployment safety layer 下評估。因此主表中不再把 `+ common SafetyNet` 寫在每個方法名稱後面，因為它不是方法差異，而是共同環境。

## 6. 分層實驗設計

### 6.0 為什麼要做 Margin Sweep

`soc_margin` 不是隨意調參，也不是把真實安全邊界改掉。真實 SoC 安全邊界始終是 `20-80%`；strict violation 也始終用 `20-80%` 計算。Margin 只作用在 deployment safety layer 的 internal projection bounds：

```text
true bounds: 20-80%
soc_margin = 0.04
projection bounds: 24-76%
```

換句話說，margin 是 safety filter / shield 的保守緩衝。它回答的是：

> 在多大的保守部署緩衝下，各 policy 可以通過 zero strict violation gate？

這個設計可以用 safe RL 文獻中的三個概念支持：

- **Safety layer / action projection**：策略提出 raw action，部署前由 safety layer 投影或修正為可接受 action。這類做法常見於 continuous-action safe RL，例如 Dalal et al. 的 safety layer。
- **Runtime shielding**：shield 在執行前監控並修正可能違反 safety specification 的行為，例如 Alshiekh et al. 的 safe RL via shielding。
- **Constraint-threshold sensitivity / profile**：safe RL 與 constrained RL 常會分析安全門檻、cost limit、penalty multiplier 或 projection conservativeness 對 reward-safety trade-off 的影響。因此 margin sweep 是 certifiability / robustness sensitivity analysis，而不是單純找最高分 hyperparameter。

因此，本研究的 margin sweep 有兩個用途：

1. 找出讓所有 learned policies 通過 strict safety gate 的最小共同部署緩衝。
2. 比較各方法需要多小的 margin 就能達到 0 violation，也就是 certifiability。

需要注意的是，margin 不保證單調改善安全。當 margin 太大時，internal feasible set 會變窄，projection 可能過度干預 policy trajectory。尤其在 flow-rate control 中，flow action、pump auxiliary loss 與有效功率限制會和 battery action projection 互相作用，因此可能出現「中等 margin 最穩，但過大 margin 反而 recovery 變差」的 non-monotonic behavior。這應被寫成 safety layer 與 actuator dynamics 的診斷結果，而不是實驗任意性。

### 6.1 Layer 1: Raw Policy Diagnostics

回答問題：

> 不加共同 margin 時，各 learned controller 自己學到多安全？

這層只作診斷，不作主 profit comparison。

| Method | Net profit | Violation steps | Strict hours | Strict kWh | Gate |
|---|---:|---:|---:|---:|---|
| SAC | -1534.420 | 938/1536 | 14.656 | 189.585 | Fail |
| SAC + reward safety penalty | -1599.707 | 655/1536 | 10.234 | 119.039 | Fail |
| SAC+SN | -1544.922 | 227/1536 | 3.547 | 58.617 | Fail |
| SAC+SN+OCC | -1532.215 | 476/1536 | 7.438 | 2.436 | Fail |
| CORAL | -1520.490 | 345/1536 | 5.391 | 3.013 | Fail |
| PPO | -1616.295 | 716/1536 | 11.188 | 195.548 | Fail |
| PPO+SN | -1595.227 | 304/1536 | 4.750 | 2.102 | Fail |
| Safety-first greedy | -1726.835 | 0/1536 | 0.000 | 0.000 | Pass |
| Balanced greedy | -1766.000 | 0/1536 | 0.000 | 0.000 | Pass |
| Profit-first greedy | -1829.203 | 0/1536 | 0.000 | 0.000 | Pass |

觀察：

- Raw learned policies profit 通常較高。
- 但多數 fail strict safety gate。
- SAC+SN+OCC 與 CORAL 的 strict kWh 明顯小於 raw SAC/PPO，表示 boundary-aware 設計確實有幫助。
- 因此不能只看 profit 判斷 winner。

### 6.2 Layer 2: Margin Sensitivity / Certifiability

回答問題：

> 需要多少共同 safety margin，才能讓 learned controller 達到 0 strict violation？

| Common margin | Projection bounds | CORAL violation steps | CORAL net | SAC+SN+OCC violation steps | SAC+SN+OCC net | PPO+SN violation steps | PPO+SN net | All learned safe? |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.030 | 23-77% | 1/1536 | -1569.518 | 1/1536 | -1551.743 | 1/1536 | -1610.821 | No |
| 0.035 | 23.5-76.5% | 0/1536 | -1573.949 | 1/1536 | -1575.335 | 1/1536 | -1626.169 | No |
| 0.036 | 23.6-76.4% | 0/1536 | -1570.445 | 1/1536 | -1573.615 | 1/1536 | -1626.462 | No |
| 0.038 | 23.8-76.2% | 1/1536 | -1567.028 | 1/1536 | -1563.752 | 1/1536 | -1647.172 | No |
| 0.040 | 24-76% | 0/1536 | -1571.089 | 0/1536 | -1567.566 | 0/1536 | -1612.649 | Yes |
| 0.045 | 24.5-75.5% | 0/1536 | -1574.616 | 0/1536 | -1577.345 | 0/1536 | -1619.409 | Yes |

觀察：

- `0.02` margin 不足以讓所有 learned baselines 都安全。
- `0.04` 是目前測到能讓所有 learned baselines 0 violation 的最小共同 margin。
- CORAL 在 `0.035` / `0.036` 已達 0 strict violation，而 SAC+SN+OCC 仍有小殘留。
- PPO+SN 也在 `0.04` 後達到 0 strict violation，且在共同安全層下 profit 穩定優於 greedy baselines，是重要的安全 learned baseline。
- 因此 CORAL 的安全故事不是「profit 永遠第一」，而是「在較小 safety margin 下更早可被 safety-certify」。

### 6.3 Layer 3: Main Fair Comparison Under Common Safety Layer

回答問題：

> 在所有 learned controllers 都使用同一個 deployment safety layer 後，誰的 profit 較好？

這一層的公平性定義是「同一個部署環境」。也就是說，所有 learned controllers 都面對相同的 true SoC bounds、相同的 projection margin，以及相同 rollout days。這適合回答系統部署者的問題：如果我只允許一個共同安全層，哪個控制器在這個部署設定下最有經濟效益？

共同設定：

```text
true SoC bounds: 20-80%
common safety margin: 0.04
internal projection bounds: 24-76%
```

| Method | Net profit | Violation steps | Strict hours | Strict kWh | Gate |
|---|---:|---:|---:|---:|---|
| SAC+SN+OCC | -1567.566 | 0/1536 | 0.000 | 0.000 | Pass |
| CORAL | -1571.089 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC+SN | -1576.740 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC | -1578.759 | 0/1536 | 0.000 | 0.000 | Pass |
| PPO+SN | -1612.649 | 0/1536 | 0.000 | 0.000 | Pass |
| PPO | -1638.157 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC + reward safety penalty | -1639.182 | 0/1536 | 0.000 | 0.000 | Pass |
| Safety-first greedy | -1726.835 | 0/1536 | 0.000 | 0.000 | Pass |
| Balanced greedy | -1766.000 | 0/1536 | 0.000 | 0.000 | Pass |
| Profit-first greedy | -1829.203 | 0/1536 | 0.000 | 0.000 | Pass |

觀察：

- 在共同 safety layer 下，所有 learned controllers 都 0 strict violation。
- Learned controllers 全部優於 greedy baselines。
- SAC+SN+OCC profit 最好。
- CORAL 非常接近，且在 margin sensitivity 中較早達到 0 violation。
- 各 learned controllers 之間差距不大，因此不應宣稱某方法大幅壓倒性勝出。

### 6.4 Layer 3b: Per-Method Minimum Certified Margin

回答問題：

> 如果每個 RL 方法都允許使用自己剛好達到 0 strict violation 的最小 margin，誰在「各自最小安全化」後表現較好？

這一層的公平性定義不同：它不是同一個部署環境，而是「每個方法都允許做自己的 safety certification」。這對論文很有價值，因為它可以分開比較：

- common margin fairness：所有方法在同一個安全層下部署。
- per-method certification fairness：每個方法用自己所需的最小保守 buffer 部署。

依照目前測過的 margin grid，結果如下：

| Method | Smallest tested 0-violation margin | Projection bounds | Net profit at that margin | Violation steps |
|---|---:|---|---:|---:|
| CORAL | 0.035 | 23.5-76.5% | -1573.949 | 0/1536 |
| SAC | 0.038 | 23.8-76.2% | -1578.059 | 0/1536 |
| SAC+SN | 0.038 | 23.8-76.2% | -1587.711 | 0/1536 |
| SAC+SN+OCC | 0.040 | 24-76% | -1567.566 | 0/1536 |
| PPO | 0.040 | 24-76% | -1638.157 | 0/1536 |
| PPO+SN | 0.040 | 24-76% | -1612.649 | 0/1536 |
| SAC + reward safety penalty | 0.040 | 24-76% | -1639.182 | 0/1536 |

觀察：

- CORAL 需要的最小測試 margin 較小，這支持「比較容易被 safety-certify」的說法。
- SAC+SN+OCC 雖然需要 `0.04` 才在目前 grid 上通過 gate，但在通過 gate 後 profit 最好。
- 因為 `0.038` 出現小幅 non-monotonic 結果，per-method margin 應寫成「smallest tested 0-violation margin」，不要過度宣稱數學上的絕對最小 margin。
- 論文主結果可以同時放 Layer 3 與 Layer 3b：前者是部署公平，後者是 certification fairness。

### 6.5 Layer 4: Flow-Rate Control Extension

Flow-rate control 沒有被納入 no-flow 主結論，這是刻意分開，而不是可以直接混在同一張主表。原因是 flow action 改變了 action space、pump auxiliary loss、有效功率限制與訓練難度，因此它應該是 actuator extension。

我已用相同 20/80 strict protocol 重新訓練 flow-control 的同一組 learned methods，並跑出相同格式的三層評估。完整細節另見：

```text
docs/scaled_commercial_flow_rate_strict_retrain_report_zh.md
```

核心結果如下：

| Flow result | Conclusion |
|---|---|
| Layer 1 raw diagnostics | SAC+SN、SAC+SN+OCC、CORAL raw flow rollout 已達 0 strict violation；SAC、SAC penalty、PPO、PPO+SN 與 flow heuristics 仍 fail。 |
| Layer 2 margin sensitivity | SAC/CORAL family 在 `0.02-0.12` common margins 下可維持 0 strict violation；PPO/PPO+SN 測到 `0.20` 仍未通過。 |
| Layer 3 common safety layer | 尚未找到讓所有 learned methods 都 strict-safe 的共同 margin；若只看 SAC/CORAL family，`0.02` 是目前最小可行 common margin。 |

這代表 flow-control 可以支持「actuator-space extension / stress test」的論文敘述，但不能取代 no-flow 主結論。合理寫法是：

- 主論文結果仍以 no-flow strict safety gate 作為核心。
- flow-control 顯示 SafetyNet/OCC/CORAL 在更複雜 action space 中仍有 certifiability 價值。
- flow-control 也暴露出 PPO-family 與 heuristic baselines 在 flow dynamics 下的 strict safety weakness。
- 不應宣稱所有 learned methods 在 flow-control 下都能被共同 safety margin certified。

## 7. 論文建議寫法

### 7.1 主要貢獻

建議主張：

> 本研究建立一個 strict safety-gated EMS evaluation protocol。結果顯示，在未加共同部署安全層時，learned policies 雖可取得較高收益，但常違反 SoC 安全限制；在加入共同 deployment safety layer 並通過 0 strict violation gate 後，learned controllers 仍能相對 greedy baselines 保留更好的經濟效益。

### 7.2 OCC / CORAL 的定位

可以這樣描述：

- SAC+SN+OCC 在共同 safety layer 下 profit 最高。
- CORAL 在 profit 上非常接近。
- CORAL 在 per-method minimum certified margin 分析中，以較小 tested margin 達成 0 strict violation，顯示較好的 certifiability。
- OCC 顯示 boundary-aware critic 對降低越界幅度與穩定 profit 有幫助。

### 7.3 不建議主張

避免說：

- CORAL 全面大幅勝出。
- RL 在任何情況下都比 greedy 好。
- SafetyNet margin 改變了真實安全邊界。
- PV/grid 是簡單二元切換。

真實安全邊界仍是 `20-80%`；margin 只是在 deployment safety layer 中提前保守投影。

## 8. Reproducibility

建立 scaled dataset：

```powershell
py thesis_sim\code\build_scaled_commercial_dataset.py --target-peak-load-kw 60
```

主要 config：

```text
thesis_sim/configs/thesis_scaled_commercial_60kw_noflow.yaml
thesis_sim/configs/thesis_scaled_commercial_60kw_flow.yaml
```

主要結果：

```text
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_strict_20_80/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_noflow/fair_rollout_common_margin_004/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/fair_rollout_strict_20_80/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/fair_rollout_common_margin_004/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_noflow_tuning/decision_notes.md
```

Canvas：

```text
C:\Users\Administrator\.cursor\projects\c-Users-Administrator-Downloads-HaoYuResearch\canvases\strict-safety-gate-results.canvas.tsx
```

## 9. Limitations

1. 這是 scaled design-space scenario，不是商用硬體實測驗證。
2. PV/load ratio 與時間型態來自 P302 clean-window data，但商用尺度是假設性放大。
3. 目前主結論是 no-flow setting；flow-control 已用相同 strict safety protocol 重新訓練並完成三層評估，但尚未找到 all-method common safety layer，因此應列為 actuator-space extension / stress test，而非取代 no-flow 主結論。
4. Common SafetyNet margin 是部署環境設定，應共同套用於所有 learned controllers；per-method margin 則應明確標示為 certification analysis。
5. 在共同 safety layer 下，各 learned controllers 差距不大；論文重點應放在 strict safety-gated evaluation、certifiability，以及 learned EMS 在安全部署後仍保有經濟價值。

## 9.1 Future Extension: Grid Export / Sell-Back Scenario

目前實驗固定：

```text
allow_grid_export: false
allow_grid_trading: false
```

因此本報告的 profit 結論只代表「不允許賣電回電網」的 behind-the-meter EMS 情境。後續可另建 sell-back / grid-export scenario，讓 EMS 在 PV surplus、battery SoC、TOU price 與 feed-in tariff 之間學習是否充電、自用或賣電。

這個 extension 應作為新情境，而不是直接併入目前主表，原因是：

- reward structure 會改變，profit 來源不再只有 grid cost reduction。
- action interpretation 會改變，需要明確區分 battery discharge to load、grid export、PV curtailment 與 battery charging。
- safety gate 仍應維持 20-80% strict SoC accounting，但 economic objective 會有新 trade-off。
- 必須重新訓練或至少重新 rollout，不應用目前 no-export checkpoint 直接宣稱可支援 sell-back。

論文可先把 sell-back 寫成 future work：

> A grid-export extension can further test whether safety-certified EMS policies remain economically useful when PV surplus can be sold back to the grid. This would require a separate reward model and evaluation protocol because the profit mechanism differs from the no-export behind-the-meter scenario considered in the main experiments.

## 10. 短版摘要

本實驗將 P302 clean-window PV/load 時序以相同倍率放大到 60 kW peak-load 的小型商用微電網情境，保留 PV/load ratio 與時間型態。系統採用 60 kW / 240 kWh 電池，真實 SoC 安全邊界為 20-80%。為避免舊式 clipping 低估風險，本研究使用 strict SoC accounting：一旦 SoC 出界，會持續累計出界時間與出界幅度。

結果顯示，raw learned policies 通常 profit 較高，但多數不通過 strict safety gate。加入共同 deployment safety layer 後，在 `soc_margin = 0.04`、internal projection bounds 為 24-76% 的共同環境下，所有 learned controllers 皆達到 0 strict violation，且 profit 優於 greedy baselines。SAC+SN+OCC 在共同安全層下 profit 最佳，CORAL 非常接近；若改用 per-method minimum certified margin，CORAL 在目前測過的 margin grid 中需要較小 margin 即可達到 0 violation，顯示較好的 safety certifiability。flow-control 目前應作為延伸與限制討論，因為既有 flow checkpoints 在 strict safety protocol 下尚未穩定。

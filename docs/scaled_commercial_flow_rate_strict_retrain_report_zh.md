# Scaled Commercial Flow-Rate Control Strict Retrain Report

## 1. 實驗目的

本實驗把 no-flow strict safety gate 的三層評估流程，完整移植到 flow-rate control action space。這不是沿用舊 flow checkpoints，也不是只在舊模型外面套 margin；本次重新訓練同一組 learned methods，並使用相同 20-80% strict SoC accounting。

本次目標是回答：

> 加入 flow-rate control 之後，同一組 learned controllers 是否仍能在 strict SoC safety gate 下被部署？

## 2. 重訓設定

核心設定：

| Item | Setting |
|---|---|
| Scenario | scaled commercial 60 kW flow-control |
| True SoC bounds | 20-80% |
| Post-step SoC clipping | disabled |
| Flow action | enabled |
| Training episodes | 1000 |
| Rollout days | 16 |
| Total rollout steps | 1536 |
| PPO output | `thesis_sim/outputs/scaled_commercial_60kw_flow/ppo_baselines_strict_v1` |

新增 strict flow configs：

```text
configs/baselines/research/scaled_flow_strict_sac_raw.yaml
configs/baselines/research/scaled_flow_strict_sac_penalty.yaml
configs/baselines/research/scaled_flow_strict_sac_train_safetynet.yaml
configs/baselines/research/scaled_flow_strict_sac_sn_occ.yaml
configs/baselines/research/scaled_flow_strict_ours_full.yaml
```

重訓後 SAC/CORAL checkpoints：

```text
experiments/seminar_ablation_scaled_commercial60_flow_strict_sac_raw/models/final_sac_model.pth
experiments/seminar_ablation_scaled_commercial60_flow_strict_sac_penalty/models/final_sac_model.pth
experiments/seminar_ablation_scaled_commercial60_flow_strict_sac_train_safetynet/models/final_sac_model.pth
experiments/seminar_ablation_scaled_commercial60_flow_strict_sac_sn_occ/models/final_sac_model.pth
experiments/seminar_ablation_scaled_commercial60_flow_strict_ours_full/models/final_sac_model.pth
```

## 3. Layer 1: Raw Policy Diagnostics

回答問題：

> 不加共同 deployment safety layer 時，各 controller 在 flow action space 下是否本身 strict-safe？

| Method | Net profit | Violation steps | Strict hours | Strict kWh | Gate |
|---|---:|---:|---:|---:|---|
| Safety-first greedy | -2004.696 | 281/1536 | 4.391 | 18557.619 | Fail |
| Profit-first greedy | -2457.025 | 413/1536 | 6.453 | 24710.140 | Fail |
| Balanced greedy | -2138.080 | 286/1536 | 4.469 | 18871.490 | Fail |
| SAC | -2583.421 | 138/1536 | 2.156 | 2.081 | Fail |
| SAC + reward safety penalty | -2550.860 | 100/1536 | 1.563 | 1.616 | Fail |
| SAC+SN | -2583.778 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC+SN+OCC | -2552.756 | 0/1536 | 0.000 | 0.000 | Pass |
| CORAL | -2561.899 | 0/1536 | 0.000 | 0.000 | Pass |
| PPO | -2110.194 | 716/1536 | 11.188 | 12868.192 | Fail |
| PPO+SN | -2084.913 | 247/1536 | 3.859 | 4191.527 | Fail |

觀察：

- Flow action space 下，原本 heuristic baselines 不再自動通過 strict safety gate。
- SAC+SN、SAC+SN+OCC、CORAL 在 raw strict flow rollout 中已經 0 strict violation。
- PPO/PPO+SN 的 profit 看起來較好，但 strict violation 很嚴重，因此不能作為 deployment winner。
- Flow-control 的主難點不是單純 profit，而是 action space 增加後 safety behavior 更分裂。

## 4. Layer 2: Margin Sensitivity / Certifiability

回答問題：

> 加入共同 SafetyNet margin 後，哪些 learned methods 可以被 strict-certify？

這裡的 margin sweep 應解讀為 safety layer conservativeness analysis，而不是任意調參。True SoC bounds 固定為 `20-80%`；margin 只是把 deployment projection bounds 收緊，例如 `soc_margin = 0.04` 對應 internal projection bounds `24-76%`。Violation 仍然只依據 true bounds 判定。

此做法對應 safe RL 文獻中的 safety layer、action projection、runtime shielding 與 constraint-threshold sensitivity analysis。也就是說，我們不是在找「哪個 margin 分數最高」，而是在問：

> 在多大的 deployment buffer 下，policy 可以被 strict safety gate certifies？

在 flow-rate control 中，margin sweep 特別重要，因為 action space 多了 flow fraction，並引入 pump auxiliary loss 與 flow-dependent effective power limits。SafetyNet 主要投影 battery power action，並不是完整的 formal invariant controller；因此 projection margin 與 flow dynamics 之間可能產生 non-monotonic interaction。

| Common margin | Projection bounds | SAC/CORAL family all safe? | All learned safe? | CORAL steps | CORAL net | SAC+SN+OCC steps | SAC+SN+OCC net | PPO+SN steps | PPO+SN net |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0.020 | 22-78% | Yes | No | 0/1536 | -2546.764 | 0/1536 | -2535.772 | 234/1536 | -2079.760 |
| 0.030 | 23-77% | Yes | No | 0/1536 | -2540.700 | 0/1536 | -2530.947 | 245/1536 | -2080.277 |
| 0.035 | 23.5-76.5% | Yes | No | 0/1536 | -2537.062 | 0/1536 | -2526.053 | 228/1536 | -2081.091 |
| 0.036 | 23.6-76.4% | Yes | No | 0/1536 | -2535.200 | 0/1536 | -2525.798 | 228/1536 | -2081.202 |
| 0.038 | 23.8-76.2% | Yes | No | 0/1536 | -2536.323 | 0/1536 | -2523.329 | 239/1536 | -2081.983 |
| 0.040 | 24-76% | Yes | No | 0/1536 | -2533.191 | 0/1536 | -2524.264 | 247/1536 | -2081.517 |
| 0.045 | 24.5-75.5% | Yes | No | 0/1536 | -2529.005 | 0/1536 | -2519.592 | 241/1536 | -2081.436 |
| 0.060 | 26-74% | Yes | No | 0/1536 | -2516.230 | 0/1536 | -2509.186 | 227/1536 | -2081.027 |
| 0.080 | 28-72% | Yes | No | 0/1536 | -2430.360 | 0/1536 | -2425.097 | 247/1536 | -2058.503 |
| 0.100 | 30-70% | Yes | No | 0/1536 | -2368.205 | 0/1536 | -2362.735 | 243/1536 | -2040.174 |
| 0.120 | 32-68% | Yes | No | 0/1536 | -2292.194 | 0/1536 | -2291.238 | 235/1536 | -1976.329 |
| 0.160 | 36-64% | No | No | 64/1536 | -2215.304 | 63/1536 | -2219.649 | 245/1536 | -1963.943 |
| 0.200 | 40-60% | No | No | 99/1536 | -2154.856 | 99/1536 | -2153.971 | 234/1536 | -1960.508 |

觀察：

- SAC/CORAL family 在 `0.02` 到 `0.12` 的共同 margins 下都可以保持 0 strict violation。
- PPO/PPO+SN 即使 margin 提高到 `0.20`，仍沒有通過 strict gate。
- margin 並非越大越穩；在 `0.16` 之後，SAC/CORAL family 反而開始重新出現 strict violation，代表 flow-control 下 projection 與 flow/pump dynamics 有 non-monotonic interaction。
- 因此 flow-control 的 certifiability 結論比 no-flow 更複雜。

這個 non-monotonic 現象可以這樣解釋：

- Margin 太小時，projection 不夠保守，可能擋不住下一步或後續 trajectory 的 SoC drift。
- Margin 適中時，projection 提供足夠 buffer，同時保留 recovery 空間。
- Margin 太大時，internal feasible set 過窄，battery action 被過度限制；若 flow/pump dynamics 仍持續造成能量損失或有效功率限制，policy 反而更難把 SoC 帶回安全區。
- Projection-based safety filters 可能產生 action aliasing：許多不同 raw actions 被壓到相同/相近 safe action，導致 closed-loop behavior 變得不像訓練時學到的 policy。

因此，flow-control 結果應寫成：適中 margin 可 certify SAC/CORAL family，但目前 safety layer 對完整 flow-action dynamics 不是 formal guarantee；過大 margin 反而揭示了 projection 與 actuator dynamics 的互動限制。

## 5. Layer 3: Main Fair Comparison Under Common Safety Layer

回答問題：

> 在共同 deployment safety layer 下，能否得到和 no-flow 一樣的「所有 learned methods 皆 strict-safe」主表？

答案是：目前不能。測試到 `soc_margin = 0.20` 為止，PPO 與 PPO+SN 仍未通過 strict gate，因此 **flow-rate control 沒有找到 all-method common safety layer**。

若只看 SAC/CORAL family，最小共同可行 margin 是 `0.02`：

| Method | Net profit | Violation steps | Strict hours | Strict kWh | Gate |
|---|---:|---:|---:|---:|---|
| SAC + reward safety penalty | -2532.889 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC+SN+OCC | -2535.772 | 0/1536 | 0.000 | 0.000 | Pass |
| CORAL | -2546.764 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC+SN | -2567.632 | 0/1536 | 0.000 | 0.000 | Pass |
| SAC | -2567.729 | 0/1536 | 0.000 | 0.000 | Pass |
| PPO+SN | -2079.760 | 234/1536 | 3.656 | 4369.487 | Fail |
| PPO | -2024.836 | 694/1536 | 10.844 | 13433.345 | Fail |
| Safety-first greedy | -2004.696 | 281/1536 | 4.391 | 18557.619 | Fail |
| Balanced greedy | -2138.080 | 286/1536 | 4.469 | 18871.490 | Fail |
| Profit-first greedy | -2457.025 | 413/1536 | 6.453 | 24710.140 | Fail |

這一層的 thesis wording 應該很保守：

> Under flow-rate control, the SAC/CORAL-family controllers can be made strict-safe under a common deployment margin, but the full learned baseline set cannot yet be certified because PPO-based policies remain unsafe. Therefore, the flow-control result should be presented as an actuator-space extension and stress test, not as a replacement for the no-flow main result.

## 6. Thesis Interpretation

可主張：

- Flow-rate control 使 action space 更複雜，安全性不再像 no-flow 那樣容易由 common margin 統一解決。
- SAC+SN、SAC+SN+OCC、CORAL 在 raw flow strict rollout 已達到 0 strict violation，顯示 SafetyNet/OCC/CORAL 設計在 flow action space 仍有安全價值。
- PPO/PPO+SN 雖然 profit 較高，但無法通過 strict gate，因此不能當 deployment-safe winner。
- Flow-control 應作為 thesis extension：它展示了更複雜 actuator dynamics 下，安全學習與 certification 更困難。

不應主張：

- flow-control 全面優於 no-flow。
- 所有 learned methods 都能在 flow-control 下被共同 margin safety-certify。
- PPO profit 較高就代表 PPO 最好；它沒有通過 strict safety gate。

## 7. Reproducibility

產生 configs：

```powershell
py thesis_sim\code\build_scaled_flow_strict_configs.py --episodes 1000 --seed 20260628
```

主要 rollout outputs：

```text
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_fair_rollout_strict_20_80/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_002/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_0030/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_0035/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_0036/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_0038/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_004/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_0045/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_006/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_008/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_010/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_012/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_016/fair_rollout_summary.csv
thesis_sim/outputs/scaled_commercial_60kw_flow/strict_retrain_common_margin_020/fair_rollout_summary.csv
```

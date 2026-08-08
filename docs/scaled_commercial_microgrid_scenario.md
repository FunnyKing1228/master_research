# Scaled Commercial Microgrid Scenario

## 目的

原始 P302 平台的功率尺度太小，expert/greedy/RL 之間的經濟差距很容易被硬體限制、量測雜訊與單日狀態影響壓縮。本 scenario 將同一份 clean-window PV/load 時序放大到小型商用微電網尺度，讓 EMS 的能源搬移、SoC recovery、SafetyNet projection 與 flow-rate control 差異更容易被觀察。

這不是宣稱實體 P302 硬體已變成商用系統，而是建立一個 **physically motivated scale-up / design-space analysis**：

> 保留實測資料中的太陽能支援與負載時間型態、保留 PV/load ratio，僅改變元件額定尺度，檢查不同 EMS 方法在較合理的商用能量容量下是否能拉開差距。

## 外部規格依據

本 scenario 採用小型商用 / C&I microgrid 的合理範圍：

- DOE commercial reference / prototype buildings 提供 small office、medium office、retail、warehouse 等標準商用建築 archetype，可作為商用負載情境來源。
- 小型商用 solar+storage 示範案可見約 `75 kW` PV、`29 kW / 64.5 kWh` battery 的實地配置。
- C&I battery 常用 `2-4 h` duration 做 demand-charge reduction、TOU arbitrage 或短期 resilience。
- Commercial battery 系統常見從 `100-500 kWh` 到 `1 MWh`，依 peak shaving 或 backup duration sizing。
- Vanadium / redox flow battery 文獻中 pump/auxiliary power 常約為額定功率的 `8-15%`；本 scenario 採 lower-bound `8%` 作為 pump max power。

參考來源：

- DOE Commercial Reference Buildings: https://www.energy.gov/cmei/buildings/commercial-reference-buildings
- DOE / PNNL Prototype Building Models: https://www.energycodes.gov/prototype-building-models
- CEC small commercial solar+storage demonstration report: https://www.energy.ca.gov/sites/default/files/2024-03/CEC-500-2024-018.pdf
- Commercial battery sizing discussion: https://www.surgepv.com/blog/commercial-battery-storage-sizing
- Vanadium flow battery operational experience: https://www.mdpi.com/2313-0105/5/3/52

## 資料縮放方法

來源資料：

```text
data/processed/thesis_coral_stage1_training_clean_windows.csv
```

輸出資料：

```text
data/processed/thesis_scaled_commercial_60kw_clean_windows.csv
data/processed/thesis_scaled_commercial_60kw_clean_windows_meta.json
```

縮放規則：

1. 讀取原始 `Solar` 與 `Consumption`。
2. 設定目標 peak load 為 `60 kW`。
3. 使用同一個 scale factor 同時放大 `Solar` 與 `Consumption`。
4. 因為兩者同倍率放大，所以每個時間點的 PV/load ratio 保持不變。
5. `PV_bool` 由 `Solar / Consumption >= 0.8` 重新計算，與既有 PV support sufficiency 定義一致。

目前 dataset summary：

| 指標 | 原始 P302 clean-window | scaled commercial |
|---|---:|---:|
| peak load | `0.009423 kW` | `60.000 kW` |
| peak PV | `0.012612 kW` | `80.308 kW` |
| mean load | `0.005551 kW` | `35.349 kW` |
| mean PV | `0.002179 kW` | `13.875 kW` |
| peak PV/load ratio | `1.338` | `1.338` |
| energy PV/load ratio | `0.393` | `0.393` |

## 元件定義

主要 config：

```text
thesis_sim/configs/thesis_scaled_commercial_60kw_flow.yaml
thesis_sim/configs/thesis_scaled_commercial_60kw_noflow.yaml
```

建議主 scenario：

| 元件 | 規格 | 理由 |
|---|---:|---|
| Peak load | `60 kW` | 小型商用 / light commercial 可解釋尺度，且由原始資料 peak load 等比例縮放。 |
| Peak PV | `80.3 kW` | 由同倍率縮放自然得到，保留 PV/load ratio。 |
| Battery power | `60 kW` | 可在 PV 不足時服務 peak load；符合 solo battery-supply guard。 |
| Battery energy | `240 kWh` | `60 kW x 4 h`，符合 C&I 常見 2-4 小時 battery duration。 |
| SoC bounds | `0.20-0.90` | 保留安全 headroom，同時給商用儲能較大可用容量。 |
| Battery efficiency | `0.90` | 商用系統含 power electronics / auxiliary 的保守 round-trip 近似。 |
| Pump max power | `4.8 kW` | `60 kW x 8%`，對應 flow battery auxiliary lower-bound。 |
| Pump curve | `P_pump(Q) = 4.8 kW x Q^3` | 沿用現有 simulator 的 cubic hydraulic relation。 |

## 論文寫法

第 3 章可以說明：

> 在 P302 實體平台完成 command-level / policy-in-loop sanity validation 後，本研究進一步建立一個商用尺度的 hypothetical design-space scenario。此 scenario 不改變實測 PV/load ratio，也不宣稱代表特定地點之真實太陽能資源，而是用相同時間型態測試 EMS 在較大能量與功率尺度下的安全、收益與 recovery trade-off。

第 4 章可以比較：

- no-flow commercial baseline：SAC、SAC+SafetyNet、SAC+SafetyNet+OCC、CORAL、expert/greedy。
- flow-control commercial extension：同一組方法，但 action space 多一個 flow-rate control。
- held-out / historical robustness：使用 scaled held-out 或 clean-window segments，但需標註為 scaled scenario validation，不是 P302 sim-to-real validation。
- recovery-aware metrics：final SoC、low-SoC duration、recovery-adjusted net profit。

第 5 章可以回扣：

- 若 CORAL 贏：說明在較大商用尺度下，learning-based EMS 的策略空間才足以展現價值。
- 若仍未贏：說明 strong expert/greedy 在單電池、單 PV/load stream 下仍很強，未來需更複雜設備組合、更多不確定性或 expert imitation/warm start。

## 執行指令

建立 scaled dataset：

```powershell
py thesis_sim\code\build_scaled_commercial_dataset.py --target-peak-load-kw 60
```

快速檢查 no-flow config：

```powershell
py thesis_sim\code\run_eval.py --config thesis_sim\configs\thesis_scaled_commercial_60kw_noflow.yaml --steps 8 --no-write
```

快速檢查 flow-control config：

```powershell
py thesis_sim\code\run_eval.py --config thesis_sim\configs\thesis_scaled_commercial_60kw_flow.yaml --steps 8 --no-write
```

後續正式訓練時，建議先跑 short smoke，再跑 500-1000 episodes；不要直接把 scaled scenario 結果寫成實體 P302 驗證。

# pymgrid 可回售電力延伸實驗報告

## 為什麼改用 pymgrid

前一版 sell-back 實驗直接使用本研究實體平台整理後的資料集。該資料集適合 behind-the-meter、自用優先、不可回售電力的情境，因為其中 `Solar` 欄位較接近實際量測或平台取用到的 PV 輸出，而不一定代表當下 PV 可發出的最大潛力。因此，該資料集缺乏完整的 PV surplus / curtailed PV potential，不適合作為可回售電力情境的主要證據。

本次改用 `pymgrid25 microgrid_4` 建立獨立 sell-back benchmark。pymgrid 的 PV production 來自 TMY3 / NREL 類型資料，load profile 來自 OpenEI building load profiles，因此較適合被解讀為 PV production potential 與建築負載需求。

## 資料與設定

- 來源：`pymgrid25 microgrid_4`
- 轉換後資料：`data/processed/pymgrid25_mg4_sellback_60kw.csv`
- metadata：`data/processed/pymgrid25_mg4_sellback_60kw_meta.json`
- 時間解析度：1 小時
- 評估長度：365 個 24 小時視窗
- 尺度：peak load 60 kW，PV 與 load 用同一倍率縮放以保留原始 PV/load 關係
- 電池：240 kWh，60 kW charge/discharge
- SoC 安全範圍：20% 至 80%，不裁切 SoC
- 回售價格：當下 TOU 購電價格的 50%

資料特性：

- peak load：60.0 kW
- peak PV：88.2 kW
- PV/load energy ratio：0.742
- PV surplus 發生比例：22.8% of hours
- 平均 PV surplus：約 117.2 kWh/day

這代表此資料集確實具有可回售電力情境所需的 PV surplus。

## 經濟指標定義

為避免將「回售收入」、「帳單成本」與「控制效益」混在一起，本延伸實驗將經濟結果分成三個指標：

- `Profit`：回售收入減去購電成本，即 `export revenue - grid import cost`。若為負值，代表場域整體仍是淨用電戶，售電收入不足以抵過購電成本。
- `Net bill`：實際電費帳單，即 `grid import cost - export revenue`。這是 `Profit` 的相反數，越低越好。
- `Benefit`：相對於不使用電池控制的基準帳單改善量，即 `baseline net bill - controller net bill`。若為正值，表示 EMS 控制相較於同一資料、同一電價與同一回售價格下的無電池基準更省錢。

本資料集的無電池基準帳單為 648.450 元/day。因此，sell-back 情境下不必要求 `Profit` 必然為正；比較合理的主經濟判讀是 `Benefit` 是否為正，以及在安全限制下 `Net bill` 是否降低。

## No-flow 結果

| Method | Profit | Net bill | Benefit | Export kWh | Export rev | Strict steps | Strict h | Strict kWh | Max kWh | Min SoC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Safety-first greedy | -454.586 | 454.586 | 193.864 | 117.243 | 243.268 | 0.000 | 0.000 | 0.000 | 0.000 | 0.448 |
| Profit-first greedy | -457.378 | 457.378 | 191.072 | 117.575 | 244.038 | 0.000 | 0.000 | 0.000 | 0.000 | 0.486 |
| Balanced greedy | -465.775 | 465.775 | 182.675 | 117.305 | 243.441 | 0.000 | 0.000 | 0.000 | 0.000 | 0.470 |
| SAC | -429.146 | 429.146 | 219.304 | 117.789 | 243.581 | 10.808 | 10.808 | 80.444 | 40.341 | 0.181 |
| SAC + reward safety penalty | -429.317 | 429.317 | 219.134 | 117.777 | 243.594 | 10.756 | 10.756 | 81.587 | 39.458 | 0.181 |
| SAC + SafetyNet projection | -460.858 | 460.858 | 187.592 | 117.715 | 243.470 | 2.589 | 2.589 | 2.391 | 33.150 | 0.209 |
| SAC + SafetyNet + OCC | -460.057 | 460.057 | 188.393 | 117.730 | 243.512 | 2.742 | 2.742 | 2.773 | 35.567 | 0.207 |
| CORAL | -460.818 | 460.818 | 187.632 | 117.738 | 243.519 | 2.534 | 2.534 | 2.625 | 34.811 | 0.208 |
| PPO | -410.524 | 410.524 | 237.926 | 117.093 | 242.730 | 9.082 | 9.082 | 72.069 | 44.113 | 0.184 |
| PPO + SafetyNet | -400.650 | 400.650 | 247.800 | 108.386 | 224.765 | 2.592 | 2.592 | 3.084 | 32.120 | 0.239 |

No-flow 結果顯示，所有方法的 `Benefit` 皆為正，代表可回售資料集下 EMS 控制確實能降低相對於無電池基準的淨帳單。不過，learned methods 雖然帳單改善更大，多數仍會產生 strict SoC violation。若要求 0 strict violation，三個 greedy baseline 仍是安全基準。CORAL 與 SafetyNet 類方法可以降低 violation energy，但沒有完全消除 violation。

## Flow-control 結果

| Method | Profit | Net bill | Benefit | Export kWh | Export rev | Strict steps | Strict h | Strict kWh | Max kWh | Min SoC | Pump Wh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Safety-first greedy | -601.149 | 601.149 | 47.301 | 116.238 | 240.883 | 3.956 | 3.956 | 7112.543 | 4925.810 | -4.766 | 1146.930 |
| Profit-first greedy | -895.516 | 895.516 | -247.066 | 116.950 | 242.291 | 4.126 | 4.126 | 7801.319 | 4919.304 | -5.126 | 3757.346 |
| Balanced greedy | -696.375 | 696.375 | -47.925 | 116.390 | 241.222 | 4.123 | 4.123 | 7886.866 | 4967.665 | -5.196 | 1474.710 |
| SAC | -1060.873 | 1060.873 | -412.423 | 112.750 | 234.869 | 17.707 | 17.707 | 121.822 | 22.679 | 0.557 | 11660.078 |
| SAC + reward safety penalty | -1060.731 | 1060.731 | -412.281 | 112.741 | 234.801 | 17.696 | 17.696 | 117.812 | 22.966 | 0.557 | 11658.085 |
| SAC + SafetyNet projection | -1033.795 | 1033.795 | -385.345 | 108.573 | 227.339 | 0.000 | 0.000 | 0.000 | 0.000 | 0.554 | 25190.599 |
| SAC + SafetyNet + OCC | -1032.740 | 1032.740 | -384.290 | 108.536 | 227.273 | 0.000 | 0.000 | 0.000 | 0.000 | 0.554 | 25151.499 |
| CORAL | -1033.494 | 1033.494 | -385.044 | 108.348 | 226.847 | 0.000 | 0.000 | 0.000 | 0.000 | 0.554 | 25458.708 |
| PPO | -663.501 | 663.501 | -15.051 | 117.179 | 242.678 | 0.000 | 0.000 | 0.000 | 0.000 | 0.515 | 2152.425 |
| PPO + SafetyNet | -656.356 | 656.356 | -7.905 | 117.179 | 242.678 | 0.000 | 0.000 | 0.000 | 0.000 | 0.515 | 1068.029 |

Flow-control 結果顯示，SafetyNet-based SAC/CORAL 方法達到 0 strict violation，但 pump energy 過高，導致 `Net bill` 大幅上升且 `Benefit` 轉為明顯負值。PPO 與 PPO + SafetyNet 在此初版設定中也達到 0 strict violation，且 pump energy 較低，因此經濟表現優於 SAC/CORAL flow-control 方法；但它們的 `Benefit` 仍略為負值，代表目前 flow-control reward 尚未調整到能在安全與帳單改善上同時勝過無電池基準。

## 初步結論

這版 pymgrid sell-back 實驗修正了資料問題：PV 欄位可被解釋為 production potential，且確實存在顯著 PV surplus。因此，此設定比前一版直接使用實體平台資料的 sell-back 嘗試更公平、更可信。

但目前結果不應被寫成 CORAL 在 sell-back 情境大幅勝出。較合理的解讀是：

- no-flow 下，所有方法皆有正 `Benefit`，表示可回售情境下電池控制確實能降低淨帳單；但 learned policies 的 strict safety 仍未完全滿足。
- flow-control 下，SafetyNet/CORAL 可以達到 strict safety，但目前 pump usage 太高，使 `Benefit` 明顯為負。
- PPO 類方法在 flow-control sell-back 初版中表現保守，反而形成較好的 zero-violation / net bill trade-off，但仍未勝過無電池基準。

因此，pymgrid sell-back 可作為論文的延伸實驗，但目前較適合呈現為「可回售情境下的初步 benchmark 測試」，而不是主結論。

## 後續建議

若要讓此實驗更有論文價值，建議下一步集中在 flow-control reward 與 action regularization：

- 提高 pump energy penalty 或加入 flow-use regularization。
- 對 SAC/CORAL flow-control 重新訓練，避免為了安全而過度使用高流速。
- no-flow 可再測試 SafetyNet margin 或更嚴格 OCC threshold，看能否消除少量 strict violation。
- 也可測試不同 feed-in tariff ratio，例如 0.3、0.5、0.8，確認結論是否穩定。

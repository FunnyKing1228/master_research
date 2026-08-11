# 實驗目錄使用說明

本頁是人員操作簡版；AI／維護者請讀 [`README_AI.md`](README_AI.md)。

## 用途

`experiments/` 保存本機訓練與驗證產物，例如封存設定、log、checkpoint、單日及連續多日結果。這些產物預設不進 Git；可攜的實驗定義應放在 `configs/`，代表狀態則登記於 [`docs/handover/experiment_inventory.md`](../docs/handover/experiment_inventory.md)。

目前候選必須依序通過單日、連續 3 日與連續 5 日 rollout。單日只可作 behavior check；3/5 日若有 SoC 漂移、跨日退化或硬體語意錯誤，即不得稱為 thesis-ready。

## 開始命令

從 repository 根目錄執行；日期與 dataset 必須換成實際存在且連續的資料：

```powershell
py core\train_sac_microgrid.py --config configs\experiments\p302\config_p302_v22_flow_power_limited.yaml --name <唯一實驗名稱>

py data\scripts\figures\generate_selected_day_validation.py `
  --experiment <實驗名稱> --model best_sac_model.pth `
  --dates "<存在的日期>" --output-subdir selected_day_validation_best

py data\scripts\figures\plot_hybrid_model_window.py `
  --experiment <實驗名稱> --model best_sac_model.pth `
  --dataset data\processed\<資料檔>.csv `
  --start-date "<起始日 00:00:00>" --days 3 `
  --output-subdir crossday_3_best --style single_day_thesis

py data\scripts\figures\plot_hybrid_model_window.py `
  --experiment <實驗名稱> --model best_sac_model.pth `
  --dataset data\processed\<資料檔>.csv `
  --start-date "<起始日 00:00:00>" --days 5 `
  --output-subdir crossday_5_best --style single_day_thesis
```

`best_sac_model.pth` 與 `final_sac_model.pth` 都要在相同資料、日期、seed 與 safety 設定下比較；不可只憑名稱選用。歷史 v16sp 曾經人工驗證後選 final，目前 v22 部署則選 best。

## 輸入與輸出

- 輸入：`configs/` 下可攜 YAML、其指定 dataset、seed、驗證日期／窗口，以及明確的 checkpoint。
- 輸出：`experiments/<name>/configs/`、`logs/`、`models/`、`results/`；皆為本機衍生產物，Git clone 不保證存在。
- 清冊：每輪記錄設定、checkpoint 類型、單日／3 日／5 日結果、淘汰原因與檔案是否實際存在；見[實驗清冊](../docs/handover/experiment_inventory.md)。

## 下一步

先固定 PV support／blocking 邏輯與 `solo_only` 放電語意，再做短訓練與單日 smoke；通過後比較 best/final，接著跑連續 3 日、5 日，最後才更新清冊、論文圖或 release manifest。`v16s_aggr1000`、`v16s_crossday3_warm200_v7` 與所有 `partialassist`／`partial_assist`／`pa_antidrift` 已淘汰，只可追溯，不可回復為候選。

## 三個禁止事項

1. 禁止提交任何 checkpoint、results、CSV、圖片、log 或整個實驗資料夾；只有本 README 與 `README_AI.md` 可被追蹤。
2. 禁止只看單日或只看 `best`／`final` 檔名下結論；必須以相同條件完成單日、連續 3 日與連續 5 日驗證。
3. 禁止把 PV/grid 寫成二選一，或讓 battery 成為第三個 partial-assist 來源；battery 放電必須符合可獨立承擔負載的 `solo_only` 語意。

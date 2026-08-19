# Windows 部署打包說明

本頁是人員操作簡版；AI／維護者請讀 [`README_AI.md`](README_AI.md)。

## 用途

`packaging/` 保存 GUI release 的 PyInstaller 規格；正式、可重現的來源鏈是 repository source 與 checkpoint 經 `_deploy.ps1` 建置，不是 `Microgrid_AI` 裡的歷史手工 build。

## 開始命令

在 repository 根目錄執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\_deploy.ps1
```

建置前先核對 checkpoint、封存 experiment config、source 狀態與 [release manifest](../docs/handover/release_manifest.md)。目前正式來源位置如下：

- GUI source：`gui/ai_control_gui.py`
- 控制 source：`control/run_deployment.py`、`control/io_protocol.py`；人員版協定說明見 [`../control/README.md`](../control/README.md)
- PyInstaller spec：`packaging/build_release.spec`
- 建置入口：`_deploy.ps1`
- 暫存 onedir：`dist/P302_AI_GUI/`
- release 留存根目錄：`%USERPROFILE%\Downloads\Microgrid_AI`
- 目前 release：`%USERPROFILE%\Downloads\Microgrid_AI\release_v22_flow_power_limited`

## 輸入與輸出

- 輸入：目前 `_deploy.ps1` 指定的 `experiments/v22_flow_power_limited_gpu300/models/best_sac_model.pth`、封存 config、GUI/control/core source 與 SoH artifacts。
- 輸出：完整 `dist/P302_AI_GUI/` onedir，再複製到上述 `Microgrid_AI\release_v22_flow_power_limited`。不可只交付 exe；`_internal/`、runtime、模型與設定缺一不可。
- `Microgrid_AI` 中的 `build_*.spec`、`build_*.bat`、`package.ps1`、`dist_v22*`、舊 release 與人工 overlay 只供比對／緊急驗證，不是 source SSOT。

## 打包前先確認

- 廠商 P302 軟體另外安裝在實驗用電腦，不會包進 GUI。
- 電池電壓量測修正已搬回正式程式碼，但還沒有從目前 source 重新打包並完成實驗電腦驗收。
- 因此重新打包時，先輸出並測試新版本，不要直接刪除或覆蓋目前可用的 release。

若只是使用既有 GUI，不需要處理這項差異。若要重新打包，請先讓 AI／維護者依 [`README_AI.md`](README_AI.md) 與 [release manifest](../docs/handover/release_manifest.md) 完成比對。

## 下一步

先保留目前 release，再建立新的測試包。確認 GUI、模型、資料讀寫與停止功能都正常後，才把新版本設為正式 release，並更新 [release manifest](../docs/handover/release_manifest.md)。

## 三個禁止事項

1. 不要把廠商 P302 軟體包進 GUI release。
2. 新版本尚未測試完成前，不要覆蓋目前可用的 release。
3. 不要只複製單一 exe；必須保留完整 release 資料夾。

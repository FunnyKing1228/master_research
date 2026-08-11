# Packaging AI Handover

本頁供 AI／自動化代理判定 Windows GUI release 的 source、artifact 與 vendor 邊界；先讀 [`../AGENTS.md`](../AGENTS.md)，人員操作簡版見 [`README.md`](README.md)。Repository 是 source SSOT；stable pre-measure／probe 已回寫 source，但正式重建前仍須先查 manifest。

## Source map

- GUI entry：`gui/ai_control_gui.py`
- Deployment runtime：`control/run_deployment.py`
- Vendor protocol parser/writer：`control/io_protocol.py`
- Environment／policy semantics：`core/microgrid_env.py`
- Build entry：`_deploy.ps1`
- PyInstaller spec：`packaging/build_release.spec`
- PyInstaller onedir output：`dist/P302_AI_GUI/`
- Release root：`%USERPROFILE%\Downloads\Microgrid_AI`
- Current release：`%USERPROFILE%\Downloads\Microgrid_AI\release_v22_flow_power_limited`
- Current packaged runtime：`%USERPROFILE%\Downloads\Microgrid_AI\release_v22_flow_power_limited\_internal\control\run_deployment.py`
- Current model source：`experiments/v22_flow_power_limited_gpu300/models/best_sac_model.pth`
- Build handover：[`docs/handover/deployment_build.md`](../docs/handover/deployment_build.md)
- Artifact/hash authority：[`docs/handover/release_manifest.md`](../docs/handover/release_manifest.md)
- Human entry：[`docs/HANDOVER_zh.md`](../docs/HANDOVER_zh.md)

正式鏈：

```text
checkpoint + experiment config + repository source
  -> _deploy.ps1
  -> packaging/build_release.spec
  -> dist/P302_AI_GUI/
  -> CPU-only cleanup
  -> %USERPROFILE%\Downloads\Microgrid_AI\release_v22_flow_power_limited
```

`dist/P302_AI_GUI/` 是完整 onedir，必須同時保留 exe 與 `_internal/`。`Microgrid_AI` 內其他 `build_*.spec`、batch、`package.ps1`、`dist_v22*`、歷史 release／overlay 是次要手工環境，不是可重現 source。

## Invariants

1. Vendor P302 軟體永遠與 repository、GUI onedir、release zip 分離；正式 vendor 軟體只安裝在實驗用電腦。
2. `_deploy.ps1` 內 `%USERPROFILE%\Downloads\P302_AI_V4.0\...` 只是開發機 GUI integration-test 預設，不是 build input，也不代表實驗電腦版本。
3. `P302V2.4`、`P302_AI_v2.5`、`P302_AI_V4.0` 的 `Data.txt`／`Command.txt` protocol 可能不同；每一 vendor 版本必須在對應實驗電腦逐版驗收，不可推定相容。
4. 每次 release 必須綁定 experiment、best/final 類型、checkpoint/config hash、repo revision、dirty 狀態、source/package diff 與 zip hash。
5. GUI、control、spec、script、model/config 與 packaged content 必須一起驗證；package 可啟動不代表硬體 protocol 或安全 guard 已通過。
6. `mode 3` 用於 charge/rest/pre-measure；`mode 4` 是明確停止 motor/battery。零功率仍保留實體 battery PP，不以 `PP=00` 取代。
7. 不得把 artifact 反向當成 source SSOT，也不得把「source 已同步」描述成「新 package 已重建並完成硬體驗收」。

## Current synchronization：重建前硬性檢查

2026-08-11 已將 current packaged `run_deployment.py` 的 stable pre-measure／probe 功能完整回寫 repository。文字 diff 已歸零，且 deployment／I/O／environment 測試合計 `175 passed`。目前 source 與 package 都包含：

- 有最短／最長等待、週期採樣、連續穩定樣本與電壓容差；
- 有 `PreMeasureResult` 與 stable／recovering／confirmed-undervoltage 判定；
- 放電前執行 battery solo-only probe，檢查完整負載覆蓋、最低電壓與最大壓降；
- probe 後回 mode 3 rest；
- deployment CSV 含 `pre_measure_*`、`probe_*`、`voltage_confirmation_pending` 等欄位；
- 不穩定／回升中的低電壓先禁止放電，不直接冒充 confirmed cutoff。

尚未完成的是由目前 source 正式重建 onedir、重新計算 artifact hash，以及在實驗電腦做 vendor／硬體驗收。重建時仍不得直接覆寫 current release；先產生獨立測試包並完成驗收。

## 技術債

- Stable pre-measure／probe 已 source-controlled，但正式重建與實驗電腦驗收尚未完成。
- Vendor version → protocol 尚無完整相容性矩陣。
- `Microgrid_AI` 次要手工 build 可能含歷史 dependency、overlay 或未同步修正。
- `_deploy.ps1` 在 release 鎖定時有 `release_v16` fallback；名稱可能誤導，需消除鎖定後確認實際輸出。
- Packaged EXE 沒有 `--self-test`；驗收紀錄不得聲稱存在。
- Build script marker checks 不能取代 unit、compile、CPU load、GUI launch、dry-run 與實驗電腦硬體測試。

## 驗證與逐版 protocol 驗收

1. 建置前：記錄 repo revision／dirty diff，核對 checkpoint/config 存在及 hash，先比對 repository 與 packaged `run_deployment.py`、`io_protocol.py`。
2. Source 檢查：確認 stable pre-measure／probe、CSV schema、guard、dry-run 測試與 packaged 文字內容仍一致。
3. Repository 驗證：執行 `py -m pytest tests\test_io_protocol.py tests\test_deployment.py` 與 `py -m compileall control core gui`。
4. Package 驗證：檢查完整 onedir、runtime、checkpoint/config/hash、CPU model load、GUI 啟動與不寫真機命令的隔離 dry-run。
5. 每一 vendor 版本在其實驗電腦分別驗收：記錄機器、vendor 版本／exe 路徑、Data/Command 樣本與 schema、parser/writer、pre-measure、probe、standby、停止及實際硬體結果。
6. 完成後重建 zip、重算 SHA256，更新 [`release_manifest.md`](../docs/handover/release_manifest.md)；未知欄位標記待驗，禁止沿用舊值。

詳細 handover：[`deployment_build.md`](../docs/handover/deployment_build.md) · [`release_manifest.md`](../docs/handover/release_manifest.md) · [`HANDOVER_zh.md`](../docs/HANDOVER_zh.md)。

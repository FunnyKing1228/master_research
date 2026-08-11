# 部署建置交接

本文件描述「從模型 checkpoint 產生可交付 Windows release」的建置鏈；這不是 GUI 操作手冊。操作 GUI、連接真機與啟停控制請另見 `docs/deployment_guide.md`。

## 正式、可重現的建置路徑

目前正式來源鏈如下：

```text
experiments/v22_flow_power_limited_gpu300/models/best_sac_model.pth
  -> _deploy.ps1
  -> packaging/build_release.spec
  -> dist/P302_AI_GUI/                  （PyInstaller onedir）
  -> CPU-only 清理
  -> %USERPROFILE%\Downloads\Microgrid_AI\release_v22_flow_power_limited
  -> 內容檢查、測試與封裝
```

以 repository 根目錄為工作目錄執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\_deploy.ps1
```

`_deploy.ps1` 與 `packaging/build_release.spec` 共同固定下列輸入：

- experiment：`v22_flow_power_limited_gpu300`
- checkpoint：`experiments/v22_flow_power_limited_gpu300/models/best_sac_model.pth`
- experiment config：`experiments/v22_flow_power_limited_gpu300/configs/experiment_config.yaml`
- GUI 入口：`gui/ai_control_gui.py`
- Python：優先使用 `.venv_deploy_cpu/Scripts/python.exe`，不存在時才退回 `py`

建置前必須確認 checkpoint 與 experiment config 都存在。spec 在缺檔時會直接失敗，避免產生來源不明的 release。

## onedir 與 CPU-only 清理

`packaging/build_release.spec` 使用 `EXE(..., exclude_binaries=True)` 加 `COLLECT(...)`，輸出是完整 **onedir**，不是單一 exe：

```text
dist/P302_AI_GUI/
  P302_AI_GUI.exe
  _internal/
```

不可只複製 `P302_AI_GUI.exe`，也不可把不完整的 `_internal` 當成 release；`base_library.zip`、Python runtime、Torch、控制程式、模型與設定都依賴完整 onedir。

PyInstaller 完成後，`_deploy.ps1` 會從 `_internal` 清除 CPU 部署不需要的內容，包括：

- CUDA、cuDNN、cuBLAS、cuSPARSE、cuSOLVER、cuFFT、NVRTC 等 NVIDIA DLL；
- `scs`、`clarabel` 等部署未使用 solver；
- TensorFlow、TensorBoard、Keras、AWS 套件及部分 legacy/MKL 檔案。

這一步只能刪除已明確排除的部署非必要項目。若增加 runtime dependency，必須先更新 spec 與清理白名單，再重建測試，不能直接在 release 中猜測刪檔。

## 產生 release

清理後，腳本把完整 `dist/P302_AI_GUI` 複製至：

```text
%USERPROFILE%\Downloads\Microgrid_AI\release_v22_flow_power_limited
```

並補入或建立：

- 根目錄與 `_internal` 使用的 `load_pattern.txt`；
- `soh_models/` 與可取得的 `.pth`、`.pkl`、`.npz`；
- `config_gui.json`，其模型路徑指向 `_internal/models/best_sac_model.pth`；
- 預設 CPU device 與本機 vendor、log 路徑。

## 廠商 P302 軟體不屬於 release

`P302V2.4`、`P302_AI_v2.5`、`P302_AI_V4.0` 是同一套廠商控制軟體的不同版本。正式廠商軟體應由實驗用電腦另外安裝，永遠不併入本 repository 或 GUI onedir release。

`_deploy.ps1` 目前寫入的：

```text
%USERPROFILE%\Downloads\P302_AI_V4.0\P302_AI_V4.0\P302_AI.exe
```

只是這台開發／打包電腦為了「打包後立即測試 GUI」而保留的預設測試路徑，不代表所有實驗都使用 V4.0，也不是 PyInstaller build input。搬到實驗用電腦後，應在 GUI 設定中選擇該電腦實際安裝的廠商版本與執行檔。

不同廠商版本的 `Data.txt`／`Command.txt` protocol 有差異。每次 GUI 整合測試與真機部署都必須記錄：

- 實驗電腦與廠商軟體版本；
- `P302_AI.exe` 實際路徑；
- Data/Command 範例與欄位格式；
- parser／writer 相容性測試結果。

目前沒有一份完整維護的「廠商版本 → protocol」相容性表。不能因開發機上的某個版本測試成功，就推定其他實驗電腦版本完全相容；每個實際版本都要在對應實驗電腦重新驗收。

舊 release 若被鎖定，腳本可能改名為時間戳備份；若仍無法取代，現有腳本含 `release_v16` fallback。遇到 fallback 時不可把它誤記為 v16 模型，應先排除檔案鎖定，再重新執行並確認實際輸出目錄。

## 建置後檢查與測試

`_deploy.ps1` 內建的檢查至少涵蓋：

- exe、`config_gui.json`、`load_pattern.txt`、checkpoint、experiment config；
- `control/run_deployment.py`、`core/microgrid_env.py` 與 SoH artifacts；
- PV support / blocking、hybrid current、flow-power guard、solo-only 放電相關 marker；
- mode 3 rest/pre-measure、實體 PP、0% rest flow、50% pre-measure flow；
- v22 flow action、flow available-power limit 與 deployment load scale；
- CPU 清理後總容量是否異常。

建置完成仍需做以下檢查：

```powershell
py -m pytest tests\test_io_protocol.py tests\test_deployment.py
py -m compileall control core gui
git status --short
```

另外應在無真機輸出的安全條件下確認：

1. `P302_AI_GUI.exe` 可啟動，沒有 `Failed to start embedded python interpreter`。
2. `_internal/base_library.zip` 與必要 runtime 存在。
3. packaged checkpoint/config 的名稱與 hash 符合本次 manifest。
4. CPU 機器可載入模型，且不要求 CUDA DLL。
5. dry-run 或隔離測試不會意外寫入真機 `Command.txt`。
6. 真正交付前再執行 Data/Command protocol、pre-measure、standby 與停止路徑的硬體測試。

## `Microgrid_AI` 次要手工 build

`%USERPROFILE%\Downloads\Microgrid_AI` 內另有 `build_*.spec`、`build_*.bat`、`package.ps1`、`dist_v22*`、舊 release 與人工 overlay。這些是歷史／次要手工建置環境，可能含本機修改、舊 dependency 或 release-only hotfix。

它們可用於診斷、比對或緊急人工驗證，但不是正式可重現來源。正式重建應回到本 repository 的 checkpoint → `_deploy.ps1` → `packaging/build_release.spec` 鏈。若手工 build 有必要保留的修正，應先明確移植回 repository、補測試，再由正式鏈重新產生 release；不得把手工覆寫後的資料夾描述成已與 source 同步。

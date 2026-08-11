# Release Manifest

本文件記錄 2026-08-11 當時可確認的 model／release 狀態。它是交接快照，不代表 repository、`Microgrid_AI` 手工 build 與已壓縮 zip 永遠自動同步。

## 目前可確認的 v22 主線

| 項目 | 2026-08-11 可確認值 |
|---|---|
| experiment | `v22_flow_power_limited_gpu300` |
| 選用 checkpoint | `experiments/v22_flow_power_limited_gpu300/models/best_sac_model.pth` |
| checkpoint SHA256 | `A8FB03C3531D8C495CF04AB99BCB73695AD53A16D52919A4831D678D2B88247D` |
| experiment config | `experiments/v22_flow_power_limited_gpu300/configs/experiment_config.yaml` |
| experiment config SHA256 | `5654C4538A82C8FD48EEC474AF32B1167124355592623D2EE54B2036A234096D` |
| 查核時 repository HEAD | `b969dd8e00ea0de728a140746bfd87fac1da2f16` |
| 查核時 worktree | **dirty**；此 commit 無法單獨重現當時 release |
| 正式 build script | `_deploy.ps1` |
| PyInstaller spec | `packaging/build_release.spec` |
| release 目錄 | `%USERPROFILE%\Downloads\Microgrid_AI\release_v22_flow_power_limited` |
| release zip | `%USERPROFILE%\Downloads\Microgrid_AI\release_v22_flow_power_limited.zip` |
| zip SHA256 | `BA8EBF2739CC31DB7DCFB829F25AE39872F6204A41776C216D3000786F1505F0` |

上述 SHA256 是 **2026-08-11 當時已建立 zip 的值**。只要重新建置、overlay、修改設定、補檔或重新壓縮，就必須重新計算並更新本 manifest；不可沿用舊 hash。

可用下列命令重驗：

```powershell
Get-FileHash `
  "$env:USERPROFILE\Downloads\Microgrid_AI\release_v22_flow_power_limited.zip" `
  -Algorithm SHA256
```

## v16sp final 與 v22 best：不同歷史候選

這兩條紀錄不可互相覆寫或假裝是同一模型：

1. 歷史 v16sp deployment-aligned 候選：
   ```text
   experiments/v16sp_guided_teacher_v5_hybrid50_2000_deployalign/models/final_sac_model.pth
   ```
   當時經 best/final 圖人工比較後，特例選擇 **final**，並曾對應 `%USERPROFILE%\Downloads\Microgrid_AI\release_v16`。

2. 目前 v22 flow-power-limited 候選：
   ```text
   experiments/v22_flow_power_limited_gpu300/models/best_sac_model.pth
   ```
   現行 `_deploy.ps1` 與 `packaging/build_release.spec` 都指向此 **best**，release 名稱為 `release_v22_flow_power_limited`。

「舊實驗常見 best 優於 final」不是永遠規則；v16sp 該輪是人工驗證後採 final，v22 現行 manifest 則明確採 best。論文、圖表或 release 說明都應寫出 experiment 與 checkpoint 類型。

## vendor protocol 相容性與差異

`P302V2.4`、`P302_AI_v2.5`、`P302_AI_V4.0` 是同一套廠商控制軟體的不同版本，不是 AI repository 版本。正式廠商程式應存在於實驗用電腦；本開發／打包電腦留下的副本只用於 GUI 打包後的整合測試。`_deploy.ps1` 內的 V4.0 路徑只是本機測試預設，不代表所有實驗電腦或 release 都使用 V4.0。

廠商軟體永遠與 GUI release 分開，由實驗用電腦另外安裝。這些廠商版本的 `Data.txt`／`Command.txt` protocol 有差異，而且目前沒有完整維護的版本相容性表，因此每次 release 驗收必須寫明「在哪一台實驗電腦、使用哪個廠商版本、測過哪種格式」。目前 source 與 release 內的 `control/io_protocol.py` 比對沒有內容差異；parser／writer 已處理的格式包括：

- `Data.txt` header 新格式可含 `YYYYMMDDhhmmss,{load_groups}`；`vendor_load_count` 會進 raw/deployment CSV。
- MPPT 新格式可多出 bus V/I/P；舊格式只有 Solar/MPPT 六欄。
- load 新格式可同列 load 與 grid V/I/P；舊格式只有 load 三欄。
- battery 資料支援既有六欄與含 charge voltage 的七欄。
- `Command.txt` 使用 situation code、時間戳／load groups、`PP,power_mW,flow_percent,`。
- mode 3 用於 charge、一般 rest 與 pre-measure；mode 4 是明確停止 motor/battery，不可當一般 standby。
- 零功率命令仍保留實體 battery PP，不使用 `PP=00` 取代。

因此，資料分析必須記錄實際 vendor controller 版本與 CSV schema。parser 能讀數種格式不表示各廠商版本的欄位、行為與 semantics 完全相同，也不能以開發機上的測試版本取代實驗電腦驗收。

## stable pre-measure／probe source 同步狀態

2026-08-11 已把 current release 的 stable pre-measure／probe 功能包回寫到 repository：

```text
repository:
  control/run_deployment.py

current release:
  %USERPROFILE%\Downloads\Microgrid_AI\
    release_v22_flow_power_limited\_internal\control\run_deployment.py
```

目前 repository source 已包含：

- pre-measure 最短／最長等待、週期採樣、連續樣本穩定度與電壓容差；
- `PreMeasureResult`、stable／recovering／confirmed-undervoltage 判斷；
- 放電前 battery solo-only probe、完整負載功率覆蓋、最低電壓與最大壓降檢查；
- probe 後回到 mode 3 rest；
- `pre_measure_*`、`probe_*`、`voltage_confirmation_pending` 等 deployment CSV 欄位；
- 不穩定／仍回升的低電壓先禁止放電，但不直接假裝成已確認 cutoff。

回寫後 `git diff --no-index` 顯示 repository 與 packaged `run_deployment.py` 的文字內容一致；兩者 SHA256 仍不同，是工作樹與 package 檔案的換行位元不同。Repository 測試結果為 `175 passed`（deployment 101、I/O 39、environment 35）。

仍未完成的項目：

1. 由正式 `_deploy.ps1` 從目前 source 重建 onedir；
2. 在不連接真機命令的條件下驗證 GUI、CPU model load、pre-measure／probe 與停止路徑；
3. 到各 vendor 版本的實驗電腦做 Data/Command 與硬體驗收；
4. 重建 zip、重算 SHA256、更新本 manifest。

在完成正式重建前，既有 zip hash 仍代表舊 package artifact；不得把「source 已同步」寫成「新 release 已重新驗收」。

2026-08-11 回寫後證據：

| 檔案 | repository SHA256 | packaged SHA256 | 結果 |
|---|---|---|---|
| `control/io_protocol.py` | `DED5F84FD567512952931A2A431D49A371D841444CD812472AB3CA0ADB65F667` | `DED5F84FD567512952931A2A431D49A371D841444CD812472AB3CA0ADB65F667` | 一致 |
| `control/run_deployment.py` | `ADF72D707C85ADADA78ED8EED39B2B0EBB4015A76081CDEA194589E3BE02DB5E` | `36AF23CB1F6A73702128BD101E2A3CD5BBA3C33F0C318D196611649BD85C681A` | 文字內容一致；換行位元不同 |

模型與 experiment config 在 repository experiment 與 packaged release 中的 SHA256 都分別等於本頁上方記錄值。Source 同步沒有替換模型檔，也沒有重建既有 zip。

## Release 驗收紀錄

每次新 release 至少應新增或更新：

- 建置日期、電腦與 Python／PyInstaller 版本；
- repository revision 與未 commit diff 狀態；
- experiment、checkpoint 類型及 checkpoint SHA256；
- experiment config SHA256；
- release 目錄名稱與 zip SHA256；
- source 與 packaged `run_deployment.py`／`io_protocol.py` 是否一致；
- unit tests、compile、exe 啟動、CPU model load、dry-run 與硬體 protocol 測試結果；
- CSV schema 是否增加／刪除欄位；
- 已知 release-only overlay 或未同步 hotfix。

若任一欄無法確認，應標註「未知／待驗」，不要用舊 release 的值補上。

## 2026-08-11 交接 smoke check

本次交接實際完成：

- `tests/test_microgrid_env.py`、`tests/test_deployment.py`、`tests/test_io_protocol.py`：`163 passed`。
- 訓練、selected-day、3-day／5-day validation CLI 的 `--help`：通過。
- `config_p302_v22_flow_power_limited.yaml` 載入、dataset 存在及 `solo_only`／PV support／flow power limit：通過。
- `v22_flow_power_limited_gpu300/best` 實跑單日、連續 3 日與連續 5 日 rollout：通過，輸出在同實驗的 `results/handover_smoke_*` 本機目錄。
- `tools/plotting_handoff` 七個 Python 檔案編譯：通過。
- 無 MC 與 1000-run MC 範例 pipeline：各產生兩張 PNG，通過。
- 九份交接 Markdown 相對連結與 `git diff --check`：通過。
- packaged EXE 可啟動 GUI；另一次帶 deployment 參數的呼叫也成功進入 embedded Python／argparse。此版本**沒有 `--self-test` 參數**，不得在驗收表中假裝有內建 self-test。

已知非阻擋警告：

- Python 環境仍會顯示 legacy `gym` 已停止維護的警告。
- validation 會警告選用的 `pymgrid` `Microgrid` 建構方式不相容，之後依設計改用本專案的 `MicrogridEnvironment` 與指定 CSV；三個 rollout 都正常完成。

本次沒有重新執行完整 PyInstaller build，也沒有覆寫既有 release／zip；只驗證 build script 與 spec 確實綁定本頁模型、既有 package 可啟動，以及既有 zip/hash。下一次 source 同步後仍必須正式重建並重跑完整 release 驗收。

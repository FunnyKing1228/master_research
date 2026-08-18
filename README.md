# Microgrid RL 研究與部署程式

這是微電網強化學習研究的 **source-only 交接 repository**，集中保存程式碼、設定、測試與可重現流程。

本 repository 不等於完整實驗環境，也不代表任何模型或 GUI release 已完成最終驗收。

## 從哪裡開始？

- **新接手者**：先讀 [`docs/HANDOVER_zh.md`](docs/HANDOVER_zh.md)
- **AI／維護者**：先讀 [`AGENTS.md`](AGENTS.md)
- **目前部署與 artifact 狀態**：查閱 [`docs/handover/release_manifest.md`](docs/handover/release_manifest.md)
- **以前遇過的問題與解法**：查閱 [`PROJECT_ISSUES_AND_SOLUTIONS.md`](PROJECT_ISSUES_AND_SOLUTIONS.md)
- **newHW（LFP，推論為離網、待確認）遷移交接**：先讀 [生命週期對照](docs/handover/newHW_lifecycle_mapping.md)，再依需查閱 [遷移紀錄](docs/handover/newHW_migration_log.md)、[待補資料與私人 artifacts](docs/handover/newHW_pending_data.md)、[新增檔案對照](docs/handover/newHW_file_changes.md)

## 依工作選入口

1. [資料準備](data/README.md)
2. [模型訓練](core/README.md)
3. [實驗與跨日驗證](experiments/README.md)
4. [部署與 GUI 打包](packaging/README.md)
5. [現場 CSV 與繪圖](tools/plotting_handoff/README.md)

每個區域另有 `README_AI.md`，記錄 source map、限制、驗證方式與已知技術債。

## Repository 內容

```text
configs/                 訓練、驗證與部署設定
control/                 即時控制流程與 Data.txt／Command.txt protocol
core/                    SAC、環境、安全層與 SoH inference source
data/scripts/            資料處理、baseline 與繪圖腳本
docs/                    交接與版本狀態文件
experiments/README*.md   實驗流程說明（不含實驗產物）
gui/                     Windows 操作介面
packaging/               PyInstaller 打包流程
tests/                   單元與 regression tests
tools/plotting_handoff/  可攜式四面板繪圖工具與小型範例資料
```

## 不包含的內容

公開 repository 刻意不保存：

- `data/raw/`、`data/processed/` 的實際研究資料；
- `experiments/` 訓練產物與 RL／SoH 模型權重；
- `outputs/`、`thesis_sim/` 與大型圖表產物；
- GUI release、ZIP、EXE 與 vendor P302 軟體；
- `config_gui.json`、本機 log、虛擬環境與個人路徑。

因此，clone 後必須另外取得符合設定檔 schema 的 processed dataset、指定 checkpoint、SoH model，以及實驗電腦上的 vendor 軟體，才能進行完整訓練或部署。

## 目前重要狀態

- Stable pre-measure／probe 已回寫 source 並通過單元測試。
- 尚未由目前 source 正式重建 GUI release，也尚未完成實驗電腦硬體驗收。
- 模型是否可部署或可作為論文證據，必須以實驗清冊、跨日驗證與 release manifest 為準，不能只看單日結果或 checkpoint 名稱。
- newHW 目前完成隔離遷移骨架與 in-sample 診斷；最新 300-episode／SoC 20–80% 試驗在同一窗口達 finite-window oracle，但 raw policy 仍依賴 SafetyNet，模型尚未通過泛化驗證。部署仍被 I/O 規格缺口阻擋；系統為離網架構屬推論而非硬體端確認，詳見生命週期對照。

## 基本驗證

```powershell
py -m pytest tests\test_deployment.py tests\test_io_protocol.py tests\test_microgrid_env.py
py -m compileall control core gui data\scripts tests
```

本專案採用 [MIT License](LICENSE)。

# 我要做什麼？

這是新接手者的繁中入口。先在 `conformal-microgrid-rl`（程式碼、設定、測試與可重現流程的 **SSOT**）執行 `git status --short --untracked-files=all`，保留既有未提交內容，再選一條流程。

## 1. 資料準備

- **入口**：[資料準備](../data/README.md)
- **主要命令／產物**：依實驗 YAML 準備 `data/processed/*.csv`；原始現場 CSV 保持不可變且不提交。
- **下一步**：確認欄位、單位、時間戳與切分後，進入[模型訓練](../core/README.md)。

## 2. 模型訓練

- **入口**：[模型訓練](../core/README.md)
- **主要命令／產物**：`py core\train_sac_microgrid.py --config <YAML> --name <唯一名稱>`；產生 `experiments/<name>/configs/`、`logs/`、`models/`、`results/`。
- **下一步**：不要依 `best`／`final` 檔名定案，進入[實驗驗證](../experiments/README.md)。

## 3. 實驗驗證

- **入口**：[實驗驗證](../experiments/README.md)
- **主要命令／產物**：依序跑單日、連續 3 日、連續 5 日 rollout；產生可追溯的圖、指標與驗證摘要。
- **下一步**：更新[實驗清冊](handover/experiment_inventory.md)；只有跨日合格的候選才可進入部署評估。

## 4. 部署打包

- **入口**：[部署打包](../packaging/README.md)
- **主要命令／產物**：`powershell -ExecutionPolicy Bypass -File .\_deploy.ps1`；產生完整 PyInstaller onedir release。
- **下一步**：逐項驗收並更新 [release manifest](handover/release_manifest.md)，不可用 release 反向取代 source。

## 5. 現場 CSV 與繪圖

- **入口**：[現場 CSV 與繪圖](../tools/plotting_handoff/README.md)
- **主要命令／產物**：從 `raw_data_v2_*.csv` 與 `deployment_v2_*.csv` 產生四面板 PNG；原始 CSV、圖與 PDF 不提交。
- **下一步**：記錄 runtime、vendor 版本、CSV schema、時間窗與繪圖命令，再判讀結果。

## 如果要修改程式

一般操作只要依上面五條流程進入對應的 `README.md`。AI／維護者請改從 [`AGENTS.md`](../AGENTS.md) 開始，再依任務閱讀各目錄的 `README_AI.md`；技術限制、歷史問題與版本差異不放在人員導航首頁。

# AI／維護者固定閱讀順序

任何修改都先讀本檔與 [`.cursorrules`](.cursorrules)，再依任務選一條，不必先把全部文件讀完：

- 資料準備／schema：[`data/README_AI.md`](data/README_AI.md)
- 模型訓練／環境／agent：[`core/README_AI.md`](core/README_AI.md)
- 實驗／選模／跨日驗證：[`experiments/README_AI.md`](experiments/README_AI.md)
- 部署／打包／vendor protocol：[`packaging/README_AI.md`](packaging/README_AI.md)
- 現場 CSV／繪圖／MC replay：[`tools/plotting_handoff/README_AI.md`](tools/plotting_handoff/README_AI.md)

涉及模型選擇、部署、打包或現場行為時，額外核對 [`docs/handover/release_manifest.md`](docs/handover/release_manifest.md)。

## newHW 固定入口與限制

- newHW 相關任務先讀 [`docs/handover/newHW_lifecycle_mapping.md`](docs/handover/newHW_lifecycle_mapping.md)，確認各站狀態與阻擋方；需要實際執行時再讀 [`docs/handover/newHW_reproduce.md`](docs/handover/newHW_reproduce.md)。
- 禁止修改 P302 檔案，亦不得調整 newHW 物理參數或 reward 權重；後者屬研究決策。
- newHW 站④部署被 I/O 規格缺口阻擋；規格取得前不要嘗試部署或沿用 P302 打包鏈。

## 固定規則

- 先執行 `git status --short --untracked-files=all`，保留所有既有未提交內容；未獲明確要求不得 commit。
- `conformal-microgrid-rl` 是 source、設定、測試與可重現流程的 SSOT。checkpoint、圖、CSV、release 與外部工作區都不能反向取代 source。
- 廠商 P302 軟體與 AI repository／GUI release 分離；每台實驗電腦的 vendor 版本及 `Data.txt`／`Command.txt` protocol 必須個別驗收。
- Stable pre-measure／probe 已回寫 repository source 並通過單元測試，但尚未由目前 source 正式重建與完成實驗電腦驗收；不得把「source 已同步」描述成「新 release 已驗收」。
- 技術細節以 [`docs/handover/`](docs/handover/) 現有文件為準；不要把入口 README 擴寫成第二套規格。

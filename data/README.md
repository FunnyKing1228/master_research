# 實際部署資料

這裡用來保存從實驗電腦帶回來的實際部署資料。

每個實驗日會有兩種 CSV：

- `deployment_v2_YYYY-MM-DD.csv`：每個控制決策時間點的模型、命令與系統狀態。
- `raw_data_v2_YYYY-MM-DD.csv`：較密集的原始量測資料。

## 資料怎麼放

將同一天的兩個檔案成對放入 `data/raw/`：

```text
data/raw/
  deployment_v2_2026-07-17.csv
  raw_data_v2_2026-07-17.csv
  deployment_v2_2026-07-18.csv
  raw_data_v2_2026-07-18.csv
```

注意：

- 兩個檔名的日期必須相同。
- 每一天都應有一組；缺少其中一個就表示該日資料不完整。
- 保留原始檔名與內容，不要先手動合併或覆寫。
- `data/raw/` 只保存在本機，不會提交到公開 repository。

## newHW 資料放哪裡？

newHW 是另一套硬體與 schema，不可混入上面的 P302 `data/raw/`。它使用獨立目錄：

```text
data/newHW/
  raw/Data140826.csv
  processed/training_newHW_15min.csv
  processed/newHW_data_quality_summary.json

data/scripts/newHW/
  prepare_data_newHW.py
  rollout_newHW.py
  analyze_energy_bound_newHW.py
  figures/plot_data_newHW.py
```

各檔案的用途：

| 檔案 | 內容 |
|---|---|
| `raw/Data140826.csv` | 私下取得的 5 秒原始感測資料；不可手動修改，且應先核對 SHA256 |
| `training_newHW_15min.csv` | 將原始資料清理並重採樣成 15 分鐘後的訓練輸入 |
| `newHW_data_quality_summary.json` | 原始列數、時間範圍、缺口、遮罩筆數、假設值與 source hash |
| `prepare_data_newHW.py` | 套用欄位修正、SoC 重建、固定負載假設與 15 分鐘重採樣 |
| `rollout_newHW.py` | 讀 checkpoint 做 in-sample rollout，輸出圖、summary JSON 與逐步 audit CSV |
| `analyze_energy_bound_newHW.py` | 計算目前假設下的能量上界，協助判斷模型與物理供需差距 |
| `plot_data_newHW.py` | 畫資料品質圖，不負責訓練模型 |

raw、processed CSV 與 rollout audit CSV 都被 `.gitignore` 排除。完整取得方式、已知 hash 與限制見
[`../docs/handover/newHW_reproduce.md`](../docs/handover/newHW_reproduce.md)。

需要清理資料、建立訓練資料、診斷問題或畫圖時，請交由維護者或 AI 依目的選擇正確工具；技術細節見 [`README_AI.md`](README_AI.md)。

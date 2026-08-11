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

需要清理資料、建立訓練資料、診斷問題或畫圖時，請交由維護者或 AI 依目的選擇正確工具；技術細節見 [`README_AI.md`](README_AI.md)。

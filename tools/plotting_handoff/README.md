# 實際部署資料畫圖

這個工具會把實驗電腦帶回來的 deployment／raw CSV 畫成容易查看的 PNG 圖片。

一般使用者只需要照下面步驟操作。`mc_replay`、欄位轉換與其他技術設定不需要自行處理。

## 第一步：準備資料

先把同一天的兩個 CSV 成對放入 repository 的 `data/raw/`：

```text
data/raw/
  deployment_v2_2026-07-17.csv
  raw_data_v2_2026-07-17.csv
  deployment_v2_2026-07-18.csv
  raw_data_v2_2026-07-18.csv
```

兩個檔名的日期必須相同。缺少其中一個，該日資料就不完整。

## 第二步：打開正確的終端機

1. 使用 Cursor 開啟 `conformal-microgrid-rl` 資料夾。
2. 在上方選單選擇 **Terminal → New Terminal**。
3. 確認終端機目前位置的最後一段是 `conformal-microgrid-rl`。

下面所有命令都貼在這個終端機執行。

## 第三步：第一次使用時安裝畫圖套件

這個命令只需要在第一次使用，或換新電腦時執行：

```powershell
py -m pip install -r tools\plotting_handoff\requirements.txt
```

等畫面停止跑動並再次出現輸入提示後，再進行下一步。

## 第四步：產生圖片

貼上下面整行命令：

```powershell
py tools\plotting_handoff\data_verification\dataset_to_figures.py --data-dir data\raw --output-dir tools\plotting_handoff\my_output
```

這個命令會讀取 `data/raw/` 裡所有日期成對的資料，並把圖片存到：

```text
tools/plotting_handoff/my_output/
```

正常完成時，終端機會顯示 `Saved:`，通常會產生兩張圖：

- `*_power.png`：查看負載、PV、電池功率、流量與 SoC。
- `*_voltage_current.png`：查看負載、PV、SoC、流量及電池電壓／電流。

直接用圖片檢視器開啟 PNG 即可。

## 想先測試工具是否正常

Repository 內附三天範例資料。執行：

```powershell
py tools\plotting_handoff\data_verification\dataset_to_figures.py --data-dir tools\plotting_handoff\dataset --output-dir tools\plotting_handoff\test_output
```

若看到 `Saved:`，並在 `tools/plotting_handoff/test_output/` 找到兩張 PNG，表示基本畫圖功能正常。

## 如果失敗

先檢查：

1. deployment 與 raw CSV 是否成對，而且日期相同。
2. 檔案是否真的放在 `data/raw/`。
3. 終端機位置是否為 `conformal-microgrid-rl`。
4. 第三步的安裝命令是否成功。

仍無法完成時，不要自行修改 CSV 或程式。把終端機最後一段錯誤訊息完整複製給維護者或 AI。

## 看圖時要注意

- PV 與市電可能同時支援負載，不能把圖解讀成兩者只能選一個。
- 電池放電不是與 PV／市電一起分攤負載。
- 一天的圖看起來正常，不代表模型能連續多日穩定運作。

需要指定日期／時間、使用 MC replay、修改圖片格式或解讀欄位時，請交由維護者或 AI 處理；技術細節見 [`README_AI.md`](README_AI.md)。

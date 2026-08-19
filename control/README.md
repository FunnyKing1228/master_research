# P302 Data.txt／Command.txt 協定

本頁是人員操作簡版，說明 AI GUI 如何與 **P302 廠商控制軟體**交換資料。
實際 parser／writer 以 [`io_protocol.py`](io_protocol.py) 為準；部署與版本狀態另見
[`../docs/handover/release_manifest.md`](../docs/handover/release_manifest.md)。

> **這不是 newHW 協定。** newHW 的 I/O 規格尚未取得，
> [`io_protocol_newHW.py`](io_protocol_newHW.py) 仍會拋出 `NotImplementedError`，
> 不可假設 newHW 沿用 P302 的文字檔格式。

## 資料怎麼流動？

```text
P302 廠商軟體
  └─ 寫入 Data.txt（量測）
         ↓
GUI／control/run_deployment.py
  └─ 解析量測、建立 state、模型推論、套用 SafetyNet
         ↓
     寫入 Command.txt（命令）
         ↓
P302 廠商軟體讀取並交給硬體
```

- `Data.txt` 是廠商軟體給 AI 的**輸入**。
- `Command.txt` 是 AI 給廠商軟體的**輸出**。
- GUI release 與廠商軟體是兩套獨立程式；廠商軟體不會包進 GUI release。
- 不同 P302 廠商版本可能有不同欄位；parser 能讀多種格式，不代表所有版本已完成硬體驗收。

## Data.txt：AI 讀到什麼？

目前支援的完整新格式範例：

```text
20260320120000,3
1600,500,8000,1500,450,6750,1200,300,3600,
550,33,400,2200,100,220000,
1,101,720,500,1200,332,1000,
```

各行意義：

| 行 | 格式 | 單位與說明 |
|---|---|---|
| 1. header | `YYYYMMDDhhmmss,load_groups` | UTC+8 時間戳；第二欄為目前負載組數，可省略 |
| 2. MPPT | `SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P[,BusV,BusI,BusP]` | 電壓原始值 ÷100 = V；電流為 mA；功率為 mW |
| 3. load/grid | `LoadV,LoadI,LoadP[,GridV,GridI,GridP]` | 電壓原始值 ÷100 = V；電流為 mA；功率為 mW |
| 4 起 battery | `PP,SoC,BattV,ChargeV,Current,Temp,Speed` | PP 為電池 ID；SoC ÷10 = %；電壓 ÷100 = V；電流 mA；溫度 ÷10 = °C；speed ÷10 = % |

### 舊格式也可能出現

- MPPT 舊格式只有前 6 欄，沒有 Bus V／I／P。
- load 舊格式只有 Load V／I／P，沒有 grid；更舊資料可能完全沒有 load 行。
- battery 舊格式為 `PP,SoC,BattV,Current,Temp,Speed`，沒有 `ChargeV`。
- battery current 可為負值；現有測試將舊格式負電流保留為負值，不會轉成 0。

`read_vendor_data_file()` 會回傳：

```text
mppt              Solar／MPPT 六個值
mppt_bus          Bus V／I／P，舊格式為 None
load              Load V／I／P，缺少時為 None
grid              Grid V／I／P，舊格式為 None
batteries         依 PP 分組的電池量測
timestamp         Data.txt 時間戳
vendor_load_count header 提供的負載組數
```

目前正式部署讀取時會拒絕超過 60 秒的 stale Data.txt，且使用
`clear_after_read=False`，不會在每次讀取後清空檔案。`io_protocol.py` 的通用函式
預設值可能不同，呼叫者必須明示自己的讀取語意。

## Command.txt：AI 寫出什麼？

目前部署格式固定為三行：

```text
{situation_code}
YYYYMMDDhhmmss,{load_groups}
PP,{power_mW},{flow_percent},
```

例如 pre-measure：

```text
3
20260316120000,4
01,0,50,
```

欄位說明：

| 欄位 | 說明 |
|---|---|
| `situation_code` | 決定充電、放電、rest 或停止路徑 |
| timestamp | UTC+8，格式 `YYYYMMDDhhmmss` |
| `load_groups` | 目前負載組數 |
| `PP` | 實體電池 ID，例如 `01` |
| `power_mW` | 功率**大小**，單位 mW；方向由 situation code 決定，不靠負號 |
| `flow_percent` | 液流幫浦百分比，限制在 0–100 |

### situation code 的現行部署語意

| code | 現行用途 |
|---:|---|
| 1 | battery discharge；只在 solo-discharge 與部署 guard 允許時使用 |
| 2 | 目前正式部署邏輯不使用 |
| 3 | battery charge、一般 rest、pre-measure，以及 grid／PV 路徑 |
| 4 | 明確 shutdown／motor off；**不是一般 standby** |

兩個重要例子：

- 一般 rest：`mode 3 + 實體 PP + 0 mW + 0% flow`
- pre-measure：`mode 3 + 實體 PP + 0 mW + 50% flow`

零功率時仍保留實體 PP，例如 `01,0,0,`；不可用 `PP=00` 取代，否則 flow 命令可能失去目標。

## 哪些檔案負責什麼？

| 檔案 | 用途 |
|---|---|
| `control/io_protocol.py` | Data.txt parser、Command.txt writer、格式與單位轉換 |
| `control/run_deployment.py` | 正式 AI 控制迴圈、stale check、pre-measure、SafetyNet 與命令寫入 |
| `control/solar_test_collect.py` | 手動／太陽能收資料流程，以及 rest／pre-measure 命令 |
| `gui/ai_control_gui.py` | 選擇 Data.txt／Command.txt 路徑、啟動控制流程與顯示內容 |
| `tests/test_io_protocol.py` | 新舊格式、負電流、clear-after-read 與 Command.txt regression tests |

`io_protocol.py` 前段另有 8 位 signed／unsigned field 的通用 helper；目前
`run_deployment.py` 實際使用的是本頁描述的 vendor Data.txt／Command.txt 路徑，
不要把兩種格式混在一起。

## 操作與安全注意事項

1. **第一次接新廠商版本先用 dry-run。** dry-run 會解析與計算，但跳過 Command.txt 寫入。
2. 每次驗收都記錄實驗電腦、廠商軟體版本、Data.txt／Command.txt 樣本與實際 schema。
3. 不要把「parser 可解析」描述成「硬體已驗收」；仍需確認 ack、更新頻率、stale data、停止與失聯行為。
4. mode 4 會走 shutdown／motor-off 語意，不可拿來當日常 standby。
5. 新 release 尚未驗收前，不要覆蓋目前可用 release，也不要在真機上用未確認格式試寫命令。

## 最小回歸測試

在 repository root 執行：

```powershell
py -m pytest tests\test_io_protocol.py -q
```

這只驗證 repository parser／writer，不取代特定廠商版本與實驗電腦硬體驗收。

# Microgrid Deployment-Safe RL — 交接說明

本 repo 為 **P302 實體微電網 + SAC／CORAL 安全部署** 的軟體交接副本（Private）。  
用途：後續接手者能快速知道「從哪裡跑、改什麼有風險、什麼先別碰」。

> 詳細英文交接筆記見 [`docs/handover.md`](docs/handover.md)  
> 硬體部署步驟見 [`docs/deployment_guide.md`](docs/deployment_guide.md)

---

## 這個專案是什麼

- 在模擬器訓練 **SAC** 微電網控制器
- 部署時用 **CORAL / SafetyNet** 把不安全動作投影回可行域，再寫入硬體
- 與原廠控制器用 **`Data.txt` / `Command.txt`** 檔案協定溝通
- 附 **Tkinter 操作 GUI** 與 **PyInstaller Windows 打包**

研究／工程重點是 **sim-to-real 與部署安全**，不是完整電化學電池模型。

---

## 目錄速覽

| 路徑 | 內容 |
|------|------|
| `core/` | 環境 `microgrid_env.py`、SAC `sac_agent.py`、訓練、SafetyNet、SoH |
| `control/` | 即時部署迴圈、`io_protocol.py`（Data/Command） |
| `gui/` | 操作介面 `ai_control_gui.py` |
| `configs/` | 實驗／baseline／部署相關設定 |
| `data/scripts/` | 診斷、圖表、前處理腳本 |
| `docs/` | 部署、方法、交接文件 |
| `packaging/` | exe 打包 spec／bat |
| `tests/` | 單元／回歸測試 |
| `examples/` | 獨立範例環境 |
| `soh_models/` | SoH 模型放置處（介面已接，硬體上尚未充分驗證） |
| `outputs/` | 部分診斷輸出樣本（大檔／原始 log 請勿再 commit） |

**本交接副本已移除 `thesis_sim/`**（論文 Chapter-4 模擬沙盒與大量 outputs，非硬體部署必要）。

---

## 接手先看這幾個檔

```text
README.md                      ← 你現在看的這一頁
docs/handover.md               ← 硬體限制、SoC/SoH/Flow 現況、勿亂改清單
docs/deployment_guide.md       ← 部署與 GUI 打包
control/run_deployment.py      ← 主部署迴圈
control/io_protocol.py         ← Data.txt / Command.txt
core/microgrid_env.py          ← 訓練／模擬環境
core/sac_agent.py              ← SAC
core/safety_net.py             ← 安全投影
gui/ai_control_gui.py          ← 操作 GUI
configs/config_p302_sim.yaml   ← 常用模擬設定入口之一
requirements.txt               ← Python 依賴
_deploy.ps1                    ← Windows 一鍵打包
```

歷史 P302 實驗設定：`configs/experiments/p302/`  
Baseline／ablation：`configs/baselines/`

---

## 環境與快速指令

```bash
pip install -r requirements.txt
pytest

# 訓練（範例）
python core/train_sac_microgrid.py --config configs/config_p302_sim.yaml

# 部署（開發模式；路徑依現場機器修改）
python control/run_deployment.py \
  --data-file path/to/Data.txt \
  --command-file path/to/Command.txt \
  --model-path path/to/best_sac_model.pth \
  --battery-pp 01

# Windows GUI 打包
powershell -ExecutionPolicy Bypass -File ./_deploy.ps1
```

改程式前建議：

```bash
python -m compileall core control data/scripts tests gui
pytest
```

---

## 硬體部署（極短版）

1. 先開原廠控制器，確認 `Data.txt` 有在更新  
2. 開 GUI（原始碼或打包版）→ 選原廠 exe、RL checkpoint、log 資料夾  
3. 設定初始 SoC、電池 PP、負載數、current mode  
4. **預設關閉 SoH 改容量**（除非該電池老化資料已驗證）  
5. Start AI control → 監看 log／CSV／voltage cutoff／最終命令  
6. 先停 AI，再停原廠控制器  

寫進硬體的命令可能 ≠ raw policy：中間會被 safety guard 擋下或改寫。請在 log 裡分開看 **raw intent / corrected / final command**。

---

## 目前實務結論（務必讀）

1. **Flow rate**  
   小電池平台上馬達功耗可不小於放電能力。硬體部署請 **固定 flow**；介面與 log 可留，但不要宣稱已完成即時 flow 經濟最佳化。

2. **SoC**  
   主要是庫侖計＋校正的操作估計，不是真電荷量測。電壓 cutoff／回復會影響判讀；若要做 OCV，請用 **靜置電壓實驗**，不要直接從部署 log 硬套曲線。細節見 `docs/handover.md`。

3. **SoH**  
   程式介面與 GUI 欄位已接好，**尚未在本實驗電池上驗證到可放心改容量**。勿開 `--soh-use-for-capacity` 上真機，除非有電池專屬老化驗證。

4. **高風險修改（先別動，或務必加測試）**  
   - `control/io_protocol.py` 欄位解析  
   - `control/run_deployment.py` guard 順序  
   - 電流正負號慣例  
   - SoCTracker 容量／效率語意  

---

## 不要 commit 進 git 的東西

- 原始部署 CSV／大量 raw log  
- 產出的圖、PDF、打包好的 release 資料夾  
- 本機 `config_gui.json`、原廠控制器路徑、個人絕對路徑  
- 大型 RL checkpoint（除非實驗室另有 artifact 規範）  

見 `docs/repository_hygiene.md`。

---

## 建議後續工作（給下一位）

1. 硬體先穩 **安全充放電控制**  
2. 量測各 flow 設定的馬達功耗，再決定要不要把 motor 當輔助負載建模  
3. 做受控 SoC 靜置電壓校正  
4. 有電池專屬老化資料後，再談 SoH 改容量  
5. 維持 log：raw policy ↔ safety correction ↔ final command 可分開分析  

---

## 授權／範圍

本副本僅供研究室內部交接與延續實驗使用。大型資料與模型檔可能不在本 repo，需另向原作者或實驗室索取本機備份路徑。

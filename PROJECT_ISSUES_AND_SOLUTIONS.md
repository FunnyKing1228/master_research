# Project Issues And Solutions

> **歷史問題紀錄，不是目前 release 清單。** 本檔保留問題演進與舊決策；目前候選、checkpoint、package、source divergence 與 hash 一律以 [`docs/handover/release_manifest.md`](docs/handover/release_manifest.md) 和 [`docs/handover/experiment_inventory.md`](docs/handover/experiment_inventory.md) 為準。下方「目前主候選」是當時的 v16sp 快照，不能覆蓋現行 v22 manifest。

本檔用來記錄這個專案在訓練、部署對齊、圖表判讀與論文撰寫過程中，已確認的問題、得到的教訓、做過的修正，以及目前仍待驗證的假說。

目標不是寫成正式報告，而是讓之後的人或未來的自己可以快速接續思路，不要再重踩同一批坑。

## 怎麼查、怎麼新增

- 遇到問題時，先用錯誤訊息、檔名、模型名稱或現象關鍵字搜尋本檔。
- 本檔回答「以前發生過什麼、為什麼、怎麼處理」；目前 release 與模型狀態仍查頁首連結的 manifest／inventory。
- 只有根因與處理方式已經由程式、測試、log 或硬體結果確認後，才新增為「已確認」；未確認內容必須明確標成待驗證假說。
- 新紀錄放在本節下方、舊紀錄上方，使用以下格式：

```markdown
## YYYY-MM-DD 問題名稱

### 症狀
- 使用者看到什麼錯誤或異常？

### 根因
- 經證據確認的原因是什麼？

### 處理
- 修改了哪些 source、設定或操作？

### 驗證
- 跑了哪些測試或硬體檢查？結果如何？

### 狀態
- 已解決／部分解決／待實驗電腦驗收／待驗證假說
```

## 目前主候選

- 目前已完成 release 打包、準備部署到另一台電腦的模型：
  - `experiments/v16sp_guided_teacher_v5_hybrid50_2000_deployalign/models/final_sac_model.pth`
- 對應驗證圖：
  - `experiments/v16sp_guided_teacher_v5_hybrid50_2000_deployalign/results/selected_day_validation_final`
- 對應 release：
  - `%USERPROFILE%\Downloads\Microgrid_AI\release_v16`
- 注意：
  - 這個候選的價值不只是 reward，而是它已把 deployment-style observation、continuous operation、guard-in-the-loop、battery response stress 一起納入。
  - 使用者在本輪檢查後，已明確選擇 `final` 作為當前 release 版本，而不是繼續沿用更早期的 `best checkpoint` 主線。
  - 這代表「此次部署候選」與「歷史上常見 best 優於 final 的經驗」需要分開描述，不能混為一談。


## 2026-08-07 修復紀錄（release 起不來 / Data.txt 負載組數未入 CSV）

### 症狀
- `Microgrid_AI\release\P302_AI_GUI.exe` 彈出 `Failed to start embedded python interpreter!`
- 使用者看不到可選的 load pattern，也無法測 load pattern

### 根因
- `release\_internal` 被掏空（約 166 檔，缺 `base_library.zip` / 完整 runtime），不是完整 PyInstaller onedir
- 先前只把 `load_pattern.txt` 丟進壞掉的 `release`，沒有重建整包

### 處理
- 以完整 `release_v22_flow_power_limited` 重建 `Microgrid_AI\release`
- 覆寫最新 `control/io_protocol.py`、`control/run_deployment.py`
- Data.txt 第一行 `YYYYMMDDhhmmss,{N}` 現在會解析為 `vendor_load_count`
- CSV 新增：`vendor_load_count`、`load_power_per_unit_w`、`load_power_est_w`（組數 × 每組功率）
- 每組功率可由 `config_gui.json` 的 `load_power_per_unit_w` 設定（預設 0.1）
- GUI 也加了「每組功率(W)」欄位，但需完整 PyInstaller 重打包後介面才會出現；目前 overlay 包已可經 config 生效

### 今晚驗證重點
1. EXE 可正常啟動
2. log 出現 `[LOAD] Loaded schedule from ...\load_pattern.txt`
3. Command.txt / Data.txt 第一行負載組數隨 schedule 變化
4. `raw_data_v2_*.csv` / `deployment_v2_*.csv` 有上述三欄

## 已確認的重要教訓

### 0. OCC 必須把 CORAL 修正成本回傳到 actor，不能只出現在 loss 數字上

問題：
- 在 `v8b` 低 SafetyNet 介入訓練後，`violations_attempted` 有下降，但 `safety_projected_meaningful` 仍然偏高。
- 檢查後確認 `core/sac_agent.py` 的 actor loss 中，`occ_pred_pi` 使用了 `.detach()`。
- 這代表 `beta_occ * OCC` 雖然被加進 loss 數值，但 OCC 對 action 的梯度不會回傳到 actor。

後果：
- CORAL/SafetyNet 仍會修正危險 action，但模型本身不會透過 OCC 充分學到「少做會被修正的 raw action」。
- 形成表面上有 `CRTSN + OCC`，但 OCC 對 policy internalization 幫助不足的 bug。

處理：
- 已移除 actor loss 中 OCC 項的斷梯度。
- actor 更新時暫時 freeze OCC head 參數，只讓 OCC 對 `actions_pi` 的梯度回傳到 actor，避免 actor optimizer 污染 OCC head。
- replay buffer 另外保存 `occ_action`，讓 SAC critics 繼續使用實際執行的 safe action，但 OCC head 學的是 raw policy action 對應的 CORAL/opportunity cost。

狀態：
- 已修正於 `core/sac_agent.py` 與 `core/train_sac_microgrid.py`。
- 既有 `v8b` 訓練結果不會自動改變；需要重新訓練才會看到 SafetyNet 介入率是否下降。

### 1. 不可把太陽能與市電簡化成單一二元切換

問題：
- 早期曾想用簡單 boolean、瞬時電壓大小或「誰比較大」來判定現在是不是太陽能供電。
- 但真機市電不是完美常數電壓，而且實際上可能出現太陽能與市電共同支援負載。

後果：
- PV 狀態在臨界區間會抖動。
- RL 看到的限制變成非平穩，單日也許還行，但跨日容易崩。
- 圖表也很容易誤導，好像只有單一來源在供電。

處理：
- 拆成兩個觀測概念：
  - `pv_bool`：是否達到「足以視為 PV active」的條件。
  - `pv_support_ratio`：PV 對負載的連續支援比例。

狀態：
- 已實作，且目前是主線設定。

### 1-1. Situation 2 / Grid Support 不應作為正式放電情境

問題：
- 部署資料顯示，`situation_code=2` 時雖然模型命令放電、電池電流也可能變成負值，但市電功率沒有退場，甚至可能升高。
- 這代表 Grid Support 不是「電池部分支援負載」，而是電池被市電拓撲壓住後形成的無效放電/耗電風險。

處理：
- 部署端在寫入 `Command.txt` 前，若放電無法形成 `Scenario 1`，直接改成待機，不再輸出 `Scenario 2`。
- 模擬環境與測試同步更新：無效放電被歸為 `Scenario 4`，避免訓練與驗證統計把它當成合法情境。
- SoC/電流方向推估也只把 `Scenario 1` 視為放電命令。

狀態：
- 已修正於 `control/run_deployment.py`、`core/microgrid_env.py` 與相關測試。

### 2. 電池不能被當成與市電/PV 並聯的第三個部分支援來源

問題：
- 使用者明確指出：太陽能與市電因電壓相近，可以互補支援。
- 但電池放電端約 `5.6V`，**不能**像太陽能與市電那樣做並聯式部分支援。

結論：
- 只要是「放電」這件事，電池必須符合 `solo_only` 的物理精神：
  - 電池若放電，就必須能單獨 cover 當下負載。
  - 若不能獨供，就不該掉 SoC。

後果：
- 先前把 `partial_assist` 當成改善穩定性的主線方向，對真機而言是物理不對齊。
- 先前某些看似穩定的結果，不應直接宣稱部署正確。

處理：
- 已將環境主線改回 `solo_only`。
- 單元測試與設定檔也已同步更新，不再把 battery partial assist 當成合法主線。

狀態：
- 已修正，這是目前不可退讓的硬體前提。

### 3. 只 warm-start actor、不 warm-start critics，會嚴重漂移

問題：
- 先前 warm-start 只載入 actor，但 critics 仍是隨機初始化。

後果：
- 原本還可以的 actor 很容易被亂掉的 critics 拖壞。
- 出現「起手看似可以，但越訓越怪」。

處理：
- 新增 warm-start 模式：
  - `actor_only`
  - `actor_critics`
  - `full_agent`

結果：
- `actor_critics` 明顯比 `actor_only` 穩。

狀態：
- 已修正，長訓練主線目前都用完整 warm-start。

### 4. 長訓練退化不只和 reward 有關，也和更新節奏有關

問題：
- 即使 reward 稍微調整，長訓練後段仍常退化。
- 其中一個原因是 critics、actor、alpha 同步高頻更新，actor 太早被拖著跑。

處理：
- 在 `core/sac_agent.py` 新增：
  - `actor_update_interval`
  - `alpha_update_interval`
  - `actor_warmup_updates`
  - `freeze_alpha`

結果：
- 長訓練穩定度有改善。
- 最佳點往後推。
- final 不再像更早期版本那樣嚴重崩落。

狀態：
- 部分解決，但還沒徹底解決跨日行為怪異的問題。

### 5. 訓練與部署的電池效率語意必須完全一致

問題：
- 曾發現訓練端與部署端對 `battery_efficiency = 0.95` 的解讀不同。
- 訓練環境把它當成單程效率，部署端一度把它當成 round-trip efficiency 再開根號。

結論：
- 主線定義應為：`0.95` 是**單程效率**。
- round-trip efficiency 約為 `0.95 * 0.95 = 0.9025`。

處理：
- 已修正 `control/run_deployment.py` 的 `SoCTracker`。
- 部署端與測試已對齊訓練環境。

狀態：
- 已修正。

### 6. 電價來源不是主要問題；問題更可能出在 reward 如何使用它

使用者疑問：
- `price_norm` 是否多餘？
- 能不能不要讓模型自己「從時間猜電價」，而是直接給它固定 TOU 表查出的單一 scalar price？

確認結果：
- 目前資料流中的 `price` 本來就是由固定 TOU 規則 `get_taipower_tou_price(hour, is_wknd)` 產生。
- 也就是說，主線其實早就已經是在用「由固定 TOU 表查出的單一 scalar price」。

結論：
- 問題不在於 price 來源不夠固定。
- 問題更可能在 reward shaping 對 price 的誘導方式，讓模型出現不符合經濟直覺的動作。

### 7. 強太陽日中的「保守充電」不一定是壞事，可能是穩健策略

觀察：
- 在 `2026-04-10` 這類強太陽日中，模型不是一看到 PV 變強就立刻用最大充電功率灌電池。
- 它常呈現出先小充、再逐步加大的樣子，看起來像是「不夠果斷」。

拆解後發現：
- 這種行為不一定代表模型不懂充電。
- 更合理的解釋是：模型學到一種偏保守但穩健的策略：
  - 先小充；
  - 確認 PV 還撐得住；
  - 再慢慢加大充電功率。

為什麼會這樣：
- 在大多數日子裡，真正可用來充電的 PV 餘電很少，而且集中在少數時段。
- 如果一開始就把充電功率拉太高，很容易讓市電補進來一起充電。
- 目前 reward 與 teacher 也沒有把「只要有餘電就盡量滿功率吃掉」教得非常強。

如何解讀這種圖：
- 不能單純把「沒有滿功率充電」解讀為失敗。
- 若該時段的 `pv_to_battery` 持續為正，且 `grid_draw_w` 很低甚至為零，
  就比較像是在穩健利用 PV 餘電，而不是在亂充。

可用於論文的說法：
- 模型在高太陽時段呈現保守但穩健的漸進式充電行為。
- 其充電策略不是盲目追求最大充電功率，而是傾向先確認 PV 餘電足以支撐，再逐步提升充電力度，以降低市電補入的風險。

### 8. 但保守策略若走過頭，也會變成「明明有餘電卻沒有好好充」

觀察：
- 在 hybrid 連續多日驗證中，`Day A+3`（對應分析日 `2026-05-07`）出現更值得警覺的情況：
- 當日其實屬於強太陽模板日，中午前後有明顯 PV 餘裕，且 SoC 起點很低。
- 但模型仍只做了很小幅度的充電，SoC 大致只從 `0.105` 上升到 `0.181`，沒有把可見的 PV surplus 充分轉成儲能。

這代表什麼：
- 這已經不只是「保守但穩健」而已。
- 更精確地說，模型目前同時存在兩種可能都會出現的行為：
  - 在某些強太陽時段，呈現合理的漸進式保守充電；
  - 但在另一些本來應該更積極充電的情境下，又會保守過頭，導致明顯低度利用 PV 餘電。

論文上怎麼寫比較安全：
- 不宜把所有「小充、慢充」都宣稱成成功策略。
- 比較安全的說法是：
  - 模型整體傾向採取保守的充電策略；
  - 此策略在部分情境下可解讀為穩健；
  - 但在連續多日驗證中，仍可觀察到對明確 PV surplus 利用不足的案例，顯示模型尚未完全學會在「可安心充電」的時段積極回收太陽能。

實務建議：
- 若目前重點是論文呈現、而不是立刻再調模型，則正文中的跨日主圖應優先挑選「3 天連續且行為可解釋」的視窗。
- `5 天` 圖較適合放成補充驗證或 limitation 佐證，避免把目前尚未完全合理的長時窗行為硬當成主結果。

## 目前仍待處理的問題

### A. 經濟上不合理的無效充放電

使用者觀察：
- 在沒有太陽、且同一電價時段內，模型看起來仍有一波充放電或不必要的電池介入。
- 這在經濟上不合理，因為沒有免費 PV、也沒有價差套利，電池吞吐只會吃掉效率損耗。

目前判斷：
- 這個懷疑是合理且重要的主線問題。
- 目前 reward 並沒有真正懲罰「同價位內、無 PV 時的無效吞吐」。
- 因此模型可能把某些 reward shaping 當成比金流本身更重要的訊號。

目前最強的懷疑：
- `v16sp` 類 reward 中的離峰充電誘因，可能讓模型學成：
  - 即使這次充電本身不划算，
  - 甚至後面又在類似價位把電放掉，
  - 還是會先去充，因為 reward 會補分。

後續方向：
- 優先先做最小改動驗證，不要同時大改很多東西。
- 已新增一版只拿掉離峰充電 bonus 的設定：
  - `configs/experiments/p302/config_p302_v16sp_solo_antidrift_v6_no_offpeak_bonus.yaml`
- 這版保留「尖峰有電應考慮放電」的提醒，但不再主動鼓勵離峰充電。

狀態：
- `v6` 已啟動訓練，等待結果。

### B. 多日模擬仍會出現看不懂的充放電抖動

使用者觀察：
- 單日看起來不一定很差，但拉到 3 日、5 日後，行為常開始變得難以解釋。
- 其中包含：
  - SoC 高但不果斷放電。
  - 同價位時段內似乎有無意義充放。
  - 早上或非直覺時段也有電池介入。

目前判斷：
- 這不一定代表單一 bug，而可能是多個因素疊加：
  - `solo_only` 物理使可放電情境本來就少。
  - reward 對離峰充電/尖峰放電的塑形過度直接。
  - 模型在跨日 horizon 下仍沒有學到穩定且可解釋的能量規劃。

結論：
- 單日圖可以作為局部觀察，但不能當最終品質證據。
- 所有 thesis-ready 圖與部署候選，必須優先通過多日檢查。

### C. final model 仍常不如 best checkpoint

問題：
- 多次實驗都顯示 best checkpoint 的行為比 final model 更合理。

處理原則：
- 不要再預設 final model 才是主模型。
- 論文、圖表、部署候選與人工檢查都應以 best checkpoint 為主。

狀態：
- 這條經驗在多個舊實驗中仍成立，但不應機械套用到所有新版本。
- 在 `v16sp_guided_teacher_v5_hybrid50_2000_deployalign` 這一輪中，使用者已根據最新 deployment-aligned validation 圖選擇 `final` 作為 release 候選，因此本輪應以該 `final` 模型為準。

### C.1 本輪 release 的例外：deployment-aligned v5 採用 final

背景：
- `v16sp_guided_teacher_v5_hybrid50_2000_deployalign` 不是單純 reward 微調，而是把多個 sim-to-real 對齊項一起放進訓練主線：
  - deployment-style aggregation / state builder
  - continuous operation mode
  - deployment guard style
  - battery zero-response / response noise stress
- 訓練完成後，已另外產出 `best` 與 `final` 的單日與多日驗證圖供人工比對。

結論：
- 這一輪不是直接套用「永遠用 best」的舊規則，而是以最新 validation 圖人工審查後，選擇 `final_sac_model.pth` 作為 release 與明日部署版本。

補充：
- release 端的 `packaging/build_release.spec` 與 `_deploy.ps1` 也已同步改為指向這個 `final` 模型，避免打包時又誤包到舊版本或 `best` checkpoint。

### D. 最新 sim-to-real gap 盤點：哪些要在模擬端建模，哪些要在實體端修

背景：
- 使用者在 `2026-04-10 ~ 2026-04-20` 的部署總覽中觀察到：
  - 某些幾乎沒有太陽能的日子，模型仍會大幅充放電；
  - `4/17` 之後到 `4/20` 的控制行為又變得很不穩；
  - 中間還混有 `situation_code = 4`、電池疑似被移走、電壓異常等系統層問題。

核心結論：
- 「訓練資料來自真實資料」不等於「部署時的閉環行為就會自然對齊」。
- 目前至少同時存在兩類落差：
  - 模擬端尚未充分建模的閉環/觀測/硬體效果；
  - 實體端本身的量測、保護與流程問題。

#### D.1 應優先從模擬端建模的項目

1. episode reset 與真實長時間連續控制不一致
- 模擬環境每個 episode 都會 `reset()`：
  - `current_soc`
  - `prev_action_kw`
  - `_soc_obs_buffer`
  - episode 累積量
- 但部署端不是每 15 分鐘 reset 控制器，只是清 `DataBuffer`，`SoCTracker`、上一動作、保護狀態都持續延續。
- 後果：
  - 模型學到的是「短/中期 episode 決策」；
  - 真機考驗的是「跨多天連續閉環控制」。
- 建議：
  - 後續若要更貼近部署，應增加長連續 rollout 訓練/驗證；
  - 必要時加入「跨窗格不 reset 內部狀態」的模擬模式。

2. 時間特徵語意可能不完全一致
- 部署端 `hour / day_of_week` 直接來自真實時鐘。
- 模擬環境則是根據 episode step 推算時間位置。
- 若 episode 起點不是實際日曆時間對齊，模型在模擬中看到的「幾點、星期幾」可能只是相對時間。
- 建議：
  - 檢查訓練資料切片後，時間特徵是否與真實 timestamp 一致；
  - 後續若仍有 gap，應改成直接從 dataset timestamp 建 observation，而非只用 step 推算。

3. SoC 演化機制不一致
- 模擬環境中，SoC 主要由 `action_kw` 和效率直接更新。
- 部署端中，SoC 由 `SoCTracker` 以電流積分估算，受下列因素影響：
  - 電流符號模式 `current-mode`
  - 感測器是否回報 0
  - `synthetic` 電流推估
  - 長間隔截斷、斷線、異常值
- 建議：
  - 模擬端應逐步加入 SoC 估測誤差、電流方向錯誤、零值段、長間隔等壓力模型；
  - 至少在驗證階段要模擬「action != measured current effect」。

4. 真機觀測不是乾淨單點值，而是 15 分鐘聚合後的結果
- 部署端 state 不是直接吃單點真值，而是吃：
  - 15 分鐘 buffer 平均
  - 可能不完整的樣本數
  - 實測值與 fallback 混合
- 其中負載還有一個重要差異：
  - 若實測 `load_p_mean_mW` 太低，部署端會退回 schedule fallback。
- 建議：
  - 模擬/驗證端應加入 deployment-style aggregation；
  - 不要只用理想化的 `load/pv/price` 單步真值評估模型。

5. 放電控制權在真機上比模擬弱
- 部署端註解已明寫：放電量不完全由 AI 自由決定，會受負載與硬體條件制約。
- 若模擬仍讓 agent 以為自己可直接精準控制放電功率，就會學到真機不存在的 action-effect 關係。
- 建議：
  - 模擬端應繼續堅持 `solo_only` 與真實可放電條件；
  - 若有必要，再進一步建模「命令放電不等於實際可放電」。

6. 部署端 guard 太多，但訓練端未完整共同建模
- 真機最終行為會被下列機制改寫：
  - CORAL SafetyNet
  - `SoC <= 0.05` 強制充電
  - `SoC <= 0.10` 禁止放電
  - `SoC >= 0.90` 禁止充電
  - `pv_active` 時禁止放電
  - 電壓截止與日鎖定
- 若模擬端沒有把這些 guard 共同納入，policy 在 sim 學到的最優行為不一定是真機最終看到的行為。
- 建議：
  - 後續至少要在 validation / rollout 階段使用與部署一致的 guard；
  - 更進一步則是在訓練端把關鍵 guard 納入環境動力。

#### D.2 應優先從實體端 / 部署端修正的項目

1. 感測器與接線異常不是模型能自己克服的
- 部署資料中已出現：
  - 電池電壓不合理飆高
  - `V=0, I=0`
  - 市電電壓異常掉落
  - 電池疑似被移走或脫離正常接線
- 這些不是 reward 調一調就能解決。
- 建議：
  - 先把感測與接線可靠性視為 deployment blocking issue；
  - 若原始 telemetry 不可信，任何 sim-to-real 討論都會被污染。

2. 部署端控制流程是連續運行，不是每窗格自動 reset
- 完整重啟程式後，大部分記憶變數會清掉：
  - `SoCTracker`
  - `DataBuffer`
  - CORAL residual buffer
  - `last_action_kw`
  - 各種 cutoff/警告狀態
- 但若不中斷程式，只是繼續跑，則很多狀態會延續。
- 結論：
  - 若懷疑「前一段壞狀態拖到下一段」，不能只看 15 分鐘窗格，必須確認當時是否真的完整重啟。

3. `initial_soc` 與真實電池狀態未必完全一致
- 部署端 SoC 是從 `--initial-soc` 起始，再靠 `SoCTracker` 積分更新。
- 若啟動時輸入的初始 SoC 與真實電池實際狀態不同，後續所有 state 都會偏掉。
- 建議：
  - 啟動前要有一致、可信的 SoC 校正流程；
  - 不要只靠人工估一個大概值。

4. `current-mode` 是部署端特有風險來源
- 部署端支援：
  - `signed`
  - `invert`
  - `unsigned`
  - `synthetic`
- 這代表真機電流語意本身可能不穩定，甚至需要人工選模式補救。
- 結論：
  - 只要 `current-mode` 還需要靠人工切換，真機 SoC 估測就仍屬高風險區。

5. 實體端仍存在資料 fallback 與保護鎖定的隱性狀態
- 例如：
  - 低負載實測時回退到排程值
  - 電壓截止後可能整天鎖 standby
  - `last_sit_code` 會持續影響 synthetic current 推估
- 這些在部署端都屬於「隱性內部狀態」，若當下沒被明確記錄，很容易事後誤判為模型突然失常。
- 建議：
  - 後續 deployment log 應明確標記：
    - cutoff 是否 active
    - day lock 是否 active
    - current-mode
    - fallback 是否啟用

#### D.3 目前分類後的實務優先順序

第一優先：實體端先修
- 感測器可靠性
- 電池是否確實接上
- `initial_soc` 校正流程
- `current-mode` 與電流語意確認
- deployment log 把隱性保護狀態寫清楚

第二優先：模擬端補建模
- deployment-style aggregation
- 長時間連續閉環驗證
- 時間特徵與真實 timestamp 對齊
- SoC estimation noise / sensor fault / guard-in-the-loop 建模

第三優先：再談 reward / policy 微調
- 在前兩層未對齊前，若只繼續調 reward，很容易把系統層問題誤當成 policy 問題。

#### D.4 若廠商端資訊與韌體暫時動不了，應立即聚焦的可控項

前提：
- 廠商端目前難以快速修改。
- 因此短期內最有價值的方向，不是等待硬體變好，而是先把「收到的資訊如何處理成模型輸入」與「模擬端如何重現部署條件」做好。

可直接分成兩大塊：

| 類別 | 我們現在可做的事 | 目的 |
| --- | --- | --- |
| Deployment-side preprocessing | 固定並明文化 `Data.txt -> 15 分鐘 buffer -> aggregation -> state vector` 的流程 | 讓模型實際吃到的 observation 可追蹤、可重播、可除錯 |
| Simulation-side modeling | 在模擬/離線 replay 中重建 deployment 端的 aggregation、fallback、SoC estimation、guard | 讓訓練/驗證看到的不是理想值，而是接近真機的輸入與限制 |

##### D.4.1 接收到資訊後，應如何處理給模型

1. 把 deployment 前處理當成正式系統的一部分，而不是零散腳本
- 必須固定下列步驟：
  - 原始讀值輪詢頻率
  - 15 分鐘窗格對齊方式
  - aggregation 統計欄位
  - completeness 檢查
  - load fallback 條件
  - `pv_support_ratio / pv_bool / pv_active` 的最終定義
- 重點不是「算得快」，而是「每次都算得一樣」。

2. 所有 deployment state 都應可離線重播
- 任何一次真實部署，都應能從 raw log 重新產出：
  - 每個 15 分鐘窗格的 `agg`
  - 每一步最終 state vector
  - guard 介入前 action
  - guard 介入後 action
- 這樣才能真正回答：
  - 是模型本身錯？
  - 還是前處理把 state 變形了？

3. 將「隱性狀態」顯性化記錄
- 之後 deployment log 應明確記錄：
  - `current_mode`
  - `completeness`
  - `load_fallback_used`
  - `voltage_cutoff_active`
  - `voltage_cutoff_day_locked`
  - `pv_active_block_applied`
  - `soc_high_block_applied`
  - `soc_low_block_applied`
  - `coral_clipped`
- 否則事後很容易把 guard 造成的結果誤看成 policy 行為。

4. 將 state 建構流程抽成可共用模組
- 理想狀態是：
  - 訓練驗證可呼叫同一套 state builder
  - deployment 也呼叫同一套 state builder
- 若訓練/部署各自維護自己的 state 邏輯，後面一定又會慢慢漂開。

##### D.4.2 模擬端應優先補哪些建模

1. 先補 deployment-style aggregation replay
- 不是直接餵乾淨 CSV 欄位給環境，
- 而是先把資料做成像部署端一樣的 15 分鐘聚合 state。
- 這是最優先、也最不依賴廠商的改進。

2. 補 load fallback 與缺測條件
- 若真機在某些條件下會從實測負載退回排程負載，
- 模擬/驗證端也應該能重現這件事。
- 否則模型在 sim 看到的 load 與 real 看到的 load 根本不是同一種訊號。

3. 補 SoC estimation noise / fault model
- 不需要一開始就做很複雜。
- 最少可先加入：
  - 電流為 0 的區段
  - 電流符號錯誤
  - 長間隔截斷
  - 初始 SoC 偏差
- 這比繼續微調 reward 更可能解釋部署時的偏差。

4. 補 guard-in-the-loop validation
- 至少在驗證與 rollout 圖中，應讓下列機制一起生效：
  - `soc <= 0.10` 禁止放電
  - `soc >= 0.90` 禁止充電
  - `pv_active` 禁止放電
  - CORAL / projection
  - voltage cutoff 類保護
- 若 policy 評估時不帶 guard，就不是真正的 deployment behavior。

5. 補長時間連續控制驗證
- 不能只看單日或每回合 reset。
- 必須加入：
  - 連續 3~5 天
  - 不重置內部狀態的 replay
  - 觀察 SoC、action、guard 觸發是否慢慢漂掉

##### D.4.3 立即執行順序（只做我們能控制的部分）

1. 先建立「raw log -> aggregated state -> final deployed state/action」的離線重播工具
- 目標：任何一天部署都能精準重建模型真正看到的 state。

2. 再建立 deployment-style validation pipeline
- 用和真機一致的 state builder + guard，離線回放歷史資料。

3. 再把這套 pipeline 接回模擬端
- 讓訓練/驗證不再只依賴理想化 CSV 欄位。

4. 最後才決定 reward / actor / teacher 還要不要繼續改
- 若前處理與模擬條件還沒對齊，太早改 reward 很容易改錯方向。

一句話總結：
- 既然短期內動不了廠商端，就應把主力放在「把部署端 observation pipeline 完全吃透、固定、可重播」，再把它搬回模擬端。

### E. `2026-04-22 ~ 2026-04-24` 新部署觀察：不能只看 final action，必須分開看 raw policy 與安全層修正

背景：
- 使用者補上 `4/21 ~ 4/24` 的新資料後，觀察到：
  - `4/22` 下午看起來沒有成功大膽充電；
  - `4/22` 午夜卻有一小段充電；
  - `4/23`、`4/24` 看起來幾乎完全沒有動作。

分析後的更精確結論：

1. `4/22` 下午不是「明明有很大 PV surplus 卻不敢充」
- 部署 log 顯示當天下午最強的幾個窗格雖然 `pv_support_ratio` 接近 `1.0`，
  但 `pv_kw - load_kw` 仍幾乎全程為負。
- 也就是說：
  - PV 幾乎能 cover load；
  - 但沒有真正明確的 surplus 可供電池放心充電。
- 因此當天下午只出現小幅充電，不能直接判為錯誤；更準確的說法是：
  - 模型面對「接近足供但未形成明確餘電」的情境時仍偏保守。

2. `4/22` 午夜確實有夜間補電行為
- `22:00` 之後，raw / final action 轉為小幅正值，SoC 由約 `0.10` 拉升到約 `0.20`。
- 這代表：
  - 模型在 TOU 轉便宜後，確實會做小幅夜間補電；
  - 此行為未必違規，但其經濟合理性仍需進一步討論。

3. `4/23`、`4/24` 不能簡化成「陰天所以不動」
- 真實 log 顯示：
  - `pv_kw` 幾乎為零；
  - 但 raw policy 多數時間其實想放電；
  - final action 則被裁成接近 `0`。
- 也就是說：
  - 看起來「沒動作」，
  - 不代表 policy 主動學會待機，
  - 很可能只是安全層一直在幫它踩煞車。

重要教訓：
- 後續 deployment 分析不能只看 final action。
- 必須同時看：
  - `action_raw_kw`
  - `action_power_kw`
  - `coral_clipped`
  - `guard_delta_mW`
  - 各 guard flag
- 否則容易把「被 safety layer 擋下來」誤判成「policy 自己很安全」。

### F. 真正的問題不是 final action 安不安全，而是 raw policy 仍過度依賴邊界 / CORAL / guard

使用者觀察：
- 模型不應該一直靠邊界修正才不過充過放。
- 例如 `SoC ≈ 9.3%` 的區段，即使 final action 被擋成 `0`，也不能算 policy 成功。

目前判斷：
- 這個觀點是正確的。
- 若 raw policy 在低 SoC 時仍持續提出放電動作，代表：
  - policy 尚未內化安全邊界；
  - runtime safety layer 只是在部署端替 policy 擦屁股。

實務解讀：
- 不能只用 final behavior 宣稱安全。
- 後續驗證至少應同時追蹤：
  - `guard intervention rate`
  - `coral clipped rate`
  - `mean |raw-final|`
  - 低 SoC 區間的 raw discharge tendency
  - 高 SoC 區間的 raw charge tendency

後續方向：
- 下一輪訓練的目標不只是讓 final action 安全，
  而是要讓 policy 自己少提出會被修正的 raw action。
- 具體可做法包括：
  - 對 `|raw-final|` 或 `delta_kw / Pmax` 加 penalty；
  - 增加低 SoC / 高 SoC 起點覆蓋；
  - 將低 SoC 想放電、高 SoC 想充電，直接視為 unsafe tendency。

### G. `pv/load > 0.8` 目前只能算 heuristic，不足以當作「可放心充電」的正式背書

背景：
- 目前 `pv_bool` 的主線定義採用 `pv/load >= 0.8`。
- 這原本是使用者提出的一個相對保守 heuristic。

最新理解：
- `pv/load >= 0.8` 比較適合表示：
  - PV 已相當活躍；
  - 或接近足供負載。
- 但它不等價於：
  - 有高可信度的 PV surplus 可供電池充電。

新部署資料給出的證據：
- `4/22` 下午多個窗格 `pv_support_ratio` 接近 `1.0`，
  但 `pv_kw - load_kw` 仍為負。
- 這證明：
  - `pv/load >= 0.8` 不能直接當成「現在安全可充」的證據。

目前較合理的分類方式：
- `pv_active`：
  - 表示 bus 上實際有 PV 參與，適合用於 deployment guard。
- `pv_support_ratio >= 0.8`：
  - 可保留為「接近足供」的 soft hint。
- 真正決定是否可放心充電的訊號：
  - 應改成看保守 surplus 是否為正，
  - 例如 `pv_lower_bound - load_upper_bound >= charge_margin`。

狀態：
- 目前還沒有足夠晴天 deployment 樣本，可用經驗法反推出可靠的充電門檻。
- 因此短期內：
  - `0.8` 可暫時保留為 heuristic；
  - 但不應在論文或方法章中把它寫成被正式證明的最終充電門檻。

### H. CORAL / conformal-adaptive loop 在訓練端與部署端的落地程度不一致

背景：
- 使用者注意到：
  - 專案裡明明有 conformal residual buffer、tube、adaptive boundary 相關變數；
  - 但 deployment 行為看起來像是只剩 static clipping。

檢查後的結論：

1. 訓練端的 module-level `project(...)` 確實有接上 conformal tube
- 在 `core/safety_net.py` 的模組函式 `project(...)` 中，
  `_conformal_tube()` 會被轉成 `soc_min_eff / soc_max_eff`。
- 也就是說：
  - 模擬 / 訓練迴圈中的 safety projection，
  - 比較接近有 adaptive tube 的版本。

2. 部署端使用的 `SafetyNet` class 版本則沒有把 tube 真正接上
- deployment loop 會：
  - 更新 `coral_delta`
  - 更新 residual buffer
  - 記錄 `coral_residual_count`
- 但 `SafetyNet.project()` / `bounds()` 目前：
  - 主要只使用靜態 `safe_soc_min / safe_soc_max`
  - 沒有把 `_conformal_tube()` 用進去
  - 也沒有在 deployment 主迴圈中呼叫 `update_buffer_after_episode()`

因此目前更準確的描述應是：
- 訓練端：
  - `adaptive safety projection` 有部分落地
- 部署端：
  - `static SafetyNet projection + conformal diagnostics`
  - 尚未形成 fully adaptive conformal loop

重要影響：
- 目前不能把 deployment 版 CORAL 寫成完整的 adaptive boundary system。
- 論文若要誠實描述，應說：
  - deployment 端已有 residual accumulation 與 action correction；
  - 但 residual-driven boundary adaptation 尚未 fully wired in。

## 最近幾輪實驗與得到的發現

### 待跑：公平比較安全思考能力與部署表現

目標：
- 論文賣點不是「外部 hard guard 把所有模型都修安全」，而是模型本身能學會少做危險 raw action。
- 同時不能為了安全犧牲過多經濟效益，所以比較時必須同時看安全內化程度與 profit / energy-management 成效。

比較設計 1：Raw Policy Comparison
- 所有方法使用同一資料集、同一環境、同一 SoC 邊界、同一訓練 episode 數與同一 validation dates。
- 評估重點放在 raw policy 本身，不把外部 safety layer 修正後的結果當成唯一成績。
- 主要指標：
  - `violations_attempted`
  - raw action 是否常碰 SoC / PV / load feasibility 邊界
  - raw policy 的 `net_profit`、`PV-to-battery`、`useful_discharge`
- 這組比較回答：「哪個模型本身比較懂安全？」

比較設計 2：Deployment-Aware Comparison
- 所有 baseline 在部署/驗證時都套同一套 hard safety layer，因為真機不可能允許 unsafe action 直接輸出。
- 這個 safety layer 視為硬體保護與部署平台條件，而不是某個模型的專屬優勢。
- 主要指標：
  - `safety_projected_meaningful`
  - `projection_delta_mean_w`
  - `projection_delta_max_w`
  - final safe policy 的 `net_profit` / `cost saving`
  - `PV-to-battery`、`useful_discharge`
- 這組比較回答：「在所有方法都必須安全部署的前提下，誰比較少依賴安全層、又能保有利潤？」

公平性原則：
- 若「同一個方法」只是同一套 deployment hard guard，套給所有 baseline 是公平的。
- 若「同一個方法」包含 `CRTSN` / `OCC` / adaptive loop，則它是本研究方法本體，不能直接套給 baseline 後再說 baseline。
- 應做 ablation：
  - SAC
  - SAC + deployment hard guard only
  - SAC + reward safety penalty
  - SAC + SafetyNet projection during training
  - SAC + SafetyNet + OCC
  - SAC + SafetyNet + OCC + adaptive loop（完整方法）

狀態：
- 尚未正式跑 baseline/ablation。
- 之後所有比較都應同步報告「安全內化」與「經濟效益」，避免只用 reward 或只用 violation 其中一邊講故事。

### `v16sp_solo_antidrift_v3_longwarmup1000`

- 設計：
  - `solo_only`
  - 長 warmup
  - scalar `price_obs`
  - anti-drift 更新節奏
- 結果：
  - `Best eval reward`: `133.5564`
  - `Final eval reward`: `83.7218`
  - `Avg SoC violations`: `56.15`
- 解讀：
  - 這是目前表面指標最好的一條主線。
  - 但使用者從 3 日/5 日圖仍觀察到可疑的充放電行為，因此不能只看 reward 就宣稱成功。

### `v16sp_solo_antidrift_v4_noprice_longwarmup1000`

- 設計：
  - 直接拿掉 scalar `price_obs`
- 結果：
  - `Best eval reward`: `-2.7324`
  - `Final eval reward`: `-17.7387`
  - `Avg SoC violations`: `50.62`
- 解讀：
  - 表現明顯很差。
  - 說明「把 price 特徵直接拿掉」不是正確方向。

### `v16sp_solo_antidrift_v5_touonehot_longwarmup1000`

- 設計：
  - 拿掉 scalar price，改成 TOU one-hot 特徵
- 結果：
  - `Best eval reward`: `1.7973`
  - `Final eval reward`: `-17.5820`
  - `Avg SoC violations`: `50.54`
- 解讀：
  - 也很差。
  - 說明把已知電價帶拆成 one-hot，既增加 state 維度，也沒有帶來更好的學習結果。

### `v16sp_solo_antidrift_v6_no_offpeak_bonus`

- 設計：
  - 保留 scalar `price_obs`
  - 保留 `solo_only`
  - 把 `v17_offpeak_charge_bonus` 從 `0.20` 降為 `0.0`
  - 保留 `v17_peak_discharge_bonus = 0.35`
  - 保留 `v17_peak_idle_penalty = 0.12`
- 目的：
  - 驗證「模型是否因為離峰充電 bonus，而寧可做出經濟上不合理的充放電」。
- 狀態：
  - 已啟動訓練，結果待補。

## 公平 baseline / ablation 比較原則

我們的方法不能被描述成「只是靠外部 SafetyNet 保護」。核心論點應該是：

> 外部 safety layer 是真機部署必要條件，但我們要證明 policy 本身也學會少做危險 action，並且在安全後仍保有經濟效益。

因此比較時不能只看「最後有沒有違規」。如果所有方法最後都套同一層 hard guard，大家都可以被修到安全。真正要看的指標是 raw policy 是否已經內化安全限制，以及安全修正後是否仍有用。

### 主要評估指標

- `violations_attempted`：模型原始意圖有多常想違規。
- `safety_projected_meaningful`：SafetyNet 實際介入多少次。
- `projection_delta_mean_w` / `projection_delta_max_w`：每次被修正的平均與最大幅度。
- `net_profit` / `grid_savings_twd`：安全後是否還保有經濟效益。
- `pv_to_battery_wh`：是否真的利用太陽能充電。
- `useful_discharge_wh`：是否真的有效使用電池供負載。
- `soc_min` / `soc_max`：是否落在安全區間內，不只靠 hard clip 修掉。

### 1. Raw Policy Comparison

目的：

> 回答「哪個模型本身比較懂安全？」

做法：

- 所有方法使用同一資料、同一環境、同一 SoC 邊界、同一訓練 episode 數。
- 評估時重點看 raw action，不把外部 SafetyNet 的修正當成方法成績。
- 可以允許統一記錄 projected action，但主表要清楚分 raw 與 safe。

適合比較的方法：

- `SAC`
- `SAC + reward safety penalty`
- `SAC + guided teacher`
- `PPO` / `PPO-Lagrangian`
- `DDPG` / `TD3`（若時間允許）
- 我們的方法的 raw policy

重點指標：

- `violations_attempted`
- raw action 對 SoC 邊界的距離
- raw charge/discharge timing
- raw policy 是否在低 SoC 還想放電、高 SoC 還想充電
- raw policy 是否在 PV 明顯不足時亂充/亂放

### 2. Deployment-Aware Comparison

目的：

> 回答「在真實部署都必須安全的前提下，誰比較少依賴安全層、又能保有利潤？」

做法：

- 所有方法部署時都套同一套 hard safety layer / SafetyNet。
- 這是公平的，因為真機不可能放任 baseline 直接輸出危險動作。
- SafetyNet 在這裡是實驗平台與硬體保護，不是某個演算法的專屬加分。
- 但必須額外報告誰被修正最多、修正幅度最大。

重點指標：

- `safety_projected_meaningful`
- `projection_delta_mean_w`
- `projection_delta_max_w`
- `net_profit`
- `grid_savings_twd`
- `pv_to_battery_wh`
- `useful_discharge_wh`
- `situation_code = 1` 的有效放電次數

### 什麼算公平，什麼不公平

公平：

- 讓所有 baseline 都套同一個外部 deployment hard guard。
- 讓所有方法用同一套 SoC 上下界、電池功率限制、PV/load 資料、TOU 價格。
- 讓所有方法在同一批日期上 rollout。

不公平：

- 把 `OCC`、`adaptive loop`、`CRTSN residual learning` 也直接加到 baseline 上，然後說 baseline 表現如何。
- 因為這些是我們方法的核心，不是單純的硬體保護層。

### 建議 ablation 組合

為了證明不是靠 safety layer 作弊，應該逐步比較：

1. `SAC`
2. `SAC + hard safety at deployment only`
3. `SAC + reward safety penalty`
4. `SAC + guided teacher`
5. `SAC + SafetyNet projection during training`
6. `SAC + SafetyNet + OCC`
7. `SAC + SafetyNet + OCC + adaptive loop`，也就是完整方法

如果時間有限，最低限度也要保留：

1. `SAC`
2. `SAC + SafetyNet`
3. `PPO-Lagrangian`
4. `PPO + SafetyNet`
5. `OURS`

### 報告時的核心說法

可以這樣說：

> We do not evaluate safety only by final violations, because all deployed controllers require a hard protection layer. Instead, we report how often the raw policy attempts unsafe actions, how much the safety layer needs to correct it, and whether the corrected policy still preserves economic value.

中文說法：

> 我們不是只看最後有沒有違規，因為真機部署一定會有硬體保護。真正公平的比較是：模型原始策略本身多常想做危險動作、安全層實際介入多少、修正幅度多大，以及修正後是否仍能保有節費與太陽能利用效益。

## 當前建議

如果下一輪要繼續，優先順序建議如下：

1. 先驗證 `v6` 是否真的減少同價位內的無效充放電。
2. 若 `v6` 有改善，再決定是否需要額外補 throughput penalty。
3. 若 `v6` 沒改善，再考慮更明確地把「無 PV、無價差時的電池吞吐」視為壞行為。
4. 所有比較都優先看多日 rollout，而不是只看單日圖或單一 eval reward。
5. 長訓練流程仍以 best checkpoint 為主，不追求 final 一定最好看。

## 文件維護規則

之後每次遇到新問題，至少要補三件事：

- 問題是什麼。
- 我們怎麼確認它存在。
- 我們試了什麼，結果如何。

不要只寫「改過了」，要寫清楚「為什麼改、改了有沒有用」。

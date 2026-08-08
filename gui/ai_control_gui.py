#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P302 微電網 AI 控制介面 — CORAL Framework
==========================================
模式：
  1. AI 控制 (CORAL)     → control/run_deployment.py
  2. 太陽能測試 (收資料)  → control/solar_test_collect.py
  3. 待機 / 手動         → control/solar_test_collect.py (scenario 3, zero power, rest 0% flow)

SLFB 電池規格（V16，P302 主控板）：
  功率  : 5.6W (5.6V × 1A)，容量 2000mAh = 11.2Wh
  放電  : 1A × 5.6V = 5.6W  (自動依負載決定)
  充電  : 1A × 8.5V = 8.5W  (可控制)
  截止  : 4.2V 以下切市電
  負載  : GUI 顯示暫沿用 4 組 × 0.1W = 0.4W；0.58W/燈條對齊待後續處理
  效率  : 95% RTE
"""

import os
import csv
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
from datetime import datetime

# ======================================================================
# ======================================================================
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(GUI_DIR)
APP_ROOT = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else PROJECT_ROOT
CONFIG_FILE = os.path.join(APP_ROOT, "config_gui.json")

CONTROL_DIR = os.path.join(PROJECT_ROOT, "control")
DEPLOYMENT_SCRIPT = os.path.join(CONTROL_DIR, "run_deployment.py")
SOLAR_TEST_SCRIPT = os.path.join(CONTROL_DIR, "solar_test_collect.py")


def _default_model_path() -> str:
    candidates = [
        os.path.join(APP_ROOT, "models", "best_sac_model.pth"),
        os.path.join(APP_ROOT, "_internal", "models", "best_sac_model.pth"),
        os.path.join(PROJECT_ROOT, "models", "best_sac_model.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[1] if getattr(sys, "frozen", False) else candidates[0]


def _default_soh_model_path() -> str:
    candidates = [
        os.path.join(APP_ROOT, "soh_models"),
        os.path.join(APP_ROOT, "_internal", "soh_models"),
        os.path.join(PROJECT_ROOT, "soh_models"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[1] if getattr(sys, "frozen", False) else candidates[0]


def _default_log_dir() -> str:
    return os.path.join(APP_ROOT, "results", "deployment")


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False

P302_DEFAULTS = {
    "vendor_dir": "",
    "vendor_exe": "",
    "model_path": _default_model_path(),
    "soh_model_path": _default_soh_model_path(),
    "initial_soc": 20.0,
    "load_count": 4,
    "log_dir": _default_log_dir(),
    "device": "cpu",
    "poll_sec": 10.0,
    "window_min": 15,
    "current_mode": "hybrid",
    "manual_scenario": 3,
    "dry_run_enabled": True,
    "pv_surplus_charge_only": True,
    "cutoff_soc_fallback_enabled": True,
    "cutoff_soc_fallback_percent": 20.0,
    "soh_prediction_enabled": False,
    "soh_use_for_capacity": False,
    "soh_health_protection_enabled": False,
    "soh_low_voltage_v": 4.2,
    "soh_low_voltage_samples": 3,
    "soh_recover_v": 5.0,
    "soh_recovery_samples": 12,
    "use_watchdog": True,
    "watchdog_interval_sec": 60,
}


class ConfigManager:
    @staticmethod
    def load():
        defaults = dict(P302_DEFAULTS)
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    defaults.update(json.load(f))
            except Exception:
                pass
        if not defaults.get("log_dir") or not _is_writable_dir(defaults["log_dir"]):
            defaults["log_dir"] = _default_log_dir()
        return defaults

    @staticmethod
    def save(config):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False


# ======================================================================
# ======================================================================
def check_vendor_running(vendor_exe_path: str) -> bool:
    try:
        exe_name = os.path.basename(vendor_exe_path).lower()
        out = subprocess.check_output(
            ["tasklist"], encoding="utf-8", errors="ignore", timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        return exe_name in out.lower()
    except Exception:
        return False


# ======================================================================
# ======================================================================
class ScriptProcess:
    """管理腳本進程，支援即時日誌"""

    def __init__(self, name: str, cmd: list, log_callback=None):
        self.name = name
        self.cmd = cmd
        self.proc = None
        self.log_callback = log_callback
        self._stop_logging = False

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _read_output(self, pipe):
        try:
            while not self._stop_logging:
                line = pipe.readline()
                if not line:
                    if self.proc and self.proc.poll() is not None:
                        remaining = pipe.read()
                        if remaining and self.log_callback:
                            self.log_callback(remaining if isinstance(remaining, str) else remaining.decode('utf-8', errors='replace'))
                    break
                text = line.rstrip() if isinstance(line, str) else line.decode('utf-8', errors='replace').rstrip()
                if text and self.log_callback:
                    self.log_callback(text + "\n")
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def start(self, cwd: str = None):
        if self.is_running():
            return False
        try:
            self._stop_logging = False
            self.proc = subprocess.Popen(
                self.cmd, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                bufsize=0, text=True, encoding='utf-8', errors='replace',
            )
            t = threading.Thread(target=self._read_output, args=(self.proc.stdout,), daemon=True)
            t.start()
            time.sleep(0.5)
            if self.proc.poll() is not None:
                if self.log_callback:
                    self.log_callback(f"[ERROR] Process exited immediately (code={self.proc.returncode})\n")
                return False
            return True
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"[ERROR] Failed to start process: {e}\n")
            return False

    def stop(self):
        if not self.is_running():
            return
        self._stop_logging = True
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except Exception:
            pass
        finally:
            if self.log_callback:
                self.log_callback(f"[{datetime.now().strftime('%H:%M:%S')}] [{self.name}] 已停止\n")
            self.proc = None


# ======================================================================
# ======================================================================
class AIControlGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("P302 微電網 AI 控制 V16 — CORAL Framework")
        self.geometry("1400x850")

        self.config = ConfigManager.load()
        self.vendor_status_var = tk.StringVar(value="--")
        self.scenario_status_var = tk.StringVar(value="已停止")
        self.soh_status_var = tk.StringVar(value="SoH: --  |  time: --")
        self.current_scenario = None
        self.vendor_proc = None

        # Watchdog
        self.watchdog_thread = None
        self.watchdog_stop_flag = threading.Event()
        self.watchdog_cmd_backup = []
        self.watchdog_project_root = ""
        self.watchdog_restart_count = 0

        self.last_data_content = ""
        self.last_command_content = ""
        self._last_soh_status_scan = 0.0

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.after(100, self._update_vendor_status)
        self.after(300, self._poll_status)
        self.after(500, self._poll_data_command)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = 6
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 10, "bold"))
        style.configure("Info.TLabel", font=("Consolas", 9))
        style.configure("Mode.TRadiobutton", font=("Microsoft JhengHei UI", 10))
        style.configure("Start.TButton", font=("Microsoft JhengHei UI", 10, "bold"))

        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=pad, pady=pad)

        left = ttk.Frame(main_paned)
        main_paned.add(left, weight=1)
        self._build_left_panel(left, pad)

        right = ttk.Frame(main_paned)
        main_paned.add(right, weight=1)
        self._build_right_panel(right, pad)

    def _build_left_panel(self, parent, pad):
        """左欄：廠商程式設定 + 日誌 + Data/Command 顯示"""
        frm = ttk.LabelFrame(parent, text="廠商程式 (P302)")
        frm.pack(fill="x", padx=pad, pady=(pad, 2))

        row = ttk.Frame(frm)
        row.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row, text="程式資料夾:").pack(side="left")
        self.vendor_dir_var = tk.StringVar(value=self.config.get("vendor_dir", ""))
        ttk.Entry(row, textvariable=self.vendor_dir_var, width=35).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row, text="瀏覽", command=self._browse_vendor_dir).pack(side="left")

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row2, text="EXE 路徑:   ").pack(side="left")
        self.vendor_exe_var = tk.StringVar(value=self.config.get("vendor_exe", ""))
        ttk.Entry(row2, textvariable=self.vendor_exe_var, width=35).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row2, text="瀏覽", command=self._browse_vendor_exe).pack(side="left")

        row_s = ttk.Frame(frm)
        row_s.pack(fill="x", padx=pad, pady=4)
        ttk.Label(row_s, text="狀態:").pack(side="left")
        ttk.Label(row_s, textvariable=self.vendor_status_var, width=10).pack(side="left", padx=4)
        ttk.Button(row_s, text="檢查", command=self._update_vendor_status).pack(side="left", padx=2)
        ttk.Button(row_s, text="啟動", command=self._launch_vendor).pack(side="left", padx=2)
        ttk.Button(row_s, text="開啟資料夾", command=self._open_vendor_folder).pack(side="left", padx=2)

        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=pad, pady=(2, pad))

        frm_log = ttk.Frame(nb)
        nb.add(frm_log, text="執行日誌")
        self.log_text = scrolledtext.ScrolledText(frm_log, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.log_text.config(state=tk.DISABLED)

        frm_dc = ttk.Frame(nb)
        nb.add(frm_dc, text="Data / Command")
        paned = ttk.PanedWindow(frm_dc, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=2, pady=2)

        frm_d = ttk.LabelFrame(paned, text="Data.txt (韌體輸入)")
        paned.add(frm_d, weight=1)
        self.data_text = scrolledtext.ScrolledText(frm_d, wrap=tk.NONE, font=("Consolas", 9), height=8)
        self.data_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.data_text.config(state=tk.DISABLED)

        frm_c = ttk.LabelFrame(paned, text="Command.txt (AI 輸出)")
        paned.add(frm_c, weight=1)
        self.cmd_text = scrolledtext.ScrolledText(frm_c, wrap=tk.NONE, font=("Consolas", 9), height=8)
        self.cmd_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.cmd_text.config(state=tk.DISABLED)

    def _build_right_panel(self, parent, pad):
        """右欄：模式選擇 + 參數設定 + 控制按鈕"""
        canvas = tk.Canvas(parent, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        cw = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        frm_mode = ttk.LabelFrame(scrollable, text="運行模式")
        frm_mode.pack(fill="x", padx=pad, pady=pad)

        self.mode_var = tk.StringVar(value="ai")
        modes = [
            ("ai",    "1. AI 控制 (CORAL Framework)",
             "使用訓練好的 SAC 模型 + 安全框架\n"
             "15 分鐘聚合資料 → 推論 → 輸出 Command.txt\n"
             "包含：SoCTracker、DataBuffer、Scenario 1-4、TOU 電價"),
            ("solar", "2. 太陽能測試 (收 MPPT 資料)",
             "平時電池 0 power、0% flow (Scenario 3)；需要量測電壓時可用 50% pre-flow\n"
             "指定負載組數，市電自動補足\n"
             "資料存入 CSV，可用於重新訓練"),
            ("standby", "3. 待機 (僅維持通訊)",
             "電池 0 power、0% flow (Scenario 3)，負載可自訂\n"
             "AI 決策/量測前才以 50% 預流動約 20-25 秒"),
        ]

        for val, label, desc in modes:
            row = ttk.Frame(frm_mode)
            row.pack(fill="x", padx=pad, pady=2)
            rb = ttk.Radiobutton(row, text=label, variable=self.mode_var, value=val,
                                 style="Mode.TRadiobutton", command=self._on_mode_change)
            rb.pack(anchor="w")
            ttk.Label(row, text=desc, font=("Microsoft JhengHei UI", 8),
                      foreground="gray", justify="left").pack(anchor="w", padx=24)

        self.frm_ai = ttk.LabelFrame(scrollable, text="AI 控制參數")
        self.frm_ai.pack(fill="x", padx=pad, pady=2)

        row_m = ttk.Frame(self.frm_ai)
        row_m.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_m, text="模型檔案 (.pth):").pack(side="left")
        self.model_path_var = tk.StringVar(value=self.config.get("model_path", ""))
        ttk.Entry(row_m, textvariable=self.model_path_var, width=30).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_m, text="瀏覽", command=self._browse_model).pack(side="left")

        row_model_hint = ttk.Frame(self.frm_ai)
        row_model_hint.pack(fill="x", padx=pad, pady=1)
        ttk.Label(
            row_model_hint,
            text="部署建議：平時 rest 0%；AI 決策/量測前 mode 3 + 50% 預流動約 20-25 秒；active flow 由部署端保護為 60-100%。",
            foreground="gray",
            font=("Microsoft JhengHei UI", 8),
        ).pack(anchor="w")

        row_soc = ttk.Frame(self.frm_ai)
        row_soc.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_soc, text="初始 SoC (%):").pack(side="left")
        self.initial_soc_var = tk.StringVar(value=str(self.config.get("initial_soc", 50.0)))
        ttk.Entry(row_soc, textvariable=self.initial_soc_var, width=10).pack(side="left", padx=4)
        ttk.Label(row_soc, text="(SoCTracker 起始值，0-100)").pack(side="left")

        row_dev = ttk.Frame(self.frm_ai)
        row_dev.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_dev, text="推論裝置:").pack(side="left")
        self.device_var = tk.StringVar(value=self.config.get("device", "cpu"))
        ttk.Combobox(row_dev, textvariable=self.device_var, values=["cpu", "cuda"],
                     width=8, state="readonly").pack(side="left", padx=4)

        row_win = ttk.Frame(self.frm_ai)
        row_win.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_win, text="聚合窗格 (min):").pack(side="left")
        self.window_min_var = tk.StringVar(value=str(self.config.get("window_min", 15)))
        ttk.Entry(row_win, textvariable=self.window_min_var, width=8).pack(side="left", padx=4)
        ttk.Label(row_win, text="(預設 15 分鐘，每窗格推論一次)").pack(side="left")

        row_init = ttk.Frame(self.frm_ai)
        row_init.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_init, text="首次動作:").pack(side="left")
        self.initial_action_var = tk.StringVar(value="standby")
        ttk.Combobox(row_init, textvariable=self.initial_action_var,
                     values=["standby", "random"], width=10, state="readonly").pack(side="left", padx=4)
        ttk.Label(row_init, text="(首個 15 分鐘的動作)").pack(side="left")

        row_cur = ttk.Frame(self.frm_ai)
        row_cur.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_cur, text="電流模式:").pack(side="left")
        self.current_mode_var = tk.StringVar(value=self.config.get("current_mode", "hybrid"))
        cb_cur = ttk.Combobox(row_cur, textvariable=self.current_mode_var,
                     values=["hybrid", "signed", "invert", "unsigned", "synthetic"],
                     width=10, state="readonly")
        cb_cur.pack(side="left", padx=4)
        ttk.Label(row_cur, text="hybrid=實測優先(推薦) | synthetic=純合成 | unsigned=全正").pack(side="left")

        row_deploy_guard = ttk.Frame(self.frm_ai)
        row_deploy_guard.pack(fill="x", padx=pad, pady=2)
        self.dry_run_var = tk.BooleanVar(value=self.config.get("dry_run_enabled", True))
        ttk.Checkbutton(row_deploy_guard, text="Dry-run 只記錄不寫 Command.txt",
                        variable=self.dry_run_var).pack(side="left")
        self.pv_surplus_charge_only_var = tk.BooleanVar(
            value=self.config.get("pv_surplus_charge_only", True)
        )
        ttk.Checkbutton(row_deploy_guard, text="PV surplus 才允許充電",
                        variable=self.pv_surplus_charge_only_var).pack(side="left", padx=(12, 0))

        row_cutoff = ttk.Frame(self.frm_ai)
        row_cutoff.pack(fill="x", padx=pad, pady=2)
        self.cutoff_soc_fallback_var = tk.BooleanVar(
            value=self.config.get("cutoff_soc_fallback_enabled", True)
        )
        ttk.Checkbutton(row_cutoff, text="電壓 cutoff 後 fallback SoC",
                        variable=self.cutoff_soc_fallback_var).pack(side="left")
        self.cutoff_soc_fallback_percent_var = tk.StringVar(
            value=str(self.config.get("cutoff_soc_fallback_percent", 20.0))
        )
        ttk.Entry(row_cutoff, textvariable=self.cutoff_soc_fallback_percent_var,
                  width=8).pack(side="left", padx=4)
        ttk.Label(row_cutoff, text="% (建議 20；0 只在想標記完全耗盡時使用)").pack(side="left")

        # SoH-aware passive monitoring / optional health lock
        row_soh = ttk.Frame(self.frm_ai)
        row_soh.pack(fill="x", padx=pad, pady=2)
        self.soh_health_protection_var = tk.BooleanVar(
            value=self.config.get("soh_health_protection_enabled", False)
        )
        ttk.Checkbutton(row_soh, text="SoH health lock (選用保護)",
                        variable=self.soh_health_protection_var).pack(side="left")
        ttk.Label(row_soh, text="低壓").pack(side="left", padx=(8, 2))
        self.soh_low_voltage_v_var = tk.StringVar(
            value=str(self.config.get("soh_low_voltage_v", 4.2))
        )
        ttk.Entry(row_soh, textvariable=self.soh_low_voltage_v_var,
                  width=5).pack(side="left")
        ttk.Label(row_soh, text="V ×").pack(side="left", padx=(2, 2))
        self.soh_low_voltage_samples_var = tk.StringVar(
            value=str(self.config.get("soh_low_voltage_samples", 3))
        )
        ttk.Entry(row_soh, textvariable=self.soh_low_voltage_samples_var,
                  width=4).pack(side="left")
        ttk.Label(row_soh, text="筆；恢復").pack(side="left", padx=(8, 2))
        self.soh_recover_v_var = tk.StringVar(
            value=str(self.config.get("soh_recover_v", 5.0))
        )
        ttk.Entry(row_soh, textvariable=self.soh_recover_v_var,
                  width=5).pack(side="left")
        ttk.Label(row_soh, text="V ×").pack(side="left", padx=(2, 2))
        self.soh_recovery_samples_var = tk.StringVar(
            value=str(self.config.get("soh_recovery_samples", 12))
        )
        ttk.Entry(row_soh, textvariable=self.soh_recovery_samples_var,
                  width=4).pack(side="left")
        ttk.Label(row_soh, text="筆後解鎖").pack(side="left", padx=(2, 0))

        row_soh_status = ttk.Frame(self.frm_ai)
        row_soh_status.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_soh_status, text="前次 SoH 紀錄:").pack(side="left")
        ttk.Label(
            row_soh_status,
            textvariable=self.soh_status_var,
            relief="sunken",
            anchor="w",
            padding=(4, 2),
        ).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Label(row_soh_status, text="(唯讀)").pack(side="left")

        row_soh_model = ttk.Frame(self.frm_ai)
        row_soh_model.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_soh_model, text="SoH prediction model:").pack(side="left")
        self.soh_model_path_var = tk.StringVar(
            value=self.config.get("soh_model_path", _default_soh_model_path())
        )
        ttk.Entry(row_soh_model, textvariable=self.soh_model_path_var,
                  width=28).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_soh_model, text="瀏覽", command=self._browse_soh_model).pack(side="left")

        row_soh_pred = ttk.Frame(self.frm_ai)
        row_soh_pred.pack(fill="x", padx=pad, pady=2)
        self.soh_prediction_var = tk.BooleanVar(
            value=self.config.get("soh_prediction_enabled", False)
        )
        ttk.Checkbutton(row_soh_pred, text="啟用 SoH prediction",
                        variable=self.soh_prediction_var).pack(side="left")
        self.soh_use_for_capacity_var = tk.BooleanVar(
            value=self.config.get("soh_use_for_capacity", False)
        )
        ttk.Checkbutton(row_soh_pred, text="用 SoH 修正 SoC 容量(實驗)",
                        variable=self.soh_use_for_capacity_var).pack(side="left", padx=(12, 0))

        frm_common = ttk.LabelFrame(scrollable, text="通用設定")
        frm_common.pack(fill="x", padx=pad, pady=2)

        row_load = ttk.Frame(frm_common)
        row_load.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_load, text="負載組數 (0-4):").pack(side="left")
        self.load_count_var = tk.StringVar(value=str(self.config.get("load_count", 4)))
        ttk.Spinbox(row_load, from_=0, to=4, textvariable=self.load_count_var,
                     width=5).pack(side="left", padx=4)
        ttk.Label(row_load, text="(顯示暫沿用每組約 0.1W；0.58W/燈條對齊待後續處理；AI 模式依 load_pattern.txt 排程)").pack(side="left")

        row_manual_scenario = ttk.Frame(frm_common)
        row_manual_scenario.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_manual_scenario, text="非 AI Scenario:").pack(side="left")
        self.manual_scenario_var = tk.StringVar(value=str(self.config.get("manual_scenario", 3)))
        ttk.Combobox(row_manual_scenario, textvariable=self.manual_scenario_var,
                     values=["4", "3"], width=5, state="readonly").pack(side="left", padx=4)
        ttk.Label(
            row_manual_scenario,
            text="solar/standby 用；預設 3=電池 0 power：平時 0% flow、量測前可 50% pre-flow；4 僅明確關馬達/電池",
        ).pack(side="left")

        row_pp = ttk.Frame(frm_common)
        row_pp.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_pp, text="電池 PP ID:").pack(side="left")
        self.battery_pp_var = tk.StringVar(value="01")
        ttk.Entry(row_pp, textvariable=self.battery_pp_var, width=5).pack(side="left", padx=4)
        ttk.Label(row_pp, text="(01-10，預設 01)").pack(side="left")

        row_poll = ttk.Frame(frm_common)
        row_poll.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_poll, text="Data.txt 輪詢 (秒):").pack(side="left")
        self.poll_sec_var = tk.StringVar(value=str(self.config.get("poll_sec", 10.0)))
        ttk.Entry(row_poll, textvariable=self.poll_sec_var, width=8).pack(side="left", padx=4)
        ttk.Label(row_poll, text="(所有模式共用，預設 10 秒)").pack(side="left")

        row_log = ttk.Frame(frm_common)
        row_log.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_log, text="日誌目錄:").pack(side="left")
        self.log_dir_var = tk.StringVar(value=self.config.get("log_dir", ""))
        ttk.Entry(row_log, textvariable=self.log_dir_var, width=25).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_log, text="瀏覽", command=self._browse_log_dir).pack(side="left")

        # ── Watchdog ──
        frm_wd = ttk.LabelFrame(scrollable, text="Watchdog (自動重啟)")
        frm_wd.pack(fill="x", padx=pad, pady=2)

        self.use_watchdog_var = tk.BooleanVar(value=self.config.get("use_watchdog", True))
        ttk.Checkbutton(frm_wd, text="啟用 Watchdog (進程異常停止時自動重啟)",
                        variable=self.use_watchdog_var).pack(anchor="w", padx=pad, pady=2)

        row_wd = ttk.Frame(frm_wd)
        row_wd.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_wd, text="檢查間隔 (秒):").pack(side="left")
        self.wd_interval_var = tk.StringVar(value=str(self.config.get("watchdog_interval_sec", 60)))
        ttk.Entry(row_wd, textvariable=self.wd_interval_var, width=8).pack(side="left", padx=4)

        frm_info = ttk.LabelFrame(scrollable, text="SLFB 電池規格 (參考)")
        frm_info.pack(fill="x", padx=pad, pady=2)

        info_text = (
            "  SLFB V16: 系統電流 1A，2Ah / 11.2Wh\n"
            "  充電: 8.5W (8.5V x 1A)  放電: 5.6W (5.6V x 1A)\n"
            "  負載顯示暫沿用 4 組 × 0.1W = 0.4W；0.58W/燈條對齊待後續處理\n"
            "  截止: V < 4.2V  效率: 95% RTE\n"
            "  放電自動(依負載決定)，充電可控\n"
            "  Command.txt: 功率(mW) + 流速(0-100%) + Scenario(1-4)\n"
            "  Scenario: 1=放電全包 2=放電+市電(目前不用) 3=市電/充電/rest/pre-measure 4=明確停機"
        )
        ttk.Label(frm_info, text=info_text, font=("Consolas", 8),
                  justify="left").pack(anchor="w", padx=pad, pady=4)

        frm_ctrl = ttk.Frame(scrollable)
        frm_ctrl.pack(fill="x", padx=pad, pady=pad)

        ttk.Label(frm_ctrl, text="狀態:").pack(side="left")
        ttk.Label(frm_ctrl, textvariable=self.scenario_status_var, width=25).pack(side="left", padx=4)

        btn_frame = ttk.Frame(scrollable)
        btn_frame.pack(fill="x", padx=pad, pady=2)

        self.btn_start = ttk.Button(btn_frame, text="▶ 啟動", command=self._start_scenario,
                                    style="Start.TButton")
        self.btn_start.pack(side="left", padx=4)
        ttk.Button(btn_frame, text="■ 停止", command=self._stop_scenario).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="儲存設定", command=self._save_config).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="清除日誌", command=self._clear_log).pack(side="left", padx=4)

        self._on_mode_change()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _on_mode_change(self):
        """根據選擇的模式顯示/隱藏 AI 參數"""
        mode = self.mode_var.get()
        if mode == "ai":
            self.frm_ai.pack(fill="x", padx=6, pady=2, after=self.frm_ai.master.winfo_children()[0])
            for w in self.frm_ai.master.winfo_children():
                if isinstance(w, ttk.LabelFrame) and w.cget("text") == "運行模式":
                    self.frm_ai.pack(fill="x", padx=6, pady=2, after=w)
                    break
        else:
            pass

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg)
        if float(self.log_text.index('end-1c').split('.')[0]) > 5000:
            self.log_text.delete('1.0', '1000.0')
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        for widget in [self.log_text, self.data_text, self.cmd_text]:
            widget.config(state=tk.NORMAL)
            widget.delete(1.0, tk.END)
            widget.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _start_scenario(self):
        if self.current_scenario and self.current_scenario.is_running():
            messagebox.showwarning("Warning", "A scenario is already running. Please stop it first.")
            return

        mode = self.mode_var.get()
        vendor_dir = self.vendor_dir_var.get()

        if not vendor_dir or not os.path.isdir(vendor_dir):
            messagebox.showerror("Error", "Please set a valid vendor software folder first.")
            return

        data_file = os.path.join(vendor_dir, "Data.txt")
        command_file = os.path.join(vendor_dir, "Command.txt")
        pp = self.battery_pp_var.get().strip() or "01"
        load_count = int(self.load_count_var.get() or 4)
        manual_scenario = int(self.manual_scenario_var.get() or 3)

        if mode == "ai":
            cmd, scenario_name = self._build_ai_cmd(data_file, command_file, pp)
        elif mode == "solar":
            cmd, scenario_name = self._build_solar_cmd(data_file, command_file, pp, load_count, manual_scenario)
        else:  # standby
            cmd, scenario_name = self._build_standby_cmd(data_file, command_file, pp, load_count, manual_scenario)

        if cmd is None:
            return

        self._log(f"\n{'='*60}\n")
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Start: {scenario_name}\n")
        self._log(f"  Command: {' '.join(cmd)}\n")
        self._log(f"  工作目錄: {PROJECT_ROOT}\n")
        self._log(f"{'='*60}\n\n")

        self.current_scenario = ScriptProcess(scenario_name, cmd, log_callback=self._log)
        if self.current_scenario.start(cwd=PROJECT_ROOT):
            self.scenario_status_var.set(f"Running: {scenario_name}")
            self._log(f"[OK] Process started (PID: {self.current_scenario.proc.pid})\n")

            # Watchdog
            if self.use_watchdog_var.get():
                self._start_watchdog(cmd, scenario_name)
        else:
            self.scenario_status_var.set("Start failed")

    def _build_ai_cmd(self, data_file, command_file, pp):
        """建構 AI 控制命令"""
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file does not exist: {model_path}")
            return None, None
        soh_model_path = self.soh_model_path_var.get()
        if self.soh_prediction_var.get() and (not soh_model_path or not os.path.exists(soh_model_path)):
            messagebox.showerror("Error", f"SoH model path does not exist: {soh_model_path}")
            return None, None

        try:
            initial_soc_pct = float(self.initial_soc_var.get())
            initial_soc = initial_soc_pct / 100.0  # % → 0~1
        except ValueError:
            messagebox.showerror("Error", "Invalid initial SoC format")
            return None, None

        try:
            cutoff_soc_fallback_pct = float(self.cutoff_soc_fallback_percent_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid cutoff fallback SoC format")
            return None, None

        try:
            soh_low_voltage_v = float(self.soh_low_voltage_v_var.get())
            soh_low_voltage_samples = int(self.soh_low_voltage_samples_var.get())
            soh_recover_v = float(self.soh_recover_v_var.get())
            soh_recovery_samples = int(self.soh_recovery_samples_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid SoH health protection parameter format")
            return None, None

        device = self.device_var.get()
        window_min = self.window_min_var.get() or "15"
        poll_sec = self.poll_sec_var.get() or "10"
        initial_action = self.initial_action_var.get()
        log_dir = self.log_dir_var.get() or os.path.join(PROJECT_ROOT, "results", "deployment")

        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--mode", "deployment"]
        else:
            cmd = [sys.executable, DEPLOYMENT_SCRIPT]

        cmd += [
            "--data-file", os.path.normpath(data_file),
            "--command-file", os.path.normpath(command_file),
            "--model-path", os.path.normpath(model_path),
            "--battery-pp", pp,
            "--initial-soc", str(initial_soc),
            "--poll-sec", str(poll_sec),
            "--window-min", str(window_min),
            "--device", device,
            "--log-dir", os.path.normpath(log_dir),
            "--initial-action", initial_action,
            "--current-mode", self.current_mode_var.get(),
            "--cutoff-soc-fallback-percent", str(cutoff_soc_fallback_pct),
            "--soh-model-path", os.path.normpath(soh_model_path),
            "--soh-low-voltage-v", str(soh_low_voltage_v),
            "--soh-low-voltage-samples", str(soh_low_voltage_samples),
            "--soh-recover-v", str(soh_recover_v),
            "--soh-recovery-samples", str(soh_recovery_samples),
            "--coral",
        ]
        if self.cutoff_soc_fallback_var.get():
            cmd.append("--cutoff-soc-fallback")
        else:
            cmd.append("--no-cutoff-soc-fallback")
        if self.dry_run_var.get():
            cmd.append("--dry-run")
        if self.pv_surplus_charge_only_var.get():
            cmd.append("--pv-surplus-charge-only")
        else:
            cmd.append("--no-pv-surplus-charge-only")
        if self.soh_health_protection_var.get():
            cmd.append("--soh-health-protection")
        else:
            cmd.append("--no-soh-health-protection")
        if self.soh_prediction_var.get():
            cmd.append("--soh-prediction")
        else:
            cmd.append("--no-soh-prediction")
        if self.soh_use_for_capacity_var.get():
            cmd.append("--soh-use-for-capacity")
        else:
            cmd.append("--no-soh-use-for-capacity")

        return cmd, "AI 控制 (SAC + CORAL)"

    def _build_solar_cmd(self, data_file, command_file, pp, load_count, scenario):
        """建構太陽能測試命令"""
        log_dir = self.log_dir_var.get() or os.path.join(PROJECT_ROOT, "results", "solar_test")
        poll_sec = self.poll_sec_var.get() or "10"

        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--mode", "solar_test"]
        else:
            cmd = [sys.executable, SOLAR_TEST_SCRIPT]

        cmd += [
            "--data-file", os.path.normpath(data_file),
            "--command-file", os.path.normpath(command_file),
            "--battery-pp", pp,
            "--load-count", str(load_count),
            "--scenario", str(scenario),
            "--poll-sec", str(poll_sec),
            "--log-dir", os.path.normpath(log_dir),
        ]

        return cmd, f"太陽能測試 (Scenario {scenario})"

    def _build_standby_cmd(self, data_file, command_file, pp, load_count, scenario):
        """建構待機命令（重用 solar_test_collect.py）"""
        log_dir = self.log_dir_var.get() or os.path.join(PROJECT_ROOT, "results", "standby")
        poll_sec = self.poll_sec_var.get() or "10"

        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--mode", "solar_test"]
        else:
            cmd = [sys.executable, SOLAR_TEST_SCRIPT]

        cmd += [
            "--data-file", os.path.normpath(data_file),
            "--command-file", os.path.normpath(command_file),
            "--battery-pp", pp,
            "--load-count", str(load_count),
            "--scenario", str(scenario),
            "--poll-sec", str(poll_sec),
            "--log-dir", os.path.normpath(log_dir),
        ]

        return cmd, f"待機 (Scenario {scenario})"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _stop_scenario(self):
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_stop_flag.set()
            self.watchdog_thread.join(timeout=2)
            self.watchdog_thread = None
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [Watchdog] Stopped\n")

        if self.current_scenario:
            self.current_scenario.stop()
            self.current_scenario = None
            self.scenario_status_var.set("已停止")

        self.watchdog_cmd_backup = []
        self.watchdog_restart_count = 0

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------
    def _start_watchdog(self, cmd, scenario_name):
        self.watchdog_cmd_backup = cmd
        self.watchdog_project_root = PROJECT_ROOT
        self.watchdog_restart_count = 0
        self.watchdog_stop_flag.clear()

        interval = int(self.wd_interval_var.get() or 60)
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop, args=(interval, scenario_name), daemon=True)
        self.watchdog_thread.start()
        self._log(f"[Watchdog] Started (interval {interval}s)\n")

    def _watchdog_loop(self, interval, scenario_name):
        max_restarts = 10
        restart_times = []

        try:
            while not self.watchdog_stop_flag.is_set():
                if self.watchdog_stop_flag.wait(timeout=interval):
                    break

                if self.current_scenario is None or not self.current_scenario.is_running():
                    now = time.time()
                    restart_times = [t for t in restart_times if t > now - 3600]
                    if len(restart_times) >= max_restarts:
                        self._log(f"[Watchdog] Too many restarts; auto-restart stopped\n")
                        break

                    self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [Watchdog] Process stopped; restarting...\n")
                    self.current_scenario = ScriptProcess(scenario_name, self.watchdog_cmd_backup,
                                                          log_callback=self._log)
                    if self.current_scenario.start(cwd=self.watchdog_project_root):
                        self.watchdog_restart_count += 1
                        restart_times.append(now)
                        self.scenario_status_var.set(f"Running: {scenario_name} (restart #{self.watchdog_restart_count})")
                        self._log(f"[Watchdog] Restart successful (#{self.watchdog_restart_count})\n")
                    else:
                        self._log(f"[Watchdog] Restart failed\n")
        except Exception as e:
            self._log(f"[Watchdog] Error: {e}\n")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _poll_status(self):
        if self.current_scenario:
            if not self.current_scenario.is_running():
                self.scenario_status_var.set("已停止 (進程結束)")
                self.current_scenario = None
        self.after(2000, self._poll_status)

    def _parse_soh_time_from_filename(self, filename: str):
        match = re.search(r"(\d{8})_(\d{6})", filename)
        if not match:
            return None
        try:
            return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    def _parse_datetime_safe(self, value: str):
        if not value:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(str(value).strip(), fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(str(value).strip())
        except ValueError:
            return None

    def _short_time(self, dt_obj) -> str:
        return dt_obj.strftime("%m/%d %H:%M") if dt_obj else "--"

    def _iter_soh_prediction_files(self):
        roots = []
        log_dir = self.log_dir_var.get() if hasattr(self, "log_dir_var") else ""
        for root in [log_dir, os.path.join(PROJECT_ROOT, "data"), os.path.join(APP_ROOT, "data")]:
            if root and os.path.isdir(root) and root not in roots:
                roots.append(root)
        for root in roots:
            for dirpath, _, filenames in os.walk(root):
                if "soh_predictions.csv" in filenames:
                    yield os.path.join(dirpath, "soh_predictions.csv")
                if "soh_online_predictions.csv" in filenames:
                    yield os.path.join(dirpath, "soh_online_predictions.csv")

    def _find_latest_soh_prediction(self):
        latest = None
        for path in self._iter_soh_prediction_files():
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        if str(row.get("status", "")).upper() != "OK":
                            continue
                        try:
                            soh = float(row.get("soh", ""))
                        except ValueError:
                            continue
                        file_name = row.get("file", "")
                        dt_obj = self._parse_datetime_safe(row.get("timestamp", ""))
                        if dt_obj is None:
                            dt_obj = self._parse_soh_time_from_filename(file_name)
                        if dt_obj is None:
                            dt_obj = datetime.fromtimestamp(os.path.getmtime(path))
                        item = (dt_obj, soh, file_name)
                        if latest is None or item[0] > latest[0]:
                            latest = item
            except Exception:
                continue
        return latest

    def _find_latest_soh_candidate(self):
        log_dir = self.log_dir_var.get() if hasattr(self, "log_dir_var") else ""
        if not log_dir or not os.path.isdir(log_dir):
            return None
        paths = [
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.startswith("deployment_v2_") and name.endswith(".csv")
        ]
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        latest = None
        for path in paths[:7]:
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        if str(row.get("soh_record_candidate", "")).strip() not in ("1", "True", "true"):
                            continue
                        dt_obj = self._parse_datetime_safe(
                            row.get("soh_last_record_time") or row.get("timestamp")
                        )
                        if dt_obj is None:
                            continue
                        reason = row.get("soh_record_reason", "")
                        item = (dt_obj, reason)
                        if latest is None or item[0] > latest[0]:
                            latest = item
            except Exception:
                continue
        return latest

    def _update_soh_status(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_soh_status_scan < 10.0:
            return
        self._last_soh_status_scan = now

        pred = self._find_latest_soh_prediction()
        cand = self._find_latest_soh_candidate()
        if cand and (pred is None or cand[0] > pred[0]):
            self.soh_status_var.set(f"SoH: 待離線估計  |  time: {self._short_time(cand[0])}")
            return
        if pred:
            self.soh_status_var.set(
                f"SoH: {pred[1] * 100:.1f}%  |  time: {self._short_time(pred[0])}"
            )
            return
        self.soh_status_var.set("SoH: --  |  time: --")

    def _poll_data_command(self):
        """定期讀取並更新 Data.txt / Command.txt 顯示"""
        self._update_soh_status()
        vendor_dir = self.vendor_dir_var.get()
        if not vendor_dir or not os.path.isdir(vendor_dir):
            self.after(5000, self._poll_data_command)
            return

        # Data.txt
        data_file = os.path.join(vendor_dir, "Data.txt")
        if os.path.exists(data_file):
            try:
                size = os.path.getsize(data_file)
                if size < 10240:  # < 10KB
                    with open(data_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content != self.last_data_content:
                        self.last_data_content = content
                        self._update_text_widget(self.data_text, self._format_data_display(content))
            except Exception:
                pass

        # Command.txt
        cmd_file = os.path.join(vendor_dir, "Command.txt")
        if os.path.exists(cmd_file):
            try:
                size = os.path.getsize(cmd_file)
                if size < 10240:
                    with open(cmd_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content != self.last_command_content:
                        self.last_command_content = content
                        self._update_text_widget(self.cmd_text, self._format_command_display(content))
            except Exception:
                pass

        self.after(1000, self._poll_data_command)

    def _format_data_display(self, raw: str) -> str:
        """格式化 Data.txt 顯示（支援新版 MPPT-Bus + 負載格式）"""
        if not raw:
            return "(空)"
        lines = [ln.strip() for ln in raw.strip().split('\n') if ln.strip()]
        out = []
        idx = 0
        
        if idx < len(lines):
            parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
            if len(parts[0]) >= 14 and parts[0][:14].isdigit():
                ts = parts[0][:14]
                load_info = f", 負載={parts[1]}組" if len(parts) > 1 else ""
                out.append(f"時間: {ts[:4]}/{ts[4:6]}/{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}{load_info}")
                idx += 1
        
        has_bus = False
        if idx < len(lines):
            parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
            if len(parts) >= 6:
                first_field = parts[0]
                is_battery = first_field.isdigit() and 1 <= int(first_field) <= 10
                if not is_battery:
                    try:
                        sv = float(parts[0]) / 100.0
                        si = float(parts[1])
                        sp = float(parts[2])
                        mv = float(parts[3]) / 100.0
                        mi = float(parts[4])
                        mp = float(parts[5])
                        out.append(f"Solar: {sv:.2f}V {si:.0f}mA {sp:.0f}mW ({sp/1000:.3f}W)")
                        out.append(f"MPPT : {mv:.2f}V {mi:.0f}mA {mp:.0f}mW ({mp/1000:.3f}W)")
                        if len(parts) >= 9:
                            bv2 = float(parts[6]) / 100.0
                            bi2 = float(parts[7])
                            bp2 = float(parts[8])
                            out.append(f"Bus  : {bv2:.2f}V {bi2:.0f}mA {bp2:.0f}mW ({bp2/1000:.3f}W)")
                            has_bus = True
                    except (ValueError, IndexError):
                        out.append(f"MPPT: {lines[idx]}")
                    idx += 1
        
        if idx < len(lines) and has_bus:
            parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
            if len(parts) >= 3:
                first_field = parts[0]
                is_battery = (first_field.isdigit() and 1 <= int(first_field) <= 10
                              and len(parts) >= 6 and len(parts) <= 7)
                if not is_battery:
                    try:
                        lv = float(parts[0]) / 100.0
                        li = float(parts[1])
                        lp = float(parts[2])
                        out.append(f"負載 : {lv:.2f}V {li:.0f}mA {lp:.0f}mW ({lp/1000:.3f}W)")
                        if len(parts) >= 6:
                            gv = float(parts[3]) / 100.0
                            gi = float(parts[4])
                            gp = float(parts[5])
                            out.append(f"市電 : {gv:.2f}V {gi:.0f}mA {gp:.0f}mW ({gp/1000:.3f}W)")
                    except (ValueError, IndexError):
                        out.append(f"Load: {lines[idx]}")
                    idx += 1
        
        while idx < len(lines):
            line = lines[idx]
            idx += 1
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 7:
                try:
                    pp = parts[0]
                    soc = float(parts[1]) / 10.0
                    bv = float(parts[2]) / 100.0
                    cv = float(parts[3]) / 100.0
                    bi = float(parts[4])
                    temp = float(parts[5]) / 10.0
                    speed = float(parts[6]) / 10.0
                    out.append(f"電池{pp}: SoC={soc:.1f}% V={bv:.2f}V 充電V={cv:.2f}V I={bi:.0f}mA T={temp:.1f}°C 流速={speed:.0f}%")
                except (ValueError, IndexError):
                    out.append(line)
            elif len(parts) >= 6:
                try:
                    pp = parts[0]
                    soc = float(parts[1]) / 10.0
                    bv = float(parts[2]) / 100.0
                    bi = float(parts[3])
                    temp = float(parts[4]) / 10.0
                    speed = float(parts[5]) / 10.0
                    out.append(f"電池{pp}: SoC={soc:.1f}% V={bv:.2f}V I={bi:.0f}mA T={temp:.1f}°C 流速={speed:.0f}%")
                except (ValueError, IndexError):
                    out.append(line)
            else:
                out.append(line)
        return '\n'.join(out) if out else raw

    def _format_command_display(self, raw: str) -> str:
        """格式化 Command.txt 顯示"""
        if not raw:
            return "(空)"
        lines = raw.strip().split('\n')
        out = []
        scenario_names = {1: "放電全包", 2: "放電+市電", 3: "市電充電/rest/pre-measure", 4: "明確停機"}

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) == 1 and line in "1234":
                code = int(line)
                out.append(f"情況碼: {code} ({scenario_names.get(code, '?')})")
            elif len(line) >= 14 and line[:14].isdigit():
                ts = line[:14]
                rest = line[14:]
                load_info = ""
                if rest.startswith(','):
                    parts = rest.split(',')
                    if len(parts) >= 2 and parts[1].strip().isdigit():
                        load_info = f", 負載={parts[1].strip()}組"
                out.append(f"時間: {ts[:4]}/{ts[4:6]}/{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}{load_info}")
            else:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if len(parts) >= 3:
                    try:
                        pp = parts[0]
                        power_mw = int(parts[1])
                        flow = int(parts[2])
                        power_w = power_mw / 1000.0
                        out.append(f"  電池{pp}: {power_mw}mW ({power_w:.3f}W) 流速={flow}%")
                    except (ValueError, IndexError):
                        out.append(f"  {line}")
                else:
                    out.append(f"  {line}")
        return '\n'.join(out) if out else raw

    def _update_text_widget(self, widget, text):
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _browse_vendor_dir(self):
        d = filedialog.askdirectory(title="選擇廠商程式資料夾",
                                     initialdir=self.vendor_dir_var.get() or "")
        if d:
            self.vendor_dir_var.set(d)
            exe = os.path.join(d, "P302.exe")
            if os.path.exists(exe):
                self.vendor_exe_var.set(exe)

    def _browse_vendor_exe(self):
        f = filedialog.askopenfilename(
            title="選擇廠商程式 EXE",
            filetypes=[("執行檔", "*.exe"), ("所有", "*.*")])
        if f:
            self.vendor_exe_var.set(f)
            self.vendor_dir_var.set(os.path.dirname(f))

    def _browse_model(self):
        f = filedialog.askopenfilename(
            title="選擇 SAC 模型",
            initialdir=os.path.dirname(self.model_path_var.get()) if self.model_path_var.get() else "",
            filetypes=[("模型", "*.pth"), ("所有", "*.*")])
        if f:
            self.model_path_var.set(f)

    def _browse_soh_model(self):
        current = self.soh_model_path_var.get()
        initialdir = current if os.path.isdir(current) else os.path.dirname(current)
        d = filedialog.askdirectory(title="選擇 SoH 模型資料夾", initialdir=initialdir or "")
        if d:
            self.soh_model_path_var.set(d)
            return
        f = filedialog.askopenfilename(
            title="選擇 SoH 模型檔案",
            initialdir=initialdir or "",
            filetypes=[("SoH 模型/Scaler", "*.pth *.pkl *.npz *.npy"), ("所有", "*.*")])
        if f:
            self.soh_model_path_var.set(f)

    def _browse_log_dir(self):
        d = filedialog.askdirectory(title="選擇日誌輸出目錄",
                                     initialdir=self.log_dir_var.get() or "")
        if d:
            self.log_dir_var.set(d)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _update_vendor_status(self):
        exe = self.vendor_exe_var.get()
        if not exe or not os.path.exists(exe):
            self.vendor_status_var.set("路徑無效")
            return
        running = check_vendor_running(exe)
        self.vendor_status_var.set("執行中 ✓" if running else "未執行")

    def _launch_vendor(self):
        exe = self.vendor_exe_var.get()
        if not os.path.exists(exe):
            messagebox.showerror("Error", f"Not found: {exe}")
            return
        try:
            self.vendor_proc = subprocess.Popen([exe], cwd=os.path.dirname(exe))
            self._update_vendor_status()
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Vendor software started\n")
        except Exception as e:
            self._log(f"[ERROR] Failed to start vendor software: {e}\n")

    def _open_vendor_folder(self):
        d = self.vendor_dir_var.get()
        if os.path.isdir(d):
            os.startfile(d)
        else:
            messagebox.showerror("Error", f"Folder does not exist: {d}")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _save_config(self):
        self.config.update({
            "vendor_dir": self.vendor_dir_var.get(),
            "vendor_exe": self.vendor_exe_var.get(),
            "model_path": self.model_path_var.get(),
            "soh_model_path": self.soh_model_path_var.get(),
            "soh_prediction_enabled": self.soh_prediction_var.get(),
            "soh_use_for_capacity": self.soh_use_for_capacity_var.get(),
            "initial_soc": float(self.initial_soc_var.get() or 50),
            "load_count": int(self.load_count_var.get() or 4),
            "log_dir": self.log_dir_var.get(),
            "device": self.device_var.get(),
            "poll_sec": float(self.poll_sec_var.get() or 10),
            "window_min": int(self.window_min_var.get() or 15),
            "use_watchdog": self.use_watchdog_var.get(),
            "watchdog_interval_sec": int(self.wd_interval_var.get() or 60),
            "current_mode": self.current_mode_var.get(),
            "manual_scenario": int(self.manual_scenario_var.get() or 3),
            "dry_run_enabled": self.dry_run_var.get(),
            "pv_surplus_charge_only": self.pv_surplus_charge_only_var.get(),
            "cutoff_soc_fallback_enabled": self.cutoff_soc_fallback_var.get(),
            "cutoff_soc_fallback_percent": float(self.cutoff_soc_fallback_percent_var.get() or 20),
            "soh_health_protection_enabled": self.soh_health_protection_var.get(),
            "soh_low_voltage_v": float(self.soh_low_voltage_v_var.get() or 4.2),
            "soh_low_voltage_samples": int(self.soh_low_voltage_samples_var.get() or 3),
            "soh_recover_v": float(self.soh_recover_v_var.get() or 5.0),
            "soh_recovery_samples": int(self.soh_recovery_samples_var.get() or 12),
        })
        if ConfigManager.save(self.config):
            messagebox.showinfo("OK", "Settings saved")
        else:
            messagebox.showerror("Error", "Failed to save settings")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _on_closing(self):
        self._stop_scenario()
        if self.vendor_proc:
            try:
                if self.vendor_proc.poll() is None:
                    self.vendor_proc.terminate()
            except Exception:
                pass
        self.destroy()


# ======================================================================
# Entry point
# ======================================================================
def _resolve_control_path():
    """找到 control/ 目錄並加入 sys.path，確保子腳本可匯入"""
    candidates = [
        CONTROL_DIR,
        os.path.join(os.path.dirname(sys.executable), "_internal", "control"),  # PyInstaller one-dir
    ]
    if hasattr(sys, '_MEIPASS'):
        candidates.insert(0, os.path.join(sys._MEIPASS, "control"))

    for d in candidates:
        if os.path.isdir(d):
            if d not in sys.path:
                sys.path.insert(0, d)
            parent = os.path.dirname(d)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return d
    return None


def main():
    """
    主入口。支援 --mode 參數以在 PyInstaller EXE 中切換模式：
      P302_AI_GUI.exe                          → GUI
      P302_AI_GUI.exe --mode solar_test [...]  → solar_test_collect.main()
      P302_AI_GUI.exe --mode deployment [...]  → run_deployment.main()
    """
    if len(sys.argv) >= 3 and sys.argv[1] == "--mode":
        mode = sys.argv[2]
        sys.argv = [sys.argv[0]] + sys.argv[3:]

        _resolve_control_path()

        if mode == "solar_test":
            from solar_test_collect import main as solar_main
            solar_main()
            return
        elif mode == "deployment":
            from run_deployment import main as deploy_main
            deploy_main()
            return
        else:
            print(f"Unknown mode: {mode}", file=sys.stderr)
            sys.exit(1)

    app = AIControlGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

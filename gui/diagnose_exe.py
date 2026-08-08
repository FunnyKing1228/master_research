#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXE 診斷工具：檢查跨裝置可能出現的問題
"""

import os
import sys
import platform

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def diagnose():
    """診斷 EXE 環境"""
    print("=" * 70)
    print("EXE 環境診斷")
    print("=" * 70)
    print()
    
    is_frozen = getattr(sys, 'frozen', False)
    print(f"[1] 執行模式: {'EXE (PyInstaller)' if is_frozen else 'Python 腳本'}")
    
    print(f"[2] sys.executable: {sys.executable}")
    if is_frozen:
        exe_dir = os.path.dirname(sys.executable)
        print(f"    EXE 目錄: {exe_dir}")
        print(f"    EXE 目錄是否存在: {os.path.exists(exe_dir)}")
        
        control_exe = os.path.join(exe_dir, "run_online_control.exe")
        print(f"    run_online_control.exe 路徑: {control_exe}")
        print(f"    run_online_control.exe 是否存在: {os.path.exists(control_exe)}")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"    腳本目錄: {script_dir}")
    
    cwd = os.getcwd()
    print(f"[3] 當前工作目錄: {cwd}")
    print(f"    工作目錄是否存在: {os.path.exists(cwd)}")
    print(f"    工作目錄可寫入: {os.access(cwd, os.W_OK)}")
    
    try:
        file_path = __file__
        print(f"[4] __file__: {file_path}")
        print(f"    __file__ 是否存在: {os.path.exists(file_path)}")
    except NameError:
        print(f"[4] __file__: 不存在（EXE 模式下正常）")
    
    print(f"[5] 作業系統: {platform.system()} {platform.release()}")
    print(f"    Python 版本: {sys.version}")
    print(f"    架構: {platform.machine()}")
    
    print(f"[6] 預設編碼: {sys.getdefaultencoding()}")
    print(f"    檔案系統編碼: {sys.getfilesystemencoding()}")
    
    print(f"[7] sys.path (前 5 個):")
    for i, path in enumerate(sys.path[:5]):
        print(f"    [{i}] {path}")
        print(f"        存在: {os.path.exists(path)}")
    
    print(f"[8] 關鍵模組檢查:")
    modules_to_check = [
        'io_protocol',
        'data_collector',
        'app.io_protocol',
        'app.data_collector',
    ]
    for mod_name in modules_to_check:
        try:
            __import__(mod_name)
            print(f"    [OK] {mod_name}: 可匯入")
        except ImportError as e:
            print(f"    [FAIL] {mod_name}: 無法匯入 ({e})")
    
    print(f"[9] 檔案權限測試:")
    test_file = os.path.join(cwd, "test_write_permission.tmp")
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("test")
        os.remove(test_file)
        print(f"    [OK] 當前目錄可寫入")
    except Exception as e:
        print(f"    [FAIL] 當前目錄無法寫入: {e}")
    
    print()
    print("=" * 70)
    print("診斷完成")
    print("=" * 70)

if __name__ == "__main__":
    diagnose()


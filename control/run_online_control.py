import os
import sys
import time
import math
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

core_dir = os.path.join(parent_dir, 'core')
sys.path.insert(0, core_dir)
control_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, control_dir)

from sac_agent import SACAgent
from safety_net import project as safety_project
from io_protocol import (
	read_vendor_data_file,
	write_control_file_vendor,
	format_ts,
	TZ_UTC8,
)

STANDBY_FLOW_SITUATION_CODE = 3
STANDBY_SITUATION_CODE = 3
PRE_MEASURE_SITUATION_CODE = 3
FLOW_REST_PCT = 0.0
FLOW_PRE_MEASURE_PCT = 50.0
VOLTAGE_RECOVERY_SECONDS = 25.0
FLOW_IDLE_PCT = FLOW_REST_PCT  # Backward-compatible public alias for rest flow.


def determine_situation(a_kw: float, load_kw: float, pv_kw: float) -> int:
	"""
	
	
	
	
	Args:
	
	Returns:
	"""
	net_load = max(0.0, load_kw - pv_kw)
	
	if a_kw < -0.001:
		discharge_kw = abs(a_kw)
		if discharge_kw >= net_load - 0.001:
			return 1
		else:
			return 2
	elif a_kw > 0.001:
		return 3
	else:
		return STANDBY_FLOW_SITUATION_CODE


def pick_latest(ids: Dict[str, Tuple[datetime, float, float, float, float]], key: str) -> Optional[Tuple[datetime, float, float, float, float]]:
	return ids.get(key, None)


def pick_first_battery(ids: Dict[str, Tuple[datetime, float, float, float, float]]) -> Optional[Tuple[str, Tuple[datetime, float, float, float, float]]]:
	candidates = [(k, v) for k, v in ids.items() if k and k.startswith("B") and len(k) == 3 and k[1:].isdigit()]
	if not candidates:
		return None
	candidates.sort(key=lambda kv: int(kv[0][1:]))
	return candidates[0]


def _get_tou_price(hour: int, day_of_week: int) -> float:
	"""Documentation for this public API is provided in English."""
	if day_of_week >= 5:
		return 2.06
	if hour < 9:
		return 2.06
	elif hour < 16:
		return 4.69
	elif hour < 22:
		return 7.13
	else:
		return 4.69


def build_agent_state(status: Dict[str, Tuple[datetime, float, float, float, float]], price: float, fallback_batt_id: str = "B01") -> Optional[Tuple[np.ndarray, datetime, str]]:
	"""
	state = [soc(0..1), load_kw, pv_kw, price_norm, hour, day_of_week]
	
	"""
	bpick = status.get(f"B{int(fallback_batt_id[-2:]):02d}", None)
	if bpick is None:
		pair = pick_first_battery(status)
		if pair is None:
			return None
		bid, bval = pair
	else:
		bid, bval = (f"B{int(fallback_batt_id[-2:]):02d}", bpick)

	b_ts, b_v, b_i, b_flow, b_soc = bval
	soc_frac = float(b_soc) / 100.0  # 0..1

	pv = pick_latest(status, "PV")
	ld = pick_latest(status, "LD")
	pv_kw = 0.0
	load_kw = 0.0
	ts_ref = b_ts
	if pv is not None:
		ts_ref = max(ts_ref, pv[0])
		pv_kw = max(0.0, float(pv[1]) * float(pv[2]) / 1000.0)
	if ld is not None:
		ts_ref = max(ts_ref, ld[0])
		load_kw = max(0.0, float(ld[1]) * float(ld[2]) / 1000.0)

	dt = ts_ref
	hour = float(dt.hour)
	day_of_week = float(dt.weekday())  # 0..6

	tou_price = _get_tou_price(int(dt.hour), int(dt.weekday()))
	price_norm = float(np.clip(tou_price / 10.0, 0.0, 1.0))

	state = np.array([soc_frac, load_kw, pv_kw, price_norm, hour, day_of_week], dtype=np.float32)
	return state, ts_ref, bid


def load_agent(model_path: str, state_dim: int = 6, action_dim: int = 1, device: str = "cpu") -> SACAgent:
	"""
	"""
	checkpoint = torch.load(model_path, map_location=device)
	
	if 'actor' in checkpoint:
		actor_state = checkpoint['actor']
		if 'fc1.weight' in actor_state:
			hidden_dim = actor_state['fc1.weight'].shape[0]
		else:
			hidden_dim = 256
	else:
		hidden_dim = 256
	
	evidential_enabled = checkpoint.get('evidential_enabled', False)
	lambda_evi = checkpoint.get('lambda_evi', 1e-3)
	beta_risk = checkpoint.get('beta_risk', 0.5)
	beta_occ = checkpoint.get('beta_occ', 0.3)
	
	agent = SACAgent(
		state_dim=state_dim,
		action_dim=action_dim,
		device=device,
		hidden_dim=hidden_dim,
		evidential_enabled=evidential_enabled,
		lambda_evi=lambda_evi,
		beta_risk=beta_risk,
		beta_occ=beta_occ,
	)
	
	agent.load(model_path)
	agent.device = device
	return agent


def main():
	parser = argparse.ArgumentParser(description="Online control loop (15-min) using text-file protocol")
	parser.add_argument("--status-file", type=str, required=True, help="元件→模型 狀態輸入檔路徑")
	parser.add_argument("--command-file", type=str, required=True, help="模型→元件 命令輸出檔路徑")
	parser.add_argument("--model-path", type=str, required=True, help="SAC 模型 .pth 檔案")
	parser.add_argument("--battery-id", type=str, default="01", help="目標電池 ID（兩位數字，如 01）")
	parser.add_argument("--pmax-kw", type=float, required=True, help="電池額定功率（kW）")
	parser.add_argument("--soc-min", type=float, default=0.10, help="SoC 下界（0..1）")
	parser.add_argument("--soc-max", type=float, default=0.90, help="SoC 上界（0..1）")
	parser.add_argument("--ramp-kw", type=float, default=0.0, help="每步最大功率變化（kW）；0 表示不限")
	parser.add_argument("--price", type=float, default=1.0, help="缺省價格")
	parser.add_argument("--flow-per-kw", type=float, default=0.0, help="舊參數（已廢棄）：使用 --use-power-to-flow 代替")
	parser.add_argument("--use-power-to-flow", action="store_true", help="使用一維查表法：流速百分比 = (功率/最大功率) * 100")
	parser.add_argument("--interval-sec", type=int, default=900, help="輪詢週期（秒），預設 15 分鐘")
	parser.add_argument("--max-age-sec", type=int, default=180, help="狀態/命令過期秒數（丟棄）")
	parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="推論裝置")
	parser.add_argument("--use-safetynet", action="store_true", help="對動作套用 SafetyNet 投影")
	args = parser.parse_args()

	device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
	agent = load_agent(args.model_path, state_dim=6, action_dim=1, device=device)

	prev_action_kw: float = 0.0
	pp = f"{int(args.battery_id):02d}"
	previous_command: Optional[Tuple[float, float]] = None
	last_write_time: Optional[datetime] = None
	last_sit_code: int = STANDBY_SITUATION_CODE
	
	print("=" * 70)
	print("AI 控制程式已啟動")
	print("=" * 70)
	print(f"狀態檔案: {args.status_file}")
	print(f"命令檔案: {args.command_file}")
	print(f"電池 ID: {pp}")
	print(f"最大功率: {args.pmax_kw} kW")
	print(f"讀取間隔: {args.interval_sec} 秒")
	print("=" * 70)
	print()
	print("程式會持續運行，每秒更新一次 Command.txt")
	print("請在 GUI 中點擊「AI控制」按鈕啟用 AI 模式")
	print()
	print("-" * 70)

	while True:
		try:
			mppt_data, status_map_raw = read_vendor_data_file(args.status_file, max_age_sec=args.max_age_sec, clear_after_read=False)
			status_map: Dict[str, Tuple[datetime, float, float, float, float]] = {}
			for pp_key in sorted(status_map_raw.keys(), key=lambda x: int(x) if x.isdigit() else 999):
				ts, soc_pct, volt_v, curr_a, temp_c, speed = status_map_raw[pp_key]
				status_map[f"B{pp_key.zfill(2)}"] = (ts, volt_v, curr_a, speed, soc_pct)

			state_pack = build_agent_state(status_map, price=args.price, fallback_batt_id=f"B{pp}")

			if state_pack is None:
				now = datetime.now(TZ_UTC8)
				success = write_control_file_vendor(args.command_file, {pp: (now, 0.0, FLOW_IDLE_PCT)}, 
				                                     global_ts=now,
				                                     require_empty=True, max_wait_sec=0.2, max_retries=5,
				                                     situation_code=STANDBY_FLOW_SITUATION_CODE)
				if not success:
					success = write_control_file_vendor(args.command_file, {pp: (now, 0.0, FLOW_IDLE_PCT)}, 
					                                     global_ts=now,
					                                     require_empty=False, max_wait_sec=0.1, max_retries=3,
					                                     situation_code=STANDBY_FLOW_SITUATION_CODE)
				if success:
					previous_command = (0.0, FLOW_IDLE_PCT)
					last_write_time = now
					current_time_str = now.strftime("%H:%M:%S")
					print(f"[{current_time_str}] [無狀態資料] 寫入 rest 指令 (mode 3, 0.0W, {FLOW_IDLE_PCT:.0f}%)")
				
				loop_start_time = time.time()
				next_state_read_time = loop_start_time + args.interval_sec
				last_power_w = 0.0
				last_flow_percent = FLOW_IDLE_PCT
				
				while time.time() < next_state_read_time:
					current_check_time = datetime.now(TZ_UTC8)
					time_since_last_write = (current_check_time - last_write_time).total_seconds() if last_write_time else float('inf')
					
					if time_since_last_write >= 1.0:
						write_ts = current_check_time
						success = write_control_file_vendor(args.command_file, {pp: (write_ts, last_power_w, last_flow_percent)}, 
						                                     global_ts=write_ts,
						                                     require_empty=True, max_wait_sec=0.1, max_retries=2,
						                                     situation_code=STANDBY_FLOW_SITUATION_CODE)
						if not success:
							success = write_control_file_vendor(args.command_file, {pp: (write_ts, last_power_w, last_flow_percent)}, 
							                                     global_ts=write_ts,
							                                     require_empty=False, max_wait_sec=0.05, max_retries=1,
							                                     situation_code=STANDBY_FLOW_SITUATION_CODE)
						
						if success:
							last_write_time = current_check_time
							current_time_str = current_check_time.strftime("%H:%M:%S")
							power_kw = last_power_w / 1000.0
							print(f"[{current_time_str}] [持續更新-無資料] 情況{STANDBY_FLOW_SITUATION_CODE}(rest) 功率={power_kw:6.3f}kW ({last_power_w:7.1f}W), 流速={last_flow_percent:5.1f}%")
					
					time.sleep(1.0)
				
				continue

			state_vec, ts_ref, bid = state_pack

			action_norm = agent.select_action(state_vec, evaluate=True)
			a_raw_kw = float(action_norm[0]) * float(args.pmax_kw)

			a_safe_kw = a_raw_kw
			if args.use_safetynet:
				a_safe_kw, _, _ = safety_project(
					state=np.array([state_vec[0]], dtype=np.float32),
					action=np.array([a_raw_kw], dtype=np.float32),
					prev_action=float(prev_action_kw),
					pmax=float(args.pmax_kw),
					ramp_kw=float(args.ramp_kw) if args.ramp_kw > 0 else None,
					soc_bounds=(float(args.soc_min), float(args.soc_max)),
					env=None,
				)
			a_safe_kw = float(a_safe_kw)
			prev_action_kw = float(a_safe_kw)

			load_kw_val = float(state_vec[1])  # state: [soc, load, pv, price, hour, dow]
			pv_kw_val = float(state_vec[2])
			sit_code = determine_situation(a_safe_kw, load_kw_val, pv_kw_val)

			power_w = float(abs(a_safe_kw)) * 1000.0  # kW → W
			if args.use_power_to_flow:
				power_percent = (abs(a_safe_kw) / float(args.pmax_kw)) * 100.0 if args.pmax_kw > 0 else 0.0
				flow_percent = max(0.0, min(100.0, power_percent))
			elif args.flow_per_kw > 0:
				flow_percent = float(abs(a_safe_kw)) * float(args.flow_per_kw)
			else:
				flow_percent = 25.0
			if abs(a_safe_kw) <= 0.001:
				flow_percent = FLOW_IDLE_PCT

			current_command = (power_w, flow_percent)
			now_ts = datetime.now(TZ_UTC8)
			
			command_changed = (previous_command is None or 
			                   abs(current_command[0] - previous_command[0]) > 0.01 or 
			                   abs(current_command[1] - previous_command[1]) > 0.01)
			
			time_since_last_write = (now_ts - last_write_time).total_seconds() if last_write_time else float('inf')
			should_write = command_changed or (time_since_last_write >= 1.0)
			
			if should_write:
				write_ts = now_ts
				
				success = write_control_file_vendor(args.command_file, {pp: (write_ts, power_w, flow_percent)}, 
				                                     global_ts=write_ts,
				                                     require_empty=True, max_wait_sec=0.2, max_retries=5,
				                                     situation_code=sit_code)
				if not success:
					success = write_control_file_vendor(args.command_file, {pp: (write_ts, power_w, flow_percent)}, 
					                                     global_ts=write_ts,
					                                     require_empty=False, max_wait_sec=0.1, max_retries=3,
					                                     situation_code=sit_code)
				
				if success:
					previous_command = current_command
					last_write_time = now_ts
					current_time_str = now_ts.strftime("%H:%M:%S")
					power_kw = power_w / 1000.0
					direction = "放電" if a_safe_kw < -0.001 else ("充電" if a_safe_kw > 0.001 else "待機")
					status = "[變化]" if command_changed else "[更新時間戳]"
					print(f"[{current_time_str}] {status} 情況{sit_code}({direction}) 功率={power_kw:6.3f}kW ({power_w:7.1f}W), 流速={flow_percent:5.1f}%")
			
			loop_start_time = time.time()
			next_state_read_time = loop_start_time + args.interval_sec
			last_power_w = power_w
			last_flow_percent = flow_percent
			last_sit_code = sit_code
			
			while time.time() < next_state_read_time:
				current_check_time = datetime.now(TZ_UTC8)
				time_since_last_write = (current_check_time - last_write_time).total_seconds() if last_write_time else float('inf')
				
				if time_since_last_write >= 1.0:
					write_ts = current_check_time
					
					success = write_control_file_vendor(args.command_file, {pp: (write_ts, last_power_w, last_flow_percent)}, 
					                                     global_ts=write_ts,
					                                     require_empty=True, max_wait_sec=0.1, max_retries=2,
					                                     situation_code=last_sit_code)
					if not success:
						success = write_control_file_vendor(args.command_file, {pp: (write_ts, last_power_w, last_flow_percent)}, 
						                                     global_ts=write_ts,
						                                     require_empty=False, max_wait_sec=0.05, max_retries=1,
						                                     situation_code=last_sit_code)
					
					if success:
						last_write_time = current_check_time
				
				time.sleep(1.0)

		except Exception as e:
			now = datetime.now(TZ_UTC8)
			try:
				success = write_control_file_vendor(args.command_file, {pp: (now, 0.0, FLOW_IDLE_PCT)}, 
				                                     global_ts=now,
				                                     require_empty=False, max_wait_sec=0.1, max_retries=2,
				                                     situation_code=STANDBY_FLOW_SITUATION_CODE)
				if success:
					previous_command = (0.0, FLOW_IDLE_PCT)
					last_write_time = now
			except Exception:
				pass
		finally:
			time.sleep(max(1, int(args.interval_sec)))


if __name__ == "__main__":
	main()



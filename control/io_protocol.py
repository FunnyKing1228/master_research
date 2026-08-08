import os
import io
import time
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional, List


TZ_UTC8 = timezone(timedelta(hours=8))


def parse_ts(ts_str: str) -> datetime:
	"""YYYYMMDDhhmmss（UTC+8）"""
	ts_str = ts_str.strip()
	dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
	return dt.replace(tzinfo=TZ_UTC8)


def format_ts(dt: Optional[datetime] = None) -> str:
	if dt is None:
		dt = datetime.now(TZ_UTC8)
	else:
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=TZ_UTC8)
		dt = dt.astimezone(TZ_UTC8)
	return dt.strftime("%Y%m%d%H%M%S")


def _to_7digits(value: float) -> str:
	iv = int(round(abs(value) * 100.0))
	return f"{iv:07d}"[-7:]


def parse_signed_field(s: str) -> float:
	"""Documentation for this public API is provided in English."""
	s = s.strip()
	if len(s) != 8 or not s[0] in ("0", "1") or not s[1:].isdigit():
		raise ValueError(f"Invalid signed field: {s!r}")
	mag = int(s[1:]) / 100.0
	return mag if s[0] == "0" else -mag


def format_signed_field(value: float) -> str:
	sign = "0" if value >= 0 else "1"
	return f"{sign}{_to_7digits(value)}"


def parse_unsigned_field(s: str) -> float:
	"""Documentation for this public API is provided in English."""
	s = s.strip()
	if len(s) != 8 or s[0] != "0" or not s[1:].isdigit():
		raise ValueError(f"Invalid unsigned field: {s!r}")
	return int(s[1:]) / 100.0


def format_unsigned_field(value: float) -> str:
	return f"0{_to_7digits(max(0.0, value))}"


# PP = "01"..."99"

def parse_control_line(line: str) -> Tuple[str, datetime, float, float]:
	parts = [p.strip() for p in line.strip().split(",")]
	if len(parts) < 4:
		raise ValueError(f"Invalid control line: {line!r}")
	pp = parts[0]
	ts = parse_ts(parts[1])
	s_power_w = parse_signed_field(parts[2])     # W
	f_flow_ccm = parse_unsigned_field(parts[3])  # cc/min
	return pp, ts, s_power_w, f_flow_ccm


def format_control_line(pp: str, ts: datetime, power_w: float, flow_ccm: float) -> str:
	pp2 = f"{int(pp):02d}" if pp.isdigit() else pp
	return f"{pp2},{format_ts(ts)},{format_signed_field(power_w)},{format_unsigned_field(flow_ccm)}, , , , , , "


def command_pp_for_power(pp: str, power_w: float) -> str:
	"""Keep the battery PP even at zero power so flow commands still target it."""
	return f"{int(pp):02d}" if str(pp).isdigit() else str(pp)


def read_control_file(path: str, max_age_sec: Optional[int] = None) -> Dict[str, Tuple[datetime, float, float]]:
	"""Documentation for this public API is provided in English."""
	results: Dict[str, Tuple[datetime, float, float]] = {}
	try:
		if not os.path.exists(path):
			return results
		with io.open(path, "r", encoding="utf-8") as f:
			lines = [ln for ln in f.readlines() if ln.strip()]
	except Exception:
		return results
	finally:
		try:
			with io.open(path, "w", encoding="utf-8") as w:
				w.write("")
		except Exception:
			pass

	now_ts = datetime.now(TZ_UTC8)
	for ln in lines:
		try:
			pp, ts, p_w, f_ccm = parse_control_line(ln)
		except Exception:
			continue
		if max_age_sec is not None and (now_ts - ts).total_seconds() > float(max_age_sec):
			continue
		if pp not in results or ts > results[pp][0]:
			results[pp] = (ts, p_w, f_ccm)
	return results


def write_control_file(path: str, commands: Dict[str, Tuple[datetime, float, float]]) -> None:
	"""Documentation for this public API is provided in English."""
	lines: List[str] = []
	for pp, (ts, power_w, flow_ccm) in commands.items():
		lines.append(format_control_line(pp, ts, power_w, flow_ccm))
	out = "\n".join(lines) + ("\n" if lines else "")
	with io.open(path, "w", encoding="utf-8") as f:
		f.write(out)


def check_file_available(path: str, max_wait_sec: float = 0.1, check_empty: bool = True) -> bool:
	"""
	
	
	Args:
	
	Returns:
	"""
	if not os.path.exists(path):
		return True
	
	if check_empty:
		try:
			if os.path.getsize(path) > 0:
				with io.open(path, "r", encoding="utf-8") as f:
					content = f.read().strip()
					if content:
						return False
		except (IOError, OSError):
			return False
	
	try:
		with io.open(path, "r+", encoding="utf-8") as f:
			pass
		return True
	except (IOError, OSError, PermissionError):
		return False


def write_control_file_vendor(path: str, commands: Dict[str, Tuple[datetime, float, float]], 
                              global_ts: Optional[datetime] = None,
                              require_empty: bool = True,
                              max_wait_sec: float = 0.1,
                              max_retries: int = 3,
                              load_count: Optional[int] = None,
                              situation_code: Optional[int] = None) -> bool:
	"""
	
	
		YYYYMMDDhhmmss
		...
	
	
	
	
	Args:
	
	Returns:
	"""
	for attempt in range(max_retries):
		if check_file_available(path, max_wait_sec=max_wait_sec, check_empty=require_empty):
			break
		if attempt < max_retries - 1:
			time.sleep(max_wait_sec)
		else:
			return False
	
	if not commands:
		ts_line = format_ts(global_ts) if global_ts else format_ts()
		if load_count is not None:
			ts_line = f"{ts_line},{load_count}"
		if situation_code is not None:
			content = f"{int(situation_code)}\n{ts_line}\n"
		else:
			content = ts_line + "\n"
	else:
		if global_ts is None:
			all_ts = [ts for ts, _, _ in commands.values()]
			if all_ts:
				global_ts = min(all_ts)
			else:
				global_ts = datetime.now(TZ_UTC8)
		
		ts_line = format_ts(global_ts)
		has_id0 = "0" in commands
		if load_count is not None:
			ts_line = f"{ts_line},{load_count}"
		elif has_id0:
			ts_line = f"{ts_line},0"
		
		lines: List[str] = []
		if situation_code is not None:
			lines.append(str(int(situation_code)))
		lines.append(ts_line)
		
		for pp in sorted(commands.keys()):
			_, power_w, flow_percent = commands[pp]
			power_mw = int(round(float(power_w) * 1000.0))
			flow_int = int(round(max(0.0, min(100.0, float(flow_percent)))))
			command_pp = command_pp_for_power(pp, power_w)
			
			power_str = f"{power_mw}"
			flow_str = f"{flow_int}"
			line = f"{command_pp},{power_str},{flow_str},"
			lines.append(line)
		
		content = "\n".join(lines) + "\n"
	
	try:
		dir_name = os.path.dirname(path) or "."
		base_name = os.path.basename(path)
		fd, temp_path = tempfile.mkstemp(
			prefix=base_name + ".",
			suffix=".tmp",
			dir=dir_name,
			text=True
		)
		
		try:
			with os.fdopen(fd, "w", encoding="utf-8") as f:
				f.write(content)
				f.flush()
				if hasattr(os, 'fsync'):
					try:
						os.fsync(f.fileno())
					except (OSError, AttributeError):
						pass
			
			time.sleep(0.01)
			
			if os.path.exists(path):
				try:
					os.remove(path)
				except (IOError, OSError):
					pass
			
			shutil.move(temp_path, path)
			return True
			
		except Exception:
			try:
				if os.path.exists(temp_path):
					os.remove(temp_path)
			except Exception:
				pass
			return False
			
	except Exception:
		try:
			with io.open(path, "w", encoding="utf-8") as f:
				f.write(content)
				f.flush()
				if hasattr(os, 'fsync'):
					try:
						os.fsync(f.fileno())
					except (OSError, AttributeError):
						pass
			time.sleep(0.01)
			return True
		except Exception:
			return False


# ID ∈ {PV, LD, B01..B99}

def parse_status_line(line: str) -> Tuple[str, datetime, float, float, float, float]:
	parts = [p.strip() for p in line.strip().split(",")]
	if len(parts) < 6:
		raise ValueError(f"Invalid status line: {line!r}")
	idv = parts[0]
	ts = parse_ts(parts[1])
	v = parse_unsigned_field(parts[2])     # V
	i = parse_signed_field(parts[3])
	f = parse_unsigned_field(parts[4]) if parts[4] else 0.0
	soc = parse_unsigned_field(parts[5]) if parts[5] else 0.0  # %
	return idv, ts, v, i, f, soc


def format_status_line(idv: str, ts: datetime, volt_v: float, curr_a: float, flow_ccm: float, soc_pct: float) -> str:
	return ",".join([
		idv,
		format_ts(ts),
		format_unsigned_field(volt_v),
		format_signed_field(curr_a),
		format_unsigned_field(flow_ccm),
		format_unsigned_field(soc_pct),
		" ", " ", " "
	])


def read_status_file(path: str, max_age_sec: Optional[int] = None) -> Dict[str, Tuple[datetime, float, float, float, float]]:
	"""
	{ ID: (ts, volt_v, curr_a, flow_ccm, soc_pct) }
	"""
	results: Dict[str, Tuple[datetime, float, float, float, float]] = {}
	try:
		if not os.path.exists(path):
			return results
		with io.open(path, "r", encoding="utf-8") as f:
			lines = [ln for ln in f.readlines() if ln.strip()]
	except Exception:
		return results
	finally:
		try:
			with io.open(path, "w", encoding="utf-8") as w:
				w.write("")
		except Exception:
			pass

	now_ts = datetime.now(TZ_UTC8)
	for ln in lines:
		try:
			idv, ts, v, i, flow, soc = parse_status_line(ln)
		except Exception:
			continue
		if max_age_sec is not None and (now_ts - ts).total_seconds() > float(max_age_sec):
			continue
		if idv not in results or ts > results[idv][0]:
			results[idv] = (ts, v, i, flow, soc)
	return results


#
#


class _VendorDataResult(dict):
	"""
	
		                    result['grid'], result['batteries']
	"""
	def __init__(self, mppt, mppt_bus, load, grid, batteries, timestamp):
		super().__init__(
			mppt=mppt,
			mppt_bus=mppt_bus,
			load=load,
			grid=grid,
			batteries=batteries,
			timestamp=timestamp,
		)
		self._tuple = (mppt, batteries)
	
	def __iter__(self):
		"""Documentation for this public API is provided in English."""
		return iter(self._tuple)
	
	def __len__(self):
		return 2

def parse_vendor_data_line(line: str) -> Tuple[str, float, float, float, float, float, float]:
	"""
	
	
		- SOC: 0.1% (101 = 10.1%)
		- TEMP: 0.1°C (332 = 33.2°C)
	
	Returns:
		(pp, soc_pct, volt_v, charge_v, curr_ma, temp_c, speed)
	"""
	parts = [p.strip() for p in line.strip().split(",") if p.strip()]
	if len(parts) < 6:
		raise ValueError(f"Invalid vendor data line: {line!r}")
	
	pp = parts[0].zfill(2)
	soc_raw = int(parts[1]) if parts[1].isdigit() else 0
	soc_pct = float(soc_raw) / 10.0
	
	bv_raw = int(parts[2]) if parts[2].isdigit() else 0
	volt_v = float(bv_raw) / 100.0
	
	if len(parts) >= 7:
		cv_raw = int(parts[3]) if parts[3].isdigit() else 0
		charge_v = float(cv_raw) / 100.0
		
		try:
			ci_raw = int(parts[4])
		except (ValueError, TypeError):
			ci_raw = 0
		curr_ma = float(ci_raw)
		
		temp_raw = int(parts[5]) if parts[5].isdigit() else 0
		temp_c = float(temp_raw) / 10.0
		
		speed_raw = int(parts[6]) if parts[6].isdigit() else 0
		speed = float(speed_raw) / 10.0
	else:
		charge_v = 0.0
		
		try:
			bi_raw = int(parts[3])
		except (ValueError, TypeError):
			bi_raw = 0
		curr_ma = float(bi_raw)
		
		temp_raw = int(parts[4]) if parts[4].isdigit() else 0
		temp_c = float(temp_raw) / 10.0
		
		speed_raw = int(parts[5]) if parts[5].isdigit() else 0
		speed = float(speed_raw) / 10.0
	
	return pp, soc_pct, volt_v, charge_v, curr_ma, temp_c, speed


def parse_mppt_line(line: str) -> Tuple[float, float, float, float, float, float]:
	"""
	
	
		- SolarV/MPPT_V: 0.01V（1600 = 16.00V）
	
	Returns:
		(solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw)
	"""
	parts = [p.strip() for p in line.strip().split(",") if p.strip()]
	if len(parts) < 6:
		raise ValueError(f"Invalid MPPT line: {line!r}")
	
	solar_v_raw = int(parts[0]) if parts[0].isdigit() else 0
	solar_v = float(solar_v_raw) / 100.0
	
	solar_i_raw = int(parts[1]) if parts[1].isdigit() else 0
	solar_i_ma = float(solar_i_raw)
	
	solar_p_raw = int(parts[2]) if parts[2].isdigit() else 0
	solar_p_mw = float(solar_p_raw)
	
	mppt_v_raw = int(parts[3]) if parts[3].isdigit() else 0
	mppt_v = float(mppt_v_raw) / 100.0
	
	mppt_i_raw = int(parts[4]) if parts[4].isdigit() else 0
	mppt_i_ma = float(mppt_i_raw)
	
	mppt_p_raw = int(parts[5]) if parts[5].isdigit() else 0
	mppt_p_mw = float(mppt_p_raw)
	
	return solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw


def parse_mppt_line_v2(line: str) -> Tuple[
	Tuple[float, float, float, float, float, float],
	Optional[Tuple[float, float, float]]
]:
	"""
	
		SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,BusV,BusI,BusP,
	
		SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,
	
		- V: 0.01V（1600 = 16.00V）
		- I: 1mA（500 = 500 mA）
		- P: 1mW（8000 = 8000 mW）
	
	Returns:
		(mppt_6tuple, mppt_bus_3tuple_or_None)
		mppt_6tuple: (solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw)
	"""
	parts = [p.strip() for p in line.strip().split(",") if p.strip()]
	if len(parts) < 6:
		raise ValueError(f"Invalid MPPT line: {line!r}")
	
	solar_v = float(int(parts[0]) if parts[0].isdigit() else 0) / 100.0
	solar_i_ma = float(int(parts[1]) if parts[1].isdigit() else 0)
	solar_p_mw = float(int(parts[2]) if parts[2].isdigit() else 0)
	mppt_v = float(int(parts[3]) if parts[3].isdigit() else 0) / 100.0
	mppt_i_ma = float(int(parts[4]) if parts[4].isdigit() else 0)
	mppt_p_mw = float(int(parts[5]) if parts[5].isdigit() else 0)
	
	mppt_6 = (solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw)
	
	mppt_bus = None
	if len(parts) >= 9:
		bus_v = float(int(parts[6]) if parts[6].isdigit() else 0) / 100.0
		bus_i_ma = float(int(parts[7]) if parts[7].isdigit() else 0)
		bus_p_mw = float(int(parts[8]) if parts[8].isdigit() else 0)
		mppt_bus = (bus_v, bus_i_ma, bus_p_mw)
	
	return mppt_6, mppt_bus


def parse_load_line(line: str) -> Tuple[Tuple[float, float, float], Optional[Tuple[float, float, float]]]:
	"""
	
	
		- V: 0.01V（1200 = 12.00V）
		- I: 1mA（5500 = 5500 mA）
		- P: 1mW（6600 = 6600 mW）
	
	Returns:
		( (load_v, load_i_ma, load_p_mw),
	"""
	parts = [p.strip() for p in line.strip().split(",") if p.strip()]
	if len(parts) < 3:
		raise ValueError(f"Invalid load line: {line!r}")
	
	load_v = float(int(parts[0]) if parts[0].isdigit() else 0) / 100.0   # 0.01V
	load_i_ma = float(int(parts[1]) if parts[1].isdigit() else 0)        # mA
	load_p_mw = float(int(parts[2]) if parts[2].isdigit() else 0)        # mW
	
	grid = None
	if len(parts) >= 6:
		grid_v = float(int(parts[3]) if parts[3].isdigit() else 0) / 100.0
		grid_i_ma = float(int(parts[4]) if parts[4].isdigit() else 0)
		grid_p_mw = float(int(parts[5]) if parts[5].isdigit() else 0)
		grid = (grid_v, grid_i_ma, grid_p_mw)
	
	return (load_v, load_i_ma, load_p_mw), grid


def read_vendor_data_file(path: str, max_age_sec: Optional[int] = None, 
                          clear_after_read: bool = True) -> Dict:
	"""
	
	
	
	
	
	Args:
	
	Returns:
		dict with keys:
			'mppt':      (solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw) or None
			'mppt_bus':  (bus_v, bus_i_ma, bus_p_mw) or None
			'load':      (load_v, load_i_ma, load_p_mw) or None
			'batteries': { PP: (ts, soc_pct, volt_v, charge_v, curr_ma, temp_c, speed) }
			'timestamp': datetime or None
		
	"""
	results: Dict[str, Tuple] = {}
	mppt_data: Optional[Tuple[float, float, float, float, float, float]] = None
	mppt_bus_data: Optional[Tuple[float, float, float]] = None
	load_data: Optional[Tuple[float, float, float]] = None
	grid_data: Optional[Tuple[float, float, float]] = None
	file_ts: Optional[datetime] = None
	read_ts = datetime.now(TZ_UTC8)
	
	try:
		if not os.path.exists(path):
			return _VendorDataResult(mppt_data, mppt_bus_data, load_data, grid_data, results, file_ts)
		
		lines: List[str] = []
		try:
			with io.open(path, "r", encoding="utf-8") as f:
				lines = [ln for ln in f.readlines() if ln.strip()]
		except (IOError, OSError, PermissionError):
			return _VendorDataResult(mppt_data, mppt_bus_data, load_data, grid_data, results, file_ts)
		
		if not lines:
			return _VendorDataResult(mppt_data, mppt_bus_data, load_data, grid_data, results, file_ts)
		
		file_ts = read_ts
		try:
			first_line = lines[0].strip()
			if len(first_line) >= 14:
				ts_part = first_line[:14]
				if ts_part.isdigit():
					file_ts = parse_ts(ts_part)
					lines = lines[1:]
		except Exception:
			pass
		
		if max_age_sec is not None:
			age_sec = (datetime.now(TZ_UTC8) - file_ts).total_seconds()
			if age_sec > float(max_age_sec):
				return _VendorDataResult(None, None, None, None, {}, file_ts)
		
		if lines:
			first_data_line = lines[0].strip()
			first_parts = [p.strip() for p in first_data_line.split(",") if p.strip()]
			if len(first_parts) >= 6:
				first_field = first_parts[0]
				if not (first_field.isdigit() and 1 <= int(first_field) <= 10):
					try:
						mppt_6, mppt_bus_data = parse_mppt_line_v2(first_data_line)
						mppt_data = mppt_6
						lines = lines[1:]
					except Exception:
						lines = lines[1:]
		
		if lines:
			load_line = lines[0].strip()
			load_parts = [p.strip() for p in load_line.split(",") if p.strip()]
			if len(load_parts) >= 3:
				first_field = load_parts[0]
				is_battery = (first_field.isdigit() and 1 <= int(first_field) <= 10 
				              and len(load_parts) >= 6 and len(load_parts) <= 7)
				if not is_battery:
					try:
						load_tuple, grid_tuple = parse_load_line(load_line)
						load_data = load_tuple
						grid_data = grid_tuple  # None if old 3-field format
						lines = lines[1:]
					except Exception:
						lines = lines[1:]
		
		for ln in lines:
			try:
				pp, soc_pct, volt_v, charge_v, curr_ma, temp_c, speed = parse_vendor_data_line(ln)
			except Exception:
				continue
			results[pp] = (file_ts, soc_pct, volt_v, charge_v, curr_ma, temp_c, speed)
		
	except Exception:
		return _VendorDataResult(mppt_data, mppt_bus_data, load_data, grid_data, results, file_ts)
	finally:
		if clear_after_read:
			try:
				with io.open(path, "w", encoding="utf-8") as w:
					w.write("")
					w.flush()
					if hasattr(os, 'fsync'):
						try:
							os.fsync(w.fileno())
						except (OSError, AttributeError):
							pass
				time.sleep(0.01)
			except (IOError, OSError, PermissionError):
				pass
	
	return _VendorDataResult(mppt_data, mppt_bus_data, load_data, grid_data, results, file_ts)



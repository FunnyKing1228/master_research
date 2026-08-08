import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings

import math
from typing import Tuple, Optional
from collections import deque


warnings.filterwarnings('ignore')

_conformal_window = 2880
_conformal_delta = 0.1
_residual_buffer: deque[float] = deque(maxlen=_conformal_window)


def set_conformal_params(window: int = 2880, delta: float = 0.1):
    global _conformal_window, _conformal_delta, _residual_buffer
    _conformal_window = max(10, int(window))
    _conformal_delta = float(np.clip(delta, 1e-4, 0.5))
    # Recreate the buffer to match the new window.
    old_vals = list(_residual_buffer) if _residual_buffer else []
    _residual_buffer = deque(old_vals[-_conformal_window:], maxlen=_conformal_window)


def update_conformal_residual(residual: float):
    try:
        r = float(abs(residual))
    except Exception:
        return
    _residual_buffer.append(r)


def clear_residual_buffer():
    """Clear the conformal residual buffer while preserving its window size."""
    global _residual_buffer
    _residual_buffer = deque(maxlen=_conformal_window)


def get_residual_count() -> int:
    """Return the current number of residual samples."""
    try:
        return int(len(_residual_buffer))
    except Exception:
        return 0


def _conformal_tube() -> float:
    if not _residual_buffer:
        return 0.0
    arr = np.asarray(_residual_buffer, dtype=float)
    q = min(0.999, max(0.0, 1.0 - _conformal_delta))
    try:
        return float(np.quantile(arr, q))
    except Exception:
        return float(np.max(arr))

class SafetyNet:
    """
    SafetyNet action projection and shielding system.
    
    Responsibilities:
    - compute safe action bounds,
    - project unsafe actions back into the safe set,
    - manage dynamic safety buffers,
    - support N-step preview constraints,
    - shrink boundaries to reduce false-safe decisions.
    """
    
    def __init__(
        self,
        battery_capacity_kwh: float = 100.0,
        battery_power_kw: float = 50.0,
        battery_efficiency: float = 0.95,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        initial_buffer_ratio: float = 0.05,  # Initial SoC buffer ratio.
        min_buffer_ratio: float = 0.02,  # Lower bound for adaptive buffer shrinkage.
        buffer_decay_episodes: int = 10,  # Safe episodes required before shrinking.
        buffer_decay_rate: float = 0.01,  # Buffer shrinkage per decay event.
        boundary_epsilon: float = 0.005,  # Extra inward boundary margin.
        time_step: float = 1.0,  # Step length in hours.
        n_step_preview: int = 2,  # Preview horizon in steps.
        enable_n_step_preview: bool = True
    ):
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_power_kw = battery_power_kw
        self.battery_efficiency = battery_efficiency
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.initial_buffer_ratio = initial_buffer_ratio
        self.min_buffer_ratio = min_buffer_ratio
        self.buffer_decay_episodes = buffer_decay_episodes
        self.buffer_decay_rate = buffer_decay_rate
        self.boundary_epsilon = boundary_epsilon
        self.time_step = time_step
        self.n_step_preview = n_step_preview
        self.enable_n_step_preview = enable_n_step_preview
        
        # Dynamic buffer management.
        self.current_buffer_ratio = initial_buffer_ratio
        self.consecutive_safe_episodes = 0
        
        # Compute the current safety buffer.
        self._update_safe_bounds()
        
        print(f"SafetyNet Phase-1 initialized:")
        print(f"  - Battery capacity: {battery_capacity_kwh:.1f} kWh")
        print(f"  - Battery power: {battery_power_kw:.1f} kW")
        print(f"  - Initial safe SoC range: [{self.safe_soc_min:.3f}, {self.safe_soc_max:.3f}]")
        print(f"  - Buffer ratio: {self.current_buffer_ratio:.1%}")
        print(f"  - N-step preview: {n_step_preview}")
        print(f"  - Boundary epsilon: {boundary_epsilon:.3f}")
    
    def _update_safe_bounds(self):
        """Update safe SoC bounds using the current buffer and inward margin."""
        buffer_size = (self.soc_max - self.soc_min) * self.current_buffer_ratio
        self.safe_soc_min = self.soc_min + buffer_size + self.boundary_epsilon
        self.safe_soc_max = self.soc_max - buffer_size - self.boundary_epsilon
    
    def _calculate_soc_bounds(self, current_soc: float) -> Tuple[float, float]:
        """
        Compute safe action bounds from the current SoC.
        
        Args:
            current_soc: Current battery SoC.
            
        Returns:
            (min_power, max_power): Safe power range in kW.
        """
        # Base power limits.
        base_min_power = -self.battery_power_kw
        base_max_power = self.battery_power_kw
        
        # Check whether SoC is inside the safety buffer.
        in_buffer_zone = (current_soc <= self.safe_soc_min or 
                         current_soc >= self.safe_soc_max)
        
        if in_buffer_zone:
            # Inside the buffer, shrink the allowed power range.
            base_min_power *= 0.5  # Use a fixed scaling factor.
            base_max_power *= 0.5
        
        return base_min_power, base_max_power
    
    def _calculate_preview_bounds(self, current_soc: float) -> Tuple[float, float]:
        """
        Compute single-step feasible charge/discharge bounds.

        Bounds are derived from SoC, efficiency, and time step, then made
        conservative with safe SoC limits and optional n-step preview shrinkage.
        """
        dt = max(float(self.time_step), 1e-9)
        eta = max(float(self.battery_efficiency), 1e-9)
        cap = max(float(self.battery_capacity_kwh), 1e-9)
        # Use conservative safe bounds.
        soc_min_eff = float(getattr(self, 'safe_soc_min', self.soc_min))
        soc_max_eff = float(getattr(self, 'safe_soc_max', self.soc_max))
        # Remaining charge/discharge energy in kWh.
        charge_room_kwh = max(0.0, (soc_max_eff - current_soc) * cap)
        discharge_room_kwh = max(0.0, (current_soc - soc_min_eff) * cap)
        # Convert to single-step power limits in kW.
        max_charge_kw = charge_room_kwh / (dt * eta)  # Charging divides by efficiency.
        max_discharge_kw = discharge_room_kwh * eta / dt  # Discharging multiplies by efficiency.
        # Shrink by n-step preview for conservatism.
        if self.enable_n_step_preview and int(self.n_step_preview) > 1:
            factor = 1.0 / float(int(self.n_step_preview))
            max_charge_kw *= factor
            max_discharge_kw *= factor
        # Clamp to physical power limits.
        max_charge_kw = min(max_charge_kw, self.battery_power_kw)
        max_discharge_kw = min(max_discharge_kw, self.battery_power_kw)
        return -max_discharge_kw, max_charge_kw
    
    def bounds(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Compute safe action bounds for a given environment state.
        
        Args:
            state: Environment state [SoC, load, pv, price, hour, day].
            
        Returns:
            (a_low, a_high): Safe action range.
        """
        current_soc = state[0]
        
        # Base bounds.
        base_min, base_max = self._calculate_soc_bounds(current_soc)
        
        # Preview bounds independent of a specific action.
        preview_min, preview_max = self._calculate_preview_bounds(current_soc)
        
        # Take the intersection.
        a_low = max(base_min, preview_min)
        a_high = min(base_max, preview_max)
        
        # Ensure bounds are valid.
        a_low = max(a_low, -self.battery_power_kw)
        a_high = min(a_high, self.battery_power_kw)
        
        return a_low, a_high
    
    def project(self, state: np.ndarray, action_raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Project a raw action into the safe action set.
        
        Args:
            state: Environment state.
            action_raw: Raw action.
            
        Returns:
            (action_safe, info): Safe action and projection diagnostics.
        """
        action_value = float(action_raw[0])
        
        # Compute safety bounds.
        a_low, a_high = self.bounds(state)
        
        # Project the action.
        action_safe = np.clip(action_value, a_low, a_high)
        
        # Check whether the action was clipped.
        clipped = abs(action_safe - action_value) > 1e-6
        
        # Compute clipping magnitude.
        if clipped:
            if action_value < a_low:
                clip_ratio = (a_low - action_value) / abs(action_value) if action_value != 0 else 1.0
            else:
                clip_ratio = (action_value - a_high) / abs(action_value) if action_value != 0 else 1.0
        else:
            clip_ratio = 0.0
        
        # Prepare diagnostics.
        info = {
            'clipped': clipped,
            'clip_ratio': clip_ratio,
            'original_action': action_value,
            'safe_bounds': (a_low, a_high),
            'soc': state[0],
            'in_buffer_zone': (state[0] <= self.safe_soc_min or state[0] >= self.safe_soc_max)
        }
        
        return np.array([action_safe], dtype=np.float32), info
    
    def get_safety_metrics(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Return safety metrics for a given state.
        
        Args:
            state: Environment state.
            
        Returns:
            Dictionary of safety metrics.
        """
        current_soc = state[0]
        
        # Compute distance to bounds.
        distance_to_min = current_soc - self.soc_min
        distance_to_max = self.soc_max - current_soc
        
        # Compute distance to safe bounds.
        distance_to_safe_min = current_soc - self.safe_soc_min
        distance_to_safe_max = self.safe_soc_max - current_soc
        
        # Classify safety level.
        buffer_size = (self.soc_max - self.soc_min) * self.current_buffer_ratio
        if current_soc < self.safe_soc_min:
            safety_level = "danger_low"
        elif current_soc > self.safe_soc_max:
            safety_level = "danger_high"
        elif current_soc < self.safe_soc_min + buffer_size * 0.5:
            safety_level = "warning_low"
        elif current_soc > self.safe_soc_max - buffer_size * 0.5:
            safety_level = "warning_high"
        else:
            safety_level = "safe"
        
        return {
            'safety_level': safety_level,
            'distance_to_min': distance_to_min,
            'distance_to_max': distance_to_max,
            'distance_to_safe_min': distance_to_safe_min,
            'distance_to_safe_max': distance_to_safe_max,
            'buffer_utilization': 1.0 - min(distance_to_safe_min, distance_to_safe_max) / buffer_size,
            'current_soc': current_soc,
            'safe_range': (self.safe_soc_min, self.safe_soc_max)
        }
    
    def update_parameters(self, **kwargs):
        """
        Dynamically update SafetyNet parameters.
        
        Args:
            **kwargs: Parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} to {value}")
        
        # Recompute the safety buffer.
        if 'buffer_ratio' in kwargs or 'soc_min' in kwargs or 'soc_max' in kwargs:
            self._update_safe_bounds()
            print(f"Recalculated safe SoC range: [{self.safe_soc_min:.3f}, {self.safe_soc_max:.3f}]")

    def update_buffer_after_episode(self, realized_violations: int):
        """Update the dynamic safety buffer after an episode."""
        if realized_violations == 0:
            self.consecutive_safe_episodes += 1
            if self.consecutive_safe_episodes >= self.buffer_decay_episodes:
                # Shrink the buffer.
                new_buffer = self.current_buffer_ratio - self.buffer_decay_rate
                if new_buffer >= self.min_buffer_ratio:
                    self.current_buffer_ratio = new_buffer
                    self._update_safe_bounds()
                    print(f"SafetyNet: Buffer shrunk to {self.current_buffer_ratio:.1%}")
        else:
            # Reset the safe-episode counter.
            self.consecutive_safe_episodes = 0
    
    def _n_step_soc_prediction(self, current_soc: float, action: float, env) -> float:
        """Predict SoC after applying the same action for N steps."""
        if not self.enable_n_step_preview:
            return current_soc
        
        soc = current_soc
        for step in range(self.n_step_preview):
            # Use the environment prediction method.
            if hasattr(env, 'predict_soc_raw'):
                soc = env.predict_soc_raw(soc, action)
            else:
                # Fallback prediction.
                energy_change_kwh = action * self.time_step
                if energy_change_kwh > 0:
                    energy_change_kwh *= self.battery_efficiency
                else:
                    energy_change_kwh /= self.battery_efficiency
                soc += energy_change_kwh / self.battery_capacity_kwh
        
        return soc


class SafetyNetWrapper:
    """
    Thin wrapper that exposes a unified SafetyNet interface.
    """
    
    def __init__(self, safety_net: SafetyNet):
        self.safety_net = safety_net
    
    def __call__(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Project an action through the wrapped SafetyNet."""
        return self.safety_net.project(state, action)
    
    def bounds(self, state: np.ndarray) -> Tuple[float, float]:
        """Return safe action bounds."""
        return self.safety_net.bounds(state)
    
    def get_safety_metrics(self, state: np.ndarray) -> Dict[str, Any]:
        """Return safety metrics."""
        return self.safety_net.get_safety_metrics(state)


def create_safety_net(
    battery_capacity_kwh: float = 100.0,
    battery_power_kw: float = 50.0,
    battery_efficiency: float = 0.95,
    soc_min: float = 0.1,
    soc_max: float = 0.9,
    buffer_ratio: float = 0.08,
    buffer_scale_kappa: float = 0.5,
    enable_one_step_preview: bool = True
) -> SafetyNet:
    """
    Create a configured SafetyNet instance.
    
    Args:
        battery_capacity_kwh: Battery capacity in kWh.
        battery_power_kw: Battery power limit in kW.
        battery_efficiency: Battery efficiency.
        soc_min: Minimum allowed SoC.
        soc_max: Maximum allowed SoC.
        buffer_ratio: Safety buffer ratio.
        buffer_scale_kappa: Boundary scaling factor.
        enable_one_step_preview: Whether to enable one-step preview.
        
    Returns:
        Configured SafetyNet instance.
    """
    return SafetyNet(
        battery_capacity_kwh=battery_capacity_kwh,
        battery_power_kw=battery_power_kw,
        battery_efficiency=battery_efficiency,
        soc_min=soc_min,
        soc_max=soc_max,
        buffer_ratio=buffer_ratio,
        buffer_scale_kappa=buffer_scale_kappa,
        enable_one_step_preview=enable_one_step_preview
    ) 


def _clip(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))


def _apply_ramp_limit(a_new: float, a_prev: float, ramp_kw: Optional[float]) -> float:
	if ramp_kw is None or ramp_kw <= 0:
		return a_new
	return _clip(a_new, a_prev - ramp_kw, a_prev + ramp_kw)


def _solve_a_for_soc_target(soc: float, soc_target: float, env) -> float:
	"""Solve the action a in kW required to reach a target SoC.
	E = (soc_target - soc) * cap
	If E >= 0 (charging): a = E / (dt * eta)
	If E < 0 (discharging): a = E * eta / dt
	"""
	dt = float(getattr(env, 'time_step', 1.0))
	eta = float(getattr(env, 'battery_efficiency', 1.0))
	cap = float(getattr(env, 'battery_capacity_kwh', 1.0))
	if cap <= 0 or dt <= 0 or eta <= 0:
		return 0.0
	E = (soc_target - soc) * cap
	if E >= 0:
		return E / (dt * eta)
	else:
		return E * eta / dt


def _project_soc_safe(soc: float, a: float, soc_min: float, soc_max: float, env) -> float:
	"""Adjust a to land on the nearest SoC boundary if it would violate bounds."""
	predict = getattr(env, 'predict_soc_raw', None)
	if predict is None:
		return a
	soc_next = float(predict(soc, a))
	if soc_min <= soc_next <= soc_max:
		return a
	# Choose the nearest boundary target to avoid crossing the other side.
	target = soc_min if soc_next < soc_min else soc_max
	return _solve_a_for_soc_target(soc, target, env)



def project(
	state: np.ndarray,
	action: np.ndarray,
	prev_action: float,
	pmax: float,
	ramp_kw: Optional[float],
	soc_bounds: Tuple[float, float],
	env,
	pmin: Optional[float] = None,
	pmax_positive: Optional[float] = None,
) -> Tuple[float, bool, float]:
	"""Project an action through Pmax, ramp, SoC, Pmax, and final scaling checks.

	Returns (a_safe, changed, delta_kw), where delta_kw = |a_safe - a_raw|.
	"""
	a_raw = float(action[0])
	soc = float(state[0])
	soc_min, soc_max = soc_bounds
	# Conformal risk tube: shrink bounds by residual quantile.
	tube = _conformal_tube()
	soc_min_eff = float(soc_min + tube)
	soc_max_eff = float(soc_max - tube)
	changed = False
	lower_kw = -abs(float(pmax if pmin is None else pmin))
	upper_kw = abs(float(pmax if pmax_positive is None else pmax_positive))
	# 1) Pmax
	a1 = _clip(a_raw, lower_kw, upper_kw)
	changed = changed or (abs(a1 - a_raw) > 1e-8)
	# 2) ramp, if provided.
	a2 = _apply_ramp_limit(a1, float(prev_action), ramp_kw) if ramp_kw is not None else a1
	# 2) SoC boundary projection.
	a3 = _project_soc_safe(soc, a2, soc_min_eff, soc_max_eff, env)
	changed = changed or (abs(a3 - a2) > 1e-8)
	# 3) Apply Pmax correction again.
	a4 = _clip(a3, lower_kw, upper_kw)
	if abs(a4 - a3) > 1e-8:
		changed = True
	# 4) Final verification: scale into the safe range if still out of bounds.
	predict = getattr(env, 'predict_soc_raw', None)
	if predict is not None:
		soc_next = float(predict(soc, a4))
		if soc_next < soc_min_eff or soc_next > soc_max_eff:
			low, high = 0.0, 1.0
			base = a4
			for _ in range(12):
				mid = (low + high) * 0.5
				a_try = base * mid
				soc_try = float(predict(soc, a_try))
				if soc_min_eff <= soc_try <= soc_max_eff:
					low = mid
				else:
					high = mid
			a4 = base * low
			changed = True
	delta_kw = abs(float(a4) - a_raw)
	return float(a4), changed, float(delta_kw)

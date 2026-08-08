from __future__ import annotations

from typing import Sequence


def denormalize_price_twd(price_norm: float) -> float:
    """Convert normalized TOU price back to approximate TWD/kWh."""
    return float(price_norm) * 10.0


def rule_expert_action_from_state(
    state: Sequence[float],
    battery_charge_power_kw: float,
    battery_discharge_power_kw: float,
    soc_min: float = 0.10,
    soc_max: float = 0.90,
    mode: str = "conservative_tou",
) -> float:
    """
    Simple explainable expert for warm-start demos.

    Policy idea:
    1. If PV exists and battery is not near full, charge strongly.
    2. If no PV and solo-discharge is feasible, discharge according to chosen expert mode.
    3. If SoC is extremely low during off-peak, allow light grid charging as recovery.
    4. Otherwise stay idle.

    Output is normalized SAC action in [-1, 1].
    """
    soc = float(state[0])
    load_kw = float(state[1])
    pv_obs = float(state[2])
    price_twd = denormalize_price_twd(float(state[3]))

    pv_active = pv_obs > 0.5
    discharge_feasible = load_kw <= (battery_discharge_power_kw / 0.99 + 1e-9)

    soc_charge_ceiling = soc_max - 0.02
    soc_discharge_floor = soc_min + 0.05
    soc_recovery_floor = soc_min + 0.02
    soc_aggressive_floor = soc_min + 0.10

    if pv_active and soc < soc_charge_ceiling:
        return 1.0

    if not pv_active and discharge_feasible:
        if mode == "aggressive_discharge":
            if soc > soc_aggressive_floor:
                return -1.0
        else:
            if soc > soc_discharge_floor and price_twd >= 4.69:
                return -1.0

    if (not pv_active) and soc < soc_recovery_floor and price_twd <= 2.06:
        return 0.6

    return 0.0

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')


class LagrangianMultiplier:
    """
    
    """
    
    def __init__(
        self,
        target_cost_rate: float = 0.05,
        learning_rate: float = 1e-3,
        min_value: float = 0.0,
        max_value: float = 10.0,
        update_frequency: int = 1000,
        window_size: int = 1000
    ):
        self.target_cost_rate = target_cost_rate
        self.learning_rate = learning_rate
        self.min_value = min_value
        self.max_value = max_value
        self.update_frequency = update_frequency
        self.window_size = window_size
        
        self.lambda_value = 1.0
        
        self.cost_history = []
        self.violation_history = []
        self.step_count = 0
        
        print(f"✓ LagrangianMultiplier initialized:")
        print(f"  - Target cost rate: {target_cost_rate:.1%}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Value range: [{min_value}, {max_value}]")
        print(f"  - Update frequency: {update_frequency}")
        print(f"  - Window size: {window_size}")
    
    def update(self, cost: float, violation: bool = False):
        """
        
        Args:
        """
        self.cost_history.append(cost)
        self.violation_history.append(violation)
        self.step_count += 1
        
        if len(self.cost_history) > self.window_size:
            self.cost_history.pop(0)
            self.violation_history.pop(0)
    
    def get_current_cost_rate(self) -> float:
        """Documentation for this public API is provided in English."""
        if not self.violation_history:
            return 0.0
        return np.mean(self.violation_history)
    
    def get_current_avg_cost(self) -> float:
        """Documentation for this public API is provided in English."""
        if not self.cost_history:
            return 0.0
        return np.mean(self.cost_history)
    
    def update_multiplier(self) -> float:
        """
        
        Returns:
        """
        if self.step_count % self.update_frequency != 0:
            return self.lambda_value
        
        current_cost_rate = self.get_current_cost_rate()
        
        cost_gap = current_cost_rate - self.target_cost_rate
        lambda_update = self.lambda_value + self.learning_rate * cost_gap
        
        self.lambda_value = np.clip(lambda_update, self.min_value, self.max_value)
        
        print(f"Lagrangian update: cost_rate={current_cost_rate:.3f}, "
              f"target={self.target_cost_rate:.3f}, lambda={self.lambda_value:.3f}")
        
        return self.lambda_value
    
    def get_multiplier(self) -> float:
        """Documentation for this public API is provided in English."""
        return self.lambda_value
    
    def reset(self):
        """Documentation for this public API is provided in English."""
        self.cost_history.clear()
        self.violation_history.clear()
        self.step_count = 0
        print("✓ LagrangianMultiplier statistics reset")


class ConstraintCostCalculator:
    """
    
    """
    
    def __init__(
        self,
        soc_violation_weight: float = 1.0,
        action_violation_weight: float = 0.5,
        buffer_violation_weight: float = 0.3,
        grid_stability_weight: float = 0.2
    ):
        self.soc_violation_weight = soc_violation_weight
        self.action_violation_weight = action_violation_weight
        self.buffer_violation_weight = buffer_violation_weight
        self.grid_stability_weight = grid_stability_weight
        
        print(f"✓ ConstraintCostCalculator initialized:")
        print(f"  - SoC violation weight: {soc_violation_weight}")
        print(f"  - Action violation weight: {action_violation_weight}")
        print(f"  - Buffer violation weight: {buffer_violation_weight}")
        print(f"  - Grid stability weight: {grid_stability_weight}")
    
    def calculate_cost(
        self,
        soc_violations: int = 0,
        action_violations: int = 0,
        buffer_hits: int = 0,
        grid_instability: float = 0.0,
        safety_clipped: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        
        Args:
            
        Returns:
        """
        soc_cost = soc_violations * self.soc_violation_weight
        action_cost = action_violations * self.action_violation_weight
        buffer_cost = buffer_hits * self.buffer_violation_weight
        grid_cost = grid_instability * self.grid_stability_weight
        
        safety_cost = 1.0 if safety_clipped else 0.0
        
        total_cost = soc_cost + action_cost + buffer_cost + grid_cost + safety_cost
        
        cost_breakdown = {
            'soc_violations': soc_cost,
            'action_violations': action_cost,
            'buffer_hits': buffer_cost,
            'grid_instability': grid_cost,
            'safety_clipped': safety_cost,
            'total_cost': total_cost
        }
        
        return cost_breakdown
    
    def calculate_violation_rate(self, violations: List[bool]) -> float:
        """Documentation for this public API is provided in English."""
        if not violations:
            return 0.0
        return np.mean(violations)
    
    def is_constraint_satisfied(self, cost: float, threshold: float = 0.1) -> bool:
        """Documentation for this public API is provided in English."""
        return cost <= threshold


class LagrangianSACAgent:
    """
    """
    
    def __init__(
        self,
        base_agent,
        constraint_calculator: ConstraintCostCalculator,
        lagrangian_multiplier: LagrangianMultiplier
    ):
        self.base_agent = base_agent
        self.constraint_calculator = constraint_calculator
        self.lagrangian_multiplier = lagrangian_multiplier
        
        print("✓ LagrangianSACAgent initialized")
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Documentation for this public API is provided in English."""
        return self.base_agent.select_action(state, evaluate)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Documentation for this public API is provided in English."""
        self.base_agent.store_transition(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, Any]:
        """Documentation for this public API is provided in English."""
        return self.base_agent.update()
    
    def calculate_constraint_cost(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Documentation for this public API is provided in English."""
        return self.constraint_calculator.calculate_cost(**info)
    
    def update_lagrangian_multiplier(self, cost: float, violation: bool = False):
        """Documentation for this public API is provided in English."""
        self.lagrangian_multiplier.update(cost, violation)
        return self.lagrangian_multiplier.update_multiplier()
    
    def get_modified_reward(self, original_reward: float, constraint_cost: float) -> float:
        """Documentation for this public API is provided in English."""
        lambda_val = self.lagrangian_multiplier.get_multiplier()
        modified_reward = original_reward - lambda_val * constraint_cost
        return modified_reward
    
    def get_constraint_stats(self) -> Dict[str, Any]:
        """Documentation for this public API is provided in English."""
        return {
            'lambda_value': self.lagrangian_multiplier.get_multiplier(),
            'current_cost_rate': self.lagrangian_multiplier.get_current_cost_rate(),
            'current_avg_cost': self.lagrangian_multiplier.get_current_avg_cost(),
            'target_cost_rate': self.lagrangian_multiplier.target_cost_rate
        }


def create_lagrangian_multiplier(
    target_cost_rate: float = 0.05,
    learning_rate: float = 1e-3,
    min_value: float = 0.0,
    max_value: float = 10.0,
    update_frequency: int = 1000,
    window_size: int = 1000
) -> LagrangianMultiplier:
    """
    """
    return LagrangianMultiplier(
        target_cost_rate=target_cost_rate,
        learning_rate=learning_rate,
        min_value=min_value,
        max_value=max_value,
        update_frequency=update_frequency,
        window_size=window_size
    )


def create_constraint_calculator(
    soc_violation_weight: float = 1.0,
    action_violation_weight: float = 0.5,
    buffer_violation_weight: float = 0.3,
    grid_stability_weight: float = 0.2
) -> ConstraintCostCalculator:
    """
    """
    return ConstraintCostCalculator(
        soc_violation_weight=soc_violation_weight,
        action_violation_weight=action_violation_weight,
        buffer_violation_weight=buffer_violation_weight,
        grid_stability_weight=grid_stability_weight
    ) 
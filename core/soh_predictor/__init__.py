"""Battery state-of-health prediction helpers.

This package integrates Transformer-based SoH inference from the companion
SoH predictor project and exposes offline and online prediction utilities for
RL simulation and deployment.
"""

from .inference import SoHPredictor
from .online import OnlineSoHPredictor

__all__ = ["SoHPredictor", "OnlineSoHPredictor"]

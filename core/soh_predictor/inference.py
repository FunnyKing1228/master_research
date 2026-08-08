"""
SOH Predictor — Inference Module
=================================

  - Transformer Encoder (d_model=64, heads=4, layers=4)


"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =====================================================================
# Model Architecture (must match training)
# =====================================================================
ARCH = {
    "seq_len": 256,
    "d_model": 64,
    "d_ff": 128,
    "num_heads": 4,
    "num_layers": 4,
    "token_length": 64,
}

N_POINTS = 253  # resampled charge voltage curve points


class TransformerModel(nn.Module):
    """Transformer Encoder for SOH prediction."""

    def __init__(self, input_dim=1, seq_len=256, d_model=64, nhead=4,
                 num_layers=4, d_ff=128, token_length=64):
        super().__init__()
        self.d_model = d_model
        self.token_length = token_length
        self.input_proj = nn.Linear(seq_len * input_dim, d_model * token_length)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
                batch_first=True, norm_first=True
            ),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.squeeze(-1)
        x = self.input_proj(x)
        x = x.view(batch_size, self.d_model, self.token_length)
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.fc_out(x)
        return out.squeeze(-1)


# =====================================================================
# Lightweight Scaler (no pickle dependency)
# =====================================================================
class NpzScaler:
    """StandardScaler that loads from .npz file."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.mean_ = data["mean"]
        self.scale_ = data["scale"]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


# =====================================================================
# Feature Extraction
# =====================================================================
def extract_features_from_arrays(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
) -> Optional[np.ndarray]:
    """

    Features:

    Returns:
        np.ndarray of shape (256,) or None if extraction fails.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for SoH prediction. Install with: pip install scipy")

    try:
        t = np.asarray(time_s, dtype=float)
        v = np.asarray(voltage_v, dtype=float)
        c = np.asarray(current_a, dtype=float)

        if len(t) < 10:
            return None

        # --- Detect charge phase (voltage rising to peak) ---
        peak_idx = int(np.argmax(v))
        if peak_idx < 5:
            return None

        t_charge = t[:peak_idx + 1]
        v_charge = v[:peak_idx + 1]

        if len(t_charge) < 10:
            return None

        # --- Normalize voltage to [0, 1] ---
        v_min = v_charge.min()
        v_max = v_charge.max()
        if v_max - v_min < 0.01:
            return None
        v_norm = (v_charge - v_min) / (v_max - v_min)

        # --- Resample to N_POINTS ---
        t_rel = t_charge - t_charge[0]
        if t_rel[-1] <= 0:
            return None
        t_resample = np.linspace(0, t_rel[-1], N_POINTS)
        fn = interp1d(t_rel, v_norm, kind="linear", fill_value="extrapolate")
        v_resampled = fn(t_resample)

        if np.any(np.isnan(v_resampled)) or np.any(np.isinf(v_resampled)):
            return None

        # --- Current mode (binned) ---
        c_valid = c[c != 0]
        if len(c_valid) > 0:
            c_lo, c_hi = c_valid.min(), c_valid.max()
            if c_hi > c_lo:
                bins = np.linspace(c_lo, c_hi, 101)
                binned = np.clip(np.digitize(c_valid, bins), 0, 100)
                mode_idx = int(np.argmax(np.bincount(binned)))
                current_mode = (bins[max(0, mode_idx - 1)] + bins[min(100, mode_idx)]) / 2
            else:
                current_mode = float(c_lo)
        else:
            current_mode = 0.0

        features = np.append(v_resampled, [current_mode, v.min(), v.max()])
        assert features.shape == (256,), f"Feature dim mismatch: {features.shape}"
        return features

    except Exception:
        return None


def extract_features_from_csv(csv_path: str) -> Optional[np.ndarray]:
    """Documentation for this public API is provided in English."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required. Install with: pip install pandas")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"time", "voltage", "current"}
    if not required.issubset(df.columns):
        return None

    return extract_features_from_arrays(
        df["time"].values,
        df["voltage"].values,
        df["current"].values,
    )


# =====================================================================
# SoH Predictor Class
# =====================================================================
class SoHPredictor:
    """

        soh = predictor.predict_from_csv("cycle.csv")
        soh = predictor.predict_from_arrays(time, voltage, current)

    Parameters:
        device: "cpu" / "cuda" / "mps"
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        device: str = "cpu",
    ):
        _dir = os.path.dirname(os.path.abspath(__file__))
        _model_dir = os.path.join(_dir, "model")

        if model_path is None:
            model_path = os.path.join(_model_dir, "merged_slot_v8_transformer.pth")
        if scaler_path is None:
            npz_path = os.path.join(_model_dir, "scaler_params.npz")
            pkl_path = os.path.join(_model_dir, "merged_slot_v8_transformer_scaler.pkl")
            scaler_path = npz_path if os.path.exists(npz_path) else pkl_path

        # Device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # Load model
        self.model = TransformerModel(
            input_dim=1,
            seq_len=ARCH["seq_len"],
            d_model=ARCH["d_model"],
            nhead=ARCH["num_heads"],
            num_layers=ARCH["num_layers"],
            d_ff=ARCH["d_ff"],
            token_length=ARCH["token_length"],
        )
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        # Load scaler
        if scaler_path.endswith(".npz"):
            self.scaler = NpzScaler(scaler_path)
        else:
            import joblib
            self.scaler = joblib.load(scaler_path)

        self._loaded = True

    def predict_from_arrays(
        self,
        time_s: np.ndarray,
        voltage_v: np.ndarray,
        current_a: np.ndarray,
    ) -> Optional[float]:
        """

        Args:

        Returns:
        """
        features = extract_features_from_arrays(time_s, voltage_v, current_a)
        if features is None:
            return None
        return self._predict(features)

    def predict_from_csv(self, csv_path: str) -> Optional[float]:
        """


        Returns:
        """
        features = extract_features_from_csv(csv_path)
        if features is None:
            return None
        return self._predict(features)

    def _predict(self, features: np.ndarray) -> float:
        """Documentation for this public API is provided in English."""
        x = self.scaler.transform(features.reshape(1, -1))
        x = np.nan_to_num(x)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            soh = self.model(x_tensor).item()
        return float(np.clip(soh, 0.0, 1.0))

    def is_loaded(self) -> bool:
        """Documentation for this public API is provided in English."""
        return self._loaded

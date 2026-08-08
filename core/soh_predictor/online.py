from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .inference import SoHPredictor


class OnlineSoHPredictor:
    """Online SoH predictor for naturally observed deployment segments."""

    PREDICTION_HEADER = [
        "timestamp",
        "file",
        "soh",
        "status",
        "method",
        "capacity_mah",
        "capacity_wh",
        "model_path",
        "samples",
        "duration_sec",
        "voltage_min",
        "voltage_max",
        "current_a_mean",
        "message",
    ]

    def __init__(
        self,
        model_path: str,
        log_dir: str,
        nominal_capacity_mah: float,
        nominal_capacity_wh: float,
        *,
        capacity_min_soh: float = 0.50,
        capacity_max_soh: float = 1.05,
        device: str = "cpu",
    ):
        self.model_path = str(model_path or "")
        self.log_dir = Path(log_dir)
        self.nominal_capacity_mah = float(nominal_capacity_mah)
        self.nominal_capacity_wh = float(nominal_capacity_wh)
        self.capacity_min_soh = float(capacity_min_soh)
        self.capacity_max_soh = float(capacity_max_soh)
        self.device = device

        self.out_dir = self.log_dir / "soh_online_segments"
        self.cycles_dir = self.out_dir / "cycles"
        self.cycles_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_csv = self.out_dir / "soh_online_predictions.csv"
        self._ensure_prediction_header()

        self.predictor: Optional[SoHPredictor] = None
        self.load_message = ""
        self._load_predictor()

    def predict_from_readings(
        self,
        readings: Iterable[Any],
        *,
        step: int,
        timestamp: datetime,
    ) -> dict[str, Any]:
        rows = [r for r in readings if getattr(r, "batt_v", 0.0) > 0]
        if len(rows) < 10:
            return self._result(None, "SKIP", "INSUFFICIENT", "", "too few samples", 0, 0, 0, 0, 0)

        segment_path, stats, arrays = self._write_segment(rows, step, timestamp)
        soh = None
        method = "MODEL_PREDICT"
        message = self.load_message

        if self.predictor is not None:
            try:
                soh = self.predictor.predict_from_arrays(*arrays)
            except Exception as exc:
                message = f"model inference failed: {exc}"

        if soh is None:
            soh = self._fallback_estimate(stats)
            method = "ONLINE_FALLBACK"
            message = message or "model not loaded; used voltage/current fallback"

        soh = float(np.clip(soh, 0.0, 1.5))
        capacity_soh = float(np.clip(soh, self.capacity_min_soh, self.capacity_max_soh))
        result = self._result(
            soh,
            "OK",
            method,
            segment_path.name,
            message,
            stats["samples"],
            stats["duration_sec"],
            stats["voltage_min"],
            stats["voltage_max"],
            stats["current_a_mean"],
            timestamp=timestamp,
            capacity_soh=capacity_soh,
        )
        self._append_prediction(result)
        return result

    def _load_predictor(self) -> None:
        model_file, scaler_file = self._resolve_model_files()
        if not model_file:
            self.load_message = "No .pth SoH model found under selected path"
            return
        try:
            self.predictor = SoHPredictor(
                model_path=str(model_file),
                scaler_path=str(scaler_file) if scaler_file else None,
                device=self.device,
            )
            self.load_message = f"loaded model={model_file.name}"
        except Exception as exc:
            self.predictor = None
            self.load_message = f"failed to load SoH model: {exc}"

    def _resolve_model_files(self) -> tuple[Optional[Path], Optional[Path]]:
        root = Path(self.model_path)
        search_root = root if root.is_dir() else root.parent
        model_file = root if root.is_file() and root.suffix.lower() == ".pth" else None
        if model_file is None and search_root.exists():
            models = sorted(search_root.rglob("*.pth"))
            model_file = models[0] if models else None

        scaler_file = None
        if search_root.exists():
            scalers = sorted(search_root.rglob("scaler_params.npz"))
            scalers += sorted(search_root.rglob("*scaler*.pkl"))
            scaler_file = scalers[0] if scalers else None
        return model_file, scaler_file

    def _write_segment(
        self,
        rows: list[Any],
        step: int,
        timestamp: datetime,
    ) -> tuple[Path, dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        first_ts = getattr(rows[0], "timestamp", timestamp)
        name = f"cycle_soh_online_{timestamp:%Y%m%d_%H%M%S}_{step:04d}.csv"
        path = self.cycles_dir / name

        time_s = []
        voltage_v = []
        current_a = []
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "voltage", "current"])
            for r in rows:
                ts = getattr(r, "timestamp", first_ts)
                t_sec = max(0.0, (ts - first_ts).total_seconds())
                voltage = float(getattr(r, "batt_v", 0.0))
                current = float(getattr(r, "batt_i_ma", 0.0)) / 1000.0
                writer.writerow([f"{t_sec:.3f}", f"{voltage:.6f}", f"{current:.6f}"])
                time_s.append(t_sec)
                voltage_v.append(voltage)
                current_a.append(current)

        stats = {
            "samples": float(len(rows)),
            "duration_sec": float(max(time_s) if time_s else 0.0),
            "voltage_min": float(min(voltage_v) if voltage_v else 0.0),
            "voltage_max": float(max(voltage_v) if voltage_v else 0.0),
            "current_a_mean": float(np.mean(current_a)) if current_a else 0.0,
        }
        arrays = (
            np.asarray(time_s, dtype=float),
            np.asarray(voltage_v, dtype=float),
            np.asarray(current_a, dtype=float),
        )
        return path, stats, arrays

    def _fallback_estimate(self, stats: dict[str, float]) -> float:
        v_mean = 0.5 * (stats["voltage_min"] + stats["voltage_max"])
        v_score = float(np.clip((v_mean - 5.6) / 1.8, 0.0, 1.0))
        current_score = float(np.clip(stats["current_a_mean"] / 0.20, 0.0, 1.0))
        sag_penalty = float(np.clip((stats["voltage_max"] - stats["voltage_min"]) / 1.5, 0.0, 0.5))
        return float(np.clip(0.45 + 0.45 * v_score + 0.10 * current_score - 0.20 * sag_penalty, 0.30, 1.20))

    def _ensure_prediction_header(self) -> None:
        if self.prediction_csv.exists() and self.prediction_csv.stat().st_size > 0:
            return
        with self.prediction_csv.open("w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=self.PREDICTION_HEADER).writeheader()

    def _append_prediction(self, result: dict[str, Any]) -> None:
        self._ensure_prediction_header()
        with self.prediction_csv.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.PREDICTION_HEADER)
            writer.writerow({k: result.get(k, "") for k in self.PREDICTION_HEADER})

    def _result(
        self,
        soh: Optional[float],
        status: str,
        method: str,
        file_name: str,
        message: str,
        samples: float,
        duration_sec: float,
        voltage_min: float,
        voltage_max: float,
        current_a_mean: float,
        *,
        timestamp: Optional[datetime] = None,
        capacity_soh: Optional[float] = None,
    ) -> dict[str, Any]:
        timestamp = timestamp or datetime.now()
        capacity_soh = float(capacity_soh if capacity_soh is not None else (soh or 1.0))
        return {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "file": file_name,
            "soh": "" if soh is None else f"{float(soh):.6f}",
            "status": status,
            "method": method,
            "capacity_mah": f"{self.nominal_capacity_mah * capacity_soh:.2f}",
            "capacity_wh": f"{self.nominal_capacity_wh * capacity_soh:.3f}",
            "model_path": self.model_path,
            "samples": int(samples),
            "duration_sec": f"{duration_sec:.1f}",
            "voltage_min": f"{voltage_min:.3f}",
            "voltage_max": f"{voltage_max:.3f}",
            "current_a_mean": f"{current_a_mean:.4f}",
            "message": message,
        }

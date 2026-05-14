"""CSI window → activity class inference (numpy-only runtime).

Weights are produced by trainer.py and stored as .npz. Architecture:
  flatten(WINDOW_SIZE × num_sc) → Linear(256) → ReLU
                                → Linear(128) → ReLU
                                → Linear(num_classes)   (logits)

Runtime applies the same per-subcarrier z-score (mu, sigma) saved at
training time, then softmax over the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .activity_classes import CLASS_NAMES, NUM_CLASSES, class_name

logger = logging.getLogger(__name__)

WINDOW_SIZE: int = 32


@dataclass
class ActivityPrediction:
    """One classifier output."""
    class_idx:   int
    class_name:  str
    confidence:  float          # softmax probability of the winning class [0, 1]
    probs:       np.ndarray     # full per-class probabilities (num_classes,)


class ActivityClassifier:
    """Loads trained weights and runs softmax inference on a CSI window."""

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        self._w1: Optional[np.ndarray] = None
        self._b1: Optional[np.ndarray] = None
        self._w2: Optional[np.ndarray] = None
        self._b2: Optional[np.ndarray] = None
        self._w3: Optional[np.ndarray] = None
        self._b3: Optional[np.ndarray] = None
        self._mu:    Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._num_sc: int = 0
        self._num_classes: int = NUM_CLASSES
        self._class_names: List[str] = list(CLASS_NAMES)
        self._loaded: bool = False
        self._weights_path = weights_path

        if weights_path is not None and Path(weights_path).exists():
            self.load(weights_path)

    # ------------------------------------------------------------------

    def load(self, path: Path) -> bool:
        try:
            data = np.load(str(path), allow_pickle=True)
            self._w1 = data["w1"]; self._b1 = data["b1"]
            self._w2 = data["w2"]; self._b2 = data["b2"]
            self._w3 = data["w3"]; self._b3 = data["b3"]
            self._num_sc = int(data["num_sc"])
            self._num_classes = int(data["num_classes"]) if "num_classes" in data.files \
                                else int(self._w3.shape[0])
            self._class_names = [str(n) for n in data["class_names"]] \
                                if "class_names" in data.files else list(CLASS_NAMES)

            # Old pose-model weights don't carry mu/sigma — reject them so we
            # don't silently run the wrong normalisation on a classifier path.
            if "mu" not in data.files or "sigma" not in data.files:
                logger.warning(
                    "Weights at %s lack mu/sigma — looks like the old pose "
                    "model. Retrain with the activity classifier.", path,
                )
                self._loaded = False
                return False

            self._mu    = data["mu"].astype(np.float32)
            self._sigma = data["sigma"].astype(np.float32) + 1e-6
            self._loaded = True
            logger.info(
                "ActivityClassifier loaded from %s  (num_sc=%d, num_classes=%d)",
                path, self._num_sc, self._num_classes,
            )
            return True
        except Exception as exc:
            logger.warning("ActivityClassifier load failed: %s", exc)
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def weights_path(self) -> Optional[Path]:
        return self._weights_path

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

    # ------------------------------------------------------------------

    def predict(self, frames: List[List[float]]) -> Optional[ActivityPrediction]:
        if not self._loaded or len(frames) < WINDOW_SIZE:
            return None
        try:
            arr = self._normalise_window(frames[-WINDOW_SIZE:])
            if arr is None:
                return None
            logits = self._forward(arr.flatten())
            probs = self._softmax(logits)
            idx = int(np.argmax(probs))
            return ActivityPrediction(
                class_idx=idx,
                class_name=self._class_names[idx] if idx < len(self._class_names) else class_name(idx),
                confidence=float(probs[idx]),
                probs=probs,
            )
        except Exception as exc:
            logger.debug("predict error: %s", exc)
            return None

    # ------------------------------------------------------------------

    def _normalise_window(self, window: List[List[float]]) -> Optional[np.ndarray]:
        nc = self._num_sc
        arr = np.zeros((WINDOW_SIZE, nc), dtype=np.float32)
        for i, frame in enumerate(window):
            n = min(len(frame), nc)
            if n:
                arr[i, :n] = frame[:n]
        return (arr - self._mu) / self._sigma

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max()
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h1 = self._relu(self._w1 @ x + self._b1)
        h2 = self._relu(self._w2 @ h1 + self._b2)
        return self._w3 @ h2 + self._b3

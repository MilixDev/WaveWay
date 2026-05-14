"""CSI sample storage for activity classification.

One class label per session — fixed at construction time and replicated
across every sample. The collector is a passive accumulator; the recording
thread owns timing and serial reads.

Output files per session:
  <dir>/<session>_csi.npy     float32 (N, num_sc)
  <dir>/<session>_labels.npy  int64   (N,)
  <dir>/<session>_ts.npy      float64 (N,)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .activity_classes import class_name

logger = logging.getLogger(__name__)


class DataCollector:
    """Accumulates (CSI amplitudes, class label) pairs for one session."""

    def __init__(
        self, output_dir: Path, session_name: str, class_idx: int
    ) -> None:
        self._out_dir = Path(output_dir)
        self._session = session_name
        self._class_idx = int(class_idx)
        self._csi_rows: List[np.ndarray] = []
        self._ts_rows:  List[float]      = []

    @property
    def n_samples(self) -> int:
        return len(self._ts_rows)

    @property
    def class_idx(self) -> int:
        return self._class_idx

    @property
    def class_name(self) -> str:
        return class_name(self._class_idx)

    def add_sample(self, csi_amplitudes: List[float], timestamp: float) -> bool:
        if not csi_amplitudes:
            return False
        self._csi_rows.append(np.array(csi_amplitudes, dtype=np.float32))
        self._ts_rows.append(timestamp)
        return True

    def save(self) -> Optional[Path]:
        if not self._ts_rows:
            logger.warning("DataCollector: no samples — nothing saved")
            return None

        self._out_dir.mkdir(parents=True, exist_ok=True)

        # Rows can have different subcarrier counts if the ESP32 hiccuped
        # mid-session — pad to the longest length with zeros.
        max_sc = max(len(r) for r in self._csi_rows)
        csi_mat = np.zeros((len(self._csi_rows), max_sc), dtype=np.float32)
        for i, row in enumerate(self._csi_rows):
            csi_mat[i, : len(row)] = row

        labels = np.full(len(self._csi_rows), self._class_idx, dtype=np.int64)
        ts_arr = np.array(self._ts_rows, dtype=np.float64)

        np.save(str(self._out_dir / f"{self._session}_csi.npy"),    csi_mat)
        np.save(str(self._out_dir / f"{self._session}_labels.npy"), labels)
        np.save(str(self._out_dir / f"{self._session}_ts.npy"),     ts_arr)

        logger.info(
            "DataCollector saved %d samples (class=%s, idx=%d) → %s",
            len(self._ts_rows), self.class_name, self._class_idx, self._out_dir,
        )
        return self._out_dir

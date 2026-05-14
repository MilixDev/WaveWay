from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QRectF, QRect, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from core.csi_reader import Frame

# -----------------------------------------------------------------------
# Colormap: dark navy → blue → cyan → yellow-green → red
# -----------------------------------------------------------------------

def _build_cmap() -> np.ndarray:
    """256-entry RGB colormap from low (dark) to high (bright red)."""
    pts = [
        (0.00, 13,  17,  23),
        (0.12, 10,  30,  90),
        (0.30, 0,   80,  200),
        (0.50, 0,   190, 190),
        (0.70, 50,  210, 50),
        (0.85, 230, 210, 0),
        (1.00, 248, 40,  20),
    ]
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for j in range(len(pts) - 1):
            t0, r0, g0, b0 = pts[j]
            t1, r1, g1, b1 = pts[j + 1]
            if t0 <= t <= t1:
                v = (t - t0) / (t1 - t0)
                cmap[i] = [
                    int(r0 + v * (r1 - r0)),
                    int(g0 + v * (g1 - g0)),
                    int(b0 + v * (b1 - b0)),
                ]
                break
    return cmap


_CMAP = _build_cmap()
_NUM_COLS = 220   # time columns shown


class HeatmapView(QWidget):
    """Waterfall CSI heatmap.

    X axis → time (newest frame on the right).
    Y axis → subcarrier index (0 at top).
    Color  → normalised amplitude via a blue-to-red colormap.

    Each column is per-frame normalised so that subcarrier patterns are
    visible regardless of overall signal strength changes.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frames: List[List[float]] = []
        self._img: Optional[QImage] = None
        self.setMinimumHeight(160)

    def update_data(self, frames: List[Frame]) -> None:
        """Accept a new snapshot of buffered frames and schedule repaint."""
        self._frames = [f[1] for f in frames[-_NUM_COLS:]]
        self._img = None  # invalidate cached image
        self.update()

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, w, h, QColor("#161b22"))

        p.setPen(QPen(QColor("#e6edf3")))
        p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        p.drawText(14, 22, "MAPA DE CALOR CSI  ·  Subportadora × Tiempo")

        if not self._frames:
            p.setPen(QPen(QColor("#8b949e")))
            p.setFont(QFont("Consolas", 10))
            p.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "Sin señal")
            p.end()
            return

        px, py = 14, 30
        pw, ph = w - 28, h - py - 20

        if pw < 4 or ph < 4:
            p.end()
            return

        if self._img is None:
            self._img = self._build_image()

        if self._img is not None:
            p.drawImage(QRectF(px, py, pw, ph), self._img)

        # Axis labels
        p.setPen(QPen(QColor("#8b949e")))
        p.setFont(QFont("Consolas", 8))
        p.drawText(px, py + ph + 14, "← Más antiguo")
        p.drawText(
            QRect(px, py + ph + 4, pw, 14),
            Qt.AlignmentFlag.AlignRight,
            "Más reciente →",
        )
        p.end()

    # ------------------------------------------------------------------

    def _build_image(self) -> Optional[QImage]:
        frames = self._frames
        if not frames:
            return None

        # Determine consistent subcarrier count (mode of lengths)
        lengths = [len(f) for f in frames if f]
        if not lengths:
            return None
        num_sc = int(np.bincount(lengths).argmax())
        if num_sc == 0:
            return None

        num_frames = len(frames)

        # Build float matrix (num_sc rows × num_frames cols)
        arr = np.zeros((num_sc, num_frames), dtype=np.float32)
        for j, frame in enumerate(frames):
            n = min(len(frame), num_sc)
            if n:
                arr[:n, j] = frame[:n]

        # Per-column (per-frame) normalisation → emphasises subcarrier patterns
        col_min = arr.min(axis=0, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        denom = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
        norm = (arr - col_min) / denom   # shape (num_sc, num_frames), float in [0, 1]

        # Flip so subcarrier 0 is at the top visually
        norm = np.flipud(norm)

        # Apply colormap via lookup table
        idx = np.clip((norm * 255).astype(np.uint8), 0, 255)
        rgb = _CMAP[idx]                  # shape (num_sc, num_frames, 3)
        rgb = np.ascontiguousarray(rgb)

        height, width = rgb.shape[0], rgb.shape[1]
        img = QImage(
            rgb.tobytes(),
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        )
        return img.copy()  # detach from the numpy buffer

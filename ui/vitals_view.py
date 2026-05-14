import math
from collections import deque
from typing import Deque, List, Optional

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from core.models import Detection

_BG = QColor("#161b22")
_BORDER = QColor("#30363d")
_ACCENT = QColor("#58a6ff")
_PINK = QColor("#f778ba")
_GREEN = QColor("#3fb950")
_YELLOW = QColor("#d29922")
_TEXT = QColor("#e6edf3")
_TEXT_SEC = QColor("#8b949e")

_WAVE_HISTORY = 200
_RATE_HISTORY = 30   # samples used for stability calculation


class BreathingView(QWidget):
    """Vitals display: breathing rate (large), heart rate (optional secondary), waveform, stability.

    Heart rate is shown only when Detection.heart_rate > 0 (i.e. the user has
    enabled the experimental toggle and the estimator has a confident peak).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._detection: Optional[Detection] = None
        self._wave: Deque[float] = deque([0.0] * _WAVE_HISTORY, maxlen=_WAVE_HISTORY)
        self._rate_history: Deque[float] = deque(maxlen=_RATE_HISTORY)
        self._phi: float = 0.0
        self.setMinimumSize(300, 360)

    def update_detection(self, detection: Detection) -> None:
        self._detection = detection
        active = detection.present and detection.breathing_rate > 0

        if active:
            br_hz = detection.breathing_rate / 60.0
            self._phi = (self._phi + br_hz * 0.1 * 2 * math.pi) % (2 * math.pi)
            self._wave.append(math.sin(self._phi))
            self._rate_history.append(detection.breathing_rate)
        else:
            self._wave.append(0.0)

        self.update()

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, w, h, _BG)

        d = self._detection
        breath_active = d is not None and d.present and d.breathing_rate > 0
        heart_active = d is not None and d.present and d.heart_rate > 0

        # Header
        p.setPen(QPen(_TEXT))
        p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header = "RESPIRACIÓN" + ("  ·  LATIDO (exp)" if heart_active else "")
        p.drawText(14, 22, header)

        pad = 14
        # ---- Breathing numeric ----
        br = d.breathing_rate if breath_active else 0.0
        if breath_active:
            p.setPen(QPen(_ACCENT))
            p.setFont(QFont("Consolas", 40, QFont.Weight.Bold))
            p.drawText(pad, 74, f"{br:.0f}")
            p.setPen(QPen(_TEXT_SEC))
            p.setFont(QFont("Segoe UI", 13))
            p.drawText(pad + 80, 72, "rpm")
            p.setFont(QFont("Consolas", 9))
            normal = "normal" if 12 <= br <= 20 else ("lento" if br < 12 else "rápido")
            conf_pct = int(d.breathing_confidence * 100)
            p.drawText(pad + 80, 90, f"({normal})  conf {conf_pct}%")
        else:
            p.setPen(QPen(_TEXT_SEC))
            p.setFont(QFont("Consolas", 32, QFont.Weight.Bold))
            p.drawText(pad, 74, "—")
            p.setFont(QFont("Consolas", 9))
            p.drawText(pad + 28, 68, "Sin señal")

        # ---- Heart numeric (secondary line; only when active) ----
        heart_line_y = 112
        if heart_active:
            hr = d.heart_rate
            p.setPen(QPen(_PINK))
            p.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
            p.drawText(pad, heart_line_y, f"♥ {hr:.0f}")
            p.setPen(QPen(_TEXT_SEC))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(pad + 70, heart_line_y - 2, "bpm")
            p.setFont(QFont("Consolas", 8))
            hr_conf_pct = int(d.heart_confidence * 100)
            p.drawText(pad + 70, heart_line_y + 12, f"exp · conf {hr_conf_pct}%")

        # ---- Waveform ----
        wx = pad
        wy = heart_line_y + 8 if heart_active else 100
        ww = w - 2 * pad
        wh = h - wy - 56
        if ww > 4 and wh > 4:
            self._draw_waveform(p, wx, wy, ww, wh, breath_active)

        # ---- Stability bar ----
        self._draw_stability(p, pad, h - 46, w - 2 * pad, 36)
        p.end()

    # ------------------------------------------------------------------

    def _draw_waveform(
        self, p: QPainter, x: int, y: int, w: int, h: int, active: bool
    ) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(13, 17, 23)))
        p.drawRect(x, y, w, h)
        p.setPen(QPen(QColor(_BORDER.red(), _BORDER.green(), _BORDER.blue(), 70), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(x, y, w, h)

        if not active:
            p.setPen(QPen(_TEXT_SEC))
            p.setFont(QFont("Consolas", 9))
            p.drawText(
                QRect(x, y, w, h),
                Qt.AlignmentFlag.AlignCenter,
                "Sin señal",
            )
            return

        samples = list(self._wave)
        n = len(samples)
        if n < 2:
            return

        cy = y + h // 2
        scale = h / 2.4

        # Baseline
        p.setPen(
            QPen(QColor(_BORDER.red(), _BORDER.green(), _BORDER.blue(), 90), 1, Qt.PenStyle.DashLine)
        )
        p.drawLine(x, cy, x + w, cy)

        # Wave path
        path = QPainterPath()
        path.moveTo(x, cy - samples[0] * scale)
        for i in range(1, n):
            xi = x + (i / (n - 1)) * w
            yi = cy - samples[i] * scale
            path.lineTo(xi, yi)

        p.setPen(QPen(_ACCENT, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Label
        p.setPen(QPen(_TEXT_SEC))
        p.setFont(QFont("Consolas", 8))
        p.drawText(x + 4, y + 13, "onda a frecuencia detectada")

    def _draw_stability(
        self, p: QPainter, x: int, y: int, w: int, h: int
    ) -> None:
        """Stability bar based on std-dev of recent rate estimates."""
        p.setPen(QPen(_TEXT_SEC))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(x, y + 13, "Estabilidad de la medición")

        rates = list(self._rate_history)
        bar_y = y + 18
        bar_h = 10

        # Track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(30, 38, 50)))
        p.drawRect(x, bar_y, w, bar_h)

        if len(rates) >= 4:
            std = float(np.std(rates))
            # std < 1 rpm → perfect, std >= 6 rpm → 0
            stability = float(np.clip(1.0 - std / 6.0, 0.0, 1.0))
            fill_w = max(4, int(stability * w))

            if stability > 0.7:
                col = _GREEN
                lbl = "buena"
            elif stability > 0.4:
                col = _YELLOW
                lbl = "regular"
            else:
                col = QColor("#f85149")
                lbl = "inestable"

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawRect(x, bar_y, fill_w, bar_h)

            p.setPen(QPen(_TEXT_SEC))
            p.setFont(QFont("Consolas", 9))
            p.drawText(x + fill_w + 6, bar_y + bar_h - 1, lbl)
        else:
            p.setPen(QPen(_TEXT_SEC))
            p.setFont(QFont("Consolas", 9))
            p.drawText(x + 4, bar_y + bar_h - 1, "acumulando datos…")

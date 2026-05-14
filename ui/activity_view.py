"""Activity panel — schematic silhouette + class label over the CSI spectrum.

Replaces the COCO-skeleton panel from earlier versions. The silhouette
switches with the predicted class (vacío / parado / sentado / caminando
/ tirado), breathes with the detected respiration rate, and pulses on
the chest when heart rate is available.

Without a trained model, falls back to a "vacío" silhouette when no
presence is detected and a placeholder otherwise.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from core.models import Detection
from core.training.activity_classes import CLASS_NAMES
from core.training.model import ActivityPrediction

_BG = QColor("#0d1117")
_PANEL = QColor("#161b22")
_TEXT = QColor("#e6edf3")
_TEXT_SEC = QColor("#8b949e")

# One accent per class — visually distinct on the dark background.
_CLASS_COLOR: List[QColor] = [
    QColor(139, 148, 158),   # vacío
    QColor(88, 166, 255),    # parado
    QColor(63, 185, 80),     # sentado
    QColor(210, 153, 34),    # caminando
    QColor(248, 81, 73),     # tirado
]


def _clamp(v: float) -> int:
    return max(0, min(255, int(v)))


class ActivityView(QWidget):
    """Silhouette + class label driven by Detection + ActivityPrediction."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._detection: Optional[Detection] = None
        self._prediction: Optional[ActivityPrediction] = None
        self._model_loaded: bool = False
        self.setMinimumSize(240, 420)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)   # ~30 fps so breathing/walking animations are smooth

    def update_detection(
        self,
        detection: Detection,
        prediction: Optional[ActivityPrediction] = None,
        model_loaded: bool = False,
    ) -> None:
        self._detection = detection
        self._prediction = prediction
        self._model_loaded = model_loaded

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return

        now = time.time()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, w, h, _BG)

        p.setPen(QPen(_TEXT))
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(12, 22, "ESPECTRO  ·  ACTIVIDAD")

        d = self._detection
        present = d is not None and d.present

        # Class selection: ML prediction first, then "vacío" if nobody is here.
        cls_idx: Optional[int] = None
        cls_conf: float = 0.0
        if self._prediction is not None and present:
            cls_idx = self._prediction.class_idx
            cls_conf = self._prediction.confidence
        elif not present:
            cls_idx, cls_conf = 0, 1.0

        if cls_idx is not None:
            base = _CLASS_COLOR[cls_idx]
            fade = 0.4 + 0.6 * max(0.0, min(1.0, cls_conf))
            cr = _clamp(base.red()   * fade + _TEXT_SEC.red()   * (1 - fade))
            cg = _clamp(base.green() * fade + _TEXT_SEC.green() * (1 - fade))
            cb = _clamp(base.blue()  * fade + _TEXT_SEC.blue()  * (1 - fade))
        else:
            cr, cg, cb = _TEXT_SEC.red(), _TEXT_SEC.green(), _TEXT_SEC.blue()

        pad = 16
        stage_x, stage_y = pad, 40
        stage_w = w - 2 * pad
        stage_h = h - stage_y - 110   # leaves room for label + bar + footer

        # ---- Vital animations ----
        breath_dy = 0.0
        if present and d.breathing_rate > 1.0 and d.breathing_confidence > 0.0:
            phase = 2.0 * math.pi * (d.breathing_rate / 60.0) * now
            breath_dy = -math.sin(phase) * 0.018 * min(d.breathing_confidence, 1.0)

        heart_pulse = 0.0
        if present and d.heart_rate > 20.0 and d.heart_confidence > 0.0:
            phase_01 = (now * (d.heart_rate / 60.0)) % 1.0
            if phase_01 < 0.20:
                heart_pulse = (1.0 - phase_01 / 0.20) * d.heart_confidence

        tilt_x = int(d.lateral_offset * stage_w * 0.10) if present else 0
        scale  = (1.0 - d.distance_ratio * 0.12) if present else 1.0

        # ---- Silhouette ----
        if cls_idx is None:
            self._draw_placeholder(p, stage_x, stage_y, stage_w, stage_h)
        else:
            self._draw_silhouette(
                p, cls_idx, cr, cg, cb,
                stage_x, stage_y, stage_w, stage_h,
                breath_dy, heart_pulse, tilt_x, scale, now,
            )

        # ---- Class label ----
        label_y = stage_y + stage_h + 14
        p.setPen(QPen(QColor(cr, cg, cb)))
        p.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        if cls_idx is not None:
            label = CLASS_NAMES[cls_idx].upper()
        elif not self._model_loaded:
            label = "SIN MODELO"
        else:
            label = "—"
        p.drawText(QRect(0, label_y, w, 32), Qt.AlignmentFlag.AlignCenter, label)

        # ---- Confidence bar ----
        bar_y, bar_h = label_y + 38, 8
        bar_pad = 36
        bar_w = w - 2 * bar_pad
        if bar_w > 20 and cls_idx is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(30, 38, 50)))
            p.drawRect(bar_pad, bar_y, bar_w, bar_h)
            fill_w = int(max(0.0, min(1.0, cls_conf)) * bar_w)
            p.setBrush(QBrush(QColor(cr, cg, cb)))
            p.drawRect(bar_pad, bar_y, fill_w, bar_h)
            p.setPen(QPen(_TEXT_SEC))
            p.setFont(QFont("Consolas", 9))
            p.drawText(bar_pad, bar_y + bar_h + 14,
                       f"confianza  {int(cls_conf * 100):3d}%")

        # ---- Footer ----
        footer: List[str] = []
        if present:
            if d.breathing_rate > 1.0:
                footer.append(f"~{d.breathing_rate:.0f} rpm  "
                              f"{int(d.breathing_confidence * 100)}%")
            if d.heart_rate > 20.0:
                footer.append(f"♥ {d.heart_rate:.0f} bpm  "
                              f"{int(d.heart_confidence * 100)}%")
        footer.append("[ML]" if (self._model_loaded and self._prediction) else "[base]")

        p.setFont(QFont("Consolas", 9))
        p.setPen(QPen(_TEXT_SEC))
        p.drawText(QRect(0, h - 22, w, 20),
                   Qt.AlignmentFlag.AlignCenter,
                   "  ".join(footer))
        p.end()

    # ------------------------------------------------------------------
    # Silhouette drawing
    # ------------------------------------------------------------------

    def _draw_placeholder(self, p: QPainter, sx: int, sy: int, sw: int, sh: int) -> None:
        p.setPen(QPen(_TEXT_SEC, 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(sx + sw * 0.15, sy + sh * 0.15, sw * 0.7, sh * 0.7))
        p.setFont(QFont("Consolas", 10))
        p.setPen(QPen(_TEXT_SEC))
        p.drawText(QRectF(sx, sy, sw, sh), Qt.AlignmentFlag.AlignCenter,
                   "Entrena un modelo para\nclasificar la actividad")

    def _draw_silhouette(
        self, p: QPainter, cls_idx: int,
        cr: int, cg: int, cb: int,
        sx: int, sy: int, sw: int, sh: int,
        breath_dy: float, heart_pulse: float,
        tilt_x: int, scale: float, now: float,
    ) -> None:
        cx = sx + sw / 2 + tilt_x

        def npos(nx: float, ny: float) -> QPointF:
            """Map a (nx, ny) in [0,1]² to the stage in pixels."""
            px = cx + (nx - 0.5) * sw * scale
            py = sy + ny * sh * scale + sh * (1 - scale) * 0.5
            return QPointF(px, py)

        col = QColor(cr, cg, cb)
        alpha_main = 220

        # 3-layer "glow + bright core" line — gives the silhouette the vital
        # look without us having to compute proper Gaussian blur.
        def glow_line(a: QPointF, b: QPointF, w_main: float = 3.0) -> None:
            for lw, la in ((w_main * 3.0, 35), (w_main * 1.7, 90), (w_main, alpha_main)):
                p.setPen(QPen(QColor(cr, cg, cb, la), lw))
                p.drawLine(a, b)

        def head_circle(pt: QPointF, r: float) -> None:
            p.setPen(QPen(QColor(cr, cg, cb, alpha_main), 2))
            p.setBrush(QBrush(QColor(cr, cg, cb, 50)))
            p.drawEllipse(pt, r, r)

        def heart_glow(pt: QPointF, r: float) -> None:
            if heart_pulse <= 0.05:
                return
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(255, 90, 140, int(heart_pulse * 110))))
            p.drawEllipse(pt, r * (1 + heart_pulse * 0.6), r * (1 + heart_pulse * 0.6))

        if cls_idx == 0:
            self._draw_empty(p, npos, sw, col, now)
        elif cls_idx == 1:
            self._draw_standing(npos, sw, breath_dy, glow_line, head_circle, heart_glow)
        elif cls_idx == 2:
            self._draw_sitting(p, npos, sw, breath_dy, col, glow_line, head_circle, heart_glow)
        elif cls_idx == 3:
            self._draw_walking(p, npos, sw, breath_dy, col, now, glow_line, head_circle, heart_glow)
        elif cls_idx == 4:
            self._draw_lying(p, npos, sw, breath_dy, col, glow_line, head_circle, heart_glow)

    # -- per-class silhouettes -------------------------------------------------

    def _draw_empty(self, p, npos, sw, col, now):
        # Room outline + animated wifi arcs in the centre.
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 120),
                      1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        top_left = npos(0.20, 0.30)
        bot_right = npos(0.80, 0.90)
        p.drawRect(QRectF(top_left.x(), top_left.y(),
                          bot_right.x() - top_left.x(),
                          bot_right.y() - top_left.y()))

        center = npos(0.50, 0.60)
        cx, cy = center.x(), center.y()
        t = (now % 2.0) / 2.0
        for i, r_norm in enumerate((0.07, 0.13, 0.20)):
            alpha = int(140 * (1 - t) if (now * 0.5 + i * 0.33) % 1.0 < 0.7 else 60)
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), alpha), 2))
            p.drawArc(QRectF(cx - sw * r_norm, cy - sw * r_norm,
                             sw * 2 * r_norm, sw * 2 * r_norm),
                      45 * 16, 90 * 16)

    def _draw_standing(self, npos, sw, breath_dy, glow_line, head_circle, heart_glow):
        dy = breath_dy * 0.6
        head = npos(0.50, 0.18 + dy)
        head_circle(head, sw * 0.075)

        glow_line(npos(0.50, 0.27 + dy), npos(0.50, 0.58))         # spine
        glow_line(npos(0.42, 0.30 + dy), npos(0.58, 0.30 + dy))    # shoulders
        glow_line(npos(0.45, 0.58),       npos(0.55, 0.58))         # hips
        glow_line(npos(0.42, 0.30 + dy), npos(0.38, 0.46 + dy))    # arms
        glow_line(npos(0.38, 0.46 + dy), npos(0.36, 0.62))
        glow_line(npos(0.58, 0.30 + dy), npos(0.62, 0.46 + dy))
        glow_line(npos(0.62, 0.46 + dy), npos(0.64, 0.62))
        glow_line(npos(0.45, 0.58), npos(0.44, 0.78))               # legs
        glow_line(npos(0.44, 0.78), npos(0.44, 0.96))
        glow_line(npos(0.55, 0.58), npos(0.56, 0.78))
        glow_line(npos(0.56, 0.78), npos(0.56, 0.96))

        heart_glow(npos(0.50, 0.40 + breath_dy * 0.5), sw * 0.05)

    def _draw_sitting(self, p, npos, sw, breath_dy, col, glow_line, head_circle, heart_glow):
        dy = breath_dy * 0.6

        # Chair hint — dashed rectangle behind the figure.
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 110),
                      1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        ct = npos(0.36, 0.68)
        cb_ = npos(0.62, 0.95)
        p.drawRect(QRectF(ct.x(), ct.y(), cb_.x() - ct.x(), cb_.y() - ct.y()))

        head_circle(npos(0.45, 0.30 + dy), sw * 0.07)

        glow_line(npos(0.46, 0.38 + dy), npos(0.49, 0.68))
        glow_line(npos(0.38, 0.40 + dy), npos(0.52, 0.40 + dy))
        glow_line(npos(0.44, 0.68),       npos(0.54, 0.68))
        glow_line(npos(0.38, 0.40 + dy), npos(0.32, 0.55 + dy))
        glow_line(npos(0.32, 0.55 + dy), npos(0.34, 0.68))
        glow_line(npos(0.52, 0.40 + dy), npos(0.55, 0.55 + dy))
        glow_line(npos(0.55, 0.55 + dy), npos(0.58, 0.68))
        # Legs go forward then down (knees roughly at hip height)
        glow_line(npos(0.44, 0.68), npos(0.68, 0.70))
        glow_line(npos(0.68, 0.70), npos(0.74, 0.92))
        glow_line(npos(0.54, 0.68), npos(0.72, 0.74))
        glow_line(npos(0.72, 0.74), npos(0.80, 0.94))

        heart_glow(npos(0.48, 0.48 + breath_dy * 0.5), sw * 0.045)

    def _draw_walking(self, p, npos, sw, breath_dy, col, now,
                      glow_line, head_circle, heart_glow):
        # 1.2 Hz leg swing → arms swing in counter-phase.
        swing = math.sin(now * 2 * math.pi * 1.2)
        arm = -swing

        dy = breath_dy * 0.6
        head_circle(npos(0.50, 0.16 + dy), sw * 0.075)

        # Motion lines drifting behind the figure
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 70), 1))
        for y in (0.40, 0.50, 0.62):
            p.drawLine(npos(0.18, y), npos(0.32, y))

        glow_line(npos(0.50, 0.26 + dy), npos(0.50, 0.56))
        glow_line(npos(0.42, 0.30 + dy), npos(0.58, 0.30 + dy))
        glow_line(npos(0.45, 0.56),       npos(0.55, 0.56))
        # Arms swing
        glow_line(npos(0.42, 0.30 + dy), npos(0.40 - 0.05 * arm, 0.46 + dy))
        glow_line(npos(0.40 - 0.05 * arm, 0.46 + dy), npos(0.40 - 0.10 * arm, 0.60))
        glow_line(npos(0.58, 0.30 + dy), npos(0.60 + 0.05 * arm, 0.46 + dy))
        glow_line(npos(0.60 + 0.05 * arm, 0.46 + dy), npos(0.60 + 0.10 * arm, 0.60))
        # Legs alternate
        glow_line(npos(0.45, 0.56), npos(0.43 + 0.06 * swing, 0.76))
        glow_line(npos(0.43 + 0.06 * swing, 0.76), npos(0.42 + 0.12 * swing, 0.96))
        glow_line(npos(0.55, 0.56), npos(0.57 - 0.06 * swing, 0.76))
        glow_line(npos(0.57 - 0.06 * swing, 0.76), npos(0.58 - 0.12 * swing, 0.96))

        heart_glow(npos(0.50, 0.40 + breath_dy * 0.5), sw * 0.05)

    def _draw_lying(self, p, npos, sw, breath_dy, col, glow_line, head_circle, heart_glow):
        chest_dy = breath_dy * 0.6

        # Ground line under the figure
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 110),
                      1, Qt.PenStyle.DashLine))
        p.drawLine(npos(0.10, 0.78), npos(0.92, 0.78))

        head_circle(npos(0.18, 0.62), sw * 0.07)

        glow_line(npos(0.25, 0.62), npos(0.55, 0.62))           # spine
        glow_line(npos(0.28, 0.58 + chest_dy), npos(0.28, 0.66 + chest_dy))  # shoulders
        glow_line(npos(0.55, 0.60), npos(0.55, 0.64))           # hips
        # Arms outwards (slightly)
        glow_line(npos(0.28, 0.58 + chest_dy), npos(0.40, 0.52 + chest_dy))
        glow_line(npos(0.40, 0.52 + chest_dy), npos(0.50, 0.50))
        glow_line(npos(0.28, 0.66 + chest_dy), npos(0.40, 0.72 + chest_dy))
        glow_line(npos(0.40, 0.72 + chest_dy), npos(0.50, 0.74))
        # Legs straight along the floor
        glow_line(npos(0.55, 0.60), npos(0.72, 0.58))
        glow_line(npos(0.72, 0.58), npos(0.88, 0.58))
        glow_line(npos(0.55, 0.64), npos(0.72, 0.66))
        glow_line(npos(0.72, 0.66), npos(0.88, 0.66))

        heart_glow(npos(0.35, 0.62 + chest_dy), sw * 0.045)

import math
from typing import Optional

from PyQt6.QtCore import QPointF, QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from core.models import Detection

_PANEL = QColor("#161b22")
_BORDER = QColor("#30363d")
_ACCENT = QColor("#58a6ff")
_GREEN = QColor("#3fb950")
_YELLOW = QColor("#d29922")
_RED = QColor("#f85149")
_TEXT = QColor("#e6edf3")
_TEXT_SEC = QColor("#8b949e")
_GRID = QColor(22, 30, 42)

_ZONE_FILL_IDLE   = {
    "Cerca": QColor(63,  185,  80, 14),
    "Medio": QColor(88,  166, 255, 14),
    "Lejos": QColor(210, 153,  34, 14),
}
_ZONE_FILL_ACTIVE = {
    "Cerca": QColor(63,  185,  80, 45),
    "Medio": QColor(88,  166, 255, 45),
    "Lejos": QColor(210, 153,  34, 45),
}
_ZONE_EDGE = {
    "Cerca": QColor(63,  185,  80),
    "Medio": QColor(88,  166, 255),
    "Lejos": QColor(210, 153,  34),
}
_ZONE_LABELS = ["Cerca", "Medio", "Lejos"]


class RadarView(QWidget):
    """Top-down room view.

    Shows three coarse distance zones (Cerca / Medio / Lejos) as background
    bands, and — when a person is detected — draws a stylised human silhouette
    at the estimated position with a heat-glow aura proportional to activity.

    The silhouette position comes from the detector's distance_ratio /
    lateral_offset estimates.  It represents "a person is approximately here",
    not a centimetre-accurate location.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._detection: Optional[Detection] = None
        self._pulse: float = 0.0
        self.setMinimumSize(360, 440)

    def update_detection(self, detection: Detection) -> None:
        self._detection = detection
        self._pulse = (self._pulse + 0.2) % (2 * math.pi)
        self.update()

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, w, h, _PANEL)

        p.setPen(QPen(_TEXT))
        p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        p.drawText(14, 22, "RADAR  ·  Zona de detección")

        mx, mt = 52, 42
        rx, ry = mx, mt
        rw = w - 2 * mx
        rh = h - mt - mx + 8
        if rw < 10 or rh < 10:
            p.end()
            return

        self._draw_grid(p, rx, ry, rw, rh)
        self._draw_room(p, rx, ry, rw, rh)
        self._draw_zones(p, rx, ry, rw, rh)
        self._draw_fresnel_edge(p, rx, ry, rw, rh)
        self._draw_devices(p, rx, ry, rw, rh)
        self._draw_silhouette(p, rx, ry, rw, rh)
        self._draw_status(p, w, h)
        p.end()

    # ------------------------------------------------------------------
    # Rooms / zones
    # ------------------------------------------------------------------

    def _draw_grid(self, p, rx, ry, rw, rh) -> None:
        p.setPen(QPen(_GRID, 1))
        for i in range(1, 4):
            x = rx + i * rw // 3
            p.drawLine(x, ry, x, ry + rh)
        for i in range(1, 6):
            y = ry + i * rh // 6
            p.drawLine(rx, y, rx + rw, y)

    def _draw_room(self, p, rx, ry, rw, rh) -> None:
        p.setPen(QPen(_BORDER, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rx, ry, rw, rh)

    def _draw_zones(self, p, rx, ry, rw, rh) -> None:
        active = self._detection.distance_zone if self._detection else "—"
        zone_w = rw // 3
        for i, name in enumerate(_ZONE_LABELS):
            x0 = rx + i * zone_w
            w0 = zone_w if i < 2 else rw - 2 * zone_w
            fill = _ZONE_FILL_ACTIVE[name] if name == active else _ZONE_FILL_IDLE[name]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(fill))
            p.drawRect(x0, ry, w0, rh)
            if i > 0:
                edge = _ZONE_EDGE[name] if name == active else _BORDER
                p.setPen(QPen(edge, 1, Qt.PenStyle.DashLine))
                p.drawLine(x0, ry, x0, ry + rh)
            cx_z = x0 + w0 // 2
            cy_z = ry + 14
            if name == active:
                p.setPen(QPen(_ZONE_EDGE[name]))
                p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            else:
                p.setPen(QPen(QColor(70, 80, 95)))
                p.setFont(QFont("Consolas", 8))
            p.drawText(QRect(x0, ry + 4, w0, 16), Qt.AlignmentFlag.AlignCenter, name)

    def _draw_fresnel_edge(self, p, rx, ry, rw, rh) -> None:
        cx = rx + rw // 2
        cy = ry + rh // 2
        p.setPen(QPen(QColor(88, 166, 255, 40), 1, Qt.PenStyle.DotLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), rw * 0.82 / 2, rh * 0.46 / 2)

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------

    def _draw_devices(self, p, rx, ry, rw, rh) -> None:
        cy = ry + rh // 2
        # ESP32
        dw, dh = 22, 42
        dx = rx + 5
        dy = cy - dh // 2
        p.setPen(QPen(_ACCENT, 1))
        p.setBrush(QBrush(QColor(88, 166, 255, 40)))
        p.drawRect(dx, dy, dw, dh)
        p.setPen(QPen(_ACCENT, 2))
        p.drawLine(dx + dw // 2, dy, dx + dw // 2, dy - 7)
        p.drawLine(dx + dw // 2 - 4, dy - 7, dx + dw // 2 + 4, dy - 7)
        p.setPen(QPen(_ACCENT))
        p.setFont(QFont("Consolas", 7))
        p.drawText(rx + 2, dy + dh + 13, "ESP32")
        # Router
        rw2, rh2 = 22, 36
        rtx = rx + rw - rw2 - 5
        rty = cy - rh2 // 2
        p.setPen(QPen(_GREEN, 1))
        p.setBrush(QBrush(QColor(63, 185, 80, 40)))
        p.drawRect(rtx, rty, rw2, rh2)
        p.setPen(QPen(_GREEN, 2))
        for off in (-4, 4):
            ax = rtx + rw2 // 2 + off
            p.drawLine(ax, rty, ax, rty - 7)
        p.setPen(QPen(_GREEN))
        p.setFont(QFont("Consolas", 7))
        p.drawText(rtx + 1, rty + rh2 + 13, "Router")

    # ------------------------------------------------------------------
    # Human silhouette
    # ------------------------------------------------------------------

    def _draw_silhouette(self, p, rx, ry, rw, rh) -> None:
        if self._detection is None or not self._detection.present:
            return
        d = self._detection

        # Position in room pixels
        pad_x, pad_y = 68, 32
        cx = rx + pad_x + int(d.distance_ratio * (rw - 2 * pad_x))
        cy = ry + rh // 2 + int(d.lateral_offset * (rh // 2 - pad_y))

        # Colour by confidence
        if d.confidence > 0.66:
            col = _GREEN
        elif d.confidence > 0.33:
            col = _YELLOW
        else:
            col = _RED

        # Scale relative to room
        sc = min(rw, rh) * 0.13

        # ---- Heat aura (3 concentric soft ellipses) ----
        pulse_extra = math.sin(self._pulse) * 0.12
        activity_extra = d.activity_level * 0.4
        for radius_factor, alpha in [(1.8 + pulse_extra + activity_extra, 18),
                                      (1.2 + pulse_extra * 0.6, 32),
                                      (0.75, 50)]:
            r_x = sc * radius_factor * 0.9
            r_y = sc * radius_factor * 1.4
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), alpha)))
            p.drawEllipse(QPointF(cx, cy), r_x, r_y)

        # ---- Head ----
        head_r = sc * 0.30
        head_cy = cy - sc * 0.72
        p.setPen(QPen(col, 1))
        p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 130)))
        p.drawEllipse(QPointF(cx, head_cy), head_r, head_r)

        # ---- Shoulders line ----
        shoulder_w = sc * 0.52
        shoulder_y = cy - sc * 0.35
        p.setPen(QPen(col, 2))
        p.drawLine(QPointF(cx - shoulder_w, shoulder_y),
                   QPointF(cx + shoulder_w, shoulder_y))

        # ---- Torso ----
        torso_w = sc * 0.38
        torso_h = sc * 0.44
        torso_cy = cy + sc * 0.10
        p.setPen(QPen(col, 1))
        p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 90)))
        p.drawEllipse(QPointF(cx, torso_cy), torso_w, torso_h)

        # ---- Neck ----
        neck_top = head_cy + head_r
        neck_bot = shoulder_y
        p.setPen(QPen(col, 2))
        p.drawLine(QPointF(cx, neck_top), QPointF(cx, neck_bot))

        # ---- Label ----
        p.setFont(QFont("Consolas", 8))
        p.setPen(QPen(col))
        lbl = f"{d.distance_zone}  {int(d.activity_level * 100)}%"
        p.drawText(
            QRect(int(cx) - 50, int(cy + sc * 0.65), 100, 14),
            Qt.AlignmentFlag.AlignCenter,
            lbl,
        )

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _draw_status(self, p, w, h) -> None:
        p.setFont(QFont("Consolas", 9))
        if self._detection and self._detection.present:
            d = self._detection
            txt = (
                f"Zona: {d.distance_zone}   "
                f"Confianza: {int(d.confidence * 100)}%   "
                f"Actividad: {int(d.activity_level * 100)}%"
            )
            p.setPen(QPen(_GREEN))
        else:
            txt = "Sin detección"
            p.setPen(QPen(_TEXT_SEC))
        p.drawText(14, h - 10, txt)

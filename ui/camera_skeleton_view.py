"""Live webcam view with COCO-17 skeleton overlay.

Used in the recording tab so the user can visually confirm the pose being
captured while CSI samples are being saved. Camera capture + MediaPipe pose
inference run in a background QThread to keep the UI responsive.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QRect, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from core.training.pose_extractor import PoseExtractor

logger = logging.getLogger(__name__)

_PANEL = "#161b22"
_TEXT = "#e6edf3"
_TEXT_SEC = "#8b949e"
_ACCENT = "#58a6ff"
_GREEN = "#3fb950"
_YELLOW = "#d29922"
_RED = "#f85149"

# COCO-17 skeleton edges (mirrors ui/skeleton_view.py:BONES_17)
_BONES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

_VIS_THRESH = 0.3
# Joints required for a "good" pose — same set used by compute_bbox
_TORSO_JOINTS = {5, 6, 11, 12}


class _CameraWorker(QThread):
    """Reads BGR frames from the default webcam and runs pose extraction."""

    frame_ready = pyqtSignal(object, object)   # (rgb ndarray HxWx3 uint8, keypoints | None)
    error       = pyqtSignal(str)

    def __init__(self, cam_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._cam_index = cam_index
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.error.emit("opencv-python no instalado")
            return

        # On Windows DSHOW opens faster and avoids the long MSMF probe
        cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._cam_index)
        if not cap.isOpened():
            self.error.emit("Cámara no disponible")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        extractor = PoseExtractor()
        if not extractor.available:
            self.error.emit("MediaPipe no disponible (mostrando solo cámara)")

        try:
            while not self._stop:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    self.msleep(30)
                    continue
                # Mirror so movements on screen feel natural
                bgr = cv2.flip(bgr, 1)
                kps = extractor.extract(bgr) if extractor.available else None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb, kps)
                self.msleep(33)
        finally:
            extractor.close()
            cap.release()


class CameraSkeletonView(QWidget):
    """Webcam view with the COCO-17 skeleton overlaid in real time."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._kps:   Optional[List[Tuple[float, float, float]]] = None
        self._error: Optional[str] = None
        self._worker: Optional[_CameraWorker] = None
        self.setMinimumSize(360, 270)
        self.setStyleSheet(f"background: {_PANEL};")

    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._error = None
        self._worker = _CameraWorker(parent=self)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        self._worker.wait(2000)
        self._worker = None

    # ------------------------------------------------------------------

    def _on_frame(self, rgb_frame, keypoints) -> None:
        self._frame = rgb_frame
        self._kps   = keypoints
        self.update()

    def _on_error(self, msg: str) -> None:
        self._error = msg
        self.update()

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, w, h, QColor(_PANEL))

        p.setPen(QPen(QColor(_TEXT)))
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(12, 18, "Cámara  ·  Esqueleto COCO-17")

        if self._frame is None:
            p.setPen(QPen(QColor(_TEXT_SEC)))
            p.setFont(QFont("Consolas", 10))
            p.drawText(
                QRect(0, 0, w, h),
                Qt.AlignmentFlag.AlignCenter,
                self._error or "Iniciando cámara…",
            )
            p.end()
            return

        rgb = self._frame
        fh, fw = rgb.shape[:2]
        img = QImage(rgb.tobytes(), fw, fh, fw * 3, QImage.Format.Format_RGB888)

        # Letterbox the camera frame into the widget, preserving aspect ratio
        top_margin = 26
        bot_margin = 16
        side_pad   = 12
        avail_w = w - 2 * side_pad
        avail_h = h - top_margin - bot_margin
        if avail_w < 10 or avail_h < 10:
            p.end()
            return
        scale = min(avail_w / fw, avail_h / fh)
        out_w = int(fw * scale)
        out_h = int(fh * scale)
        out_x = side_pad + (avail_w - out_w) // 2
        out_y = top_margin + (avail_h - out_h) // 2
        p.drawImage(QRect(out_x, out_y, out_w, out_h), img)

        # ----- Skeleton overlay -----
        if self._kps is not None and len(self._kps) == 17:
            def to_screen(idx: int) -> Tuple[float, float, float]:
                x, y, v = self._kps[idx]
                return (out_x + x * out_w, out_y + y * out_h, v)

            # Bones: dark outer glow + bright inner line
            for i, j in _BONES:
                ax, ay, av = to_screen(i)
                bx, by, bv = to_screen(j)
                if av < _VIS_THRESH or bv < _VIS_THRESH:
                    continue
                p.setPen(QPen(QColor(88, 166, 255, 90), 6))
                p.drawLine(int(ax), int(ay), int(bx), int(by))
                p.setPen(QPen(QColor(63, 185, 80, 230), 2))
                p.drawLine(int(ax), int(ay), int(bx), int(by))

            # Joints — torso anchors highlighted, others smaller
            for idx in range(17):
                x, y, v = to_screen(idx)
                if v < _VIS_THRESH:
                    continue
                is_torso = idx in _TORSO_JOINTS
                r = 6 if is_torso else 4
                color = QColor(255, 180, 40, 230) if is_torso else QColor(255, 220, 90, 200)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(color))
                p.drawEllipse(int(x - r), int(y - r), r * 2, r * 2)

            n_vis = sum(1 for _, _, vv in self._kps if vv >= _VIS_THRESH)
            torso_ok = all(self._kps[i][2] >= _VIS_THRESH for i in _TORSO_JOINTS)
            status_color = _GREEN if (torso_ok and n_vis >= 10) else _YELLOW
            p.setPen(QPen(QColor(status_color)))
            p.setFont(QFont("Consolas", 9))
            p.drawText(12, h - 4, f"{n_vis}/17 visibles  ·  torso {'OK' if torso_ok else 'incompleto'}")
        else:
            p.setPen(QPen(QColor(_TEXT_SEC)))
            p.setFont(QFont("Consolas", 9))
            p.drawText(12, h - 4, "Sin pose detectada — colócate frente a la cámara")

        p.end()

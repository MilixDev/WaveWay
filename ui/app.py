from PyQt6.QtCore import Qt, QRect, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.csi_reader import CSIReader
from core.detector import Detector
from core.training.model import ActivityClassifier, WINDOW_SIZE
from .activity_view import ActivityView
from .heatmap_view import HeatmapView
from .radar_view import RadarView
from .vitals_view import BreathingView

_BG = "#0d1117"
_PANEL = "#161b22"
_BORDER = "#30363d"
_ACCENT = "#58a6ff"
_GREEN = "#3fb950"
_RED_ERR = "#f85149"
_YELLOW = "#d29922"
_TEXT = "#e6edf3"
_TEXT_SEC = "#8b949e"


def _h_sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {_BORDER}; border: none;")
    return f


def _v_sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setFixedWidth(1)
    f.setStyleSheet(f"background: {_BORDER}; border: none;")
    return f


# -----------------------------------------------------------------------
# Calibration status bar
# -----------------------------------------------------------------------

class CalibrationBar(QWidget):
    """Thin bar showing calibration state: uncalibrated / progress / done."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state: str = "none"   # "none" | "running" | "done"
        self._progress: float = 0.0
        self._threshold: float = 0.02
        self._live_var: float = 0.0
        self.setFixedHeight(26)
        self.setStyleSheet(f"background: {_BG};")

    def set_none(self, live_variance: float = 0.0, threshold: float = 0.02) -> None:
        self._state = "none"
        self._live_var = live_variance
        self._threshold = threshold
        self.update()

    def set_running(self, progress: float) -> None:
        self._state = "running"
        self._progress = progress
        self.update()

    def set_done(self, threshold: float) -> None:
        self._state = "done"
        self._threshold = threshold
        self.update()

    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, w, h, QColor(_BG))

        pad = 14
        font = QFont("Consolas", 9)
        p.setFont(font)

        if self._state == "none":
            ratio = self._live_var / (self._threshold + 1e-9)
            bar_x, bar_w, bar_h = pad, 140, 8
            bar_y = (h - bar_h) // 2
            # Track
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(30, 38, 50)))
            p.drawRect(bar_x, bar_y, bar_w, bar_h)
            # Fill — colour shifts from blue→green as signal approaches threshold
            fill_w = max(0, min(bar_w, int(ratio * bar_w * 0.5)))
            fill_col = QColor(_GREEN) if ratio > 0.8 else QColor(_ACCENT)
            p.setBrush(QBrush(fill_col))
            p.drawRect(bar_x, bar_y, fill_w, bar_h)
            # Labels
            p.setPen(QPen(QColor(_YELLOW)))
            p.drawText(
                bar_x + bar_w + 8, 17,
                f"Sin calibrar  ·  var={self._live_var:.4f}  umbral={self._threshold:.4f}"
                f"  —  Calibrar con sala vacía",
            )

        elif self._state == "running":
            bar_x = pad
            bar_w = min(300, w // 3)
            bar_h = 8
            bar_y = (h - bar_h) // 2

            # Track
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(30, 38, 50)))
            p.drawRect(bar_x, bar_y, bar_w, bar_h)

            # Fill
            fill_w = max(4, int(self._progress * bar_w))
            p.setBrush(QBrush(QColor(_ACCENT)))
            p.drawRect(bar_x, bar_y, fill_w, bar_h)

            p.setPen(QPen(QColor(_ACCENT)))
            pct = int(self._progress * 100)
            p.drawText(bar_x + bar_w + 10, 17, f"Calibrando sala vacía…  {pct}%")

        elif self._state == "done":
            # Full green bar
            bar_x, bar_w, bar_h = pad, 120, 8
            bar_y = (h - bar_h) // 2
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(30, 50, 34)))
            p.drawRect(bar_x, bar_y, bar_w, bar_h)
            p.setBrush(QBrush(QColor(_GREEN)))
            p.drawRect(bar_x, bar_y, bar_w, bar_h)

            p.setPen(QPen(QColor(_GREEN)))
            p.drawText(
                bar_x + bar_w + 10, 17,
                f"Calibrado  ·  umbral adaptado: {self._threshold:.4f}",
            )

        p.end()


# -----------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window for WaveWay — WiFi CSI presence detection."""

    UPDATE_MS: int = 100

    def __init__(self) -> None:
        super().__init__()
        self.reader = CSIReader(port="COM6", baudrate=921600)
        self.detector = Detector(self.reader)
        self._running = False
        self._training_window = None

        # Load activity classifier if weights exist
        from pathlib import Path
        _weights = Path(__file__).resolve().parent.parent / "training_data" / "activity_model.npz"
        self._activity_model = ActivityClassifier(_weights if _weights.exists() else None)

        self.setWindowTitle("WaveWay")
        self.setMinimumSize(1280, 820)
        self._apply_style()
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(self.UPDATE_MS)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background-color: {_BG};
                color: {_TEXT};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {_PANEL};
                color: {_TEXT};
                border: 1px solid {_BORDER};
                padding: 4px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #21262d;
                border-color: {_ACCENT};
                color: {_ACCENT};
            }}
            QPushButton:disabled {{
                color: {_TEXT_SEC};
                border-color: {_BORDER};
                background-color: {_BG};
            }}
            QPushButton:pressed {{ background-color: {_BG}; }}
            """
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        vlay.addWidget(self._build_topbar())
        vlay.addWidget(_h_sep())
        self.cal_bar = CalibrationBar()
        vlay.addWidget(self.cal_bar)
        vlay.addWidget(_h_sep())
        vlay.addWidget(self._build_body(), stretch=1)

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet(f"background: {_PANEL};")

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(14)

        name = QLabel("WaveWay")
        name.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        name.setStyleSheet(f"color: {_ACCENT}; background: transparent;")
        lay.addWidget(name)

        lay.addWidget(_v_sep())

        self.lbl_mode = QLabel("Sin señal")
        self.lbl_mode.setFont(QFont("Consolas", 10))
        self.lbl_mode.setStyleSheet(f"color: {_TEXT_SEC}; background: transparent;")
        lay.addWidget(self.lbl_mode)

        lay.addStretch()

        # Calibrate button
        self.btn_cal = QPushButton("Calibrar")
        self.btn_cal.setFixedSize(100, 30)
        self.btn_cal.setToolTip("Captura el ruido de fondo (sala vacía, 5 s)")
        self.btn_cal.clicked.connect(self._on_calibrate)
        self.btn_cal.setEnabled(False)
        lay.addWidget(self.btn_cal)

        lay.addWidget(_v_sep())

        # Train button
        self.btn_train = QPushButton("Entrenar")
        self.btn_train.setFixedSize(100, 30)
        self.btn_train.setToolTip("Abre el modo de grabación y entrenamiento de poses")
        self.btn_train.clicked.connect(self._on_open_training)
        lay.addWidget(self.btn_train)

        lay.addWidget(_v_sep())

        # Detection mode toggle
        self.btn_mode = QPushButton("Modo: Ambiente")
        self.btn_mode.setFixedSize(140, 30)
        self.btn_mode.setToolTip(
            "Ambiente: detecta movimiento en toda la sala\n"
            "LOS: solo detecta cuando bloqueas el camino directo ESP32↔Router"
        )
        self.btn_mode.setCheckable(True)
        self.btn_mode.setChecked(False)
        self.btn_mode.clicked.connect(self._on_toggle_mode)
        lay.addWidget(self.btn_mode)

        lay.addWidget(_v_sep())

        # Activity panel toggle
        self.btn_activity = QPushButton("Actividad ▼")
        self.btn_activity.setFixedSize(110, 30)
        self.btn_activity.setToolTip("Mostrar / ocultar el panel de actividad")
        self.btn_activity.setCheckable(True)
        self.btn_activity.setChecked(True)
        self.btn_activity.clicked.connect(self._on_toggle_activity)
        lay.addWidget(self.btn_activity)

        lay.addWidget(_v_sep())

        # Heart-rate toggle (experimental)
        self.btn_heart = QPushButton("♥ Latido OFF")
        self.btn_heart.setFixedSize(120, 30)
        self.btn_heart.setToolTip(
            "Estimación experimental de frecuencia cardíaca\n"
            "Requiere sujeto estático, ≥20 s de datos, SNR muy alta.\n"
            "Banda: 48–150 bpm."
        )
        self.btn_heart.setCheckable(True)
        self.btn_heart.setChecked(False)
        self.btn_heart.clicked.connect(self._on_toggle_heart)
        lay.addWidget(self.btn_heart)

        lay.addWidget(_v_sep())

        # Start/stop button
        self.btn_toggle = QPushButton("Iniciar")
        self.btn_toggle.setFixedSize(104, 30)
        self.btn_toggle.clicked.connect(self._on_toggle)
        lay.addWidget(self.btn_toggle)

        lay.addWidget(_v_sep())

        self.lbl_conn = QLabel("● Desconectado")
        self.lbl_conn.setFont(QFont("Consolas", 10))
        self.lbl_conn.setStyleSheet(f"color: {_TEXT_SEC}; background: transparent;")
        lay.addWidget(self.lbl_conn)

        lay.addWidget(_v_sep())

        self.lbl_frames = QLabel("frames: 0")
        self.lbl_frames.setFont(QFont("Consolas", 9))
        self.lbl_frames.setStyleSheet(f"color: {_TEXT_SEC}; background: transparent;")
        lay.addWidget(self.lbl_frames)

        return bar

    def _build_body(self) -> QWidget:
        body = QWidget()
        hlay = QHBoxLayout(body)
        hlay.setContentsMargins(0, 0, 0, 0)
        hlay.setSpacing(0)

        # Left: radar
        self.radar = RadarView()
        self.radar.setFixedWidth(420)
        self.radar.setStyleSheet(
            f"background: {_PANEL}; border-right: 1px solid {_BORDER};"
        )
        hlay.addWidget(self.radar)

        # Right panel
        right = QWidget()
        right.setStyleSheet(f"background: {_BG};")
        rvlay = QVBoxLayout(right)
        rvlay.setContentsMargins(0, 0, 0, 0)
        rvlay.setSpacing(0)

        # Top row: activity (left) + breathing (right)
        top = QWidget()
        thlay = QHBoxLayout(top)
        thlay.setContentsMargins(0, 0, 0, 0)
        thlay.setSpacing(0)

        self.activity = ActivityView()
        self.activity.setStyleSheet(
            f"background: {_PANEL}; border-right: 1px solid {_BORDER};"
            f" border-bottom: 1px solid {_BORDER};"
        )
        thlay.addWidget(self.activity, stretch=2)

        self.breathing = BreathingView()
        self.breathing.setStyleSheet(
            f"background: {_PANEL}; border-bottom: 1px solid {_BORDER};"
        )
        thlay.addWidget(self.breathing, stretch=3)

        rvlay.addWidget(top, stretch=3)
        rvlay.addWidget(_h_sep())

        # Bottom: heatmap
        self.heatmap = HeatmapView()
        self.heatmap.setMinimumHeight(170)
        self.heatmap.setStyleSheet(f"background: {_PANEL};")
        rvlay.addWidget(self.heatmap, stretch=2)

        hlay.addWidget(right, stretch=1)
        return body

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _on_calibrate(self) -> None:
        if not self._running:
            return
        self.detector.start_calibration()
        self.btn_cal.setEnabled(False)

    def _on_toggle_mode(self) -> None:
        los = self.btn_mode.isChecked()
        self.detector.detection_mode = "los" if los else "ambiente"
        if los:
            self.btn_mode.setText("Modo: LOS")
            self.btn_mode.setStyleSheet(
                f"background: #1a2a1a; border: 1px solid {_GREEN}; color: {_GREEN};"
            )
        else:
            self.btn_mode.setText("Modo: Ambiente")
            self.btn_mode.setStyleSheet("")

    def _on_toggle_activity(self) -> None:
        visible = self.btn_activity.isChecked()
        self.activity.setVisible(visible)
        self.btn_activity.setText("Actividad ▼" if visible else "Actividad ►")

    def _on_toggle_heart(self) -> None:
        on = self.btn_heart.isChecked()
        self.detector.heart_enabled = on
        if on:
            self.btn_heart.setText("♥ Latido ON")
            self.btn_heart.setStyleSheet(
                f"background: #2d1b1b; border: 1px solid {_RED_ERR}; color: {_RED_ERR};"
            )
        else:
            self.btn_heart.setText("♥ Latido OFF")
            self.btn_heart.setStyleSheet("")

    def _on_open_training(self) -> None:
        from .training_window import TrainingWindow
        if self._training_window is None or not self._training_window.isVisible():
            self._training_window = TrainingWindow(self.reader, parent=self)
            self._training_window.finished.connect(self._reload_pose_model)
            self._training_window.show()
        else:
            self._training_window.raise_()
            self._training_window.activateWindow()

    def _reload_pose_model(self) -> None:
        from pathlib import Path
        _weights = Path(__file__).resolve().parent.parent / "training_data" / "activity_model.npz"
        if _weights.exists():
            self._activity_model.load(_weights)
            self.btn_train.setText("Entrenar ✓")
            self.btn_train.setStyleSheet(
                f"color: #3fb950; border-color: #3fb950;"
            )

    def _start(self) -> None:
        self.reader.start()
        self._timer.start()
        self._running = True
        self.btn_toggle.setText("Detener")
        self.btn_toggle.setStyleSheet(
            "background: #2d1b1b; border: 1px solid #f85149;"
            " color: #f85149; padding: 4px 16px; font-size: 12px;"
        )
        self.btn_cal.setEnabled(True)

    def _stop(self) -> None:
        self._timer.stop()
        self.reader.stop()
        self._running = False
        self.btn_toggle.setText("Iniciar")
        self.btn_toggle.setStyleSheet("")
        self.btn_cal.setEnabled(False)
        self._reset_labels()

    def _reset_labels(self) -> None:
        self.lbl_mode.setText("Sin señal")
        self.lbl_mode.setStyleSheet(f"color: {_TEXT_SEC}; background: transparent;")
        self.lbl_conn.setText("● Desconectado")
        self.lbl_conn.setStyleSheet(f"color: {_TEXT_SEC}; background: transparent;")

    def _tick(self) -> None:
        detection = self.detector.detect()
        frames = self.reader.get_frames()
        connected = self.reader.connected

        # --- Calibration bar ---
        if self.detector.calibrating:
            self.cal_bar.set_running(self.detector.calibration_progress)
            self.btn_cal.setEnabled(False)
        elif self.detector.is_calibrated:
            self.cal_bar.set_done(self.detector.presence_threshold)
            self.btn_cal.setEnabled(True)
            self.btn_cal.setText("Recalibrar")
        else:
            self.cal_bar.set_none(
                live_variance=self.detector.live_variance,
                threshold=self.detector.presence_threshold,
            )

        # --- Connection indicator ---
        if connected:
            self.lbl_conn.setText("● Conectado")
            self.lbl_conn.setStyleSheet(f"color: {_GREEN}; background: transparent;")
        else:
            self.lbl_conn.setText("● Sin señal")
            self.lbl_conn.setStyleSheet(f"color: {_RED_ERR}; background: transparent;")

        # --- Mode label ---
        latest = self.reader.get_latest()
        if detection.present:
            self.lbl_mode.setText("Real · Detección activa")
            self.lbl_mode.setStyleSheet(f"color: {_GREEN}; background: transparent;")
        elif connected and latest:
            self.lbl_mode.setText("Real · En espera")
            self.lbl_mode.setStyleSheet(f"color: {_ACCENT}; background: transparent;")
        else:
            self.lbl_mode.setText("Sin señal")
            self.lbl_mode.setStyleSheet(f"color: {_TEXT_SEC}; background: transparent;")

        # --- Frame counter ---
        self.lbl_frames.setText(f"frames: {len(frames)}")

        # --- Activity classifier inference (if trained model available) ---
        prediction = None
        if self._activity_model.is_loaded and detection.present:
            raw_frames = [f[1] for f in frames]
            prediction = self._activity_model.predict(raw_frames)

        # --- Sub-views ---
        self.radar.update_detection(detection)
        self.activity.update_detection(
            detection, prediction, model_loaded=self._activity_model.is_loaded,
        )
        self.breathing.update_detection(detection)
        self.heatmap.update_data(frames)

    def closeEvent(self, event) -> None:
        self.reader.stop()
        super().closeEvent(event)

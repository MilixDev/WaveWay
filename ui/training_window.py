"""Training window — three-tab UI for recording sessions and training a classifier.

Tab 1 — Grabación
  Class dropdown · webcam preview with skeleton overlay · live CSI heatmap
  Session name + record button at the bottom. The webcam is a visual aid
  only; pose data is NOT saved — sessions only store CSI + the chosen class.

Tab 2 — Entrenamiento
  Per-class sample summary, training button, epoch progress, val-loss graph.

Tab 3 — Muestras
  List of recorded sessions with rename / delete / refresh.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QRect, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.csi_reader import CSIReader
from core.training.activity_classes import CLASS_NAMES, NUM_CLASSES, class_name
from core.training.data_collector import DataCollector
from core.training.trainer import EPOCHS, train as train_model
from ui.camera_skeleton_view import CameraSkeletonView

logger = logging.getLogger(__name__)

_BG     = "#0d1117"
_PANEL  = "#161b22"
_BORDER = "#30363d"
_ACCENT = "#58a6ff"
_GREEN  = "#3fb950"
_RED    = "#f85149"
_YELLOW = "#d29922"
_TEXT   = "#e6edf3"
_TEXT_SEC = "#8b949e"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR      = _PROJECT_ROOT / "training_data"
WEIGHTS_PATH  = DATA_DIR / "activity_model.npz"


# -----------------------------------------------------------------------
# Live CSI waterfall (recording tab)
# -----------------------------------------------------------------------

class LiveHeatmap(QWidget):
    """Mid-size CSI waterfall plot — fed by the same buffer the reader fills."""

    _MAX_FRAMES = 220

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frames: List[List[float]] = []
        self._img: Optional[QImage] = None
        self.setMinimumSize(400, 240)
        self.setStyleSheet(f"background: {_PANEL};")

    def update_data(self, frames) -> None:
        self._frames = [f[1] for f in frames[-self._MAX_FRAMES:]]
        self._img = None
        self.update()

    def paintEvent(self, event) -> None:
        from ui.heatmap_view import _CMAP
        pw, ph = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, pw, ph, QColor(_PANEL))

        p.setPen(QPen(QColor(_TEXT)))
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(12, 18, "CSI  ·  Espectro en tiempo real")

        if not self._frames:
            p.setPen(QPen(QColor(_TEXT_SEC)))
            p.setFont(QFont("Consolas", 10))
            p.drawText(QRect(0, 0, pw, ph), Qt.AlignmentFlag.AlignCenter, "Sin señal")
            p.end()
            return

        if self._img is None:
            self._img = self._build_image()

        if self._img is not None:
            p.drawImage(QRectF(12, 24, pw - 24, ph - 36), self._img)

        p.setPen(QPen(QColor(_TEXT_SEC)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(12, ph - 4, "← Antiguo")
        p.drawText(QRect(12, ph - 14, pw - 24, 12),
                   Qt.AlignmentFlag.AlignRight, "Reciente →")
        p.end()

    def _build_image(self) -> Optional[QImage]:
        from ui.heatmap_view import _CMAP
        frames = self._frames
        if not frames:
            return None
        lengths = [len(f) for f in frames if f]
        if not lengths:
            return None
        num_sc = int(np.bincount(lengths).argmax())
        if num_sc == 0:
            return None
        nf = len(frames)
        arr = np.zeros((num_sc, nf), dtype=np.float32)
        for j, frame in enumerate(frames):
            n = min(len(frame), num_sc)
            if n:
                arr[:n, j] = frame[:n]
        col_min = arr.min(axis=0, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        denom = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
        norm = np.flipud((arr - col_min) / denom)
        idx = np.clip((norm * 255).astype(np.uint8), 0, 255)
        rgb = np.ascontiguousarray(_CMAP[idx])
        h, w = rgb.shape[:2]
        img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return img.copy()


# -----------------------------------------------------------------------
# Loss graph
# -----------------------------------------------------------------------

class LossGraph(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._losses: List[float] = []
        self.setMinimumHeight(120)
        self.setStyleSheet(f"background: {_PANEL};")

    def add_loss(self, loss: float) -> None:
        self._losses.append(loss)
        self.update()

    def reset(self) -> None:
        self._losses.clear()
        self.update()

    def paintEvent(self, event) -> None:
        pw, ph = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, pw, ph, QColor(_PANEL))

        p.setPen(QPen(QColor(_TEXT)))
        p.setFont(QFont("Consolas", 9))
        p.drawText(8, 14, "Val loss (cross-entropy) por época")

        if len(self._losses) < 2:
            p.end()
            return

        pad = 20
        gw = pw - 2 * pad
        gh = ph - pad - 20
        if gw < 10 or gh < 10:
            p.end()
            return

        mn = min(self._losses)
        mx = max(self._losses)
        if mx - mn < 1e-9:
            mx = mn + 1e-4

        pts = []
        for i, v in enumerate(self._losses):
            x = pad + int(i / (len(self._losses) - 1) * gw)
            y = 20 + int((1 - (v - mn) / (mx - mn)) * gh)
            pts.append((x, y))

        p.setPen(QPen(QColor(_ACCENT), 1))
        for i in range(1, len(pts)):
            p.drawLine(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])

        p.setPen(QPen(QColor(_TEXT_SEC)))
        p.drawText(pad, 20 + gh + 12, f"{mn:.4f}")
        p.setFont(QFont("Consolas", 9))
        p.drawText(pad, 30, f"{mx:.4f}")
        p.end()


# -----------------------------------------------------------------------
# Worker threads
# -----------------------------------------------------------------------

class RecordingThread(QThread):
    """Pulls CSI from the reader and feeds a DataCollector at SAMPLE_PERIOD."""

    sample_ready = pyqtSignal(int, float)   # (n_samples, elapsed_s)
    finished_sig = pyqtSignal(object)       # output Path | None

    SAMPLE_PERIOD = 0.10                    # seconds between saved samples (10 Hz)

    def __init__(
        self, reader: CSIReader, session_name: str, class_idx: int, parent=None
    ) -> None:
        super().__init__(parent)
        self._reader     = reader
        self._session    = session_name
        self._class_idx  = class_idx
        self._stop_req   = False

    def request_stop(self) -> None:
        self._stop_req = True

    def run(self) -> None:
        # Take over the reader only if the main window hasn't already started it.
        reader_was_running = self._reader.running
        if not reader_was_running:
            self._reader.start()
            time.sleep(0.5)   # give the serial port time to deliver a few frames

        collector = DataCollector(
            output_dir=DATA_DIR,
            session_name=self._session,
            class_idx=self._class_idx,
        )

        t_start     = time.time()
        t_last_samp = 0.0

        while not self._stop_req:
            now = time.time()
            if now - t_last_samp >= self.SAMPLE_PERIOD:
                latest = self._reader.get_latest()
                if latest:
                    if collector.add_sample(latest, now):
                        self.sample_ready.emit(collector.n_samples, now - t_start)
                t_last_samp = now
            else:
                time.sleep(0.005)

        if not reader_was_running:
            self._reader.stop()

        out_path = collector.save()
        self.finished_sig.emit(out_path)


class TrainingThread(QThread):
    progress_sig = pyqtSignal(int, int, float)
    finished_sig = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def run(self) -> None:
        def _cb(epoch, total, loss):
            self.progress_sig.emit(epoch, total, loss)
        ok = train_model(DATA_DIR, WEIGHTS_PATH, progress_cb=_cb)
        self.finished_sig.emit(ok)


# -----------------------------------------------------------------------
# Main training window
# -----------------------------------------------------------------------

class TrainingWindow(QMainWindow):
    """Standalone window opened from the main app's 'Entrenar' button."""

    finished = pyqtSignal()

    def __init__(self, reader: CSIReader, parent=None) -> None:
        super().__init__(parent)
        self._reader = reader
        self._recording = False
        self._rec_thread:   Optional[RecordingThread] = None
        self._train_thread: Optional[TrainingThread]  = None
        self._heatmap_timer = QTimer(self)
        self._heatmap_timer.setInterval(100)
        self._heatmap_timer.timeout.connect(self._tick_heatmap)

        self.setWindowTitle("WaveWay — Modo Entrenamiento (clases de actividad)")
        self.setMinimumSize(900, 620)
        self._apply_style()
        self._build_ui()
        self._heatmap_timer.start()

    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{ background: {_BG}; color: {_TEXT};
                font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; }}
            QTabWidget::pane {{ border: 1px solid {_BORDER}; }}
            QTabBar::tab {{ background: {_PANEL}; color: {_TEXT_SEC};
                padding: 6px 20px; border: 1px solid {_BORDER}; }}
            QTabBar::tab:selected {{ color: {_ACCENT}; border-bottom: 2px solid {_ACCENT}; }}
            QPushButton {{ background: {_PANEL}; color: {_TEXT};
                border: 1px solid {_BORDER}; padding: 5px 16px; font-size: 12px; }}
            QPushButton:hover {{ border-color: {_ACCENT}; color: {_ACCENT}; }}
            QPushButton:disabled {{ color: {_TEXT_SEC}; }}
            QLineEdit, QComboBox {{ background: #21262d; color: {_TEXT};
                border: 1px solid {_BORDER}; padding: 3px 8px; font-size: 12px; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{ background: {_PANEL}; color: {_TEXT};
                border: 1px solid {_BORDER}; selection-background-color: {_ACCENT}; }}
            QProgressBar {{ background: #21262d; border: 1px solid {_BORDER};
                text-align: center; color: {_TEXT}; }}
            QProgressBar::chunk {{ background: {_ACCENT}; }}
            """
        )

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_recording_tab(),  "  Grabación  ")
        tabs.addTab(self._build_training_tab(),   "  Entrenamiento  ")
        tabs.addTab(self._build_sessions_tab(),   "  Muestras  ")
        tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(tabs)

    # ---- Grabación tab ----

    def _build_recording_tab(self) -> QWidget:
        w = QWidget()
        vlay = QVBoxLayout(w)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        # Class selector strip at top
        vlay.addWidget(self._build_class_selector())
        vlay.addWidget(self._h_sep())

        # Body: camera + skeleton (left)  ·  CSI heatmap (right)
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(12, 12, 12, 12)
        body_lay.setSpacing(12)

        self._camera  = CameraSkeletonView()
        self._heatmap = LiveHeatmap()
        body_lay.addWidget(self._camera,  stretch=1)
        body_lay.addWidget(self._heatmap, stretch=1)
        vlay.addWidget(body, stretch=1)

        vlay.addWidget(self._h_sep())
        vlay.addWidget(self._build_recording_controls())

        self._camera.start()
        return w

    def _build_class_selector(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"background: {_PANEL};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(14)

        lbl = QLabel("Clase de actividad:")
        lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lay.addWidget(lbl)

        self._class_combo = QComboBox()
        for i, name in enumerate(CLASS_NAMES):
            self._class_combo.addItem(f"  {i}  ·  {name}  ", userData=i)
        self._class_combo.setFixedWidth(220)
        self._class_combo.setFont(QFont("Consolas", 11))
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        lay.addWidget(self._class_combo)

        self._lbl_class_hint = QLabel("")
        self._lbl_class_hint.setStyleSheet(f"color: {_TEXT_SEC};")
        self._update_class_hint(0)
        lay.addWidget(self._lbl_class_hint)

        lay.addStretch()
        return bar

    def _on_class_changed(self, idx: int) -> None:
        cls = self._class_combo.currentData()
        if cls is None:
            return
        self._update_class_hint(int(cls))
        # Only auto-rename when the field still holds a previous class default;
        # don't overwrite a name the user typed by hand.
        if not hasattr(self, "_session_edit"):
            return
        current = self._session_edit.text()
        if not current or any(current.startswith(c + "_") for c in CLASS_NAMES):
            self._session_edit.setText(f"{class_name(int(cls))}_001")

    def _update_class_hint(self, idx: int) -> None:
        hints = {
            0: "Sala vacía  —  sin nadie en el espacio sensado.",
            1: "Sujeto de pie quieto, brazos relajados.",
            2: "Sujeto sentado en silla / sofá / borde de cama.",
            3: "Sujeto caminando con paso normal por el espacio.",
            4: "Sujeto tumbado (suelo, cama, simulación de caída).",
        }
        self._lbl_class_hint.setText(hints.get(idx, ""))

    def _build_recording_controls(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet(f"background: {_PANEL};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(14)

        lay.addWidget(QLabel("Sesión:"))
        self._session_edit = QLineEdit("vacío_001")
        self._session_edit.setFixedWidth(180)
        lay.addWidget(self._session_edit)

        self._btn_record = QPushButton("⏺  Grabar")
        self._btn_record.setFixedSize(120, 30)
        self._btn_record.clicked.connect(self._on_record_toggle)
        lay.addWidget(self._btn_record)

        self._lbl_samples = QLabel("Muestras: 0")
        self._lbl_samples.setStyleSheet(f"color: {_TEXT_SEC};")
        lay.addWidget(self._lbl_samples)

        self._lbl_elapsed = QLabel("Tiempo: 0 s")
        self._lbl_elapsed.setStyleSheet(f"color: {_TEXT_SEC};")
        lay.addWidget(self._lbl_elapsed)

        lay.addStretch()

        self._lbl_status = QLabel("Listo")
        self._lbl_status.setStyleSheet(f"color: {_TEXT_SEC};")
        lay.addWidget(self._lbl_status)
        return bar

    # ---- Entrenamiento tab ----

    def _build_training_tab(self) -> QWidget:
        w = QWidget()
        vlay = QVBoxLayout(w)
        vlay.setContentsMargins(20, 20, 20, 20)
        vlay.setSpacing(14)

        self._lbl_dataset = QLabel("Buscando sesiones guardadas…")
        self._lbl_dataset.setStyleSheet(f"color: {_TEXT_SEC};")
        vlay.addWidget(self._lbl_dataset)
        self._refresh_dataset_info()

        vlay.addWidget(self._h_sep())

        row = QWidget()
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(0, 0, 0, 0)

        self._btn_train = QPushButton("▶  Iniciar entrenamiento")
        self._btn_train.setFixedSize(220, 32)
        self._btn_train.clicked.connect(self._on_train)
        rlay.addWidget(self._btn_train)

        self._lbl_epoch = QLabel("")
        self._lbl_epoch.setStyleSheet(f"color: {_TEXT_SEC};")
        rlay.addWidget(self._lbl_epoch)
        rlay.addStretch()
        vlay.addWidget(row)

        self._progress = QProgressBar()
        self._progress.setRange(0, EPOCHS)
        self._progress.setValue(0)
        self._progress.setFixedHeight(18)
        vlay.addWidget(self._progress)

        self._loss_graph = LossGraph()
        vlay.addWidget(self._loss_graph, stretch=1)

        vlay.addWidget(self._h_sep())

        self._lbl_model = QLabel(self._model_status())
        self._lbl_model.setStyleSheet(f"color: {_TEXT_SEC};")
        vlay.addWidget(self._lbl_model)
        return w

    # ---- Muestras tab ----

    def _build_sessions_tab(self) -> QWidget:
        w = QWidget()
        vlay = QVBoxLayout(w)
        vlay.setContentsMargins(16, 16, 16, 16)
        vlay.setSpacing(10)

        hdr = QLabel("Sesiones grabadas")
        hdr.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        hdr.setStyleSheet(f"color: {_TEXT};")
        vlay.addWidget(hdr)

        self._session_list = QListWidget()
        self._session_list.setStyleSheet(
            f"background: {_PANEL}; color: {_TEXT}; border: 1px solid {_BORDER};"
            f" font-family: Consolas; font-size: 12px;"
        )
        self._session_list.setAlternatingRowColors(True)
        self._session_list.currentRowChanged.connect(self._on_session_selected)
        vlay.addWidget(self._session_list, stretch=1)

        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(10)

        self._btn_rename = QPushButton("Renombrar")
        self._btn_rename.setFixedHeight(30)
        self._btn_rename.setEnabled(False)
        self._btn_rename.clicked.connect(self._on_rename_session)
        btn_lay.addWidget(self._btn_rename)

        self._btn_delete = QPushButton("Eliminar")
        self._btn_delete.setFixedHeight(30)
        self._btn_delete.setEnabled(False)
        self._btn_delete.setStyleSheet(
            f"QPushButton {{ color: {_RED}; border-color: {_BORDER}; }}"
            f"QPushButton:hover {{ border-color: {_RED}; }}"
        )
        self._btn_delete.clicked.connect(self._on_delete_session)
        btn_lay.addWidget(self._btn_delete)

        btn_lay.addStretch()

        self._btn_refresh = QPushButton("↻  Actualizar")
        self._btn_refresh.setFixedHeight(30)
        self._btn_refresh.clicked.connect(self._refresh_session_list)
        btn_lay.addWidget(self._btn_refresh)
        vlay.addWidget(btn_row)

        self._lbl_session_info = QLabel("")
        self._lbl_session_info.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 11px;")
        vlay.addWidget(self._lbl_session_info)
        return w

    def _refresh_session_list(self) -> None:
        self._session_list.clear()
        self._btn_rename.setEnabled(False)
        self._btn_delete.setEnabled(False)
        self._lbl_session_info.setText("")

        sessions = self._get_sessions()
        if not sessions:
            item = QListWidgetItem("  Sin sesiones grabadas todavía.")
            item.setForeground(QColor(_TEXT_SEC))
            self._session_list.addItem(item)
            return

        per_class = [0] * NUM_CLASSES
        total = 0
        for name, n_samples, size_kb, cls_idx in sessions:
            cls_str = class_name(cls_idx) if cls_idx is not None else "?"
            item = QListWidgetItem(
                f"  {name}   —   [{cls_str}]   {n_samples} muestras   ({size_kb} KB)"
            )
            item.setData(256, name)
            self._session_list.addItem(item)
            total += n_samples
            if cls_idx is not None and 0 <= cls_idx < NUM_CLASSES:
                per_class[cls_idx] += n_samples

        counts_str = "  ".join(
            f"{CLASS_NAMES[i]}={per_class[i]}" for i in range(NUM_CLASSES)
        )
        self._lbl_session_info.setText(
            f"{len(sessions)} sesión(es)  ·  {total} muestras totales  "
            f"(~{total / 10 / 60:.1f} min)\n{counts_str}"
        )

    def _get_sessions(self) -> List[Tuple[str, int, int, Optional[int]]]:
        """List (name, n_samples, size_kb, class_idx_or_None) for each session on disk."""
        csi_files = sorted(DATA_DIR.glob("*_csi.npy"))
        result = []
        for cf in csi_files:
            name = cf.stem.replace("_csi", "")
            try:
                arr = np.load(str(cf))
                n_samples = arr.shape[0]
                del arr
            except Exception:
                n_samples = 0

            # Class is constant across the session, so the first label is enough.
            cls_idx: Optional[int] = None
            lf = DATA_DIR / f"{name}_labels.npy"
            if lf.exists():
                try:
                    labels = np.load(str(lf))
                    if labels.size > 0:
                        cls_idx = int(labels[0])
                except Exception:
                    pass

            size_kb = sum(
                f.stat().st_size for f in DATA_DIR.glob(f"{name}_*.npy")
                if f.exists()
            ) // 1024
            result.append((name, n_samples, size_kb, cls_idx))
        return result

    def _on_session_selected(self, row: int) -> None:
        item = self._session_list.item(row)
        has_data = item is not None and item.data(256) is not None
        self._btn_rename.setEnabled(has_data)
        self._btn_delete.setEnabled(has_data)

    def _selected_session_name(self) -> Optional[str]:
        item = self._session_list.currentItem()
        if item is None:
            return None
        return item.data(256)

    def _session_files(self, name: str) -> List[Path]:
        """All files belonging to a session. Includes legacy pose suffixes so
        renaming/deleting old pose-format sessions still cleans them up."""
        suffixes = ["_csi.npy", "_labels.npy", "_ts.npy",
                    "_pose.npy", "_bbox.npy"]
        return [DATA_DIR / f"{name}{s}" for s in suffixes]

    def _on_rename_session(self) -> None:
        old_name = self._selected_session_name()
        if not old_name:
            return
        new_name, ok = QInputDialog.getText(
            self, "Renombrar sesión", "Nuevo nombre:", text=old_name
        )
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        errors = []
        for old_path in self._session_files(old_name):
            if old_path.exists():
                new_path = old_path.with_name(
                    old_path.name.replace(old_name, new_name, 1)
                )
                try:
                    old_path.rename(new_path)
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            QMessageBox.warning(self, "Error", "\n".join(errors))
        self._refresh_session_list()
        self._refresh_dataset_info()

    def _on_delete_session(self) -> None:
        name = self._selected_session_name()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Eliminar sesión",
            f"¿Eliminar la sesión «{name}» y sus archivos?\nEsta acción no se puede deshacer.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        for p in self._session_files(name):
            if p.exists():
                try:
                    p.unlink()
                except Exception as exc:
                    QMessageBox.warning(self, "Error", str(exc))
        self._refresh_session_list()
        self._refresh_dataset_info()

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self._refresh_dataset_info()
        elif index == 2:
            self._refresh_session_list()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _h_sep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet(f"background: {_BORDER}; border: none;")
        return f

    def _model_status(self) -> str:
        if WEIGHTS_PATH.exists():
            return f"Modelo entrenado: {WEIGHTS_PATH}"
        return "Sin modelo entrenado aún."

    def _refresh_dataset_info(self) -> None:
        sessions = self._get_sessions()
        labelled = [s for s in sessions if s[3] is not None]
        legacy = len(sessions) - len(labelled)
        if not sessions:
            self._lbl_dataset.setText(
                f"Sin sesiones en {DATA_DIR}/  —  graba al menos una sesión primero."
            )
            return
        per_class = [0] * NUM_CLASSES
        for _, n, _, cls_idx in labelled:
            if cls_idx is not None and 0 <= cls_idx < NUM_CLASSES:
                per_class[cls_idx] += n
        counts_str = "  ".join(
            f"{CLASS_NAMES[i]}={per_class[i]}" for i in range(NUM_CLASSES)
        )
        legacy_str = f"  ·  {legacy} sesión(es) sin etiqueta (ignoradas)" if legacy else ""
        self._lbl_dataset.setText(
            f"{len(labelled)} sesión(es) etiquetada(s){legacy_str}\n{counts_str}"
        )

    # ------------------------------------------------------------------
    # Slots — recording
    # ------------------------------------------------------------------

    def _on_record_toggle(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        session = self._session_edit.text().strip() or "session"
        class_idx = int(self._class_combo.currentData() or 0)

        if (DATA_DIR / f"{session}_csi.npy").exists():
            QMessageBox.warning(
                self, "Sesión existente",
                f"Ya existe una sesión con el nombre «{session}».\nCámbiale el nombre.",
            )
            return

        self._recording = True
        self._btn_record.setText("⏹  Detener")
        self._btn_record.setStyleSheet(
            f"background: #2d1b1b; border: 1px solid {_RED}; color: {_RED};"
        )
        self._lbl_status.setText(f"Grabando · clase: {class_name(class_idx)}")
        self._lbl_status.setStyleSheet(f"color: {_RED};")
        self._session_edit.setEnabled(False)
        self._class_combo.setEnabled(False)

        self._rec_thread = RecordingThread(
            self._reader, session, class_idx, parent=None,
        )
        self._rec_thread.sample_ready.connect(self._on_sample)
        self._rec_thread.finished_sig.connect(self._on_recording_done)
        self._rec_thread.start()

    def _stop_recording(self) -> None:
        self._recording = False
        self._btn_record.setEnabled(False)
        self._lbl_status.setText("Guardando…")
        if self._rec_thread:
            self._rec_thread.request_stop()

    def _on_sample(self, n: int, elapsed: float) -> None:
        self._lbl_samples.setText(f"Muestras: {n}")
        self._lbl_elapsed.setText(f"Tiempo: {elapsed:.0f} s")

    def _on_recording_done(self, out_path) -> None:
        self._recording = False
        self._btn_record.setText("⏺  Grabar")
        self._btn_record.setStyleSheet("")
        self._btn_record.setEnabled(True)
        self._session_edit.setEnabled(True)
        self._class_combo.setEnabled(True)
        if out_path:
            self._lbl_status.setText(f"Guardado en {out_path}")
            self._lbl_status.setStyleSheet(f"color: {_GREEN};")
            # Bump the numeric suffix so the next recording doesn't clash.
            name = self._session_edit.text()
            try:
                base, num = name.rsplit("_", 1)
                self._session_edit.setText(f"{base}_{int(num)+1:03d}")
            except ValueError:
                pass
        else:
            self._lbl_status.setText("Error al guardar")
            self._lbl_status.setStyleSheet(f"color: {_RED};")
        self._refresh_dataset_info()

    def _tick_heatmap(self) -> None:
        frames = self._reader.get_frames()
        self._heatmap.update_data(frames)

    # ------------------------------------------------------------------
    # Slots — training
    # ------------------------------------------------------------------

    def _on_train(self) -> None:
        self._refresh_dataset_info()
        sessions = list(DATA_DIR.glob("*_labels.npy"))
        if not sessions:
            self._lbl_epoch.setText("Sin sesiones etiquetadas — graba primero.")
            return

        self._btn_train.setEnabled(False)
        self._progress.setValue(0)
        self._loss_graph.reset()
        self._lbl_epoch.setText("Entrenando…")
        self._lbl_model.setText("Entrenando modelo…")

        self._train_thread = TrainingThread(parent=None)
        self._train_thread.progress_sig.connect(self._on_train_progress)
        self._train_thread.finished_sig.connect(self._on_train_done)
        self._train_thread.start()

    def _on_train_progress(self, epoch: int, total: int, loss: float) -> None:
        self._progress.setValue(epoch)
        self._lbl_epoch.setText(f"Época {epoch}/{total}  ·  val loss {loss:.4f}")
        self._loss_graph.add_loss(loss)

    def _on_train_done(self, ok: bool) -> None:
        self._btn_train.setEnabled(True)
        if ok:
            self._lbl_epoch.setText("Entrenamiento completado.")
            self._lbl_model.setText(self._model_status())
            self._lbl_model.setStyleSheet(f"color: {_GREEN};")
        else:
            self._lbl_epoch.setText("Error durante el entrenamiento.")
            self._lbl_model.setStyleSheet(f"color: {_RED};")

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._heatmap_timer.stop()
        if hasattr(self, "_camera"):
            self._camera.stop()
        if self._rec_thread and self._rec_thread.isRunning():
            self._rec_thread.request_stop()
            self._rec_thread.wait(3000)
        if self._train_thread and self._train_thread.isRunning():
            self._train_thread.terminate()
        super().closeEvent(event)
        self.finished.emit()

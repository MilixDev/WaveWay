import re
import time
import logging
import threading
from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np
import serial

logger = logging.getLogger(__name__)

# (unix_timestamp, per-subcarrier amplitudes)
Frame = Tuple[float, List[float]]


# 802.11n HT20 LLTF subcarrier layout (64 FFT bins, ESP32 default CSI dump).
# Drops: left guards 0..5, right guards 59..63, DC at 32, pilots at 11/25/39/53
# (SC indices -21, -7, +7, +21). What remains: 48 usable data subcarriers.
_HT20_NULL_IDX: Set[int] = set(range(0, 6)) | {32} | set(range(59, 64))
_HT20_PILOT_IDX: Set[int] = {11, 25, 39, 53}
_HT20_DATA_IDX: List[int] = [
    i for i in range(64)
    if i not in _HT20_NULL_IDX and i not in _HT20_PILOT_IDX
]


class CSIReader:
    """Reads CSI data from an ESP32 (classic, 2.4 GHz HT20) over serial.

    Expected line format:
        CSI_DATA,<seq>,<mac>,...,"[I0,Q0,I1,Q1,...,IN,QN]"

    Per subcarrier: amplitude = sqrt(I^2 + Q^2). Frames are then masked
    to keep only the 48 data subcarriers when input is 64 (LLTF) or
    128 (LLTF+HT-LTF concatenated) raw bins.
    """

    BUFFER_SIZE: int = 3000          # ~30 s @ 100 Hz → FFT res ≈ 0.033 Hz (~2 rpm)
    _IQ_PATTERN: re.Pattern = re.compile(r'"?\[(-?\d+(?:,\s*-?\d+)*)\]"?')
    _RATE_LOG_EVERY: int = 200       # log measured CSI rate every N frames

    def __init__(self, port: str = "COM6", baudrate: int = 921600) -> None:
        self.port = port
        self.baudrate = baudrate
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._serial: Optional[serial.Serial] = None
        self._raw_length_warned: bool = False
        self._frame_counter: int = 0
        self._rate_window_start: float = 0.0
        self._measured_rate_hz: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True while the serial port is open and actively reading."""
        return self._connected

    @property
    def running(self) -> bool:
        """True between start() and stop()."""
        return self._running

    @property
    def measured_rate_hz(self) -> float:
        """Most recently measured CSI frame rate, or 0 before first window."""
        return self._measured_rate_hz

    def start(self) -> None:
        """Spawn the background reading thread (no-op if already running)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CSIReader"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the reading thread and close the serial port."""
        self._running = False
        self._connected = False
        ser, self._serial = self._serial, None
        if ser:
            try:
                ser.close()
            except Exception:
                pass

    def get_frames(self) -> List[Frame]:
        """Return a snapshot of all buffered (timestamp, amplitudes) frames."""
        with self._lock:
            return list(self._buffer)

    def get_latest(self) -> List[float]:
        """Return the most recent amplitude frame, or [] if the buffer is empty."""
        with self._lock:
            if self._buffer:
                return list(self._buffer[-1][1])
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Try to open the serial port. Returns True on success."""
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self._connected = True
            logger.info("Connected to %s @ %d baud", self.port, self.baudrate)
            return True
        except serial.SerialException as exc:
            logger.warning("Cannot open %s: %s", self.port, exc)
            self._connected = False
            return False

    def _parse_line(self, line: str) -> Optional[List[float]]:
        """Parse one CSI_DATA line into a list of per-subcarrier amplitudes."""
        if not line.startswith("CSI_DATA"):
            return None
        match = self._IQ_PATTERN.search(line)
        if not match:
            return None
        try:
            values = [int(v.strip()) for v in match.group(1).split(",")]
        except ValueError:
            return None
        if len(values) < 4:
            return None
        amplitudes: List[float] = []
        for i in range(0, len(values) - 1, 2):
            I_val = values[i]
            Q_val = values[i + 1]
            amplitudes.append(float(np.sqrt(I_val * I_val + Q_val * Q_val)))
        if not amplitudes:
            return None
        return self._apply_subcarrier_mask(amplitudes)

    def _apply_subcarrier_mask(self, amps: List[float]) -> List[float]:
        """Drop DC, guard bands, and pilots when the raw frame matches HT20 layout.

        Returns the original list unchanged for unknown lengths, logging a
        one-shot warning so the firmware/SC layout can be diagnosed.
        """
        n = len(amps)
        if n == 64 or n == 128:
            # n==128: LLTF + HT-LTF concatenated; keep just the LLTF half.
            return [amps[i] for i in _HT20_DATA_IDX]
        if not self._raw_length_warned:
            logger.warning(
                "CSI frame has %d subcarriers (expected 64 or 128). "
                "Passing through unfiltered — verify firmware SC layout.", n,
            )
            self._raw_length_warned = True
        return amps

    def _run(self) -> None:
        """Main loop: connect → read lines → reconnect on failure."""
        while self._running:
            if not self._connect():
                time.sleep(2.0)
                continue
            ser = self._serial  # local ref — avoids race with stop() setting self._serial to None
            try:
                while self._running and ser and ser.is_open:
                    try:
                        raw = ser.readline()
                    except (serial.SerialException, TypeError, OSError,
                            AttributeError, ValueError) as exc:
                        # AttributeError/ValueError: pyserial Win32 backend
                        # raises these when the port is closed mid-readline().
                        if self._running:
                            logger.warning("Read error on %s: %s", self.port, exc)
                        self._connected = False
                        break
                    if not raw:
                        continue
                    try:
                        line = raw.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue
                    amplitudes = self._parse_line(line)
                    if amplitudes is not None:
                        now = time.time()
                        with self._lock:
                            self._buffer.append((now, amplitudes))
                        self._frame_counter += 1
                        if self._frame_counter % self._RATE_LOG_EVERY == 0:
                            if self._rate_window_start > 0.0:
                                dt = now - self._rate_window_start
                                if dt > 0.0:
                                    self._measured_rate_hz = self._RATE_LOG_EVERY / dt
                                    logger.info(
                                        "CSI rate: %.1f Hz  (subcarriers/frame: %d)",
                                        self._measured_rate_hz, len(amplitudes),
                                    )
                            self._rate_window_start = now
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
                self._connected = False
            if self._running:
                logger.info("Disconnected from %s — retrying in 2 s", self.port)
                time.sleep(2.0)

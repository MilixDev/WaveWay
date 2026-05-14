import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .csi_reader import CSIReader, Frame
from .models import Detection
from .vitals import (
    BREATH_BAND_HZ,
    HEART_BAND_HZ,
    VitalEstimate,
    estimate_vital,
)

logger = logging.getLogger(__name__)


class Detector:
    """Process CSI frames into presence, activity, position, and vital signs.

    Presence detection blends two complementary paths:
      1. Temporal variance  — off-axis motion (multipath changes).
      2. Amplitude attenuation — on-LOS blocking (signal drops when a person
         is on the direct path between ESP32 and router).
    Both are normalised so a single threshold applies to their max.

    Vital signs are estimated by core.vitals.estimate_vital() using
    concentration-weighted PSD across subcarriers, throttled to once a
    second (the FFT pipeline is too heavy to run at every UI tick).
    Heart-rate estimation is opt-in via self.heart_enabled.
    """

    MIN_FRAMES_PRESENCE: int = 3
    MIN_FRAMES_VITALS:   int = 1500   # ≥15 s @ 100 Hz → ~0.067 Hz res (~4 rpm) for breathing FFT
    MIN_FRAMES_HEART:    int = 2000   # ≥20 s @ 100 Hz — heart band is wider, needs more samples

    # Vital re-computation cadence (seconds). Detect() runs at ~10 Hz from UI;
    # estimate_vital is ~50–100 ms of CPU, so we throttle it.
    VITAL_REFRESH_S: float = 1.0

    def __init__(
        self, reader: CSIReader, presence_threshold: float = 0.02
    ) -> None:
        self.reader = reader
        self.presence_threshold = presence_threshold
        # "los"      → attenuation only (person must block the direct path)
        # "ambiente"  → max(variance, attenuation×0.8)  (any room motion)
        self.detection_mode: str = "ambiente"
        # Heart-rate estimation is experimental and opt-in
        self.heart_enabled: bool = False
        # EMA state
        self._sm_breathing: float = 0.0
        self._sm_heart: float = 0.0
        self._sm_activity: float = 0.0
        self._sm_distance: float = 0.5
        self._sm_lateral: float = 0.0
        self._sm_confidence: float = 0.0
        # Latest vital estimates (kept between recomputes for throttling)
        self._last_breath: VitalEstimate = VitalEstimate(0.0, 0.0, 0.0, 0.0)
        self._last_heart:  VitalEstimate = VitalEstimate(0.0, 0.0, 0.0, 0.0)
        self._last_vital_ts: float = 0.0
        # Diagnostics (read by UI)
        self._last_variance: float = 0.0
        self._last_attenuation: float = 0.0
        # Calibration
        self._calibrating: bool = False
        self._calibrated: bool = False
        self._cal_calls: int = 0
        self._cal_target: int = 50          # ticks × 100 ms = 5 s
        self._cal_variances: List[float] = []
        self._cal_amp_frames: List[List[float]] = []
        self._baseline_amps: Optional[np.ndarray] = None   # empty-room amplitude

    # ------------------------------------------------------------------
    # Calibration API
    # ------------------------------------------------------------------

    def start_calibration(self) -> None:
        """Begin 5-second baseline capture. Room must be empty."""
        self._calibrating = True
        self._calibrated = False
        self._cal_calls = 0
        self._cal_variances = []
        self._cal_amp_frames = []
        logger.info("Calibration started (%d ticks)", self._cal_target)

    @property
    def calibrating(self) -> bool:
        return self._calibrating

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_progress(self) -> float:
        if self._calibrating:
            return min(1.0, self._cal_calls / self._cal_target)
        return 1.0 if self._calibrated else 0.0

    @property
    def live_variance(self) -> float:
        """Most recent combined variance/attenuation value (read by UI)."""
        return self._last_variance

    # ------------------------------------------------------------------
    # Detection API
    # ------------------------------------------------------------------

    def detect(self) -> Detection:
        """Run the full detection pipeline and return a smoothed Detection."""
        frames: List[Frame] = self.reader.get_frames()

        if len(frames) < self.MIN_FRAMES_PRESENCE:
            self._sm_activity = self._ema(self._sm_activity, 0.0, alpha=0.15)
            self._sm_confidence = self._ema(self._sm_confidence, 0.0, alpha=0.15)
            self._sm_breathing *= 0.93
            self._sm_heart *= 0.93
            return Detection(
                present=False,
                confidence=0.0,
                activity_level=self._sm_activity,
                velocity=0.0,
                breathing_rate=0.0,
                breathing_confidence=0.0,
                heart_rate=0.0,
                heart_confidence=0.0,
                distance_ratio=self._sm_distance,
                lateral_offset=self._sm_lateral,
                distance_zone="—",
            )

        timestamps = [f[0] for f in frames]
        amp_matrix = [f[1] for f in frames]
        arr = np.array(amp_matrix, dtype=float)

        # ---- Calibration sample collection ----
        if self._calibrating:
            v_now = self._variance_from_arr(arr)
            self._cal_variances.append(v_now)
            if amp_matrix:
                self._cal_amp_frames.append(amp_matrix[-1])
            self._cal_calls += 1
            if self._cal_calls >= self._cal_target:
                self._finish_calibration()

        # ---- Dual detection ----
        variance = self._variance_from_arr(arr)        # off-axis / motion
        attenuation = self._attenuation_from_arr(arr)  # on-LOS blockage

        if self.detection_mode == "los":
            combined = float(attenuation)
        else:
            combined = float(max(variance, attenuation * 0.8))

        self._last_variance = combined
        self._last_attenuation = attenuation

        present = combined > self.presence_threshold
        raw_confidence = float(np.clip(combined / (self.presence_threshold * 3.0), 0.0, 1.0))
        velocity = self._frame_delta_arr(arr)

        raw_activity = float(np.clip(combined / (self.presence_threshold * 5.0), 0.0, 1.0))
        self._sm_activity = self._ema(self._sm_activity, raw_activity, alpha=0.3)
        self._sm_confidence = self._ema(
            self._sm_confidence,
            raw_confidence if present else 0.0,
            alpha=0.3,
        )

        # ---- Vital signs (breathing + experimental heart rate) ----
        breath_now, heart_now = self._update_vitals(arr, timestamps, present)

        # ---- Position ----
        distance_ratio, lateral_offset = self._estimate_position_arr(arr)
        self._sm_distance = self._ema(self._sm_distance, distance_ratio, alpha=0.1)
        self._sm_lateral = self._ema(self._sm_lateral, lateral_offset, alpha=0.1)

        if present:
            d = self._sm_distance
            zone = "Cerca" if d < 0.33 else ("Medio" if d < 0.67 else "Lejos")
        else:
            zone = "—"

        return Detection(
            present=present,
            confidence=self._sm_confidence,
            activity_level=self._sm_activity,
            velocity=velocity,
            breathing_rate=breath_now,
            breathing_confidence=self._last_breath.confidence,
            heart_rate=heart_now,
            heart_confidence=self._last_heart.confidence,
            distance_ratio=self._sm_distance,
            lateral_offset=self._sm_lateral,
            distance_zone=zone,
        )

    # ------------------------------------------------------------------
    # Private — calibration
    # ------------------------------------------------------------------

    def _finish_calibration(self) -> None:
        if len(self._cal_variances) >= 5:
            v_arr = np.array(self._cal_variances)
            mean_v = float(v_arr.mean())
            std_v = float(v_arr.std())
            new_thresh = float(np.clip(mean_v + 4.0 * std_v, 0.002, 1.0))
            self.presence_threshold = new_thresh
            logger.info("Calibration: baseline=%.4f  threshold=%.4f", mean_v, new_thresh)

        if self._cal_amp_frames:
            cal_arr = np.array(self._cal_amp_frames, dtype=float)
            self._baseline_amps = cal_arr.mean(axis=0)   # per-subcarrier mean
            logger.info("Stored baseline amplitudes shape=%s", self._baseline_amps.shape)

        self._calibrating = False
        self._calibrated = True
        self._cal_calls = 0
        self._cal_variances.clear()
        self._cal_amp_frames.clear()

    # ------------------------------------------------------------------
    # Private — signal processing
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(prev: float, new: float, alpha: float = 0.2) -> float:
        return alpha * new + (1.0 - alpha) * prev

    def _variance_from_arr(self, arr: np.ndarray) -> float:
        """Coefficient of variation of consecutive frame deltas.

        Mirrors what the CSI heatmap shows: high when frames differ
        significantly relative to the local amplitude level.
        """
        if arr.shape[0] < 2:
            return 0.0
        try:
            diffs = np.abs(np.diff(arr, axis=0))
            amp_mean = arr.mean(axis=0) + 1e-9
            return float((diffs.mean(axis=0) / amp_mean).mean())
        except Exception:
            return 0.0

    def _attenuation_from_arr(self, arr: np.ndarray) -> float:
        """Detect on-LOS presence via amplitude drop below baseline.

        When a person stands between ESP32 and router they attenuate
        the direct path signal.  This is complementary to variance-based
        detection which works for off-axis / moving targets.
        """
        try:
            if self._baseline_amps is not None:
                baseline = self._baseline_amps
            else:
                # Estimate: 80th-percentile of buffer ≈ empty-room level
                baseline = np.percentile(arr, 80, axis=0)

            current = arr[-1]
            n = min(len(current), len(baseline))
            if n == 0:
                return 0.0
            drop = (baseline[:n] - current[:n]) / (baseline[:n] + 1e-9)
            return float(np.maximum(drop, 0).mean())
        except Exception:
            return 0.0

    def _frame_delta_arr(self, arr: np.ndarray) -> float:
        if arr.shape[0] < 2:
            return 0.0
        try:
            return float(np.abs(arr[-1] - arr[-2]).mean())
        except Exception:
            return 0.0

    def _sample_rate(self, timestamps: List[float]) -> float:
        if len(timestamps) < 2:
            return 10.0
        duration = timestamps[-1] - timestamps[0]
        return (len(timestamps) - 1) / duration if duration > 0 else 10.0

    def _update_vitals(
        self, arr: np.ndarray, timestamps: List[float], present: bool
    ) -> Tuple[float, float]:
        """Refresh breathing (+ heart if enabled) estimates, throttled and smoothed.

        Returns (breathing_rate_rpm, heart_rate_bpm). Either is 0 when the
        signal isn't ready or confidence is below the gate.
        """
        # When no person, let the smoothed values decay so the UI numbers fade.
        if not present:
            self._sm_breathing *= 0.96
            self._sm_heart *= 0.96
            return 0.0, 0.0

        now = time.time()
        need_recompute = (now - self._last_vital_ts) >= self.VITAL_REFRESH_S
        sample_rate = self._sample_rate(timestamps)

        # Breathing
        if need_recompute and arr.shape[0] >= self.MIN_FRAMES_VITALS:
            self._last_breath = estimate_vital(
                arr, sample_rate, BREATH_BAND_HZ, min_snr=3.0
            )
        if self._last_breath.confidence > 0.0:
            self._sm_breathing = self._ema(
                self._sm_breathing, self._last_breath.rate_bpm, alpha=0.15
            )
        else:
            self._sm_breathing *= 0.97

        # Heart (experimental, opt-in)
        if self.heart_enabled:
            if need_recompute and arr.shape[0] >= self.MIN_FRAMES_HEART:
                # Reject breathing harmonics using the RAW breathing peak, not
                # the confidence-gated rate — even a low-SNR breathing peak is
                # enough to mark out the 5th/6th harmonic that otherwise
                # dominates the heart band.
                breath_hz = (
                    self._last_breath.raw_rate_bpm / 60.0
                    if self._last_breath.raw_rate_bpm > 0.0 else None
                )
                self._last_heart = estimate_vital(
                    arr, sample_rate, HEART_BAND_HZ,
                    min_snr=3.0,
                    reject_harmonics_of_hz=breath_hz,
                )
            if self._last_heart.confidence > 0.0:
                self._sm_heart = self._ema(
                    self._sm_heart, self._last_heart.rate_bpm, alpha=0.12
                )
            else:
                self._sm_heart *= 0.97
        else:
            self._last_heart = VitalEstimate(0.0, 0.0, 0.0, 0.0)
            self._sm_heart *= 0.96

        if need_recompute:
            self._last_vital_ts = now

        # Gate near-zero noise out of the UI numbers
        br_out = self._sm_breathing if self._sm_breathing > 2.0 else 0.0
        hr_out = self._sm_heart if self._sm_heart > 30.0 else 0.0
        return br_out, hr_out

    def _estimate_position_arr(self, arr: np.ndarray) -> Tuple[float, float]:
        try:
            latest = arr[-1]
            n = len(latest)
            if n < 2:
                return 0.5, 0.0
            total = latest.sum() + 1e-9
            centroid = float(np.dot(np.arange(n, dtype=float), latest) / total)
            dist = float(np.clip(centroid / n, 0.0, 1.0))
            mid = n // 2
            lat = float(np.clip((latest[mid:].sum() - latest[:mid].sum()) / total, -1.0, 1.0))
            return dist, lat
        except Exception:
            return 0.5, 0.0

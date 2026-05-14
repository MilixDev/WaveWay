"""CSI vital-sign estimation: breathing rate and (experimental) heart rate.

Pipeline per call to estimate_vital():
    1. Hampel filter per subcarrier            → removes impulse noise / corrupt frames
    2. FFT-domain bandpass per subcarrier      → isolates target frequency band
    3. PCA (PC1)                               → concentrates vital signal across SCs
    4. Welch periodogram                       → low-variance PSD estimate
    5. In-band peak + SNR vs out-of-band noise → rate + confidence

Pure numpy. No scipy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Default bands
BREATH_BAND_HZ: Tuple[float, float] = (0.15, 0.50)   #   9–30 rpm
HEART_BAND_HZ:  Tuple[float, float] = (0.80, 2.50)   #  48–150 bpm


@dataclass
class VitalEstimate:
    """Result of one vital-sign estimation."""
    rate_bpm:     float    # breaths-per-minute or beats-per-minute (gated by confidence)
    raw_rate_bpm: float    # peak freq × 60, regardless of confidence (for internal use)
    confidence:   float    # 0..1 from in-band peak SNR
    snr:          float    # raw peak / noise ratio


# ----------------------------------------------------------------------
# 1. Hampel filter — outlier rejection
# ----------------------------------------------------------------------

def hampel_1d(x: np.ndarray, k: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    """Vectorised Hampel filter: replace outliers with local median.

    For each sample i, compute median and MAD over a centred window of
    half-width k. If |x_i - median| > n_sigma * 1.4826 * MAD, replace x_i
    with the median.
    """
    n = len(x)
    if n < 2 * k + 1:
        return x.copy()
    pad = np.pad(x, k, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
    med = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - med[:, None]), axis=1) * 1.4826
    out = x.copy()
    valid = mad > 1e-9
    deviation = np.abs(x - med)
    repl = valid & (deviation > n_sigma * mad)
    out[repl] = med[repl]
    return out


def hampel_columns(X: np.ndarray, k: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    """Apply hampel_1d to each column of (N, M)."""
    return np.apply_along_axis(hampel_1d, 0, X, k=k, n_sigma=n_sigma)


# ----------------------------------------------------------------------
# 2. Welch periodogram (Hann window, 50% overlap)
# ----------------------------------------------------------------------

def pick_nperseg(
    n_samples: int, sample_rate: float, band_width_hz: float,
    bins_in_band: int = 10, min_seg: int = 64,
) -> int:
    """Choose a Welch segment size that puts ~bins_in_band frequency bins in the band.

    Frequency resolution is sample_rate / nperseg. To get K bins inside a band
    of width W, we need nperseg ≈ sample_rate * K / W. Clamped to the available
    sample count so welch_psd produces at least one segment.
    """
    if band_width_hz <= 0 or sample_rate <= 0:
        return max(min_seg, min(n_samples, 512))
    target = int(round(sample_rate * bins_in_band / band_width_hz))
    return max(min_seg, min(n_samples, target))


def welch_psd(
    x: np.ndarray, sample_rate: float, nperseg: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD. Returns (freqs, psd)."""
    n = len(x)
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    nperseg = min(nperseg, n)
    if nperseg < 8:
        nperseg = n
    noverlap = nperseg // 2
    step = max(1, nperseg - noverlap)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / max(1, nperseg - 1))
    win_norm = (win ** 2).sum() * sample_rate
    psds = []
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg] - x[start:start + nperseg].mean()
        S = np.abs(np.fft.rfft(seg * win)) ** 2
        psds.append(S)
    if not psds:
        return np.array([0.0]), np.array([0.0])
    psd = np.mean(psds, axis=0) / max(win_norm, 1e-12)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    return freqs, psd


def welch_psd_batch(
    X: np.ndarray, sample_rate: float, nperseg: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD on every column of X simultaneously.

    Returns (freqs (F,), psd (F, M)). FFTs are batched along axis=0 of the
    windowed segments so this is roughly M× faster than calling welch_psd
    per subcarrier in a Python loop.
    """
    n, m = X.shape
    nperseg = min(max(8, nperseg), n)
    nf = nperseg // 2 + 1
    noverlap = nperseg // 2
    step = max(1, nperseg - noverlap)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / max(1, nperseg - 1))
    win_norm = (win ** 2).sum() * sample_rate

    psds = []
    for start in range(0, n - nperseg + 1, step):
        seg = X[start:start + nperseg, :]
        seg = seg - seg.mean(axis=0, keepdims=True)
        S = np.abs(np.fft.rfft(seg * win[:, None], axis=0)) ** 2
        psds.append(S)
    if not psds:
        return np.zeros(nf), np.zeros((nf, m))

    psd = np.mean(psds, axis=0) / max(win_norm, 1e-12)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    return freqs, psd


def concentration_weighted_psd(
    arr: np.ndarray, sample_rate: float, low: float, high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine per-subcarrier PSDs weighted by in-band energy concentration.

    Per subcarrier: weight = (in_band_power / total_power) ** 2.
    SCs whose spectrum is dominated by the target band contribute strongly;
    SCs dominated by drift or out-of-band noise contribute almost nothing.

    Empirically more robust than PCA-on-bandpassed-data when the vital
    signal isn't the dominant variance direction (which is common at
    2.4 GHz with a single antenna).
    """
    n_samples = arr.shape[0]
    nperseg = pick_nperseg(n_samples, sample_rate, high - low, bins_in_band=10)
    freqs, psd_mat = welch_psd_batch(arr, sample_rate, nperseg)

    in_band = (freqs >= low) & (freqs <= high)
    if not in_band.any():
        return freqs, np.zeros_like(freqs)

    total = psd_mat.sum(axis=0)                          # (M,)
    in_band_pow = psd_mat[in_band, :].sum(axis=0)        # (M,)
    safe = total > 1e-12
    weights = np.zeros_like(total)
    weights[safe] = (in_band_pow[safe] / total[safe]) ** 2

    if weights.sum() < 1e-9:
        # No subcarrier shows meaningful in-band concentration — fall back
        # to plain average so we still report SOMETHING (low SNR will gate it).
        return freqs, psd_mat.mean(axis=1)
    weights /= weights.sum()
    return freqs, psd_mat @ weights                      # (F,)


# ----------------------------------------------------------------------
# 5. Peak / SNR in band
# ----------------------------------------------------------------------

def peak_snr_in_band(
    freqs: np.ndarray, psd: np.ndarray, low: float, high: float,
    excl_bins: int = 2,
) -> Tuple[float, float]:
    """Find the peak in [low, high] Hz and its SNR vs the in-band noise floor.

    The noise floor is the median of the PSD *within* the band, excluding
    ±excl_bins around the peak. Measuring noise inside the band (rather than
    outside) is critical when the signal has already been bandpass-filtered:
    out-of-band PSD is ~0 by construction, which would otherwise saturate SNR.

    Returns (peak_freq_hz, snr). (0, 0) when the band is empty.
    """
    if freqs.size < 2:
        return 0.0, 0.0
    in_band = (freqs >= low) & (freqs <= high)
    if not in_band.any():
        return 0.0, 0.0
    band_psd = psd[in_band]
    band_freqs = freqs[in_band]
    peak_local = int(np.argmax(band_psd))
    peak_power = float(band_psd[peak_local])
    peak_freq = float(band_freqs[peak_local])
    # In-band noise floor — median excluding ±excl_bins around the peak.
    keep = np.ones(band_psd.shape[0], dtype=bool)
    lo = max(0, peak_local - excl_bins)
    hi = min(band_psd.shape[0], peak_local + excl_bins + 1)
    keep[lo:hi] = False
    if keep.any():
        noise = float(np.median(band_psd[keep]))
    else:
        noise = peak_power
    snr = peak_power / (noise + 1e-12)
    return peak_freq, snr


def reject_harmonics(
    freqs: np.ndarray, psd: np.ndarray, fundamental_hz: float,
    n_harmonics: int = 7, width_hz: float = 0.05,
) -> np.ndarray:
    """Zero PSD bins within ±width_hz of integer multiples of fundamental_hz.

    Used to suppress breathing harmonics before heart-rate peak search.
    Without this, the 5th–7th harmonics of breathing land squarely in the
    heart-rate band (≈11 rpm × 5..6 = 55..66 bpm) and are picked as
    "heart rate".
    """
    if fundamental_hz <= 0 or freqs.size < 2:
        return psd
    out = psd.copy()
    for n in range(2, n_harmonics + 1):
        f_harm = n * fundamental_hz
        mask = np.abs(freqs - f_harm) <= width_hz
        out[mask] = 0.0
    return out


def snr_to_confidence(snr: float, min_snr: float = 3.0, sat_snr: float = 15.0) -> float:
    """Map raw SNR ratio to a [0, 1] confidence score.

    Below min_snr: zero (peak indistinguishable from noise).
    At sat_snr or above: one (strong peak).
    """
    if snr <= min_snr:
        return 0.0
    if snr >= sat_snr:
        return 1.0
    return float((snr - min_snr) / (sat_snr - min_snr))


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------

def estimate_vital(
    csi_amps: np.ndarray,
    sample_rate: float,
    band_hz: Tuple[float, float],
    min_snr: float = 3.0,
    reject_harmonics_of_hz: Optional[float] = None,
) -> VitalEstimate:
    """Estimate one vital sign from a buffer of CSI amplitudes.

    csi_amps: float ndarray (N samples, M subcarriers)
    sample_rate: measured CSI rate in Hz
    band_hz: (low, high) frequency band to search for the rate peak
    reject_harmonics_of_hz: when set, zero PSD bins around integer multiples
        of this frequency. Use this when estimating heart rate, passing the
        breathing fundamental so its harmonics don't masquerade as heartbeat.

    Welch nperseg is chosen automatically so the band always contains
    ~10 frequency bins, regardless of sample_rate / band width.

    Returns a VitalEstimate. rate_bpm is 0 when confidence is 0.
    """
    low, high = band_hz
    if csi_amps.ndim != 2 or csi_amps.shape[0] < 32 or sample_rate < 4 * high:
        return VitalEstimate(rate_bpm=0.0, raw_rate_bpm=0.0, confidence=0.0, snr=0.0)

    arr = np.asarray(csi_amps, dtype=np.float32)

    # 1. Despike per subcarrier
    arr = hampel_columns(arr, k=7, n_sigma=3.0)
    # 2. Combine subcarriers by in-band energy concentration → single PSD
    freqs, psd = concentration_weighted_psd(arr, sample_rate, low, high)
    # 3. Optional: suppress breathing harmonics before heart-rate peak search
    if reject_harmonics_of_hz is not None and reject_harmonics_of_hz > 0:
        psd = reject_harmonics(freqs, psd, reject_harmonics_of_hz)
    # 4. Peak SNR using in-band noise floor (excludes peak ±2 bins)
    peak_freq, snr = peak_snr_in_band(freqs, psd, low, high)

    conf = snr_to_confidence(snr, min_snr=min_snr, sat_snr=min_snr * 5.0)
    raw_rate_bpm = peak_freq * 60.0
    rate_bpm = raw_rate_bpm if conf > 0.0 else 0.0
    return VitalEstimate(
        rate_bpm=rate_bpm, raw_rate_bpm=raw_rate_bpm,
        confidence=conf, snr=float(snr),
    )

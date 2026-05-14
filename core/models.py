from dataclasses import dataclass, field
import time


@dataclass
class Detection:
    """Result of one CSI detection cycle.

    Attributes:
        present:               Person detected in the Fresnel zone.
        confidence:            Smoothed detection confidence [0, 1].
        activity_level:        Normalised motion intensity [0, 1].
        velocity:              Raw frame-to-frame amplitude delta.
        breathing_rate:        Smoothed breathing rate in rpm; 0 if unavailable.
        breathing_confidence:  Confidence of the breathing estimate [0, 1].
        heart_rate:            Smoothed heart rate in bpm; 0 if unavailable or disabled.
        heart_confidence:      Confidence of the heart-rate estimate [0, 1].
        distance_ratio:        Smoothed LOS position [0 = ESP32 side, 1 = router side].
        lateral_offset:        Smoothed lateral offset [-1, 1].
        distance_zone:         Coarse zone label: "Cerca" | "Medio" | "Lejos" | "—".
        timestamp:             Unix time of this detection.
    """

    present: bool
    confidence: float
    activity_level: float
    velocity: float
    breathing_rate: float
    breathing_confidence: float
    heart_rate: float
    heart_confidence: float
    distance_ratio: float
    lateral_offset: float
    distance_zone: str
    timestamp: float = field(default_factory=time.time)

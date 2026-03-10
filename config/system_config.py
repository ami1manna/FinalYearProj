"""
config/system_config.py
========================
Every tunable system constant lives here.

Change values in this file — never hard-code magic numbers in module files.
"""

from dataclasses import dataclass, field


@dataclass
class ControllerConfig:
    """
    Gain-scheduled Skyhook controller parameters.

    Damping range
    -------------
    c_min  : softest setting — comfortable on flat roads
    c_max  : stiffest setting — used for CRITICAL potholes at high speed

    Preview timing (total actuator delay = latency + rise_time + safety_margin)
    -------------
    actuator_latency : hardware response delay (s)
    rise_time        : time for actuator to reach target damping (s)
    safety_margin    : extra buffer to ensure damping is ready before impact (s)
    ramp_window      : how far ahead (in seconds) to start ramping damping up (s)
    """

    # Damping coefficient bounds (N·s/m)
    c_min: float = 800.0
    c_max: float = 4000.0

    # Speed normalisation reference (~90 km/h)
    v_max: float = 25.0

    # Actuator timing (seconds)
    actuator_latency: float = 0.020
    rise_time: float = 0.030
    safety_margin: float = 0.050
    ramp_window: float = 0.300

    # Target damping ratio for sprung mass
    zeta_target: float = 0.7

    @property
    def total_delay(self) -> float:
        """Total actuator delay: latency + rise_time + safety_margin (seconds)."""
        return self.actuator_latency + self.rise_time + self.safety_margin


@dataclass
class SystemConfig:
    """
    Pipeline-level operational parameters.

    detection_conf   : YOLO minimum confidence threshold (0–1)
    max_potholes     : max detections processed per frame (performance limit)
    depth_interval   : run MiDaS every N frames (higher = faster, less accurate)
    speed_buffer_len : frames to median-smooth speed over
    score_buffer_len : frames to median-smooth severity score over
    resize_width     : all frames are resized to this width before processing
    resize_height    : all frames are resized to this height before processing
    """

    # Detection
    detection_conf: float = 0.4
    max_potholes: int = 3

    # Depth estimation (MiDaS)
    depth_interval: int = 15          # frames between full MiDaS calls
    use_real_depth: bool = False       # set True to enable MiDaS (requires GPU for realtime)

    # Smoothing buffers
    speed_buffer_len: int = 10
    score_buffer_len: int = 7

    # Frame resize target
    resize_width: int = 640
    resize_height: int = 384

    # Severity depth thresholds (for relative-depth → score mapping)
    depth_score_min: float = 0.02
    depth_score_max: float = 0.18

"""
Preview-Based Gain-Scheduled Skyhook Suspension Controller
===========================================================
Implements semi-active suspension control with road preview.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class RoadSeverity(Enum):
    """Road disturbance severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SuspensionState:
    """Suspension system state at one corner."""
    z_s: float  # sprung mass displacement (m)
    z_s_dot: float  # sprung mass velocity (m/s)
    z_u: float  # unsprung mass displacement (m)
    z_u_dot: float  # unsprung mass velocity (m/s)


@dataclass
class ControllerConfig:
    """Gain-scheduled Skyhook controller configuration."""
    # Damping coefficient bounds
    c_min: float = 800.0  # N·s/m (soft damping)
    c_max: float = 4000.0  # N·s/m (hard damping)
    
    # Speed normalization (for speed-dependent gain)
    v_max: float = 25.0  # m/s (~90 km/h)
    
    # Preview timing
    actuator_latency: float = 0.020  # seconds (20ms)
    rise_time: float = 0.030  # seconds (30ms)
    safety_margin: float = 0.050  # seconds (50ms)
    ramp_window: float = 0.300  # seconds (300ms)
    
    # Skyhook damping ratio
    zeta_target: float = 0.7  # target damping ratio


class GainScheduledSkyhookController:
    """
    Preview-based gain-scheduled Skyhook controller.
    
    Adjusts damping coefficient based on:
    1. Pothole severity (from detection)
    2. Vehicle speed
    3. Time-to-impact (preview)
    """
    
    def __init__(self, config=None):
        """
        Initialize controller.
        
        Parameters:
        -----------
        config : ControllerConfig, optional
            Controller configuration
        """
        self.config = config if config else ControllerConfig()
        
        # State history for preview
        self.upcoming_events = []  # list of (time, severity, distance)
        
    def schedule_event(self, detection_time, severity_score, distance_m):
        """
        Schedule upcoming pothole event for preview control.
        
        Parameters:
        -----------
        detection_time : float
            Time of detection (seconds since start)
        severity_score : int
            Pothole severity score (0-100)
        distance_m : float
            Distance to pothole (meters)
        """
        event = {
            'time': detection_time,
            'severity': severity_score,
            'distance': distance_m
        }
        self.upcoming_events.append(event)
        
        # Keep only upcoming events (prune past events)
        current_time = detection_time
        self.upcoming_events = [e for e in self.upcoming_events 
                               if e['time'] >= current_time - 1.0]
    
    def gain_schedule(self, severity_score, speed_mps, t_remaining):
        """
        Compute damping coefficient based on severity, speed, and preview.
        
        Parameters:
        -----------
        severity_score : int
            Pothole severity (0-100)
        speed_mps : float
            Vehicle forward speed (m/s)
        t_remaining : float
            Time until impact (seconds)
            
        Returns:
        --------
        c : float
            Scheduled damping coefficient (N·s/m)
        """
        cfg = self.config
        
        # Base damping from severity
        sev_factor = np.clip(severity_score / 100.0, 0, 1)
        c_base = cfg.c_min + (cfg.c_max - cfg.c_min) * sev_factor
        
        # Speed-dependent gain (higher speed → stiffer damping)
        speed_factor = min(1.0, speed_mps / cfg.v_max)
        c_speed = c_base * (0.6 + 0.4 * speed_factor)
        
        # Preview ramp (gradually increase damping as impact approaches)
        if t_remaining is not None and t_remaining < cfg.ramp_window:
            # Ramp from current to target over ramp_window
            ramp_factor = 1.0 - max(0.0, t_remaining / cfg.ramp_window)
            c = cfg.c_min + (c_speed - cfg.c_min) * ramp_factor
        else:
            c = c_speed
        
        # Clamp to bounds
        c = np.clip(c, cfg.c_min, cfg.c_max)
        
        return c
    
    def skyhook_force(self, state, c):
        """
        Compute Skyhook damping force.
        
        Skyhook concept: damper attached to "sky" (inertial reference).
        Force proportional to sprung mass absolute velocity.
        
        Semi-active implementation: can only dissipate energy.
        F_d = c * (z_s_dot - z_u_dot) when energy is dissipated,
        F_d = 0 otherwise.
        
        Parameters:
        -----------
        state : SuspensionState
            Current suspension state
        c : float
            Damping coefficient
            
        Returns:
        --------
        F : float
            Damping force (N)
        """
        # Relative velocity (compression positive)
        v_rel = state.z_s_dot - state.z_u_dot
        
        # Skyhook logic: F = c * v_rel when dissipative
        # Semi-active constraint: can only dissipate energy
        # (force opposes relative velocity)
        
        if v_rel * state.z_s_dot > 0:
            # Dissipating energy
            F = c * v_rel
        else:
            # Would add energy → clamp to passive minimum
            F = self.config.c_min * v_rel
        
        return F
    
    def compute_control(self, state, severity_score, speed_mps, 
                       distance_m=None, current_time=None):
        """
        Compute control force for current state.
        
        Parameters:
        -----------
        state : SuspensionState
            Current suspension state
        severity_score : int
            Pothole severity (0-100, or 0 if no pothole)
        speed_mps : float
            Vehicle speed (m/s)
        distance_m : float, optional
            Distance to upcoming pothole (if known)
        current_time : float, optional
            Current time (for preview scheduling)
            
        Returns:
        --------
        F : float
            Control force (N)
        c : float
            Applied damping coefficient (N·s/m)
        """
        # Compute time-to-impact if distance known
        t_impact = None
        t_remaining = None
        
        if distance_m is not None and speed_mps > 0.1:
            t_impact = distance_m / speed_mps
            total_delay = (self.config.actuator_latency + 
                          self.config.rise_time + 
                          self.config.safety_margin)
            t_remaining = t_impact - total_delay
        
        # Gain scheduling
        c = self.gain_schedule(severity_score, speed_mps, t_remaining)
        
        # Skyhook control law
        F = self.skyhook_force(state, c)
        
        return F, c


def classify_severity(score):
    """
    Classify severity score into categories.
    
    Parameters:
    -----------
    score : int
        Severity score (0-100)
        
    Returns:
    --------
    severity : RoadSeverity
    """
    if score < 25:
        return RoadSeverity.LOW
    elif score < 50:
        return RoadSeverity.MEDIUM
    elif score < 75:
        return RoadSeverity.HIGH
    else:
        return RoadSeverity.CRITICAL


# Integration example
if __name__ == "__main__":
    # Controller setup
    controller = GainScheduledSkyhookController()
    
    # Example: pothole detected ahead
    severity = 65  # HIGH severity
    distance = 12.0  # meters ahead
    speed = 15.0  # m/s (54 km/h)
    
    # Current suspension state (example)
    state = SuspensionState(
        z_s=0.02,  # 2cm sprung mass displacement
        z_s_dot=-0.15,  # moving downward at 0.15 m/s
        z_u=0.01,  # 1cm unsprung mass displacement
        z_u_dot=-0.10  # moving downward at 0.10 m/s
    )
    
    # Compute control
    F, c = controller.compute_control(
        state, 
        severity_score=severity,
        speed_mps=speed,
        distance_m=distance
    )
    
    print(f"Pothole severity: {severity}/100 ({classify_severity(severity).name})")
    print(f"Distance: {distance:.1f}m, Speed: {speed:.1f}m/s ({speed*3.6:.1f}km/h)")
    print(f"Time to impact: {distance/speed:.2f}s")
    print(f"Scheduled damping: c = {c:.1f} N·s/m")
    print(f"Control force: F = {F:.1f} N")

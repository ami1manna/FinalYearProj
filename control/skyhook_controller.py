"""
control/skyhook_controller.py
==============================
Preview-based gain-scheduled Skyhook suspension controller.

Theory
------
The Skyhook concept attaches a virtual damper between the sprung mass
(vehicle body) and an inertial "sky" reference. This maximally isolates
the body from road disturbances.

Because a purely active Skyhook damper would require energy input, we
use the *semi-active* approximation:
    - When the force would dissipate energy → apply it at the scheduled c.
    - When it would add energy → fall back to passive minimum (c_min).

Gain scheduling
---------------
The damping coefficient c is varied in real-time based on three factors:

1. Severity   : higher severity → higher c_base (linear, 0–100 scale)
2. Speed      : higher speed    → stiffer multiplier (0.6 – 1.0 × c_base)
3. Preview    : as time-to-impact shrinks below ramp_window, c ramps from
                c_min up to the scheduled c_speed for a smooth transition.

Usage
-----
    from control import SkyhookController, SuspensionState
    from config import ControllerConfig

    ctrl = SkyhookController(ControllerConfig())
    F, c = ctrl.compute(
        state         = SuspensionState(z_s_dot=-0.15, z_u_dot=-0.10),
        severity      = 65,
        speed_mps     = 15.0,
        distance_m    = 12.0,
    )
    # F  → control force in Newtons
    # c  → applied damping coefficient in N·s/m
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .suspension_state import SuspensionState
from config.system_config import ControllerConfig


class SkyhookController:
    """
    Gain-scheduled preview Skyhook suspension controller.

    Parameters
    ----------
    config : ControllerConfig
        All tunable controller constants. Defaults are used if not provided.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.cfg = config if config is not None else ControllerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        state: SuspensionState,
        severity: int,
        speed_mps: float,
        distance_m: Optional[float] = None,
    ):
        """
        Compute the Skyhook damping force and scheduled coefficient.

        Parameters
        ----------
        state : SuspensionState
            Current suspension corner state (velocities in m/s).
        severity : int
            Pothole severity score [0, 100].  0 = smooth road.
        speed_mps : float
            Current forward vehicle speed (m/s).
        distance_m : float, optional
            Distance to the upcoming pothole in metres.
            Used to compute time-to-impact for preview ramping.

        Returns
        -------
        F : float
            Damping force to apply (Newtons).
        c : float
            Damping coefficient used (N·s/m). Useful for logging/display.
        """
        t_remaining = self._time_remaining(distance_m, speed_mps)
        c = self._gain_schedule(severity, speed_mps, t_remaining)
        F = self._skyhook_force(state, c)
        return F, c

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gain_schedule(
        self,
        severity: int,
        speed_mps: float,
        t_remaining: Optional[float],
    ) -> float:
        """
        Determine the target damping coefficient.

        Three-stage computation:
        1. severity  → c_base   (linear mapping over c_min – c_max)
        2. speed     → c_speed  (multiplier: 0.6 at 0 m/s, 1.0 at v_max)
        3. preview   → c        (linear ramp over ramp_window)

        Parameters
        ----------
        severity : int        Pothole severity [0, 100].
        speed_mps : float     Vehicle speed (m/s).
        t_remaining : float   Seconds until pothole arrives at wheel.

        Returns
        -------
        c : float   Clamped damping coefficient [c_min, c_max] in N·s/m.
        """
        cfg = self.cfg

        # 1 ─ Severity → base damping
        sev_factor = float(np.clip(severity / 100.0, 0.0, 1.0))
        c_base = cfg.c_min + (cfg.c_max - cfg.c_min) * sev_factor

        # 2 ─ Speed → stiffness multiplier
        speed_factor = min(1.0, speed_mps / cfg.v_max)
        c_speed = c_base * (0.6 + 0.4 * speed_factor)

        # 3 ─ Preview ramp
        if t_remaining is not None and t_remaining < cfg.ramp_window:
            # Linearly interpolate from c_min (far) to c_speed (at impact)
            ramp_frac = float(np.clip(1.0 - t_remaining / cfg.ramp_window, 0.0, 1.0))
            c = cfg.c_min + (c_speed - cfg.c_min) * ramp_frac
        else:
            c = c_speed

        return float(np.clip(c, cfg.c_min, cfg.c_max))

    def _skyhook_force(self, state: SuspensionState, c: float) -> float:
        """
        Apply the semi-active Skyhook damping law.

        F = c * v_rel   when dissipating energy (v_rel * z_s_dot > 0)
        F = c_min * v_rel  otherwise (passive fallback)

        Parameters
        ----------
        state : SuspensionState
        c : float   Scheduled damping coefficient (N·s/m).

        Returns
        -------
        F : float   Damping force (N).
        """
        v_rel = state.relative_velocity   # z_s_dot - z_u_dot

        if v_rel * state.z_s_dot > 0:
            # Dissipative regime → apply full scheduled damping
            return c * v_rel
        else:
            # Would inject energy → fall back to passive minimum
            return self.cfg.c_min * v_rel

    def _time_remaining(
        self, distance_m: Optional[float], speed_mps: float
    ) -> Optional[float]:
        """
        Compute how many seconds remain before the wheel hits the pothole,
        after subtracting total actuator delay.

        Parameters
        ----------
        distance_m : float or None   Distance to pothole (metres).
        speed_mps : float            Current speed (m/s).

        Returns
        -------
        t_remaining : float or None
        """
        if distance_m is None or speed_mps < 0.1:
            return None

        t_impact = distance_m / speed_mps
        return t_impact - self.cfg.total_delay

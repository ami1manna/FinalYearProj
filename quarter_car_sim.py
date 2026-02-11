"""
Quarter-Car Suspension Simulator
=================================
Simulates suspension dynamics for validation and comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dataclasses import dataclass


@dataclass
class VehicleParameters:
    """Quarter-car model parameters."""
    m_s: float = 453.0  # sprung mass (kg)
    m_u: float = 36.0  # unsprung mass (kg)
    k_s: float = 17658.0  # suspension spring stiffness (N/m)
    k_t: float = 183887.0  # tire stiffness (N/m)
    c_passive: float = 1500.0  # passive damping (N·s/m)


class QuarterCarSimulator:
    """
    Quarter-car suspension dynamics simulator.
    
    State vector: [z_s, z_s_dot, z_u, z_u_dot]
    where:
        z_s = sprung mass displacement
        z_u = unsprung mass displacement
        z_r = road displacement (input)
    """
    
    def __init__(self, params=None):
        """
        Initialize simulator.
        
        Parameters:
        -----------
        params : VehicleParameters, optional
            Vehicle parameters
        """
        self.params = params if params else VehicleParameters()
        
    def dynamics(self, state, t, z_r_func, F_control_func):
        """
        Quarter-car dynamics equations.
        
        Parameters:
        -----------
        state : array [z_s, z_s_dot, z_u, z_u_dot]
            Current state
        t : float
            Current time
        z_r_func : callable
            Road profile function z_r(t)
        F_control_func : callable
            Control force function F(t, state)
            
        Returns:
        --------
        state_dot : array
            State derivatives
        """
        z_s, z_s_dot, z_u, z_u_dot = state
        
        # Road input
        z_r = z_r_func(t)
        
        # Control force (positive = compression)
        F_control = F_control_func(t, state)
        
        # Suspension force
        F_spring = self.params.k_s * (z_s - z_u)
        F_damper = F_control  # controllable damping
        F_susp = F_spring + F_damper
        
        # Tire force
        F_tire = self.params.k_t * (z_u - z_r)
        
        # Equations of motion
        z_s_ddot = (-F_susp) / self.params.m_s
        z_u_ddot = (F_susp - F_tire) / self.params.m_u
        
        return [z_s_dot, z_s_ddot, z_u_dot, z_u_ddot]
    
    def simulate(self, t_span, initial_state, road_profile, control_law, dt=0.001):
        """
        Run simulation.
        
        Parameters:
        -----------
        t_span : tuple
            (t_start, t_end) in seconds
        initial_state : array
            Initial state [z_s, z_s_dot, z_u, z_u_dot]
        road_profile : callable
            Road displacement function z_r(t)
        control_law : callable
            Control force function F(t, state)
        dt : float
            Time step
            
        Returns:
        --------
        results : dict
            Simulation results with time history
        """
        t = np.arange(t_span[0], t_span[1], dt)
        
        # Integrate
        states = odeint(
            self.dynamics,
            initial_state,
            t,
            args=(road_profile, control_law)
        )
        
        # Extract results
        z_s = states[:, 0]
        z_s_dot = states[:, 1]
        z_u = states[:, 2]
        z_u_dot = states[:, 3]
        
        # Compute additional metrics
        z_s_ddot = np.gradient(z_s_dot, dt)  # sprung mass acceleration
        
        # Suspension travel
        susp_travel = z_s - z_u
        
        # Tire force
        z_r = np.array([road_profile(ti) for ti in t])
        tire_force = self.params.k_t * (z_u - z_r)
        
        # Control force
        control_force = np.array([
            control_law(ti, states[i]) for i, ti in enumerate(t)
        ])
        
        return {
            'time': t,
            'z_s': z_s,
            'z_s_dot': z_s_dot,
            'z_s_ddot': z_s_ddot,
            'z_u': z_u,
            'susp_travel': susp_travel,
            'tire_force': tire_force,
            'control_force': control_force,
            'road_profile': z_r
        }


# Road profiles

def generate_pothole(depth_m, width_m, severity_factor=1.0):
    """
    Generate pothole profile (half-sine bump/dip).
    
    Parameters:
    -----------
    depth_m : float
        Pothole depth (meters)
    width_m : float
        Pothole width (meters)
    severity_factor : float
        Severity multiplier (0-1)
        
    Returns:
    --------
    profile_func : callable
        Road profile function z_r(t, x, v)
    """
    effective_depth = depth_m * severity_factor
    
    def profile(t, x_pothole=10.0, v_vehicle=15.0):
        """
        Road profile as function of time.
        
        Parameters:
        -----------
        t : float
            Time (seconds)
        x_pothole : float
            Longitudinal position of pothole (meters)
        v_vehicle : float
            Vehicle speed (m/s)
            
        Returns:
        --------
        z_r : float
            Road displacement at current position
        """
        # Current vehicle position
        x = v_vehicle * t
        
        # Distance from pothole center
        dx = abs(x - x_pothole)
        
        if dx < width_m / 2:
            # Inside pothole
            z_r = -effective_depth * np.sin(np.pi * dx / (width_m / 2))**2
        else:
            z_r = 0.0
        
        return z_r
    
    return profile


def random_road_profile(psd_roughness='C'):
    """
    Generate random road profile based on ISO 8608 power spectral density.
    
    Parameters:
    -----------
    psd_roughness : str
        Road class: 'A' (very good) to 'E' (very poor)
        
    Returns:
    --------
    profile_func : callable
    """
    # ISO 8608 roughness coefficients (×10^-6 m^3/cycle)
    roughness_map = {
        'A': 16, 'B': 64, 'C': 256, 'D': 1024, 'E': 4096
    }
    Gq = roughness_map.get(psd_roughness, 256) * 1e-6
    
    # Generate frequency-domain representation
    n_freq = 100
    f = np.linspace(0.01, 10, n_freq)  # spatial frequency (cycles/m)
    
    # PSD: Gq(n) = Gq0 * (n/n0)^-w, where w=2 (standard)
    n0 = 0.1  # reference spatial frequency
    PSD = Gq * (f / n0)**(-2)
    
    # Random phase
    phase = np.random.uniform(0, 2*np.pi, n_freq)
    
    def profile(t, v=15.0):
        x = v * t
        z_r = np.sum(np.sqrt(2 * PSD) * np.sin(2 * np.pi * f * x + phase))
        return z_r
    
    return profile


# Control laws

def passive_control(c_passive):
    """
    Passive suspension control.
    
    Parameters:
    -----------
    c_passive : float
        Passive damping coefficient
        
    Returns:
    --------
    control_func : callable
    """
    def control(t, state):
        z_s, z_s_dot, z_u, z_u_dot = state
        v_rel = z_s_dot - z_u_dot
        return c_passive * v_rel
    
    return control


def skyhook_control(c_sky):
    """
    Skyhook damping control.
    
    Parameters:
    -----------
    c_sky : float
        Skyhook damping coefficient
        
    Returns:
    --------
    control_func : callable
    """
    def control(t, state):
        z_s, z_s_dot, z_u, z_u_dot = state
        v_rel = z_s_dot - z_u_dot
        
        # Skyhook: dissipate energy when possible
        if v_rel * z_s_dot > 0:
            return c_sky * v_rel
        else:
            # Passive minimum
            return 800.0 * v_rel
    
    return control


def preview_skyhook_control(c_min, c_max, severity_profile):
    """
    Preview-based gain-scheduled skyhook control.
    
    Parameters:
    -----------
    c_min, c_max : float
        Damping coefficient bounds
    severity_profile : callable
        Severity as function of time severity(t)
        
    Returns:
    --------
    control_func : callable
    """
    def control(t, state):
        z_s, z_s_dot, z_u, z_u_dot = state
        v_rel = z_s_dot - z_u_dot
        
        # Gain schedule based on preview
        severity = severity_profile(t)
        c = c_min + (c_max - c_min) * severity
        
        # Skyhook logic
        if v_rel * z_s_dot > 0:
            return c * v_rel
        else:
            return c_min * v_rel
    
    return control


# Comparison and plotting

def compare_controllers(road_profile, severity_profile=None):
    """
    Compare passive, Skyhook, and preview Skyhook controllers.
    
    Parameters:
    -----------
    road_profile : callable
        Road profile function
    severity_profile : callable, optional
        Severity function for preview control
    """
    sim = QuarterCarSimulator()
    
    # Initial conditions (at rest)
    x0 = [0.0, 0.0, 0.0, 0.0]
    
    # Time span
    t_span = (0, 3.0)
    
    # Controllers
    controllers = {
        'Passive': passive_control(1500.0),
        'Skyhook': skyhook_control(2500.0),
        'Preview Skyhook': preview_skyhook_control(
            800.0, 4000.0,
            severity_profile if severity_profile else lambda t: 0.5
        )
    }
    
    results = {}
    
    for name, controller in controllers.items():
        print(f"Simulating {name}...")
        results[name] = sim.simulate(t_span, x0, road_profile, controller)
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot 1: Road profile
    t = results['Passive']['time']
    axes[0].plot(t, results['Passive']['road_profile'] * 1000, 'k-', linewidth=1.5)
    axes[0].set_ylabel('Road (mm)')
    axes[0].set_title('Road Profile')
    axes[0].grid(True)
    
    # Plot 2: Sprung mass acceleration
    for name, res in results.items():
        axes[1].plot(res['time'], res['z_s_ddot'], label=name)
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('Sprung Mass Acceleration (Ride Comfort)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Suspension travel
    for name, res in results.items():
        axes[2].plot(res['time'], res['susp_travel'] * 1000, label=name)
    axes[2].set_ylabel('Travel (mm)')
    axes[2].set_title('Suspension Travel')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: Tire force variation
    for name, res in results.items():
        axes[3].plot(res['time'], res['tire_force'], label=name)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Tire Force (N)')
    axes[3].set_title('Tire Force (Road Holding)')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/claude/suspension_comparison.png', dpi=150)
    print("Plot saved to suspension_comparison.png")
    
    # Compute metrics
    print("\n=== Performance Metrics ===")
    for name, res in results.items():
        rms_accel = np.sqrt(np.mean(res['z_s_ddot']**2))
        peak_accel = np.max(np.abs(res['z_s_ddot']))
        rms_travel = np.sqrt(np.mean(res['susp_travel']**2))
        rms_tire = np.sqrt(np.mean((res['tire_force'] - np.mean(res['tire_force']))**2))
        
        print(f"\n{name}:")
        print(f"  RMS Acceleration: {rms_accel:.3f} m/s²")
        print(f"  Peak Acceleration: {peak_accel:.3f} m/s²")
        print(f"  RMS Susp Travel: {rms_travel*1000:.2f} mm")
        print(f"  RMS Tire Load Var: {rms_tire:.1f} N")


if __name__ == "__main__":
    print("Quarter-Car Suspension Simulation\n")
    
    # Define pothole
    depth = 0.05  # 5cm deep pothole
    width = 0.6   # 60cm wide
    severity = 0.7  # 70% severity
    
    road = generate_pothole(depth, width, severity)
    
    # Define severity preview (ramps up before pothole at t=0.67s)
    def severity_sched(t, t_pothole=10.0/15.0, preview_window=0.3):
        if t < t_pothole - preview_window:
            return 0.0
        elif t < t_pothole:
            # Ramp up
            return (t - (t_pothole - preview_window)) / preview_window
        else:
            # Full severity during/after pothole
            return 1.0
    
    # Run comparison
    compare_controllers(road, severity_sched)

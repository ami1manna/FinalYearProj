"""
Optical Flow-based Vehicle Speed Estimation
============================================
Estimates forward velocity using ground plane feature tracking.
"""

import numpy as np
import cv2
from collections import deque


class OpticalFlowSpeedEstimator:
    """
    Estimate vehicle speed using optical flow on ground plane features.
    Uses Lucas-Kanade sparse optical flow with feature tracking.
    """
    
    def __init__(self, ipm_estimator, fps=30, buffer_size=10):
        """
        Initialize speed estimator.
        
        Parameters:
        -----------
        ipm_estimator : IPMDistanceEstimator
            Calibrated IPM estimator for ground plane mapping
        fps : float
            Camera frame rate
        buffer_size : int
            Number of speed estimates to smooth over
        """
        self.ipm = ipm_estimator
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7
        )
        
        # Optical flow parameters (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # State
        self.prev_gray = None
        self.prev_points = None
        self.speed_buffer = deque(maxlen=buffer_size)
        
        # ROI for ground plane features (lower half of image)
        self.roi_y_start = 0.5  # start at middle of image
        self.roi_y_end = 0.9    # end near bottom (avoid hood)
        
    def reset(self):
        """Reset tracking state."""
        self.prev_gray = None
        self.prev_points = None
        self.speed_buffer.clear()
    
    def _extract_ground_features(self, gray_frame):
        """
        Detect features in ground plane ROI.
        
        Parameters:
        -----------
        gray_frame : np.ndarray
            Grayscale image
            
        Returns:
        --------
        points : np.ndarray
            Feature points in ground ROI
        """
        h, w = gray_frame.shape
        
        # Define ROI (lower portion of image = ground)
        y_start = int(h * self.roi_y_start)
        y_end = int(h * self.roi_y_end)
        
        roi = gray_frame[y_start:y_end, :]
        
        # Detect features
        points = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
        
        if points is not None:
            # Adjust coordinates to full image
            points[:, 0, 1] += y_start
        
        return points
    
    def estimate_speed(self, frame):
        """
        Estimate vehicle speed from current frame.
        
        Parameters:
        -----------
        frame : np.ndarray
            Current BGR frame
            
        Returns:
        --------
        speed_mps : float
            Estimated speed in meters/second (None if unavailable)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = self._extract_ground_features(gray)
            return None
        
        # Skip if no previous features
        if self.prev_points is None or len(self.prev_points) < 5:
            self.prev_gray = gray
            self.prev_points = self._extract_ground_features(gray)
            return None
        
        # Calculate optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        # Filter good matches
        good_prev = self.prev_points[status == 1]
        good_curr = curr_points[status == 1]
        
        if len(good_prev) < 5:
            # Too few features, re-detect
            self.prev_gray = gray
            self.prev_points = self._extract_ground_features(gray)
            return None
        
        # Convert pixel displacements to ground plane displacements
        displacements_m = []
        
        for p_prev, p_curr in zip(good_prev, good_curr):
            # Convert to ground coordinates
            X_prev, Y_prev, _ = self.ipm.image_to_ground(p_prev[0], p_prev[1])
            X_curr, Y_curr, _ = self.ipm.image_to_ground(p_curr[0], p_curr[1])
            
            if X_prev is None or X_curr is None:
                continue
            
            # Forward displacement (X-axis)
            dx = X_curr - X_prev
            
            displacements_m.append(dx)
        
        if len(displacements_m) < 3:
            self.prev_gray = gray
            self.prev_points = self._extract_ground_features(gray)
            return None
        
        # Robust median displacement
        median_displacement = np.median(displacements_m)
        
        # Speed = displacement / time
        # Note: displacement is negative when moving forward (features move backward in image)
        speed_mps = abs(median_displacement) / self.dt
        
        # Smooth speed estimate
        self.speed_buffer.append(speed_mps)
        smoothed_speed = np.median(list(self.speed_buffer))
        
        # Update state
        self.prev_gray = gray
        self.prev_points = good_curr.reshape(-1, 1, 2)
        
        # Re-detect features periodically to maintain tracking
        if len(self.prev_points) < 20:
            new_features = self._extract_ground_features(gray)
            if new_features is not None:
                self.prev_points = new_features
        
        return float(smoothed_speed)
    
    def get_speed_kmh(self, frame):
        """
        Get speed in km/h.
        
        Parameters:
        -----------
        frame : np.ndarray
            Current BGR frame
            
        Returns:
        --------
        speed_kmh : float
            Speed in km/h (None if unavailable)
        """
        speed_mps = self.estimate_speed(frame)
        
        if speed_mps is None:
            return None
        
        return speed_mps * 3.6  # m/s to km/h


# Alternative: Simple speed estimator using known vehicle speed (CAN bus / GPS)
class CANSpeedReader:
    """
    Read vehicle speed from CAN bus or GPS.
    This is a placeholder - actual implementation depends on hardware.
    """
    
    def __init__(self, source='CAN'):
        """
        Parameters:
        -----------
        source : str
            'CAN' for CAN bus, 'GPS' for GPS module
        """
        self.source = source
        # Initialize CAN/GPS interface here
        
    def get_speed_mps(self):
        """
        Read current speed in m/s.
        
        Returns:
        --------
        speed : float
            Vehicle speed in meters/second
        """
        # Placeholder implementation
        # In production, read from actual CAN bus or GPS
        
        if self.source == 'CAN':
            # Read from CAN bus (requires python-can library)
            # speed = read_can_speed()
            pass
        elif self.source == 'GPS':
            # Read from GPS module (requires gpsd or similar)
            # speed = read_gps_speed()
            pass
        
        # Return dummy value for demonstration
        return 15.0  # 54 km/h
    
    def get_speed_kmh(self):
        """Get speed in km/h."""
        return self.get_speed_mps() * 3.6


if __name__ == "__main__":
    # Example usage
    from ipm_distance import IPMDistanceEstimator, create_default_camera_matrix
    
    # Setup
    K = create_default_camera_matrix(640, 384, focal_length=500)
    ipm = IPMDistanceEstimator(K, None, camera_height=1.2, pitch_angle=15)
    
    speed_estimator = OpticalFlowSpeedEstimator(ipm, fps=30)
    
    # In video loop:
    # cap = cv2.VideoCapture("video.mp4")
    # while True:
    #     ret, frame = cap.read()
    #     speed_mps = speed_estimator.estimate_speed(frame)
    #     if speed_mps:
    #         print(f"Speed: {speed_mps:.1f} m/s ({speed_mps*3.6:.1f} km/h)")
    
    print("Speed estimator initialized. Use estimate_speed() in video loop.")

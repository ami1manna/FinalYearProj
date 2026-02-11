"""
Inverse Perspective Mapping (IPM) for Distance Estimation
==========================================================
Converts image coordinates to real-world ground plane coordinates
for accurate monocular distance estimation.
"""

import numpy as np
import cv2


class IPMDistanceEstimator:
    """
    IPM-based distance estimator for monocular camera.
    
    Requires camera calibration (intrinsics + extrinsics).
    Assumes flat road surface.
    """
    
    def __init__(self, camera_matrix, dist_coeffs, camera_height, pitch_angle):
        """
        Initialize IPM Distance Estimator.
        
        Parameters:
        -----------
        camera_matrix : np.ndarray (3x3)
            Camera intrinsic matrix K
        dist_coeffs : np.ndarray
            Lens distortion coefficients
        camera_height : float
            Camera height above ground in meters
        pitch_angle : float
            Camera pitch angle in degrees (positive = looking down)
        """
        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_height = camera_height
        self.pitch_rad = np.deg2rad(pitch_angle)
        
        # Compute homography matrix
        self.H_inv = self._compute_homography()
        
    def _compute_homography(self):
        """
        Compute inverse homography for image-to-ground transformation.
        
        Returns:
        --------
        H_inv : np.ndarray (3x3)
            Inverse homography matrix
        """
        # Simplified IPM for flat ground plane
        # Using camera geometry: distance = (h * f) / (v - v0)
        # where v0 is the horizon line (principal point y-coordinate)
        
        fx = self.K[0, 0]  # focal length in x
        fy = self.K[1, 1]  # focal length in y
        cx = self.K[0, 2]  # principal point x
        cy = self.K[1, 2]  # principal point y
        
        # Account for pitch angle
        # The effective horizon line shifts with pitch
        cy_effective = cy - fy * np.tan(self.pitch_rad)
        
        # Create transformation matrix from image to ground
        # This is a simplified version that works better for our use case
        H_inv = np.array([
            [(self.camera_height / fx), 0, -cx * (self.camera_height / fx)],
            [0, (self.camera_height / fy), -cy_effective * (self.camera_height / fy)],
            [0, 0, 1]
        ])
        
        return H_inv
    
    def image_to_ground(self, u, v):
        """
        Convert image pixel to ground plane coordinates.
        
        Parameters:
        -----------
        u, v : float
            Image coordinates (pixels)
            
        Returns:
        --------
        X, Y, distance : tuple
            Ground coordinates (X forward, Y lateral) in meters
            and Euclidean distance from camera base
        """
        # Undistort point if distortion coefficients provided
        if self.dist_coeffs is not None:
            pts = np.array([[[u, v]]], dtype=np.float32)
            pts_undist = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
            u, v = pts_undist[0, 0]
        
        # Homogeneous coordinates
        p_img = np.array([u, v, 1.0])
        
        # Transform to ground plane
        p_ground = self.H_inv @ p_img
        
        # Normalize
        w = p_ground[2]
        if abs(w) < 1e-9:
            return None, None, None
            
        X = p_ground[0] / w
        Y = p_ground[1] / w
        
        # Euclidean distance
        distance = np.sqrt(X**2 + Y**2)
        
        return float(X), float(Y), float(distance)
    
    def bbox_bottom_distance(self, x1, y1, x2, y2):
        """
        Estimate distance using bottom-center of bounding box.
        This corresponds to contact point with the ground.
        
        Parameters:
        -----------
        x1, y1, x2, y2 : float
            Bounding box coordinates
            
        Returns:
        --------
        distance : float
            Distance in meters (None if estimation fails)
        """
        # Use bottom-center point
        u = (x1 + x2) / 2
        v = y2  # bottom edge
        
        # Simple pinhole camera model for distance estimation
        # distance = (h * f) / (v - cy) where cy is principal point y
        fx = self.K[0, 0]
        fy = self.K[1, 1] 
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        # Account for pitch angle - shift the principal point
        cy_effective = cy - fy * np.tan(self.pitch_rad)
        
        # Distance from camera to point on ground plane
        if abs(v - cy_effective) < 1e-6:
            return None  # Point too close to horizon
            
        distance = (self.camera_height * fx) / abs(v - cy_effective)
        
        return distance


# Calibration helper functions

def calibrate_from_reference(image, known_distance_m, known_pixel_height):
    """
    Quick calibration using a known reference object.
    
    Parameters:
    -----------
    image : np.ndarray
        Camera image
    known_distance_m : float
        Actual distance to reference object in meters
    known_pixel_height : float
        Height of reference object in pixels
        
    Returns:
    --------
    focal_length : float
        Estimated focal length
        
    Example:
    --------
    # Place a 1.7m tall person at 10m distance
    # Measure their height in the image: 150 pixels
    f = calibrate_from_reference(img, 10.0, 150)
    """
    # Simple pinhole formula: f = (pixel_height * distance) / real_height
    # Assuming standard reference (e.g., lane marking 3m, person 1.7m)
    REFERENCE_HEIGHT_M = 1.7  # average human height
    
    focal_length = (known_pixel_height * known_distance_m) / REFERENCE_HEIGHT_M
    
    return focal_length


def create_default_camera_matrix(image_width, image_height, focal_length=None):
    """
    Create camera intrinsic matrix with default parameters.
    
    Parameters:
    -----------
    image_width, image_height : int
        Image dimensions
    focal_length : float, optional
        If None, uses image width as rough estimate
        
    Returns:
    --------
    K : np.ndarray (3x3)
        Camera intrinsic matrix
    """
    if focal_length is None:
        # Rough estimate: focal_length ≈ image_width
        focal_length = image_width
    
    # Principal point at image center
    cx = image_width / 2
    cy = image_height / 2
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


# Example usage
if __name__ == "__main__":
    # Example: typical dash cam setup
    img_width, img_height = 640, 384
    
    # Camera parameters (example values - MUST be calibrated for your setup)
    K = create_default_camera_matrix(img_width, img_height, focal_length=500)
    dist = None  # or np.array([k1, k2, p1, p2, k3]) for lens distortion
    
    camera_height = 1.2  # meters above ground
    pitch_angle = 15  # degrees, looking down
    
    estimator = IPMDistanceEstimator(K, dist, camera_height, pitch_angle)
    
    # Test with bbox (example pothole detection)
    x1, y1, x2, y2 = 200, 250, 350, 320
    
    distance = estimator.bbox_bottom_distance(x1, y1, x2, y2)
    
    if distance:
        print(f"Estimated distance: {distance:.2f} meters")
    else:
        print("Distance estimation failed")

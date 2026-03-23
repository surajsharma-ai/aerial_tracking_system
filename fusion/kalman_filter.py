"""
fusion/kalman_filter.py - Kalman filter implementation for sensor fusion
"""

import numpy as np
import pandas as pd
from typing import Tuple

class KalmanFilter:
    """
    Extended Kalman Filter for sensor fusion
    State vector: [x, y, z, vx, vy, vz]
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.state_dim = 6  # position + velocity
        self.measurement_dim = 3  # position only
        
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(self.state_dim)
        
        # State covariance matrix
        self.P = np.eye(self.state_dim) * 1000.0
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        q = 10.0  # process noise magnitude
        self.Q = np.eye(self.state_dim) * q
        self.Q[3:, 3:] *= 2.0  # Higher noise for velocity
        
    def predict(self) -> np.ndarray:
        """Prediction step"""
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state
    
    def update(self, measurement: np.ndarray, measurement_noise: float):
        """
        Update step with new measurement
        
        Args:
            measurement: [x, y, z] position measurement
            measurement_noise: measurement noise standard deviation
        """
        # Measurement noise covariance
        R = np.eye(self.measurement_dim) * (measurement_noise ** 2)
        
        # Innovation
        y = measurement - (self.H @ self.state)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate"""
        return self.state[:3]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate"""
        return self.state[3:]
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (trace of position covariance)"""
        return np.trace(self.P[:3, :3])


class SensorFusion:
    """
    Combines multiple sensor measurements using Kalman filtering
    """
    
    def __init__(self, dt: float = 0.1):
        self.kf = KalmanFilter(dt)
        self.fusion_history = []
        
    def initialize_from_measurement(self, measurement: np.ndarray):
        """Initialize filter with first measurement"""
        self.kf.state[:3] = measurement
        self.kf.state[3:] = 0  # Zero initial velocity
        
    def process_measurements(self, measurements_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process multi-sensor measurements through Kalman filter
        
        Args:
            measurements_df: DataFrame with sensor measurements
        
        Returns:
            DataFrame with fused estimates
        """
        fusion_results = []
        
        # Group measurements by time
        for time, group in measurements_df.groupby('time'):
            # Prediction step
            self.kf.predict()
            
            # Update with each available measurement
            for _, measurement in group.iterrows():
                if measurement['detected']:
                    meas_vector = np.array([
                        measurement['x_measured'],
                        measurement['y_measured'],
                        measurement['z_measured']
                    ])
                    
                    self.kf.update(meas_vector, measurement['noise_std'])
            
            # Record fused estimate
            fusion_results.append({
                'time': time,
                'x_fused': self.kf.state[0],
                'y_fused': self.kf.state[1],
                'z_fused': self.kf.state[2],
                'vx_fused': self.kf.state[3],
                'vy_fused': self.kf.state[4],
                'vz_fused': self.kf.state[5],
                'uncertainty': self.kf.get_position_uncertainty()
            })
        
        return pd.DataFrame(fusion_results)
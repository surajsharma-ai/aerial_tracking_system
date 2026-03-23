"""
object_simulator.py - High-speed object trajectory simulator
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class HighSpeedObjectSimulator:
    """Simulates realistic trajectories of high-speed aerial objects with various maneuvers"""
    
    def __init__(self, initial_position: np.ndarray, initial_velocity: np.ndarray, dt: float = 0.1):
        """
        Initialize the simulator
        
        Args:
            initial_position: Initial position [x, y, z] in meters
            initial_velocity: Initial velocity [vx, vy, vz] in m/s
            dt: Time step in seconds
        """
        self.position = initial_position.astype(float)
        self.velocity = initial_velocity.astype(float)
        self.dt = dt
        
        # Physics constants
        self.gravity = 9.81  # m/s^2
        self.wind = np.array([0.0, 0.0, 0.0])  # m/s
        self.air_density = 1.225  # kg/m^3 at sea level
        self.drag_coefficient = 0.1  # Dimensionless
        
        # Storage for trajectory
        self.trajectory = []
        self.times = []
        
    def simulate_trajectory(self, duration: float, maneuver_pattern: Optional[List[Tuple]] = None):
        """
        Simulate object trajectory with optional maneuvers
        
        Args:
            duration: Total simulation time in seconds
            maneuver_pattern: List of (time, maneuver_type, intensity) tuples
                Example: [(20, 'turn', 1.2), (50, 'climb', 1.1)]
        """
        if maneuver_pattern is None:
            maneuver_pattern = []
        
        num_steps = int(duration / self.dt)
        current_position = self.position.copy()
        current_velocity = self.velocity.copy()
        
        # Create maneuver schedule
        maneuver_dict = {t: (m, i) for t, m, i in maneuver_pattern}
        
        # Reset trajectory storage
        self.trajectory = [current_position.copy()]
        self.times = [0.0]
        
        for step in range(1, num_steps):
            current_time = step * self.dt
            
            # Apply maneuvers if scheduled for this time
            acceleration = np.array([0.0, 0.0, -self.gravity])
            
            # Check if there's a maneuver at this time
            for maneuver_time in sorted(maneuver_dict.keys()):
                if abs(current_time - maneuver_time) < self.dt / 2:
                    maneuver_type, intensity = maneuver_dict[maneuver_time]
                    accel_delta = self._get_maneuver_acceleration(
                        maneuver_type, intensity, current_velocity
                    )
                    acceleration += accel_delta
            
            # Apply continuous maneuver effects (smooth transitions)
            for maneuver_time in sorted(maneuver_dict.keys()):
                maneuver_type, intensity = maneuver_dict[maneuver_time]
                # Apply maneuver for ~10 seconds from its start time
                if maneuver_time <= current_time <= maneuver_time + 10:
                    accel_delta = self._get_maneuver_acceleration(
                        maneuver_type, intensity, current_velocity
                    ) * 0.5  # Reduced acceleration for continuous effect
                    acceleration += accel_delta
            
            # Update velocity and position (simple Euler integration)
            current_velocity += acceleration * self.dt
            current_position += current_velocity * self.dt
            
            # Store trajectory
            self.trajectory.append(current_position.copy())
            self.times.append(current_time)
    
    def _get_maneuver_acceleration(self, maneuver_type: str, intensity: float, 
                                  velocity: np.ndarray) -> np.ndarray:
        """Get acceleration vector for a specific maneuver"""
        accel = np.array([0.0, 0.0, 0.0])
        speed = np.linalg.norm(velocity)
        
        if maneuver_type == 'turn':
            # Horizontal turn maneuver
            # Increase acceleration perpendicular to velocity
            if speed > 0:
                # Turn in xy-plane with intensity
                angle = intensity * 0.1  # rad/s^2
                turn_accel = speed * angle
                # Perpendicular direction (approximate)
                perp_dir = np.array([-velocity[1], velocity[0], 0])
                if np.linalg.norm(perp_dir) > 0:
                    perp_dir = perp_dir / np.linalg.norm(perp_dir)
                    accel = perp_dir * turn_accel
        
        elif maneuver_type == 'climb':
            # Vertical climb maneuver
            accel[2] = 5.0 * intensity  # m/s^2 upward
        
        elif maneuver_type == 'dive':
            # Vertical dive maneuver
            accel[2] = -5.0 * intensity  # m/s^2 downward
        
        elif maneuver_type == 'spiral':
            # Combination of turn and climb
            if speed > 0:
                angle = intensity * 0.1
                turn_accel = speed * angle
                perp_dir = np.array([-velocity[1], velocity[0], 0])
                if np.linalg.norm(perp_dir) > 0:
                    perp_dir = perp_dir / np.linalg.norm(perp_dir)
                    accel = perp_dir * turn_accel
                    accel[2] += 2.0 * intensity  # Add climb component
        
        return accel
    
    def get_trajectory_dataframe(self) -> pd.DataFrame:
        """Return trajectory as a pandas DataFrame"""
        positions = np.array(self.trajectory)
        
        # Calculate velocities by numerical differentiation
        velocities = np.zeros_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) / self.dt
        velocities[0] = velocities[1]  # Use next velocity for first point
        
        df = pd.DataFrame({
            'time': self.times,
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
            'vx': velocities[:, 0],
            'vy': velocities[:, 1],
            'vz': velocities[:, 2],
        })
        
        return df

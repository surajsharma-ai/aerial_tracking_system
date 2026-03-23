import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Optional
from dataclasses import dataclass

@dataclass
class SensorCharacteristics:
    """Sensor performance parameters"""
    name: str
    position_noise_std: float
    update_rate: float
    detection_probability: float
    max_range: float
    
class MultiSensorSimulator:
    """
    Simulates multiple sensor types observing aerial objects
    """
    
    def __init__(self):
        self.sensors = {
            'radar': SensorCharacteristics(
                name='Radar',
                position_noise_std=50.0,
                update_rate=10.0,
                detection_probability=0.98,
                max_range=100000.0
            ),
            'satellite': SensorCharacteristics(
                name='Satellite',
                position_noise_std=100.0,
                update_rate=1.0,
                detection_probability=0.85,
                max_range=500000.0
            ),
            'thermal': SensorCharacteristics(
                name='Thermal',
                position_noise_std=150.0,
                update_rate=5.0,
                detection_probability=0.75,
                max_range=50000.0
            )
        }
        
    def add_measurement_noise(self, 
                             true_position: np.ndarray,
                             noise_std: float) -> np.ndarray:
        """Add Gaussian noise to position measurement"""
        noise = np.random.normal(0, noise_std, size=3)
        return true_position + noise
    
    def check_detection(self, 
                       true_position: np.ndarray,
                       sensor: SensorCharacteristics) -> bool:
        """Determine if sensor detects object"""
        distance = np.linalg.norm(true_position)
        
        if distance > sensor.max_range:
            return False
        
        return np.random.random() < sensor.detection_probability
    
    def should_update(self, time: float, sensor: SensorCharacteristics) -> bool:
        """Check if sensor should provide update at this time"""
        update_interval = 1.0 / sensor.update_rate
        return (time % update_interval) < 0.01
    
    def generate_sensor_measurements(self,
                                    trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-sensor measurements for entire trajectory
        
        Returns:
            DataFrame with columns: time, sensor_type, x_measured, y_measured, z_measured, detected
        """
        measurements = []
        
        for idx, row in trajectory_df.iterrows():
            time = row['time']
            true_position = np.array([row['x'], row['y'], row['z']])
            
            for sensor_name, sensor in self.sensors.items():
                if not self.should_update(time, sensor):
                    continue
                
                detected = self.check_detection(true_position, sensor)
                
                if detected:
                    measured_position = self.add_measurement_noise(
                        true_position, sensor.position_noise_std
                    )
                    
                    measurements.append({
                        'time': time,
                        'sensor_type': sensor_name,
                        'x_measured': measured_position[0],
                        'y_measured': measured_position[1],
                        'z_measured': measured_position[2],
                        'x_true': true_position[0],
                        'y_true': true_position[1],
                        'z_true': true_position[2],
                        'detected': True,
                        'noise_std': sensor.position_noise_std
                    })
                else:
                    measurements.append({
                        'time': time,
                        'sensor_type': sensor_name,
                        'x_measured': np.nan,
                        'y_measured': np.nan,
                        'z_measured': np.nan,
                        'x_true': true_position[0],
                        'y_true': true_position[1],
                        'z_true': true_position[2],
                        'detected': False,
                        'noise_std': sensor.position_noise_std
                    })
        
        return pd.DataFrame(measurements)
    
    def add_systematic_bias(self, 
                           measurements_df: pd.DataFrame,
                           sensor_type: str,
                           bias_vector: np.ndarray) -> pd.DataFrame:
        """Add systematic bias to specific sensor"""
        mask = measurements_df['sensor_type'] == sensor_type
        measurements_df.loc[mask, 'x_measured'] += bias_vector[0]
        measurements_df.loc[mask, 'y_measured'] += bias_vector[1]
        measurements_df.loc[mask, 'z_measured'] += bias_vector[2]
        return measurements_df
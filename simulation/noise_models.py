"""
simulation/noise_models.py - Noise models for realistic sensor simulation
"""

import numpy as np
from typing import Tuple

class NoiseModel:
    """Base class for noise models"""
    
    def generate(self, size: int = 1) -> np.ndarray:
        """Generate noise samples"""
        raise NotImplementedError

class GaussianNoise(NoiseModel):
    """Gaussian (Normal) noise model"""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def generate(self, size: int = 1) -> np.ndarray:
        return np.random.normal(self.mean, self.std, size)

class UniformNoise(NoiseModel):
    """Uniform noise model"""
    
    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.low = low
        self.high = high
    
    def generate(self, size: int = 1) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size)

class MultiPathNoise(NoiseModel):
    """
    Multi-path interference noise (common in radar)
    Combination of direct signal and reflected signals
    """
    
    def __init__(self, direct_std: float = 10.0, 
                 reflection_std: float = 30.0,
                 reflection_prob: float = 0.3):
        self.direct_std = direct_std
        self.reflection_std = reflection_std
        self.reflection_prob = reflection_prob
    
    def generate(self, size: int = 1) -> np.ndarray:
        noise = np.random.normal(0, self.direct_std, size)
        
        # Add multi-path reflections
        reflection_mask = np.random.random(size) < self.reflection_prob
        reflections = np.random.normal(0, self.reflection_std, size)
        noise[reflection_mask] += reflections[reflection_mask]
        
        return noise

class CorrelatedNoise(NoiseModel):
    """
    Time-correlated noise (e.g., atmospheric effects)
    Uses AR(1) process
    """
    
    def __init__(self, std: float = 20.0, correlation: float = 0.7):
        self.std = std
        self.correlation = correlation
        self.previous = 0.0
    
    def generate(self, size: int = 1) -> np.ndarray:
        noise = np.zeros(size)
        
        for i in range(size):
            noise[i] = (self.correlation * self.previous + 
                       np.random.normal(0, self.std * np.sqrt(1 - self.correlation**2)))
            self.previous = noise[i]
        
        return noise

class OutlierNoise(NoiseModel):
    """
    Noise with occasional outliers (sensor glitches)
    """
    
    def __init__(self, base_std: float = 15.0,
                 outlier_std: float = 100.0,
                 outlier_prob: float = 0.05):
        self.base_std = base_std
        self.outlier_std = outlier_std
        self.outlier_prob = outlier_prob
    
    def generate(self, size: int = 1) -> np.ndarray:
        noise = np.random.normal(0, self.base_std, size)
        
        # Add outliers
        outlier_mask = np.random.random(size) < self.outlier_prob
        outliers = np.random.normal(0, self.outlier_std, size)
        noise[outlier_mask] = outliers[outlier_mask]
        
        return noise

class QuantizationNoise(NoiseModel):
    """
    Quantization noise from digital conversion
    """
    
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
    
    def generate(self, size: int = 1) -> np.ndarray:
        # Uniform noise within quantization step
        return np.random.uniform(-self.resolution/2, 
                                self.resolution/2, 
                                size)

class CompositeSensorNoise:
    """
    Realistic composite noise model combining multiple sources
    """
    
    def __init__(self, sensor_type: str = 'radar'):
        if sensor_type == 'radar':
            self.noise_models = {
                'thermal': GaussianNoise(0, 10),
                'multipath': MultiPathNoise(5, 20, 0.2),
                'quantization': QuantizationNoise(0.5),
                'outliers': OutlierNoise(10, 50, 0.02)
            }
            self.weights = [0.4, 0.3, 0.2, 0.1]
            
        elif sensor_type == 'satellite':
            self.noise_models = {
                'atmospheric': CorrelatedNoise(30, 0.8),
                'thermal': GaussianNoise(0, 20),
                'quantization': QuantizationNoise(1.0)
            }
            self.weights = [0.5, 0.3, 0.2]
            
        elif sensor_type == 'thermal':
            self.noise_models = {
                'thermal': GaussianNoise(0, 40),
                'atmospheric': CorrelatedNoise(50, 0.6),
                'outliers': OutlierNoise(30, 100, 0.03)
            }
            self.weights = [0.5, 0.3, 0.2]
        else:
            self.noise_models = {'gaussian': GaussianNoise(0, 20)}
            self.weights = [1.0]
    
    def generate(self, size: int = 1) -> np.ndarray:
        """Generate composite noise"""
        total_noise = np.zeros(size)
        
        for (name, model), weight in zip(self.noise_models.items(), self.weights):
            total_noise += weight * model.generate(size)
        
        return total_noise

def add_realistic_sensor_noise(measurements: np.ndarray,
                               sensor_type: str = 'radar') -> np.ndarray:
    """
    Add realistic noise to sensor measurements
    
    Args:
        measurements: (N, 3) array of positions
        sensor_type: 'radar', 'satellite', or 'thermal'
    
    Returns:
        Noisy measurements
    """
    noise_model = CompositeSensorNoise(sensor_type)
    
    noisy_measurements = measurements.copy()
    for i in range(3):  # x, y, z
        noise = noise_model.generate(len(measurements))
        noisy_measurements[:, i] += noise
    
    return noisy_measurements
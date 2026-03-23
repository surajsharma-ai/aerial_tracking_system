"""
fusion/sensor_combiner.py - Advanced sensor combination strategies
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SensorWeight:
    """Weight for sensor in fusion"""
    radar: float = 0.5
    satellite: float = 0.3
    thermal: float = 0.2


class WeightedAverageFusion:
    """Simple weighted average of sensor measurements"""
    
    def __init__(self, weights: SensorWeight = None):
        self.weights = weights or SensorWeight()
    
    def fuse(self, measurements: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine measurements using weighted average
        
        Args:
            measurements: dict of sensor_name -> measurement
        
        Returns:
            Fused position estimate
        """
        fused = np.zeros(3)
        total_weight = 0.0
        
        if measurements.get('radar') is not None:
            fused += self.weights.radar * measurements['radar']
            total_weight += self.weights.radar
        
        if measurements.get('satellite') is not None:
            fused += self.weights.satellite * measurements['satellite']
            total_weight += self.weights.satellite
        
        if measurements.get('thermal') is not None:
            fused += self.weights.thermal * measurements['thermal']
            total_weight += self.weights.thermal
        
        if total_weight > 0:
            fused /= total_weight
        
        return fused


class CovarianceIntersectionFusion:
    """
    Covariance Intersection for unknown correlations
    More robust than Kalman when sensor correlations unknown
    """
    
    def __init__(self):
        self.sensor_covariances = {
            'radar': np.eye(3) * 50**2,
            'satellite': np.eye(3) * 100**2,
            'thermal': np.eye(3) * 150**2
        }
    
    def fuse(self, measurements: Dict[str, np.ndarray]) -> tuple:
        """
        Fuse using Covariance Intersection
        
        Returns:
            fused_estimate, fused_covariance
        """
        valid_sensors = [k for k, v in measurements.items() if v is not None]
        
        if not valid_sensors:
            return np.zeros(3), np.eye(3) * 1000
        
        if len(valid_sensors) == 1:
            sensor = valid_sensors[0]
            return measurements[sensor], self.sensor_covariances[sensor]
        
        # Covariance Intersection formula
        omega = 0.5  # Simplified equal weighting
        
        P_fused_inv = np.zeros((3, 3))
        weighted_measurements = np.zeros(3)
        
        for sensor in valid_sensors:
            P = self.sensor_covariances[sensor]
            P_inv = np.linalg.inv(P)
            
            P_fused_inv += omega * P_inv
            weighted_measurements += omega * P_inv @ measurements[sensor]
        
        P_fused = np.linalg.inv(P_fused_inv)
        fused_estimate = P_fused @ weighted_measurements
        
        return fused_estimate, P_fused


class AdaptiveWeightedFusion:
    """
    Adaptive fusion that adjusts weights based on recent performance
    """
    
    def __init__(self):
        self.weights = SensorWeight()
        self.error_history = {
            'radar': [],
            'satellite': [],
            'thermal': []
        }
        self.window_size = 20
    
    def update_errors(self, sensor_errors: Dict[str, float]):
        """Update error history"""
        for sensor, error in sensor_errors.items():
            self.error_history[sensor].append(error)
            if len(self.error_history[sensor]) > self.window_size:
                self.error_history[sensor] = self.error_history[sensor][-self.window_size:]
    
    def compute_adaptive_weights(self) -> SensorWeight:
        """Compute weights inversely proportional to recent errors"""
        avg_errors = {}
        
        for sensor in ['radar', 'satellite', 'thermal']:
            if self.error_history[sensor]:
                avg_errors[sensor] = np.mean(self.error_history[sensor])
            else:
                avg_errors[sensor] = 1.0
        
        # Inverse error weighting
        inv_errors = {k: 1.0 / (v + 1.0) for k, v in avg_errors.items()}
        total = sum(inv_errors.values())
        
        return SensorWeight(
            radar=inv_errors['radar'] / total,
            satellite=inv_errors['satellite'] / total,
            thermal=inv_errors['thermal'] / total
        )
    
    def fuse(self, measurements: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse with adaptive weights"""
        weights = self.compute_adaptive_weights()
        
        fused = np.zeros(3)
        total_weight = 0.0
        
        if measurements.get('radar') is not None:
            fused += weights.radar * measurements['radar']
            total_weight += weights.radar
        
        if measurements.get('satellite') is not None:
            fused += weights.satellite * measurements['satellite']
            total_weight += weights.satellite
        
        if measurements.get('thermal') is not None:
            fused += weights.thermal * measurements['thermal']
            total_weight += weights.thermal
        
        if total_weight > 0:
            fused /= total_weight
        
        return fused


class MultiHypothesisFusion:
    """
    Multiple Hypothesis Tracking (MHT) approach
    Maintains multiple possible tracks
    """
    
    def __init__(self, max_hypotheses: int = 5):
        self.max_hypotheses = max_hypotheses
        self.hypotheses = []
    
    def fuse(self, measurements: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate and score multiple hypotheses
        Return best hypothesis
        """
        new_hypotheses = []
        
        # Generate hypotheses from each sensor
        for sensor, measurement in measurements.items():
            if measurement is not None:
                new_hypotheses.append({
                    'estimate': measurement,
                    'source': sensor,
                    'score': self._score_measurement(measurement, sensor)
                })
        
        # Also generate fusion hypotheses
        fusion_methods = [
            WeightedAverageFusion(),
            CovarianceIntersectionFusion()
        ]
        
        for method in fusion_methods:
            try:
                if isinstance(method, CovarianceIntersectionFusion):
                    fused, _ = method.fuse(measurements)
                else:
                    fused = method.fuse(measurements)
                
                new_hypotheses.append({
                    'estimate': fused,
                    'source': 'fusion',
                    'score': self._score_measurement(fused, 'fusion')
                })
            except:
                pass
        
        # Keep top hypotheses
        new_hypotheses.sort(key=lambda x: x['score'], reverse=True)
        self.hypotheses = new_hypotheses[:self.max_hypotheses]
        
        # Return best hypothesis
        if self.hypotheses:
            return self.hypotheses[0]['estimate']
        else:
            return np.zeros(3)
    
    def _score_measurement(self, measurement: np.ndarray, source: str) -> float:
        """Score a measurement (simplified)"""
        base_scores = {
            'radar': 0.8,
            'satellite': 0.6,
            'thermal': 0.5,
            'fusion': 0.9
        }
        
        return base_scores.get(source, 0.5)
"""
test_sensor_fusion.py - Unit tests for sensor fusion
"""

import pytest
import numpy as np
from fusion.kalman_filter import KalmanFilter, SensorFusion
from fusion.sensor_combiner import WeightedAverageFusion, SensorWeight

class TestSensorFusion:
    """Test sensor fusion algorithms"""
    
    def test_kalman_filter_initialization(self):
        """Test Kalman filter initialization"""
        kf = KalmanFilter(dt=0.1)
        
        assert kf.state.shape == (6,)
        assert kf.P.shape == (6, 6)
        assert kf.F.shape == (6, 6)
        assert kf.H.shape == (3, 6)
    
    def test_kalman_filter_prediction(self):
        """Test Kalman filter prediction step"""
        kf = KalmanFilter(dt=0.1)
        kf.state = np.array([0, 0, 1000, 100, 50, 10])
        
        predicted_state = kf.predict()
        
        # Position should have moved
        assert predicted_state[0] > 0  # x increased
        assert predicted_state[1] > 0  # y increased
        assert predicted_state[2] > 1000  # z increased
    
    def test_kalman_filter_update(self):
        """Test Kalman filter update step"""
        kf = KalmanFilter(dt=0.1)
        kf.state = np.array([0, 0, 1000, 100, 50, 10])
        
        measurement = np.array([10, 5, 1001])
        measurement_noise = 50.0
        
        updated_state = kf.update(measurement, measurement_noise)
        
        assert updated_state.shape == (6,)
    
    def test_weighted_average_fusion(self):
        """Test weighted average sensor fusion"""
        weights = SensorWeight(radar=0.5, satellite=0.3, thermal=0.2)
        fusion = WeightedAverageFusion(weights)
        
        measurements = {
            'radar': np.array([100, 200, 1000]),
            'satellite': np.array([105, 205, 1005]),
            'thermal': np.array([95, 195, 995])
        }
        
        fused = fusion.fuse(measurements)
        
        assert fused.shape == (3,)
        # Should be close to weighted average
        expected = (0.5 * measurements['radar'] + 
                   0.3 * measurements['satellite'] + 
                   0.2 * measurements['thermal'])
        np.testing.assert_array_almost_equal(fused, expected)
    
    def test_weighted_fusion_missing_sensors(self):
        """Test fusion with missing sensors"""
        fusion = WeightedAverageFusion()
        
        measurements = {
            'radar': np.array([100, 200, 1000]),
            'satellite': None,
            'thermal': None
        }
        
        fused = fusion.fuse(measurements)
        
        # Should return radar measurement when others missing
        np.testing.assert_array_equal(fused, measurements['radar'])

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
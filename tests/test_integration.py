"""
test_integration.py - Integration tests for complete system
"""

import pytest
import numpy as np
import pandas as pd
from simulation.object_simulator import HighSpeedObjectSimulator
from simulation.sensor_simulator import MultiSensorSimulator
from models.physics_models import PhysicsPredictor
from fusion.kalman_filter import SensorFusion

class TestSystemIntegration:
    """Integration tests for complete tracking system"""
    
    def test_end_to_end_tracking(self):
        """Test complete tracking pipeline"""
        
        # 1. Generate trajectory
        simulator = HighSpeedObjectSimulator(
            initial_position=np.array([0, 0, 10000]),
            initial_velocity=np.array([300, 200, 50]),
            dt=0.1
        )
        simulator.simulate_trajectory(duration=10.0)
        trajectory_df = simulator.get_trajectory_dataframe()
        
        assert len(trajectory_df) > 0
        assert 'x' in trajectory_df.columns
        assert 'y' in trajectory_df.columns
        assert 'z' in trajectory_df.columns
        
        # 2. Simulate sensors
        sensor_sim = MultiSensorSimulator()
        measurements_df = sensor_sim.generate_sensor_measurements(trajectory_df)
        
        assert len(measurements_df) > 0
        assert 'sensor_type' in measurements_df.columns
        
        # 3. Sensor fusion
        fusion = SensorFusion(dt=0.1)
        
        first_valid = measurements_df[measurements_df['detected'] == True].iloc[0]
        fusion.initialize_from_measurement(
            np.array([first_valid['x_measured'],
                     first_valid['y_measured'],
                     first_valid['z_measured']])
        )
        
        fusion_df = fusion.process_measurements(measurements_df)
        
        assert len(fusion_df) > 0
        assert 'x_fused' in fusion_df.columns
        
        # 4. Verify tracking performance
        merged = trajectory_df.merge(fusion_df, on='time', how='inner')
        
        true_pos = merged[['x', 'y', 'z']].values
        fused_pos = merged[['x_fused', 'y_fused', 'z_fused']].values
        
        errors = np.linalg.norm(true_pos - fused_pos, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        
        # RMSE should be reasonable (< 200m for this setup)
        assert rmse < 200.0
    
    def test_physics_prediction_pipeline(self):
        """Test physics prediction pipeline"""
        
        # Generate simple trajectory
        simulator = HighSpeedObjectSimulator(
            initial_position=np.array([0, 0, 10000]),
            initial_velocity=np.array([100, 100, 0]),
            dt=0.1
        )
        simulator.simulate_trajectory(duration=5.0)
        trajectory_df = simulator.get_trajectory_dataframe()
        
        # Physics predictor
        predictor = PhysicsPredictor(dt=0.1)
        
        predictions = []
        for idx, row in trajectory_df.iterrows():
            if idx < 5:
                predictor.update(
                    row[['x', 'y', 'z']].values,
                    row[['vx', 'vy', 'vz']].values
                )
                continue
            
            pred, uncertainty, model_type = predictor.predict_next()
            predictions.append(pred)
            
            predictor.update(
                row[['x', 'y', 'z']].values,
                row[['vx', 'vy', 'vz']].values
            )
        
        assert len(predictions) > 0
        
        # Calculate prediction accuracy
        true_positions = trajectory_df[['x', 'y', 'z']].values[5:]
        predictions = np.array(predictions)
        
        errors = np.linalg.norm(predictions - true_positions, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Physics predictions should be reasonable
        assert rmse < 100.0  # For constant velocity, should be quite accurate

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
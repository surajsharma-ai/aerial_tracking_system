"""
test_physics_models.py - Unit tests for physics models
"""

import pytest
import numpy as np
from models.physics_models import (
    PhysicsState,
    ConstantVelocityModel,
    ConstantAccelerationModel,
    CoordinatedTurnModel,
    PhysicsPredictor
)

class TestPhysicsModels:
    """Test physics-based prediction models"""
    
    def test_constant_velocity_prediction(self):
        """Test constant velocity model"""
        model = ConstantVelocityModel(dt=0.1)
        
        state = PhysicsState(
            position=np.array([0.0, 0.0, 1000.0]),
            velocity=np.array([100.0, 50.0, 10.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            theta=0.0,
            phi=0.0
        )
        
        next_state = model.predict(state)
        
        # Check position updated correctly
        expected_pos = np.array([10.0, 5.0, 1001.0])  # pos + vel*dt
        np.testing.assert_array_almost_equal(next_state.position, expected_pos)
        
        # Velocity should remain constant
        np.testing.assert_array_equal(next_state.velocity, state.velocity)
    
    def test_constant_acceleration_prediction(self):
        """Test constant acceleration model"""
        model = ConstantAccelerationModel(dt=0.1)
        
        state = PhysicsState(
            position=np.array([0.0, 0.0, 1000.0]),
            velocity=np.array([100.0, 50.0, 10.0]),
            acceleration=np.array([10.0, 5.0, -9.81]),
            theta=0.0,
            phi=0.0
        )
        
        next_state = model.predict(state)
        
        # Check position: pos + vel*dt + 0.5*acc*dt^2
        expected_pos = np.array([10.05, 5.025, 1000.951])
        np.testing.assert_array_almost_equal(next_state.position, expected_pos, decimal=2)
        
        # Check velocity: vel + acc*dt
        expected_vel = np.array([101.0, 50.5, 9.019])
        np.testing.assert_array_almost_equal(next_state.velocity, expected_vel, decimal=2)
    
    def test_coordinated_turn(self):
        """Test coordinated turn model"""
        model = CoordinatedTurnModel(dt=0.1)
        
        state = PhysicsState(
            position=np.array([0.0, 0.0, 1000.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            theta=0.0,
            phi=0.0
        )
        
        turn_rate = 0.1  # radians per second
        next_state = model.predict(state, turn_rate)
        
        # Position should have moved
        assert next_state.position[0] > state.position[0]
        
        # Heading should have changed
        assert next_state.theta > state.theta
    
    def test_physics_predictor(self):
        """Test physics predictor wrapper"""
        predictor = PhysicsPredictor(dt=0.1)
        
        # Initialize with some observations
        for i in range(10):
            position = np.array([i * 10.0, i * 5.0, 1000.0])
            velocity = np.array([100.0, 50.0, 0.0])
            predictor.update(position, velocity)
        
        # Make prediction
        pred_position, uncertainty, model_type = predictor.predict_next()
        
        assert pred_position.shape == (3,)
        assert uncertainty > 0
        assert model_type in ['constant_velocity', 'constant_acceleration', 'coordinated_turn']
    
    def test_multi_step_prediction(self):
        """Test multi-step prediction"""
        predictor = PhysicsPredictor(dt=0.1)
        
        # Initialize
        for i in range(5):
            position = np.array([i * 10.0, 0.0, 1000.0])
            velocity = np.array([100.0, 0.0, 0.0])
            predictor.update(position, velocity)
        
        # Predict 10 steps ahead
        result = predictor.predict_trajectory(n_steps=10)
        
        assert result['predictions'].shape == (10, 3)
        assert len(result['uncertainties']) == 10
        assert len(result['model_types']) == 10

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
"""
models/physics_models.py - Physics-based trajectory prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class PhysicsState:
    """Physical state of the object"""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    theta: float  # heading angle (radians)
    phi: float  # elevation angle (radians)
    
class ConstantVelocityModel:
    """Simple constant velocity physics model"""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
    
    def predict(self, state: PhysicsState) -> PhysicsState:
        """
        Predict next state using constant velocity assumption
        
        x_next = x + vx * dt
        y_next = y + vy * dt
        z_next = z + vz * dt
        """
        new_position = state.position + state.velocity * self.dt
        
        return PhysicsState(
            position=new_position,
            velocity=state.velocity.copy(),
            acceleration=state.acceleration.copy(),
            theta=state.theta,
            phi=state.phi
        )
    
    def predict_n_steps(self, state: PhysicsState, n_steps: int) -> np.ndarray:
        """Predict n steps ahead"""
        predictions = []
        current_state = state
        
        for _ in range(n_steps):
            current_state = self.predict(current_state)
            predictions.append(current_state.position.copy())
        
        return np.array(predictions)

class ConstantAccelerationModel:
    """Constant acceleration physics model"""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
    
    def predict(self, state: PhysicsState) -> PhysicsState:
        """
        Predict using constant acceleration
        
        x_next = x + vx*dt + 0.5*ax*dt²
        vx_next = vx + ax*dt
        """
        # Update position
        new_position = (state.position + 
                       state.velocity * self.dt + 
                       0.5 * state.acceleration * self.dt**2)
        
        # Update velocity
        new_velocity = state.velocity + state.acceleration * self.dt
        
        return PhysicsState(
            position=new_position,
            velocity=new_velocity,
            acceleration=state.acceleration.copy(),
            theta=state.theta,
            phi=state.phi
        )
    
    def predict_n_steps(self, state: PhysicsState, n_steps: int) -> np.ndarray:
        """Predict n steps ahead"""
        predictions = []
        current_state = state
        
        for _ in range(n_steps):
            current_state = self.predict(current_state)
            predictions.append(current_state.position.copy())
        
        return np.array(predictions)

class CoordinatedTurnModel:
    """Coordinated turn model for aerial objects"""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
    
    def predict(self, state: PhysicsState, turn_rate: float = 0.0) -> PhysicsState:
        """
        Predict with turn dynamics
        
        x_next = x + v*cos(theta)*dt
        y_next = y + v*sin(theta)*dt
        theta_next = theta + turn_rate*dt
        """
        speed = np.linalg.norm(state.velocity[:2])  # Horizontal speed
        
        # Update heading
        new_theta = state.theta + turn_rate * self.dt
        
        # Update horizontal position
        dx = speed * np.cos(new_theta) * self.dt
        dy = speed * np.sin(new_theta) * self.dt
        dz = state.velocity[2] * self.dt
        
        new_position = state.position + np.array([dx, dy, dz])
        
        # Update velocity direction
        new_velocity = np.array([
            speed * np.cos(new_theta),
            speed * np.sin(new_theta),
            state.velocity[2]
        ])
        
        return PhysicsState(
            position=new_position,
            velocity=new_velocity,
            acceleration=state.acceleration.copy(),
            theta=new_theta,
            phi=state.phi
        )
    
    def predict_n_steps(self, state: PhysicsState, n_steps: int, 
                       turn_rate: float = 0.0) -> np.ndarray:
        """Predict n steps ahead with turning"""
        predictions = []
        current_state = state
        
        for _ in range(n_steps):
            current_state = self.predict(current_state, turn_rate)
            predictions.append(current_state.position.copy())
        
        return np.array(predictions)

class HybridPhysicsModel:
    """
    Hybrid physics model that combines multiple motion models
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.cv_model = ConstantVelocityModel(dt)
        self.ca_model = ConstantAccelerationModel(dt)
        self.ct_model = CoordinatedTurnModel(dt)
        
    def estimate_motion_type(self, state_history: list) -> str:
        """
        Estimate which motion model is most appropriate
        based on recent state history
        """
        if len(state_history) < 3:
            return 'constant_velocity'
        
        # Calculate recent acceleration
        recent_velocities = [s.velocity for s in state_history[-3:]]
        vel_changes = np.diff(recent_velocities, axis=0)
        avg_accel = np.mean(np.linalg.norm(vel_changes, axis=1))
        
        # Calculate heading changes
        recent_thetas = [s.theta for s in state_history[-3:]]
        theta_changes = np.diff(recent_thetas)
        avg_turn_rate = np.mean(np.abs(theta_changes)) / self.dt
        
        # Decision logic
        if avg_turn_rate > 0.05:  # Significant turning
            return 'coordinated_turn'
        elif avg_accel > 5.0:  # Significant acceleration
            return 'constant_acceleration'
        else:
            return 'constant_velocity'
    
    def predict(self, state: PhysicsState, 
                state_history: list = None,
                turn_rate: float = 0.0) -> Tuple[PhysicsState, str]:
        """
        Predict using most appropriate model
        
        Returns:
            predicted_state, model_type
        """
        if state_history:
            model_type = self.estimate_motion_type(state_history)
        else:
            model_type = 'constant_velocity'
        
        if model_type == 'coordinated_turn':
            predicted = self.ct_model.predict(state, turn_rate)
        elif model_type == 'constant_acceleration':
            predicted = self.ca_model.predict(state)
        else:
            predicted = self.cv_model.predict(state)
        
        return predicted, model_type
    
    def predict_n_steps(self, state: PhysicsState, n_steps: int,
                       state_history: list = None) -> Dict:
        """
        Predict multiple steps with model selection
        
        Returns:
            Dictionary with predictions and metadata
        """
        predictions = []
        uncertainties = []
        model_types = []
        
        current_state = state
        history = state_history if state_history else []
        
        for step in range(n_steps):
            # Predict next state
            next_state, model_type = self.predict(current_state, history)
            
            predictions.append(next_state.position.copy())
            model_types.append(model_type)
            
            # Uncertainty increases with prediction horizon
            uncertainty = 10.0 * (step + 1)  # Simple linear growth
            uncertainties.append(uncertainty)
            
            # Update history
            history.append(current_state)
            current_state = next_state
        
        return {
            'predictions': np.array(predictions),
            'uncertainties': np.array(uncertainties),
            'model_types': model_types,
            'final_state': current_state
        }

class PhysicsPredictor:
    """
    Main physics-based prediction engine
    Provides baseline predictions for ML correction
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.model = HybridPhysicsModel(dt)
        self.state_history = []
        
    def state_from_observation(self, position: np.ndarray, 
                              velocity: np.ndarray = None) -> PhysicsState:
        """Create physics state from observation"""
        
        if velocity is None:
            # Estimate velocity from history if available
            if len(self.state_history) > 0:
                velocity = (position - self.state_history[-1].position) / self.dt
            else:
                velocity = np.zeros(3)
        
        # Calculate heading and elevation
        theta = np.arctan2(velocity[1], velocity[0])
        horizontal_speed = np.linalg.norm(velocity[:2])
        phi = np.arctan2(velocity[2], horizontal_speed)
        
        # Estimate acceleration from history
        if len(self.state_history) > 1:
            prev_vel = self.state_history[-1].velocity
            acceleration = (velocity - prev_vel) / self.dt
        else:
            acceleration = np.zeros(3)
        
        return PhysicsState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            theta=theta,
            phi=phi
        )
    
    def update(self, position: np.ndarray, velocity: np.ndarray = None):
        """Update with new observation"""
        state = self.state_from_observation(position, velocity)
        self.state_history.append(state)
        
        # Keep only recent history
        if len(self.state_history) > 50:
            self.state_history = self.state_history[-50:]
    
    def predict_next(self) -> Tuple[np.ndarray, float, str]:
        """
        Predict next position using physics
        
        Returns:
            position, uncertainty, model_type
        """
        if not self.state_history:
            raise ValueError("No state history available")
        
        current_state = self.state_history[-1]
        next_state, model_type = self.model.predict(current_state, self.state_history)
        
        # Estimate uncertainty based on model confidence
        uncertainty = self.estimate_uncertainty(model_type)
        
        return next_state.position, uncertainty, model_type
    
    def predict_trajectory(self, n_steps: int = 10) -> Dict:
        """
        Predict full trajectory
        
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.state_history:
            raise ValueError("No state history available")
        
        current_state = self.state_history[-1]
        return self.model.predict_n_steps(current_state, n_steps, self.state_history)
    
    def estimate_uncertainty(self, model_type: str) -> float:
        """Estimate prediction uncertainty based on model type"""
        base_uncertainties = {
            'constant_velocity': 20.0,
            'constant_acceleration': 35.0,
            'coordinated_turn': 50.0
        }
        return base_uncertainties.get(model_type, 30.0)
    
    def get_physics_features(self) -> np.ndarray:
        """
        Extract physics-based features for ML model
        
        Returns:
            Feature vector containing:
            - current position
            - current velocity
            - current acceleration
            - heading, elevation
            - speed, turn rate
        """
        if not self.state_history:
            return np.zeros(12)
        
        state = self.state_history[-1]
        
        # Calculate derived features
        speed = np.linalg.norm(state.velocity)
        horizontal_speed = np.linalg.norm(state.velocity[:2])
        
        # Turn rate from history
        if len(self.state_history) > 1:
            theta_change = state.theta - self.state_history[-2].theta
            turn_rate = theta_change / self.dt
        else:
            turn_rate = 0.0
        
        features = np.array([
            *state.position,      # x, y, z (3)
            *state.velocity,      # vx, vy, vz (3)
            *state.acceleration,  # ax, ay, az (3)
            state.theta,          # heading (1)
            state.phi,            # elevation (1)
            speed,                # total speed (1)
            turn_rate             # turn rate (1)
        ])
        
        return features
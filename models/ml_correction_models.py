"""
models/ml_correction_models.py - ML models for correcting physics predictions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from typing import Tuple, Dict
import joblib

class LinearCorrectionModel:
    """
    Simple linear regression for error correction
    Fast and interpretable baseline
    """
    
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.is_trained = False
    
    def prepare_features(self, physics_pred: np.ndarray,
                        sensor_measurements: np.ndarray,
                        physics_features: np.ndarray) -> np.ndarray:
        """
        Combine features for correction
        
        Features:
        - Physics prediction (3)
        - Sensor measurements (9 = 3 sensors × 3 coords)
        - Physics features (12)
        """
        # Flatten sensor measurements
        sensor_flat = sensor_measurements.flatten()
        
        # Combine all features
        features = np.concatenate([
            physics_pred,
            sensor_flat,
            physics_features
        ])
        
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train correction model
        
        X: features
        y: correction needed (true_position - physics_prediction)
        """
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict_correction(self, features: np.ndarray) -> np.ndarray:
        """Predict position correction"""
        if not self.is_trained:
            return np.zeros(3)
        
        return self.model.predict(features.reshape(1, -1))[0]
    
    def save(self, path: str):
        """Save model"""
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """Load model"""
        self.model = joblib.load(path)
        self.is_trained = True

class RandomForestCorrectionModel:
    """
    Random Forest for non-linear error patterns
    Better at capturing complex relationships
    """
    
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
    
    def prepare_features(self, physics_pred: np.ndarray,
                        sensor_measurements: np.ndarray,
                        physics_features: np.ndarray,
                        past_errors: np.ndarray = None) -> np.ndarray:
        """
        Enhanced feature preparation with error history
        """
        sensor_flat = sensor_measurements.flatten()
        
        features = [
            physics_pred,
            sensor_flat,
            physics_features
        ]
        
        # Add past error information if available
        if past_errors is not None:
            features.append(past_errors.flatten())
        
        return np.concatenate(features)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict_correction(self, features: np.ndarray) -> np.ndarray:
        """Predict correction with uncertainty estimate"""
        if not self.is_trained:
            return np.zeros(3), np.zeros(3)
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(features.reshape(1, -1))[0] 
                               for tree in self.model.estimators_])
        
        # Mean prediction
        correction = np.mean(predictions, axis=0)
        
        # Uncertainty (standard deviation across trees)
        uncertainty = np.std(predictions, axis=0)
        
        return correction, uncertainty
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_trained:
            return None
        return self.model.feature_importances_
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_trained = True

class LSTMCorrectionModel(nn.Module):
    """
    LSTM-based correction model
    Best for learning temporal error patterns
    """
    
    def __init__(self, 
                 feature_dim: int = 24,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 output_dim: int = 3):
        super(LSTMCorrectionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layers
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        x: (batch, seq_len, feature_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Process features
        x = self.feature_fc(x)
        
        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]
        
        # Correction prediction
        correction = self.output_fc(last_output)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_fc(last_output)
        
        return correction, uncertainty, hidden

class HybridCorrectionSystem:
    """
    Hybrid system combining Physics + ML Correction
    
    Final Prediction = Physics Prediction + ML Correction
    """
    
    def __init__(self, 
                 physics_predictor,
                 correction_model_type: str = 'random_forest'):
        """
        Args:
            physics_predictor: PhysicsPredictor instance
            correction_model_type: 'linear', 'random_forest', or 'lstm'
        """
        self.physics_predictor = physics_predictor
        self.correction_model_type = correction_model_type
        
        # Initialize correction model
        if correction_model_type == 'linear':
            self.correction_model = LinearCorrectionModel()
        elif correction_model_type == 'random_forest':
            self.correction_model = RandomForestCorrectionModel(n_estimators=100)
        elif correction_model_type == 'lstm':
            self.correction_model = LSTMCorrectionModel()
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.correction_model.to(self.device)
        else:
            raise ValueError(f"Unknown model type: {correction_model_type}")
        
        self.prediction_history = []
        self.error_history = []
    
    def predict(self, 
                sensor_measurements: Dict[str, np.ndarray],
                return_components: bool = False) -> Dict:
        """
        Make hybrid prediction
        
        Args:
            sensor_measurements: dict of sensor_name -> position measurement
            return_components: if True, return physics and correction separately
        
        Returns:
            Dictionary with prediction and metadata
        """
        # Step 1: Physics-based prediction
        physics_pred, physics_uncertainty, model_type = self.physics_predictor.predict_next()
        
        # Step 2: Prepare features for ML
        physics_features = self.physics_predictor.get_physics_features()
        
        # Combine sensor measurements
        sensor_array = self._combine_sensor_measurements(sensor_measurements)
        
        # Get recent errors if available
        past_errors = np.array(self.error_history[-5:]) if self.error_history else None
        
        # Step 3: ML correction
        features = self._prepare_ml_features(
            physics_pred, sensor_array, physics_features, past_errors
        )
        
        ml_correction = self._predict_correction(features)
        
        # Step 4: Combine predictions
        final_prediction = physics_pred + ml_correction
        
        # Store prediction
        result = {
            'final_prediction': final_prediction,
            'physics_prediction': physics_pred,
            'ml_correction': ml_correction,
            'physics_uncertainty': physics_uncertainty,
            'model_type': model_type,
            'sensor_measurements': sensor_measurements
        }
        
        self.prediction_history.append(result)
        
        if return_components:
            return result
        else:
            return final_prediction
    
    def predict_trajectory(self, n_steps: int = 10) -> Dict:
        """
        Predict future trajectory using hybrid approach
        """
        # Get physics trajectory
        physics_result = self.physics_predictor.predict_trajectory(n_steps)
        
        # Apply ML correction to each step
        corrected_predictions = []
        
        for i, physics_pred in enumerate(physics_result['predictions']):
            # Simple correction (could be enhanced with step-specific features)
            if self.correction_model.is_trained:
                # Use average correction pattern
                avg_correction = self._get_average_correction()
                corrected_pred = physics_pred + avg_correction
            else:
                corrected_pred = physics_pred.copy()
            
            corrected_predictions.append(corrected_pred)
        
        return {
            'predictions': np.array(corrected_predictions),
            'physics_predictions': physics_result['predictions'],
            'uncertainties': physics_result['uncertainties'],
            'model_types': physics_result['model_types']
        }
    
    def update_with_truth(self, true_position: np.ndarray):
        """
        Update with ground truth for learning
        """
        if self.prediction_history:
            last_pred = self.prediction_history[-1]
            error = true_position - last_pred['final_prediction']
            physics_error = true_position - last_pred['physics_prediction']
            
            self.error_history.append(physics_error)
            
            # Keep limited history
            if len(self.error_history) > 100:
                self.error_history = self.error_history[-100:]
    
    def _combine_sensor_measurements(self, measurements: Dict) -> np.ndarray:
        """Combine measurements from multiple sensors"""
        # Expected sensors: radar, satellite, thermal
        combined = []
        
        for sensor_name in ['radar', 'satellite', 'thermal']:
            if sensor_name in measurements and measurements[sensor_name] is not None:
                combined.append(measurements[sensor_name])
            else:
                combined.append(np.zeros(3))  # Missing measurement
        
        return np.array(combined)
    
    def _prepare_ml_features(self, physics_pred, sensor_array, 
                           physics_features, past_errors):
        """Prepare features for ML model"""
        if self.correction_model_type == 'random_forest':
            return self.correction_model.prepare_features(
                physics_pred, sensor_array, physics_features, past_errors
            )
        else:
            return self.correction_model.prepare_features(
                physics_pred, sensor_array, physics_features
            )
    
    def _predict_correction(self, features: np.ndarray) -> np.ndarray:
        """Predict correction using ML model"""
        if not self.correction_model.is_trained:
            return np.zeros(3)
        
        if self.correction_model_type == 'lstm':
            # LSTM requires sequence
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            features_tensor = features_tensor.to(self.device)
            
            with torch.no_grad():
                correction, _, _ = self.correction_model(features_tensor)
                correction = correction.cpu().numpy()[0]
            
            return correction
        elif self.correction_model_type == 'random_forest':
            correction, _ = self.correction_model.predict_correction(features)
            return correction
        else:
            return self.correction_model.predict_correction(features)
    
    def _get_average_correction(self) -> np.ndarray:
        """Get average correction from recent history"""
        if not self.prediction_history:
            return np.zeros(3)
        
        recent_corrections = [p['ml_correction'] for p in self.prediction_history[-10:]]
        return np.mean(recent_corrections, axis=0)
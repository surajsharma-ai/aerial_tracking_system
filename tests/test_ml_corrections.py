"""
test_ml_corrections.py - Unit tests for ML correction models
"""

import pytest
import numpy as np
from models.ml_correction_models import (
    LinearCorrectionModel,
    RandomForestCorrectionModel,
)

class TestMLCorrectionModels:
    """Test ML correction models"""
    
    def test_linear_correction_model(self):
        """Test linear correction model"""
        model = LinearCorrectionModel()
        
        # Generate synthetic training data
        n_samples = 100
        X = np.random.randn(n_samples, 24)  # 24 features
        y = np.random.randn(n_samples, 3)   # 3D correction
        
        # Train
        model.train(X, y)
        assert model.is_trained
        
        # Predict
        test_features = np.random.randn(24)
        correction = model.predict_correction(test_features)
        
        assert correction.shape == (3,)
    
    def test_random_forest_correction(self):
        """Test random forest correction model"""
        model = RandomForestCorrectionModel(n_estimators=10)
        
        # Generate synthetic training data
        n_samples = 100
        X = np.random.randn(n_samples, 24)
        y = np.random.randn(n_samples, 3)
        
        # Train
        model.train(X, y)
        assert model.is_trained
        
        # Predict with uncertainty
        test_features = np.random.randn(24)
        correction, uncertainty = model.predict_correction(test_features)
        
        assert correction.shape == (3,)
        assert uncertainty.shape == (3,)
        assert np.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        model = RandomForestCorrectionModel(n_estimators=10)
        
        n_samples = 100
        X = np.random.randn(n_samples, 24)
        y = np.random.randn(n_samples, 3)
        
        model.train(X, y)
        
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == 24
        assert np.all(importance >= 0)
        assert np.isclose(np.sum(importance), 1.0)  # Should sum to 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
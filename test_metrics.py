"""
test_metrics.py - Test metrics module
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing metrics module...")

try:
    import numpy as np
    print("✓ NumPy imported")
    
    import pandas as pd
    print("✓ Pandas imported")
    
    from sklearn.metrics import mean_squared_error
    print("✓ Sklearn imported")
    
    from utils.metrics import TrackingMetrics
    print("✓ TrackingMetrics imported")
    
    from utils.metrics import MetricsVisualizer
    print("✓ MetricsVisualizer imported")
    
    # Test basic functionality
    print("\nTesting TrackingMetrics...")
    
    predicted = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    actual = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]])
    
    errors = TrackingMetrics.position_error(predicted, actual)
    print(f"  Position errors: {errors}")
    
    rmse = TrackingMetrics.rmse(predicted, actual)
    print(f"  RMSE: {rmse:.4f}")
    
    mae = TrackingMetrics.mae(predicted, actual)
    print(f"  MAE: {mae:.4f}")
    
    print("\n✅ All metrics tests passed!")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nMake sure you have installed all dependencies:")
    print("  pip install numpy pandas scikit-learn matplotlib")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
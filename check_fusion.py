"""
check_fusion.py - Verify fusion module
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Checking fusion module...")

try:
    from fusion.kalman_filter import KalmanFilter, SensorFusion
    print("✓ KalmanFilter imported successfully")
    print("✓ SensorFusion imported successfully")
    
    # Test instantiation
    kf = KalmanFilter(dt=0.1)
    print(f"✓ KalmanFilter created: state shape = {kf.state.shape}")
    
    sf = SensorFusion(dt=0.1)
    print(f"✓ SensorFusion created")
    
    print("\n✅ All fusion module checks passed!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure:")
    print("  1. File exists: fusion/kalman_filter.py")
    print("  2. File exists: fusion/__init__.py")
    print("  3. Files are not empty")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
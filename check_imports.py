"""
check_imports.py - Diagnose import issues
"""

import sys
from pathlib import Path

print("="*60)
print("Checking Project Imports")
print("="*60)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n1. Testing simulation package...")
try:
    from simulation.object_simulator import HighSpeedObjectSimulator
    print("  ✓ HighSpeedObjectSimulator")
except ImportError as e:
    print(f"  ✗ HighSpeedObjectSimulator: {e}")

try:
    from simulation.sensor_simulator import MultiSensorSimulator
    print("  ✓ MultiSensorSimulator")
except ImportError as e:
    print(f"  ✗ MultiSensorSimulator: {e}")

try:
    from simulation.sensor_simulator import SensorCharacteristics
    print("  ✓ SensorCharacteristics")
except ImportError as e:
    print(f"  ✗ SensorCharacteristics: {e}")

print("\n2. Testing models package...")
try:
    from models.physics_models import PhysicsPredictor
    print("  ✓ PhysicsPredictor")
except ImportError as e:
    print(f"  ✗ PhysicsPredictor: {e}")

try:
    from models.ml_correction_models import RandomForestCorrectionModel
    print("  ✓ RandomForestCorrectionModel")
except ImportError as e:
    print(f"  ✗ RandomForestCorrectionModel: {e}")

print("\n3. Testing fusion package...")
try:
    from fusion.kalman_filter import KalmanFilter, SensorFusion
    print("  ✓ KalmanFilter, SensorFusion")
except ImportError as e:
    print(f"  ✗ Kalman filter: {e}")

print("\n4. Testing utils package...")
try:
    from utils.metrics import TrackingMetrics
    print("  ✓ TrackingMetrics")
except ImportError as e:
    print(f"  ✗ TrackingMetrics: {e}")

try:
    from utils.helpers import create_directory_structure
    print("  ✓ Helpers")
except ImportError as e:
    print(f"  ✗ Helpers: {e}")

print("\n5. Testing visualization package...")
try:
    from visualization.plots import TrajectoryVisualizer
    print("  ✓ TrajectoryVisualizer")
except ImportError as e:
    print(f"  ✗ TrajectoryVisualizer: {e}")

print("\n" + "="*60)
print("Import Check Complete")
print("="*60)
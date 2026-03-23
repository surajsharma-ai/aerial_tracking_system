"""
check_utils.py - Verify utils module
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Checking utils module...")

try:
    from utils.helpers import create_directory_structure
    print("✓ create_directory_structure imported")
    
    from utils.helpers import Logger
    print("✓ Logger imported")
    
    from utils.helpers import DataProcessor
    print("✓ DataProcessor imported")
    
    from utils.helpers import ConfigManager
    print("✓ ConfigManager imported")
    
    from utils.helpers import ModelSaver
    print("✓ ModelSaver imported")
    
    from utils.helpers import calculate_trajectory_statistics
    print("✓ calculate_trajectory_statistics imported")
    
    from utils.metrics import TrackingMetrics
    print("✓ TrackingMetrics imported")
    
    from utils.metrics import MetricsVisualizer
    print("✓ MetricsVisualizer imported")
    
    # Test create_directory_structure
    print("\nTesting create_directory_structure...")
    create_directory_structure()
    
    print("\n✅ All utils module checks passed!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure:")
    print("  1. File exists: utils/helpers.py")
    print("  2. File exists: utils/metrics.py")
    print("  3. File exists: utils/__init__.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
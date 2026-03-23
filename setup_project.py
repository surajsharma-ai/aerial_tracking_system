"""
setup_project.py - Complete project setup including directory structure and .gitkeep files
"""

import os
from pathlib import Path

def setup_project_structure():
    """Create complete directory structure with .gitkeep files"""
    
    print("="*60)
    print("Setting up Aerial Tracking System Project Structure")
    print("="*60)
    
    # Define all directories
    directories = [
        'data/raw',
        'data/processed',
        'data/simulated',
        'models/saved_models',
        'simulation',
        'fusion',
        'visualization',
        'utils',
        'tests',
        'logs',
        'results/plots',
        'results/metrics',
        'scripts',
        'notebooks',
        'docs/images',
    ]
    
    project_root = Path(__file__).parent
    
    # Directories that need .gitkeep
    gitkeep_dirs = [
        'data/raw',
        'data/processed',
        'data/simulated',
        'models/saved_models',
        'logs',
        'results/plots',
        'results/metrics',
    ]
    
    # Create all directories
    print("\n📁 Creating directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    # Create .gitkeep files
    print("\n📄 Creating .gitkeep files...")
    for directory in gitkeep_dirs:
        gitkeep_path = project_root / directory / '.gitkeep'
        gitkeep_path.touch()
        print(f"  ✓ {directory}/.gitkeep")
    
    # Create __init__.py files for packages
    print("\n🐍 Creating __init__.py files...")
    packages = ['simulation', 'models', 'fusion', 'visualization', 'utils', 'tests']
    
    for package in packages:
        init_path = project_root / package / '__init__.py'
        if not init_path.exists():
            init_path.touch()
            print(f"  ✓ {package}/__init__.py")
    
    # Verify structure
    print("\n✅ Project structure created successfully!")
    print("\nDirectory tree:")
    print_tree(project_root, max_depth=2)

def print_tree(directory, prefix="", max_depth=2, current_depth=0):
    """Print directory tree structure"""
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, entry in enumerate(entries):
            if entry.name.startswith('.') and entry.name not in ['.gitkeep']:
                continue
            
            is_last = i == len(entries) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{entry.name}")
            
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                print_tree(entry, prefix + extension, max_depth, current_depth + 1)
    except PermissionError:
        pass

if __name__ == "__main__":
    setup_project_structure()
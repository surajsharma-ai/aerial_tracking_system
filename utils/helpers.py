"""
utils/helpers.py - Helper functions for data processing and utilities
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import json
import pickle
from pathlib import Path
import os

class DataProcessor:
    """Data preprocessing and normalization utilities"""
    
    @staticmethod
    def normalize_trajectory(data: np.ndarray,
                           mean: np.ndarray = None,
                           std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize trajectory data
        
        Args:
            data: (N, features) trajectory data
            mean: precomputed mean (optional)
            std: precomputed std (optional)
        
        Returns:
            normalized_data, mean, std
        """
        if mean is None:
            mean = np.mean(data, axis=0)
        if std is None:
            std = np.std(data, axis=0)
            std[std == 0] = 1.0  # Avoid division by zero
        
        normalized = (data - mean) / std
        return normalized, mean, std
    
    @staticmethod
    def denormalize_trajectory(normalized_data: np.ndarray,
                              mean: np.ndarray,
                              std: np.ndarray) -> np.ndarray:
        """Reverse normalization"""
        return normalized_data * std + mean
    
    @staticmethod
    def interpolate_missing_measurements(df: pd.DataFrame,
                                        time_col: str = 'time',
                                        method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing sensor measurements"""
        df_sorted = df.sort_values(time_col)
        df_interpolated = df_sorted.interpolate(method=method)
        return df_interpolated
    
    @staticmethod
    def resample_trajectory(df: pd.DataFrame,
                           target_dt: float = 0.1,
                           time_col: str = 'time') -> pd.DataFrame:
        """Resample trajectory to uniform time steps"""
        df_sorted = df.sort_values(time_col).copy()
        
        # Create uniform time grid
        t_min = df_sorted[time_col].min()
        t_max = df_sorted[time_col].max()
        new_times = np.arange(t_min, t_max, target_dt)
        
        # Interpolate each column
        result = pd.DataFrame({time_col: new_times})
        
        for col in df_sorted.columns:
            if col != time_col:
                result[col] = np.interp(new_times, df_sorted[time_col], df_sorted[col])
        
        return result


class ConfigManager:
    """Configuration management"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def save_config(config: Dict, config_path: str):
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    @staticmethod
    def get_default_config() -> Dict:
        """Return default configuration"""
        return {
            'simulation': {
                'duration': 100.0,
                'dt': 0.1,
                'initial_position': [0, 0, 10000],
                'initial_velocity': [300, 200, 50]
            },
            'sensors': {
                'radar': {
                    'noise_std': 50.0,
                    'update_rate': 10.0,
                    'detection_prob': 0.98
                },
                'satellite': {
                    'noise_std': 100.0,
                    'update_rate': 1.0,
                    'detection_prob': 0.85
                },
                'thermal': {
                    'noise_std': 150.0,
                    'update_rate': 5.0,
                    'detection_prob': 0.75
                }
            },
            'model': {
                'sequence_length': 20,
                'prediction_horizon': 10,
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 100
            }
        }


class ModelSaver:
    """Model and data saving utilities"""
    
    @staticmethod
    def save_model(model, path: str):
        """Save PyTorch model"""
        import torch
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model(model, path: str):
        """Load PyTorch model"""
        import torch
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        print(f"Model loaded from {path}")
        return model
    
    @staticmethod
    def save_scaler(mean: np.ndarray, std: np.ndarray, path: str):
        """Save normalization parameters"""
        scaler_params = {'mean': mean, 'std': std}
        with open(path, 'wb') as f:
            pickle.dump(scaler_params, f)
        print(f"Scaler saved to {path}")
    
    @staticmethod
    def load_scaler(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load normalization parameters"""
        with open(path, 'rb') as f:
            scaler_params = pickle.load(f)
        return scaler_params['mean'], scaler_params['std']


class Logger:
    """Simple logging utility"""
    
    def __init__(self, log_file: str = 'logs/training.log'):
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, print_console: bool = True):
        """Log message to file and optionally console"""
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        if print_console:
            print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')


def create_directory_structure():
    """Create project directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'data/simulated',
        'models/saved_models',
        'logs',
        'results/plots',
        'results/metrics'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created successfully")


def calculate_trajectory_statistics(trajectory_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive trajectory statistics"""
    
    # Calculate distances
    positions = trajectory_df[['x', 'y', 'z']].values
    distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    
    stats = {
        'duration': trajectory_df['time'].max(),
        'total_distance': np.sum(distances),
        'mean_speed': trajectory_df['speed'].mean() if 'speed' in trajectory_df.columns else np.mean(distances) / (trajectory_df['time'].iloc[1] - trajectory_df['time'].iloc[0]),
        'max_speed': trajectory_df['speed'].max() if 'speed' in trajectory_df.columns else np.max(distances) / (trajectory_df['time'].iloc[1] - trajectory_df['time'].iloc[0]),
        'min_speed': trajectory_df['speed'].min() if 'speed' in trajectory_df.columns else np.min(distances) / (trajectory_df['time'].iloc[1] - trajectory_df['time'].iloc[0]),
        'mean_altitude': trajectory_df['z'].mean(),
        'max_altitude': trajectory_df['z'].max(),
        'min_altitude': trajectory_df['z'].min(),
        'position_range': {
            'x': [trajectory_df['x'].min(), trajectory_df['x'].max()],
            'y': [trajectory_df['y'].min(), trajectory_df['y'].max()],
            'z': [trajectory_df['z'].min(), trajectory_df['z'].max()]
        }
    }
    
    return stats


def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def format_time(seconds: float) -> str:
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """Pretty print metrics dictionary"""
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print("="*50 + "\n")
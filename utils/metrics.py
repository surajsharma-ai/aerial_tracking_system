"""
utils/metrics.py - Performance evaluation metrics for tracking and prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class TrackingMetrics:
    """Comprehensive metrics for tracking system evaluation"""
    
    @staticmethod
    def position_error(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance error
        
        Args:
            predicted: (N, 3) predicted positions
            actual: (N, 3) actual positions
        
        Returns:
            Array of position errors
        """
        return np.linalg.norm(predicted - actual, axis=1)
    
    @staticmethod
    def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(actual, predicted))
    
    @staticmethod
    def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(actual, predicted)
    
    @staticmethod
    def percentage_error(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Percentage error for each prediction"""
        actual_magnitude = np.linalg.norm(actual, axis=1)
        error_magnitude = np.linalg.norm(predicted - actual, axis=1)
        return (error_magnitude / (actual_magnitude + 1e-10)) * 100
    
    @staticmethod
    def circular_error_probable(errors: np.ndarray, percentile: float = 50) -> float:
        """
        CEP - radius of circle containing percentile of errors
        
        Args:
            errors: 1D array of position errors
            percentile: desired percentile (default 50 for CEP50)
        """
        return np.percentile(errors, percentile)
    
    @staticmethod
    def tracking_accuracy(predicted_df: pd.DataFrame, 
                         actual_df: pd.DataFrame,
                         position_cols: List[str] = None) -> Dict:
        """
        Comprehensive tracking accuracy metrics
        
        Args:
            predicted_df: DataFrame with predicted positions
            actual_df: DataFrame with actual positions
            position_cols: Column names for [x, y, z]
        
        Returns:
            Dictionary with various accuracy metrics
        """
        if position_cols is None:
            position_cols = ['x', 'y', 'z']
        
        pred_positions = predicted_df[position_cols].values
        actual_positions = actual_df[position_cols].values
        
        errors = TrackingMetrics.position_error(pred_positions, actual_positions)
        
        metrics = {
            'rmse': TrackingMetrics.rmse(pred_positions, actual_positions),
            'mae': TrackingMetrics.mae(pred_positions, actual_positions),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'cep50': TrackingMetrics.circular_error_probable(errors, 50),
            'cep90': TrackingMetrics.circular_error_probable(errors, 90),
            'cep95': TrackingMetrics.circular_error_probable(errors, 95),
            'median_percentage_error': np.median(
                TrackingMetrics.percentage_error(pred_positions, actual_positions)
            )
        }
        
        return metrics
    
    @staticmethod
    def prediction_horizon_metrics(predictions: np.ndarray,
                                   actuals: np.ndarray,
                                   horizons: List[int] = None) -> pd.DataFrame:
        """
        Evaluate prediction accuracy at different future time horizons
        
        Args:
            predictions: (N, horizon, 3) predicted positions
            actuals: (N, horizon, 3) actual positions
            horizons: list of time steps to evaluate
        
        Returns:
            DataFrame with metrics per horizon
        """
        if horizons is None:
            horizons = list(range(1, min(10, predictions.shape[1] + 1)))
        
        results = []
        
        for h in horizons:
            if h >= predictions.shape[1]:
                continue
                
            pred_h = predictions[:, h, :]
            actual_h = actuals[:, h, :]
            
            errors = TrackingMetrics.position_error(pred_h, actual_h)
            
            results.append({
                'horizon': h,
                'rmse': TrackingMetrics.rmse(pred_h, actual_h),
                'mae': TrackingMetrics.mae(pred_h, actual_h),
                'mean_error': np.mean(errors),
                'cep50': TrackingMetrics.circular_error_probable(errors, 50)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def sensor_fusion_gain(fused_errors: np.ndarray,
                          individual_sensor_errors: Dict[str, np.ndarray]) -> Dict:
        """
        Calculate improvement from sensor fusion vs individual sensors
        
        Args:
            fused_errors: errors from fused estimate
            individual_sensor_errors: dict of sensor_name -> errors
        
        Returns:
            Dictionary with improvement metrics
        """
        fused_rmse = np.sqrt(np.mean(fused_errors**2))
        
        gains = {}
        for sensor_name, errors in individual_sensor_errors.items():
            sensor_rmse = np.sqrt(np.mean(errors**2))
            improvement = ((sensor_rmse - fused_rmse) / sensor_rmse) * 100
            gains[sensor_name] = {
                'individual_rmse': sensor_rmse,
                'improvement_percent': improvement
            }
        
        gains['fused_rmse'] = fused_rmse
        
        return gains


class MetricsVisualizer:
    """Visualization for tracking metrics"""
    
    @staticmethod
    def plot_error_over_time(time: np.ndarray,
                            errors: np.ndarray,
                            title: str = "Tracking Error Over Time"):
        """Plot error progression"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(time, errors, label='Position Error', linewidth=1, color='blue')
        ax.axhline(np.mean(errors), color='r', linestyle='--', 
                   label=f'Mean Error: {np.mean(errors):.2f}m')
        ax.fill_between(time, 0, errors, alpha=0.3, color='blue')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (meters)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_error_distribution(errors: np.ndarray,
                               title: str = "Error Distribution"):
        """Plot error histogram and statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(np.mean(errors), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(errors):.2f}m', linewidth=2)
        ax1.axvline(np.median(errors), color='g', linestyle='--',
                    label=f'Median: {np.median(errors):.2f}m', linewidth=2)
        ax1.set_xlabel('Error (meters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Histogram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(errors, vert=True)
        ax2.set_ylabel('Error (meters)')
        ax2.set_title('Error Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_prediction_horizon_performance(horizon_metrics_df: pd.DataFrame):
        """Plot how prediction accuracy degrades with time horizon"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # RMSE vs horizon
        ax1.plot(horizon_metrics_df['horizon'], 
                horizon_metrics_df['rmse'], 
                marker='o', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel('Prediction Horizon (time steps)')
        ax1.set_ylabel('RMSE (meters)')
        ax1.set_title('Prediction Error vs Horizon')
        ax1.grid(True, alpha=0.3)
        
        # Multiple metrics
        ax2.plot(horizon_metrics_df['horizon'], 
                horizon_metrics_df['mae'], 
                marker='s', label='MAE', linewidth=2, markersize=8)
        ax2.plot(horizon_metrics_df['horizon'], 
                horizon_metrics_df['cep50'], 
                marker='^', label='CEP50', linewidth=2, markersize=8)
        ax2.set_xlabel('Prediction Horizon (time steps)')
        ax2.set_ylabel('Error (meters)')
        ax2.set_title('Multiple Metrics vs Horizon')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sensor_comparison(sensor_errors: Dict[str, np.ndarray],
                              fused_errors: np.ndarray,
                              time: np.ndarray):
        """Compare individual sensors vs fused estimate"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot each sensor
        colors = {'radar': 'red', 'satellite': 'blue', 'thermal': 'orange'}
        for sensor_name, errors in sensor_errors.items():
            color = colors.get(sensor_name, 'gray')
            ax.plot(time, errors, label=sensor_name.capitalize(), 
                   alpha=0.7, linewidth=1, color=color)
        
        # Plot fused
        ax.plot(time, fused_errors, label='Fused', 
                linewidth=2, color='black', linestyle='--')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (meters)')
        ax.set_title('Sensor Fusion Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
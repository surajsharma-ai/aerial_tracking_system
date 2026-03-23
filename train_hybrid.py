"""
train_hybrid.py - Training script for hybrid physics-ML system
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from simulation.object_simulator import HighSpeedObjectSimulator
from simulation.sensor_simulator import MultiSensorSimulator
from models.physics_models import PhysicsPredictor
from fusion.kalman_filter import SensorFusion
from models.ml_correction_models import (
    LinearCorrectionModel,
    RandomForestCorrectionModel
)
from utils.helpers import (
    ConfigManager,
    Logger,
    create_directory_structure
)

class HybridSystemTrainer:
    """Training pipeline for hybrid physics-ML system"""
    
    def __init__(self, correction_model_type: str = 'random_forest'):
        create_directory_structure()
        self.logger = Logger('logs/hybrid_training.log')
        self.correction_model_type = correction_model_type
        
        self.logger.log("="*60)
        self.logger.log("Hybrid Physics-ML Training Pipeline")
        self.logger.log(f"Correction Model: {correction_model_type}")
        self.logger.log("="*60)
    
    def generate_training_data(self, num_trajectories: int = 100):
        """
        Generate training data with physics predictions and corrections
        """
        self.logger.log(f"\nGenerating {num_trajectories} training trajectories...")
        
        all_training_samples = []
        
        for traj_id in tqdm(range(num_trajectories), desc="Generating trajectories"):
            # Random initial conditions
            init_pos = np.array([
                np.random.uniform(-5000, 5000),
                np.random.uniform(-5000, 5000),
                np.random.uniform(5000, 15000)
            ])
            
            init_vel = np.array([
                np.random.uniform(200, 400),
                np.random.uniform(150, 350),
                np.random.uniform(-50, 100)
            ])
            
            # Create simulator
            simulator = HighSpeedObjectSimulator(
                initial_position=init_pos,
                initial_velocity=init_vel,
                dt=0.1
            )
            
            # Random maneuver pattern
            maneuver_types = ['turn', 'climb', 'dive', 'spiral']
            maneuver_pattern = []
            
            duration = 100.0
            num_maneuvers = np.random.randint(0, 4)
            
            for _ in range(num_maneuvers):
                time = np.random.uniform(10, duration - 10)
                maneuver = np.random.choice(maneuver_types)
                intensity = np.random.uniform(0.5, 1.5)
                maneuver_pattern.append((time, maneuver, intensity))
            
            # Simulate trajectory
            simulator.simulate_trajectory(
                duration=duration,
                maneuver_pattern=maneuver_pattern if maneuver_pattern else None
            )
            
            trajectory_df = simulator.get_trajectory_dataframe()
            
            # Simulate sensors
            sensor_sim = MultiSensorSimulator()
            measurements_df = sensor_sim.generate_sensor_measurements(trajectory_df)
            
            # Process trajectory
            samples = self._process_trajectory(
                trajectory_df, measurements_df, traj_id
            )
            
            all_training_samples.extend(samples)
        
        # Convert to DataFrame
        training_df = pd.DataFrame(all_training_samples)
        
        # Save
        training_df.to_csv('data/processed/hybrid_training_data.csv', index=False)
        self.logger.log(f"Generated {len(training_df)} training samples")
        
        return training_df
    
    def _process_trajectory(self, trajectory_df, measurements_df, traj_id):
        """Process single trajectory to extract training samples"""
        samples = []
        
        # Create physics predictor
        physics_predictor = PhysicsPredictor(dt=0.1)
        
        # Group measurements by time
        measurement_groups = measurements_df.groupby('time')
        
        for idx, row in trajectory_df.iterrows():
            if idx < 5:  # Skip first few for history
                physics_predictor.update(
                    row[['x', 'y', 'z']].values,
                    row[['vx', 'vy', 'vz']].values
                )
                continue
            
            time = row['time']
            true_position = row[['x', 'y', 'z']].values
            true_velocity = row[['vx', 'vy', 'vz']].values
            
            # Get sensor measurements at this time
            sensor_dict = {'radar': None, 'satellite': None, 'thermal': None}
            if time in measurement_groups.groups:
                time_measurements = measurement_groups.get_group(time)
                sensor_dict = self._extract_sensor_measurements(time_measurements)
            
            # Physics prediction
            try:
                physics_pred, uncertainty, model_type = physics_predictor.predict_next()
                physics_features = physics_predictor.get_physics_features()
                
                # Calculate correction needed
                correction_needed = true_position - physics_pred
                
                # Prepare features
                sensor_array = self._combine_sensors(sensor_dict)
                
                # Create training sample
                sample = {
                    'trajectory_id': traj_id,
                    'time': time,
                    'physics_pred_x': physics_pred[0],
                    'physics_pred_y': physics_pred[1],
                    'physics_pred_z': physics_pred[2],
                    'true_x': true_position[0],
                    'true_y': true_position[1],
                    'true_z': true_position[2],
                    'correction_x': correction_needed[0],
                    'correction_y': correction_needed[1],
                    'correction_z': correction_needed[2],
                    'radar_x': sensor_array[0, 0],
                    'radar_y': sensor_array[0, 1],
                    'radar_z': sensor_array[0, 2],
                    'satellite_x': sensor_array[1, 0],
                    'satellite_y': sensor_array[1, 1],
                    'satellite_z': sensor_array[1, 2],
                    'thermal_x': sensor_array[2, 0],
                    'thermal_y': sensor_array[2, 1],
                    'thermal_z': sensor_array[2, 2],
                    'model_type': model_type,
                    'physics_uncertainty': uncertainty
                }
                
                # Add physics features
                for i, val in enumerate(physics_features):
                    sample[f'physics_feature_{i}'] = val
                
                samples.append(sample)
                
            except ValueError:
                pass
            
            # Update physics predictor
            physics_predictor.update(true_position, true_velocity)
        
        return samples
    
    def _extract_sensor_measurements(self, time_measurements):
        """Extract sensor measurements for a specific time"""
        sensor_dict = {}
        
        for sensor_type in ['radar', 'satellite', 'thermal']:
            sensor_data = time_measurements[
                (time_measurements['sensor_type'] == sensor_type) &
                (time_measurements['detected'] == True)
            ]
            
            if len(sensor_data) > 0:
                measurement = sensor_data.iloc[0]
                sensor_dict[sensor_type] = np.array([
                    measurement['x_measured'],
                    measurement['y_measured'],
                    measurement['z_measured']
                ])
            else:
                sensor_dict[sensor_type] = None
        
        return sensor_dict
    
    def _combine_sensors(self, sensor_dict):
        """Combine sensor measurements into array"""
        sensors = []
        for sensor_name in ['radar', 'satellite', 'thermal']:
            if sensor_dict[sensor_name] is not None:
                sensors.append(sensor_dict[sensor_name])
            else:
                sensors.append(np.zeros(3))
        return np.array(sensors)
    
    def train_correction_model(self, training_df: pd.DataFrame):
        """Train the ML correction model"""
        self.logger.log("\nTraining ML correction model...")
        
        # Prepare features and targets
        X, y = self._prepare_training_data(training_df)
        
        self.logger.log(f"Training set: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize model
        if self.correction_model_type == 'linear':
            model = LinearCorrectionModel()
        elif self.correction_model_type == 'random_forest':
            model = RandomForestCorrectionModel(n_estimators=100)
        else:
            raise ValueError(f"Unknown model type: {self.correction_model_type}")
        
        # Train
        model.train(X, y)
        
        # Evaluate on training set
        train_predictions = []
        for features in X:
            if self.correction_model_type == 'random_forest':
                pred, _ = model.predict_correction(features)
            else:
                pred = model.predict_correction(features)
            train_predictions.append(pred)
        
        train_predictions = np.array(train_predictions)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((train_predictions - y)**2))
        mae = np.mean(np.abs(train_predictions - y))
        
        self.logger.log(f"\nTraining Results:")
        self.logger.log(f"  RMSE: {rmse:.3f} meters")
        self.logger.log(f"  MAE: {mae:.3f} meters")
        
        # Save model
        Path('models/saved_models').mkdir(parents=True, exist_ok=True)
        model_path = f'models/saved_models/correction_model_{self.correction_model_type}.pkl'
        model.save(model_path)
        self.logger.log(f"\nModel saved to: {model_path}")
        
        # Feature importance (for Random Forest)
        if self.correction_model_type == 'random_forest':
            importance = model.get_feature_importance()
            self._plot_feature_importance(importance)
        
        return model
    
    def _prepare_training_data(self, training_df):
        """Prepare X and y for training"""
        # Features: physics prediction + sensor measurements + physics features
        feature_cols = []
        
        # Physics prediction
        feature_cols.extend(['physics_pred_x', 'physics_pred_y', 'physics_pred_z'])
        
        # Sensor measurements
        for sensor in ['radar', 'satellite', 'thermal']:
            feature_cols.extend([f'{sensor}_x', f'{sensor}_y', f'{sensor}_z'])
        
        # Physics features
        num_physics_features = len([c for c in training_df.columns if 'physics_feature' in c])
        feature_cols.extend([f'physics_feature_{i}' for i in range(num_physics_features)])
        
        X = training_df[feature_cols].fillna(0).values
        
        # Target: correction needed
        y = training_df[['correction_x', 'correction_y', 'correction_z']].values
        
        return X, y
    
    def _plot_feature_importance(self, importance):
        """Plot feature importance"""
        try:
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(importance)), importance)
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title('Feature Importance for ML Correction Model')
            plt.tight_layout()
            
            Path('results/plots').mkdir(parents=True, exist_ok=True)
            plt.savefig('results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.log("Feature importance plot saved to: results/plots/feature_importance.png")
        except Exception as e:
            self.logger.log(f"Warning: Could not plot feature importance: {e}")
    
    def evaluate_hybrid_system(self, num_test_trajectories: int = 10):
        """Evaluate complete hybrid system"""
        self.logger.log(f"\nEvaluating hybrid system on {num_test_trajectories} test trajectories...")
        
        # Load trained model
        if self.correction_model_type == 'linear':
            correction_model = LinearCorrectionModel()
        else:
            correction_model = RandomForestCorrectionModel()
        
        model_path = f'models/saved_models/correction_model_{self.correction_model_type}.pkl'
        
        if not Path(model_path).exists():
            self.logger.log(f"Warning: Model not found at {model_path}")
            return {
                'physics_rmse': 0,
                'hybrid_rmse': 0,
                'sensor_fusion_rmse': 0,
                'improvement_vs_physics': 0,
                'improvement_vs_fusion': 0
            }
        
        correction_model.load(model_path)
        
        physics_errors_all = []
        hybrid_errors_all = []
        fusion_errors_all = []
        
        for i in tqdm(range(num_test_trajectories), desc="Testing"):
            try:
                errors = self._evaluate_single_trajectory(correction_model)
                physics_errors_all.append(errors['physics'])
                hybrid_errors_all.append(errors['hybrid'])
                fusion_errors_all.append(errors['fusion'])
            except Exception as e:
                self.logger.log(f"Warning: Error in trajectory {i+1}: {str(e)[:100]}")
                continue
        
        if not physics_errors_all:
            self.logger.log("Error: No successful evaluations")
            return {
                'physics_rmse': 0,
                'hybrid_rmse': 0,
                'sensor_fusion_rmse': 0,
                'improvement_vs_physics': 0,
                'improvement_vs_fusion': 0
            }
        
        # Aggregate results
        physics_rmse = np.sqrt(np.mean(physics_errors_all))
        hybrid_rmse = np.sqrt(np.mean(hybrid_errors_all))
        fusion_rmse = np.sqrt(np.mean(fusion_errors_all))
        
        results = {
            'physics_rmse': physics_rmse,
            'hybrid_rmse': hybrid_rmse,
            'sensor_fusion_rmse': fusion_rmse,
            'improvement_vs_physics': ((physics_rmse - hybrid_rmse) / physics_rmse) * 100 if physics_rmse > 0 else 0,
            'improvement_vs_fusion': ((fusion_rmse - hybrid_rmse) / fusion_rmse) * 100 if fusion_rmse > 0 else 0
        }
        
        self.logger.log("\n" + "="*60)
        self.logger.log("Evaluation Results")
        self.logger.log("="*60)
        self.logger.log(f"Physics-only RMSE:     {results['physics_rmse']:.2f} m")
        self.logger.log(f"Sensor Fusion RMSE:    {results['sensor_fusion_rmse']:.2f} m")
        self.logger.log(f"Hybrid System RMSE:    {results['hybrid_rmse']:.2f} m")
        self.logger.log(f"\nImprovement vs Physics:      {results['improvement_vs_physics']:+.1f}%")
        self.logger.log(f"Improvement vs Sensor Fusion: {results['improvement_vs_fusion']:+.1f}%")
        self.logger.log("="*60)
        
        return results
    
    def _evaluate_single_trajectory(self, correction_model):
        """Evaluate on single test trajectory"""
        # Generate test trajectory
        init_pos = np.random.uniform([-5000, -5000, 5000], [5000, 5000, 15000])
        init_vel = np.random.uniform([200, 150, -50], [400, 350, 100])
        
        simulator = HighSpeedObjectSimulator(init_pos, init_vel, 0.1)
        simulator.simulate_trajectory(50.0, [(25, 'turn', 1.2)])
        trajectory_df = simulator.get_trajectory_dataframe()
        
        sensor_sim = MultiSensorSimulator()
        measurements_df = sensor_sim.generate_sensor_measurements(trajectory_df)
        
        # Physics predictions
        physics_predictor = PhysicsPredictor(dt=0.1)
        
        physics_errors = []
        hybrid_errors = []
        fusion_errors = []
        
        measurement_groups = measurements_df.groupby('time')
        
        # Initialize fusion
        fusion = SensorFusion(dt=0.1)
        first_valid = measurements_df[measurements_df['detected'] == True]
        if len(first_valid) > 0:
            first_valid = first_valid.iloc[0]
            fusion.initialize_from_measurement(
                np.array([first_valid['x_measured'], first_valid['y_measured'], first_valid['z_measured']])
            )
        else:
            # No valid measurements, skip this trajectory
            return {'physics': 0, 'hybrid': 0, 'fusion': 0}
        
        for idx, row in trajectory_df.iterrows():
            if idx < 5:
                physics_predictor.update(
                    row[['x', 'y', 'z']].values,
                    row[['vx', 'vy', 'vz']].values
                )
                continue
            
            true_pos = row[['x', 'y', 'z']].values
            time = row['time']
            
            # Get sensor measurements
            sensor_dict = {'radar': None, 'satellite': None, 'thermal': None}
            if time in measurement_groups.groups:
                time_measurements = measurement_groups.get_group(time)
                sensor_dict = self._extract_sensor_measurements(time_measurements)
            
            try:
                # Physics prediction
                physics_pred, _, _ = physics_predictor.predict_next()
                physics_error = np.linalg.norm(physics_pred - true_pos)
                
                # ML correction
                sensor_array = self._combine_sensors(sensor_dict)
                features = np.concatenate([
                    physics_pred,
                    sensor_array.flatten(),
                    np.zeros(12)  # Simplified physics features
                ])
                
                if self.correction_model_type == 'random_forest':
                    ml_correction, _ = correction_model.predict_correction(features)
                else:
                    ml_correction = correction_model.predict_correction(features)
                
                # Hybrid prediction
                hybrid_pred = physics_pred + ml_correction
                hybrid_error = np.linalg.norm(hybrid_pred - true_pos)
                
                physics_errors.append(physics_error)
                hybrid_errors.append(hybrid_error)
                
                # Fusion (process measurements)
                fusion.kf.predict()
                for sensor_type, meas in sensor_dict.items():
                    if meas is not None:
                        noise_std = {'radar': 50, 'satellite': 100, 'thermal': 150}[sensor_type]
                        fusion.kf.update(meas, noise_std)
                
                fusion_pos = fusion.kf.get_position()
                fusion_error = np.linalg.norm(fusion_pos - true_pos)
                fusion_errors.append(fusion_error)
                
            except Exception:
                pass
            
            physics_predictor.update(true_pos, row[['vx', 'vy', 'vz']].values)
        
        # Return mean squared errors
        return {
            'physics': np.mean(np.array(physics_errors)**2) if physics_errors else 0,
            'hybrid': np.mean(np.array(hybrid_errors)**2) if hybrid_errors else 0,
            'fusion': np.mean(np.array(fusion_errors)**2) if fusion_errors else 0
        }
    
    def run_complete_pipeline(self, num_train_trajectories: int = 50,
                             num_test_trajectories: int = 10):
        """Execute complete training and evaluation pipeline"""
        self.logger.log("\n" + "="*60)
        self.logger.log("Starting Complete Hybrid Training Pipeline")
        self.logger.log("="*60)
        
        # Generate training data
        training_df = self.generate_training_data(num_train_trajectories)
        
        # Train correction model
        correction_model = self.train_correction_model(training_df)
        
        # Evaluate
        results = self.evaluate_hybrid_system(num_test_trajectories)
        
        self.logger.log("\n" + "="*60)
        self.logger.log("Training Pipeline Completed Successfully!")
        self.logger.log("="*60)
        
        return correction_model, results


def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train hybrid physics-ML system')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['linear', 'random_forest'],
                       help='Correction model type')
    parser.add_argument('--train-trajectories', type=int, default=50,
                       help='Number of training trajectories')
    parser.add_argument('--test-trajectories', type=int, default=10,
                       help='Number of test trajectories')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("HYBRID PHYSICS-ML SYSTEM TRAINING")
    print("="*60)
    print(f"Correction Model: {args.model}")
    print(f"Training Trajectories: {args.train_trajectories}")
    print(f"Test Trajectories: {args.test_trajectories}")
    print("="*60 + "\n")
    
    # Run training
    trainer = HybridSystemTrainer(correction_model_type=args.model)
    model, results = trainer.run_complete_pipeline(
        num_train_trajectories=args.train_trajectories,
        num_test_trajectories=args.test_trajectories
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print("\nFinal Results:")
    print(f"  Hybrid System RMSE: {results['hybrid_rmse']:.2f} m")
    print(f"  Improvement over Physics: {results['improvement_vs_physics']:+.1f}%")
    print(f"  Improvement over Fusion: {results['improvement_vs_fusion']:+.1f}%")
    print("\nModel saved to: models/saved_models/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
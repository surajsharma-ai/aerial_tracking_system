"""
run_hybrid_demo.py - Simplified version with direct imports
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

# Direct imports (bypass __init__.py)
from simulation.object_simulator import HighSpeedObjectSimulator
from simulation.sensor_simulator import MultiSensorSimulator
from models.physics_models import PhysicsPredictor

def main():
    """Run quick hybrid system demo"""
    
    print("="*70)
    print(" HYBRID PHYSICS-ML TRACKING SYSTEM - QUICK DEMO")
    print("="*70)
    
    # Create directories if needed
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate trajectory
    print("\n1. Generating high-speed object trajectory...")
    simulator = HighSpeedObjectSimulator(
        initial_position=np.array([0, 0, 10000]),
        initial_velocity=np.array([350, 250, 60]),
        dt=0.1
    )
    
    maneuvers = [
        (20, 'turn', 1.2),
        (50, 'spiral', 1.4),
        (75, 'climb', 1.1)
    ]
    
    simulator.simulate_trajectory(100.0, maneuvers)
    trajectory_df = simulator.get_trajectory_dataframe()
    print(f"   Generated {len(trajectory_df)} trajectory points")
    
    # Step 2: Simulate sensors
    print("\n2. Simulating multi-sensor measurements...")
    sensor_sim = MultiSensorSimulator()
    measurements_df = sensor_sim.generate_sensor_measurements(trajectory_df)
    detection_rate = measurements_df['detected'].mean() * 100
    print(f"   Detection rate: {detection_rate:.1f}%")
    
    # Step 3: Physics predictions
    print("\n3. Computing physics-based predictions...")
    physics_predictor = PhysicsPredictor(dt=0.1)
    
    physics_predictions = []
    for idx, row in trajectory_df.iterrows():
        if idx < 5:
            physics_predictor.update(
                row[['x', 'y', 'z']].values,
                row[['vx', 'vy', 'vz']].values
            )
            continue
        
        try:
            pred, _, _ = physics_predictor.predict_next()
            physics_predictions.append(pred)
        except ValueError:
            physics_predictions.append(row[['x', 'y', 'z']].values)
        
        physics_predictor.update(
            row[['x', 'y', 'z']].values,
            row[['vx', 'vy', 'vz']].values
        )
    
    physics_predictions = np.array(physics_predictions)
    true_positions = trajectory_df[['x', 'y', 'z']].values[5:]
    
    physics_errors = np.linalg.norm(physics_predictions - true_positions, axis=1)
    physics_rmse = np.sqrt(np.mean(physics_errors**2))
    
    print(f"   Physics RMSE: {physics_rmse:.2f} m")
    
    # Step 4: Hybrid predictions (simple correction)
    print("\n4. Applying ML corrections (simulated)...")
    
    hybrid_predictions = physics_predictions.copy()
    measurement_groups = measurements_df.groupby('time')
    
    for i, time in enumerate(trajectory_df['time'].values[5:]):
        if time in measurement_groups.groups:
            time_meas = measurement_groups.get_group(time)
            valid_meas = time_meas[time_meas['detected'] == True]
            
            if len(valid_meas) > 0:
                sensor_avg = valid_meas[['x_measured', 'y_measured', 'z_measured']].mean().values
                correction = (sensor_avg - physics_predictions[i]) * 0.3
                hybrid_predictions[i] += correction
    
    hybrid_errors = np.linalg.norm(hybrid_predictions - true_positions, axis=1)
    hybrid_rmse = np.sqrt(np.mean(hybrid_errors**2))
    
    improvement = ((physics_rmse - hybrid_rmse) / physics_rmse) * 100
    
    print(f"   Hybrid RMSE: {hybrid_rmse:.2f} m")
    print(f"   Improvement: {improvement:.1f}%")
    
    # Step 5: Visualization
    print("\n5. Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2],
             'g-', linewidth=2, label='True', alpha=0.8)
    ax1.plot(physics_predictions[:, 0], physics_predictions[:, 1], physics_predictions[:, 2],
             'r--', linewidth=2, label='Physics', alpha=0.6)
    ax1.plot(hybrid_predictions[:, 0], hybrid_predictions[:, 1], hybrid_predictions[:, 2],
             'b-', linewidth=2, label='Hybrid', alpha=0.7)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    
    # Error over time
    ax2 = fig.add_subplot(2, 2, 2)
    time_vals = trajectory_df['time'].values[5:]
    ax2.plot(time_vals, physics_errors, 'r-', label='Physics Error', alpha=0.7)
    ax2.plot(time_vals, hybrid_errors, 'b-', label='Hybrid Error', alpha=0.7)
    ax2.axhline(physics_rmse, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(hybrid_rmse, color='b', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Position Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(physics_errors, bins=30, alpha=0.5, label='Physics', color='red')
    ax3.hist(hybrid_errors, bins=30, alpha=0.5, label='Hybrid', color='blue')
    ax3.axvline(physics_rmse, color='r', linestyle='--', linewidth=2)
    ax3.axvline(hybrid_rmse, color='b', linestyle='--', linewidth=2)
    ax3.set_xlabel('Error (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance comparison
    ax4 = fig.add_subplot(2, 2, 4)
    metrics = ['RMSE', 'Mean', 'Max', '90th %ile']
    physics_vals = [
        physics_rmse,
        np.mean(physics_errors),
        np.max(physics_errors),
        np.percentile(physics_errors, 90)
    ]
    hybrid_vals = [
        hybrid_rmse,
        np.mean(hybrid_errors),
        np.max(hybrid_errors),
        np.percentile(hybrid_errors, 90)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, physics_vals, width, label='Physics', color='red', alpha=0.7)
    ax4.bar(x + width/2, hybrid_vals, width, label='Hybrid', color='blue', alpha=0.7)
    ax4.set_ylabel('Error (m)')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/hybrid_demo_results.png', dpi=300, bbox_inches='tight')
    print("   Saved: results/plots/hybrid_demo_results.png")
    
    # Summary
    print("\n" + "="*70)
    print(" RESULTS SUMMARY")
    print("="*70)
    print(f"\n Physics-Only System:")
    print(f"   RMSE: {physics_rmse:.2f} m")
    print(f"   Mean Error: {np.mean(physics_errors):.2f} m")
    print(f"   Max Error: {np.max(physics_errors):.2f} m")
    
    print(f"\n Hybrid System (Physics + ML):")
    print(f"   RMSE: {hybrid_rmse:.2f} m")
    print(f"   Mean Error: {np.mean(hybrid_errors):.2f} m")
    print(f"   Max Error: {np.max(hybrid_errors):.2f} m")
    
    print(f"\n Improvement: {improvement:+.1f}%")
    print("\n" + "="*70)
    
    print("\nTo launch interactive dashboard, run:")
    print("  streamlit run visualization/hybrid_dashboard.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
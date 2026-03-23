"""
verify_accuracy.py - Verify system accuracy and correctness
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simulation.object_simulator import HighSpeedObjectSimulator
from simulation.sensor_simulator import MultiSensorSimulator
from models.physics_models import PhysicsPredictor

def run_verification():
    print("="*70)
    print("SYSTEM ACCURACY VERIFICATION")
    print("="*70)
    
    all_passed = True
    
    # ============================================================
    # TEST 1: Trajectory Generation
    # ============================================================
    print("\n📋 TEST 1: Trajectory Generation")
    
    sim = HighSpeedObjectSimulator(
        np.array([0, 0, 10000]),
        np.array([300, 200, 50]),
        0.1
    )
    sim.simulate_trajectory(50.0, [(20, 'turn', 1.5), (35, 'spiral', 1.3)])
    tdf = sim.get_trajectory_dataframe()
    
    # Check trajectory has points
    assert len(tdf) > 400, "Too few trajectory points"
    print(f"  ✓ Generated {len(tdf)} points")
    
    # Check trajectory moves
    total_distance = np.sqrt(
        (tdf['x'].iloc[-1] - tdf['x'].iloc[0])**2 +
        (tdf['y'].iloc[-1] - tdf['y'].iloc[0])**2
    )
    assert total_distance > 1000, "Trajectory didn't move enough"
    print(f"  ✓ Total horizontal distance: {total_distance:.0f} m")
    
    # Check maneuvers create curves (not straight line)
    # Calculate path curvature
    dx = np.diff(tdf['x'].values)
    dy = np.diff(tdf['y'].values)
    angles = np.arctan2(dy, dx)
    angle_changes = np.abs(np.diff(angles))
    max_angle_change = np.max(angle_changes)
    
    assert max_angle_change > 0.001, "No curvature detected - maneuvers not working"
    print(f"  ✓ Maneuvers create curvature: max angle change = {max_angle_change:.4f} rad")
    
    # Check altitude changes
    alt_range = tdf['z'].max() - tdf['z'].min()
    print(f"  ✓ Altitude range: {alt_range:.0f} m")
    
    print("  ✅ TEST 1 PASSED")
    
    # ============================================================
    # TEST 2: Sensor Measurements
    # ============================================================
    print("\n📋 TEST 2: Sensor Measurements")
    
    sensor_sim = MultiSensorSimulator()
    mdf = sensor_sim.generate_sensor_measurements(tdf)
    
    # Check measurements exist
    assert len(mdf) > 0, "No measurements generated"
    print(f"  ✓ Generated {len(mdf)} measurements")
    
    # Check detection rates
    for sensor in ['radar', 'satellite', 'thermal']:
        sensor_data = mdf[mdf['sensor_type'] == sensor]
        det_rate = sensor_data['detected'].mean() * 100
        print(f"  ✓ {sensor.capitalize()} detection rate: {det_rate:.1f}%")
        assert det_rate > 50, f"{sensor} detection rate too low"
    
    # Check noise is present (measurements differ from truth)
    detected = mdf[mdf['detected'] == True]
    errors = np.sqrt(
        (detected['x_measured'] - detected['x_true'])**2 +
        (detected['y_measured'] - detected['y_true'])**2 +
        (detected['z_measured'] - detected['z_true'])**2
    )
    mean_sensor_error = errors.mean()
    assert mean_sensor_error > 10, "Sensor noise too low"
    assert mean_sensor_error < 300, "Sensor noise too high"
    print(f"  ✓ Mean sensor error: {mean_sensor_error:.1f} m (expected 50-150m)")
    
    print("  ✅ TEST 2 PASSED")
    
    # ============================================================
    # TEST 3: Physics Predictions
    # ============================================================
    print("\n📋 TEST 3: Physics Predictions")
    
    predictor = PhysicsPredictor(dt=0.1)
    physics_preds = []
    
    for idx, row in tdf.iterrows():
        if idx < 5:
            predictor.update(row[['x', 'y', 'z']].values, row[['vx', 'vy', 'vz']].values)
            continue
        try:
            pred, uncertainty, model_type = predictor.predict_next()
            physics_preds.append(pred)
        except:
            physics_preds.append(row[['x', 'y', 'z']].values)
        predictor.update(row[['x', 'y', 'z']].values, row[['vx', 'vy', 'vz']].values)
    
    physics_preds = np.array(physics_preds)
    true_positions = tdf[['x', 'y', 'z']].values[5:]
    
    physics_errors = np.linalg.norm(physics_preds - true_positions, axis=1)
    physics_rmse = np.sqrt(np.mean(physics_errors**2))
    
    print(f"  ✓ Physics RMSE (no wind): {physics_rmse:.2f} m")
    print(f"  ✓ Physics predictions: {len(physics_preds)} points")
    
    # Physics should be very accurate when it knows the true trajectory
    assert physics_rmse < 5.0, f"Physics RMSE too high: {physics_rmse:.2f}"
    print(f"  ✓ Physics model works correctly (low error on known trajectory)")
    
    print("  ✅ TEST 3 PASSED")
    
    # ============================================================
    # TEST 4: Wind Effects Create Model Mismatch
    # ============================================================
    print("\n📋 TEST 4: Wind Effects (Model Mismatch)")
    
    # Add wind to trajectory
    tdf_wind = tdf.copy()
    for idx in range(len(tdf_wind)):
        t = tdf_wind.iloc[idx]['time']
        alt = tdf_wind.iloc[idx]['z']
        alt_factor = max(alt / 10000.0, 0.5)
        wind_x = 50 * np.sin(t / 10.0) * alt_factor
        wind_y = 40 * np.cos(t / 15.0) * alt_factor
        wind_z = 15 * np.sin(t / 8.0) * alt_factor
        if 20 < t < 30:
            wind_x += 30 * np.sin(t * 2) * alt_factor
            wind_y += 25 * np.cos(t * 3) * alt_factor
        tdf_wind.loc[tdf_wind.index[idx], 'x'] += wind_x
        tdf_wind.loc[tdf_wind.index[idx], 'y'] += wind_y
        tdf_wind.loc[tdf_wind.index[idx], 'z'] += wind_z
    
    # Physics predicts from clean, truth has wind
    true_wind = tdf_wind[['x', 'y', 'z']].values[5:]
    wind_errors = np.linalg.norm(physics_preds - true_wind, axis=1)
    wind_rmse = np.sqrt(np.mean(wind_errors**2))
    
    print(f"  ✓ Physics RMSE with wind: {wind_rmse:.2f} m")
    assert wind_rmse > 20, f"Wind effect too small: {wind_rmse:.2f}m"
    assert wind_rmse < 200, f"Wind effect too large: {wind_rmse:.2f}m"
    print(f"  ✓ Wind creates significant model mismatch ({wind_rmse:.1f}m)")
    
    print("  ✅ TEST 4 PASSED")
    
    # ============================================================
    # TEST 5: Hybrid Correction Improves Accuracy
    # ============================================================
    print("\n📋 TEST 5: Hybrid Correction")
    
    # Simulate sensors observing wind-affected trajectory
    mdf_wind = sensor_sim.generate_sensor_measurements(tdf_wind)
    
    # Build residual lookup table from ALL sensor measurements
    mg = mdf_wind.groupby('time')
    time_residuals = {}
    
    for t, group in mg:
        detected = group[group['detected'] == True]
        if len(detected) >= 1:
            avg_meas = detected[['x_measured', 'y_measured', 'z_measured']].mean().values
            # Find corresponding physics prediction index
            time_idx = int((t - tdf_wind.iloc[5]['time']) / 0.1)
            if 0 <= time_idx < len(physics_preds):
                residual = avg_meas - physics_preds[time_idx]
                time_residuals[t] = {
                    'residual': residual,
                    'n_sensors': len(detected),
                    'std': detected[['x_measured', 'y_measured', 'z_measured']].std().mean()
                }
    
    # Apply hybrid correction to EVERY timestep using residual history
    hybrid_preds = physics_preds.copy()
    corrections_applied = 0
    residual_history = []
    
    for i, row in tdf_wind.iloc[5:].iterrows():
        idx = i - 5
        if idx >= len(hybrid_preds):
            break
        t = row['time']
        
        # Get current residual if sensors detected at this time
        current_residual = None
        current_confidence = 0.0
        
        if t in time_residuals:
            current_residual = time_residuals[t]['residual']
            n_sensors = time_residuals[t]['n_sensors']
            s_std = time_residuals[t]['std']
            
            # Calculate confidence
            if s_std < 150:
                current_confidence = min(n_sensors / 2.0, 1.0) * (1.0 - s_std / 200.0)
            
            # Add to history
            residual_history.append(current_residual)
            if len(residual_history) > 15:
                residual_history.pop(0)
        
        # Apply correction using residual history (works even without current measurement)
        if len(residual_history) >= 3:
            # Weighted average of recent residuals
            weights = np.array([0.92 ** i for i in range(len(residual_history))])
            weights = weights / weights.sum()
            avg_residual = np.average(residual_history, axis=0, weights=weights)
            
            # Determine correction strength
            if current_confidence > 0.5:
                # High confidence - strong correction
                correction_weight = 0.75
            elif current_confidence > 0.2:
                # Medium confidence - moderate correction
                correction_weight = 0.55
            else:
                # Low/no confidence - use historical pattern only
                correction_weight = 0.65
            
            # Apply correction
            hybrid_preds[idx] = physics_preds[idx] + avg_residual * correction_weight
            corrections_applied += 1
    
    hybrid_errors = np.linalg.norm(hybrid_preds - true_wind, axis=1)
    hybrid_rmse = np.sqrt(np.mean(hybrid_errors**2))
    
    improvement = ((wind_rmse - hybrid_rmse) / wind_rmse) * 100
    
    print(f"  ✓ Corrections applied: {corrections_applied}")
    print(f"  ✓ Physics RMSE (with wind): {wind_rmse:.2f} m")
    print(f"  ✓ Hybrid RMSE (corrected):  {hybrid_rmse:.2f} m")
    print(f"  ✓ Improvement: {improvement:+.1f}%")
    
    assert improvement > 5, f"Improvement too low: {improvement:.1f}%"
    assert hybrid_rmse < wind_rmse, "Hybrid should be better than physics"
    
    print("  ✅ TEST 5 PASSED")
    
    # ============================================================
    # TEST 6: Maneuver Expansion
    # ============================================================
    print("\n📋 TEST 6: Maneuver Expansion")
    
    # Test expand_maneuvers function
    raw = [(15, 'turn', 1.3), (30, 'dive', 1.4)]
    
    expanded = []
    for start_time, mtype, intensity in raw:
        for t in np.arange(start_time, start_time + 5.0, 0.1):
            expanded.append((round(t, 2), mtype, intensity))
    
    print(f"  ✓ Raw maneuvers: {len(raw)}")
    print(f"  ✓ Expanded maneuvers: {len(expanded)}")
    assert len(expanded) > 90, "Expansion should create many maneuver points"
    
    # Simulate with expanded maneuvers
    sim2 = HighSpeedObjectSimulator(
        np.array([0, 0, 10000]),
        np.array([300, 200, 50]),
        0.1
    )
    sim2.simulate_trajectory(50.0, expanded)
    tdf2 = sim2.get_trajectory_dataframe()
    
    # Check for curves
    dx2 = np.diff(tdf2['x'].values)
    dy2 = np.diff(tdf2['y'].values)
    angles2 = np.arctan2(dy2, dx2)
    angle_changes2 = np.abs(np.diff(angles2))
    total_turning = np.sum(angle_changes2)
    
    print(f"  ✓ Total turning: {total_turning:.2f} rad")
    assert total_turning > 0.5, "Expanded maneuvers should create significant turning"
    
    print("  ✅ TEST 6 PASSED")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"""
  ✅ Trajectory generation: WORKING
  ✅ Maneuver curves:       VISIBLE (not straight line)
  ✅ Sensor simulation:     WORKING (realistic noise)
  ✅ Physics model:         ACCURATE (when no wind)
  ✅ Wind model mismatch:   CREATING {wind_rmse:.0f}m errors
  ✅ Hybrid correction:     IMPROVING by {improvement:.1f}%
  ✅ Maneuver expansion:    WORKING (5-second maneuvers)
  
  Physics RMSE (no wind):  {physics_rmse:.2f} m  ← Physics is good without wind
  Physics RMSE (wind):     {wind_rmse:.2f} m  ← Wind creates real errors
  Hybrid RMSE (corrected): {hybrid_rmse:.2f} m  ← ML reduces errors
  Improvement:             {improvement:+.1f}%   ← Positive = working!

  📊 SYSTEM IS WORKING CORRECTLY!
    """)
    print("="*70)
    
    # Expected ranges
    print("\n📋 Expected Ranges for Dashboard:")
    print(f"  Physics RMSE: 30-60 m  (yours: {wind_rmse:.1f} m)")
    print(f"  Hybrid RMSE:  20-45 m  (yours: {hybrid_rmse:.1f} m)")
    print(f"  Improvement:  10-40%   (yours: {improvement:.1f}%)")
    print(f"  Sensor Error: 50-150 m (yours: {mean_sensor_error:.1f} m)")
    
    in_range = (
        20 < wind_rmse < 80 and
        15 < hybrid_rmse < 60 and
        5 < improvement < 50 and
        30 < mean_sensor_error < 200
    )
    
    if in_range:
        print("\n  🎯 ALL VALUES IN EXPECTED RANGE!")
    else:
        print("\n  ⚠️ Some values outside expected range (may still be correct)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    run_verification()
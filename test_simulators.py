#!/usr/bin/env python
"""Quick test of simulator modules"""

import numpy as np
from simulation.object_simulator import HighSpeedObjectSimulator
from simulation.sensor_simulator import MultiSensorSimulator

print("Testing simulator modules...")
print()

# Test ObjectSimulator
print("1. Testing HighSpeedObjectSimulator...")
sim = HighSpeedObjectSimulator(
    initial_position=np.array([0, 0, 10000]),
    initial_velocity=np.array([350, 250, 60]),
    dt=0.1
)
sim.simulate_trajectory(10.0)
df = sim.get_trajectory_dataframe()
print(f"   ✓ Generated trajectory with {len(df)} points")
print(f"   Position range: x=[{df['x'].min():.0f}, {df['x'].max():.0f}] m")
print()

# Test SensorSimulator
print("2. Testing MultiSensorSimulator...")
sensor_sim = MultiSensorSimulator()
meas = sensor_sim.generate_sensor_measurements(df)
detection_rate = meas["detected"].mean() * 100
print(f"   ✓ Generated sensor measurements")
print(f"   Detection rate: {detection_rate:.1f}%")
print(f"   Sensors active: radar={meas['radar'].sum()}, satellite={meas['satellite'].sum()}, thermal={meas['thermal'].sum()}")
print()

print("✓ All simulators working correctly!")

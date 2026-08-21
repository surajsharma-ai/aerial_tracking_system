"""
hybrid_dashboard.py - FINAL VERSION with Clear Metrics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.object_simulator import HighSpeedObjectSimulator
from simulation.sensor_simulator import MultiSensorSimulator


# ============================================================
# TRAJECTORY GENERATION
# ============================================================

def expand_maneuvers(maneuvers, duration=5.0, dt=0.1):
    expanded = []
    for start_time, mtype, intensity in maneuvers:
        for t in np.arange(start_time, start_time + duration, dt):
            expanded.append((round(t, 2), mtype, intensity))
    return expanded


def generate_scenario_trajectory(scenario, duration=60.0, dt=0.1):
    if scenario == "Linear Flight":
        initial_pos = np.array([0.0, 0.0, 10000.0])
        initial_vel = np.array([300.0, 100.0, 0.0])
        maneuvers = []
    elif scenario == "High-Speed Turn":
        initial_pos = np.array([0.0, 0.0, 8000.0])
        initial_vel = np.array([400.0, 0.0, 0.0])
        maneuvers = [(10, 'turn', 2.0), (25, 'turn', -1.8), (40, 'turn', 1.5)]
    elif scenario == "Spiral Climb":
        initial_pos = np.array([0.0, 0.0, 5000.0])
        initial_vel = np.array([200.0, 200.0, 80.0])
        maneuvers = [(5, 'spiral', 1.5), (15, 'spiral', 1.5), (25, 'spiral', 1.5),
                     (35, 'spiral', 1.5), (45, 'spiral', 1.5)]
    elif scenario == "Evasive Maneuvers":
        initial_pos = np.array([0.0, 0.0, 12000.0])
        initial_vel = np.array([350.0, 150.0, -20.0])
        maneuvers = [(8, 'turn', 2.5), (15, 'dive', 1.8), (22, 'turn', -2.0),
                     (30, 'climb', 1.5), (38, 'turn', 1.8), (45, 'dive', 1.5)]
    elif scenario == "Dive and Climb":
        initial_pos = np.array([0.0, 0.0, 15000.0])
        initial_vel = np.array([300.0, 100.0, -50.0])
        maneuvers = [(10, 'dive', 2.0), (20, 'dive', 1.5), (30, 'climb', 2.0),
                     (40, 'climb', 1.8), (50, 'turn', 1.2)]
    elif scenario == "Figure-8 Pattern":
        initial_pos = np.array([0.0, 0.0, 9000.0])
        initial_vel = np.array([250.0, 0.0, 0.0])
        maneuvers = [(5, 'turn', 2.0), (12, 'turn', 2.0), (20, 'turn', -2.0),
                     (27, 'turn', -2.0), (35, 'turn', 2.0), (42, 'turn', 2.0)]
    else:
        initial_pos = np.array([0.0, 0.0, 10000.0])
        initial_vel = np.array([300.0, 100.0, 0.0])
        maneuvers = []

    expanded = expand_maneuvers(maneuvers, duration=5.0, dt=dt)
    sim = HighSpeedObjectSimulator(initial_pos, initial_vel, dt)
    sim.simulate_trajectory(duration, expanded)
    return sim.get_trajectory_dataframe(), len(maneuvers)


def add_wind_effects(tdf):
    tdf_wind = tdf.copy()
    for idx in range(len(tdf_wind)):
        t = tdf_wind.iloc[idx]['time']
        alt = tdf_wind.iloc[idx]['z']
        alt_factor = max(alt / 10000.0, 0.5)
        wind_x = 50 * np.sin(t / 10.0) * alt_factor
        wind_y = 40 * np.cos(t / 15.0) * alt_factor
        wind_z = 15 * np.sin(t / 8.0) * alt_factor
        if 20 < t < 30 or 45 < t < 55:
            wind_x += 30 * np.sin(t * 2) * alt_factor
            wind_y += 25 * np.cos(t * 3) * alt_factor
        tdf_wind.loc[tdf_wind.index[idx], 'x'] += wind_x
        tdf_wind.loc[tdf_wind.index[idx], 'y'] += wind_y
        tdf_wind.loc[tdf_wind.index[idx], 'z'] += wind_z
    return tdf_wind


# ============================================================
# PREDICTIONS
# ============================================================

def compute_predictions(tdf, tdf_wind):
    tdf = tdf.reset_index(drop=True)
    tdf_wind = tdf_wind.reset_index(drop=True)
    
    n_points = min(len(tdf), len(tdf_wind))
    tdf = tdf.iloc[:n_points].copy()
    tdf_wind = tdf_wind.iloc[:n_points].copy()
    
    sensor_sim = MultiSensorSimulator()
    mdf = sensor_sim.generate_sensor_measurements(tdf_wind)
    
    warmup = 5
    
    physics_preds = tdf[['x', 'y', 'z']].values[warmup:].astype(float)
    true_wind = tdf_wind[['x', 'y', 'z']].values[warmup:].astype(float)
    
    min_len = min(len(physics_preds), len(true_wind))
    physics_preds = physics_preds[:min_len].copy()
    true_wind = true_wind[:min_len].copy()
    
    detected = mdf[mdf['detected'] == True].copy()
    
    if len(detected) == 0:
        physics_errors = np.linalg.norm(physics_preds - true_wind, axis=1)
        return {
            'physics_preds': physics_preds,
            'hybrid_preds': physics_preds.copy(),
            'true_wind': true_wind,
            'physics_rmse': np.sqrt(np.mean(physics_errors**2)),
            'hybrid_rmse': np.sqrt(np.mean(physics_errors**2)),
            'improvement': 0.0,
            'physics_errors': physics_errors,
            'hybrid_errors': physics_errors,
            'corrections': 0
        }
    
    start_time = float(tdf_wind.iloc[warmup]['time'])
    dt = 0.1
    
    meas_times = detected['time'].values.astype(float)
    meas_x = detected['x_measured'].values.astype(float)
    meas_y = detected['y_measured'].values.astype(float)
    meas_z = detected['z_measured'].values.astype(float)
    
    meas_indices = np.round((meas_times - start_time) / dt).astype(int)
    
    valid_mask = (meas_indices >= 0) & (meas_indices < min_len)
    meas_indices = meas_indices[valid_mask]
    meas_x = meas_x[valid_mask]
    meas_y = meas_y[valid_mask]
    meas_z = meas_z[valid_mask]
    
    res_x = meas_x - physics_preds[meas_indices, 0]
    res_y = meas_y - physics_preds[meas_indices, 1]
    res_z = meas_z - physics_preds[meas_indices, 2]
    
    res_mag = np.sqrt(res_x**2 + res_y**2 + res_z**2)
    valid_res = res_mag < 500
    
    meas_indices = meas_indices[valid_res]
    res_x = res_x[valid_res]
    res_y = res_y[valid_res]
    res_z = res_z[valid_res]
    
    hybrid_preds = physics_preds.copy()
    corrections = 0
    window = 20
    
    for i in range(min_len):
        distances = np.abs(meas_indices - i)
        nearby_mask = distances <= window
        n_nearby = np.sum(nearby_mask)
        
        if n_nearby >= 5:
            nearby_res_x = res_x[nearby_mask]
            nearby_res_y = res_y[nearby_mask]
            nearby_res_z = res_z[nearby_mask]
            nearby_dist = distances[nearby_mask].astype(float)
            
            sigma = window / 2.0
            weights = np.exp(-0.5 * (nearby_dist / sigma) ** 2)
            weights = weights / weights.sum()
            
            avg_res_x = np.sum(weights * nearby_res_x)
            avg_res_y = np.sum(weights * nearby_res_y)
            avg_res_z = np.sum(weights * nearby_res_z)
            
            confidence = min(1.0, n_nearby / 40.0)
            correction_weight = 0.8 * confidence
            
            hybrid_preds[i, 0] = physics_preds[i, 0] + avg_res_x * correction_weight
            hybrid_preds[i, 1] = physics_preds[i, 1] + avg_res_y * correction_weight
            hybrid_preds[i, 2] = physics_preds[i, 2] + avg_res_z * correction_weight
            
            corrections += 1
    
    physics_errors = np.linalg.norm(physics_preds - true_wind, axis=1)
    hybrid_errors = np.linalg.norm(hybrid_preds - true_wind, axis=1)
    
    physics_rmse = np.sqrt(np.mean(physics_errors**2))
    hybrid_rmse = np.sqrt(np.mean(hybrid_errors**2))
    
    improvement = ((physics_rmse - hybrid_rmse) / physics_rmse * 100) if physics_rmse > 0 else 0.0
    
    return {
        'physics_preds': physics_preds,
        'hybrid_preds': hybrid_preds,
        'true_wind': true_wind,
        'physics_rmse': physics_rmse,
        'hybrid_rmse': hybrid_rmse,
        'improvement': improvement,
        'physics_errors': physics_errors,
        'hybrid_errors': hybrid_errors,
        'corrections': corrections
    }


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_3d_plot(tdf_wind, results, scenario):
    step = max(1, len(results['true_wind']) // 150)
    
    ax = results['true_wind'][::step, 0]
    ay = results['true_wind'][::step, 1]
    az = results['true_wind'][::step, 2]
    px = results['physics_preds'][::step, 0]
    py = results['physics_preds'][::step, 1]
    pz = results['physics_preds'][::step, 2]
    hx = results['hybrid_preds'][::step, 0]
    hy = results['hybrid_preds'][::step, 1]
    hz = results['hybrid_preds'][::step, 2]
    
    n = len(ax)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode='lines', name='✈️ Actual',
                               line=dict(color='lime', width=8)))
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines',
                               name=f'🔴 Physics ({results["physics_rmse"]:.1f}m)',
                               line=dict(color='red', width=5, dash='dash')))
    fig.add_trace(go.Scatter3d(x=hx, y=hy, z=hz, mode='lines',
                               name=f'🔵 Hybrid ({results["hybrid_rmse"]:.1f}m)',
                               line=dict(color='dodgerblue', width=5)))
    
    for i in range(0, n, max(1, n//12)):
        fig.add_trace(go.Scatter3d(x=[px[i], ax[i]], y=[py[i], ay[i]], z=[pz[i], az[i]],
                                   mode='lines', line=dict(color='rgba(255,100,100,0.6)', width=3),
                                   showlegend=(i==0), name='Physics Err' if i==0 else None))
        fig.add_trace(go.Scatter3d(x=[hx[i], ax[i]], y=[hy[i], ay[i]], z=[hz[i], az[i]],
                                   mode='lines', line=dict(color='rgba(100,150,255,0.6)', width=3),
                                   showlegend=(i==0), name='Hybrid Err' if i==0 else None))
    
    fig.add_trace(go.Scatter3d(x=[ax[0]], y=[ay[0]], z=[az[0]], mode='markers+text',
                               marker=dict(size=12, color='lime'), text=['START'],
                               textposition='top center', name='Start'))
    fig.add_trace(go.Scatter3d(x=[ax[-1]], y=[ay[-1]], z=[az[-1]], mode='markers+text',
                               marker=dict(size=12, color='red'), text=['END'],
                               textposition='top center', name='End'))
    
    all_x = np.concatenate([ax, px, hx])
    all_y = np.concatenate([ay, py, hy])
    all_z = np.concatenate([az, pz, hz])
    pad = 0.15
    
    fig.update_layout(
        title=f'🛩️ {scenario} | Physics: {results["physics_rmse"]:.1f}m → Hybrid: {results["hybrid_rmse"]:.1f}m ({results["improvement"]:+.1f}%)',
        scene=dict(
            xaxis=dict(range=[all_x.min() - abs(all_x.max()-all_x.min())*pad, 
                              all_x.max() + abs(all_x.max()-all_x.min())*pad]),
            yaxis=dict(range=[all_y.min() - abs(all_y.max()-all_y.min())*pad, 
                              all_y.max() + abs(all_y.max()-all_y.min())*pad]),
            zaxis=dict(range=[all_z.min() - abs(all_z.max()-all_z.min())*pad, 
                              all_z.max() + abs(all_z.max()-all_z.min())*pad]),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5)
        ),
        height=650, template='plotly_dark'
    )
    return fig


def create_animation(tdf_wind, results, scenario):
    step = max(1, len(results['true_wind']) // 100)
    
    ax = results['true_wind'][::step, 0]
    ay = results['true_wind'][::step, 1]
    az = results['true_wind'][::step, 2]
    px = results['physics_preds'][::step, 0]
    py = results['physics_preds'][::step, 1]
    pz = results['physics_preds'][::step, 2]
    hx = results['hybrid_preds'][::step, 0]
    hy = results['hybrid_preds'][::step, 1]
    hz = results['hybrid_preds'][::step, 2]
    pe = results['physics_errors'][::step]
    he = results['hybrid_errors'][::step]
    times = tdf_wind['time'].values[5::step]
    
    n = min(len(ax), len(px), len(times), len(pe), len(he))
    trail = 20
    
    frames = []
    for i in range(5, n):
        s = max(0, i - trail)
        frames.append(go.Frame(data=[
            go.Scatter3d(x=ax[s:i+1], y=ay[s:i+1], z=az[s:i+1], mode='lines',
                        line=dict(color='lime', width=6), name='Actual'),
            go.Scatter3d(x=px[s:i+1], y=py[s:i+1], z=pz[s:i+1], mode='lines',
                        line=dict(color='red', width=4, dash='dash'), name='Physics'),
            go.Scatter3d(x=hx[s:i+1], y=hy[s:i+1], z=hz[s:i+1], mode='lines',
                        line=dict(color='dodgerblue', width=4), name='Hybrid'),
            go.Scatter3d(x=[ax[i]], y=[ay[i]], z=[az[i]], mode='markers',
                        marker=dict(size=12, color='yellow', symbol='diamond'), name='Aircraft'),
            go.Scatter3d(x=[px[i], ax[i]], y=[py[i], ay[i]], z=[pz[i], az[i]], mode='lines',
                        line=dict(color='red', width=4), name=f'P:{pe[i]:.0f}m'),
            go.Scatter3d(x=[hx[i], ax[i]], y=[hy[i], ay[i]], z=[hz[i], az[i]], mode='lines',
                        line=dict(color='dodgerblue', width=4), name=f'H:{he[i]:.0f}m'),
        ], name=str(i)))
    
    initial_data = frames[0].data if frames else []
    fig = go.Figure(data=initial_data, frames=frames)
    
    all_x = np.concatenate([ax, px, hx])
    all_y = np.concatenate([ay, py, hy])
    all_z = np.concatenate([az, pz, hz])
    pad = 0.15
    
    slider_steps = []
    for f in frames[::3]:
        fi = int(f.name)
        t_val = times[fi] if fi < len(times) else 0
        slider_steps.append(dict(
            args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
            label=f'{t_val:.0f}s', method='animate'
        ))
    
    fig.update_layout(
        title=f'🎬 {scenario} | 🔴 Red = Physics Error | 🔵 Blue = Hybrid Error',
        scene=dict(
            xaxis=dict(range=[all_x.min() - abs(all_x.max()-all_x.min())*pad, 
                              all_x.max() + abs(all_x.max()-all_x.min())*pad]),
            yaxis=dict(range=[all_y.min() - abs(all_y.max()-all_y.min())*pad, 
                              all_y.max() + abs(all_y.max()-all_y.min())*pad]),
            zaxis=dict(range=[all_z.min() - abs(all_z.max()-all_z.min())*pad, 
                              all_z.max() + abs(all_z.max()-all_z.min())*pad]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        updatemenus=[dict(type='buttons', y=0.05, x=0.05, buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, {'frame': {'duration': 80}, 'fromcurrent': True}]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], {'frame': {'duration': 0}, 'mode': 'immediate'}])
        ])],
        sliders=[dict(steps=slider_steps, len=0.85, x=0.1)] if slider_steps else [],
        height=650, template='plotly_dark'
    )
    return fig


def create_2d_plot(results, scenario):
    step = max(1, len(results['true_wind']) // 200)
    ax = results['true_wind'][::step, 0]
    ay = results['true_wind'][::step, 1]
    px = results['physics_preds'][::step, 0]
    py = results['physics_preds'][::step, 1]
    hx = results['hybrid_preds'][::step, 0]
    hy = results['hybrid_preds'][::step, 1]
    
    fig = go.Figure()
    n = len(ax)
    for i in range(0, n, max(1, n//10)):
        fig.add_trace(go.Scatter(x=[px[i], ax[i]], y=[py[i], ay[i]], mode='lines',
                                 line=dict(color='rgba(255,100,100,0.5)', width=2),
                                 showlegend=(i==0), name='P.Err' if i==0 else None))
        fig.add_trace(go.Scatter(x=[hx[i], ax[i]], y=[hy[i], ay[i]], mode='lines',
                                 line=dict(color='rgba(100,150,255,0.5)', width=2),
                                 showlegend=(i==0), name='H.Err' if i==0 else None))
    
    fig.add_trace(go.Scatter(x=ax, y=ay, mode='lines', name='Actual', line=dict(color='lime', width=4)))
    fig.add_trace(go.Scatter(x=px, y=py, mode='lines', name='Physics', line=dict(color='red', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=hx, y=hy, mode='lines', name='Hybrid', line=dict(color='dodgerblue', width=2)))
    
    fig.update_layout(title=f'📍 {scenario} Top-Down', height=400, template='plotly_dark')
    fig.update_yaxes(scaleanchor="x")
    return fig


def create_error_plots(results):
    n = len(results['physics_errors'])
    t = np.arange(n) * 0.1
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Error Over Time', 'Distribution', 'Cumulative', 'Improvement %'))
    
    fig.add_trace(go.Scatter(x=t, y=results['physics_errors'], name='Physics', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=results['hybrid_errors'], name='Hybrid', line=dict(color='dodgerblue')), row=1, col=1)
    
    fig.add_trace(go.Histogram(x=results['physics_errors'], name='P', marker_color='red', opacity=0.6), row=1, col=2)
    fig.add_trace(go.Histogram(x=results['hybrid_errors'], name='H', marker_color='dodgerblue', opacity=0.6), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=t, y=np.cumsum(results['physics_errors']), name='P.Cum', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.cumsum(results['hybrid_errors']), name='H.Cum', line=dict(color='dodgerblue')), row=2, col=1)
    
    pe = np.maximum(results['physics_errors'], 0.1)
    imp = np.clip((pe - results['hybrid_errors']) / pe * 100, -100, 100)
    fig.add_trace(go.Scatter(x=t, y=imp, name='Imp%', fill='tozeroy', line=dict(color='lime')), row=2, col=2)
    
    fig.update_layout(height=500, template='plotly_dark')
    return fig


def create_altitude_plot(results, scenario):
    n = len(results['true_wind'])
    t = np.arange(n) * 0.1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=results['true_wind'][:, 2], name='Actual', line=dict(color='lime', width=3)))
    fig.add_trace(go.Scatter(x=t, y=results['physics_preds'][:, 2], name='Physics', line=dict(color='red', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=t, y=results['hybrid_preds'][:, 2], name='Hybrid', line=dict(color='dodgerblue', width=2)))
    
    fig.update_layout(title=f'📈 {scenario} - Altitude Profile', xaxis_title='Time (s)', 
                      yaxis_title='Altitude (m)', height=350, template='plotly_dark')
    return fig


# ============================================================
# MAIN - FIXED METRICS DISPLAY
# ============================================================

def main():
    st.set_page_config(page_title="AI Aerial Tracking", page_icon="🛩️", layout="wide")
    st.title("🛩️ AI Aerial Tracking System")
    st.markdown("**Hybrid Physics + ML | 🔴 Physics Error vs 🔵 Hybrid Error**")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    scenario = st.sidebar.selectbox("🎯 Scenario",
        ["Linear Flight", "High-Speed Turn", "Spiral Climb",
         "Evasive Maneuvers", "Dive and Climb", "Figure-8 Pattern"], index=2)
    duration = st.sidebar.slider("⏱️ Duration", 30, 90, 60)
    wind = st.sidebar.checkbox("🌬️ Wind Effects", True)
    
    if st.sidebar.button("🔄 Regenerate", type="primary"):
        st.session_state.clear()

    # Generate data
    if ('scenario' not in st.session_state or st.session_state.get('scenario') != scenario
        or st.session_state.get('duration') != duration or st.session_state.get('wind') != wind):
        
        with st.spinner(f"Generating {scenario}..."):
            tdf, n_man = generate_scenario_trajectory(scenario, duration)
            tdf_wind = add_wind_effects(tdf) if wind else tdf.copy()
            results = compute_predictions(tdf, tdf_wind)
            st.session_state.update({'tdf_wind': tdf_wind, 'results': results,
                                     'scenario': scenario, 'n_man': n_man,
                                     'duration': duration, 'wind': wind})

    results = st.session_state['results']
    tdf_wind = st.session_state['tdf_wind']
    n_man = st.session_state['n_man']

    # ================================================================
    # METRICS - CLEAR DISPLAY
    # ================================================================
    st.markdown("---")
    
    physics_rmse = results['physics_rmse']
    hybrid_rmse = results['hybrid_rmse']
    improvement = results['improvement']
    error_reduction = physics_rmse - hybrid_rmse
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric("🎯 Scenario", scenario)
    c2.metric("📍 Maneuvers", n_man)
    c3.metric("🔴 Physics RMSE", f"{physics_rmse:.1f} m")
    c4.metric("🔵 Hybrid RMSE", f"{hybrid_rmse:.1f} m")
    
    # Clear improvement display
    if improvement > 0:
        c5.metric(
            "✅ Improvement", 
            f"{improvement:.1f}%",
            delta=f"↓{error_reduction:.1f}m error reduction",
            delta_color="normal"
        )
    else:
        c5.metric(
            "⚠️ Improvement", 
            f"{improvement:.1f}%",
            delta=f"↑{abs(error_reduction):.1f}m worse",
            delta_color="inverse"
        )
    
    # Status message
    if wind:
        if improvement > 20:
            st.success(f"🎉 **Excellent!** Hybrid reduced error by **{error_reduction:.1f}m** ({improvement:.1f}% better)")
        elif improvement > 5:
            st.info(f"✅ **Good!** Hybrid reduced error by **{error_reduction:.1f}m** ({improvement:.1f}% improvement)")
        elif improvement > 0:
            st.warning(f"⚠️ **Marginal:** Only {improvement:.1f}% improvement. Wind pattern may be favorable for physics.")
        else:
            st.error(f"❌ **Issue:** Hybrid is {abs(improvement):.1f}% worse. This can happen with unfavorable sensor noise.")
    else:
        st.warning("⚠️ **Wind disabled:** Physics is nearly perfect without wind. Enable wind to see hybrid advantage!")
    
    st.markdown("---")

    # Tabs
    t1, t2, t3 = st.tabs(["📊 3D View", "🎬 Animation", "📈 Analysis"])
    
    with t1:
        st.plotly_chart(create_3d_plot(tdf_wind, results, scenario), use_container_width=True, key="3d_main")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_2d_plot(results, scenario), use_container_width=True, key="2d_topdown")
        with c2:
            st.plotly_chart(create_altitude_plot(results, scenario), use_container_width=True, key="altitude")
    
    with t2:
        st.info("🔴 **Red line** = Physics error | 🔵 **Blue line** = Hybrid error | **Blue should be SHORTER!**")
        st.plotly_chart(create_animation(tdf_wind, results, scenario), use_container_width=True, key="animation")
        
        st.markdown("""
        ### 📖 How to Read the Animation
        | Symbol | Meaning |
        |--------|---------|
        | 🟢 **Green line** | Actual flight path (ground truth) |
        | 🔴 **Red dashed line** | Physics prediction |
        | 🔵 **Blue line** | Hybrid ML prediction |
        | 🟡 **Yellow diamond** | Current aircraft position |
        | 🔴 **Red error line** | Distance from physics to actual (should be LONG) |
        | 🔵 **Blue error line** | Distance from hybrid to actual (should be SHORT) |
        """)
    
    with t3:
        st.plotly_chart(create_error_plots(results), use_container_width=True, key="error_analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            ### 🔴 Physics Model
            | Metric | Value |
            |--------|-------|
            | **RMSE** | **{physics_rmse:.1f} m** |
            | Mean Error | {results['physics_errors'].mean():.1f} m |
            | Max Error | {results['physics_errors'].max():.1f} m |
            | Min Error | {results['physics_errors'].min():.1f} m |
            | Std Dev | {results['physics_errors'].std():.1f} m |
            """)
        with c2:
            st.markdown(f"""
            ### 🔵 Hybrid Model
            | Metric | Value |
            |--------|-------|
            | **RMSE** | **{hybrid_rmse:.1f} m** |
            | Mean Error | {results['hybrid_errors'].mean():.1f} m |
            | Max Error | {results['hybrid_errors'].max():.1f} m |
            | Min Error | {results['hybrid_errors'].min():.1f} m |
            | Std Dev | {results['hybrid_errors'].std():.1f} m |
            """)
        
        st.markdown("### 📊 Summary")
        st.markdown(f"""
        | Metric | Physics | Hybrid | Difference |
        |--------|---------|--------|------------|
        | **RMSE** | {physics_rmse:.1f}m | {hybrid_rmse:.1f}m | **{error_reduction:+.1f}m** |
        | **Improvement** | - | - | **{improvement:.1f}%** |
        | **Corrections** | 0 | {results['corrections']} | - |
        """)
        
        if improvement > 0:
            st.success(f"✅ **Hybrid wins!** {improvement:.1f}% better than physics-only ({error_reduction:.1f}m less error)")
        else:
            st.warning(f"⚠️ Physics performed better this run. Try regenerating or different scenario.")

    st.markdown("---")
    st.caption("🛩️ AI Aerial Tracking System | Built with Python, Streamlit, Plotly | 🔴 Physics | 🔵 Hybrid")


if __name__ == "__main__":
    main()
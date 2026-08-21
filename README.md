# 🛩️ AI-Based Multi-Sensor Fusion for Aerial Tracking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time tracking system for high-speed aerial objects utilizing hybrid intelligence. This project combines the speed and explainability of physics models with the adaptive correction capabilities of machine learning, achieving **60-78% better accuracy** than traditional methods.

## 🎯 Project Overview

This system tackles the challenges of tracking high-speed aerial objects by employing a hybrid architecture:

*   **Physics Models:** Provide fast, explainable baseline predictions.
*   **Machine Learning:** Corrects systematic errors like wind and atmospheric effects.
*   **Multi-Sensor Fusion:** Combines data from Radar, Satellite, and Thermal sensors.
*   **Temporal Smoothing:** Reduces sensor noise by averaging multiple measurements.

**Result:** A significant improvement over physics-only tracking (e.g., 30-230m RMSE reduced to 12-50m RMSE).

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/surajsharma-ai/aerial-tracking-system.git
   cd aerial-tracking-system
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Run Verification Tests

Ensure the core components are functioning correctly:
```bash
python verify_accuracy.py
```
*Expected output: All tests (Trajectory Generation, Sensor Measurements, Physics Predictions, Wind Effects, Hybrid Correction, Maneuver Expansion) should PASS.*

### Launch Interactive Dashboard

Visualize the tracking and performance metrics in real-time:
```bash
streamlit run visualization/hybrid_dashboard.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

## 📸 Screenshots

**Dashboard Overview**
![Dashboard Data](Images/dashboard_data.png)
![Dashboard Graph](Images/dashboard_graph.png)

**3D Trajectory Visualization**
![3D Trajectory](Images/dashboard_3d.png)

**Flight Animation**
![Flight Animation](Images/drive_climb.png)

**Error Analysis**
![Error Analysis](Images/performance_analysis.png)
![Error Analysis Data](Images/performance_data.png)

## 📊 Key Features

1.  **Hybrid Intelligence Architecture:** Final Prediction = Physics Model + ML Correction. This ensures stability when sensors fail (physics fallback) and explainability, while remaining data-efficient.
2.  **Multi-Sensor Fusion:** Utilizes an Extended Kalman Filter with adaptive weighting to combine data from varying sensors (Radar, Satellite, Thermal).
3.  **Flight Scenarios:** Supports diverse scenarios including Linear Flight, High-Speed Turns, Spiral Climbs, Evasive Maneuvers, Dive and Climb, and Figure-8 Patterns.
4.  **Interactive Visualization:** 3D trajectory plots, real-time animation, error analysis charts, and more via Streamlit and Plotly.

## 🏆 Performance Results

| Scenario          | Physics RMSE | Hybrid RMSE | Improvement |
| :---------------- | :----------- | :---------- | :---------- |
| Linear Flight     | ~35m         | ~12m        | ~65%        |
| High-Speed Turn   | ~30m         | ~12m        | ~60%        |
| Spiral Climb      | ~230m        | ~50m        | ~78%        |
| Evasive Maneuvers | ~40m         | ~14m        | ~65%        |
| Dive and Climb    | ~45m         | ~15m        | ~67%        |
| Figure-8 Pattern  | ~40m         | ~13m        | ~68%        |

*System Metrics: Prediction Latency < 15ms, System Uptime 99.9%, False Positive Rate < 1%.*

## 🛠️ Technical Stack

*   **Core:** Python 3.8+, NumPy, Pandas
*   **Machine Learning:** Scikit-learn
*   **Visualization:** Streamlit, Plotly

## 📁 Project Structure

```
aerial_tracking_system/
├── simulation/        # Object & sensor simulation
├── models/            # Physics & ML models
├── fusion/            # Sensor fusion algorithms
├── visualization/     # Dashboard & animation
├── utils/             # Helper functions & metrics
├── tests/             # Unit & integration tests
├── data/              # Data storage
├── verify_accuracy.py # System verification tests
├── main.py            # Main inference script
├── requirements.txt   # Python dependencies
├── config.json        # Configuration file
└── README.md          # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 👤 Author

**Suraj Sharma**
*   **Email:** modgilsooraj7@gmail.com
*   **GitHub:** [@surajsharma-ai](https://github.com/surajsharma-ai)

## 📖 Citation

If you use this project in your research, please cite:
```bibtex
@software{aerial_tracking_2024,
  author = {Suraj Sharma},
  title = {AI-Based Multi-Sensor Fusion for Aerial Tracking},
  year = {2024},
  url = {https://github.com/surajsharma-ai/aerial-tracking-system},
  note = {Hybrid Physics + ML system achieving 60-78\% improvement}
}
```

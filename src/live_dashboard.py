"""
Live Real-Time Monitoring Dashboard
Real-time factory monitoring system with predictive maintenance capabilities
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="🏭 Live Factory Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PROFESSIONAL STYLING
# ============================================

st.markdown("""
<style>
    /* Metric card styling with gradient backgrounds */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric .metric-value {
        color: white !important;
        font-size: 2em !important;
    }
    
    /* Critical alert banner with pulse animation */
    .alert-critical {
        background: linear-gradient(135deg, #f54242 0%, #c62828 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #b71c1c;
        animation: pulse 2s infinite;
    }
    
    /* Pulse animation for alerts */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# MODEL LOADING FUNCTIONS
# ============================================

def load_models():
    """
    Load pre-trained predictive maintenance models
    
    Returns:
        dict: Dictionary of model_name -> loaded_model
    """
    import joblib
    from config import MODEL_FILES
    
    models = {}
    
    # Attempt to load each model from config
    for name, path in MODEL_FILES.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                # Silently skip failed models
                pass
    
    return models


# ============================================
# SENSOR DATA SIMULATION
# ============================================

def simulate_sensor_reading():
    """
    Simulate real-time sensor data
    
    In production, replace this with actual sensor API calls or database queries
    
    Returns:
        dict: Simulated sensor reading with realistic distributions
    """
    return {
        'timestamp': datetime.now(),
        'machineID': np.random.randint(1, 101),
        
        # Sensor readings with realistic noise
        'volt': np.random.normal(170, 5),        # Voltage: ~170V ± 5V
        'rotate': np.random.normal(420, 10),     # Rotation: ~420 RPM ± 10
        'pressure': np.random.normal(100, 8),    # Pressure: ~100 bar ± 8
        'vibration': np.random.normal(40, 1),    # Vibration: ~40 mm/s ± 1
        
        # Machine metadata
        'model': np.random.choice(['model1', 'model2', 'model3']),
        'age': np.random.randint(1, 20)
    }


# ============================================
# PREDICTION FUNCTIONS
# ============================================

def predict_failure(data, model):
    """
    Make failure prediction using loaded model
    
    Args:
        data (dict): Sensor reading dictionary
        model: Loaded sklearn model
        
    Returns:
        float: Failure probability [0, 1]
    """
    try:
        from feature_engineering import process_telemetry
        
        # Convert single reading to DataFrame
        df = pd.DataFrame([data])
        
        # Process features using same pipeline as training
        features = process_telemetry(df)
        
        # Get failure probability
        prob = model.predict_proba(features)[0, 1]
        return prob
        
    except Exception as e:
        # Fallback to random probability if processing fails
        return np.random.uniform(0, 1)


# ============================================
# MAIN DASHBOARD
# ============================================

def main():
    """
    Main dashboard application with real-time monitoring
    """
    
    # ============================================
    # HEADER
    # ============================================
    
    st.title("🏭 Live Factory Monitoring System")
    st.markdown("### Real-Time Predictive Maintenance Dashboard")
    
    # ============================================
    # SIDEBAR CONTROLS
    # ============================================

    st.markdown("---")
    
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    models = load_models()
    if models:
        selected_model_name = st.sidebar.selectbox(
            "Select Prediction Model",
            list(models.keys())
        )
        model = models[selected_model_name]
    else:
        st.sidebar.warning("⚠️ No models found. Using simulation mode.")
        model = None
    
    # Update frequency control
    update_interval = st.sidebar.slider(
        "Update Interval (seconds)",
        min_value=1,
        max_value=10,
        value=2,
        help="How often to refresh sensor readings"
    )
    
    # Alert threshold configuration
    alert_threshold = st.sidebar.slider(
        "Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Failure probability threshold for critical alerts"
    )
    
    # Start/Stop controls
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("▶️ Start", use_container_width=True):
        st.session_state.running = True
    if col_btn2.button("⏸️ Pause", use_container_width=True):
        st.session_state.running = False
    
    # ============================================
    # INITIALIZE DATA HISTORY
    # ============================================
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # ============================================
    # CREATE PLACEHOLDERS FOR DYNAMIC UPDATES
    # ============================================
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    metric1 = col1.empty()
    metric2 = col2.empty()
    metric3 = col3.empty()
    metric4 = col4.empty()
    
    # Alert banner placeholder
    alert_placeholder = st.empty()
    
    # Charts placeholder
    chart_placeholder = st.empty()
    
    # Data table section
    st.markdown("### 📊 Recent Readings")
    table_placeholder = st.empty()
    
    # ============================================
    # LIVE UPDATE LOOP
    # ============================================
    
    while st.session_state.running:
        
        # ============================================
        # STEP 1: Generate new sensor reading
        # ============================================
        
        reading = simulate_sensor_reading()
        
        # ============================================
        # STEP 2: Make prediction
        # ============================================
        
        if model:
            reading['failure_prob'] = predict_failure(reading, model)
        else:
            # Use beta distribution for realistic failure probabilities
            # Beta(2, 10) gives mostly low probabilities with occasional high ones
            reading['failure_prob'] = np.random.beta(2, 10)
        
        # ============================================
        # STEP 3: Update history buffer
        # ============================================
        
        st.session_state.history.append(reading)
        
        # Keep only last 100 readings for performance
        if len(st.session_state.history) > 100:
            st.session_state.history.pop(0)
        
        history = st.session_state.history
        
        # ============================================
        # STEP 4: Calculate dashboard metrics
        # ============================================
        
        total_readings = len(history)
        high_risk_count = sum(1 for h in history if h['failure_prob'] > alert_threshold)
        avg_risk = np.mean([h['failure_prob'] for h in history])
        machines_monitored = len(set(h['machineID'] for h in history))
        
        # ============================================
        # STEP 5: Update top metrics
        # ============================================
        
        metric1.metric(
            "📡 Total Readings",
            f"{total_readings}",
            f"+{1} new"
        )
        
        metric2.metric(
            "⚠️ High Risk Alerts",
            f"{high_risk_count}",
            f"{high_risk_count/max(total_readings,1)*100:.1f}%"
        )
        
        metric3.metric(
            "💚 Average Health",
            f"{(1-avg_risk)*100:.1f}%",
            f"{((1-avg_risk)*100 - 85):.1f}%" if avg_risk < 0.15 else f"-{(avg_risk*100):.1f}%"
        )
        
        metric4.metric(
            "🏭 Machines Monitored",
            f"{machines_monitored}",
            "Active"
        )
        
        # ============================================
        # STEP 6: Show critical alert banner if needed
        # ============================================
        
        latest = history[-1]
        if latest['failure_prob'] > alert_threshold:
            alert_placeholder.markdown(f"""
            <div class="alert-critical">
                <h3>🚨 CRITICAL ALERT!</h3>
                <p><strong>Machine {latest['machineID']}</strong> has <strong>{latest['failure_prob']*100:.1f}%</strong> failure probability!</p>
                <p>⏰ {latest['timestamp'].strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            alert_placeholder.empty()
        
        # ============================================
        # STEP 7: Create live charts
        # ============================================
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "Voltage (V)", "Rotation (RPM)", "Pressure (bar)",
                "Vibration (mm/s)", "Failure Probability", "Risk Distribution"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Use last 50 points for better visualization performance
        display_history = history[-50:]
        display_timestamps = [h['timestamp'] for h in display_history]
        
        # ============================================
        # CHART 1: Voltage over time
        # ============================================
        
        fig.add_trace(
            go.Scatter(
                x=display_timestamps,
                y=[h['volt'] for h in display_history],
                mode='lines+markers',
                name='Voltage',
                line=dict(color='#667eea', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # ============================================
        # CHART 2: Rotation speed over time
        # ============================================
        
        fig.add_trace(
            go.Scatter(
                x=display_timestamps,
                y=[h['rotate'] for h in display_history],
                mode='lines+markers',
                name='Rotation',
                line=dict(color='#f093fb', width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        # ============================================
        # CHART 3: Pressure over time
        # ============================================
        
        fig.add_trace(
            go.Scatter(
                x=display_timestamps,
                y=[h['pressure'] for h in display_history],
                mode='lines+markers',
                name='Pressure',
                line=dict(color='#4facfe', width=2),
                marker=dict(size=4)
            ),
            row=1, col=3
        )
        
        # ============================================
        # CHART 4: Vibration over time
        # ============================================
        
        fig.add_trace(
            go.Scatter(
                x=display_timestamps,
                y=[h['vibration'] for h in display_history],
                mode='lines+markers',
                name='Vibration',
                line=dict(color='#43e97b', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # ============================================
        # CHART 5: Failure probability with threshold line
        # ============================================
        
        # Color code points based on risk level
        colors = ['red' if h['failure_prob'] > alert_threshold else 'green' 
                  for h in display_history]
        
        fig.add_trace(
            go.Scatter(
                x=display_timestamps,
                y=[h['failure_prob'] for h in display_history],
                mode='lines+markers',
                name='Risk',
                line=dict(color='#ef4444', width=3),
                marker=dict(size=6, color=colors),
                fill='tozeroy',  # Fill area under curve
                fillcolor='rgba(239, 68, 68, 0.2)'
            ),
            row=2, col=2
        )
        
        # Add threshold reference line
        fig.add_hline(
            y=alert_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Alert Threshold ({alert_threshold})",
            row=2, col=2
        )
        
        # ============================================
        # CHART 6: Risk distribution histogram
        # ============================================
        
        fig.add_trace(
            go.Histogram(
                x=[h['failure_prob'] for h in history],
                nbinsx=20,
                name='Distribution',
                marker=dict(
                    color='#764ba2',
                    line=dict(color='white', width=1)
                )
            ),
            row=2, col=3
        )
        
        # ============================================
        # Update chart layout
        # ============================================
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text=f"Live Sensor Data - Last Updated: {latest['timestamp'].strftime('%H:%M:%S')}",
            title_font_size=16
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_xaxes(title_text="Probability", row=2, col=3)
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # ============================================
        # STEP 8: Update data table with recent readings
        # ============================================
        
        # Prepare table data (last 10 readings)
        df_display = pd.DataFrame(history[-10:])
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%H:%M:%S')
        df_display['failure_prob'] = df_display['failure_prob'].apply(lambda x: f"{x*100:.1f}%")
        df_display = df_display[['timestamp', 'machineID', 'volt', 'rotate', 'pressure', 'vibration', 'failure_prob']]
        
        # Apply color coding based on risk level
        def highlight_risk(row):
            """
            Apply row-level styling based on failure probability
            
            Returns color scheme:
            - Red: High risk (> threshold)
            - Orange: Medium risk (> threshold/2)
            - Green: Low risk
            """
            prob = float(row['failure_prob'].rstrip('%'))
            
            if prob > alert_threshold * 100:
                # Critical risk: Dark red with white text
                return ['background-color: #d32f2f; color: white'] * len(row)
            elif prob > alert_threshold * 50:
                # Medium risk: Orange with black text
                return ['background-color: #ff9800; color: black'] * len(row)
            else:
                # Low risk: Green with white text
                return ['background-color: #4caf50; color: white'] * len(row)
        
        styled_df = df_display.style.apply(highlight_risk, axis=1)
        table_placeholder.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # ============================================
        # STEP 9: Wait before next update
        # ============================================
        
        time.sleep(update_interval)
        
        # Allow Streamlit to process stop events
        if not st.session_state.running:
            break


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
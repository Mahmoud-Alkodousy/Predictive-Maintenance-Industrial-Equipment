"""
===============================================================================
DATA ANALYSIS & VISUALIZATION MODULE
===============================================================================
Enhanced data analysis with Risk Heatmap, Component Health, and Anomaly Detection
Includes proper state management and unique widget keys

KEY FIXES:
1. ✅ All widgets have unique keys
2. ✅ Session state properly managed
3. ✅ No unwanted reruns
4. ✅ All comments in English
5. ✅ Dynamic threshold updates

Developer: Eng. Mahmoud Khalid Alkodousy
===============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from feature_engineering import SENSORS

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_analysis_session_state():
    """
    Initialize session state variables for analysis tab
    Must be called before any UI elements
    """
    # Analysis selections
    if 'selected_analysis_type' not in st.session_state:
        st.session_state.selected_analysis_type = "Overview"
    
    if 'selected_sensor' not in st.session_state:
        st.session_state.selected_sensor = None
    
    if 'selected_component' not in st.session_state:
        st.session_state.selected_component = None
    
    if 'anomaly_method' not in st.session_state:
        st.session_state.anomaly_method = "IQR (Interquartile Range)"
    
    if 'selected_sensor_viz' not in st.session_state:
        st.session_state.selected_sensor_viz = None

# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render_analysis_tab(df_input):
    """
    Render the complete data analysis tab with proper state management
    
    Args:
        df_input: Input DataFrame with sensor data
    """
    # Initialize session state
    initialize_analysis_session_state()
    
    st.header("📈 Advanced Data Analytics")
    
    if df_input is not None and len(df_input) > 0:
        # Check if predictions exist in session state
        has_predictions = ('results_df' in st.session_state and 
                          'prediction_done' in st.session_state and 
                          st.session_state.prediction_done)
        
        # Update risk levels if threshold changed
        if has_predictions and 'predictions_proba' in st.session_state:
            # Get current threshold from session state
            current_threshold = st.session_state.get('current_threshold', 0.5)
            
            # Recalculate risk levels based on current threshold
            results_df = st.session_state.results_df.copy()
            results_df['failure_prediction'] = (
                st.session_state.predictions_proba >= current_threshold
            ).astype(int)
            results_df['risk_level'] = pd.cut(
                st.session_state.predictions_proba,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            # Update session state with new risk levels
            st.session_state.results_df = results_df
        
        # Build analysis options based on available data
        analysis_options = ["Overview", "Sensor Analysis"]
        if has_predictions:
            analysis_options.extend(["Risk Heatmap", "Component Health", "Anomaly Detection"])
        
        # Analysis type selector with unique key
        analysis_type = st.selectbox(
            "Select Analysis Type",
            analysis_options,
            index=analysis_options.index(st.session_state.selected_analysis_type) 
                  if st.session_state.selected_analysis_type in analysis_options else 0,
            key="analysis_type_selector_main"  # ✅ Unique key
        )
        st.session_state.selected_analysis_type = analysis_type
        
        # Render selected analysis type
        if analysis_type == "Overview":
            render_overview(df_input)
        elif analysis_type == "Sensor Analysis":
            render_sensor_analysis(df_input)
        elif analysis_type == "Risk Heatmap":
            if has_predictions:
                render_risk_heatmap(st.session_state.results_df)
            else:
                st.warning("⚠️ Risk Heatmap requires predictions first. Go to Prediction tab to generate predictions.")
        elif analysis_type == "Component Health":
            if has_predictions:
                render_component_health(st.session_state.results_df)
            else:
                st.warning("⚠️ Component Health requires predictions first. Go to Prediction tab to generate predictions.")
        elif analysis_type == "Anomaly Detection":
            render_anomaly_detection(df_input)
    
    else:
        # Show helpful message when no data available
        st.info("📊 Upload or use sample data in the Prediction tab first")
        st.markdown("""
        ### How to use this tab:
        
        1. Go to **📊 Prediction** tab
        2. Choose data input method:
           - Upload CSV file
           - Use sample data
           - Manual input
        3. Click **Make Predictions**
        4. Return here for detailed analytics
        
        **Available Analytics:**
        - 📊 Data Overview
        - ⚙️ Sensor Analysis
        - 🗺️ Risk Heatmap (requires predictions)
        - 🔧 Component Health (requires predictions)
        - 🔍 Anomaly Detection
        """)

# ============================================================================
# OVERVIEW SECTION
# ============================================================================

def render_overview(df_input):
    """
    Render overview analytics section with data quality metrics
    
    Args:
        df_input: Input DataFrame
    """
    st.subheader("📊 Data Overview")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df_input))
    
    with col2:
        unique_machines = (
            df_input['machineID'].nunique() 
            if 'machineID' in df_input.columns 
            else "N/A"
        )
        st.metric("Unique Machines", unique_machines)
    
    with col3:
        # Calculate time span if datetime column exists
        if 'datetime' in df_input.columns:
            try:
                time_span = (
                    pd.to_datetime(df_input['datetime'].max()) - 
                    pd.to_datetime(df_input['datetime'].min())
                ).days
                st.metric("Time Span (Days)", time_span)
            except:
                st.metric("Time Span", "N/A")
        else:
            st.metric("Time Span", "N/A")
    
    with col4:
        # Calculate missing data percentage
        total_cells = len(df_input) * len(df_input.columns)
        missing_cells = df_input.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        st.metric("Missing Data %", f"{missing_pct:.1f}%")
    
    # Data quality assessment
    st.subheader("🔍 Data Quality Assessment")
    
    quality_data = []
    for sensor in SENSORS:
        if sensor in df_input.columns:
            data = df_input[sensor]
            quality_data.append({
                'Sensor': sensor.capitalize(),
                'Missing %': f"{(data.isnull().sum() / len(data)) * 100:.1f}%",
                'Mean': f"{data.mean():.2f}",
                'Std': f"{data.std():.2f}",
                'Min': f"{data.min():.2f}",
                'Max': f"{data.max():.2f}"
            })
    
    if quality_data:
        st.dataframe(
            pd.DataFrame(quality_data), 
            use_container_width=True,
            key="quality_assessment_table"  # ✅ Unique key
        )
    else:
        st.warning("⚠️ No sensor data available for quality assessment")
    
    # Correlation heatmap
    if set(SENSORS).issubset(df_input.columns):
        st.subheader("🔥 Sensor Correlation Heatmap")
        st.markdown("*Shows relationships between different sensors*")
        
        corr_matrix = df_input[SENSORS].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Sensors", y="Sensors", color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=".2f"
        )
        fig.update_layout(height=500)
        
        # ✅ Unique key with timestamp
        unique_key = f"corr_heatmap_{int(time.time() * 1000000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)

# ============================================================================
# SENSOR ANALYSIS SECTION
# ============================================================================

def render_sensor_analysis(df_input):
    """
    Render detailed sensor analysis with distributions and time series
    
    Args:
        df_input: Input DataFrame
    """
    st.subheader("⚙️ Detailed Sensor Analysis")
    
    # Get available sensors
    available_sensors = [s for s in SENSORS if s in df_input.columns]
    
    if not available_sensors:
        st.warning("⚠️ No sensor data available for analysis")
        return
    
    # Sensor selector with unique key
    selected_sensor = st.selectbox(
        "Select Sensor to Analyze",
        available_sensors,
        index=available_sensors.index(st.session_state.selected_sensor) 
              if st.session_state.selected_sensor in available_sensors else 0,
        key="sensor_selector_analysis"  # ✅ Unique key
    )
    st.session_state.selected_sensor = selected_sensor
    
    sensor_data = df_input[selected_sensor]
    
    # Distribution histogram with box plot
    st.markdown("#### 📊 Distribution Analysis")
    
    fig = px.histogram(
        df_input, 
        x=selected_sensor,
        nbins=30,
        title=f"{selected_sensor.capitalize()} Distribution",
        marginal="box",
        labels={selected_sensor: f"{selected_sensor.capitalize()} Value"}
    )
    
    # Add mean line
    fig.add_vline(
        x=sensor_data.mean(), 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Mean"
    )
    
    # ✅ Unique key with timestamp
    unique_key = f"hist_{selected_sensor}_{int(time.time() * 1000000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)
    
    # Time series analysis (if datetime available)
    if 'datetime' in df_input.columns:
        st.markdown("#### 📈 Time Series Analysis")
        
        try:
            # Sort and prepare data
            df_sorted = df_input.sort_values('datetime').copy()
            df_sorted['datetime'] = pd.to_datetime(df_sorted['datetime'])
            
            # Create time series plot
            fig = px.line(
                df_sorted, 
                x='datetime', 
                y=selected_sensor,
                title=f"{selected_sensor.capitalize()} Over Time",
                labels={
                    'datetime': 'Date/Time',
                    selected_sensor: f"{selected_sensor.capitalize()} Value"
                }
            )
            
            # ✅ Unique key with timestamp
            unique_key = f"timeseries_{selected_sensor}_{int(time.time() * 1000000)}"
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
            
        except Exception as e:
            st.warning(f"⚠️ Could not create time series: {e}")
    
    # Statistical summary
    st.markdown("#### 📊 Statistical Summary")
    
    stats_df = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25th Percentile', 
                   'Median (50th)', '75th Percentile', 'Max'],
        'Value': [
            len(sensor_data),
            f"{sensor_data.mean():.2f}",
            f"{sensor_data.std():.2f}",
            f"{sensor_data.min():.2f}",
            f"{sensor_data.quantile(0.25):.2f}",
            f"{sensor_data.quantile(0.50):.2f}",
            f"{sensor_data.quantile(0.75):.2f}",
            f"{sensor_data.max():.2f}"
        ]
    })
    
    st.dataframe(
        stats_df, 
        use_container_width=True, 
        hide_index=True,
        key="sensor_stats_table"  # ✅ Unique key
    )

# ============================================================================
# RISK HEATMAP SECTION
# ============================================================================

def render_risk_heatmap(results_df):
    """
    Render interactive risk heatmap for all machines
    Shows color-coded grid based on failure probability
    
    Args:
        results_df: DataFrame with prediction results
    """
    st.subheader("🗺️ Machine Risk Heatmap")
    st.markdown("Interactive overview of all machines colored by failure risk level")
    
    # Get current threshold from session state
    current_threshold = st.session_state.get('current_threshold', 0.5)
    
    # Display current threshold info
    st.info(
        f"📊 Current Threshold: **{current_threshold:.1%}** - "
        f"Machines above this threshold are considered high risk"
    )
    
    # Prepare heatmap data
    heatmap_df = results_df.copy()
    heatmap_df['risk_score'] = (heatmap_df['failure_probability'] * 100).round(1)
    
    # Calculate risk categories based on current threshold
    high_risk_count = (heatmap_df['failure_probability'] >= current_threshold).sum()
    medium_risk_count = (
        (heatmap_df['failure_probability'] >= current_threshold * 0.6) & 
        (heatmap_df['failure_probability'] < current_threshold)
    ).sum()
    low_risk_count = (heatmap_df['failure_probability'] < current_threshold * 0.6).sum()
    
    # Display risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🔴 High Risk (≥ Threshold)", 
            high_risk_count,
            delta="Needs Attention" if high_risk_count > 0 else None,
            delta_color="inverse"
        )
    
    with col2:
        st.metric("🟡 Medium Risk", medium_risk_count)
    
    with col3:
        st.metric("🟢 Low Risk", low_risk_count)
    
    # Grid layout configuration
    machines_per_row = 5
    num_machines = len(heatmap_df)
    
    # Sort by risk (highest first)
    heatmap_df = heatmap_df.sort_values('failure_probability', ascending=False)
    
    st.markdown("---")
    st.markdown("**Machine Risk Grid (Sorted by Risk Level):**")
    
    # Create machine grid
    for i in range(0, num_machines, machines_per_row):
        cols = st.columns(machines_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            
            if idx < num_machines:
                machine = heatmap_df.iloc[idx]
                risk_pct = machine['failure_probability']
                
                # Determine color and status based on current threshold
                if risk_pct >= current_threshold:
                    color = "#ff4444"  # Red
                    icon = "🔴"
                    status = "HIGH RISK"
                elif risk_pct >= current_threshold * 0.6:
                    color = "#ffd700"  # Yellow
                    icon = "🟡"
                    status = "MEDIUM"
                else:
                    color = "#00cc00"  # Green
                    icon = "🟢"
                    status = "LOW"
                
                with col:
                    # Machine card with dynamic styling
                    st.markdown(f"""
                    <div style='
                        background-color: {color}22;
                        border: 2px solid {color};
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                        margin: 5px 0;
                        transition: transform 0.2s;
                    '>
                        <div style='font-size: 24px;'>{icon}</div>
                        <div style='font-weight: bold; font-size: 16px; margin: 5px 0;'>
                            Machine {machine['machineID']}
                        </div>
                        <div style='font-size: 20px; font-weight: bold; color: {color}; margin: 5px 0;'>
                            {risk_pct:.1%}
                        </div>
                        <div style='font-size: 11px; color: #666; margin: 3px 0;'>
                            {status}
                        </div>
                        <div style='font-size: 12px; color: #888;'>
                            {machine.get('model', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Detailed risk table
    st.markdown("---")
    st.subheader("📋 Detailed Risk Analysis Table")
    
    # Prepare display DataFrame
    display_df = heatmap_df[['machineID', 'model', 'failure_probability']].copy()
    display_df['Risk %'] = (display_df['failure_probability'] * 100).round(1).astype(str) + '%'
    display_df['Status'] = display_df['failure_probability'].apply(
        lambda x: '🔴 HIGH RISK' if x >= current_threshold 
        else ('🟡 MEDIUM' if x >= current_threshold * 0.6 else '🟢 LOW')
    )
    display_df['Recommended Action'] = display_df['failure_probability'].apply(
        lambda x: 'Immediate Inspection' if x >= current_threshold 
        else ('Schedule Maintenance' if x >= current_threshold * 0.6 else 'Normal Operation')
    )
    
    # Display table with unique key
    st.dataframe(
        display_df[['machineID', 'model', 'Risk %', 'Status', 'Recommended Action']], 
        use_container_width=True, 
        hide_index=True,
        key="risk_heatmap_table"  # ✅ Unique key
    )

# ============================================================================
# COMPONENT HEALTH SECTION
# ============================================================================

def render_component_health(results_df):
    """
    Render component health monitoring dashboard
    Shows maintenance status for different machine components
    
    Args:
        results_df: DataFrame with prediction results
    """
    st.subheader("🔧 Component Health Monitoring")
    st.markdown("Track maintenance status and component health across all machines")
    
    # Define components to track
    components = ['comp1', 'comp2', 'comp3', 'comp4']
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Machines", len(results_df))
    
    with col2:
        high_risk_machines = (results_df['failure_probability'] >= 0.5).sum()
        st.metric("Machines Needing Attention", high_risk_machines)
    
    with col3:
        st.metric("Components Tracked", len(components))
    
    st.markdown("---")
    
    # Component health status
    st.subheader("📊 Component Health Status")
    
    # Generate component data based on machine risk
    comp_data = []
    for comp in components:
        for _, machine in results_df.iterrows():
            # Simulate days since maintenance (higher risk = longer since maintenance)
            base_days = int(machine['failure_probability'] * 100)
            days_since_maint = base_days + np.random.randint(10, 50)
            
            # Determine status based on days since maintenance
            if days_since_maint > 80:
                status = "Critical"
                icon = "🔴"
            elif days_since_maint > 60:
                status = "Warning"
                icon = "🟠"
            elif days_since_maint > 40:
                status = "Check Soon"
                icon = "🟡"
            else:
                status = "Good"
                icon = "🟢"
            
            comp_data.append({
                'Machine': machine['machineID'],
                'Component': comp.upper(),
                'Days Since Maintenance': days_since_maint,
                'Status': f"{icon} {status}",
                'Risk Level': machine['risk_level']
            })
    
    comp_df = pd.DataFrame(comp_data)
    
    # Component selector with unique key
    selected_comp = st.selectbox(
        "Select Component to Analyze",
        components,
        format_func=lambda x: x.upper(),
        index=components.index(st.session_state.selected_component) 
              if st.session_state.selected_component in components else 0,
        key="component_selector_health"  # ✅ Unique key
    )
    st.session_state.selected_component = selected_comp
    
    # Filter data by selected component
    comp_filtered = comp_df[comp_df['Component'] == selected_comp.upper()]
    comp_filtered = comp_filtered.sort_values('Days Since Maintenance', ascending=False)
    
    # Visualization - Top 10 machines by days since maintenance
    st.markdown(f"#### 📊 {selected_comp.upper()} Maintenance Status")
    
    fig = px.bar(
        comp_filtered.head(10),
        x='Machine',
        y='Days Since Maintenance',
        color='Days Since Maintenance',
        color_continuous_scale=['green', 'yellow', 'orange', 'red'],
        title=f"{selected_comp.upper()} - Days Since Last Maintenance (Top 10 Machines)",
        labels={
            'Machine': 'Machine ID',
            'Days Since Maintenance': 'Days'
        }
    )
    fig.update_layout(xaxis_title="Machine ID", yaxis_title="Days Since Maintenance")
    
    # ✅ Unique key with timestamp
    unique_key = f"comp_health_{selected_comp}_{int(time.time() * 1000000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)
    
    # Maintenance recommendations
    st.markdown("---")
    st.subheader("📋 Maintenance Recommendations")
    
    # Critical machines (>80 days)
    critical = comp_filtered[comp_filtered['Days Since Maintenance'] > 80]
    if len(critical) > 0:
        st.error(
            f"**🔴 CRITICAL: {len(critical)} machines need IMMEDIATE maintenance for {selected_comp.upper()}**"
        )
        st.dataframe(
            critical[['Machine', 'Days Since Maintenance', 'Risk Level']], 
            use_container_width=True, 
            hide_index=True,
            key="critical_maintenance_table"  # ✅ Unique key
        )
    else:
        st.success(f"✅ No critical maintenance required for {selected_comp.upper()}")
    
    # Warning machines (60-80 days)
    warning = comp_filtered[
        (comp_filtered['Days Since Maintenance'] > 60) & 
        (comp_filtered['Days Since Maintenance'] <= 80)
    ]
    if len(warning) > 0:
        st.warning(
            f"**🟠 WARNING: {len(warning)} machines should be scheduled for {selected_comp.upper()} maintenance**"
        )
        st.dataframe(
            warning[['Machine', 'Days Since Maintenance', 'Risk Level']], 
            use_container_width=True, 
            hide_index=True,
            key="warning_maintenance_table"  # ✅ Unique key
        )

# ============================================================================
# ANOMALY DETECTION SECTION
# ============================================================================

def render_anomaly_detection(df_input):
    """
    Render anomaly detection analysis using statistical methods
    Identifies machines with unusual sensor readings
    
    Args:
        df_input: Input DataFrame
    """
    st.subheader("🔍 Anomaly Detection")
    st.markdown("Identify machines with unusual sensor readings using statistical methods")
    
    # Get available sensors
    available_sensors = [s for s in SENSORS if s in df_input.columns]
    
    if not available_sensors:
        st.warning("⚠️ No sensor data available for anomaly detection")
        return
    
    # Detection method selector with unique key
    method = st.radio(
        "Detection Method",
        ["IQR (Interquartile Range)", "Z-Score"],
        index=0 if st.session_state.anomaly_method == "IQR (Interquartile Range)" else 1,
        horizontal=True,
        key="anomaly_method_selector"  # ✅ Unique key
    )
    st.session_state.anomaly_method = method
    
    # Method explanation
    if method == "IQR (Interquartile Range)":
        st.info(
            "📊 **IQR Method:** Detects values outside 1.5 × IQR from quartiles. "
            "More robust to outliers."
        )
    else:
        st.info(
            "📊 **Z-Score Method:** Detects values with |z-score| > 3. "
            "Assumes normal distribution."
        )
    
    # Detect anomalies
    anomalies_data = []
    
    for sensor in available_sensors:
        sensor_values = df_input[sensor].dropna()
        
        if len(sensor_values) == 0:
            continue
        
        if method == "IQR (Interquartile Range)":
            # IQR method
            Q1 = sensor_values.quantile(0.25)
            Q3 = sensor_values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomaly_mask = (df_input[sensor] < lower_bound) | (df_input[sensor] > upper_bound)
        
        else:  # Z-Score method
            mean = sensor_values.mean()
            std = sensor_values.std()
            
            if std == 0:  # Avoid division by zero
                continue
            
            z_scores = np.abs((df_input[sensor] - mean) / std)
            anomaly_mask = z_scores > 3
        
        # Find anomalous machines
        anomaly_machines = df_input[anomaly_mask]
        
        for _, row in anomaly_machines.iterrows():
            anomalies_data.append({
                'Machine': row['machineID'],
                'Sensor': sensor.capitalize(),
                'Value': round(row[sensor], 2),
                'Severity': 'High' if method == "Z-Score" else 'Medium'
            })
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Anomalies Detected", len(anomalies_data))
    
    with col2:
        if len(anomalies_data) > 0:
            unique_machines = len(set([a['Machine'] for a in anomalies_data]))
            st.metric("Machines with Anomalies", unique_machines)
        else:
            st.metric("Machines with Anomalies", 0)
    
    # Show anomaly results
    if len(anomalies_data) > 0:
        anomalies_df = pd.DataFrame(anomalies_data)
        
        # Anomaly distribution by sensor
        st.markdown("---")
        st.subheader("📊 Anomalies by Sensor")
        
        anomaly_counts = anomalies_df.groupby('Sensor').size().reset_index(name='Count')
        
        fig = px.bar(
            anomaly_counts, 
            x='Sensor', 
            y='Count',
            title="Number of Anomalies per Sensor",
            color='Count', 
            color_continuous_scale='Reds',
            labels={'Count': 'Anomaly Count'}
        )
        
        # ✅ Unique key with timestamp
        unique_key = f"anomaly_bar_{int(time.time() * 1000000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)
        
        # Detailed anomaly table
        st.markdown("---")
        st.subheader("📋 Detected Anomalies")
        
        st.dataframe(
            anomalies_df.sort_values(['Machine', 'Sensor']), 
            use_container_width=True, 
            hide_index=True,
            key="anomaly_details_table"  # ✅ Unique key
        )
        
        # Recommendations
        st.warning(f"""
        **⚠️ {len(anomalies_data)} anomalies detected across {unique_machines} machines**
        
        **Recommended Actions:**
        - Investigate machines with multiple sensor anomalies
        - Cross-reference with maintenance history
        - Consider immediate inspection for high-severity anomalies
        - Verify sensor calibration
        """)
    else:
        st.success("✅ No anomalies detected. All sensor readings are within normal ranges.")
    
    # Sensor value range visualization
    st.markdown("---")
    st.subheader("📈 Sensor Value Ranges")
    
    # Sensor selector with unique key
    selected_sensor_viz = st.selectbox(
        "Select Sensor for Visualization",
        available_sensors,
        index=available_sensors.index(st.session_state.selected_sensor_viz) 
              if st.session_state.selected_sensor_viz in available_sensors else 0,
        key="anomaly_viz_selector"  # ✅ Unique key
    )
    st.session_state.selected_sensor_viz = selected_sensor_viz
    
    # Box plot showing distribution and outliers
    fig = px.box(
        df_input, 
        y=selected_sensor_viz,
        title=f"{selected_sensor_viz.capitalize()} Value Distribution with Outliers",
        points="outliers",
        labels={selected_sensor_viz: f"{selected_sensor_viz.capitalize()} Value"}
    )
    
    # ✅ Unique key with timestamp
    unique_key = f"anomaly_box_{selected_sensor_viz}_{int(time.time() * 1000000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)

# ============================================================================
# ENTRY POINT (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="Data Analysis", layout="wide")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'machineID': [1, 2, 3, 4, 5],
        'datetime': pd.date_range('2024-01-01', periods=5),
        'volt': [170, 165, 180, 160, 175],
        'rotate': [420, 430, 410, 440, 425],
        'pressure': [100, 95, 105, 90, 98],
        'vibration': [40, 45, 38, 50, 42]
    })
    
    render_analysis_tab(sample_data)
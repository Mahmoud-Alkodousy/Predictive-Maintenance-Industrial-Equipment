"""
3D Interactive Visualization Module
Creates interactive 3D plots for exploratory data analysis
Includes PCA analysis, sensor space visualization, and time series 3D plots
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from typing import Optional


# ============================================
# PCA 3D VISUALIZATION
# ============================================

def create_3d_pca_plot(df: pd.DataFrame, target_col: str = 'will_fail_in_24h') -> go.Figure:
    """
    Create 3D PCA visualization to reduce high-dimensional data to 3 dimensions
    
    PCA (Principal Component Analysis):
    - Projects high-dimensional data onto 3 principal components
    - Preserves maximum variance
    - Helps identify clusters and patterns
    
    Args:
        df (pd.DataFrame): Telemetry data
        target_col (str): Target column for coloring points
        
    Returns:
        go.Figure: Plotly 3D scatter plot
    """
    
    # ============================================
    # STEP 1: Prepare features for PCA
    # ============================================
    
    # Exclude non-feature columns
    exclude_cols = {'datetime', 'machineID', 'will_fail_in_24h', 'will_fail_in_48h'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df[target_col]
    
    # ============================================
    # STEP 2: Handle categorical columns
    # ============================================
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        st.info(f"🔄 Converting categorical columns: {', '.join(categorical_cols)}")
        
        # Encode categorical features to numeric
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # ============================================
    # STEP 3: Clean and normalize data
    # ============================================
    
    # Fill missing values with 0
    X = X.fillna(0)
    
    # Convert to numeric (handle any remaining non-numeric values)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # ============================================
    # STEP 4: Apply PCA transformation
    # ============================================
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # ============================================
    # STEP 5: Create 3D scatter plot
    # ============================================
    
    fig = go.Figure()
    
    # Add trace for normal machines (green)
    mask_normal = y == 0
    if mask_normal.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=X_pca[mask_normal, 0],
            y=X_pca[mask_normal, 1],
            z=X_pca[mask_normal, 2],
            mode='markers',
            name='Normal',
            marker=dict(
                size=3,
                color='green',
                opacity=0.6,
                line=dict(width=0)
            ),
            text=[f"Machine {m}" for m in df[mask_normal]['machineID']],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>Status: Normal<extra></extra>'
        ))
    
    # Add trace for failing machines (red)
    mask_fail = y == 1
    if mask_fail.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=X_pca[mask_fail, 0],
            y=X_pca[mask_fail, 1],
            z=X_pca[mask_fail, 2],
            mode='markers',
            name='Will Fail',
            marker=dict(
                size=4,
                color='red',
                opacity=0.8,
                line=dict(width=0)
            ),
            text=[f"Machine {m}" for m in df[mask_fail]['machineID']],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>Status: ⚠️ Will Fail<extra></extra>'
        ))
    
    # ============================================
    # STEP 6: Configure layout
    # ============================================
    
    # Calculate total explained variance
    total_variance = sum(pca.explained_variance_ratio_) * 100
    
    fig.update_layout(
        title={
            'text': f'3D PCA Visualization - Explained Variance: {total_variance:.1f}%',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            # Label axes with variance explained by each component
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)  # Initial viewing angle
            )
        ),
        height=700,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig


# ============================================
# SENSOR SPACE 3D VISUALIZATION
# ============================================

def create_3d_sensor_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create 3D plot showing relationships between sensors
    
    Visualizes 3 sensors as X, Y, Z axes with vibration shown as color
    Helps identify correlations and anomalies in sensor space
    
    Args:
        df (pd.DataFrame): Telemetry data
        
    Returns:
        go.Figure: Plotly 3D scatter plot
    """
    
    # ============================================
    # STEP 1: Sample data for performance
    # ============================================
    
    df_sample = df.sample(min(5000, len(df)))
    
    # ============================================
    # STEP 2: Validate required columns
    # ============================================
    
    required_cols = ['volt', 'rotate', 'pressure', 'vibration', 'machineID']
    missing_cols = [col for col in required_cols if col not in df_sample.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        return go.Figure()
    
    # ============================================
    # STEP 3: Create 3D scatter plot
    # ============================================
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=df_sample['volt'],
            y=df_sample['rotate'],
            z=df_sample['pressure'],
            mode='markers',
            marker=dict(
                size=4,
                color=df_sample['vibration'],  # Color by vibration level
                colorscale='Viridis',  # Yellow (low) to Purple (high)
                showscale=True,
                colorbar=dict(title="Vibration<br>(mm/s)"),
                opacity=0.7
            ),
            text=[f"Machine {m}<br>Vibration: {v:.2f}" 
                  for m, v in zip(df_sample['machineID'], df_sample['vibration'])],
            hovertemplate='<b>%{text}</b><br>Volt: %{x:.2f}V<br>Rotate: %{y:.2f} RPM<br>Pressure: %{z:.2f} bar<extra></extra>'
        )
    ])
    
    # ============================================
    # STEP 4: Configure layout
    # ============================================
    
    fig.update_layout(
        title='3D Sensor Space (Colored by Vibration)',
        scene=dict(
            xaxis_title='Voltage (V)',
            yaxis_title='Rotation (RPM)',
            zaxis_title='Pressure (bar)',
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.3)
            )
        ),
        height=700
    )
    
    return fig


# ============================================
# TIME SERIES 3D VISUALIZATION
# ============================================

def create_3d_time_series(df: pd.DataFrame) -> go.Figure:
    """
    Create 3D time series visualization with sensors on separate layers
    
    Shows temporal evolution of multiple sensors simultaneously
    Each sensor is plotted on a different Z-layer for clarity
    
    Args:
        df (pd.DataFrame): Time-sorted telemetry data for single machine
        
    Returns:
        go.Figure: Plotly 3D line plot
    """
    
    # ============================================
    # STEP 1: Validate data
    # ============================================
    
    if 'datetime' not in df.columns:
        st.error("❌ 'datetime' column not found")
        return go.Figure()
    
    # ============================================
    # STEP 2: Select machine and limit samples
    # ============================================
    
    # Get most common machine
    machine_id = df['machineID'].mode()[0]
    df_machine = df[df['machineID'] == machine_id].sort_values('datetime').head(500)
    
    if len(df_machine) == 0:
        st.error("❌ No data found for selected machine")
        return go.Figure()
    
    # Create numeric time index
    time_numeric = np.arange(len(df_machine))
    
    # ============================================
    # STEP 3: Create 3D plot with sensors on layers
    # ============================================
    
    fig = go.Figure()
    
    # Layer 0: Voltage (bottom)
    fig.add_trace(go.Scatter3d(
        x=time_numeric,
        y=df_machine['volt'],
        z=np.zeros(len(df_machine)),  # Z = 0
        mode='lines',
        name='Voltage',
        line=dict(color='blue', width=4),
        hovertemplate='Time: %{x}<br>Voltage: %{y:.2f}V<extra></extra>'
    ))
    
    # Layer 100: Rotation (middle)
    fig.add_trace(go.Scatter3d(
        x=time_numeric,
        y=df_machine['rotate'],
        z=np.ones(len(df_machine)) * 100,  # Z = 100
        mode='lines',
        name='Rotation',
        line=dict(color='red', width=4),
        hovertemplate='Time: %{x}<br>Rotation: %{y:.2f} RPM<extra></extra>'
    ))
    
    # Layer 200: Pressure (top)
    fig.add_trace(go.Scatter3d(
        x=time_numeric,
        y=df_machine['pressure'],
        z=np.ones(len(df_machine)) * 200,  # Z = 200
        mode='lines',
        name='Pressure',
        line=dict(color='green', width=4),
        hovertemplate='Time: %{x}<br>Pressure: %{y:.2f} bar<extra></extra>'
    ))
    
    # ============================================
    # STEP 4: Configure layout
    # ============================================
    
    fig.update_layout(
        title=f'3D Time Series - Machine {machine_id}',
        scene=dict(
            xaxis_title='Time (hours)',
            yaxis_title='Sensor Value',
            zaxis_title='Sensor Layer',
            zaxis=dict(
                tickmode='array',
                tickvals=[0, 100, 200],
                ticktext=['Voltage', 'Rotation', 'Pressure']
            ),
            camera=dict(
                eye=dict(x=2, y=-2, z=1.5)
            )
        ),
        height=700,
        showlegend=True
    )
    
    return fig


# ============================================
# MAIN STREAMLIT UI
# ============================================

def show_3d_visualizations():
    """
    Main function to display all 3D visualizations
    
    Features:
    - Tab-based interface
    - Interactive 3D plots
    - Real-time statistics
    - Error handling
    """
    
    st.title("📊 3D Interactive Visualizations")
    st.markdown("### Explore Your Data in Three Dimensions")
    
    # ============================================
    # LOAD DATA
    # ============================================
    
    from config import PROCESSED_DATA_DIR
    
    data_path = PROCESSED_DATA_DIR / "telemetry.csv"
    
    if not data_path.exists():
        st.error("❌ Processed data not found. Please run data_preprocessing.py first.")
        st.info(f"Expected path: {data_path}")
        return
    
    try:
        with st.spinner("Loading data..."):
            df = pd.read_csv(data_path, parse_dates=['datetime'])
            st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return
    
    # ============================================
    # CREATE TABS
    # ============================================
    
    tab1, tab2, tab3 = st.tabs([
        "🎯 PCA Analysis", 
        "🔬 Sensor Space", 
        "📈 Time Series 3D"
    ])
    
    # ============================================
    # TAB 1: PCA ANALYSIS
    # ============================================
    
    with tab1:
        st.markdown("#### PCA reduces high-dimensional data to 3D while preserving variance")
        
        # Configuration controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            sample_size = st.slider(
                "Sample Size", 
                1000, 
                min(50000, len(df)), 
                10000, 
                step=1000,
                help="Larger samples take longer to process"
            )
        
        with col2:
            target_col = st.selectbox(
                "Target Variable", 
                ['will_fail_in_24h', 'will_fail_in_48h']
            )
        
        # Sample data
        df_sample = df.sample(min(sample_size, len(df)))
        
        # Generate plot
        with st.spinner("Generating 3D PCA... (this may take a moment)"):
            try:
                fig = create_3d_pca_plot(df_sample, target_col=target_col)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                normal_count = (df_sample[target_col] == 0).sum()
                fail_count = (df_sample[target_col] == 1).sum()
                
                col1.metric(
                    "🟢 Normal", 
                    f"{normal_count:,}", 
                    f"{normal_count/len(df_sample)*100:.1f}%"
                )
                col2.metric(
                    "🔴 Will Fail", 
                    f"{fail_count:,}", 
                    f"{fail_count/len(df_sample)*100:.1f}%"
                )
                col3.metric("📊 Total", f"{len(df_sample):,}")
                
            except Exception as e:
                st.error(f"❌ Failed to generate PCA plot: {e}")
                with st.expander("🐛 Error Details"):
                    st.code(str(e))
        
        st.info("💡 **Tip:** Rotate the plot by dragging, zoom with scroll wheel. Green = Normal, Red = Will Fail")
    
    # ============================================
    # TAB 2: SENSOR SPACE
    # ============================================
    
    with tab2:
        st.markdown("#### Visualize relationships between sensors in 3D space")
        
        with st.spinner("Generating 3D sensor space..."):
            try:
                fig = create_3d_sensor_plot(df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show sensor statistics
                st.markdown("##### 📈 Sensor Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric(
                    "⚡ Voltage", 
                    f"{df['volt'].mean():.1f}V", 
                    f"±{df['volt'].std():.1f}V"
                )
                col2.metric(
                    "🔄 Rotation", 
                    f"{df['rotate'].mean():.0f} RPM", 
                    f"±{df['rotate'].std():.0f}"
                )
                col3.metric(
                    "💨 Pressure", 
                    f"{df['pressure'].mean():.1f} bar", 
                    f"±{df['pressure'].std():.1f}"
                )
                col4.metric(
                    "📳 Vibration", 
                    f"{df['vibration'].mean():.2f} mm/s", 
                    f"±{df['vibration'].std():.2f}"
                )
                
            except Exception as e:
                st.error(f"❌ Failed to generate sensor plot: {e}")
        
        st.info("💡 **Tip:** Color intensity shows vibration levels. Darker colors indicate higher vibration.")
    
    # ============================================
    # TAB 3: TIME SERIES 3D
    # ============================================
    
    with tab3:
        st.markdown("#### See how sensors evolve over time in 3D")
        
        # Machine selection
        available_machines = sorted(df['machineID'].unique())
        selected_machine = st.selectbox(
            "Select Machine", 
            available_machines[:10], 
            index=0
        )
        
        # Filter by machine
        df_machine = df[df['machineID'] == selected_machine].sort_values('datetime')
        
        if len(df_machine) == 0:
            st.warning(f"No data found for machine {selected_machine}")
        else:
            with st.spinner("Generating 3D time series..."):
                try:
                    fig = create_3d_time_series(df_machine)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show time range
                    st.info(
                        f"📅 Showing data from {df_machine['datetime'].min()} "
                        f"to {df_machine['datetime'].max()} ({len(df_machine)} readings)"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Failed to generate time series: {e}")
        
        st.info("💡 **Tip:** Each layer represents a different sensor. Time progresses from left to right.")


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    show_3d_visualizations()
"""
Business Intelligence Report Generator
Generates professional PDF and Excel reports with analytics and visualizations
Optimized for speed using matplotlib instead of plotly
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, 
    Spacer, PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed


# ============================================
# DATA ANALYSIS FUNCTIONS
# ============================================

@st.cache_data
def generate_summary_stats(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive summary statistics for report
    
    Args:
        df (pd.DataFrame): Telemetry data with sensor readings
        
    Returns:
        dict: Summary statistics including:
            - Machine counts
            - Reading counts
            - Date ranges
            - Failure rates
            - Average sensor values
    """
    stats = {
        # Machine and data coverage
        'total_machines': df['machineID'].nunique(),
        'total_readings': len(df),
        'date_range': f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}",
        
        # Failure metrics
        'failure_rate_24h': df['will_fail_in_24h'].mean() * 100,
        'failure_rate_48h': df['will_fail_in_48h'].mean() * 100,
        
        # Average sensor values
        'avg_voltage': df['volt'].mean(),
        'avg_rotation': df['rotate'].mean(),
        'avg_pressure': df['pressure'].mean(),
        'avg_vibration': df['vibration'].mean()
    }
    
    return stats


# ============================================
# CHART GENERATION FUNCTIONS (MATPLOTLIB)
# ============================================

def create_chart_failure_rate(df: pd.DataFrame) -> io.BytesIO:
    """
    Create daily failure rate trend chart
    
    Shows failure probability over time to identify patterns
    
    Args:
        df (pd.DataFrame): Telemetry data
        
    Returns:
        io.BytesIO: PNG image buffer
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate daily failure rates
    daily_failures = df.groupby(df['datetime'].dt.date)['will_fail_in_24h'].mean()
    
    # Plot line chart
    ax.plot(
        daily_failures.index, 
        daily_failures.values * 100, 
        color='#e74c3c',  # Red for failures
        linewidth=2, 
        marker='o', 
        markersize=4
    )
    
    # Styling
    ax.set_title("Daily Failure Rate Trend", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Failure Rate (%)", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def create_chart_sensors(df: pd.DataFrame) -> io.BytesIO:
    """
    Create sensor distribution histograms (2x2 grid)
    
    Shows distribution of sensor readings to identify anomalies
    
    Args:
        df (pd.DataFrame): Telemetry data
        
    Returns:
        io.BytesIO: PNG image buffer
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # ============================================
    # Voltage Distribution (Top-Left)
    # ============================================
    axes[0, 0].hist(
        df['volt'], 
        bins=30, 
        color='#3498db',  # Blue
        alpha=0.7, 
        edgecolor='black'
    )
    axes[0, 0].set_title("Voltage Distribution", fontweight='bold')
    axes[0, 0].set_xlabel("Voltage (V)")
    axes[0, 0].set_ylabel("Frequency")
    
    # ============================================
    # Rotation Distribution (Top-Right)
    # ============================================
    axes[0, 1].hist(
        df['rotate'], 
        bins=30, 
        color='#2ecc71',  # Green
        alpha=0.7, 
        edgecolor='black'
    )
    axes[0, 1].set_title("Rotation Distribution", fontweight='bold')
    axes[0, 1].set_xlabel("RPM")
    axes[0, 1].set_ylabel("Frequency")
    
    # ============================================
    # Pressure Distribution (Bottom-Left)
    # ============================================
    axes[1, 0].hist(
        df['pressure'], 
        bins=30, 
        color='#f39c12',  # Orange
        alpha=0.7, 
        edgecolor='black'
    )
    axes[1, 0].set_title("Pressure Distribution", fontweight='bold')
    axes[1, 0].set_xlabel("Pressure (bar)")
    axes[1, 0].set_ylabel("Frequency")
    
    # ============================================
    # Vibration Distribution (Bottom-Right)
    # ============================================
    axes[1, 1].hist(
        df['vibration'], 
        bins=30, 
        color='#e74c3c',  # Red
        alpha=0.7, 
        edgecolor='black'
    )
    axes[1, 1].set_title("Vibration Distribution", fontweight='bold')
    axes[1, 1].set_xlabel("Vibration (mm/s)")
    axes[1, 1].set_ylabel("Frequency")
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def create_chart_machine_health(df: pd.DataFrame) -> io.BytesIO:
    """
    Create top 15 machines by failure rate chart
    
    Identifies high-risk machines requiring immediate attention
    
    Args:
        df (pd.DataFrame): Telemetry data
        
    Returns:
        io.BytesIO: PNG image buffer
    """
    # Calculate failure rate per machine (top 15)
    machine_health = df.groupby('machineID')['will_fail_in_24h'].mean().sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    bars = ax.barh(
        range(len(machine_health)), 
        machine_health.values * 100, 
        color='#e74c3c'
    )
    
    # Configure axes
    ax.set_yticks(range(len(machine_health)))
    ax.set_yticklabels(machine_health.index)
    ax.set_xlabel("Failure Rate (%)", fontsize=10)
    ax.set_title("Top 15 Machines by Failure Rate", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels to bars
    for i, v in enumerate(machine_health.values * 100):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ============================================
# PDF REPORT GENERATION
# ============================================

def create_pdf_report(df: pd.DataFrame, stats: dict, charts: dict) -> io.BytesIO:
    """
    Generate professional PDF report with ReportLab
    
    Report Structure:
    1. Title Page
    2. Executive Summary (statistics table)
    3. Key Findings
    4. Visual Analytics (charts)
    5. Recommendations
    
    Args:
        df (pd.DataFrame): Filtered telemetry data
        stats (dict): Summary statistics
        charts (dict): Dictionary of chart_name -> image_buffer
        
    Returns:
        io.BytesIO: PDF file buffer
    """
    
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=0.75*inch, 
        leftMargin=0.75*inch,
        topMargin=1*inch, 
        bottomMargin=0.75*inch
    )
    
    # ============================================
    # Define Styles
    # ============================================
    
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # ============================================
    # Build Report Content
    # ============================================
    
    content = []
    
    # ============================================
    # PAGE 1: Title Page
    # ============================================
    
    content.append(Paragraph("PREDICTIVE MAINTENANCE", title_style))
    content.append(Paragraph("Business Intelligence Report", styles['Heading2']))
    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", 
        styles['Normal']
    ))
    content.append(PageBreak())
    
    # ============================================
    # PAGE 2: Executive Summary
    # ============================================
    
    content.append(Paragraph("Executive Summary", heading_style))
    
    # Summary statistics table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Machines Monitored', f"{stats['total_machines']:,}"],
        ['Total Sensor Readings', f"{stats['total_readings']:,}"],
        ['Analysis Period', stats['date_range']],
        ['24-Hour Failure Rate', f"{stats['failure_rate_24h']:.2f}%"],
        ['48-Hour Failure Rate', f"{stats['failure_rate_48h']:.2f}%"],
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # Body styling
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    
    content.append(summary_table)
    content.append(Spacer(1, 0.3*inch))
    
    # ============================================
    # KEY FINDINGS SECTION
    # ============================================
    
    content.append(Paragraph("Key Findings", heading_style))
    
    findings = [
        f"• Average sensor readings are within normal operational ranges",
        f"• Voltage: {stats['avg_voltage']:.2f}V (Target: 170V ±10V)",
        f"• Rotation: {stats['avg_rotation']:.2f} RPM (Target: 420 RPM ±30 RPM)",
        f"• Pressure: {stats['avg_pressure']:.2f} bar (Target: 100 bar ±15 bar)",
        f"• Vibration: {stats['avg_vibration']:.2f} mm/s (Target: &lt;42 mm/s)",
        f"",
        f"• Failure prediction models show {stats['failure_rate_24h']:.2f}% of readings indicate potential failures within 24 hours",
        f"• Recommend increased monitoring for high-risk machines"
    ]
    
    for finding in findings:
        content.append(Paragraph(finding, styles['Normal']))
    
    content.append(PageBreak())
    
    # ============================================
    # PAGE 3+: Visual Analytics (Charts)
    # ============================================
    
    if charts:
        content.append(Paragraph("Visual Analytics", heading_style))
        
        for chart_name, chart_buffer in charts.items():
            content.append(Paragraph(chart_name, styles['Heading3']))
            content.append(Spacer(1, 0.1*inch))
            
            # Add chart image from buffer
            chart_buffer.seek(0)
            img = Image(chart_buffer, width=5.5*inch, height=3.5*inch)
            content.append(img)
            content.append(Spacer(1, 0.2*inch))
        
        content.append(PageBreak())
    
    # ============================================
    # FINAL PAGE: Recommendations
    # ============================================
    
    content.append(Paragraph("Recommendations", heading_style))
    
    recommendations = [
        "1. <b>Immediate Actions:</b>",
        "   • Schedule maintenance for machines with &gt;70% failure probability",
        "   • Inspect sensor calibration on machines with abnormal readings",
        "",
        "2. <b>Short-term (1-2 weeks):</b>",
        "   • Implement real-time monitoring dashboard",
        "   • Set up automated alerts for critical thresholds",
        "   • Conduct training sessions for maintenance staff",
        "",
        "3. <b>Long-term (1-3 months):</b>",
        "   • Upgrade aging machines (&gt;15 years old)",
        "   • Invest in IoT sensors for better data quality",
        "   • Develop preventive maintenance schedule based on ML predictions"
    ]
    
    for rec in recommendations:
        content.append(Paragraph(rec, styles['Normal']))
        content.append(Spacer(1, 0.05*inch))
    
    # ============================================
    # Build Final PDF
    # ============================================
    
    doc.build(content)
    
    buffer.seek(0)
    return buffer


# ============================================
# STREAMLIT UI
# ============================================

def show_report_generator():
    """
    Main Streamlit interface for report generation
    
    Features:
    - Date range filtering
    - Chart selection
    - PDF/Excel export options
    - Data sampling for performance
    """
    
    st.title("📄 Business Intelligence Report Generator")
    st.markdown("### Generate Executive-Ready PDF Reports")
    
    # ============================================
    # LOAD DATA
    # ============================================
    
    from config import PROCESSED_DATA_DIR
    data_path = PROCESSED_DATA_DIR / "telemetry.csv"
    
    if not data_path.exists():
        st.error("❌ Data not found. Please run preprocessing first.")
        return
    
    with st.spinner("Loading data..."):
        df = pd.read_csv(data_path, parse_dates=['datetime'])
    
    # ============================================
    # SIDEBAR: Report Configuration
    # ============================================
    
    st.sidebar.title("⚙️ Report Settings")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['datetime'].min(), df['datetime'].max())
    )
    
    # Chart selection (empty by default for speed)
    include_charts = st.sidebar.multiselect(
        "Include Charts",
        [
            "Failure Rate Over Time", 
            "Sensor Distributions", 
            "Machine Health Summary"
        ],
        default=[],  # No charts by default for faster generation
        help="Adding charts will increase generation time"
    )
    
    # Export format selection
    report_format = st.sidebar.selectbox(
        "Report Format",
        ["PDF", "Excel"]
    )
    
    # ============================================
    # FILTER DATA BY DATE RANGE
    # ============================================
    
    df_filtered = df[
        (df['datetime'].dt.date >= date_range[0]) & 
        (df['datetime'].dt.date <= date_range[1])
    ]
    
    # ============================================
    # PERFORMANCE: Sample large datasets
    # ============================================
    
    max_rows = 30000
    if len(df_filtered) > max_rows:
        st.info(
            f"📊 Using sample of {max_rows:,} records from {len(df_filtered):,} "
            f"for faster processing"
        )
        df_display = df_filtered.sample(max_rows, random_state=42)
    else:
        df_display = df_filtered
    
    # ============================================
    # PREVIEW METRICS
    # ============================================
    
    st.markdown("### 📊 Report Preview")
    
    col1, col2, col3 = st.columns(3)
    
    stats = generate_summary_stats(df_filtered)
    
    col1.metric("Total Machines", f"{stats['total_machines']:,}")
    col2.metric("Total Readings", f"{stats['total_readings']:,}")
    col3.metric("Failure Rate (24h)", f"{stats['failure_rate_24h']:.2f}%")
    
    # ============================================
    # GENERATE REPORT BUTTON
    # ============================================
    
    if st.button("📥 Generate Report", type="primary", use_container_width=True):
        
        charts = {}
        
        # ============================================
        # STEP 1: Generate Charts (if selected)
        # ============================================
        
        if include_charts:
            with st.spinner("Generating charts... ⚡"):
                
                if "Failure Rate Over Time" in include_charts:
                    st.write("📈 Creating failure rate chart...")
                    chart_buf = create_chart_failure_rate(df_display)
                    charts["Failure Rate Over Time"] = chart_buf
                
                if "Sensor Distributions" in include_charts:
                    st.write("📊 Creating sensor distributions...")
                    chart_buf = create_chart_sensors(df_display)
                    charts["Sensor Distributions"] = chart_buf
                
                if "Machine Health Summary" in include_charts:
                    st.write("🏥 Creating machine health summary...")
                    chart_buf = create_chart_machine_health(df_display)
                    charts["Machine Health Summary"] = chart_buf
                
                st.success(f"✅ Generated {len(charts)} chart(s)")
        
        # ============================================
        # STEP 2: Generate Report File
        # ============================================
        
        with st.spinner(f"Building {report_format} report..."):
            
            if report_format == "PDF":
                # Generate PDF
                pdf_buffer = create_pdf_report(df_filtered, stats, charts)
                
                st.success("✅ Report generated successfully!")
                
                # Download button
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            elif report_format == "Excel":
                # Generate Excel
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    # Summary sheet
                    pd.DataFrame([stats]).T.to_excel(writer, sheet_name='Summary')
                    
                    # Data sheet (limited to 10k rows for performance)
                    df_filtered.head(10000).to_excel(writer, sheet_name='Data', index=False)
                
                excel_buffer.seek(0)
                
                st.success("✅ Report generated successfully!")
                
                # Download button
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_buffer,
                    file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    show_report_generator()
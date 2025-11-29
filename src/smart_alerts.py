"""
===============================================================================
SMART ALERT SYSTEM - FIXED VERSION
===============================================================================
Smart Alert System with Email Notifications and proper state management

KEY FIXES:
1. ✅ All widgets have unique keys
2. ✅ Session state properly managed
3. ✅ No unwanted reruns
4. ✅ Stable UI interactions
5. ✅ All comments in English

Version: 2.0 (Fixed State Management)
===============================================================================
"""

import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pandas as pd
from datetime import datetime, timedelta
import os

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_alert_session_state():
    """
    Initialize all session state variables for alert system
    Must be called before any UI elements
    """
    # Alert configuration
    if 'alert_recipient_email' not in st.session_state:
        st.session_state.alert_recipient_email = "mahmoudkhaledalkoudosy@gmail.com"
    
    if 'alert_threshold' not in st.session_state:
        st.session_state.alert_threshold = 0.7
    
    if 'alert_frequency' not in st.session_state:
        st.session_state.alert_frequency = "Immediate"
    
    # Notification channels
    if 'enable_email' not in st.session_state:
        st.session_state.enable_email = True
    
    if 'enable_sms' not in st.session_state:
        st.session_state.enable_sms = False
    
    if 'enable_slack' not in st.session_state:
        st.session_state.enable_slack = False
    
    # Test alert data
    if 'test_machine_id' not in st.session_state:
        st.session_state.test_machine_id = 42
    
    if 'test_failure_prob' not in st.session_state:
        st.session_state.test_failure_prob = 0.85
    
    if 'test_sensor_volt' not in st.session_state:
        st.session_state.test_sensor_volt = 165.5
    
    if 'test_sensor_rotate' not in st.session_state:
        st.session_state.test_sensor_rotate = 445.2
    
    if 'test_sensor_pressure' not in st.session_state:
        st.session_state.test_sensor_pressure = 118.7
    
    if 'test_sensor_vibration' not in st.session_state:
        st.session_state.test_sensor_vibration = 43.1
    
    # Alert history
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = generate_sample_alert_history()
    
    # Alert statistics
    if 'alert_stats_last_updated' not in st.session_state:
        st.session_state.alert_stats_last_updated = datetime.now()

# ============================================================================
# EMAIL SENDING FUNCTIONS
# ============================================================================

def send_email_alert(recipient, subject, body, html_body=None):
    """
    Send email alert with plain text and optional HTML version
    
    Args:
        recipient: Email address to send to
        subject: Email subject line
        body: Plain text email body
        html_body: Optional HTML email body
    
    Returns:
        tuple: (success: bool, message: str)
    """
    sender_email = os.getenv("ALERT_EMAIL", "mahmoudkhaledalkoudosy@gmail.com")
    sender_password = os.getenv("ALERT_EMAIL_PASSWORD", "")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    
    # Check if password is set
    if not sender_password:
        return False, "Email password not configured. Set ALERT_EMAIL_PASSWORD in .env file"
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient
        
        # Text version (always include)
        text_part = MIMEText(body, 'plain')
        msg.attach(text_part)
        
        # HTML version (optional)
        if html_body:
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return True, "Email sent successfully"
    
    except smtplib.SMTPAuthenticationError as e:
        return False, f"Authentication failed: {str(e)}\n\nPlease check:\n1. ALERT_EMAIL_PASSWORD is correct\n2. 'Less secure app access' is enabled (Gmail)\n3. Or use App Password (Gmail)"
    
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# ============================================================================
# HTML EMAIL TEMPLATE
# ============================================================================

def create_alert_html(machine_id, failure_prob, sensor_data):
    """
    Create professional HTML email template for alerts
    
    Args:
        machine_id: Machine identifier
        failure_prob: Failure probability (0-1)
        sensor_data: Dictionary with sensor readings
    
    Returns:
        str: HTML email content
    """
    # Determine risk color based on probability
    if failure_prob > 0.8:
        risk_color = "#e74c3c"
        risk_text = "CRITICAL"
    elif failure_prob > 0.6:
        risk_color = "#f39c12"
        risk_text = "HIGH"
    else:
        risk_color = "#f1c40f"
        risk_text = "MEDIUM"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 30px; 
                text-align: center; 
                border-radius: 10px 10px 0 0; 
            }}
            .header h1 {{ margin: 0; font-size: 28px; }}
            .header p {{ margin: 10px 0 0 0; font-size: 14px; opacity: 0.9; }}
            .alert-box {{ 
                background: {risk_color}; 
                color: white; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 8px; 
                text-align: center; 
            }}
            .alert-box h2 {{ margin: 0; font-size: 24px; }}
            .alert-box p {{ margin: 10px 0 0 0; font-size: 16px; }}
            .info-section {{ 
                background: #f8f9fa; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 8px; 
            }}
            .info-section h3 {{ margin-top: 0; color: #2c3e50; }}
            .sensor-grid {{ 
                display: grid; 
                grid-template-columns: repeat(2, 1fr); 
                gap: 15px; 
                margin-top: 15px;
            }}
            .sensor-card {{ 
                background: white; 
                padding: 15px; 
                border-radius: 6px; 
                border-left: 4px solid #3498db; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .sensor-card h4 {{ margin: 0 0 5px 0; color: #2c3e50; font-size: 14px; }}
            .sensor-card p {{ margin: 0; font-size: 20px; font-weight: bold; color: #3498db; }}
            .recommendations {{ margin: 15px 0; }}
            .recommendations ul {{ margin: 10px 0; padding-left: 20px; }}
            .recommendations li {{ margin: 8px 0; }}
            .footer {{ 
                text-align: center; 
                color: #7f8c8d; 
                padding: 20px; 
                font-size: 12px; 
                border-top: 1px solid #ecf0f1;
                margin-top: 20px;
            }}
            .btn {{ 
                display: inline-block; 
                padding: 12px 30px; 
                background: #3498db; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                margin-top: 20px; 
            }}
            .btn:hover {{ background: #2980b9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚨 {risk_text} ALERT</h1>
                <p>Predictive Maintenance System</p>
            </div>
            
            <div class="alert-box">
                <h2>Machine {machine_id} Requires Attention</h2>
                <p>Failure Probability: {failure_prob*100:.1f}%</p>
                <p>Risk Level: {risk_text}</p>
            </div>
            
            <div class="info-section">
                <h3>📊 Current Sensor Readings</h3>
                <div class="sensor-grid">
                    <div class="sensor-card">
                        <h4>⚡ Voltage</h4>
                        <p>{sensor_data['volt']:.2f} V</p>
                    </div>
                    <div class="sensor-card">
                        <h4>🔄 Rotation</h4>
                        <p>{sensor_data['rotate']:.2f} RPM</p>
                    </div>
                    <div class="sensor-card">
                        <h4>💨 Pressure</h4>
                        <p>{sensor_data['pressure']:.2f} bar</p>
                    </div>
                    <div class="sensor-card">
                        <h4>📳 Vibration</h4>
                        <p>{sensor_data['vibration']:.2f} mm/s</p>
                    </div>
                </div>
            </div>
            
            <div class="info-section recommendations">
                <h3>🎯 Recommended Actions</h3>
                <ul>
                    <li><strong>Immediate:</strong> Schedule inspection of Machine {machine_id}</li>
                    <li><strong>Check:</strong> Abnormal vibration patterns and sensor calibration</li>
                    <li><strong>Review:</strong> Maintenance logs for recent activities</li>
                    <li><strong>Prepare:</strong> Replacement parts if necessary</li>
                    <li><strong>Monitor:</strong> Continue tracking sensor readings closely</li>
                </ul>
            </div>
            
            <div style="text-align: center;">
                <a href="http://localhost:8501" class="btn">View Full Dashboard</a>
            </div>
            
            <div class="footer">
                <p><strong>Alert Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Predictive Maintenance System v2.0</p>
                <p style="margin-top: 10px; font-size: 11px;">
                    This is an automated alert. Do not reply to this email.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================

def generate_sample_alert_history():
    """
    Generate sample alert history for demonstration
    
    Returns:
        DataFrame: Sample alert history
    """
    # Generate timestamps for last 24 hours
    timestamps = pd.date_range(end=datetime.now(), periods=10, freq='2H')
    
    alert_history = pd.DataFrame({
        'Timestamp': timestamps,
        'Machine ID': [42, 17, 89, 42, 3, 56, 42, 23, 11, 42],
        'Risk Level': ['Critical', 'Warning', 'Normal', 'Critical', 'Warning', 
                       'Normal', 'Critical', 'Warning', 'Normal', 'Critical'],
        'Probability': [0.92, 0.68, 0.23, 0.87, 0.61, 0.31, 0.89, 0.65, 0.28, 0.91],
        'Action Taken': ['Inspected', 'Monitored', '-', 'Scheduled', 'Monitored', 
                        '-', 'Pending', 'Monitored', '-', 'Pending']
    })
    
    return alert_history

# ============================================================================
# STYLING FUNCTIONS
# ============================================================================

def highlight_risk(row):
    """
    Apply color styling based on risk level
    
    Args:
        row: DataFrame row
    
    Returns:
        list: CSS styles for each column
    """
    if row['Risk Level'] == 'Critical':
        return ['background-color: #ffcdd2; color: #b71c1c; font-weight: bold'] * len(row)
    elif row['Risk Level'] == 'Warning':
        return ['background-color: #ffe0b2; color: #e65100; font-weight: bold'] * len(row)
    else:
        return ['background-color: #c8e6c9; color: #1b5e20; font-weight: bold'] * len(row)

# ============================================================================
# MAIN UI FUNCTION
# ============================================================================

def show_alert_system():
    """
    Main function for alert system configuration and testing
    Includes proper state management and unique widget keys
    """
    # Initialize session state FIRST
    initialize_alert_session_state()
    
    st.title("🔔 Smart Alert System")
    st.markdown("### Configure Automated Notifications")
    
    # ========================================
    # SIDEBAR CONFIGURATION
    # ========================================
    st.sidebar.title("⚙️ Alert Settings")
    
    # Email settings
    st.sidebar.markdown("#### 📧 Email Configuration")
    
    recipient_email = st.sidebar.text_input(
        "Recipient Email", 
        value=st.session_state.alert_recipient_email,
        key="sidebar_recipient_email"  # ✅ Unique key
    )
    st.session_state.alert_recipient_email = recipient_email
    
    alert_threshold = st.sidebar.slider(
        "Alert Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.alert_threshold, 
        step=0.05,
        key="sidebar_alert_threshold"  # ✅ Unique key
    )
    st.session_state.alert_threshold = alert_threshold
    
    # Alert frequency
    alert_frequency = st.sidebar.selectbox(
        "Alert Frequency",
        ["Immediate", "Hourly Digest", "Daily Digest"],
        index=["Immediate", "Hourly Digest", "Daily Digest"].index(st.session_state.alert_frequency),
        key="sidebar_alert_frequency"  # ✅ Unique key
    )
    st.session_state.alert_frequency = alert_frequency
    
    # Alert channels
    st.sidebar.markdown("#### 📱 Notification Channels")
    
    enable_email = st.sidebar.checkbox(
        "Email", 
        value=st.session_state.enable_email,
        key="sidebar_enable_email"  # ✅ Unique key
    )
    st.session_state.enable_email = enable_email
    
    enable_sms = st.sidebar.checkbox(
        "SMS", 
        value=st.session_state.enable_sms,
        key="sidebar_enable_sms"  # ✅ Unique key
    )
    st.session_state.enable_sms = enable_sms
    
    enable_slack = st.sidebar.checkbox(
        "Slack", 
        value=st.session_state.enable_slack,
        key="sidebar_enable_slack"  # ✅ Unique key
    )
    st.session_state.enable_slack = enable_slack
    
    # ========================================
    # TEST ALERT SECTION
    # ========================================
    st.markdown("### 🧪 Test Alert System")
    
    col1, col2 = st.columns(2)
    
    # Left column: Input data
    with col1:
        st.markdown("#### Sample Alert Data")
        
        test_machine = st.number_input(
            "Machine ID", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.test_machine_id,
            key="test_machine_id_input"  # ✅ Unique key
        )
        st.session_state.test_machine_id = test_machine
        
        test_prob = st.slider(
            "Failure Probability", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.test_failure_prob, 
            step=0.05,
            key="test_failure_prob_slider"  # ✅ Unique key
        )
        st.session_state.test_failure_prob = test_prob
        
        # Sensor data inputs
        test_volt = st.number_input(
            "Voltage", 
            value=st.session_state.test_sensor_volt,
            key="test_sensor_volt_input"  # ✅ Unique key
        )
        st.session_state.test_sensor_volt = test_volt
        
        test_rotate = st.number_input(
            "Rotation", 
            value=st.session_state.test_sensor_rotate,
            key="test_sensor_rotate_input"  # ✅ Unique key
        )
        st.session_state.test_sensor_rotate = test_rotate
        
        test_pressure = st.number_input(
            "Pressure", 
            value=st.session_state.test_sensor_pressure,
            key="test_sensor_pressure_input"  # ✅ Unique key
        )
        st.session_state.test_sensor_pressure = test_pressure
        
        test_vibration = st.number_input(
            "Vibration", 
            value=st.session_state.test_sensor_vibration,
            key="test_sensor_vibration_input"  # ✅ Unique key
        )
        st.session_state.test_sensor_vibration = test_vibration
        
        # Collect sensor data
        test_sensor_data = {
            'volt': test_volt,
            'rotate': test_rotate,
            'pressure': test_pressure,
            'vibration': test_vibration
        }
    
    # Right column: Alert preview
    with col2:
        st.markdown("#### Alert Preview")
        
        # Show alert based on threshold
        if test_prob > alert_threshold:
            st.error(f"🚨 CRITICAL: Machine {test_machine} at {test_prob*100:.1f}% risk!")
        elif test_prob > alert_threshold * 0.7:
            st.warning(f"⚠️ WARNING: Machine {test_machine} at {test_prob*100:.1f}% risk")
        else:
            st.success(f"✅ NORMAL: Machine {test_machine} at {test_prob*100:.1f}% risk")
        
        st.markdown("---")
        
        # Display sensor readings
        st.markdown(f"**⚡ Voltage:** {test_sensor_data['volt']:.2f} V")
        st.markdown(f"**🔄 Rotation:** {test_sensor_data['rotate']:.2f} RPM")
        st.markdown(f"**💨 Pressure:** {test_sensor_data['pressure']:.2f} bar")
        st.markdown(f"**📳 Vibration:** {test_sensor_data['vibration']:.2f} mm/s")
    
    # ========================================
    # SEND TEST ALERT BUTTON
    # ========================================
    if st.button(
        "📨 Send Test Alert", 
        type="primary", 
        use_container_width=True,
        key="send_test_alert_button"  # ✅ Unique key
    ):
        
        # Email alert
        if enable_email:
            with st.spinner("Sending email..."):
                
                # Determine risk level for subject
                if test_prob > 0.8:
                    risk_label = "CRITICAL"
                elif test_prob > 0.6:
                    risk_label = "HIGH"
                else:
                    risk_label = "MEDIUM"
                
                subject = f"🚨 {risk_label} ALERT: Machine {test_machine} - {test_prob*100:.0f}% Failure Risk"
                
                # Plain text version
                text_body = f"""
PREDICTIVE MAINTENANCE ALERT - {risk_label}

Machine ID: {test_machine}
Failure Probability: {test_prob*100:.1f}%
Risk Level: {risk_label}

Current Sensor Readings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ Voltage:    {test_sensor_data['volt']:.2f} V
🔄 Rotation:   {test_sensor_data['rotate']:.2f} RPM
💨 Pressure:   {test_sensor_data['pressure']:.2f} bar
📳 Vibration:  {test_sensor_data['vibration']:.2f} mm/s

Recommended Actions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Schedule immediate inspection of Machine {test_machine}
2. Check for abnormal vibration patterns
3. Verify sensor calibration
4. Review maintenance logs for recent activities
5. Prepare replacement parts if necessary

Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is an automated alert from Predictive Maintenance System v2.0
                """
                
                # HTML version
                html_body = create_alert_html(test_machine, test_prob, test_sensor_data)
                
                # Send email
                success, message = send_email_alert(
                    recipient_email,
                    subject,
                    text_body,
                    html_body
                )
                
                # Display result
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
                    
                    # Show setup instructions
                    with st.expander("📖 Email Setup Instructions"):
                        st.markdown("""
                        ### Gmail Setup (Recommended)
                        
                        1. **Create App Password:**
                           - Go to [Google Account Security](https://myaccount.google.com/security)
                           - Enable 2-Step Verification
                           - Go to App Passwords
                           - Generate new app password for "Mail"
                        
                        2. **Update .env file:**
                           ```
                           ALERT_EMAIL=your-email@gmail.com
                           ALERT_EMAIL_PASSWORD=your-16-digit-app-password
                           SMTP_SERVER=smtp.gmail.com
                           SMTP_PORT=587
                           ```
                        
                        3. **Alternative (Less Secure):**
                           - Enable "Less secure app access" in Gmail settings
                           - Use your regular Gmail password
                           - **Not recommended for production**
                        
                        ### Other Email Providers:
                        
                        **Outlook/Hotmail:**
                        ```
                        SMTP_SERVER=smtp-mail.outlook.com
                        SMTP_PORT=587
                        ```
                        
                        **Yahoo:**
                        ```
                        SMTP_SERVER=smtp.mail.yahoo.com
                        SMTP_PORT=587
                        ```
                        """)
        
        # SMS alert (not implemented)
        if enable_sms:
            st.info("📱 SMS alerts require Twilio API (coming soon)")
            with st.expander("📱 Twilio Setup"):
                st.markdown("""
                ### SMS Alert Setup (Future Feature)
                
                1. Create Twilio account
                2. Get Account SID and Auth Token
                3. Buy a phone number
                4. Add to .env:
                   ```
                   TWILIO_ACCOUNT_SID=your-sid
                   TWILIO_AUTH_TOKEN=your-token
                   TWILIO_PHONE_NUMBER=+1234567890
                   ```
                """)
        
        # Slack alert (not implemented)
        if enable_slack:
            st.info("💬 Slack alerts require webhook URL (coming soon)")
            with st.expander("💬 Slack Setup"):
                st.markdown("""
                ### Slack Webhook Setup (Future Feature)
                
                1. Create Slack app
                2. Enable Incoming Webhooks
                3. Create webhook for your channel
                4. Add to .env:
                   ```
                   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
                   ```
                """)
    
    # ========================================
    # ALERT HISTORY SECTION
    # ========================================
    st.markdown("---")
    st.markdown("### 📜 Alert History")
    
    # Get alert history from session state
    alert_history = st.session_state.alert_history
    
    # Apply styling
    styled_df = alert_history.style.apply(highlight_risk, axis=1)
    
    # Display with unique key
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        hide_index=True,
        key="alert_history_dataframe"  # ✅ Unique key
    )
    
    # ========================================
    # ALERT STATISTICS
    # ========================================
    st.markdown("### 📈 Alert Statistics (Last 24 Hours)")
    
    # Calculate statistics
    total_alerts = len(alert_history)
    critical_count = (alert_history['Risk Level'] == 'Critical').sum()
    warning_count = (alert_history['Risk Level'] == 'Warning').sum()
    normal_count = (alert_history['Risk Level'] == 'Normal').sum()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Alerts", 
            total_alerts, 
            f"+{total_alerts - 44}" if total_alerts > 44 else "0"
        )
    
    with col2:
        st.metric(
            "Critical", 
            critical_count, 
            f"+{critical_count - 11}" if critical_count > 11 else "0"
        )
    
    with col3:
        st.metric(
            "Warnings", 
            warning_count, 
            f"+{warning_count - 26}" if warning_count > 26 else "0"
        )
    
    with col4:
        avg_response_time = 8  # Mock data
        st.metric(
            "Avg Response Time", 
            f"{avg_response_time} min", 
            "-2 min"
        )
    
    # ========================================
    # ADDITIONAL STATISTICS
    # ========================================
    st.markdown("---")
    st.markdown("### 📊 Detailed Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Risk Distribution")
        
        # Calculate percentages
        critical_pct = (critical_count / total_alerts * 100) if total_alerts > 0 else 0
        warning_pct = (warning_count / total_alerts * 100) if total_alerts > 0 else 0
        normal_pct = (normal_count / total_alerts * 100) if total_alerts > 0 else 0
        
        st.markdown(f"- 🔴 **Critical:** {critical_count} ({critical_pct:.1f}%)")
        st.markdown(f"- 🟡 **Warning:** {warning_count} ({warning_pct:.1f}%)")
        st.markdown(f"- 🟢 **Normal:** {normal_count} ({normal_pct:.1f}%)")
    
    with col2:
        st.markdown("#### Top Machines by Alerts")
        
        # Count alerts per machine
        machine_counts = alert_history['Machine ID'].value_counts().head(3)
        
        for idx, (machine_id, count) in enumerate(machine_counts.items(), 1):
            st.markdown(f"{idx}. **Machine {machine_id}:** {count} alerts")
    
    # ========================================
    # CONFIGURATION SUMMARY
    # ========================================
    st.markdown("---")
    st.markdown("### ⚙️ Current Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("**Email Settings:**")
        st.markdown(f"- Recipient: `{recipient_email}`")
        st.markdown(f"- Threshold: `{alert_threshold:.0%}`")
        st.markdown(f"- Frequency: `{alert_frequency}`")
    
    with config_col2:
        st.markdown("**Active Channels:**")
        st.markdown(f"- Email: {'✅ Enabled' if enable_email else '❌ Disabled'}")
        st.markdown(f"- SMS: {'✅ Enabled' if enable_sms else '❌ Disabled'}")
        st.markdown(f"- Slack: {'✅ Enabled' if enable_slack else '❌ Disabled'}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    show_alert_system()
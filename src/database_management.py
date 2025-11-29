"""
Machine Management Module
Complete CRUD operations + Advanced Dashboard for machines table

Developer: Eng. Mahmoud Khalid Alkodousy
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# SUPABASE CONNECTION
# ============================================

@st.cache_resource
def get_supabase_client():
    try:        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        
        if not url or not key:
            st.error("❌ Missing Supabase credentials in .env file")
            return None
        
        from supabase import create_client
        client = create_client(url, key)
        client.table("machines").select("id").limit(1).execute()
        return client
        
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase: {e}")
        return None


# ============================================
# DATABASE OPERATIONS
# ============================================

def fetch_all_machines(supabase) -> Optional[pd.DataFrame]:
    try:
        response = supabase.table("machines").select("*").order("id").execute()
        if not response.data:
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        date_cols = ['purchase_date', 'warranty_until', 'last_maintenance', 
                     'next_maintenance', 'last_oil_date', 'created_at', 'updated_at']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"❌ Error fetching machines: {e}")
        return None


def add_machine(supabase, machine_data: Dict[str, Any]) -> bool:
    try:
        clean_data = {k: v for k, v in machine_data.items() if v is not None and v != ''}
        response = supabase.table("machines").insert(clean_data).execute()
        if response.data:
            st.success("✅ Machine added successfully!")
            return True
        return False
    except Exception as e:
        st.error(f"❌ Error adding machine: {e}")
        return False


def update_machine(supabase, machine_id: int, updated_data: Dict[str, Any]) -> bool:
    try:
        clean_data = {k: v for k, v in updated_data.items() if v is not None and v != ''}
        response = supabase.table("machines").update(clean_data).eq("id", machine_id).execute()
        if response.data:
            st.success("✅ Machine updated successfully!")
            return True
        return False
    except Exception as e:
        st.error(f"❌ Error updating machine: {e}")
        return False


def delete_machine(supabase, machine_id: int) -> bool:
    try:
        supabase.table("machines").delete().eq("id", machine_id).execute()
        st.success("✅ Machine deleted successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error deleting machine: {e}")
        return False


# ============================================
# MACHINE FORM COMPONENT
# ============================================

def render_machine_form(mode: str = "add", existing_data: Optional[Dict] = None):
    st.subheader("🔧 " + ("Add New Machine" if mode == "add" else "Edit Machine"))
    with st.form(key=f"machine_form_{mode}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📋 Basic Information**")
            machine_id = st.number_input(
                "Machine ID*",
                min_value=1,
                value=existing_data.get('machine_id', 1) if existing_data else 1,
                help="Unique identifier for the machine"
            )
            serial_number = st.text_input(
                "Serial Number",
                value=existing_data.get('serial_number', '') if existing_data else ''
            )
            model = st.text_input(
                "Model",
                value=existing_data.get('model', '') if existing_data else ''
            )
            manufacturer = st.text_input(
                "Manufacturer",
                value=existing_data.get('manufacturer', '') if existing_data else ''
            )
            vendor = st.text_input(
                "Vendor",
                value=existing_data.get('vendor', '') if existing_data else ''
            )
        with col2:
            st.markdown("**📍 Location & Status**")
            location = st.text_input(
                "Location",
                value=existing_data.get('location', '') if existing_data else ''
            )
            status = st.selectbox(
                "Status",
                ["Active", "Inactive", "Maintenance", "Retired"],
                index=["Active", "Inactive", "Maintenance", "Retired"].index(
                    existing_data.get('status', 'Active')
                ) if existing_data and existing_data.get('status') in ["Active", "Inactive", "Maintenance", "Retired"] else 0
            )
            price = st.number_input(
                "Price",
                min_value=0.0,
                value=float(existing_data.get('price', 0.0)) if existing_data else 0.0,
                format="%.2f"
            )
            currency = st.selectbox(
                "Currency",
                ["USD", "EUR", "EGP", "GBP", "SAR", "AED"],
                index=["USD", "EUR", "EGP", "GBP", "SAR", "AED"].index(
                    existing_data.get('currency', 'USD')
                ) if existing_data and existing_data.get('currency') in ["USD", "EUR", "EGP", "GBP", "SAR", "AED"] else 0
            )
            purchase_date = st.date_input(
                "Purchase Date",
                value=pd.to_datetime(existing_data.get('purchase_date')).date() 
                      if existing_data and existing_data.get('purchase_date') 
                      else datetime.now().date()
            )
            warranty_until = st.date_input(
                "Warranty Until",
                value=pd.to_datetime(existing_data.get('warranty_until')).date() 
                      if existing_data and existing_data.get('warranty_until')
                      else (datetime.now() + timedelta(days=365)).date()
            )
        with col3:
            st.markdown("**🔧 Maintenance Info**")
            last_maintenance = st.date_input(
                "Last Maintenance",
                value=pd.to_datetime(existing_data.get('last_maintenance')).date() 
                      if existing_data and existing_data.get('last_maintenance')
                      else datetime.now().date()
            )
            maintenance_interval = st.number_input(
                "Maintenance Interval (days)",
                min_value=1,
                value=existing_data.get('maintenance_interval_days', 90) if existing_data else 90
            )
            next_maintenance = last_maintenance + timedelta(days=maintenance_interval)
            st.date_input(
                "Next Maintenance (Auto-calculated)",
                value=next_maintenance,
                disabled=True
            )
            last_oil_date = st.date_input(
                "Last Oil Change",
                value=pd.to_datetime(existing_data.get('last_oil_date')).date() 
                      if existing_data and existing_data.get('last_oil_date')
                      else datetime.now().date()
            )
            oil_interval = st.number_input(
                "Oil Change Interval (days)",
                min_value=1,
                value=existing_data.get('oil_interval_days', 30) if existing_data else 30
            )
            oil_type = st.text_input(
                "Oil Type",
                value=existing_data.get('oil_type', '') if existing_data else ''
            )
        notes = st.text_area(
            "📝 Notes",
            value=existing_data.get('notes', '') if existing_data else '',
            height=100
        )
        submit = st.form_submit_button(
            "➕ Add Machine" if mode == "add" else "💾 Update Machine",
            use_container_width=True,
            type="primary"
        )
        if submit:
            machine_data = {
                'machine_id': machine_id,
                'serial_number': serial_number or None,
                'model': model or None,
                'manufacturer': manufacturer or None,
                'vendor': vendor or None,
                'location': location or None,
                'status': status,
                'price': price if price > 0 else None,
                'currency': currency,
                'purchase_date': purchase_date.isoformat(),
                'warranty_until': warranty_until.isoformat(),
                'last_maintenance': last_maintenance.isoformat(),
                'next_maintenance': next_maintenance.isoformat(),
                'last_oil_date': last_oil_date.isoformat(),
                'oil_interval_days': oil_interval,
                'oil_type': oil_type or None,
                'maintenance_interval_days': maintenance_interval,
                'notes': notes or None
            }
            return machine_data
    return None


# ============================================
# DASHBOARD COMPONENTS
# ============================================

def render_kpi_cards(df: pd.DataFrame):
    col1, col2, col3, col4, col5 = st.columns(5)
    total_machines = len(df)
    active_machines = len(df[df['status'] == 'Active']) if 'status' in df.columns else 0
    maintenance_due = 0
    oil_change_due = 0
    total_value = 0
    today = pd.Timestamp.now()
    if not df.empty:
        if 'next_maintenance' in df.columns:
            maintenance_due = len(df[pd.to_datetime(df['next_maintenance'], errors='coerce') <= today + timedelta(days=7)])
        if 'last_oil_date' in df.columns and 'oil_interval_days' in df.columns:
            df_temp = df.dropna(subset=['last_oil_date', 'oil_interval_days']).copy()
            if not df_temp.empty:
                df_temp['next_oil_date'] = pd.to_datetime(df_temp['last_oil_date']) + pd.to_timedelta(df_temp['oil_interval_days'], unit='D')
                oil_change_due = len(df_temp[df_temp['next_oil_date'] <= today + timedelta(days=7)])
        if 'price' in df.columns:
            total_value = df['price'].fillna(0).sum()
    with col1:
        st.metric(label="🏭 Total Machines", value=total_machines, delta=None)
    with col2:
        st.metric(label="✅ Active Machines", value=active_machines, delta=f"{(active_machines/total_machines*100):.1f}%" if total_machines > 0 else "0%")
    with col3:
        st.metric(label="⚠️ Maintenance Due", value=maintenance_due, delta="Next 7 days", delta_color="inverse")
    with col4:
        st.metric(label="🛢️ Oil Change Due", value=oil_change_due, delta="Next 7 days", delta_color="inverse")
    with col5:
        st.metric(label="💰 Total Value", value=f"${total_value:,.0f}", delta=None)


def render_status_distribution(df: pd.DataFrame):
    if df.empty or 'status' not in df.columns:
        st.info("No status data available")
        return
    status_counts = df['status'].value_counts()
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="📊 Machine Status Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_manufacturer_breakdown(df: pd.DataFrame):
    if df.empty or 'manufacturer' not in df.columns:
        st.info("No manufacturer data available")
        return
    mfg_counts = df['manufacturer'].value_counts().head(10)
    fig = px.bar(
        x=mfg_counts.values,
        y=mfg_counts.index,
        orientation='h',
        title="🏭 Top 10 Manufacturers",
        labels={'x': 'Number of Machines', 'y': 'Manufacturer'},
        color=mfg_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=350,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_maintenance_timeline(df: pd.DataFrame):
    """Render maintenance timeline"""
    if df.empty or 'next_maintenance' not in df.columns:
        st.info("No maintenance data available")
        return

    df_maint = df[['machine_id', 'next_maintenance', 'status']].dropna()
    if df_maint.empty:
        st.info("No upcoming maintenance scheduled")
        return

    df_maint['next_maintenance'] = pd.to_datetime(df_maint['next_maintenance'], errors='coerce')
    df_maint = df_maint.dropna(subset=['next_maintenance']).copy()
    df_maint = df_maint.sort_values('next_maintenance').head(20)

    today = pd.Timestamp.now()
    df_maint['days_until'] = (df_maint['next_maintenance'] - today).dt.days
    df_maint['urgency'] = df_maint['days_until'].apply(
        lambda x: 'Overdue' if x < 0 else 'Critical' if x <= 7 else 'Soon' if x <= 30 else 'Scheduled'
    )

    color_map = {
        'Overdue': 'red',
        'Critical': 'orange',
        'Soon': 'yellow',
        'Scheduled': 'green'
    }

    fig = px.scatter(
        df_maint,
        x='next_maintenance',
        y='machine_id',
        color='urgency',
        color_discrete_map=color_map,
        title="📅 Maintenance Timeline (Next 20)",
        labels={'next_maintenance': 'Scheduled Date', 'machine_id': 'Machine ID'},
        size=[10] * len(df_maint),
        hover_data={'days_until': True, 'status': True}
    )

    # تأكيد أن محور x تاريخ
    fig.update_xaxes(type='date')

    # خط اليوم باستخدام add_shape (أكثر ثباتًا عبر نسخ Plotly)
    x_line = today.to_pydatetime()
    fig.add_shape(
        type="line",
        x0=x_line, x1=x_line, xref="x",
        y0=0, y1=1, yref="paper",
        line=dict(color="red", dash="dash")
    )
    # لابل "Today" في أعلى الشكل
    fig.add_annotation(
        x=x_line, xref="x",
        y=1, yref="paper",
        text="Today",
        showarrow=False,
        yanchor="bottom"
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)



def render_price_analysis(df: pd.DataFrame):
    if df.empty or 'price' not in df.columns:
        st.info("No price data available")
        return
    df_price = df[df['price'] > 0].copy()
    if df_price.empty:
        st.info("No price data available")
        return
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            df_price,
            x='price',
            nbins=20,
            title="💵 Price Distribution",
            labels={'price': 'Price (USD)'},
            color_discrete_sequence=['#636EFA']
        )
        fig_hist.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        if 'manufacturer' in df_price.columns:
            mfg_price = df_price.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(10)
            fig_mfg = px.bar(
                x=mfg_price.values,
                y=mfg_price.index,
                orientation='h',
                title="💰 Average Price by Manufacturer",
                labels={'x': 'Avg Price (USD)', 'y': 'Manufacturer'},
                color=mfg_price.values,
                color_continuous_scale='Greens'
            )
            fig_mfg.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_mfg, use_container_width=True)


def render_location_map(df: pd.DataFrame):
    if df.empty or 'location' not in df.columns:
        st.info("No location data available")
        return
    location_counts = df['location'].value_counts()
    fig = px.treemap(
        names=location_counts.index,
        parents=[""] * len(location_counts),
        values=location_counts.values,
        title="📍 Machines by Location",
        color=location_counts.values,
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=400)
    fig.update_traces(textinfo="label+value+percent parent")
    st.plotly_chart(fig, use_container_width=True)


def render_age_analysis(df: pd.DataFrame):
    if df.empty or 'purchase_date' not in df.columns:
        st.info("No purchase date data available")
        return
    df_age = df.dropna(subset=['purchase_date']).copy()
    if df_age.empty:
        st.info("No age data available")
        return
    df_age['purchase_date'] = pd.to_datetime(df_age['purchase_date'])
    today = pd.Timestamp.now()
    df_age['age_years'] = (today - df_age['purchase_date']).dt.days / 365.25
    df_age['age_category'] = pd.cut(
        df_age['age_years'],
        bins=[0, 1, 3, 5, 10, 100],
        labels=['< 1 year', '1-3 years', '3-5 years', '5-10 years', '> 10 years']
    )
    age_counts = df_age['age_category'].value_counts().sort_index()
    fig = px.bar(
        x=age_counts.index,
        y=age_counts.values,
        title="📆 Machine Age Distribution",
        labels={'x': 'Age Category', 'y': 'Number of Machines'},
        color=age_counts.values,
        color_continuous_scale='Purples'
    )
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ============================================
# MAIN RENDER FUNCTION
# ============================================

def render_machine_management_tab():
    supabase = get_supabase_client()
    if not supabase:
        st.error("❌ Cannot connect to database. Please check your configuration.")
        st.stop()

    MENU_OPTIONS = ["📊 Dashboard", "➕ Add Machine", "✏️ Edit/Delete", "📋 View All"]

    if "machine_menu" not in st.session_state:
        st.session_state.machine_menu = MENU_OPTIONS[0]

    # ========= Title + perfectly aligned compact dropdown =========
    header_left, header_right = st.columns([6, 1], gap="small")

    # CSS: محاذاة الأعمدة عمودياً + تصغير عرض السليكت + تقليل المسافة أسفل العنوان والخط
    st.markdown(
        """
        <style>
        /* خلي صف الأعمدة اللي فيه العنوان والسليكت مصفوف عموديًا */
        div[data-testid="stHorizontalBlock"]:has(> div > div h1) > div[data-testid="column"]{
            display:flex; align-items:center;
        }
        /* شكل العنوان ومفيش مسافة تحت */
        .mm-title{ margin:0 !important; line-height:1.08; white-space:nowrap; }

        /* صغِّر عرض السليكت */
        div[data-testid="stHorizontalBlock"]:has(> div > div h1)
          div[data-testid="stSelectbox"] > div > div{
            width: 210px !important; min-width: 210px !important;
        }
        /* اضبط محاذاة السليكت رأسياً مع العنوان (ارفع/انزل بالرقم) */
        div[data-testid="stHorizontalBlock"]:has(> div > div h1)
          div[data-testid="stSelectbox"]{
            margin-top: 30px !important;   /* ← عدّلها لو عايز أعلى/أقل */
            margin-bottom: 0 !important;
        }

        /* قلل المسافة قبل وبعد الخط */
        div[data-testid="stMarkdownContainer"] hr{
            margin-top: 6px !important;
            margin-bottom: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with header_left:
        st.markdown('<h1 class="mm-title">🏭 Machine Management System</h1>', unsafe_allow_html=True)

    def _persist_menu():
        st.session_state.machine_menu = st.session_state._machine_menu_widget

    with header_right:
        st.selectbox(
            "Navigate",
            options=MENU_OPTIONS,
            index=MENU_OPTIONS.index(st.session_state.machine_menu),
            key="_machine_menu_widget",
            label_visibility="collapsed",
            on_change=_persist_menu,
        )

    sub = st.session_state.machine_menu
    st.markdown("---")

    # =================== Fetch once ===================
    df_machines = fetch_all_machines(supabase)
    if df_machines is None:
        st.error("❌ Failed to load machines data")
        return

    # ---------------- DASHBOARD ----------------
    if sub.startswith("📊"):
        st.markdown("### 📊 Machine Fleet Dashboard")
        if df_machines.empty:
            st.info("👋 No machines in database yet.")
        else:
            render_kpi_cards(df_machines)
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1: render_status_distribution(df_machines)
            with c2: render_manufacturer_breakdown(df_machines)
            st.markdown("---")
            render_maintenance_timeline(df_machines)
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1: render_location_map(df_machines)
            with c2: render_age_analysis(df_machines)
            st.markdown("---")
            render_price_analysis(df_machines)

    # ---------------- ADD MACHINE ----------------
    elif sub.startswith("➕"):
        st.markdown("### ➕ Add Machine")
        machine_data = render_machine_form(mode="add")
        if machine_data:
            if add_machine(supabase, machine_data):
                st.balloons()
                st.rerun()

    # ---------------- EDIT/DELETE ----------------
    elif sub.startswith("✏️"):
        st.markdown("### ✏️ Edit or Delete Machine")
        if df_machines.empty:
            st.info("No machines to edit")
        else:
            machine_options = df_machines.apply(
                lambda row: f"ID: {row['machine_id']} - {row.get('model','Unknown')} ({row.get('serial_number','N/A')})",
                axis=1
            ).tolist()

            selected_machine = st.selectbox(
                "Select Machine",
                options=range(len(df_machines)),
                format_func=lambda x: machine_options[x]
            )
            selected_row = df_machines.iloc[selected_machine]
            st.markdown("---")

            if "confirm_delete" not in st.session_state:
                st.session_state["confirm_delete"] = False

            c1, c2 = st.columns([3, 1])
            with c1:
                updated_data = render_machine_form(mode="edit", existing_data=selected_row.to_dict())
                if updated_data:
                    if update_machine(supabase, selected_row['id'], updated_data):
                        st.rerun()
            with c2:
                st.markdown("### ⚠️ Danger Zone")
                st.warning("Deleting cannot be undone!")
                if st.button("🗑️ Delete Machine", type="secondary", use_container_width=True):
                    st.session_state["confirm_delete"] = True
                if st.session_state.get("confirm_delete", False):
                    st.error("Are you absolutely sure?")
                    cA, cB = st.columns(2)
                    with cA:
                        if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                            delete_machine(supabase, selected_row["id"])
                            st.session_state["confirm_delete"] = False
                            st.rerun()
                    with cB:
                        if st.button("❌ Cancel", use_container_width=True):
                            st.session_state["confirm_delete"] = False
                            st.rerun()

    # ---------------- VIEW ALL ----------------
    elif sub.startswith("📋"):
        st.markdown("### 📋 All Machines")
        if df_machines.empty:
            st.info("No machines yet.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                status_filter = st.multiselect("Status", df_machines['status'].unique())
            with c2:
                manufacturer_filter = st.multiselect("Manufacturer", df_machines['manufacturer'].dropna().unique())
            with c3:
                location_filter = st.multiselect("Location", df_machines['location'].dropna().unique())

            df_filtered = df_machines.copy()
            if status_filter:
                df_filtered = df_filtered[df_filtered['status'].isin(status_filter)]
            if manufacturer_filter:
                df_filtered = df_filtered[df_filtered['manufacturer'].isin(manufacturer_filter)]
            if location_filter:
                df_filtered = df_filtered[df_filtered['location'].isin(location_filter)]

            st.info(f"Showing {len(df_filtered)} of {len(df_machines)} machines")

            st.dataframe(df_filtered, use_container_width=True, height=500)

            csv = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"machines_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Machine Management",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_machine_management_tab()

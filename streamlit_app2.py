# ==============================================
# STREAMLIT APP v2 – TRAFFIC SIGNAL CONTROL SYSTEM
# Features:
# - Realistic road network diagram (J1 west, J2 center, J3 east, J4 south)
# - Sequential green allocation (only one junction green at a time)
# - Proportional green times based on predicted weighted load
# - Custom phase order (user can reorder junctions)
# - Visual countdown and progress bar
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Adaptive Traffic Control v2", layout="wide")
st.title("🚦 Intelligent Traffic Control System – Sequential Signals")
st.markdown("### Green times are allocated one junction at a time based on predicted congestion")

# ------------------------------
# 1. Load models (cached)
# ------------------------------
@st.cache_resource
def load_models():
    le_day = joblib.load("le_day.pkl")
    le_junction = joblib.load("le_junction.pkl")
    le_weather = joblib.load("le_weather.pkl")
    reg_car = joblib.load("reg_car.pkl")
    reg_bike = joblib.load("reg_bike.pkl")
    reg_bus = joblib.load("reg_bus.pkl")
    reg_truck = joblib.load("reg_truck.pkl")
    max_load = joblib.load("max_load.pkl")
    return le_day, le_junction, le_weather, reg_car, reg_bike, reg_bus, reg_truck, max_load

# ------------------------------
# 2. Helper functions
# ------------------------------
def weighted_load(car, bike, bus, truck):
    """Passenger Car Equivalents (PCE)"""
    return car * 1.0 + bike * 0.5 + bus * 3.0 + truck * 2.5

def green_from_load(load, max_load):
    """Map weighted load to green seconds (15‑90 sec)"""
    return int(np.clip(15 + (load / max_load) * 75, 15, 90))

def create_cycle_plan(preds, max_load, order=None):
    """
    Create list of (junction, green_seconds) in the order they will be served.
    If order is given, use that sequence; otherwise sort by weighted load descending.
    """
    if order is None:
        items = sorted(preds.items(), key=lambda x: x[1]['weighted_load'], reverse=True)
    else:
        items = [(j, preds[j]) for j in order if j in preds]
    total_load = sum(info['weighted_load'] for _, info in items)
    plan = []
    for junction, info in items:
        load = info['weighted_load']
        sec = max(15, int((load / total_load) * 90)) if total_load > 0 else 30
        plan.append((junction, sec))
    return plan

def draw_road_network(active_junction):
    """
    Draws a schematic map of junctions on a road network.
    Layout:
        J1 (west) ---- J2 (center) ---- J3 (east)
                        |
                        J4 (south)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    # Coordinates
    positions = {
        'J1': (-2, 0),
        'J2': (0, 0),
        'J3': (2, 0),
        'J4': (0, -1.5)
    }
    # Draw roads (lines)
    # East-West main road
    ax.plot([-2.5, 2.5], [0, 0], 'gray', linewidth=5, alpha=0.4, zorder=1)
    # South road from J2
    ax.plot([0, 0], [0, -2], 'gray', linewidth=5, alpha=0.4, zorder=1)
    # Junctions as circles
    for name, (x, y) in positions.items():
        color = 'lime' if name == active_junction else 'red'
        ax.scatter(x, y, s=500, c=color, edgecolors='black', zorder=3)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')
    # Road labels
    ax.text(-1, 0.3, "East‑West Road", ha='center', fontsize=9, color='gray')
    ax.text(0.2, -0.8, "South Road", ha='center', fontsize=9, color='gray')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

# ------------------------------
# 3. Load models once
# ------------------------------
try:
    le_day, le_junction, le_weather, rc, rb, rbu, rt, max_load = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please run app.py first to train and save models.")
    st.stop()

# ------------------------------
# 4. Sidebar inputs
# ------------------------------
with st.sidebar:
    st.header("📋 Input Parameters")
    hour = st.slider("Hour (0‑23)", 0, 23, 12)
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    holiday = st.checkbox("Is Holiday?")
    weather = st.selectbox("Weather", ["Sunny","Rainy","Cloudy"])
    
    st.markdown("---")
    st.subheader("⚙️ Phase Order")
    order_option = st.radio("Choose phase order", 
                            ["By congestion (busiest first)", "Custom fixed order"])
    if order_option == "Custom fixed order":
        all_junctions = list(le_junction.classes_)
        custom_order = st.multiselect("Drag to reorder (or click to select in order)", 
                                      all_junctions, default=all_junctions)
    else:
        custom_order = None
    
    predict_btn = st.button("🚦 Predict & Run Cycle", type="primary")

# ------------------------------
# 5. Main logic when button clicked
# ------------------------------
if predict_btn:
    # Encode day and weather (with fallback)
    try:
        day_enc = le_day.transform([day])[0]
    except ValueError:
        st.warning(f"Day '{day}' not in training data. Using default encoding (0).")
        day_enc = 0
    try:
        weather_enc = le_weather.transform([weather])[0]
    except ValueError:
        st.warning(f"Weather '{weather}' not in training data. Using default encoding (0).")
        weather_enc = 0
    
    # Predict for each junction
    preds = {}
    for junction in le_junction.classes_:
        j_enc = le_junction.transform([junction])[0]
        input_df = pd.DataFrame([[hour, day_enc, int(holiday), j_enc, weather_enc]],
                                 columns=['Hour','Day','IsHoliday','Junction','Weather'])
        car = max(0, int(rc.predict(input_df)[0]))
        bike = max(0, int(rb.predict(input_df)[0]))
        bus = max(0, int(rbu.predict(input_df)[0]))
        truck = max(0, int(rt.predict(input_df)[0]))
        load = weighted_load(car, bike, bus, truck)
        preds[junction] = {
            'car': car, 'bike': bike, 'bus': bus, 'truck': truck,
            'weighted_load': load,
            'individual_green': green_from_load(load, max_load)
        }
    
    # Create cycle plan
    plan = create_cycle_plan(preds, max_load, order=custom_order)
    
    # Display prediction table
    st.subheader("📊 Predicted Traffic per Junction")
    df_display = pd.DataFrame([{
        'Junction': j,
        'Cars': info['car'],
        'Bikes': info['bike'],
        'Buses': info['bus'],
        'Trucks': info['truck'],
        'Weighted Load': round(info['weighted_load'], 1),
        'Individual Green (s)': info['individual_green']
    } for j, info in preds.items()])
    st.dataframe(df_display, use_container_width=True)
    
    # Show cycle plan
    st.subheader("🔄 Sequential Signal Cycle Plan")
    st.write("Junctions are served one after another (only one green at a time).")
    plan_df = pd.DataFrame(plan, columns=["Junction", "Green Duration (seconds)"])
    st.dataframe(plan_df, use_container_width=True)
    
    # Run simulation
    if st.button("▶️ Start Signal Simulation"):
        map_placeholder = st.empty()
        timer_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        total_steps = sum(sec for _, sec in plan)
        elapsed = 0
        
        for junction, green_sec in plan:
            for remaining in range(green_sec, 0, -1):
                # Draw map with active junction highlighted
                fig = draw_road_network(junction)
                map_placeholder.pyplot(fig)
                timer_placeholder.markdown(f"## 🟢 **{junction}** – GREEN for **{remaining}** seconds")
                elapsed += 1
                progress_bar.progress(elapsed / total_steps)
                time.sleep(1)
        
        timer_placeholder.success("✅ Cycle completed")
        progress_bar.empty()
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Traffic Control System", layout="wide")
st.title("🚦 Intelligent Traffic Control System")

# Load models (cached for performance)
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

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    hour = st.slider("Hour (0-23)", 0, 23, 12)
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    holiday = st.checkbox("Is Holiday?")
    weather = st.selectbox("Weather", ["Sunny","Rainy","Cloudy"])
    predict_btn = st.button("Predict Green Times")

# Main area
if predict_btn:
    with st.spinner("Computing..."):
        le_day, le_junction, le_weather, rc, rb, rbu, rt, max_load = load_models()
        
        day_enc = le_day.transform([day])[0]
        weather_enc = le_weather.transform([weather])[0]
        
        results = []
        for junction in le_junction.classes_:
            junc_enc = le_junction.transform([junction])[0]
            inp = pd.DataFrame([[hour, day_enc, int(holiday), junc_enc, weather_enc]],
                               columns=['Hour','Day','IsHoliday','Junction','Weather'])
            car = max(0, int(rc.predict(inp)[0]))
            bike = max(0, int(rb.predict(inp)[0]))
            bus = max(0, int(rbu.predict(inp)[0]))
            truck = max(0, int(rt.predict(inp)[0]))
            load = car*1.0 + bike*0.5 + bus*3.0 + truck*2.5
            green = int(np.clip(15 + (load / max_load) * 75, 15, 90))
            results.append({
                "Junction": junction,
                "Cars": car, "Bikes": bike, "Buses": bus, "Trucks": truck,
                "Weighted Load": round(load,1),
                "Green Time (s)": green
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.success("✅ Prediction complete. Green times are for the main movement at each junction.")
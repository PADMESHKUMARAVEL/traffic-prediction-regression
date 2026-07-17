# ==============================================
# FINAL TRAFFIC CONTROL + VEHICLE SIMULATION
# ==============================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🚦 Intelligent Traffic Control System")

# ------------------------------
# LOAD MODELS
# ------------------------------
@st.cache_resource
def load_models():
    return (
        joblib.load("reg_car.pkl"),
        joblib.load("reg_bike.pkl"),
        joblib.load("reg_bus.pkl"),
        joblib.load("reg_truck.pkl"),
        joblib.load("le_day.pkl"),
        joblib.load("le_junction.pkl"),
        joblib.load("le_weather.pkl"),
        joblib.load("pca.pkl"),
    )

rc, rb, rbu, rt, le_day, le_junction, le_weather, pca = load_models()

# ------------------------------
# SIDEBAR INPUT
# ------------------------------
hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.selectbox("Day", le_day.classes_)
weather = st.sidebar.selectbox("Weather", le_weather.classes_)
holiday = st.sidebar.checkbox("Holiday")

run = st.sidebar.button("🚀 Run Simulation")

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def weighted_load(c, b, bu, t):
    return c*1 + b*0.5 + bu*3 + t*2.5

def green_time(load):
    return int(np.clip(20 + load/10, 20, 90))

# ------------------------------
# VEHICLE CLASS
# ------------------------------
class Vehicle:
    def __init__(self, x, y, direction, road):
        self.x = x
        self.y = y
        self.direction = direction
        self.road = road

    def move(self):
        if self.direction == "right":
            self.x += 0.15
        elif self.direction == "left":
            self.x -= 0.15
        elif self.direction == "up":
            self.y += 0.15
        elif self.direction == "down":
            self.y -= 0.15

# ------------------------------
# STOP LOGIC
# ------------------------------
def should_stop(v, signal):

    # J1 (center)
    if v.road == "R1":
        if abs(v.y) < 0.4 and not signal["J1"]:
            return True

    # J2 (left)
    if v.road == "R2":
        if abs(v.y-2) < 0.3 and abs(v.x + 1.5) < 0.4 and not signal["J2"]:
            return True

    # J3 (right)
    if v.road == "R3":
        if abs(v.y+2) < 0.3 and abs(v.x - 1.5) < 0.4 and not signal["J3"]:
            return True

    return False

# ------------------------------
# GREEN CONTROL (STRICT)
# ------------------------------
def can_move(v, signal):

    if v.road == "R1":
        return signal["J1"]

    if v.road == "R2":
        return signal["J2"]

    if v.road == "R3":
        return signal["J3"]

    return False

# ------------------------------
# DRAW FUNCTION
# ------------------------------
def draw(vehicles, signal_state):

    fig, ax = plt.subplots(figsize=(6,6))

    # Roads
    ax.plot([0,0], [-5,5], linewidth=6, color="blue")     # R1
    ax.plot([-5,5], [2,2], linewidth=6, color="orange")   # R2
    ax.plot([-5,5], [-2,-2], linewidth=6, color="green")  # R3

    # Junction positions
    signals = {
        "J1": (0,0),
        "J2": (-1.5,2),
        "J3": (1.5,-2)
    }

    for j,(x,y) in signals.items():
        color = "green" if signal_state[j] else "red"
        ax.scatter(x,y,s=400,c=color,edgecolors='black', zorder=3)
        ax.text(x,y,j,ha='center',va='center',color='white',fontweight='bold')

    # Vehicles
    for v in vehicles:
        ax.scatter(v.x, v.y, c="black", s=15)

    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.axis("off")

    return fig

# ------------------------------
# STATUS TABLE
# ------------------------------
def show_status(active, remaining, results):

    data = []

    for j in ["J1","J2","J3"]:
        if j in active:
            state = "🟢 GREEN"
            rem = int(remaining)   # always int
        else:
            state = "🔴 RED"
            rem = None             # FIX HERE

        data.append({
            "Junction": j,
            "Signal": state,
            "Remaining Time": rem,
            "Predicted Green Time": int(results[j]["green"])
        })

    return pd.DataFrame(data)
# ------------------------------
# MAIN EXECUTION
# ------------------------------
if run:

    d = le_day.transform([day])[0]
    w = le_weather.transform([weather])[0]

    results = {}

    for j in ["J1","J2","J3"]:
        j_enc = le_junction.transform([j])[0]

        X = pd.DataFrame([[hour,d,int(holiday),j_enc,w]],
                         columns=['Hour','Day','IsHoliday','Junction','Weather'])

        X = pca.transform(X)

        car = int(rc.predict(X)[0])
        bike = int(rb.predict(X)[0])
        bus = int(rbu.predict(X)[0])
        truck = int(rt.predict(X)[0])

        load = weighted_load(car,bike,bus,truck)
        g = green_time(load)

        results[j] = {
            "car":car,"bike":bike,"bus":bus,"truck":truck,
            "load":load,"green":g
        }

    # ------------------------------
    # DISPLAY PREDICTIONS
    # ------------------------------
    st.subheader("📊 Predicted Traffic")

    st.dataframe(pd.DataFrame([
        {
            "Junction":j,
            "Cars":v["car"],
            "Bikes":v["bike"],
            "Buses":v["bus"],
            "Trucks":v["truck"],
            "Weighted Load":round(v["load"],1),
            "Green Time":v["green"]
        }
        for j,v in results.items()
    ]), use_container_width=True)

    map_placeholder = st.empty()
    status_placeholder = st.empty()

    vehicles = []

    # ------------------------------
    # PHASE 1 → J2 & J3 GREEN
    # ------------------------------
    phase1_time = max(results["J2"]["green"], results["J3"]["green"])

    for t in range(phase1_time):

        remaining = phase1_time - t
        signal = {"J1":False, "J2":True, "J3":True}

        status_placeholder.dataframe(
            show_status(["J2","J3"], remaining, results),
            use_container_width=True
        )

        # Spawn vehicles
        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(0,-5,"up","R1"))
        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(0,5,"down","R1"))

        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(-5,2,"right","R2"))
        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(5,2,"left","R2"))

        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(-5,-2,"right","R3"))
        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(5,-2,"left","R3"))

        # Move vehicles
        new_vehicles = []
        for v in vehicles:

            if should_stop(v, signal):
                new_vehicles.append(v)
                continue

            if can_move(v, signal):
                v.move()

            new_vehicles.append(v)

        vehicles = new_vehicles

        map_placeholder.pyplot(draw(vehicles, signal))
        time.sleep(0.05)

    # ------------------------------
    # PHASE 2 → J1 GREEN
    # ------------------------------
    phase2_time = results["J1"]["green"]

    for t in range(phase2_time):

        remaining = phase2_time - t
        signal = {"J1":True, "J2":False, "J3":False}

        status_placeholder.dataframe(
            show_status(["J1"], remaining, results),
            use_container_width=True
        )

        # Spawn vehicles
        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(0,-5,"up","R1"))
        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(0,5,"down","R1"))

        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(-5,2,"right","R2"))
        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(5,2,"left","R2"))

        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(-5,-2,"right","R3"))
        if np.random.rand() < 0.3:
            vehicles.append(Vehicle(5,-2,"left","R3"))

        # Move vehicles
        new_vehicles = []
        for v in vehicles:

            if should_stop(v, signal):
                new_vehicles.append(v)
                continue

            if can_move(v, signal):
                v.move()

            new_vehicles.append(v)

        vehicles = new_vehicles

        map_placeholder.pyplot(draw(vehicles, signal))
        time.sleep(0.05)

    st.success("✅ Simulation Completed")
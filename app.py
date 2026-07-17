# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Allow requests from any frontend

# Load models and encoders
le_day = joblib.load("le_day.pkl")
le_junction = joblib.load("le_junction.pkl")
le_weather = joblib.load("le_weather.pkl")

reg_car = joblib.load("reg_car.pkl")
reg_bike = joblib.load("reg_bike.pkl")
reg_bus = joblib.load("reg_bus.pkl")
reg_truck = joblib.load("reg_truck.pkl")

max_load = joblib.load("max_load.pkl")  # 95th percentile weighted load

def weighted_load(car, bike, bus, truck):
    """Passenger Car Equivalent (PCE) weighting"""
    return car * 1.0 + bike * 0.5 + bus * 3.0 + truck * 2.5

def green_time_from_load(load, max_load_val=max_load):
    """Map weighted load to green signal time (15-90 seconds)"""
    green = 15 + (load / max_load_val) * 75
    return int(np.clip(green, 15, 90))

def safe_transform(encoder, value, default=0):
    """Handle unseen categories gracefully"""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        print(f"Warning: '{value}' not in training, using default {default}")
        return default

@app.route('/predict', methods=['POST'])
def predict():
    """Expects JSON: {hour, day, holiday, weather}"""
    data = request.get_json()
    
    # Validate required fields
    required = ['hour', 'day', 'holiday', 'weather']
    if not all(k in data for k in required):
        return jsonify({'error': f'Missing fields. Required: {required}'}), 400
    
    hour = int(data['hour'])
    day = data['day']
    holiday = int(data['holiday'])
    weather = data['weather']
    
    # Encode categoricals
    day_enc = safe_transform(le_day, day)
    weather_enc = safe_transform(le_weather, weather)
    
    results = {}
    for junction_name in le_junction.classes_:
        junction_enc = le_junction.transform([junction_name])[0]
        # Create input DataFrame
        input_df = pd.DataFrame([{
            'Hour': hour,
            'Day': day_enc,
            'IsHoliday': holiday,
            'Junction': junction_enc,
            'Weather': weather_enc
        }])
        
        # Predict counts (ensure non‑negative)
        car = max(0, int(reg_car.predict(input_df)[0]))
        bike = max(0, int(reg_bike.predict(input_df)[0]))
        bus = max(0, int(reg_bus.predict(input_df)[0]))
        truck = max(0, int(reg_truck.predict(input_df)[0]))
        
        load = weighted_load(car, bike, bus, truck)
        green = green_time_from_load(load)
        
        results[junction_name] = {
            'car': car,
            'bike': bike,
            'bus': bus,
            'truck': truck,
            'weighted_load': round(load, 1),
            'green_time': green
        }
    
    return jsonify(results)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
import streamlit as st
import pandas as pd
import pickle

# 1. Load the model and scaler from your saved pickle files
with open('delivery_time_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('delivery_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set up page configurations
st.set_page_config(page_title="Delivery ETA Predictor", page_icon="🛵", layout="centered")

st.title("🛵 Delivery Time Prediction System")
st.write("Enter the logistical parameters below to compute an accurate Estimated Time of Arrival (ETA).")
st.markdown("---")

# 2. Create the User Interface Inputs
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Delivery Distance (km)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
    traffic_level_text = st.selectbox("Current Traffic Level", ["Low", "Medium", "High"])
    vehicle = st.selectbox("Driver Vehicle Type", ["Bicycle", "Motorbike", "Car"])

with col2:
    city = st.selectbox("Select City", ["Alexandria", "Assiut", "Cairo", "Giza", "Mansoura", "Tanta", "Zagazig"])
    hour = st.slider("Order Hour (0-23)", min_value=0, max_value=23, value=12)
    day_of_week = st.slider("Day of the Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)

st.markdown("---")

# 3. Handle Encodings in the Background
# Map traffic text to its corresponding ordinal number
traffic_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
traffic_val = traffic_mapping[traffic_level_text]

# Handle Vehicle One-Hot Encoding
is_car = 1.0 if vehicle == "Car" else 0.0
is_motorbike = 1.0 if vehicle == "Motorbike" else 0.0

# Handle City One-Hot Encoding (drop_first=True drops the baseline, usually Alexandria)
city_assiut = 1.0 if city == "Assiut" else 0.0
city_cairo = 1.0 if city == "Cairo" else 0.0
city_giza = 1.0 if city == "Giza" else 0.0
city_mansoura = 1.0 if city == "Mansoura" else 0.0
city_tanta = 1.0 if city == "Tanta" else 0.0
city_zagazig = 1.0 if city == "Zagazig" else 0.0

# 4. Prediction Trigger Button
if st.button("Calculate Delivery ETA", use_container_width=True):
    
    # Put all inputs into a dictionary (so initial order doesn't matter)
    input_dict = {
        'Delivery_Distance_km': [distance],
        'Order_Hour': [hour],
        'Order_DayOfWeek': [day_of_week],
        'Traffic_Level': [traffic_val],
        'City_Assiut': [city_assiut],
        'City_Cairo': [city_cairo],
        'City_Giza': [city_giza],
        'City_Mansoura': [city_mansoura],
        'City_Tanta': [city_tanta],
        'City_Zagazig': [city_zagazig],
        'Driver_Vehicle_Car': [is_car],
        'Driver_Vehicle_Motorbike': [is_motorbike]
    }
    
    # Create the dataframe
    input_data = pd.DataFrame(input_dict)
    
    # AUTOMATICALLY snap the columns into the exact order the scaler memorized!
    try:
        expected_columns = scaler.feature_names_in_
        input_data = input_data[expected_columns]
    except KeyError as e:
        st.error(f"Mismatch in expected columns. The model is looking for a column named: {e}")
        st.stop()
    
    # Scale the input data using the loaded scaler configuration
    scaled_input = scaler.transform(input_data)
    
    # Generate prediction using the loaded regression model
    prediction = model.predict(scaled_input)[0]
    
    # Ensure ETA doesn't display unrealistic values below a baseline
    final_eta = max(15, int(prediction))
    
    # Display Result
    st.success(f"### ⏱️ Estimated Delivery Time: **{final_eta} minutes**")
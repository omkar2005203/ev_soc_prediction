import streamlit as st
import joblib
import numpy as np
import os

# Streamlit UI
st.title("SoC Prediction for EV")

# File Uploaders for Model and Scaler
uploaded_model = st.file_uploader("Upload Model (.pkl)", type=["pkl"])
uploaded_scaler = st.file_uploader("Upload Scaler (.pkl)", type=["pkl"])

# Model selection dropdown
model_options = {
    "Linear Regression": ["acceleration", "speed", "speedFactor", "energyConsumed", 
                          "energyRegen", "slope", "distance", "remainingRange", "energyRate"],
    
    "Random Forest/Neural Network": ["acceleration", "speed", "speedFactor", "energyConsumed", 
                      "energyRegen", "slope", "distance", "remainingRange", 
                      "energyRate", "batteryTemp", "motorTemp", "ambientTemp", 
                      "windSpeed", "trafficFactor", "chargingEfficiency", "regenEfficiency"]
}

selected_model = st.selectbox("Select Prediction Model", list(model_options.keys()))

# Check if both files are uploaded
if uploaded_model and uploaded_scaler:
    # Save uploaded files temporarily
    model_path = "temp/temp_model.pkl"
    scaler_path = "temp/temp_scaler.pkl"
    
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    with open(scaler_path, "wb") as f:
        f.write(uploaded_scaler.getbuffer())
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    st.success("Model and Scaler loaded successfully!")

    # Define input fields dynamically based on selected model
    inputs = {}
    for feature in model_options[selected_model]:
        inputs[feature] = st.number_input(f"{feature}", value=0.0)
    
    # Predict Button
    if st.button("Predict SoC"):
        new_data = np.array([[inputs[feature] for feature in model_options[selected_model]]])
        
        # Scale input data
        new_data_scaled = scaler.transform(new_data)
        
        # Predict SoC
        predicted_soc = model.predict(new_data_scaled)
        predicted_soc = np.clip(predicted_soc, 0, 100)  # Ensure SoC is in range
        
        st.success(f"Predicted SoC: {predicted_soc[0]:.2f}%")

    # Clean up temporary files
    os.remove(model_path)
    os.remove(scaler_path)

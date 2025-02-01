import streamlit as st
import numpy as np
import h5py
import pickle
import os

# Load the trained model
model_path = "advertising.h5"

st.title("Advertising Sales Predictor 📈")

# Check if model exists
if not os.path.exists(model_path):
    st.error("⚠️ Model file 'advertising.h5' not found! Please train and save the model.")
else:
    # Load the model from .h5 file using h5py and pickle
    with h5py.File(model_path, "r") as f:
        model_bytes = f["model"][()]  # Retrieve model bytes
        model = pickle.loads(model_bytes.tobytes())  # Deserialize model

    # User input for advertisement budgets
    TV = st.number_input("Enter TV advertising budget ($):", min_value=0.0, step=0.1)
    Radio = st.number_input("Enter Radio advertising budget ($):", min_value=0.0, step=0.1)
    Newspaper = st.number_input("Enter Newspaper advertising budget ($):", min_value=0.0, step=0.1)

    if st.button("Predict Sales 🚀"):
        input_data = np.array([[TV, Radio, Newspaper]])
        prediction = model.predict(input_data)
        st.success(f"📊 Predicted Sales: ${prediction[0]:.2f}")

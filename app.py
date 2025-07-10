import streamlit as st
import model  # This is your model.py
import pandas as pd

st.title("Online Learning Prediction with River")

# Example input fields – customize based on your features
age = st.number_input("Age", min_value=0, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert input to dictionary format for river
input_data = {"age": age, "gender": gender}

# Predict
if st.button("Predict"):
    prediction = model.predict(model.model, input_data)
    st.write(f"Predicted class: {prediction}")

# Optionally allow feedback to train the model
feedback = st.selectbox("True label (for learning)", ["None", "0", "1"])
if st.button("Train on this input"):
    if feedback in ["0", "1"]:
        model.learn(model.model, input_data, int(feedback))
        st.success("Model updated with new data.")

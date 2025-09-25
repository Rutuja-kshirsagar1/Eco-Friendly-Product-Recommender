import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model and encoders
model = joblib.load("D:\internship works\codsoft\sales_adv_prdict.pkl")


st.title(" Future sales Prediction App")

# User input form
#with st.form("movie_form"):
TV= st.number_input("Enter the amount spent for advertising on TV", min_value=0, max_value=10000, value=100)
Radio = st.number_input("Enter the amount spent for advertising on Radio",min_value=0, max_value=10000, value=50)
Newspaper=st.number_input("Enter the amount spent for advertising on Newspaper",min_value=0, max_value=10000, value=25)

if st.button("Predict Sales"):
    # Prepare input data
    new_data = np.array([[TV , Radio, Newspaper]])

    # Make prediction
    prediction = model.predict(new_data)[0]  # assuming model returns a single value

    # Display results
    st.success(f"🎯 Predicted sales: {prediction:.2f}")
    st.write(f"🧮 Predicted sales: **{prediction:.2f}**")

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('resale_price_model.pkl')

# Define the Streamlit app
st.title('Singapore Resale Flat Price Predictor')

# User inputs
town = st.selectbox('Town', df['town'].unique())
flat_type = st.selectbox('Flat Type', df['flat_type'].unique())
storey_range = st.selectbox('Storey Range', df['storey_range'].unique())
floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0)
flat_model = st.selectbox('Flat Model', df['flat_model'].unique())
lease_commence_date = st.number_input('Lease Commence Year', min_value=1990, max_value=2024, step=1)
remaining_lease = st.selectbox('Remaining Lease', ['Unknown'] + [f'{i} years' for i in range(1, 100)])
flat_age = 2024 - lease_commence_date

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'floor_area_sqm': [floor_area_sqm],
    'flat_age': [flat_age],
    'town_' + town: [1],
    'flat_type_' + flat_type: [1],
    'storey_range_' + storey_range: [1],
    'flat_model_' + flat_model: [1],
    'remaining_lease_' + remaining_lease: [1] if remaining_lease != 'Unknown' else [0]
})

# Fill missing columns with zeros
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Predict resale price
if st.button('Predict'):
    resale_price = model.predict(input_data)
    st.write(f'Predicted Resale Price: ${resale_price[0]:,.2f}')

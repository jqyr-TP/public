import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Add headphone-themed background using CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://upload.wikimedia.org/wikipedia/commons/7/7d/Ghim_Moh_night_panorama%2C_Singapore_-_20110101.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0,0,0,0.7); 
        z-index: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model = joblib.load('model.pkl') 
model_input_columns = joblib.load('model_columns.pkl')
town_options = joblib.load('town_options.pkl')
flat_type_options = joblib.load('flat_type_options.pkl')
flat_model_options = joblib.load('flat_model_options.pkl')
year_options = joblib.load('year_options.pkl')
remaining_lease_years_options = joblib.load('remaining_lease_years_options.pkl')



# Define input options (hardcoded or extracted from training data)
town = list(model.feature_names_in_)
town = [col.replace('town_', '') for col in town if col.startswith('town_')]

flat_type = [col.replace('flat_type_', '') for col in model.feature_names_in_ if col.startswith('flat_type_')]
flat_model = [col.replace('flat_model_', '') for col in model.feature_names_in_ if col.startswith('flat_model_')]
year = [col.replace('year_', '') for col in model.feature_names_in_ if col.startswith('year_')]

storey = [col.replace('storey_avg_', '') for col in model.feature_names_in_ if col.startswith('storey_avg_')]
storey = [int(s) for s in storey] if storey else list(range(1, 51))


st.title("HDB Resale Price Predictor")

# Collect user input for each feature
col1, col2 = st.columns(2)
with col1:
    selected_town = st.selectbox("Town Name", town_options,)
    selected_flat_type = st.selectbox("Flat Type", flat_type_options,)
    selected_flat_model = st.selectbox("Flat Model", flat_model_options,)
with col2:
    selected_storey = st.selectbox("Storey", storey,)
    selected_year = st.selectbox("Year of Sale", year_options)
    selected_remaining_lease_years = st.slider("Remaining Lease Years", min_value=1, max_value=99, value=60)
selected_floor_area_sqm = st.slider("Floor Area (sqm)", min_value=20, max_value=400, value=80, step=1, help="Typical HDB units range between 60–150 sqm")



# Create input DataFrame from user selections
input_dict = {
    'floor_area_sqm': [selected_floor_area_sqm],
    'storey_avg': [selected_storey],
    'town': [selected_town],
    'flat_type': [selected_flat_type],
    'flat_model': [selected_flat_model],
    'year': [selected_year],
    'remaining_lease_years': [selected_remaining_lease_years],
}

input_df = pd.DataFrame(input_dict)

# One-hot encode the categorical features to match training
input_df = pd.get_dummies(input_df, columns=['town', 'flat_type', 'flat_model'])

# Align columns with the training model input
# This is crucial: use the original model input column order
model_input_columns = joblib.load("model_columns.pkl")  # Load this separately saved during training
input_df = input_df.reindex(columns=model_input_columns, fill_value=0)



# Show prediction only after button click
if st.button("Predict Price", key="predict_button"):
    prediction = model.predict(input_df)[0]
    st.markdown(
        f"""
        <div style="background-color: green; opacity: 1; padding: 20px; border-radius: 10px;">
            <h3 style="color: #FFF;">Predicted Resale Price: ${prediction:,.2f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


if st.checkbox("Show input summary"):
    st.json(input_dict)

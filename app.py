import streamlit as st
import pandas as pd
import joblib
import os
from math import radians, sin, cos, sqrt, atan2
import json

# --- Configuration & Model Loading ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_rf.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# Load the trained model
model = None
feature_columns = []

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure it exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    with open(FEATURE_COLUMNS_PATH, 'r') as f:
        feature_columns = json.load(f)
    print(f"Feature columns loaded successfully from {FEATURE_COLUMNS_PATH}")
    print(f"Expected Features: {feature_columns}")
except FileNotFoundError:
    st.error(f"Error: Feature columns file not found at {FEATURE_COLUMNS_PATH}. Please save it from your notebook.")
    st.stop()
except Exception as e:
    st.error(f"Error loading feature columns: {e}")
    st.stop()

# --- Preprocessing Function ---
def preprocess_input_features(raw_data_dict: dict, feature_cols_order: list) -> pd.DataFrame:
    processed_data = pd.DataFrame(index=[0])

    # 1. Handle Start/End Lat/Lng
    processed_data['Start_Lat'] = raw_data_dict['start_lat']
    processed_data['Start_Lng'] = raw_data_dict['start_lng']
    processed_data['End_Lat'] = raw_data_dict['end_lat'] if raw_data_dict['end_lat'] is not None else raw_data_dict['start_lat']
    processed_data['End_Lng'] = raw_data_dict['end_lng'] if raw_data_dict['end_lng'] is not None else raw_data_dict['start_lng']

    # 2. Extract Year, Month, Day, Hour from user inputs
    processed_data['Year'] = raw_data_dict['Year']
    processed_data['Month'] = raw_data_dict['Month']
    processed_data['Day'] = raw_data_dict['Day']
    processed_data['Hour'] = raw_data_dict['Hour']

    # 3. Handle boolean features (Traffic_Signal, Crossing)
    processed_data['Traffic_Signal'] = int(raw_data_dict['Traffic_Signal'])
    processed_data['Crossing'] = int(raw_data_dict['Crossing'])

    # 4. Temperature (in Celsius)
    processed_data['Temperature(C)'] = raw_data_dict['Temperature(C)']

    # 5. Distance(km) - Haversine calculation
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of Earth in kilometers
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    processed_data['Distance(km)'] = haversine_distance(
        processed_data['Start_Lat'].iloc[0], processed_data['Start_Lng'].iloc[0],
        processed_data['End_Lat'].iloc[0], processed_data['End_Lng'].iloc[0]
    )

    # Final DataFrame with features in the correct order
    try:
        final_df = processed_data[feature_cols_order]
    except KeyError as e:
        st.error(f"Feature mismatch: A required feature for the model is missing after preprocessing. Missing: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error creating final DataFrame for prediction: {e}")
        st.stop()
    return final_df

# --- Streamlit App ---
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="⚠️",
    layout="centered"
)



# --- Streamlit App ---
st.title("⚠️ Accident Severity Predictor")
st.markdown("Fill out the form below to predict potential accident severity.")

# Use st.form to group all input widgets
with st.form("accident_prediction_form", clear_on_submit=True):
    st.subheader("Location")
    col1, col2 = st.columns(2)
    with col1:
        start_lat = st.number_input(
            "Start Latitude",
            min_value=-90.0, max_value=90.0, value=34.0, step=0.0001, format="%.4f",
            help="Geographic latitude of the accident start point."
        )
        start_lng = st.number_input(
            "Start Longitude",
            min_value=-180.0, max_value=180.0, value=-118.0, step=0.0001, format="%.4f",
            help="Geographic longitude of the accident start point."
        )
        end_lat_default = 34.0
        end_lng_default = -118.0
        end_lat_input = st.number_input(
            "End Latitude (Optional)",
            min_value=-90.0, max_value=90.0, value=end_lat_default, step=0.0001, format="%.4f",
            help="Geographic latitude of the accident end point. Defaults to Start Lat if left unchanged."
        )
        end_lng_input = st.number_input(
            "End Longitude (Optional)",
            min_value=-180.0, max_value=180.0, value=end_lng_default, step=0.0001, format="%.4f",
            help="Geographic longitude of the accident end point. Defaults to Start Lng if left unchanged."
        )

        end_lat = end_lat_input if end_lat_input != end_lat_default else None
        end_lng = end_lng_input if end_lng_input != end_lng_default else None

    with col2:
        st.subheader("Time and Conditions")
        Year = st.number_input("Year", min_value=2000, max_value=2025, value=2023, step=1, help="Year of the accident")
        Month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1, help="Month of the accident (1-12)")
        Day = st.number_input("Day", min_value=1, max_value=31, value=1, step=1, help="Day of the accident (1-31)")
        Hour = st.number_input("Hour", min_value=0, max_value=23, value=12, step=1, help="Hour of the accident (0-23)")

        TemperatureC = st.slider(
            "Temperature (°C)",
            min_value=-30.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
            help="Temperature in Celsius at the time of accident."
        )
        Traffic_Signal = st.checkbox("Traffic Signal Present?", value=False)
        Crossing = st.checkbox("Crossing Present?", value=False)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Predict Severity")

# --- Prediction Logic (only runs if form is submitted) ---
if submitted:
    # Prepare data for prediction
    raw_input_data = {
        "Temperature(C)": TemperatureC,
        "start_lat": start_lat,
        "start_lng": start_lng,
        "end_lat": end_lat,
        "end_lng": end_lng,
        "Traffic_Signal": Traffic_Signal,
        "Crossing": Crossing,
        "Year": Year,
        "Month": Month,
        "Day": Day,
        "Hour": Hour,
    }

    try:
        # Preprocess the input features
        processed_df = preprocess_input_features(raw_input_data, feature_columns)

        # Make prediction
        prediction = model.predict(processed_df)[0]
        severity_map = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
        predicted_severity = severity_map.get(prediction, "Unknown Severity")

        st.subheader("Prediction Result:")
        if predicted_severity == "Low":
            st.success(f"Predicted Accident Severity: **{predicted_severity}** ✅ No Major Concerns")
        elif predicted_severity == "Medium":
            st.warning(f"Predicted Accident Severity: **{predicted_severity}** ⚠️ Moderate Risk")
        elif predicted_severity == "High":
            st.error(f"Predicted Accident Severity: **{predicted_severity}** 🚨 High Risk")
        elif predicted_severity == "Critical":
            st.exception(f"Predicted Accident Severity: **{predicted_severity}** 🛑 Critical Risk! Take Extreme Caution.")
        else:
            st.info(f"Predicted Accident Severity: **{predicted_severity}**")

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")

st.markdown("---")
st.markdown("Application Build By Abhinav Marlingaplar.")

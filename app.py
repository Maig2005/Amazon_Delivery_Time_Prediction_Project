import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Import the necessary feature engineering functions from your utility file
try:
    from data_preparation import create_prediction_df
except ImportError:
    st.error("Error: Could not find 'data_preparation.py'. Please ensure it is in the same directory.")
    st.stop()

# --- 1. Load the Model ---
@st.cache_resource
def load_model(path='final_delivery_time_prediction_model.pkl'):
    """Loads the saved model pipeline using joblib."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {path}. Please run 'model_training.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model globally
pipeline = load_model()

# --- 2. Define Input Options (Based on Feature Engineering) ---
WEATHER_OPTIONS = ['Sunny', 'Stormy', 'Cloudy', 'Rainy', 'Windy']
TRAFFIC_OPTIONS = ['Low', 'Medium', 'High', 'Jam']
VEHICLE_OPTIONS = ['Bike', 'Car', 'Scooter', 'Electric Vehicle']
AREA_OPTIONS = ['Urban', 'Rural', 'Semi-Urban']
CATEGORY_OPTIONS = ['Food', 'Non-Food', 'Electronics', 'Others']

# --- 3. Streamlit App Layout ---
st.title("📦 Amazon Delivery Time Predictor")
st.markdown("Use the controls below to input delivery details and get an estimated delivery time (ETA) in minutes.")
st.divider()

# --- Input Form ---
with st.form("prediction_form"):
    
    # 3.1 Agent & Store Information
    st.header("1. Agent & Location Details")
    col1, col2 = st.columns(2)
    
    # Coordinates (Use standard Amazon warehouse coordinates for example)
    store_lat = col1.number_input("Store Latitude (Example: 12.97)", value=12.9716, format="%.4f")
    store_lon = col1.number_input("Store Longitude (Example: 77.59)", value=77.5946, format="%.4f")
    drop_lat = col2.number_input("Drop Latitude (Customer)", value=13.0000, format="%.4f")
    drop_lon = col2.number_input("Drop Longitude (Customer)", value=77.6500, format="%.4f")
    
    st.markdown("---")
    
    # 3.2 Agent Performance & Vehicle
    col3, col4, col5 = st.columns(3)
    
    agent_age = col3.slider("Agent Age (Years)", min_value=18, max_value=60, value=28)
    agent_rating = col4.slider("Agent Rating (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    vehicle = col5.selectbox("Vehicle Type", options=VEHICLE_OPTIONS)
    
    st.markdown("---")
    
    # 3.3 Temporal & Environmental Factors
    st.header("2. Time & Environmental Factors")
    col6, col7, col8 = st.columns(3)
    
    order_date_input = col6.date_input("Order Date", value=date.today())
    order_time_input = col7.time_input("Order Time", value=pd.to_datetime("14:30:00").time())
    
    traffic = col8.selectbox("Traffic Density", options=TRAFFIC_OPTIONS)
    
    col9, col10 = st.columns(2)
    weather = col9.selectbox("Weather Condition", options=WEATHER_OPTIONS)
    category = col10.selectbox("Item Category", options=CATEGORY_OPTIONS)
    area = st.selectbox("Area Type", options=AREA_OPTIONS)

    # 4. Prediction Button
    submitted = st.form_submit_button("Predict Delivery Time")

# --- 4. Handle Submission and Prediction ---
if submitted:
    
    # 4.1 Assemble Input Data
    input_data = {
        'Store_Latitude': store_lat,
        'Store_Longitude': store_lon,
        'Drop_Latitude': drop_lat,
        'Drop_Longitude': drop_lon,
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Weather': weather,
        'Traffic': traffic,
        'Vehicle': vehicle,
        'Area': area,
        'Category': category,
        # Format date and time for feature engineering function
        'Order_Date': order_date_input.strftime('%Y-%m-%d'),
        'Order_Time': order_time_input.strftime('%H:%M:%S'),
        # These are required by the data_preparation function but not used by the model directly
        'Pickup_Time': '15:00:00' 
    }
    
    # 4.2 Create the Feature Engineered DataFrame
    try:
        # This function handles Haversine, Day_of_Week, Order_Hour, etc.
        input_df = create_prediction_df(input_data)
        
        # 4.3 Make Prediction
        prediction_minutes = pipeline.predict(input_df)[0]
        
        # 4.4 Display Result
        st.success(f"### Predicted Delivery Time (ETA): **{prediction_minutes:.2f} minutes**")
        
        # Optional: Convert minutes to hours and minutes
        hours = int(prediction_minutes // 60)
        minutes = int(prediction_minutes % 60)
        st.info(f"This is approximately **{hours} hours and {minutes} minutes**.")
        
        st.markdown("---")
        with st.expander("Show Raw Input to Model"):
             st.dataframe(input_df)
             
    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your inputs. Details: {e}")

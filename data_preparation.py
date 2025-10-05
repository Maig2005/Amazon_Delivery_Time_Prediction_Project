# data_preparation.py

import pandas as pd
import numpy as np

# --- 1. Haversine Distance Function ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two coordinates in kilometers."""
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula components
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in kilometers is 6371 km
    R = 6371 
    return R * c

# --- 2. Time Categorization Function ---
def get_time_of_day(hour):
    """Categorizes the hour into key delivery windows."""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    else:
        return 'Evening_Night'

# --- 3. Full Data Preprocessing Function ---
def preprocess_data(df):
    """
    Applies all necessary cleaning and feature engineering steps 
    to the raw input DataFrame (or a single row for prediction).
    """
    
    # Ensure all columns exist before processing (important for single-row input)
    required_cols = [
        'Delivery_Time', 'Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude', 
        'Drop_Latitude', 'Drop_Longitude', 'Order_Date', 'Order_Time', 'Pickup_Time'
    ]
    
    # --- Cleaning/Type Conversion ---
    
    # Clean Target Variable (only needed if processing full dataset)
    if 'Delivery_Time' in df.columns:
        df['Delivery_Time'] = pd.to_numeric(df['Delivery_Time'], errors='coerce')
        df.dropna(subset=['Delivery_Time'], inplace=True)
        
    # Impute Agent_Age with mean (if working with the full training set)
    if df['Agent_Age'].isnull().any():
        mean_age = df['Agent_Age'].mean()
        df['Agent_Age'].fillna(mean_age, inplace=True)
    
    # Convert Time/Date columns to objects for feature extraction
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    
    # Safely convert Order_Time string to a datetime.time object for hour extraction
    # Need to handle single-row prediction case where 'Order_Time' might be a single string
    if isinstance(df['Order_Time'].iloc[0], str):
        df['Order_Time_dt'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.time
    elif isinstance(df['Order_Time'].iloc[0], pd.Series):
         # Handle case where it might already be an object
         df['Order_Time_dt'] = df['Order_Time'] 

    df.dropna(subset=['Order_Time_dt'], inplace=True)


    # --- Feature Engineering ---

    # 1. Haversine Distance
    df['Distance_km'] = df.apply(
        lambda row: haversine(
            row['Store_Latitude'], row['Store_Longitude'], 
            row['Drop_Latitude'], row['Drop_Longitude']
        ), axis=1
    )

    # 2. Time Features
    df['Day_of_Week'] = df['Order_Date'].dt.dayofweek
    df['Order_Hour'] = df['Order_Time_dt'].apply(lambda x: x.hour)
    df['Order_Time_Category'] = df['Order_Hour'].apply(get_time_of_day)

    
    # --- Final Feature Selection (Must match features used in training) ---
    final_features = [
        'Distance_km', 'Agent_Age', 'Agent_Rating', 'Order_Hour', 'Day_of_Week',
        'Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Order_Time_Category'
    ]
    
    # Return only the necessary columns for the final model input
    return df[final_features]

# --- Example of creating a single prediction DataFrame for Streamlit ---
def create_prediction_df(input_data):
    """
    Creates a DataFrame from Streamlit input and performs feature engineering 
    for a single prediction.
    """
    # Create a DataFrame from the input dictionary (single row)
    df_pred = pd.DataFrame([input_data])
    
    # Ensure necessary columns are converted for feature engineering
    df_pred['Order_Date'] = pd.to_datetime(df_pred['Order_Date'])
    
    # Order_Time is expected as a string like '14:30:00'
    df_pred['Order_Time_dt'] = pd.to_datetime(df_pred['Order_Time'], format='%H:%M:%S', errors='coerce').dt.time
    
    # Add engineered features
    df_pred['Distance_km'] = df_pred.apply(
        lambda row: haversine(
            row['Store_Latitude'], row['Store_Longitude'], 
            row['Drop_Latitude'], row['Drop_Longitude']
        ), axis=1
    )
    df_pred['Day_of_Week'] = df_pred['Order_Date'].dt.dayofweek
    df_pred['Order_Hour'] = df_pred['Order_Time_dt'].apply(lambda x: x.hour)
    df_pred['Order_Time_Category'] = df_pred['Order_Hour'].apply(get_time_of_day)

    # Select and return the final features used by the trained pipeline
    final_features = [
        'Distance_km', 'Agent_Age', 'Agent_Rating', 'Order_Hour', 'Day_of_Week',
        'Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Order_Time_Category'
    ]
    
    return df_pred[final_features]

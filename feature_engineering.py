# --- Feature Engineering Functions ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two coordinates in kilometers."""
    # ... (full haversine formula implementation) ...
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371 
    return R * c

def get_time_of_day(hour):
    """Categorizes the hour into key delivery windows."""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    else:
        return 'Evening_Night'

# --- Core Feature Engineering Steps from the create_prediction_df function ---

def create_prediction_df(input_data):
    """
    Creates a DataFrame from Streamlit input and performs feature engineering 
    for a single prediction row.
    """
    df_pred = pd.DataFrame([input_data])
    
    # Ensure necessary columns are converted for feature engineering
    df_pred['Order_Date'] = pd.to_datetime(df_pred['Order_Date'])
    df_pred['Order_Time_dt'] = pd.to_datetime(df_pred['Order_Time'], format='%H:%M:%S', errors='coerce').dt.time
    
    # 1. Haversine Distance
    df_pred['Distance_km'] = df_pred.apply(
        lambda row: haversine(
            row['Store_Latitude'], row['Store_Longitude'], 
            row['Drop_Latitude'], row['Drop_Longitude']
        ), axis=1
    )

    # 2. Time Features
    df_pred['Day_of_Week'] = df_pred['Order_Date'].dt.dayofweek
    df_pred['Order_Hour'] = df_pred['Order_Time_dt'].apply(lambda x: x.hour)
    df_pred['Order_Time_Category'] = df_pred['Order_Hour'].apply(get_time_of_day)

    # Select and return the final features used by the trained pipeline
    final_features = [
        'Distance_km', 'Agent_Age', 'Agent_Rating', 'Order_Hour', 'Day_of_Week',
        'Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Order_Time_Category'
    ]
    
    return df_pred[final_features]

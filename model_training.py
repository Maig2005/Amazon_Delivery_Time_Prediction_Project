# model_training.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Assuming data_preparation.py is in the same directory for feature engineering functions
from data_preparation import haversine, get_time_of_day 


# --- 1. Master Data Preparation Function (Consolidating Notebook Steps) ---

def load_and_prepare_data(file_path='amazon_delivery.csv'):
    """Loads, cleans, and engineers features for the full training dataset."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    # --- Cleaning ---
    df.drop_duplicates(inplace=True)
    df['Delivery_Time'] = pd.to_numeric(df['Delivery_Time'], errors='coerce')
    df.dropna(subset=['Delivery_Time', 'Agent_Rating'], inplace=True)
    
    # Handle Agent_Age (Imputation)
    df['Agent_Age'].fillna(df['Agent_Age'].mean(), inplace=True)
    
    # Convert Time/Date
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Order_Time_dt'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.time
    df.dropna(subset=['Order_Time_dt'], inplace=True)

    # --- Feature Engineering ---
    
    # Haversine Distance
    df['Distance_km'] = df.apply(
        lambda row: haversine(
            row['Store_Latitude'], row['Store_Longitude'], 
            row['Drop_Latitude'], row['Drop_Longitude']
        ), axis=1
    )
    
    # Time Features
    df['Day_of_Week'] = df['Order_Date'].dt.dayofweek
    df['Order_Hour'] = df['Order_Time_dt'].apply(lambda x: x.hour)
    df['Order_Time_Category'] = df['Order_Hour'].apply(get_time_of_day)

    return df

# --- 2. Main Training Function ---

def train_and_save_model(df):
    """Defines pipeline, trains Random Forest model, and saves the final pipeline."""
    
    # --- Define Features (X) and Target (y) ---
    # Must drop ALL raw, redundant, and intermediary columns
    X = df.drop(columns=[
        'Delivery_Time', 'Order_ID', 'Store_Latitude', 'Store_Longitude', 
        'Drop_Latitude', 'Drop_Longitude', 'Order_Date', 'Order_Time', 
        'Pickup_Time', 'Order_Time_dt', 'Pickup_Time_dt' 
    ])
    y = df['Delivery_Time']

    # --- Define Preprocessing Steps ---
    numerical_features = ['Distance_km', 'Agent_Age', 'Agent_Rating', 'Order_Hour', 'Day_of_Week']
    categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Order_Time_Category'] 

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' 
    )

    # --- Define Best Model Pipeline ---
    best_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])
    
    print("Training final Random Forest Regressor on all data...")
    final_pipeline.fit(X, y)
    print("Model training complete.")
    
    # --- Save the Pipeline ---
    model_filename = 'final_delivery_time_prediction_model.pkl'
    joblib.dump(final_pipeline, model_filename)
    print(f"Final trained pipeline saved successfully as {model_filename}")


# --- 3. Script Execution ---

if __name__ == '__main__':
    # Load and preprocess the data
    df_clean = load_and_prepare_data()
    
    if df_clean is not None:
        # Train and save the model
        train_and_save_model(df_clean)

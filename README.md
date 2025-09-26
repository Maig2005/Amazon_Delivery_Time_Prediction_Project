Amazon Delivery Time Prediction


1. Project Overview
This project aims to build a machine learning model to accurately predict delivery times for Amazon e-commerce orders. The prediction is based on a comprehensive set of features, including delivery agent characteristics, environmental factors (weather and traffic), and logistical details (store and drop-off coordinates, product category). The ultimate goal is to create a fully functional, user-friendly Streamlit application that provides real-time delivery time estimates.


2. Business Value
    - Enhanced Customer Satisfaction: Providing accurate delivery windows to customers.

    - Operational Efficiency: Optimizing resource allocation, delivery routes, and agent scheduling.

    - Logistics Optimization: Identifying and mitigating factors that cause delivery delays (e.g., specific traffic or weather conditions).


3. Technical Tags
Python • Supervised Regression • Pandas • Scikit-learn • Data Cleaning • Feature Engineering • Streamlit • MLflow


4. Project Structure
The repository is organized to ensure modularity, reproducibility, and clear separation of concerns, following the proposed file structure:

amazon-delivery-prediction-project/
├── data/
│   ├── raw/
│   │   └── amazon_delivery.csv   # Original dataset
│   └── processed/                # Cleaned and feature-engineered data
├── notebooks/
│   └── EDA_and_Modelling.ipynb   # Exploratory Data Analysis and model experimentation
├── src/
│   ├── data_preparation.py       # Functions for cleaning and preprocessing
│   ├── feature_engineering.py    # Functions for creating new features (e.g., Distance)
│   └── model_training.py         # Script for training, evaluating, and logging models with MLflow
├── app/
│   └── app.py                    # Streamlit application code
├── models/                       # Directory to save the final trained model
├── mlruns/                       # MLflow tracking data (parameters, metrics, artifacts)
├── requirements.txt              # List of all necessary Python dependencies
└── README.md


5. Getting Started
   
    i. Prerequisites
        a.Clone the Repository:

            git clone [your-repo-link]
            cd amazon-delivery-prediction-project

        b. Set up Environment:
            Install all required Python packages:

            pip install -r requirements.txt

    ii. Data
        The raw data, amazon_delivery.csv, is located in the data/raw/ directory.

    iii. Execution Steps
        a. Data Processing & Feature Engineering:
            Run the scripts in the src/ directory to clean the data and perform feature engineering (especially the calculation of the Euclidean distance between store and drop-off locations).

            # Example command to run your main processing script (if you create one)
            # python main_process.py
        b. Model Training and Tracking:
            The src/model_training.py script will train multiple regression models (e.g., Linear Regression, Random Forest, XGBoost) and log their parameters and performance metrics using MLflow.

            To view MLflow experiments, run:

                mlflow ui
            Then, open your browser to http://localhost:5000.

        c. Run the Streamlit Application:
            Once the best model is trained and saved in the models/ directory, the application can be launched:

                streamlit run app/app.py
            The prediction interface will open in your web browser.


7. Evaluation Metrics
Model performance is evaluated using the following regression metrics, as specified in the project scope:

    - Root Mean Squared Error (RMSE)

    - Mean Absolute Error (MAE)

    - R-squared (R^2)


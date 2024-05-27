# Shipment Delay Analysis

## Overview
This project utilizes shipment booking and GPS tracking data to analyze delivery performance and predict potential delays. The solutions for Task 1 and Task 2 are implemented in Python and involve data preprocessing, integration with the Google Maps API, and detailed analysis of on-time deliveries and delay predictions. This document provides an overview of the methodologies, project structure, and setup instructions.

## Project Structure

    project_name/
    │
    ├── data/                # Folder containing all data files.
    │
    ├── results/             # Output files and artifacts such as figures and logs.
    │
    ├──model_repository/model_v1  # trained model repository
    │
    ├── preprocessing.py     # Script for preprocessing data for model training.
    ├── model_train.py       # Script for training the initial machine learning model.
    ├── model_tuning.py      # Script for hyperparameter tuning.
    ├── predict.py           # Script for predicting delays on new data.
    ├── Task1.py             # Script for computing on-time delivery rates.
    ├── Task2.py             # Script for predicting delays and notifying shippers.
    │
    └── README.md            # Overview and documentation of the project.





## Task 1: On-Time Delivery Analysis
The `Task1.py` script evaluates the punctuality of shipments by calculating the percentage that were delivered on time, based on their scheduled delivery windows. Here’s a concise overview of the process:

- **Data Loading**: The script loads shipment booking and GPS tracking data from the `data/` directory.

- **Data Filtering**: It filters this data to focus only on shipments scheduled for delivery between October 1st and December 31st, 2023.

- **Computation**: The script determines which shipments were delivered within a 30-minute grace period of their scheduled times. This step involves comparing the actual delivery timestamps with the scheduled delivery windows to identify timely versus delayed shipments.

- **Output**: The final percentage of on-time deliveries is computed and printed.



## Task 2: Delay Prediction and Notification
The `Task2.py` script uses real-time data to predict potential shipment delays and automatically generates notifications for stakeholders. The process is outlined as follows:

- **Data Integration**: Real-time GPS data is integrated with static shipment data.

- **Google Maps API**: This API is utilized to fetch current travel times and distances,to estimate the time of arrival based on average speed (crude estimation).

- **Delay Prediction**: For each shipment, the script calculates whether the updated estimated time of arrival (ETA) will exceed the scheduled delivery time.

- **Notifications**: If a delay is predicted, the script logs this information and can be used to trigger notifications to relevant parties.

- **Output**: All findings, specifically the earliest time a possible delay is detected, are stored in the `results/processed_shipment_data.csv` and `results/delayed_shipments_and_earliest_notification_time.csv`.

## Task 3: Predicting the Likelihood of Delay

### Overview
Task 3 involves building a machine learning model to predict the likelihood of delays for shipments listed in the `New_bookings.csv` dataset. This process is broken down into several steps, each handled by a different script.

### preprocessing.py

**Purpose**: 
- Preprocess raw data from `gps_data.csv` and `shipment_bookings.csv` to create a dataset ready for model training.

**Inputs**:
- `gps_data.csv`
- `shipment_bookings.csv`

**Outputs**:
- `processed_shipment_data.csv` (stored in the `results` folder)

**Key Steps**:
- Load raw data from CSV files.
- Merge GPS data with shipment booking details based on `SHIPMENT_NUMBER`.
- Perform feature engineering by calculating distances, step counts, and ferry involvement using an external API.
- Save the processed data for later use.

### model_train.py

**Purpose**:
- Train an initial machine learning model using the preprocessed data.

**Inputs**:
- `processed_shipment_data.csv`

**Outputs**:
- `model_v1/model_pipeline.joblib` (stored in the `model_repository` folder)
- Evaluation metrics and a classification report printed to the console and stored in the `results` folder

**Key Steps**:

#### Load the preprocessed data:
- The preprocessed data includes various engineered features that provide valuable insights for training the model. This data is stored in the `processed_shipment_data.csv` file.

#### Define feature columns and target variables:
- **Feature Columns**:
  - `VEHICLE_SIZE`: Type of the vehicle used for the shipment.
  - `VEHICLE_BUILD_UP`: Configuration or build-up of the vehicle trailer.
  - `CARRIER_DISPLAY_ID`: Unique identifier for the carrier.
  - `FIRST_COLLECTION_POST_CODE`: Postcode of the first collection point.
  - `LAST_DELIVERY_POST_CODE`: Postcode of the last delivery point.
  - `FIRST_COLLECTION_LATITUDE`: Latitude of the first collection point.
  - `FIRST_COLLECTION_LONGITUDE`: Longitude of the first collection point.
  - `LAST_DELIVERY_LATITUDE`: Latitude of the last delivery point.
  - `LAST_DELIVERY_LONGITUDE`: Longitude of the last delivery point.
  - `Step_Count`: Number of steps (or segments) in the route from the collection point to the delivery point.
  - `Distance_km`: Total distance of the route in kilometers.
  - `Ferry_Involved`: Boolean indicating whether a ferry is involved in the route (1 if involved, 0 if not).

- **Target Variable**:
  - `is_delayed`: Binary variable indicating whether the shipment was delayed (1 if delayed, 0 if not).

#### Split the data into training and testing sets:
- The data is split into training and testing sets to evaluate the model's performance. Typically, this involves using 80% of the data for training and 20% for testing.

#### Set up preprocessing pipelines and the model:
- **Preprocessing Pipelines**:
  - **Numerical Features**: Features such as `FIRST_COLLECTION_LATITUDE`, `FIRST_COLLECTION_LONGITUDE`, `LAST_DELIVERY_LATITUDE`, `LAST_DELIVERY_LONGITUDE`, `Step_Count`, and `Distance_km` are scaled to standardize their ranges.
  - **Categorical Features**: Features such as `VEHICLE_SIZE`, `VEHICLE_BUILD_UP`, `CARRIER_DISPLAY_ID`, `FIRST_COLLECTION_POST_CODE`, and `LAST_DELIVERY_POST_CODE` are encoded using one-hot encoding.
  - **Binary Features**: The `Ferry_Involved` feature is converted from boolean to integer.

- **Model**:
  - I used the `RandomForestClassifier` from scikit-learn, which is an ensemble learning method that combines multiple decision trees to improve the model's accuracy and robustness.

#### Train the model and evaluate its performance:
- The model is trained using the training data and its performance is evaluated using the testing data. Evaluation metrics such as precision, recall, F1-score, and accuracy are computed to assess the model's performance. I have saved th ROC in 'results' folder

#### Save the trained model:
- The trained model is saved to the `model_v1/model_pipeline.joblib` file in the `model_repository` folder. This saved model can later be loaded for making predictions on new data.


### model_tuning.py

**Purpose**: 
- Perform hyperparameter tuning to optimize the model's performance.

**Inputs**:
- `processed_shipment_data.csv`

**Outputs**:
- `best_hyperparameters_v1.yaml` (stored in the `results` folder)

**Key Steps**:
- Load the preprocessed data.
- Define feature columns and target variables.
- Split the data into training and testing sets.
- Set up preprocessing pipelines and the model ( I used RandomForestClassifier).
- Define a parameter grid for hyperparameter tuning.
- Use GridSearchCV to find the best hyperparameters.
- Train the model with the best hyperparameters and evaluate its performance.
- Save the the best hyperparameters.

### predict.py

**Purpose**: 
- Predict the likelihood of delay for new shipments listed in `New_bookings.csv`.

**Inputs**:
- `New_bookings.csv`
- `model_v2/model_pipeline.joblib`

**Outputs**:
- `new_bookings_predictions.csv` (stored in the `results` folder)

**Key Steps**:
- Load new shipment data from `New_bookings.csv`.
- Extract and engineer features similar to the preprocessing step.
- Load the trained model.
- Make predictions on the new shipment data.
- Save the predictions along with their probabilities.

## Installation and Setup
Follow these steps to set up and run the analysis:
1. **Clone the repository**: Download the project to your local machine.
2. **Install Python**: Ensure Python 3.x is installed.
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt



## Data
Place all necessary data files within the `data/` folder.

## Execution
To run the scripts, execute the following commands in your terminal:

```bash
python Task1.py
python Task2.py


python preprocessing.py
python model_train.py
python model_tuning.py
python predict.py

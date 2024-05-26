import pandas as pd
import logging
import yaml
import os
from joblib import load
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config.yaml'):
    """ Load the YAML configuration file """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logging.error(f"Error loading the configuration file: {e}")
        raise

def load_data(file_path):
    """ Load data from a csv file """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise

def load_model(model_path):
    """ Load a trained model from a file """
    try:
        model = load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError as e:
        logging.error(f"Error loading model: {e}")
        raise

def save_predictions(predictions, file_path):
    """ Save predictions to a csv file """
    try:
        predictions.to_csv(file_path, index=False)
        logging.info(f"Predictions saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving predictions: {e}")
        raise

def convert_bool_to_int(X):
    """ Convert boolean columns to integers """
    return X.astype(int)

def get_route_info_with_road_types(origin, destination, osrm_base_url):
    """ Fetch route information including distance, number of steps, and ferry involvement """
    try:
        osrm_route_url = f"{osrm_base_url}{origin[1]},{origin[0]};{destination[1]},{destination[0]}?overview=full&geometries=geojson&steps=true"
        response = requests.get(osrm_route_url)
        if response.status_code != 200:
            return None, None, None  # Handle non-success responses

        data = response.json()
        route = data['routes'][0]
        distance = route['distance'] / 1000  # Convert to kilometers
        steps = route['legs'][0]['steps']
        num_steps = len(steps)
        ferry_involved = any('ferry' in step.get('mode', '') for step in steps)

        return distance, num_steps, ferry_involved
    except requests.RequestException as e:
        logging.error(f"Failed to fetch route data: {e}")
        return None, None, None

def main():
    config = load_config()
    data_path = os.path.join(config['data_paths']['data_dir'], 'New_bookings.csv')
    model_version = config['versioning']['model_version']
    model_path = os.path.join(config['versioning']['model_repository'], f"model_v{model_version}", 'model_pipeline.joblib')
    results_dir = config['versioning']['results_dir']
    predictions_file_path = os.path.join(results_dir, 'new_bookings_predictions.csv')
    osrm_base_url = config['api_settings']['osrm_base_url']

    data = load_data(data_path)
    #data= data.iloc[:10,:]
    
    # Define feature columns
    feature_columns = [
        'VEHICLE_SIZE', 'VEHICLE_BUILD_UP', 'CARRIER_DISPLAY_ID', 
        'FIRST_COLLECTION_POST_CODE', 'LAST_DELIVERY_POST_CODE',
        'FIRST_COLLECTION_LATITUDE', 'FIRST_COLLECTION_LONGITUDE',
        'LAST_DELIVERY_LATITUDE', 'LAST_DELIVERY_LONGITUDE'
    ]
    
    # Extract route information features
    distances = []
    step_counts = []
    ferries_involved = []
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        origin = (row['FIRST_COLLECTION_LATITUDE'], row['FIRST_COLLECTION_LONGITUDE'])
        destination = (row['LAST_DELIVERY_LATITUDE'], row['LAST_DELIVERY_LONGITUDE'])
        distance, num_steps, ferry_involved = get_route_info_with_road_types(origin, destination, osrm_base_url)

        distances.append(distance)
        step_counts.append(num_steps)
        ferries_involved.append(ferry_involved)

    data['Distance_km'] = distances
    data['Step_Count'] = step_counts
    data['Ferry_Involved'] = ferries_involved

    # Select features for prediction
    feature_columns.extend(['Distance_km', 'Step_Count', 'Ferry_Involved'])
    X = data[feature_columns]

    model = load_model(model_path)

    predictions = model.predict(X)
    prediction_probabilities = model.predict_proba(X)[:, 1]

    data['predicted_delay'] = predictions
    data['prediction_probability'] = prediction_probabilities

    save_predictions(data, predictions_file_path)

if __name__ == "__main__":
    main()

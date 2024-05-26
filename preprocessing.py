import pandas as pd
import numpy as np
import logging
import yaml
from tqdm import tqdm
import requests
import os

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

def load_data(file_name, data_dir):
    """ Load data from a csv file """
    try:
        data_path = os.path.join(data_dir, file_name)
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {file_name}")
        return data
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise

def clean_data(data):
    """ Clean data by handling missing values, outliers, and errors """
    data = data.replace("?", np.nan)
    data.dropna(inplace=True)
    logging.info("Data cleaned successfully")
    return data

def feature_engineering(shipments, gps, osrm_base_url):
    """ Perform feature engineering on the datasets """
    # Filter shipments within the specified date range
    shipments['LAST_DELIVERY_SCHEDULE_LATEST'] = pd.to_datetime(shipments['LAST_DELIVERY_SCHEDULE_LATEST'], errors='coerce')
    filtered_shipments = shipments[(shipments['LAST_DELIVERY_SCHEDULE_LATEST'] >= '2023-10-01') & (shipments['LAST_DELIVERY_SCHEDULE_LATEST'] <= '2023-12-31')]

    # Merge the GPS data with shipment booking details
    merged_data = pd.merge(gps, filtered_shipments, on='SHIPMENT_NUMBER')

    # Convert the columns to datetime type
    merged_data['RECORD_TIMESTAMP'] = pd.to_datetime(merged_data['RECORD_TIMESTAMP'], errors='coerce')
    merged_data['LAST_DELIVERY_SCHEDULE_LATEST'] = pd.to_datetime(merged_data['LAST_DELIVERY_SCHEDULE_LATEST'], errors='coerce')

    # Sort the GPS records by timestamp in descending order to get the latest position first
    merged_data = merged_data.sort_values(by='RECORD_TIMESTAMP', ascending=False)

    # Group by shipment number and take the first record, which is the latest GPS record for each shipment
    latest_positions = merged_data.groupby('SHIPMENT_NUMBER').first().reset_index()


    # New code to fetch route info
    distances = []
    step_counts = []
    ferries_involved = []

    for idx, row in tqdm(latest_positions.iterrows(), total=latest_positions.shape[0]):
        origin = f"{row['FIRST_COLLECTION_LONGITUDE']},{row['FIRST_COLLECTION_LATITUDE']}"
        destination = f"{row['LAST_DELIVERY_LONGITUDE']},{row['LAST_DELIVERY_LATITUDE']}"
        distance, num_steps, ferry_involved = get_route_info_with_road_types(origin, destination, osrm_base_url)
        
        distances.append(distance)
        step_counts.append(num_steps)
        ferries_involved.append(ferry_involved)
    
    latest_positions['Distance_km'] = distances
    latest_positions['Step_Count'] = step_counts
    latest_positions['Ferry_Involved'] = ferries_involved

    
    # Calculate if the shipment is delayed
    latest_positions['is_delayed'] = latest_positions['RECORD_TIMESTAMP'] > (latest_positions['LAST_DELIVERY_SCHEDULE_LATEST'] + pd.Timedelta(minutes=30))

    logging.info("Feature engineering complete.")
    return latest_positions

def get_route_info_with_road_types(origin, destination, base_url):
    """ Fetch route information using the OSRM API """
    url = f"{base_url}{origin};{destination}?overview=full&geometries=geojson&steps=true"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            logging.error(f"Non-successful OSRM API call with status code: {response.status_code}")
            return None, None, None

        data = response.json()
        route = data['routes'][0]
        distance = route['distance'] / 1000  # convert meters to kilometers
        steps = route['legs'][0]['steps']
        num_steps = len(steps)
        ferry_involved = any('ferry' in step.get('mode', '') for step in steps)

        return distance, num_steps, ferry_involved
    except requests.RequestException as e:
        logging.error(f"Failed to fetch route data: {e}")
        return None, None, None

def main():
    config = load_config()
    data_dir = config['data_paths']['data_dir']
    gps_file = config['data_paths']['gps_file']
    shipments_file = config['data_paths']['shipments_file']
    osrm_base_url = config['api_settings']['osrm_base_url']
    results_dir = config['results_dir']
    preprocessed_file_path = os.path.join(results_dir, config['data_paths']['preprocessed_file'])

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    gps_data = load_data(gps_file, data_dir)
    shipments_data = load_data(shipments_file, data_dir)
    shipments_data = shipments_data.iloc[:30, :]

    processed_data = feature_engineering(shipments_data, gps_data, osrm_base_url)
    print(processed_data.head())

    # Save the processed data to a CSV file
    processed_data.to_csv(preprocessed_file_path, index=False)
    logging.info(f"Processed data saved to {preprocessed_file_path}")

if __name__ == "__main__":
    main()

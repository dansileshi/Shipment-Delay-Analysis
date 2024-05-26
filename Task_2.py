import pandas as pd
import googlemaps
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import configparser
import googlemaps

# Initialize Google Maps client

# Create a parser
config = configparser.ConfigParser()
# Read configuration file
config.read('config.ini')

# Get the API key
api_key = config.get('google_maps', 'api_key')

# Use the API key to create a client
gmaps = googlemaps.Client(key=api_key)


def load_data(gps_path, shipments_path):
    """Load GPS and shipment data from CSV files."""
    gps_data = pd.read_csv(gps_path)
    shipments = pd.read_csv(shipments_path)
    return gps_data, shipments

def preprocess_data(gps_data, shipments):
    """Convert timestamps, handle missing values, and ensure all timestamps are timezone-naive."""
    gps_data['RECORD_TIMESTAMP'] = pd.to_datetime(gps_data['RECORD_TIMESTAMP'], errors='coerce')
    shipments['LAST_DELIVERY_SCHEDULE_LATEST'] = pd.to_datetime(shipments['LAST_DELIVERY_SCHEDULE_LATEST'], errors='coerce')

    gps_data = gps_data.dropna(subset=['RECORD_TIMESTAMP'])
    shipments = shipments.dropna(subset=['LAST_DELIVERY_SCHEDULE_LATEST'])

    gps_data['RECORD_TIMESTAMP'] = gps_data['RECORD_TIMESTAMP'].apply(lambda x: x.replace(tzinfo=None))
    shipments['LAST_DELIVERY_SCHEDULE_LATEST'] = shipments['LAST_DELIVERY_SCHEDULE_LATEST'].apply(lambda x: x.replace(tzinfo=None))
    
    return gps_data, shipments

def merge_data(gps_data, shipments):
    """Merge GPS data with shipment booking data on 'SHIPMENT_NUMBER'."""
    merged_data = pd.merge(gps_data, shipments, on='SHIPMENT_NUMBER', how='left')
    return merged_data

def filter_data(merged_data, start_date, end_date):
    """Filter merged data for the specified period."""
    merged_data = merged_data[(merged_data['RECORD_TIMESTAMP'] >= start_date) & (merged_data['RECORD_TIMESTAMP'] <= end_date)]
    
    return merged_data

def fetch_distance_and_calculate_eta(origin, destination, average_speed_kmh):
    """Fetch distance from Google Maps and calculate ETA."""
    try:
        route_result = gmaps.distance_matrix(origin, destination, mode="driving")
        distance_meters = route_result['rows'][0]['elements'][0]['distance']['value']
        distance_km = distance_meters / 1000.0  # Convert meters to kilometers
        eta_hours = distance_km / average_speed_kmh  # Time = Distance / Speed
        return distance_km, eta_hours
    except Exception as e:
        print(f"Error with Google Maps API: {e}")
        return None, None

def process_shipments(merged_data, average_speed_kmh=60):
    """Process each shipment, calculate ETA, and check for delays."""

    print(merged_data.shape)

    results = []
    #for shipment_number, group in merged_data.groupby('SHIPMENT_NUMBER'):
    for shipment_number, group in tqdm(merged_data.groupby('SHIPMENT_NUMBER'), desc="Processing Shipments"):

        sorted_group = group.sort_values(by='RECORD_TIMESTAMP')
        half_index = len(sorted_group) // 2
        last_half_data = sorted_group.iloc[half_index:]

        total_points = len(last_half_data)

        ## take coarse information if we have a large history of loc and timestamp is available (for computation  efficiency reasons)
        if total_points > 1000:

            last_sev_index = len(sorted_group) // 5  # Determine the midpoint
            last_half_data = sorted_group.iloc[last_sev_index:]  # Take the last half of the data

            sample_rate = 10

        elif total_points > 500:
            last_sev_index = len(sorted_group) // 3  # Determine the midpoint
            last_half_data = sorted_group.iloc[last_sev_index:]  # Take the last half of the data

            sample_rate = 3

        elif total_points > 100:
            sample_rate = 2
        else:
            sample_rate = 1
        
        last_half_data = last_half_data.iloc[::sample_rate, :]
        
        for index, row in last_half_data.iterrows():
            origin = f"{row['LAT']},{row['LON']}"
            destination = f"{row['LAST_DELIVERY_LATITUDE']},{row['LAST_DELIVERY_LONGITUDE']}"
            distance, eta = fetch_distance_and_calculate_eta(origin, destination, average_speed_kmh)
            
            if eta is not None:
                eta_time = row['RECORD_TIMESTAMP'] + pd.Timedelta(hours=eta)
                delay = eta_time > row['LAST_DELIVERY_SCHEDULE_LATEST']
                delay_minutes = (eta_time - row['LAST_DELIVERY_SCHEDULE_LATEST']).total_seconds() / 60 if delay else 0
                results.append({
                    "SHIPMENT_NUMBER": shipment_number,
                    "SHIPPER_ID": row['SHIPPER_ID'] if 'SHIPPER_ID' in row else None,
                    "LAT": row['LAT'],
                    "LON": row['LON'],
                    "Distance (km)": distance,
                    "ETA (hours)": eta,
                    "ETA Time": eta_time,
                    "Scheduled Latest Time": row['LAST_DELIVERY_SCHEDULE_LATEST'],
                    "Delay": delay,
                    "Delay (minutes)": delay_minutes,
                    "Record Timestamp": row['RECORD_TIMESTAMP']
                })
    return pd.DataFrame(results)


def main():
    gps_data_path = './Data/gps_data.csv'
    shipments_data_path = './Data/Shipment_bookings.csv'
    
    gps_data, shipments = load_data(gps_data_path, shipments_data_path)

    delayed_data = pd.read_csv('Delayed_lastest_postions.csv')
    delayed_data_ids = delayed_data['SHIPMENT_NUMBER'].tolist()

    gps_data = gps_data[gps_data['SHIPMENT_NUMBER'].isin(delayed_data_ids)]

    gps_data, shipments = preprocess_data(gps_data, shipments)

    merged_data = merge_data(gps_data, shipments)
    
    
    start_date = pd.to_datetime('2023-10-01')
    end_date = pd.to_datetime('2023-12-31')
    filtered_data = filter_data(merged_data, start_date, end_date)
    
    results_df = process_shipments(filtered_data)
    delayed_shipments = results_df[results_df['Delay']]


    # saving the results
    #results_df.to_csv('processed_shipment_data.csv')
    
    # Identify delayed shipments
    delayed_shipments = results_df[results_df['Delay'] == True]

    # Group by shipper and shipment, and get the earliest detected delay and corresponding time
    earliest_delays = delayed_shipments.sort_values(by='Record Timestamp').groupby(['SHIPMENT_NUMBER']).first().reset_index()

    # Extract relevant information for notifications
    notifications = earliest_delays[['SHIPMENT_NUMBER', 'Record Timestamp', 'Delay (minutes)']]

    notifications.to_csv('./results/delayed_shipments_and_earliest_notification_time.csv')
    
    

if __name__ == "__main__":
    main()

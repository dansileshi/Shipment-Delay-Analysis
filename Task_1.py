import pandas as pd

def load_data(file_path):
    """
    Load CSV data from the specified file path.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pandas.DataFrame: Loaded data.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def filter_data(shipments, start_date, end_date):
    """
    Filter shipments within the specified date range.
    Args:
        shipments (pandas.DataFrame): DataFrame containing shipment data.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pandas.DataFrame: Filtered data.
    """
    condition = (shipments['LAST_DELIVERY_SCHEDULE_LATEST'] >= start_date) & \
                (shipments['LAST_DELIVERY_SCHEDULE_LATEST'] <= end_date)
    return shipments[condition]

def merge_and_prepare_data(gps_data, shipments):
    """
    Merge GPS data with shipments and prepare by sorting and converting to datetime.
    Args:
        gps_data (pandas.DataFrame): GPS tracking data.
        shipments (pandas.DataFrame): Shipment booking data.
    Returns:
        pandas.DataFrame: Merged and prepared data.
    """
    merged_data = pd.merge(gps_data, shipments, on='SHIPMENT_NUMBER')
    merged_data['RECORD_TIMESTAMP'] = pd.to_datetime(merged_data['RECORD_TIMESTAMP'], errors='coerce')
    merged_data['LAST_DELIVERY_SCHEDULE_LATEST'] = pd.to_datetime(merged_data['LAST_DELIVERY_SCHEDULE_LATEST'], errors='coerce')
    return merged_data.sort_values(by='RECORD_TIMESTAMP', ascending=False)

def calculate_delay_percentage(merged_data):
    """
    Calculate the percentage of shipments that were delayed beyond the delivery window plus a grace period.
    Args:
        merged_data (pandas.DataFrame): Merged DataFrame with shipments and GPS data.
    Returns:
        float: Percentage of delayed shipments.
    """
    latest_positions = merged_data.groupby('SHIPMENT_NUMBER').first().reset_index()
    latest_positions['is_delayed'] = latest_positions['RECORD_TIMESTAMP'] > \
                                     (latest_positions['LAST_DELIVERY_SCHEDULE_LATEST'] + pd.Timedelta(minutes=30))
    return latest_positions['is_delayed'].mean() * 100

def main():
    shipments = load_data('./Data/Shipment_bookings.csv')
    gps_data = load_data('./Data/gps_data.csv')
    
    filtered_shipments = filter_data(shipments, '2023-10-01', '2023-12-31')
    merged_data = merge_and_prepare_data(gps_data, filtered_shipments)
    
    delay_percentage = calculate_delay_percentage(merged_data)
    print(f"Percentage of delayed shipments: {delay_percentage:.2f}%")

if __name__ == "__main__":
    main()

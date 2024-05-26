import pandas as pd
import numpy as np
import logging
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def convert_bool_to_int(X):
    """ Convert boolean columns to integers """
    return X.astype(int)

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
    """ Load preprocessed data from a csv file """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise

def setup_preprocessing(numerical_features, categorical_features, binary_features):
    """ Set up the preprocessing pipeline """
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    
    binary_transformer = Pipeline([
        ('to_integer', FunctionTransformer(convert_bool_to_int)),
        ('imputer', SimpleImputer(strategy='most_frequent'))])


    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)])

    return preprocessor

def create_model_pipeline(preprocessor, model):
    """ Create a model training pipeline """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)])

def plot_roc_curve(y_true, y_scores):
    """ Plot ROC curve """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return plt

def save_plot(plot, version, directory='results'):
    """ Save the plot to a file """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f'roc_curve_v{version}.png')
    plot.savefig(filepath)
    logging.info(f"Plot saved to {filepath}")
    plot.close()

def save_model(model, version, directory='model_repository'):
    """ Save the trained model to disk """
    model_directory = os.path.join(directory, f"model_v{version}")
    os.makedirs(model_directory, exist_ok=True)
    
    model_path = os.path.join(model_directory, 'model_pipeline.joblib')
    dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

    with open(os.path.join(model_directory, 'model_metadata.txt'), 'w') as f:
        f.write(f"Model version: {version}\n")

def main():
    config = load_config()
    preprocessed_file_path = os.path.join(config['results_dir'], config['data_paths']['preprocessed_file'])
    model_version = config['versioning']['model_version']
    results_dir = config['results_dir']
    seed = config['seed']
    
    data = load_data(preprocessed_file_path)
    
    # Define feature columns
    binary_features = ['Ferry_Involved']
    categorical_features = ['VEHICLE_SIZE', 'VEHICLE_BUILD_UP', 'CARRIER_DISPLAY_ID',
                            'FIRST_COLLECTION_POST_CODE', 'LAST_DELIVERY_POST_CODE']
    numerical_features = ['FIRST_COLLECTION_LATITUDE', 'FIRST_COLLECTION_LONGITUDE', 'Step_Count',
                          'LAST_DELIVERY_LATITUDE', 'LAST_DELIVERY_LONGITUDE', 'Distance_km']
        
    # Define target column
    X = data.drop('is_delayed', axis=1)
    y = data['is_delayed']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    
    # Setup preprocessing
    preprocessor = setup_preprocessing(numerical_features, categorical_features, binary_features)
    
    # Setup model
    model = RandomForestClassifier(n_estimators=config['model_params']['random_forest']['n_estimators'],
                                   criterion=config['model_params']['random_forest']['criterion'],
                                   max_depth=config['model_params']['random_forest']['max_depth'],
                                   random_state=config['model_params']['random_forest']['random_state'])
    
    # Create model pipeline
    model_pipeline = create_model_pipeline(preprocessor, model)
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    logging.info("\n" + classification_report(y_test, y_pred))
    
    # Plot and save ROC curve
    plt = plot_roc_curve(y_test, y_pred_proba)
    save_plot(plt, model_version, results_dir)
    
    # Save the trained model
    save_model(model_pipeline, model_version)

if __name__ == "__main__":
    main()

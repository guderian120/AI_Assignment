import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import ssl
import os

# Bypass SSL verification for legacy/corporate networks
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    print("Attempting to download dataset...")
    # Load the latest version
    path = kagglehub.dataset_download("subhajournal/iotintrusion")
    
    print(f"Path to dataset files: {path}")
    
    # List files in the directory
    files = os.listdir(path)
    print(f"Files found: {files}")
    
    # Try loading the CSV file if found
    csv_file = [f for f in files if f.endswith('.csv')]
    if csv_file:
        df = pd.read_csv(os.path.join(path, csv_file[0]))
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst 5 records:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())
    else:
        print("No CSV file found in the dataset download path.")

except Exception as e:
    print(f"An error occurred: {e}")

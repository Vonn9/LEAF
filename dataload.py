
"""
File: dataload.py
Role: Data Loading Module
Description: 
    Loads the dataset from a JSON file and preprocesses it to create a DataFrame
"""
import pandas as pd
import json

def load_data(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df
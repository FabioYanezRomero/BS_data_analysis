"""
utils.py

Utility functions for data loading and processing.
"""
import pandas as pd
import json
from pathlib import Path


def load_csv_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: The loaded data
    """
    return pd.read_csv(file_path)


def load_csv_directory(directory_path, file_extension='.csv'):
    """
    Load all CSV files from a directory into a dictionary of DataFrames.
    
    Args:
        directory_path: Path to the directory containing CSV files
        file_extension: File extension to filter by
        
    Returns:
        dict: Mapping of filenames to DataFrames
    """
    directory = Path(directory_path)
    data_files = {}
    
    for file_path in directory.glob(f'*{file_extension}'):
        try:
            data_files[file_path.stem] = load_csv_data(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data_files


def save_json(data, file_path, ensure_ascii=False, indent=2):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        ensure_ascii: Whether to ensure ASCII encoding
        indent: Number of spaces for indentation
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def load_json(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        dict: Loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

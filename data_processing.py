# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from technical_analysis_lib import TecnicalAnalysis

def preprocess_data_backtrader(RUN, filename, scaler=None):
    """
    Preprocess the data by computing indicators and preparing raw features for Backtrader.
    Scaled features are not included and will be computed on-the-fly in the strategy.

    :param RUN: Configuration dictionary
    :param filename: Path to the CSV file containing the timeseries data
    :param scaler: Optional pre-fitted scaler. If None, a new scaler is fitted.
    :return: Tuple of (processed DataFrame, fitted scaler, feature columns)
    """
    # Load the dataset
    data = pd.read_csv(f"{RUN['folder']}{filename}")
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)

    # Compute technical indicators and patterns
    data = TecnicalAnalysis.compute_oscillators(data)
    data = TecnicalAnalysis.find_patterns(data)
    data = TecnicalAnalysis.add_timely_data(data)

    # Filter data within the backtest period
    data = data[(data['Date'] >= RUN['back_test_start']) & (data['Date'] <= RUN['back_test_end'])]
    if data.empty:
        raise ValueError("Filtered dataframe is empty.")

    # Replace infinities and drop missing values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Define columns to exclude from scaling
    exclude_cols = ['Date']  # 'label' is not included

    # Identify feature columns to scale (exclude non-numeric or non-feature columns)
    feature_cols = [col for col in data.columns if col not in exclude_cols and col != 'Asset_name']

    data_pred = data[feature_cols].copy()

    if data_pred.empty:
        raise ValueError("Dataframe after selecting feature columns is empty.")

    # Fit scaler if not provided
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data_pred)

    # Print feature columns for verification
    print("Feature columns used for scaling:", feature_cols)

    return data, scaler, feature_cols

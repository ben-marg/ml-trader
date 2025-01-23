import pickle
import os 
from config import RUN as run_conf
import sys
import pandas as pd 
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
import numpy as np
from sklearn.preprocessing import StandardScaler



full_csv = os.path.join(run_conf['folder'], 'BTCUSDT.csv')
try:
    data = pd.read_csv(full_csv)
except Exception as ex:
    print("Error loading CSV data:", ex)
    sys.exit(1)
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

data = TecnicalAnalysis.compute_oscillators(data)
data = TecnicalAnalysis.find_patterns(data)
data = TecnicalAnalysis.add_timely_data(data)

train_start_dt = pd.to_datetime(run_conf['train_start'])
train_end_dt = pd.to_datetime(run_conf['train_end'])
backtest_start_dt = pd.to_datetime(run_conf['back_test_start'])
backtest_end_dt = pd.to_datetime(run_conf['back_test_end'])

train_mask = (data['Date'] >= train_start_dt) & (data['Date'] <= train_end_dt)
backtest_mask = (data['Date'] >= backtest_start_dt) & (data['Date'] <= backtest_end_dt)
data_train = data[train_mask].copy()
data_backtest = data[backtest_mask].copy()
cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
data_train_features = data_train.drop(columns=cols_to_drop, errors='ignore')
data_train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
data_train_features.dropna(inplace=True)

scaler = StandardScaler()

# Fit the scaler
scaler.fit(data_train_features)
print(data_train_features)
'''
# Save the scaler to a pickle file
scaler_path = 'scaler.pkl'
try:
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
except Exception as ex:
    print(f"Error saving scaler to {scaler_path}: {ex}")
    sys.exit(1)

# -----------------------------------------------
# 3. Generate NN Predictions for Backtest Data
# -----------------------------------------------
# Load the scaler from the pickle file
try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")
except Exception as ex:
    print(f"Error loading scaler from {scaler_path}: {ex}")
    sys.exit(1)

# Apply the scaler to the backtest features
data_backtest_features = data_backtest.drop(columns=cols_to_drop, errors='ignore')
data_backtest_features.replace([np.inf, -np.inf], np.nan, inplace=True)
data_backtest_features.fillna(method='ffill', inplace=True)
data_backtest_features.fillna(method='bfill', inplace=True)

X_backtest = scaler.transform(data_backtest_features)
'''
import ccxt
import time
import pandas as pd
import numpy as np
import datetime as dt
import traceback, sys
import logging
from copy import deepcopy
from config import RUN as run_conf
from NNModel_lib import NNModel  
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from sklearn.preprocessing import StandardScaler
import compute_indicators_labels_lib  # for get_dataset and feature functions
import json


# ----------------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------------
logging.basicConfig(
    filename="live_trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# ----------------------------------------------------------------------------
# Configuration Setup
# ----------------------------------------------------------------------------
# For live trading, we use a copy of your RUN configuration.
live_conf = deepcopy(run_conf)
live_conf['b_window'] = 2   # backward/look-back window (same as used in training/backtest)
live_conf['f_window'] = 2   # forward window (forecast horizon)
# Open and load the JSON file
with open('keys.json', 'r') as file:
    credentials = json.load(file)
# Binance API credentials

binance_api_key = live_conf.get('binance_api_key', credentials.get("binance-api-key"))
binance_secret  = live_conf.get('binance_secret', credentials.get("secret"))
# (trade_amount is no longer a fixed value; see compute_order_size below.)
# Model file (should match the naming/format you used during training)
model_file = live_conf.get("model_file", "./models/model.h5")

# ----------------------------------------------------------------------------
# Connect to Binance via CCXT
# ----------------------------------------------------------------------------
exchange = ccxt.binance({
    'apiKey': binance_api_key,
    'secret': binance_secret,
    'enableRateLimit': True,
})

# ----------------------------------------------------------------------------
# Prepare the Feature Scaler
# ----------------------------------------------------------------------------
try:
    scaler = StandardScaler()
    dataset = compute_indicators_labels_lib.get_dataset(live_conf)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.dropna()
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset = dataset[dataset['pct_change'] < live_conf['beta']]  # remove outliers
    print(dataset['label'].value_counts())
    labels = dataset['label'].copy()
    dataset.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', "Asset_name", "label"], inplace=True)
    columns = dataset.columns
    index = dataset.index
    X_scaler = scaler.fit_transform(dataset.values)
    dataset = pd.DataFrame(X_scaler, columns=columns, index=index)
    dataset['label'] = labels
    training_feature_cols = list(columns)
    logging.info("Scaler fitted on historical dataset. Training feature columns: %s", training_feature_cols)
except Exception as e:
    logging.error(f"Error fitting scaler: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------------
# Load the Trained Neural Network Model
# ----------------------------------------------------------------------------
input_dim = X_scaler.shape[1]
model_instance = NNModel(input_dim, 3)
try:
    # For dummy training, split a bit of historical data:
    from sklearn.model_selection import train_test_split
    X_total = scaler.transform(dataset.iloc[:, :-1])
    y_total = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, random_state=42)
    model_instance.dummy_train(X_train, y_train)
    # Now load your pre-trained model weights
    model_instance.load(model_file)
    logging.info(f"Model loaded from {model_file}.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------------
# Trading State Variables
# ----------------------------------------------------------------------------
# Track current position side ('long' or 'short'), current position size (in base asset units),
# and the capital percentage allocated (each order uses 2%, maximum allocation is 20%)
position = None            # None, 'long', or 'short'
contract_amount = 0.0      # total contracts / amount in current position
allocated_percent = 0.0    # in percent (each order adds 2% and maximum allowed is 20%)

# ----------------------------------------------------------------------------
# Data Fetching & Processing Functions
# ----------------------------------------------------------------------------
def fetch_live_data(symbol='BTCUSDT', timeframe='1m', limit=100):
    """
    Fetch recent OHLCV data from Binance.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.drop(columns=['timestamp'], inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching live data: {e}")
        return None

def process_live_data(df):
    """
    Process live OHLCV data to match the training feature set.
    """
    try:
        df_proc = df.copy()
        df_proc = TecnicalAnalysis.compute_oscillators(df_proc)
        df_proc = TecnicalAnalysis.find_patterns(df_proc)
        df_proc = TecnicalAnalysis.add_timely_data(df_proc)
        df_proc.set_index('Date', inplace=True)
        df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_proc.dropna(inplace=True)
        df_features = df_proc.copy()
        cols_to_drop = ['High', 'Low', 'Close', 'Volume', 'Asset_name', 'label']
        for col in cols_to_drop:
            if col in df_features.columns:
                df_features = df_features.drop(columns=[col])
        live_feature_cols = list(df_features.columns)
        logging.info("Live feature columns before adjustment: %s", live_feature_cols)
        extra_cols = [col for col in live_feature_cols if col not in training_feature_cols]
        if extra_cols:
            logging.info("Dropping extra columns from live data: %s", extra_cols)
            df_features = df_features.drop(columns=extra_cols)
        missing_cols = [col for col in training_feature_cols if col not in df_features.columns]
        if missing_cols:
            logging.warning("Live data is missing expected training columns: %s", missing_cols)
        logging.info("Final live feature columns: %s", list(df_features.columns))
        return df_proc, df_features
    except Exception as e:
        logging.error(f"Error processing live data: {e}")
        return None, None

# ----------------------------------------------------------------------------
# Capital & Order Size Calculation
# ----------------------------------------------------------------------------
def compute_order_size(symbol='BTCUSDT'):
    """
    Calculate the order size based on 2% of the total available capital in USDT.
    Assumes that the balance is denominated in USDT.
    """
    try:
        balance = exchange.fetch_balance()
        # Depending on your account structure, adjust where you look up your total balance.
        total_capital = balance['total']['USDT']  # total USDT balance
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        # Order value = 2% of total capital. Then compute order size (in base asset units).
        order_value = total_capital * 0.02
        order_amount = order_value / current_price
        return order_amount, total_capital, current_price
    except Exception as e:
        logging.error(f"Error computing order size: {e}")
        return None, None, None

# ----------------------------------------------------------------------------
# Trading Decision Function
# ----------------------------------------------------------------------------
def decide_trade(predicted_label, current_position, allocated_percent):
    """
    Decision logic:
      - If the signal is BUY:
            * If no position, then open a long ('buy').
            * If already long and allocated capital is less than 20%, then add to long ('buy').
            * If in a short position, then reverse to long ('reverse_to_long').
      - If the signal is SELL:
            * If no position, then open a short ('sell').
            * If already short and allocated capital is less than 20%, then add to short ('sell').
            * If in a long position, then reverse to short ('reverse_to_short').
      - Otherwise, hold.
    """
    if predicted_label == BUY:
        if current_position is None:
            return 'buy'
        elif current_position == 'long':
            if allocated_percent < 20:
                return 'buy'  # add to long position (scale in)
            else:
                return 'hold'
        elif current_position == 'short':
            return 'reverse_to_long'
    elif predicted_label == SELL:
        if current_position is None:
            return 'sell'
        elif current_position == 'short':
            if allocated_percent < 20:
                return 'sell'  # add to short position (scale in)
            else:
                return 'hold'
        elif current_position == 'long':
            return 'reverse_to_short'
    return 'hold'

# ----------------------------------------------------------------------------
# Trade Execution Function
# ----------------------------------------------------------------------------
def execute_trade(action, symbol='BTCUSDT'):
    """
    Execute the specified action. Actions:
      - 'buy': Open or add to a long position.
      - 'sell': Open or add to a short position.
      - 'reverse_to_long': Reverse a short position to long.
      - 'reverse_to_short': Reverse a long position to short.
    Each order uses 2% of total capital. If adding, ensure total allocated capital does not exceed 20%.
    """
    global position, allocated_percent, contract_amount

    try:
        order_amount, total_capital, current_price = compute_order_size(symbol)
        if order_amount is None:
            return None

        # The capital allocation per order is fixed at 2% (i.e. allocated_percent increases by 2 with each order)
        if action == 'buy':
            order = exchange.create_market_buy_order(symbol, order_amount)
            logging.info(f"Executed BUY order for {order_amount} {symbol} (long)")
            if position is None:
                position = 'long'
            allocated_percent += 2
            contract_amount += order_amount
            return order

        elif action == 'sell':
            order = exchange.create_market_sell_order(symbol, order_amount)
            logging.info(f"Executed SELL order for {order_amount} {symbol} (short)")
            if position is None:
                position = 'short'
            allocated_percent += 2
            contract_amount += order_amount
            return order

        elif action == 'reverse_to_long':
            # Close the short position first
            close_order = exchange.create_market_buy_order(symbol, contract_amount)
            logging.info(f"Closed SHORT position by buying {contract_amount} {symbol}")
            # Reset allocated capital and position variables
            allocated_percent = 0
            contract_amount = 0
            # Now open a long order
            new_order = exchange.create_market_buy_order(symbol, order_amount)
            logging.info(f"Reversed to LONG: executed BUY order for {order_amount} {symbol}")
            position = 'long'
            allocated_percent = 2
            contract_amount = order_amount
            return new_order

        elif action == 'reverse_to_short':
            # Close the long position first
            close_order = exchange.create_market_sell_order(symbol, contract_amount)
            logging.info(f"Closed LONG position by selling {contract_amount} {symbol}")
            allocated_percent = 0
            contract_amount = 0
            # Now open a short order
            new_order = exchange.create_market_sell_order(symbol, order_amount)
            logging.info(f"Reversed to SHORT: executed SELL order for {order_amount} {symbol}")
            position = 'short'
            allocated_percent = 2
            contract_amount = order_amount
            return new_order

        else:
            logging.info("Hold decision: no trade executed.")
            return None

    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None

# ----------------------------------------------------------------------------
# Main Live Trading Loop
# ----------------------------------------------------------------------------
def live_trading_loop():
    global position, allocated_percent, contract_amount

    symbol = 'BTCUSDT'
    timeframe = '4h'
    fetch_limit = 100  # number of recent candles

    while True:
        try:
            # 1. Fetch live data
            df_live = fetch_live_data(symbol, timeframe, fetch_limit)
            if df_live is None or df_live.empty:
                logging.warning("No live data fetched; skipping cycle.")
                time.sleep(60)
                continue

            # 2. Process the live data
            full_data, df_features = process_live_data(df_live)
            if df_features is None or df_features.empty:
                logging.warning("No processed data available; skipping cycle.")
                time.sleep(60)
                continue

            # 3. Scale the features for prediction
            X_live = scaler.transform(df_features)
            X_live_input = X_live[-1].reshape(1, -1)

            # 4. Predict using your NN model
            predicted_label = model_instance.predict(X_live_input)
            logging.info(f"Model predicted label: {predicted_label}")

            # 5. Decide trade action using the new logic
            action = decide_trade(predicted_label, position, allocated_percent)
            logging.info(f"Decided action: {action}")

            # 6. Execute trade if required
            if action in ['buy', 'sell', 'reverse_to_long', 'reverse_to_short']:
                order = execute_trade(action, symbol)
                if order is not None:
                    logging.info(f"Order executed: {order}")
            else:
                logging.info("Holding current position.")

            # Log current price and position state for reference
            current_price = float(df_live['Close'].iloc[-1])
            logging.info(f"Current price: {current_price}, Position: {position}, Allocated Capital: {allocated_percent}%")
            time.sleep(60)

        except Exception as e:
            logging.error(f"Exception in trading loop: {e}")
            traceback.print_exc(file=sys.stdout)
            time.sleep(60)

# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Starting live trading bot...")
    live_trading_loop()

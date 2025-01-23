import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
import backtrader as bt

# These imports assume you have these modules available.
# They can be your custom implementations.
from ccxtbt import CCXTFeed, CCXTStore
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel

import pickle
from sklearn.preprocessing import StandardScaler
import logging

# Create a logger
logger = logging.getLogger('my_logger')

# Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler('app.log')

# Create a console handler to display logs on the console (optional)

# Set the logging level for the handlers
file_handler.setLevel(logging.DEBUG)

# Create a formatter for consistent log formatting
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Attach the formatter to the handlers
file_handler.setFormatter(formatter)

# Attach the handlers to the logger
logger.addHandler(file_handler)

STOP_LOSS = 0.05
# Define your signal labels (adjust as needed)
BUY_SIGNAL = 1
HOLD_SIGNAL = 0
SELL_SIGNAL = -1

# Load API keys
with open('keys.json', 'r') as file:
    credentials = json.load(file)

binance_config = {
    'apiKey': credentials.get("testnet-key"),
    'secret': credentials.get("testnet-secret"),
    'enableRateLimit': True,
    'test': True,
    'options': {'defaultType': 'future'},
    'urls': {
        'api': {
            'public': 'https://testnet.binancefuture.com/fapi/v1',
            'private': 'https://testnet.binancefuture.com/fapi/v1',
        }
    }
}

from ccxtbt import CCXTFeed
import time

from binance.um_futures import UMFutures


client = UMFutures(binance_config.get('apiKey'), binance_config.get('secret'), base_url = 'https://testnet.binancefuture.com' )

response = client.balance(recvWindow=6000)

logging.info(response)
usdt = next((item for item in response if item.get('asset') == 'USDT'), None)
usdt_balance = usdt.get('balance')
if usdt_balance:
    print("USDT Balance:")
    print(usdt_balance)
else:
    print("USDT balance not found in the response.")
class CustomDynamicSinceCCXTFeed(CCXTFeed):
    """
    A custom feed that updates the 'since' parameter after each successful fetch.
    If no new data is returned, it will wait briefly before the next fetch.
    """
    def _load(self):
        # Call the parent _load() method to perform the actual fetching
        ohlcv = super()._load()

        # Check if we got valid data
        if not ohlcv or not isinstance(ohlcv, list) or len(ohlcv) == 0:
            # If no new data, optionally sleep for a short time to avoid constant re-fetching
            # (For example, wait 5 seconds before the next attempt.)
            time.sleep(1)
            return ohlcv  # Return empty or False; the feed logic should then wait until next call

        try:
            # Ensure that the last candle is a list (i.e. a valid candle)
            if isinstance(ohlcv[-1], list):
                # Get the timestamp of the last candle and add a small delta (e.g. 1 millisecond)
                last_ts = ohlcv[-1][0] + 1  
                # Update the numeric "since" parameter so the next fetch uses this new timestamp.
                self.p.since = last_ts
                # Optionally update a string date too if your feed uses it.
                self.p.fromdate = last_ts
                if self.store.debug:
                    print(f"Updated 'since' to: {self.p.since}")
        except Exception as ex:
            print(f"Error updating 'since': {ex}")
        
        return ohlcv



# Define Live Strategy with Model-Based Labeling
# ======================================================================
class LiveNNStrategy(bt.Strategy):
    params = (
        ('stop_loss', STOP_LOSS),
        ('window', 5),  # Number of bars to accumulate before feature computation
    )


    def __init__(self):
        self.order = None
        self.entryprice = None
        # Initialize an empty list to store incoming bar data.
        self.live_data = []
        self.last_bar_time = None

        self.position_size = 0.05
        self.leverage = 19
        self.stop_loss = 0.05 

        # Load the scaler (assumed to be saved as scaler.pkl)
        try:
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as ex:
            logger.info(f"Error loading scaler: {ex}")
            sys.exit(1)
        
        # Specify the number of features your model expects.
        # (Make sure this value matches what you used during training.)
        self.num_features = 36  # <--- adjust if necessary

        # Load your trained NN model.
        try:
            self.model = NNModel(self.num_features, 3)  # assuming 3 classes: BUY, HOLD, SELL
            self.model.load('model.h5')
        except Exception as ex:
            logger.info(f"Error loading NN model: {ex}")
            traceback.print_exc()
            sys.exit(1)

    def next(self):
    # Get the current bar's datetime.
        current_time = self.data.datetime.datetime(0)
        if self.last_bar_time is not None and current_time <= self.last_bar_time:
            logger.info(f"Duplicate or outdated bar at {current_time}, last_bar_time: {self.last_bar_time}")
            return  # Skip processing if this is not a new bar.
        
        self.last_bar_time = current_time
        logger.info(f"Processing new bar at {current_time}")

        # Create a bar dictionary with relevant data.
        bar = {
            'Date': self.data.datetime.datetime(0),
            'Open': self.data.open[0],
            'High': self.data.high[0],
            'Low': self.data.low[0],
            'Close': self.data.close[0],
            'Volume': self.data.volume[0]
        }
        
        # Append the new bar to the live_data list.
        self.live_data.append(bar)
        
        # Maintain the sliding window: keep only the last 'window' bars.
        if len(self.live_data) > self.params.window:
            self.live_data = self.live_data[-self.params.window:]
        
        logger.info(f"Sliding window size after update: {len(self.live_data)}")

        # Only proceed if the sliding window is full
        if len(self.live_data) < self.params.window:
            logger.info("Insufficient data to compute features; waiting for more bars")
            return

        # Convert the live data list to a DataFrame.
        df = pd.DataFrame(self.live_data)
        
        # --- Compute Technical Indicators ---
        try:
            df = TecnicalAnalysis.compute_oscillators(df)
            df = TecnicalAnalysis.find_patterns(df)
            df = TecnicalAnalysis.add_timely_data(df)
        except Exception as ex:
            logger.info(f"Error computing technical indicators: {ex}")
            return

        # --- Prepare Features for the Model ---
        # Drop columns that were not used during training.
        cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
        feature_df = df.drop(columns=cols_to_drop, errors='ignore')

        # Clean the features: handle infinities and NaN values.
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_df.ffill()

        # Get the feature vector from the current (latest) bar.
        try:
            features = feature_df.iloc[-1].values.reshape(1, -1)
            logger.info(features)
        except Exception as ex:
            logger.info(f"Error preparing feature vector: {ex}")
            return

        # Scale the features.
        try:
            scaled_features = self.scaler.transform(features)
            logger.info(f"Scaled features: {scaled_features}")
        except Exception as ex:
            logger.info(f"Error scaling feature vector: {ex}")
            return

        # --- Get Model Prediction (Label) ---
        try:
            prediction = self.model.predict(scaled_features)
            logger.info(f"Model prediction: {prediction}")

            # Assume the model returns a numpy array; pick the first (and only) prediction.
            label = prediction[0]
        except Exception as ex:
            logger.info(f"Error during model prediction: {ex}")
            return



        # --- Decision Logic Based on Model Label ---
        # If an order is still pending, do nothing.
        current_price = self.data.close[0]
        balance = self.broker.get_balance()
        print(balance)
        portfolio_value =  self.broker._value 
        print(portfolio_value)
        input()

        full_size = (portfolio_value * self.leverage) / current_price
        # Optionally apply a scaling factor:
        full_size = full_size * self.position_size

        logger.info(f"Calculated order size: {full_size:.4f} (Portfolio: {portfolio_value:.2f}, Leverage: {self.leverage})")

        #testing order execution 
        self.order = self.buy(exectype=bt.Order.Market, size=full_size)
        self.entryprice = current_price
        logger.info(f"ENTER LONG (Leveraged) {full_size:.4f} at {current_price:.2f}")
        input()

        if self.order:
            return
        
        if not self.position:
            # Execute a BUY order if model signals BUY.
            if label == BUY_SIGNAL:
                self.order = self.buy()
                self.entryprice = self.data.close[0]
                logger.info(f"BUY signal (model label {label}) at {self.entryprice:.2f}")
        else:
            # If the model signals SELL, exit the position.
            if label == SELL_SIGNAL:
                self.order = self.sell()
                logger.info(f"SELL signal (model label {label}) at {self.data.close[0]:.2f}")
            # Check stop loss conditions.
            elif self.data.low[0] < self.entryprice * (1 - self.params.stop_loss):
                logger.info(f"STOP LOSS HIT at {self.data.low[0]:.2f}")
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"BUY EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size}")
                
            elif order.issell():
                logger.info(f"SELL EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.info(f"Order Canceled/Margin/Rejected: {order.status}")
        # Reset order to allow new orders.
        self.order = None

# ======================================================================
# Run Live Trading
# ======================================================================
def run_live_trading():
    try:
        # Initialize CCXTStore. (If you run against testnet, consider using a custom store that mocks the balance.)
        store = CCXTStore(exchange='binanceusdm', currency='USDT', config=binance_config, retries=1, debug=True)
        # (Ensure BrokerCls is set)
        store.__class__.BrokerCls = store.__class__.BrokerCls or getattr(store, 'BrokerCls', None)
        if store.__class__.BrokerCls is None:
            from ccxtbt import CCXTBroker
            store.__class__.BrokerCls = CCXTBroker

        # Create the live data feed using the custom feed class.
        data = CustomDynamicSinceCCXTFeed(
            dataname='BTC/USDT',
            exchange='binance',
            currency='USDT',
            timeframe=bt.TimeFrame.Minutes,    # or '1m' if appropriate
            compression=240,                     # for 1m candles; adjust compression if needed
            ohlcv_limit=100,
            config=binance_config,
            retries=5,
            drop_newest=False,
            # You may remove the fromdate parameter here so that the feed begins from "now"
        )

        # Set up Cerebro
        cerebro = bt.Cerebro()
        cerebro.adddata(data)
        cerebro.addstrategy(LiveNNStrategy)
        broker = store.getbroker()
        cerebro.setbroker(broker)
    
        cerebro.run()

    except Exception as ex:
        print(f"Error in live trading: {ex}")
        traceback.print_exc()

if __name__ == "__main__":
    run_live_trading()



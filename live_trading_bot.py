#!/usr/bin/env python
import os
import sys
import json
import traceback
import time
import numpy as np
import pandas as pd
import backtrader as bt
import ccxt
import logging
import pickle
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Logging Configuration
# ------------------------------
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ------------------------------
# Strategy Signal Labels and Parameters
# ------------------------------
STOP_LOSS = 0.05
BUY_SIGNAL = 1
HOLD_SIGNAL = 0
SELL_SIGNAL = -1

# ------------------------------
# Load API Keys from keys.json
# ------------------------------
with open('keys.json', 'r') as file:
    credentials = json.load(file)

# ------------------------------
# Create a CCXT Binance Exchange Instance (Testnet)
# ------------------------------
binance_config = {
    'apiKey': credentials.get("testnet-key"),
    'secret': credentials.get("testnet-secret"),
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'test': True,
}

exchange = ccxt.binance(binance_config)
exchange.set_sandbox_mode(True)

# ------------------------------
# Example: Fetch and Print USDT Balance Using CCXT
# ------------------------------
try:
    balance = exchange.fetch_balance()
    usdt_balance = balance.get('USDT', {})
    if usdt_balance:
        print("USDT Balance:")
        print(usdt_balance)  # e.g., {'free': 15000.0, 'used': 0.0, 'total': 15000.0}
    
    else:
        print("USDT balance not found in the response.")
except Exception as e:
    logger.error("Error fetching balance: %s", e)

# ------------------------------
# Custom CCXT Data Feed for Backtrader
# ------------------------------
class CustomCCXTFeed(bt.feed.DataBase):
    """
    A custom live data feed that uses CCXT to fetch OHLCV data from Binance Testnet.
    """
    params = (
        ('exchange', None),         # a ccxt exchange instance
        ('symbol', 'BTC/USDT'),
        ('timeframe', bt.TimeFrame.Minutes),  # Backtrader's timeframe (integer constant)
        ('compression', 240),         # 1-minute candles
        ('since', None),            # starting timestamp in ms (if None, computed)
        ('limit', 100),             # number of candles to fetch per call
        ('sleep', 1),               # seconds to sleep if no new data
    )

    # Mapping from Backtrader timeframe constants to CCXT timeframe strings
    TIMEFRAME_MAP = {
        bt.TimeFrame.Seconds: '1s',
        bt.TimeFrame.Minutes: '1m',
        bt.TimeFrame.Days: '1d',
        bt.TimeFrame.Weeks: '1w',
        bt.TimeFrame.Months: '1M'
    }

    def __init__(self):
        super().__init__()
        if self.p.since is None:
            # Set "since" to a little before now
            self.p.since = self.p.exchange.milliseconds() - self.p.limit * 60 * 1000
        self._data_buffer = []

    def _load(self):
        # If buffer is empty, fetch new OHLCV data
        if not self._data_buffer:
            try:
                # Convert Backtrader timeframe to CCXT timeframe string
                ccxt_timeframe = self.TIMEFRAME_MAP.get(self.p.timeframe, '4h')
                ohlcv = self.p.exchange.fetch_ohlcv(
                    self.p.symbol,
                    timeframe=ccxt_timeframe,
                    since=self.p.since,
                    limit=self.p.limit
                )
                if ohlcv:
                    self.p.since = ohlcv[-1][0] + 1
                    self._data_buffer.extend(ohlcv)
                else:
                    time.sleep(self.p.sleep)
                    return None
            except Exception as e:
                logger.error("Error fetching data from CCXT: %s", e)
                time.sleep(self.p.sleep)
                return None

        if self._data_buffer:
            # Pop one candle from the buffer:
            candle = self._data_buffer.pop(0)
            # CCXT returns: [timestamp, open, high, low, close, volume]
            # Convert timestamp (in ms) to a float datetime for backtrader:
            dt = bt.date2num(pd.to_datetime(candle[0], unit='ms'))
            self.lines.datetime[0] = dt
            self.lines.open[0] = candle[1]
            self.lines.high[0] = candle[2]
            self.lines.low[0] = candle[3]
            self.lines.close[0] = candle[4]
            self.lines.volume[0] = candle[5]
            return True
        return False

# ------------------------------
# Live Trading Strategy with Model-Based Labeling
# ------------------------------
# (Ensure that technical_analysis_lib and NNModel_lib are available in your PYTHONPATH)
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel

class LiveNNStrategy(bt.Strategy):
    params = (
        ('stop_loss', STOP_LOSS),
        ('window', 5),  # Number of bars to accumulate before feature computation
    )

    def __init__(self):
        self.order = None
        self.entryprice = None
        self.live_data = []
        self.last_bar_time = None

        self.position_size = 0.05
        self.leverage = 10
        self.stop_loss = 0.05 

        # Load the scaler (assumed to be saved as scaler.pkl)
        try:
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as ex:
            logger.info(f"Error loading scaler: {ex}")
            sys.exit(1)
        
        self.num_features = 36  # Adjust as necessary

        # Load the neural network model (NNModel)
        try:
            self.model = NNModel(self.num_features, 3)  # assuming 3 classes: BUY, HOLD, SELL
            self.model.load('model.h5')
        except Exception as ex:
            logger.info(f"Error loading NN model: {ex}")
            traceback.print_exc()
            sys.exit(1)

    def next(self):
        # Avoid processing a new bar if there is a pending order.
        if self.order:
            logger.info("Pending order exists; skipping new order placement.")
            return

        current_time = self.data.datetime.datetime(0)
        if self.last_bar_time is not None and current_time <= self.last_bar_time:
            logger.info(f"Duplicate or outdated bar at {current_time}; last_bar_time: {self.last_bar_time}")
            return
        self.last_bar_time = current_time
        logger.info(f"Processing new bar at {current_time}")

        # Accumulate live data
        bar = {
            'Date': current_time,
            'Open': self.data.open[0],
            'High': self.data.high[0],
            'Low': self.data.low[0],
            'Close': self.data.close[0],
            'Volume': self.data.volume[0]
        }
        self.live_data.append(bar)
        if len(self.live_data) > self.params.window:
            self.live_data = self.live_data[-self.params.window:]
        logger.info(f"Sliding window size after update: {len(self.live_data)}")
        if len(self.live_data) < self.params.window:
            logger.info("Insufficient data to compute features; waiting for more bars")
            return

        # Prepare features and predict model label
        df = pd.DataFrame(self.live_data)
        try:
            df = TecnicalAnalysis.compute_oscillators(df)
            df = TecnicalAnalysis.find_patterns(df)
            df = TecnicalAnalysis.add_timely_data(df)
        except Exception as ex:
            logger.info(f"Error computing technical indicators: {ex}")
            return

        cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
        feature_df = df.drop(columns=cols_to_drop, errors='ignore')
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_df.ffill()

        try:
            features = feature_df.iloc[-1].values.reshape(1, -1)
            logger.info(f"Features: {features}")
        except Exception as ex:
            logger.info(f"Error preparing feature vector: {ex}")
            return

        try:
            scaled_features = self.scaler.transform(features)
            logger.info(f"Scaled features: {scaled_features}")
        except Exception as ex:
            logger.info(f"Error scaling feature vector: {ex}")
            return

        try:
            prediction = self.model.predict(scaled_features)
            logger.info(f"Model prediction: {prediction}")
            label = prediction[0]
        except Exception as ex:
            logger.info(f"Error during model prediction: {ex}")
            return

        # Calculate order size and check execution conditions
        current_price = self.data.close[0]
        portfolio_value = self.broker.getvalue()
        logger.info(f"Portfolio Value: {portfolio_value}")
        free_cash = self.broker.get_cash()
        logger.info(f"Free cash: {free_cash}")

        full_size = (portfolio_value * self.leverage) / current_price
        full_size *= self.position_size
        logger.info(f"Calculated order size: {full_size:.4f} (Portfolio: {portfolio_value:.2f}, Leverage: {self.leverage})")
        cash_required = full_size * current_price
        logger.info(f"Cash required: {cash_required}")
        # Place a market order (this will now only be called if no order is pending)
        try:
            if (free_cash > cash_required):
                self.order = self.buy(exectype=bt.Order.Market, size=full_size)
                self.entryprice = current_price
                logger.info(f"ENTER LONG (Leveraged) {full_size:.4f} at {current_price:.2f}")
            else:
                logger.info("Insufficient cash to execute the order")
            
        except Exception as e:
            logger.error(f"Order submission error: {e}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"BUY EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size}")
            elif order.issell():
                logger.info(f"SELL EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size}")
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.info(f"Order Canceled/Margin/Rejected: {order.status}")
            print(order)
            input()
            self.order = None
        # Handle custom "Expired" status if it occurs (assume status value 7 means expired)
        elif hasattr(order, 'status') and order.status == 7:
            logger.info("Order expired (status 7). Resetting order state.")
            self.order = None
    def stop(self):
        print("end of strategy")
        
    

# ------------------------------
# Run Live Trading with CCXT Data Feed
# ------------------------------
def run_live_trading():
    try:
        cerebro = bt.Cerebro()
        # Use a 1-minute timeframe with compression 1 for the CCXT feed.
        data = CustomCCXTFeed(
            exchange=exchange,
            symbol='BTC/USDT',
            timeframe=bt.TimeFrame.Minutes,
            compression=240,
            limit=100
        )
        cerebro.adddata(data)
        cerebro.addstrategy(LiveNNStrategy)
        
        free_balance = usdt_balance.get('free')
        print(free_balance)
        cerebro.broker.setcash(free_balance)  # starting cash
        logger.info("Starting live trading...")
        cerebro.run()
        cerebro.plot()
    except Exception as ex:
        print(f"Error in live trading: {ex}")
        traceback.print_exc()

if __name__ == "__main__":
    run_live_trading()

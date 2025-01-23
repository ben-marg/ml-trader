import backtrader as bt
from ccxtbt import CCXTStore
import ccxt
import json
import time
import logging
import datetime
import sys
import pickle
import traceback
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

STOP_LOSS = 0.05
WINDOW = 5 
# ---------------------------------------------------------------------
# Load Credentials
# ---------------------------------------------------------------------
with open('keys.json', 'r') as f:
    credentials = json.load(f)

# ---------------------------------------------------------------------
# Updated Configuration for Binance Futures Testnet
# ---------------------------------------------------------------------
binanceusdm_testnet_config = {
    'apiKey': credentials['testnet-key'],       # Use Futures Testnet key
    'secret': credentials['testnet-secret'],
    'test': True,                               # Force test environment
    'enableRateLimit': True,
    'adjustForTimeDifference': True,
    'options': {
        'defaultType': 'future',
        'sandbox': True,                        # Enforce sandbox mode
    },
    'has': {
        'fetchCurrencies': False,
        'fetchBalance': True,
    },
    'urls': {
        'api': {
            'public':  'https://testnet.binancefuture.com/fapi/v1',
            'private': 'https://testnet.binancefuture.com/fapi/v1',
        },
    },
}

# 1) Instantiate ccxt with the binanceusdm class:
exchange = ccxt.binanceusdm(binanceusdm_testnet_config)
exchange.set_sandbox_mode(True)

# 2) Create the CCXTStore using the same 'binanceusdm' name:
store = CCXTStore(
    exchange='binanceusdm',
    currency='USDT',
    config=binanceusdm_testnet_config,
    retries=5,
    debug=False
)

# ---------------------------------------------------------------------
# Custom Data Feed
# ---------------------------------------------------------------------
class CustomCCXTFeed(bt.feed.DataBase):
    params = (
        ('exchange', None),
        ('symbol', 'BTC/USDT'),
        ('dataname', None),
        ('timeframe', bt.TimeFrame.Minutes),
        # Set compression to 240 so that each candle represents 4 hours
        ('compression', 240),
        ('limit', 100),
        ('sleep', 30),
        ('since', None),
    )
    
    TIMEFRAME_MAP = {
        bt.TimeFrame.Minutes: '1m',
        bt.TimeFrame.Days: '1d',
    }
    
    def __init__(self):
        super().__init__()
        self._data_buffer = []  # Initialize the buffer
        if not self.p.dataname:
            self.p.dataname = self.p.symbol
        if self.p.since is None:
            self.p.since = int(time.time() * 1000) - (self.p.limit * 60 * 1000)
        logger.info(f"CustomCCXTFeed initialized, timeframe map = {self.TIMEFRAME_MAP}")
    
    def _load(self):
        if not self._data_buffer:
            try:
                # Use the timeframe mapping to determine the correct string for the exchange call
                ccxt_timeframe = self.TIMEFRAME_MAP.get(self.p.timeframe, '1m')
                logger.info(f"Fetching OHLCV: symbol={self.p.symbol}, tf={ccxt_timeframe}, since={self.p.since}")
                ohlcv = self.p.exchange.fetch_ohlcv(
                    self.p.symbol,
                    timeframe=ccxt_timeframe,
                    since=self.p.since,
                    limit=self.p.limit
                )
                if ohlcv:
                    logger.info(f"Received {len(ohlcv)} bars")
                    self.p.since = ohlcv[-1][0] + 1
                    self._data_buffer.extend(ohlcv)
                else:
                    logger.info("No new data returned; sleeping...")
                    time.sleep(self.p.sleep)
                    return None
            except Exception as e:
                logger.error("Error fetching data from CCXT: %s", e)
                time.sleep(self.p.sleep)
                return None
        
        if self._data_buffer:
            candle = self._data_buffer.pop(0)
            tstamp = candle[0] / 1000.0
            dt = datetime.datetime.fromtimestamp(tstamp)
            self.lines.datetime[0] = bt.date2num(dt)
            self.lines.open[0]     = float(candle[1])
            self.lines.high[0]     = float(candle[2])
            self.lines.low[0]      = float(candle[3])
            self.lines.close[0]    = float(candle[4])
            self.lines.volume[0]   = float(candle[5])
            return True
        else:
            return None

# ---------------------------------------------------------------------
# Custom Strategy with Live NN Prediction
# Now the strategy expects a sliding window of 21 candles.
# ---------------------------------------------------------------------
class LiveNNStrategy(bt.Strategy):
    params = (
        ('stop_loss', STOP_LOSS),
        ('window', WINDOW),  # Use the last 21 candles (4h each)
        ('store', None), # Pass the store to access the exchange instance
    )

    def __init__(self):
        self.order = None
        self.entryprice = None
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
        
        self.num_features = 36  # Adjust as necessary

        # Load the neural network model (NNModel)
        try:
            self.model = NNModel(self.num_features, 3)  # assuming 3 classes: BUY, HOLD, SELL
            self.model.load('model.h5')
        except Exception as ex:
            logger.info(f"Error loading NN model: {ex}")
            traceback.print_exc()
            sys.exit(1)

        if self.p.store is None:
            logger.error("Store not provided to strategy. Exiting.")
            sys.exit(1)
        self.store = self.p.store

    def next(self):
        # Avoid processing a new bar if there is a pending order.
        if self.order:
            logger.info("Pending order exists; skipping new order placement.")
            return

        # Ensure that the historical data is sufficient for our window
        if len(self.data) < self.params.window:
            logger.info(f"Insufficient bars loaded ({len(self.data)}). Waiting until {self.params.window} candles are available.")
            return

        # Build a sliding window of the last 21 candles.
        live_data = []
        for i in range(-self.params.window, 0):
            dt = self.data.datetime.datetime(i)
            bar = {
                'Date': dt,
                'Open': self.data.open[i],
                'High': self.data.high[i],
                'Low': self.data.low[i],
                'Close': self.data.close[i],
                'Volume': self.data.volume[i]
            }
            live_data.append(bar)
        logger.info(f"Sliding window rebuilt with {len(live_data)} candles.")

        # Prepare features and predict model label
        df = pd.DataFrame(live_data)
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
        feature_df.ffill(inplace=True)

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

        # Retrieve balance and portfolio value directly from the exchange via the store:
        try:
            balance = self.store.exchange.fetch_balance(params={'type': 'future'})
            portfolio_value = float(balance['total']['USDT'])
            free_cash = float(balance['free']['USDT'])
            logger.info(f"Portfolio Value: {portfolio_value}")
            logger.info(f"Free cash: {free_cash}")
        except Exception as e:
            logger.error("Error retrieving balance: %s", e)
            return

        # Calculate the order size based on the retrieved portfolio value
        current_price = self.data.close[0]
        full_size = (portfolio_value * self.leverage) / current_price
        full_size *= self.position_size
        logger.info(f"Calculated order size: {full_size:.4f} (Portfolio: {portfolio_value:.2f}, Leverage: {self.leverage})")
        cash_required = (full_size * current_price)/self.leverage 
        logger.info(f"Cash required: {cash_required}")

        try:
            if not self.position:
                # Position opening logic – use the imported BUY constant
                
                if label == BUY:
                        self.order = self.buy(exectype=bt.Order.Market, size=full_size)
                        self.entryprice = current_price
                        logger.info(f"ENTER LONG (Leveraged) {full_size:.4f} at {current_price:.2f}")
                
            else:
                # Exit when sell signal occurs
                if label == SELL:
                    self.order = self.sell(exectype=bt.Order.Market)
                    logger.info(f"SELL signal (model label {label}) at {self.data.close[0]:.2f}")
                # Check stop loss conditions.
                elif self.data.low[0] < self.entryprice * (1 - STOP_LOSS):
                    logger.info(f"STOP LOSS HIT at {self.data.low[0]:.2f}")
                    self.order = self.sell(exectype=bt.Order.Market)
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
        elif hasattr(order, 'status') and order.status == 7:
            logger.info("Order expired (status 7). Resetting order state.")
            self.order = None

    def stop(self):
        print("end of strategy")

# ---------------------------------------------------------------------
# Running the Strategy
# ---------------------------------------------------------------------
def run_live_trading():
    cerebro = bt.Cerebro()
    broker = store.getbroker()
    cerebro.setbroker(broker)
    data = CustomCCXTFeed(
        dataname='BTC/USDT',
        exchange=exchange,
        symbol='BTC/USDT',
        # Here we use the feed's own compression (which is set to 240 for 4h candles)
        timeframe=bt.TimeFrame.Minutes,
        limit=100
    )
    cerebro.adddata(data)
    # Pass the store into the strategy so that it can access the exchange
    cerebro.addstrategy(LiveNNStrategy, store=store)
    # If you want to preload historical candles, set preload=True
    cerebro.run(runonce=False, preload=True)
    cerebro.plot()

if __name__ == "__main__":
    run_live_trading()

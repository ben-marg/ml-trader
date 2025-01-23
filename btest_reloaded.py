import os
import sys
import traceback
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import backtrader as bt
import pickle
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel
from sklearn.preprocessing import StandardScaler
from config import RUN as run_conf

STOP_LOSS = 0.05
LEVERAGE = 19  # Adjust leverage as needed

class NNStrategy(bt.Strategy):
    params = (
        ('stop_loss', STOP_LOSS),
        ('leverage', LEVERAGE),
        ('features_array', None),     # Pre-scaled features array for each bar
        ('model', None),              # The NN model instance (already loaded)
    )

    def __init__(self):
        if self.p.features_array is None or self.p.model is None:
            raise ValueError("NNStrategy requires both 'features_array' and 'model'")
        self.order = None
        self.entryprice = None

    def next(self):
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]
        i = len(self) - 1  # Current bar index

        if i < 0:
            signal = HOLD
        else:
            # Use only the current bar's features
            current_features = self.p.features_array[i]
            current_features = np.expand_dims(current_features, axis=0)
            try:
                signal = self.p.model.predict(current_features)
                if isinstance(signal, (np.ndarray, list)):
                    signal = signal[0]
            except Exception as ex:
                self.log(f"Error during prediction at bar {i}: {ex}")
                signal = HOLD
            self.log(f"Predicted signal: {signal}")

        if self.order:
            self.log("An order is currently pending; skipping this bar.")
            return

        # Compute the order size using leverage
        portfolio_value = self.broker.getvalue()
        full_size = (portfolio_value / current_price) * 0.05  # Adjust scaling factor if needed
        self.log(f"Calculated order size: {full_size:.4f} (Portfolio: {portfolio_value:.2f}, Leverage: {self.p.leverage})")

        if not self.position:
            if signal == BUY:
                self.order = self.buy(exectype=bt.Order.Market, size=full_size)
                self.entryprice = current_price
                self.log(f"ENTER LONG (Leveraged) {full_size:.4f} at {current_price:.2f}")
            else:
                self.log("No BUY signal and no position; no order placed.")
        else:
            if current_price < self.entryprice * (1 - self.p.stop_loss):
                self.log(f"STOP LOSS HIT at {self.data.low[0]:.2f} (Entry: {self.entryprice:.2f})")
                self.order = self.sell(exectype=bt.Order.Market, size=self.position.size)
            elif signal == SELL:
                self.log(f"SELL signal received at {current_price:.2f}; exiting long position.")
                self.order = self.sell(exectype=bt.Order.Market, size=self.position.size)
            else:
                self.log("Holding current long position.")

    def log(self, txt, dtobj=None):
        dtobj = dtobj or self.data.datetime.date(0)
        print(f"{dtobj.isoformat()} {txt}")

    def notify_order(self, order):
        self.log(f"Order notification: Status {order.status}")
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size}")
            elif order.issell():
                self.log(f"SELL EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size}")
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: Status {order.status}")
            self.order = None
        elif order.status == 7:
            self.log("Order expired (status 7).")
            self.order = None

class PandasLabelData(bt.feeds.PandasData):
    lines = ('label',)
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('label', 'label'),
    )
def extract_metrics(strategy):
    sharpe_ratio = strategy.analyzers.sharpe.get_analysis().get('sharperatio', None)
    drawdown = strategy.analyzers.drawdown.get_analysis()
    time_return = strategy.analyzers.timereturn.get_analysis()

    # Extract drawdown metrics
    max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
    max_drawdown_pct = max_drawdown * 100 if max_drawdown < 1 else max_drawdown

    # Extract average return metrics
    avg_daily_return = np.mean(list(time_return.values())) if time_return else 0
    annual_return = (1 + avg_daily_return) ** 252 - 1 if avg_daily_return else 0

    # Print the metrics
    print("\nMetrics Summary:")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}" if sharpe_ratio else "Sharpe Ratio: N/A")
    print(f"Max Drawdown (%): {max_drawdown_pct:.2f}")
    print(f"Annual Return (%): {annual_return * 100:.2f}")
    print(f"Final Portfolio Value: {strategy.broker.getvalue():.2f}")

def run_backtest():
    # Load and preprocess data
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

    if data_backtest.empty:
        print("No data available for the backtest period!")
        sys.exit(1)

    cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
    data_train_features = data_train.drop(columns=cols_to_drop, errors='ignore')
    data_train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_train_features.dropna(inplace=True)

    # Debugging Step: Print the shape and columns of the training features
    print(f"Shape of data_train_features before cleaning: {data_train_features.shape}")
    print(f"Columns in data_train_features: {list(data_train_features.columns)}")
    if data_train_features.empty:
        print("Training data is empty after preprocessing!")
        sys.exit(1)

    # Ensure at least one numeric column exists
    numeric_cols = data_train_features.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        print("No numeric columns available for training!")
        sys.exit(1)

    # Fit the scaler only on numeric columns
    scaler = StandardScaler()
    scaler.fit(data_train_features[numeric_cols])

    # Apply the scaler to backtest data
    data_backtest_features = data_backtest.drop(columns=cols_to_drop, errors='ignore')
    data_backtest_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_backtest_features.fillna(method='ffill', inplace=True)
    data_backtest_features.fillna(method='bfill', inplace=True)

    # Debugging Step: Ensure numeric columns exist in backtest features
    numeric_cols_backtest = data_backtest_features.select_dtypes(include=[np.number]).columns
    if numeric_cols_backtest.empty:
        print("No numeric columns available in backtest features!")
        sys.exit(1)

    X_backtest = scaler.transform(data_backtest_features[numeric_cols_backtest])

    data_backtest['label'] = np.nan
    data_backtest.set_index('Date', inplace=True)

    # Load the model
    model_path = r'model.h5'
    num_features = X_backtest.shape[1]
    model = NNModel(num_features, 3)
    try:
        model.load(model_path)
    except Exception as ex:
        print(f"Error loading model from {model_path}: {ex}")
        traceback.print_exc()
        sys.exit(1)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(run_conf.get('initial_capital', 10000.0))
    cerebro.broker.setcommission(commission=run_conf.get('commission fee', 0.001))

    bt_data = PandasLabelData(dataname=data_backtest)
    cerebro.adddata(bt_data)
    cerebro.addstrategy(NNStrategy, stop_loss=STOP_LOSS, leverage=LEVERAGE, features_array=X_backtest, model=model)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    print("Running backtest...")
    cerebro.run()
    cerebro.plot(iplot=True, volume=False)

    extract_metrics(NNStrategy)



if __name__ == '__main__':
    run_backtest()

import os
import sys
import traceback
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtrader as bt

from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel
from sklearn.preprocessing import StandardScaler
from config import RUN as run_conf


# ======================================================================
# Create a custom data feed that includes a new “label” line
# ======================================================================
class PandasLabelData(bt.feeds.PandasData):
    """
    Extend the PandasData feed to include the 'label' column.
    Make sure your CSV/DataFrame has a column called 'label'.
    """
    lines = ('label',)
    params = (
        ('datetime', None),  # use default datetime (i.e. index)
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('label', 'label'),  # extra line
    )


# ======================================================================
# Define a simple NN-based strategy
# ======================================================================
class NNStrategy(bt.Strategy):
    """
    This strategy uses a pre-computed 'label' field in the data to trigger trades.
      - If label == BUY: go long
      - If label == SELL: exit (or go short if you want)
      - If the price falls below stop_loss from entry, exit
    """
    params = (
        ('stop_loss', 0.05),  # 5% stop loss
    )

    def __init__(self):
        self.order = None
        self.entryprice = None

    def next(self):
        # Avoid double-ordering
        if self.order:
            return

        if not self.position:  
            # We have no open position
            if self.data.label[0] == BUY:  # Assuming BUY=1
                self.order = self.buy()
                self.entryprice = self.data.close[0]
                self.log(f"BUY at {self.entryprice:.2f}")
        else:
            # We have an open position
            if self.data.low[0] < self.entryprice * (1 - self.params.stop_loss):
                # Price dropped below stop-loss
                self.log(f"STOP LOSS HIT at {self.data.low[0]:.2f}")
                self.order = self.sell()  # exit the position
            elif self.data.label[0] == SELL:  # Assuming SELL=-1
                self.log(f"SELL signal at {self.data.close[0]:.2f}")
                self.order = self.sell()

    def log(self, txt, dtobj=None):
        """ Logging function for this strategy"""
        dtobj = dtobj or self.data.datetime.date(0)
        print(f"{dtobj.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None
class BuyHoldStrategy(bt.Strategy):
    """
    Simple Buy and Hold Strategy.
    Buys all available capital on the first trading day and holds until the end.
    """

    def __init__(self):
        self.first_trade_done = False

    def next(self):
        if not self.first_trade_done:
            # Buy all available cash at the current price
            self.buy(size=self.broker.get_cash() / self.data.close[0])
            self.first_trade_done = True

def calculate_metrics(cerebro, strategy_name):
    """
    Calculate performance metrics for a given strategy.
    Args:
        cerebro: The Backtrader engine instance after running the strategy.
        strategy_name: Name of the strategy being evaluated.
    Returns:
        A dictionary with Sharpe ratio, max drawdown, and annual return.
    """
    # Extract portfolio value over time
    portfolio_values = cerebro.broker.get_value_dataseries()
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Sharpe ratio
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

    # Max drawdown
    max_drawdown = 0
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Annualized return
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annual_return = (1 + total_return) ** (1 / (len(portfolio_values) / 252)) - 1

    return {
        "Strategy": strategy_name,
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Annual Return (%)": round(annual_return * 100, 2),
        "Final Portfolio Value": round(portfolio_values[-1], 2),
    }

# ======================================================================
# Prepare data, compute indicators, predictions, and run Cerebro
# ======================================================================
def run_backtest():
    # Create a Cerebro engine instance
    cerebro = bt.Cerebro()

    # Set initial capital
    initial_capital = run_conf.get('initial_capital', 10000.0)
    cerebro.broker.setcash(initial_capital)

    # Set broker commission
    commission_fee = run_conf.get('commission fee', 0.001)
    cerebro.broker.setcommission(commission=commission_fee)

    # ===================================================================
    # 1. Read CSV for the full backtest range (including training portion)
    # ===================================================================
    full_csv = os.path.join(run_conf['folder'], 'BTCUSDT.csv')
    try:
        data = pd.read_csv(full_csv)
    except Exception as ex:
        print("Error loading CSV data:", ex)
        sys.exit(1)

    # Convert 'Date' and sort
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)

    # We'll apply indicators to the entire period, then subdivide
    data = TecnicalAnalysis.compute_oscillators(data)
    data = TecnicalAnalysis.find_patterns(data)
    data = TecnicalAnalysis.add_timely_data(data)

    # Filter: we only want rows between the earliest training date and the final backtest date
    train_start_dt = pd.to_datetime(run_conf['train_start'])
    train_end_dt   = pd.to_datetime(run_conf['train_end'])
    backtest_start = pd.to_datetime(run_conf['back_test_start'])
    backtest_end   = pd.to_datetime(run_conf['back_test_end'])

    # If your CSV has data older than train_start or newer than backtest_end, we can cut it:
    data = data[(data['Date'] >= train_start_dt) & (data['Date'] <= backtest_end)]

    if data.empty:
        print("No data after filtering!")
        sys.exit(1)

    # ===================================================================
    # 2. Separate the "training portion" (for fitting scaler) from the "backtest portion"
    # ===================================================================
    #   We'll define:
    #   training data: [train_start_dt, train_end_dt]
    #   backtest data: [backtest_start, backtest_end]
    #   Possibly there's overlap or a gap.

    train_mask = (data['Date'] >= train_start_dt) & (data['Date'] <= train_end_dt)
    data_train = data[train_mask].copy()

    # The "full backtest" portion covers [backtest_start, backtest_end]
    # but we want signals for everything from backtest_start to backtest_end.
    backtest_mask = (data['Date'] >= backtest_start) & (data['Date'] <= backtest_end)
    data_backtest = data[backtest_mask].copy()

    # If you want to see how many rows are in training vs. backtest:
    print(f"Training portion: {len(data_train)} rows")
    print(f"Backtest portion: {len(data_backtest)} rows")

    # ===================================================================
    # 3. Fit the scaler ONLY on the training portion (data_train)
    # ===================================================================
    #   Create a copy for scaling and drop non-numeric columns
    data_train_features = data_train.copy()
    cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
    data_train_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Replace inf, and drop or fill any nans in training
    data_train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_train_features.dropna(inplace=True)  # we must drop these for training

    # Fit scaler on training features
    scaler = StandardScaler()
    scaler.fit(data_train_features)  # only on training subset

    # ===================================================================
    # 4. Generate predictions for the FULL backtest range
    # ===================================================================
    #   We'll transform data_backtest (the entire period we want signals for).
    #   Then we apply the model to get predictions for each row.
    # ===================================================================
    data_backtest_features = data_backtest.copy()
    data_backtest_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    data_backtest_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill missing values in the backtest portion (so we don't drop rows)
    data_backtest_features.fillna(method='ffill', inplace=True)
    data_backtest_features.fillna(method='bfill', inplace=True)

    Xb = scaler.transform(data_backtest_features)  # transform backtest features

    # Load your pre-trained NN model
    model_path = 'model.h5'
    num_features = Xb.shape[1]  # must match your net input size
    model = NNModel(num_features, 3)
    model.load(model_path)

    try:
        labels = model.predict(Xb)  # predictions for the entire backtest period
    except Exception as ex:
        print("Error while predicting:", ex)
        traceback.print_exc()
        sys.exit(1)

    # Convert to a Series with the same index as data_backtest
    labels_series = pd.Series(labels, index=data_backtest.index)
    # Now attach that to data_backtest
    data_backtest['label'] = labels_series

    # For any rows that still have missing 'label' (should be none if we filled properly),
    # we can fill them with HOLD
    data_backtest['label'].fillna(HOLD, inplace=True)

    # ===================================================================
    # 5. Prepare final DataFrame for Backtrader
    # ===================================================================
    #   data_backtest is from backtest_start to backtest_end, with 'label'
    # ===================================================================
    data_backtest.set_index('Date', inplace=True)

    # Create a Backtrader data feed
    bt_data = PandasLabelData(dataname=data_backtest)

    # Add to cerebro
    cerebro.adddata(bt_data)

    # Add our strategy
    cerebro.addstrategy(NNStrategy, stop_loss=run_conf.get('stop_loss', 0.02))

    # Print the timeframe we are actually using
    print(f"Backtest data from {backtest_start.date()} to {backtest_end.date()}: {len(data_backtest)} rows")

    # ===================================================================
    # 6. Run the backtest
    # ===================================================================
    start_val = cerebro.broker.getvalue()
    print("Starting Portfolio Value: %.2f" % start_val)
    cerebro.run()
    end_val = cerebro.broker.getvalue()
    print("Final Portfolio Value: %.2f" % end_val)

    # ===================================================================
    # 7. Plot the results
    # ===================================================================
    cerebro.plot(iplot=True, volume=False)


if __name__ == '__main__':
    run_backtest()

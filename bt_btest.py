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
# Create a custom data feed that includes a new "label" line
# ======================================================================
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


# ======================================================================
# NN-based Strategy
# ======================================================================
class NNStrategy(bt.Strategy):
    params = (('stop_loss', 0.02),)

    def __init__(self):
        self.order = None
        self.entryprice = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.label[0] == BUY:
                self.order = self.buy()
                self.entryprice = self.data.close[0]
                self.log(f"BUY at {self.entryprice:.2f}")
        else:
            if self.data.low[0] < self.entryprice * (1 - self.params.stop_loss):
                self.log(f"STOP LOSS HIT at {self.data.low[0]:.2f}")
                self.order = self.sell()
            elif self.data.label[0] == SELL:
                self.log(f"SELL signal at {self.data.close[0]:.2f}")
                self.order = self.sell()

    def log(self, txt, dtobj=None):
        dtobj = dtobj or self.data.datetime.date(0)
        print(f"{dtobj.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None


# ======================================================================
# Simplified Buy-and-Hold Strategy
# ======================================================================
class BuyHoldStrategy(bt.Strategy):
    """
    Incremental Buy-and-Hold Strategy:
    Buys incrementally over several bars until all capital is exhausted, then sells on the last bar.
    """
    def __init__(self):
        self.remaining_cash = None  # Track how much cash is left for incremental buys
        self.increment_ratio = 0.1  # Buy 10% of remaining cash on each bar
        self.buying_complete = False

    def next(self):
        # Initialize remaining_cash on the first bar
        if self.remaining_cash is None:
            self.remaining_cash = self.broker.get_cash()
            print(f"Initial Cash: {self.remaining_cash:.2f}")

        # Incremental buying logic
        if not self.buying_complete:
            # Calculate the amount to invest in this increment
            increment_cash = self.remaining_cash * self.increment_ratio
            size = increment_cash / self.data.close[0]

            # Check if there's enough cash to place a meaningful order
            if size > 0 and self.remaining_cash > increment_cash:
                self.buy(size=size)
                self.remaining_cash -= increment_cash  # Deduct used cash
                self.log(f"INCREMENTAL BUY: Size {size:.4f}, Remaining Cash: {self.remaining_cash:.2f}")

            # Stop buying if remaining cash is too low
            if self.remaining_cash < self.data.close[0]:
                self.buying_complete = True
                self.log("BUYING COMPLETE: Insufficient cash for further buys.")

        # Detect the last bar and sell all positions
        if self.data._last():
            if self.position:
                self.sell(size=self.position.size)
                self.log(f"SELL: Selling all at price {self.data.close[0]:.2f}")

    def stop(self):
        # Ensure all positions are sold at the end
        if self.position:
            self.sell(size=self.position.size)
            self.log(f"STOP: Sold remaining position at price {self.data.close[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size}")
            elif order.issell():
                self.log(f"SELL EXECUTED: Price {order.executed.price:.2f}, Size {order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")

    def log(self, txt, dtobj=None):
        """
        Logging function for the strategy.
        """
        dtobj = dtobj or self.data.datetime.date(0)
        print(f"{dtobj.isoformat()} {txt}")



# ======================================================================
# Analyzer-based Metrics Calculation Function
# ======================================================================
def get_metrics_from_analyzers(cerebro_instance, strategy_instance, strategy_name):
    final_value = cerebro_instance.broker.getvalue()

    sharpe_dict = strategy_instance.analyzers.sharpe.get_analysis()
    drawdown_dict = strategy_instance.analyzers.drawdown.get_analysis()
    timereturn_dict = strategy_instance.analyzers.timereturn.get_analysis()

    # Handle possibility that 'sharperatio' is None.
    s = sharpe_dict.get('sharperatio', 0)
    if s is None:
        s = 0
    sharpe = float(s)

    max_dd_value = drawdown_dict.get('max', 0)
    if isinstance(max_dd_value, dict):
        max_dd_value = float(max_dd_value.get('drawdown', 0))
    else:
        max_dd_value = float(max_dd_value)
    if max_dd_value < 1:
        max_dd_pct = max_dd_value * 100
    else:
        max_dd_pct = max_dd_value

    daily_returns = np.array(list(timereturn_dict.values()))
    if daily_returns.size > 0:
        avg_daily_return = np.mean(daily_returns)
        neg_returns = daily_returns[daily_returns < 0]
        if len(neg_returns) > 0 and np.std(neg_returns) != 0:
            sortino_ratio = np.mean(daily_returns) / np.std(neg_returns)
        else:
            sortino_ratio = 0
        annual_return = (1 + avg_daily_return) ** 252 - 1
    else:
        sortino_ratio = 0
        annual_return = 0

    return {
        "Strategy": strategy_name,
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino_ratio, 3),
        "Max Drawdown (%)": round(max_dd_pct, 2),
        "Annual Return (%)": round(annual_return * 100, 2),
        "Final Portfolio Value": round(final_value, 2),
    }


# ======================================================================
# Main Backtest Function
# ======================================================================
def run_backtest():
    # ----------------------------
    # 1. Load & Preprocess Data
    # ----------------------------
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

    # -------------------------------
    # 2. Fit Scaler on Training Data
    # -------------------------------
    cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
    data_train_features = data_train.drop(columns=cols_to_drop, errors='ignore')
    data_train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_train_features.dropna(inplace=True)

    scaler = StandardScaler()
    scaler.fit(data_train_features)

    # -----------------------------------------------
    # 3. Generate NN Predictions for Backtest Data
    # -----------------------------------------------
    data_backtest_features = data_backtest.drop(columns=cols_to_drop, errors='ignore')
    data_backtest_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_backtest_features.fillna(method='ffill', inplace=True)
    data_backtest_features.fillna(method='bfill', inplace=True)

    X_backtest = scaler.transform(data_backtest_features)

    model_path = 'model.h5'
    num_features = X_backtest.shape[1]
    model = NNModel(num_features, 3)  # Assuming 3 classes: BUY, HOLD, SELL
    try:
        model.load(model_path)
    except Exception as ex:
        print(f"Error loading model from {model_path}: {ex}")
        traceback.print_exc()
        sys.exit(1)

    try:
        predictions = model.predict(X_backtest)
    except Exception as ex:
        print("Error during prediction:", ex)
        traceback.print_exc()
        sys.exit(1)

    data_backtest['label'] = pd.Series(predictions, index=data_backtest.index)
    data_backtest['label'].fillna(HOLD, inplace=True)
    data_backtest.set_index('Date', inplace=True)

    # ----------------------------------
    # 4. Run Strategies in Backtrader with Analyzers
    # ----------------------------------

    # --- NN-Based Strategy ---
    cerebro_nn = bt.Cerebro()
    cerebro_nn.broker.setcash(run_conf.get('initial_capital', 10000.0))
    cerebro_nn.broker.setcommission(commission=run_conf.get('commission fee', 0.001))
    bt_data_nn = PandasLabelData(dataname=data_backtest)
    cerebro_nn.adddata(bt_data_nn)
    cerebro_nn.addstrategy(NNStrategy, stop_loss=run_conf.get('stop_loss', 0.02))
    cerebro_nn.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro_nn.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro_nn.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    print("Running NN Strategy...")
    strategies_nn = cerebro_nn.run()
    strat_nn = strategies_nn[0]
    metrics_nn = get_metrics_from_analyzers(cerebro_nn, strat_nn, "NN Strategy")

    # --- Buy-and-Hold Strategy ---
    cerebro_bh = bt.Cerebro()
    cerebro_bh.broker.setcash(run_conf.get('initial_capital', 10000.0))
    cerebro_bh.broker.setcommission(commission=run_conf.get('commission fee', 0.001))
    bt_data_bh = PandasLabelData(dataname=data_backtest)
    cerebro_bh.adddata(bt_data_bh)
    cerebro_bh.addstrategy(BuyHoldStrategy)
    cerebro_bh.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro_bh.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro_bh.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    print("Running Buy-and-Hold Strategy...")
    strategies_bh = cerebro_bh.run()
    strat_bh = strategies_bh[0]
    metrics_bh = get_metrics_from_analyzers(cerebro_bh, strat_bh, "Buy-and-Hold Strategy")

    # ------------------------------------------
    # 5. Compare and Print Results
    # ------------------------------------------
    results = [metrics_nn, metrics_bh]
    results_df = pd.DataFrame(results)
    print("\nComparison of Strategies:")
    print(results_df)

    print("\nNN Strategy Portfolio Value:")
    cerebro_nn.plot(iplot=True, volume=False)

    print("\nBuy-and-Hold Strategy Portfolio Value:")
    cerebro_bh.plot(iplot=True, volume=False)


if __name__ == '__main__':
    run_backtest()

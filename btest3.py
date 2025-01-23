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

STOP_LOSS = 0.05
LEVERAGE = 19  # Adjust leverage as needed

class NNStrategy(bt.Strategy):
    params = (
        ('stop_loss', STOP_LOSS),
        ('leverage', LEVERAGE),
       # ('window_size', 5),           # You can still define a window size if desired
        ('features_array', None),     # pre-scaled features array for each bar
        ('model', None),              # the NN model instance (already loaded)
    )

    def __init__(self):
        if self.p.features_array is None or self.p.model is None:
            raise ValueError("NNStrategy requires both 'features_array' and 'model'")
        self.order = None
        self.entryprice = None

    def next(self):
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]
        i = len(self) - 1  # current bar index

        # For debugging, you might still log that you have a window, but we use only the current bar:
        if i < 0:
            signal = HOLD  # just in case, though i>=0 always.
        else:
            # Use only the current bar's features (shape becomes (1, 36))
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

        # Compute the order size using leverage:
        portfolio_value = self.broker.getvalue()
        full_size = (portfolio_value * self.p.leverage) / current_price
        # Optionally apply a scaling factor:
        full_size = full_size * 0.05  
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
# Simplified Buy-and-Hold Strategy (unchanged)
# ======================================================================
class BuyHoldStrategy(bt.Strategy):
    """
    Incremental Buy-and-Hold Strategy:
    Buys incrementally over several bars until all capital is exhausted,
    then sells on the last bar.
    """
    def __init__(self):
        self.remaining_cash = None
        self.increment_ratio = 0.01  # Adjust ratio as needed
        self.buying_complete = False

    def next(self):
        if self.remaining_cash is None:
            self.remaining_cash = self.broker.get_cash()
            print(f"Initial Cash: {self.remaining_cash:.2f}")

        if not self.buying_complete:
            increment_cash = self.remaining_cash * self.increment_ratio
            size = increment_cash / self.data.close[0]
            if size > 0 and self.remaining_cash > increment_cash:
                self.buy(size=size)
                self.remaining_cash -= increment_cash
                self.log(f"INCREMENTAL BUY: Size {size:.4f}, Remaining Cash: {self.remaining_cash:.2f}")
            if self.remaining_cash < self.data.close[0]:
                self.buying_complete = True
                self.log("BUYING COMPLETE: Insufficient cash for further buys.")

        if self.data._last():
            if self.position:
                self.sell(size=self.position.size)
                self.log(f"SELL: Selling all at price {self.data.close[0]:.2f}")

    def stop(self):
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
            self.log(f"Order Canceled/Margin/Rejected: Status {order.status}")

    def log(self, txt, dtobj=None):
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

    s = sharpe_dict.get('sharperatio', 0)
    if s is None:
        s = 0
    sharpe = float(s)

    max_dd_value = drawdown_dict.get('max', 0)
    if isinstance(max_dd_value, dict):
        max_dd_value = float(max_dd_value.get('drawdown', 0))
    else:
        max_dd_value = float(max_dd_value)
    max_dd_pct = max_dd_value * 100 if max_dd_value < 1 else max_dd_value

    daily_returns = np.array(list(timereturn_dict.values()))
    if daily_returns.size > 0:
        avg_daily_return = np.mean(daily_returns)
        neg_returns = daily_returns[daily_returns < 0]
        sortino_ratio = (np.mean(daily_returns) / np.std(neg_returns)) if (len(neg_returns) > 0 and np.std(neg_returns) != 0) else 0
        annual_return = (1 + avg_daily_return) ** 252 - 1
    else:
        sortino_ratio = 0
        annual_return = 0

    return {
        "Strategy": strategy_name,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown (%)": max_dd_pct,
        "Annual Return (%)": annual_return * 100,
        "Final Portfolio Value": final_value,
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
    # 3. Generate Scaled Features for Backtest Data
    # -----------------------------------------------
    data_backtest_features = data_backtest.drop(columns=cols_to_drop, errors='ignore')
    data_backtest_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_backtest_features.fillna(method='ffill', inplace=True)
    data_backtest_features.fillna(method='bfill', inplace=True)
    X_backtest = scaler.transform(data_backtest_features)
    # Do NOT compute predictions here; they will be computed on-the-fly.

    # To satisfy the DataFeed's expectations, add a dummy "label" column.
    data_backtest['label'] = np.nan
    data_backtest.set_index('Date', inplace=True)

    # -----------------------------------------------
    # 4. Load the Neural Network Model
    # -----------------------------------------------
    model_path = r'model.h5'
    num_features = X_backtest.shape[1]
    model = NNModel(num_features, 3)  # Assuming 3 classes: BUY, HOLD, SELL
    try:
        model.load(model_path)
    except Exception as ex:
        print(f"Error loading model from {model_path}: {ex}")
        traceback.print_exc()
        sys.exit(1)

    # ----------------------------------
    # 5. Set up Backtrader Data Feed and Run Strategies
    # ----------------------------------

    # --- NN-Based Strategy (On-the-Fly Predictions with Sliding Window) ---
    cerebro_nn = bt.Cerebro()
    cerebro_nn.broker.setcash(run_conf.get('initial_capital', 10000.0))
    cerebro_nn.broker.setcommission(commission=run_conf.get('commission fee', 0.001))
    #cerebro_nn.broker.set_coc(True)

    bt_data_nn = PandasLabelData(dataname=data_backtest)
    cerebro_nn.adddata(bt_data_nn)
    cerebro_nn.addstrategy(NNStrategy,
                           stop_loss=run_conf.get('stop_loss', STOP_LOSS),
                           leverage=LEVERAGE,
                         
                           features_array=X_backtest,
                           model=model)
    cerebro_nn.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro_nn.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro_nn.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    print("Running NN Strategy (On-the-Fly Predictions with Sliding Window)...")
    strategies_nn = cerebro_nn.run()
    strat_nn = strategies_nn[0]
    metrics_nn = get_metrics_from_analyzers(cerebro_nn, strat_nn, "NN Strategy (On-the-Fly Predictions)")

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
    # 6. Compare and Print Results
    # ------------------------------------------
    results = [metrics_nn, metrics_bh]
    results_df = pd.DataFrame(results)
    print("\nComparison of Strategies:")
    print(results_df)
    results_df.to_csv('backtest.csv', index=False)

    print("\nNN Strategy Portfolio Value:")
    cerebro_nn.plot(iplot=True, volume=False)
    print("\nBuy-and-Hold Strategy Portfolio Value:")
    cerebro_bh.plot(iplot=True, volume=False)


if __name__ == '__main__':
    run_backtest()

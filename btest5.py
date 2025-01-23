import os
import sys
import traceback
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pickle
from collections import deque

import backtrader as bt

from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel
from sklearn.preprocessing import StandardScaler
from config import RUN as run_conf

# ================================
# Constants and Configuration
# ================================
STOP_LOSS = 0.05
LEVERAGE = 15  # Adjust leverage as needed
BUFFER_SIZE = 100  # Adjust based on maximum lookback required for feature computation

# ================================
# Neural Network-Based Strategy
# ================================
class NNStrategy(bt.Strategy):
    """
    Neural Network-Based Strategy with On-the-Fly Feature Computation using Sliding Window.
    """
    params = (
        ('stop_loss', STOP_LOSS),
        ('leverage', LEVERAGE),
        ('scaler', None),       # Pre-fitted StandardScaler
        ('model', None),        # Pre-loaded Neural Network model
        ('feature_names', []),  # List of feature names used during training
        ('buffer_size', BUFFER_SIZE),  # Size of the sliding window buffer
    )

    def __init__(self):
        # Validate required parameters
        if self.p.scaler is None or self.p.model is None:
            raise ValueError("NNStrategy requires both 'scaler' and 'model'")
        if not self.p.feature_names:
            raise ValueError("NNStrategy requires 'feature_names'")

        self.order = None
        self.entryprice = None

        # Initialize a deque buffer to store past bars
        self.buffer = deque(maxlen=self.p.buffer_size)

        # Flag to indicate if the buffer has enough data to start predictions
        self.buffer_ready = False

    def next(self):
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]

        # Append current bar's data to the buffer
        bar_data = {
            'Date': self.data.datetime.datetime(0),
            'Open': self.data.open[0],
            'High': self.data.high[0],
            'Low': self.data.low[0],
            'Close': self.data.close[0],
            'Volume': self.data.volume[0],
        }
        self.buffer.append(bar_data)

        # Check if buffer is ready
        if len(self.buffer) < self.p.buffer_size:
            self.log(f"Buffer not ready: {len(self.buffer)}/{self.p.buffer_size}")
            return  # Not enough data to compute features

        if not self.buffer_ready:
            self.buffer_ready = True
            self.log("Buffer is now ready for feature computation")

        # Convert buffer to DataFrame
        buffer_df = pd.DataFrame(list(self.buffer))
        
        # Compute features using TecnicalAnalysis
        try:
            buffer_df = TecnicalAnalysis.compute_oscillators(buffer_df)
            buffer_df = TecnicalAnalysis.find_patterns(buffer_df)
            buffer_df = TecnicalAnalysis.add_timely_data(buffer_df)
        except Exception as ex:
            self.log(f"Error computing features: {ex}")
            return

        # Drop columns not used for features
        cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
        buffer_features = buffer_df.drop(columns=cols_to_drop, errors='ignore')

        # Handle missing or infinite values
        buffer_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        buffer_features.ffill()
        buffer_features.bfill()

        # Extract the latest features (current bar)
        current_features = buffer_features.iloc[-1].values.reshape(1, -1)

        # Ensure feature count matches scaler
        if current_features.shape[1] != self.p.scaler.mean_.shape[0]:
            self.log(f"Feature count mismatch: {current_features.shape[1]} vs {self.p.scaler.mean_.shape[0]}")
            return

        # Scale the features
        try:
            scaled_features = self.p.scaler.transform(current_features)
        except Exception as ex:
            self.log(f"Error during scaling features: {ex}")
            return

        # Predict using the model
        try:
            prediction = self.p.model.predict(scaled_features)
            print(prediction)
            if isinstance(prediction, (np.ndarray, list)):
                signal = prediction[0]
            else:
                signal = HOLD
            self.log(f"Predicted signal: {signal}")
        except Exception as ex:
            self.log(f"Error during prediction: {ex}")
            signal = HOLD

        # Skip if an order is pending
        if self.order:
            self.log("An order is currently pending; skipping this bar.")
            return

        # Compute the order size using leverage
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
    # ================================
    # Buy-and-Hold Strategy Definition
    # ================================
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
            self.log(f"Initial Cash: {self.remaining_cash:.2f}")

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

    # ================================
    # Analyzer-based Metrics Calculation Function
    # ================================
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

    # ================================
    # Main Backtest Function
    # ================================
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

    # Compute technical indicators and patterns
    try:
        data = TecnicalAnalysis.compute_oscillators(data)
        data = TecnicalAnalysis.find_patterns(data)
        data = TecnicalAnalysis.add_timely_data(data)
    except Exception as ex:
        print(f"Error during technical analysis computation: {ex}")
        traceback.print_exc()
        sys.exit(1)

    # Define training and backtest periods
    try:
        train_start_dt = pd.to_datetime(run_conf['train_start'])
        train_end_dt = pd.to_datetime(run_conf['train_end'])
        backtest_start_dt = pd.to_datetime(run_conf['back_test_start'])
        backtest_end_dt = pd.to_datetime(run_conf['back_test_end'])
    except Exception as ex:
        print(f"Error parsing dates from config: {ex}")
        sys.exit(1)

    # Split data into training and backtest
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

    # Save scaler and feature names
    scaler_filename = 'scaler.pkl'
    with open(scaler_filename, 'wb') as f:
        pickle.dump({'scaler': scaler, 'features': data_train_features.columns.tolist()}, f)

    print("Scaler fitted and saved.")

    # -----------------------------------------------
    # 3. Load the Neural Network Model
    # -----------------------------------------------
    model_path = 'model.h5'
    num_features = len(data_train_features.columns)
    model = NNModel(num_features, 3)  # Assuming 3 classes: BUY, HOLD, SELL
    try:
        model.load(model_path)
    except Exception as ex:
        print(f"Error loading model from {model_path}: {ex}")
        traceback.print_exc()
        sys.exit(1)

    print("Loaded model successfully.")

    # ----------------------------------
    # 4. Set up Backtrader Data Feed and Run Strategies
    # ----------------------------------

    # Initialize Cerebro instances
    cerebro_nn = bt.Cerebro()
    cerebro_bh = bt.Cerebro()

    # Set broker parameters
    initial_cash = run_conf.get('initial_capital', 10000.0)
    commission_fee = run_conf.get('commission_fee', 0.001)
    cerebro_nn.broker.setcash(initial_cash)
    cerebro_nn.broker.setcommission(commission=commission_fee)
    cerebro_bh.broker.setcash(initial_cash)
    cerebro_bh.broker.setcommission(commission=commission_fee)

    # Add data feeds using default PandasData
    bt_data_nn = bt.feeds.PandasData(dataname=data_backtest, datetime='Date')
    cerebro_nn.adddata(bt_data_nn)

    bt_data_bh = bt.feeds.PandasData(dataname=data_backtest, datetime= 'Date')
    cerebro_bh.adddata(bt_data_bh)

    # Load scaler and feature names
    try:
        with open(scaler_filename, 'rb') as f:
            scaler_data = pickle.load(f)
    except Exception as ex:
        print(f"Error loading scaler from {scaler_filename}: {ex}")
        sys.exit(1)

    scaler_loaded = scaler_data.get('scaler')
    feature_names = scaler_data.get('features', [])

    if not scaler_loaded or not feature_names:
        print("Scaler or feature names not found in scaler.pkl.")
        sys.exit(1)

    # Add strategies
    cerebro_nn.addstrategy(
        NNStrategy,
        scaler=scaler_loaded,
        model=model,
        stop_loss=STOP_LOSS,
        leverage=LEVERAGE,
        feature_names=feature_names,
        buffer_size=BUFFER_SIZE
    )
    cerebro_bh.addstrategy(
        BuyHoldStrategy
    )

    # Add analyzers
    cerebro_nn.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro_nn.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro_nn.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

    cerebro_bh.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro_bh.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro_bh.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    
    # Run strategies
    print("\nRunning NN Strategy (On-the-Fly Predictions)...")
    try:
        strategies_nn = cerebro_nn.run()
        strat_nn = strategies_nn[0]
        metrics_nn = get_metrics_from_analyzers(cerebro_nn, strat_nn, "NN Strategy (On-the-Fly Predictions)")
    except Exception as ex:
        print("Error running NN Strategy:", ex)
        traceback.print_exc()
        sys.exit(1)

    print("\nRunning Buy-and-Hold Strategy...")
    try:
        strategies_bh = cerebro_bh.run()
        strat_bh = strategies_bh[0]
        metrics_bh = get_metrics_from_analyzers(cerebro_bh, strat_bh, "Buy-and-Hold Strategy")
    except Exception as ex:
        print("Error running Buy-and-Hold Strategy:", ex)
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------
    # 5. Extract Metrics
    # ------------------------------------------
    # Already extracted during running

    # ------------------------------------------
    # 6. Compare and Print Results
    # ------------------------------------------
    results = [metrics_nn, metrics_bh]
    results_df = pd.DataFrame(results)
    print("\nComparison of Strategies:")
    print(results_df)
    results_df.to_csv('backtest_comparison.csv', index=False)

    # ------------------------------------------
    # 7. Plot Cumulative Returns
    # ------------------------------------------
    # Extract portfolio value history
    hist_nn = strat_nn.analyzers.timereturn.get_analysis()
    hist_bh = strat_bh.analyzers.timereturn.get_analysis()

    # Convert to DataFrame for plotting
    df_nn = pd.DataFrame.from_dict(hist_nn, orient='index', columns=['NN Strategy'])
    df_bh = pd.DataFrame.from_dict(hist_bh, orient='index', columns=['Buy-and-Hold'])
    df_combined = pd.concat([df_nn, df_bh], axis=1).dropna()

    # Calculate cumulative returns
    df_combined = (1 + df_combined).cumprod()

    # Plotting
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#eeeeee')

    # Plot cumulative returns
    df_combined.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    filename = os.path.basename(full_csv).split('.')[0]
    ax.set_title(f"Strategy Comparison for {filename}")
    ax.legend()
    ax.grid(True)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    # Save the plot
    plot_filename = os.path.join(run_conf['reports'], 'backtest_comparison.png')
    fig.tight_layout()
    fig.savefig(plot_filename)
    plt.show()

    return {
        'du': (
            metrics_bh["Final Portfolio Value"],
            metrics_bh["Sharpe Ratio"],
            metrics_bh["Max Drawdown (%)"],
            metrics_bh["Annual Return (%)"]
        ),
        'nn': (
            metrics_nn["Final Portfolio Value"],
            metrics_nn["Sharpe Ratio"],
            metrics_nn["Max Drawdown (%)"],
            metrics_nn["Annual Return (%)"]
        )
    }

if __name__ == '__main__':
    backtest_results = run_backtest()
    print("\nBacktest Results:", backtest_results)

# btest_backtrader.py

import os
import sys
import traceback
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import backtrader as bt
import pickle
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel  # Ensure this is compatible with your ML framework
from sklearn.preprocessing import StandardScaler
from config import RUN as run_conf
from data_processing import preprocess_data_backtrader

STOP_LOSS = 0.05
LEVERAGE = 19  # Adjust leverage as needed

# Define your custom data feed class
class CustomPandasData(bt.feeds.PandasData):
    """
    Custom Pandas Data Feed without scaled features.
    """
    lines = ()  # No additional lines for scaled features
    params = (
        ('datetime', 'Date'),      # Map 'Date' to datetime
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        # No scaled features included
    )

class BuyHoldStrategy(bt.Strategy):
    """
    Incremental Buy-and-Hold Strategy:
    Buys incrementally over several bars until all capital is exhausted,
    then sells on the last bar.
    """
    def __init__(self):
        self.remaining_cash = None
        self.increment_ratio = 0.2  # Adjust ratio as needed
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

class LiveNNStrategy(bt.Strategy):
    params = (
        ('stop_loss', STOP_LOSS),
        ('leverage', LEVERAGE),
        ('scaler', None),  # Pre-fitted scaler
        ('model', None),   # The NN model instance (already loaded)
    )

    def __init__(self):
        if self.p.scaler is None or self.p.model is None:
            raise ValueError("LiveNNStrategy requires both 'scaler' and 'model'")
        self.order = None
        self.entryprice = None

        # Define raw feature names required for prediction
        self.feature_names = ['rsi', 'macd', 'price_change']  # Replace with actual feature names

    def next(self):
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]

        try:
            # Collect raw feature values for the current bar
            features = []
            for feat in self.feature_names:
                feature_value = getattr(self.data, feat)[0]
                features.append(feature_value)
                self.log(f"Feature {feat}: {feature_value}")

            # Convert features to numpy array and reshape for scaler
            raw_features = np.array(features).reshape(1, -1)

            # Scale features using the pre-fitted scaler
            scaled_features = self.p.scaler.transform(raw_features)

            # Predict using the NN model
            prediction = self.p.model.predict(scaled_features)
            signal = np.argmax(prediction, axis=1)[0] if isinstance(prediction, np.ndarray) else HOLD

            self.log(f"Predicted signal: {signal}")

        except Exception as ex:
            self.log(f"Error processing features or predicting signal: {ex}")
            signal = HOLD

        # Skip if an order is pending
        if self.order:
            self.log("An order is currently pending; skipping this bar.")
            return

        # Compute order size using leverage
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

    def log(self, txt, dtobj=None):
        dtobj = dtobj or self.data.datetime.date(0)
        print(f"{dtobj.isoformat()} {txt}")

def run_backtest_single_coin(RUN, filename, mdl_name=r'model.h5', suffix=""):
    """
    Backtest a coin whose timeseries is contained in filename.
    It uses last model trained.
    Backtest period selected in RUN config dictionary
    :param suffix: 
    :param mdl_name: 
    :param RUN: 
    :param filename: 
    :return: a dictionary with dummy (du) and neural net (nn) statistics of backtest
    """
    try:
        # Preprocess data and fit scaler
        processed_data, scaler = preprocess_data_backtrader(RUN, filename)

        # Verify the DataFrame columns and dtypes
        print("\nColumns in processed_data:")
        print(processed_data.columns.tolist())

        print("\nData types in processed_data:")
        print(processed_data.dtypes)

        print("\nSample data:")
        print(processed_data.head())

        # Verify the number of data points and date range
        print("\nNumber of data points:", len(processed_data))
        print("Date range:", processed_data['Date'].min(), "to", processed_data['Date'].max())

        # Save the scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # Load the scaler
        scaler = pickle.load(open('scaler.pkl', 'rb'))

        # Define feature names (must match the ones used in LiveNNStrategy)
        feature_names = ['rsi', 'macd', 'price_change']  # Replace with actual feature names used in LiveNNStrategy

        # Check if required features are present
        missing_features = [feat for feat in feature_names if feat not in processed_data.columns]
        if missing_features:
            print(f"Error: The following required features are missing from the data feed: {missing_features}")
            sys.exit(1)

        # Load the trained model
        model = NNModel(len(feature_names), 3)  # Adjust based on your model's architecture
        model.load(mdl_name)

        print("\nLoaded model and scaler successfully.")

        # Prepare data for Backtrader
        # Backtrader expects the Date column to be the index
        processed_data.set_index('Date', inplace=True)

        # Initialize Cerebro instances
        cerebro_nn = bt.Cerebro()
        cerebro_du = bt.Cerebro()

        # Add data feed
        bt_data = CustomPandasData(dataname=processed_data)
        cerebro_nn.adddata(bt_data)
        cerebro_du.adddata(bt_data)

        # Add strategies
        cerebro_nn.addstrategy(
            LiveNNStrategy,
            scaler=scaler,
            model=model,
            stop_loss=RUN['stop_loss'],
            leverage=LEVERAGE
        )
        cerebro_du.addstrategy(
            DummyStrategy,
            stop_loss=RUN['stop_loss'],
            leverage=LEVERAGE
        )

        # Set broker parameters
        initial_cash = 10000.0
        cerebro_nn.broker.setcash(initial_cash)
        cerebro_nn.broker.setcommission(commission=RUN['commission_fee'])
        cerebro_du.broker.setcash(initial_cash)
        cerebro_du.broker.setcommission(commission=RUN['commission_fee'])

        # Add analyzers
        cerebro_nn.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro_nn.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro_nn.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

        cerebro_du.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro_du.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro_du.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

        # Run strategies
        print("\nRunning LiveNNStrategy...")
        strategies_nn = cerebro_nn.run()
        strat_nn = strategies_nn[0]

        print("\nRunning DummyStrategy...")
        strategies_du = cerebro_du.run()
        strat_du = strategies_du[0]

        # Extract metrics
        def get_metrics(strategy, name):
            sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            drawdown = strategy.analyzers.drawdown.get_analysis().get('max', 0)
            time_return = strategy.analyzers.timereturn.get_analysis()
            if time_return:
                # Calculate average daily return
                daily_returns = list(time_return.values())
                avg_daily_return = np.mean(daily_returns)
                # Calculate annualized return
                annual_return = (1 + avg_daily_return) ** 252 - 1
            else:
                annual_return = 0
            return {
                "Strategy": name,
                "Sharpe Ratio": round(sharpe, 3),
                "Max Drawdown (%)": round(drawdown, 2),
                "Annualized Return (%)": round(annual_return * 100, 2),
                "Final Portfolio Value": round(strategy.broker.getvalue(), 2)
            }

        metrics_nn = get_metrics(strat_nn, "LiveNNStrategy")
        metrics_du = get_metrics(strat_du, "DummyStrategy")

        # Print and compare metrics
        results_df = pd.DataFrame([metrics_nn, metrics_du])
        print("\nComparison of Strategies:")
        print(results_df)

        # Save metrics to CSV
        results_df.to_csv(f"{RUN['reports']}{filename.split('.')[0]}_backtest_comparison_{suffix}.csv", index=False)

        # Extract portfolio value history
        hist_nn = strat_nn.analyzers.timereturn.get_analysis()
        hist_du = strat_du.analyzers.timereturn.get_analysis()

        # Convert to DataFrame for plotting
        df_nn = pd.DataFrame.from_dict(hist_nn, orient='index', columns=['LiveNNStrategy'])
        df_du = pd.DataFrame.from_dict(hist_du, orient='index', columns=['DummyStrategy'])
        df_combined = pd.concat([df_nn, df_du], axis=1).dropna()

        # Calculate cumulative returns
        df_combined = (1 + df_combined).cumprod()

        # Plotting
        plt.rcParams['font.size'] = 14
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#eeeeee')

        # Plot cumulative returns
        df_combined.plot(ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.set_title(f"{filename.split('.')[0]} Backtest Comparison")
        ax.legend()
        ax.grid(True)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

        # Save the plot
        plot_filename = f"{RUN['reports']}{filename.split('.')[0]}_backtest_comparison_{suffix}.png"
        fig.tight_layout()
        fig.savefig(plot_filename)
        plt.show()

        return {
            'du': (metrics_du["Final Portfolio Value"], metrics_du["Sharpe Ratio"], metrics_du["Max Drawdown (%)"], metrics_du["Annualized Return (%)"]),
            'nn': (metrics_nn["Final Portfolio Value"], metrics_nn["Sharpe Ratio"], metrics_nn["Max Drawdown (%)"], metrics_nn["Annualized Return (%)"])
        }
    except Exception as e:
        print(e)
if __name__ == '__main__':
        backtest_results = run_backtest_single_coin(run_conf, 'BTCUSDT.csv', mdl_name='model.h5', suffix="test")
        print("\nBacktest Results:", backtest_results)

import os
import sys
import traceback
import datetime as dt
import numpy as np
import pandas as pd
import backtrader as bt
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from NNModel_lib import NNModel
from sklearn.preprocessing import StandardScaler
from config import RUN as run_conf

STOP_LOSS = 0.05
LEVERAGE = 19

class LiveNNStrategy(bt.Strategy):
    params = (
        ('stop_loss', STOP_LOSS),
        ('leverage', LEVERAGE),
        ('scaler', None),  # Pretrained scaler
        ('model', None),   # Pretrained NN model
    )

    def __init__(self):
        if self.p.scaler is None or self.p.model is None:
            raise ValueError("LiveNNStrategy requires both 'scaler' and 'model'")
        self.order = None
        self.entryprice = None
        self.log_data = []  # Store log data for later analysis

    def next(self):
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]

        # Extract past data up to the current bar
        lookback_data = {
            'Date': [self.data.datetime.date(-i) for i in range(len(self))],
            'Open': [self.data.open[-i] for i in range(len(self))],
            'High': [self.data.high[-i] for i in range(len(self))],
            'Low': [self.data.low[-i] for i in range(len(self))],
            'Close': [self.data.close[-i] for i in range(len(self))],
            'Volume': [self.data.volume[-i] for i in range(len(self))],
        }
        df = pd.DataFrame(lookback_data)

        # Compute features using only past data
        try:
            df = TecnicalAnalysis.compute_oscillators(df)
            df = TecnicalAnalysis.find_patterns(df)
            df = TecnicalAnalysis.add_timely_data(df)
        except Exception as ex:
            self.log(f"Error computing features: {ex}")
            return

        # Drop unnecessary columns
        cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
        features = df.drop(columns=cols_to_drop, errors='ignore').iloc[-1:].values

        # Scale features
        try:
            scaled_features = self.p.scaler.transform(features)
        except Exception as ex:
            self.log(f"Error scaling features: {ex}")
            return

        # Get prediction from the NN model
        try:
            prediction = self.p.model.predict(scaled_features)
            if isinstance(prediction, (np.ndarray, list)):
                signal = prediction[0]
            else:
                signal = prediction
            self.log(f"Predicted signal: {signal}")
        except Exception as ex:
            self.log(f"Error during prediction: {ex}")
            signal = HOLD

        # Compute order size
        portfolio_value = self.broker.getvalue()
        full_size = (portfolio_value * self.p.leverage) / current_price
        order_size = full_size * 0.05

        # Execute trading logic
        if not self.position:
            if signal == BUY:
                self.order = self.buy(size=order_size)
                self.entryprice = current_price
                self.log(f"BUY order placed at {current_price:.2f} for size {order_size:.4f}")
        else:
            if current_price < self.entryprice * (1 - self.p.stop_loss):
                self.log(f"STOP LOSS HIT at {current_price:.2f}")
                self.order = self.sell(size=self.position.size)
            elif signal == SELL:
                self.log(f"SELL signal received at {current_price:.2f}")
                self.order = self.sell(size=self.position.size)

    def log(self, txt):
        dtobj = self.data.datetime.datetime(0)
        print(f"{dtobj.isoformat()} {txt}")
        self.log_data.append(f"{dtobj.isoformat()} {txt}")

    def stop(self):
        with open('strategy_log.txt', 'w') as f:
            for entry in self.log_data:
                f.write(entry + '\n')

# ======================================================================
# Main Backtest Function
# ======================================================================
def run_backtest():
    full_csv = os.path.join(run_conf['folder'], 'ALGOUSDT.csv')
    try:
        data = pd.read_csv(full_csv)
    except Exception as ex:
        print("Error loading CSV data:", ex)
        sys.exit(1)

    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)

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

    # Fit scaler on training data
    cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Asset_name']
    data_train_features = data_train.drop(columns=cols_to_drop, errors='ignore')
    data_train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_train_features.dropna(inplace=True)

    scaler = StandardScaler()
    scaler.fit(data_train_features)

    # Prepare backtest data
    data_backtest['label'] = np.nan
    data_backtest.set_index('Date', inplace=True)

    # Load NN model
    model_path = r'model.h5'
    num_features = len(data_train_features.columns)
    model = NNModel(num_features, 3)  # Assuming 3 classes: BUY, HOLD, SELL
    try:
        model.load(model_path)
    except Exception as ex:
        print(f"Error loading model: {ex}")
        sys.exit(1)

    # Set up Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(run_conf.get('initial_capital', 10000.0))
    cerebro.broker.setcommission(commission=run_conf.get('commission fee', 0.001))

    bt_data = PandasLabelData(dataname=data_backtest)
    cerebro.adddata(bt_data)
    cerebro.addstrategy(LiveNNStrategy, scaler=scaler, model=model)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

    print("Running Live NN Strategy...")
    strategies = cerebro.run()
    strat = strategies[0]

    print("\nPortfolio Value:", cerebro.broker.getvalue())
    cerebro.plot()

if __name__ == '__main__':
    run_backtest()

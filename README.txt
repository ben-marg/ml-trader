This code has been adapted from the below research paper:

https://www.sciencedirect.com/science/article/pii/S0957417423023084?via%3Dihub

This code is a thorough exploration of all the topics discussed in the paper, as well as an attempt at practical implementation. 
The following changes/developments were made:

1. Library upgrades/modernization
2. Backtesting strategy with Backtrader
3. Thorough testing with unseen data
4. A framework for a live implementation via Binance Futures.



The main conclusions of the study are:

1. The current testing framework leads to severe overfitting. That is, the model performs exceptionally well on the training data, but exceptionally poorly on the test data.
2. The model has the potential for good performance, but needs some significant changes 
3. Integrating a testing pipeline with Backtrader directly will lead to much better results.
4. The number of technical indicators used was excessive, and only made overfitting more of an issue. It is best to limit the number of indicators used to avoid overfitting and unneeded complexity. 

For that reason, this code was open-sourced. 
______________________________________________________________



USE INSTRUCTIONS:

1. Follow the instructions in the original paper to train/test the model, except by sure to ommit the last 2-3 years of data to save for testing. 


2. Run btest, btest2, ... btest4 to view the backtesting results of the strategy. 

Changing the time period of the testing will result in different results. It is critical to study the effects of overfitting by fitting the model on all but the last N% of the data, where N can range from 10-30%. Doing so will reveal incrdible performance during the training period, but dismal performance during the testing period. 

POTENTIAL FIXES:

The most obvious fix for the overfitting would be to add regularization. However, it can be difficult to justify optimizing for classification accuracy when what we really care about is the backtest performance. At the end of the day, we could be right a lot and still lose money, depending on how big/small the wins are compared to the losses. For this reason, the current pipeline from a practical perspective doesn't make much sense. 

So what can we do instead?

One solution, albeit complicated, would be to modify the pipeline as follows:

Current:

Train model --> Maximize classification accuracy 
Test model --> Classification accuracy 
Test model --> Backtest performance 



Modified:

Train model --> Backtest performance
Test model  --> Backtest performance








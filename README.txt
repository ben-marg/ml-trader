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
4. The number of technical indicators used was excessive, and only made overfitting more of an issue. It is best to limit the number of indiactors used to avoid overfitting and unneeded complexity. 

For that reason, this code was open-sourced. If any progress is made or anyone has any feedback, feel free to reach out to me  at beniamarg3@gmail.com. I have been working on some of the ideas developed in this study, and they have shown significant promise. 




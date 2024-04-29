# region imports
from AlgorithmImports import *
from transformers import AutoTokenizer
from transformers import pipeline
import joblib
import talib
from helper_function import *
import pytz
import pandas as pd
# endregion


class BuyAndHoldBitcoinEthAlgorithm(QCAlgorithm):
    def Initialize(self):
        # INS in-sample
        # self.SetStartDate(2022, 4, 10)
        # self.SetEndDate(2023,3,31)
        
        # Out of Sample (OOS) 1 
        # self.SetStartDate(2021, 5, 14)
        # self.SetEndDate(2021, 8, 10)

        # OOS 2
        # self.SetStartDate(2021, 9, 10)
        # self.SetEndDate(2021, 12, 10)


        # OOS 3
        # self.SetStartDate(2024, 3, 31)
        # self.SetEndDate(2024, 4, 27)

        # stress test
        self.SetStartDate(2021, 12, 1)
        self.SetEndDate(2022, 7, 1)

        self.SetCash(10000000)

        # Add Bitcoin data
        self.AddCrypto("BTCUSD", Resolution.Daily)
        self.AddCrypto("ETHUSD", Resolution.Daily)

        # Flag to check if Bitcoin has been bought
        self.bought_btc = False

    def OnData(self, data):
        if not self.bought_btc:
            # Buy Bitcoin & ETH 80/20
            self.SetHoldings("BTCUSD", 0.8)
            self.SetHoldings("ETHUSD", 0.2)
            self.bought_btc = True

        # Check if it's the last day of the backtest
        if self.Time.date() >= self.EndDate.date():
            self.Liquidate()  # Liquidate all positions

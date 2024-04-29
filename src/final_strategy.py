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

class MySlippageModel:
    """Class that defines our slippage model"""
    def GetSlippageApproximation(self, asset: Security, order: Order) -> float:
        """Slippage formula"""
        slippage = asset.Price * 0.0001 * np.log10(2*float(order.AbsoluteQuantity))
        return slippage


class FatGreenHorse(QCAlgorithm):
    """Main Algo Class"""
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

        self.SetCash(10000000)  # Setting initial Cash
        self.SetWarmUp(30)  # Warm up for 30 days

        # Adding instruments
        self.AddEquity("SPY", Resolution.Daily)
        self.btc_security = self.AddCrypto("BTCUSD", Resolution.Daily)
        self.btc_symbol = self.btc_security.symbol
        self.eth_security = self.AddCrypto("ETHUSD", Resolution.Daily)
        self.sol_security = self.AddCrypto("SOLUSD", Resolution.Daily)
        self.symbol = self.AddEquity("coin", Resolution.Daily).Symbol
        
        # Slippage (uncomment for slippage)
        self.btc_security.SetSlippageModel(MySlippageModel())

        # Adding data sources
        self.dataset_symbol = self.AddData(TiingoNews, self.symbol, Resolution.Daily).Symbol
        self.treas = self.add_data(USTreasuryYieldCurveRate, "USTYCR", Resolution.Daily).symbol
        self.vix = self.add_data(CBOE, "VIX", Resolution.Daily).symbol
        
        # setting up FINBERT
        bert_res = self.set_up_bert()
        self.pipe = bert_res[0]
        self.tokenizer = bert_res[1]
        self.signal_model = self.load_rf_model()

        # Initializing empty DF for Sentiment scores
        self.btc_sentiment_df = pd.DataFrame()
        self.coin_sentiment_df = pd.DataFrame()

        # Initiliazing list to store preds
        self.predicted_15dars = []

        # Initializing runtime flags
        self.first_run = True
        self.bought_btc = False
        self.shorted_btc = False
        self.old_portfolio_value = 0
        self.days_portfolio_decline = 0


    def set_up_bert(self) -> tuple:
        """Loads FINBERT from QC object store"""
        path = self.ObjectStore.GetFilePath("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
        pipe = pipeline("text-classification", model=path)

        self.debug(f'{pipe("bitcoin pushes to fresh record high after breaching $62,000 yesterday")}')
        tokenizer = AutoTokenizer.from_pretrained(path)

        return pipe, tokenizer

    def load_rf_model(self):
        """Loads trained Random Forest Model from QC object store"""
        path = self.ObjectStore.GetFilePath("group_4_crypto_trading_with_sentiment_sprin_2024/random_forest_model.pkl")
        return joblib.load(path)

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        # Risk management 1 - If value drops by more than 5% from previous OnData call, Liquidate
        if self.portfolio.TotalPortfolioValue < self.old_portfolio_value * 0.95:
            self.debug(f"Liquidating at: {self.time}")
            self.Liquidate()

        # Check if data contains BTCUSD info
        if data.ContainsKey("BTCUSD"):
            # Getting necessary historical BTC data for passing to our RF ML model
            df = self.History(self.btc_symbol, 37).droplevel(0) 
            
            # Extracting technical indicators
            ti = technical_indicators(df)
            stoch_fastk = ti['STOCH_FASTK'].values[-1]
            stoch_fastd =  ti['STOCH_FASTD'].values[-1]
            stoch_fast_d_rolling_10 =  ti['STOCH_FASTD'].rolling(10).sum().values[-1]
            aroonosc = ti['AROONOSC'].values[-1][-1]
            mfi = ti['MFI'].values[-1]
            roc = ti['ROC'].values[-1]
            rsi = ti['RSI'].values[-1]
            roc_rolling_10 = ti['RSI'].rolling(10).sum().values[-1]
            willr = ti['WILLR'].values[-1]
            mom_rolling_10 = ti['MOM'].rolling(10).sum().values[-1]
            natr = ti['NATR'].values[-1]
            mom = ti['MOM'].values[-1]
            cmo = ti['CMO'].values[-1]
            willr_rolling_10 = ti['WILLR'].rolling(10).sum().values[-1]
            macdhist = ti['MACDHIST'].values[-1]
            stoch_fastd_rolling_10 =  ti['STOCH_FASTD'].rolling(10).sum().values[-1]
            plus_di_rolling_10 = ti['PLUS_DI'].rolling(10).sum().values[-1]
            macdhist_rolling_10 = ti['MACDHIST'].rolling(10).sum().values[-1]
            plus_di = ti['PLUS_DI'].values[-1]
            stoch_fastk_rolling_10 = ti['STOCH_FASTK'].rolling(10).sum().values[-1]
            cci = ti['CCI'].values[-1]
            ULTOSC_rolling_10 = ti['ULTOSC'].rolling(10).sum().values[-1]
            ultosc = ti['ULTOSC'].values[-1]
            minus_di = ti['MINUS_DI'].values[-1]
            mfi_rolling_10 = ti['MFI'].rolling(10).sum().values[-1]
            macd = ti['MACD'].values[-1]
            cci_rolling_10 = ti['CCI'].rolling(10).sum().values[-1]
            ht_phasor_quad_rolling_10 = ti['HT_PHASOR_quadrature'].rolling(10).sum().values[-1]
            trange_rolling_10 = ti['TRANGE'].rolling(10).sum().values[-1]
            ht_dcphase = ti['HT_DCPHASE'].values[-1]

            # Getting last 37 days of News data from TIINGO
            history_coin = process_coinbase(self.History(self.dataset_symbol, 37,  Resolution.Daily).droplevel(0))
            bitcoin_news = history_coin[1]
            coinbase_news = history_coin[0]

            # Filtering new news data since the last day
            if not self.first_run:
                new_btc_news = bitcoin_news[bitcoin_news.index > self.last_day.replace(tzinfo=pytz.UTC)]
                new_coin_news = coinbase_news[coinbase_news.index > self.last_day.replace(tzinfo=pytz.UTC)]
                self.last_day = self.Time
            else:
                new_btc_news = bitcoin_news
                new_coin_news = coinbase_news
                self.last_day = self.Time
                self.first_run = False

            # Analyzing sentiment of new news data
            new_btc_news, new_coin_news = get_sentiment(self, new_btc_news, new_coin_news)

            # Concatenating new news data with existing data
            if new_btc_news.shape[0] > 0:
                self.btc_sentiment_df = pd.concat([self.btc_sentiment_df, new_btc_news], ignore_index=True)
            
            if new_btc_news.shape[0] > 0:
                self.coin_sentiment_df = pd.concat([self.coin_sentiment_df, new_coin_news], ignore_index=True)

            # Calculating sentiment scores
            sentiment_scores = get_sentiment_scores(self.btc_sentiment_df, self.coin_sentiment_df)

            coinbase_neutral_count_rolling_10 = sentiment_scores[0]
            coinbase_positive_avg_score_rolling_10 = sentiment_scores[1]
            coinbase_positive_avg_score_rolling_30 = sentiment_scores[2]
            btc_total_news_score_rolling_30 = sentiment_scores[3]
            coinbase_total_news_score = sentiment_scores[4]

            # Creating dataframe with feature order for model prediction
            indicators_order = [aroonosc, roc_rolling_10, mom_rolling_10, roc, mfi, willr, rsi, cmo,
                                mom, natr, willr_rolling_10, macdhist, stoch_fast_d_rolling_10, 
                                plus_di, plus_di_rolling_10, stoch_fastk_rolling_10, cci, 
                                stoch_fastk, macdhist_rolling_10, btc_total_news_score_rolling_30, 
                                ultosc, coinbase_positive_avg_score_rolling_10, stoch_fastd, 
                                ht_phasor_quad_rolling_10, coinbase_total_news_score, 
                                coinbase_neutral_count_rolling_10, minus_di, trange_rolling_10, 
                                coinbase_positive_avg_score_rolling_30, ht_dcphase]
            
            df = pd.DataFrame([indicators_order], columns=['AROONOSC', 'ROC_rolling_10', 'MOM_rolling_10', 'ROC', 
                                                            'MFI', 'WILLR', 'RSI', 'CMO', 'MOM', 'NATR', 
                                                            'WILLR_rolling_10', 'MACDHIST', 'STOCH_FASTD_rolling_10', 
                                                            'PLUS_DI', 'PLUS_DI_rolling_10', 'STOCH_FASTK_rolling_10', 
                                                            'CCI', 'STOCH_FASTK', 'MACDHIST_rolling_10', 
                                                            'btc_total_news_score_rolling_30', 'ULTOSC', 
                                                            'coinbase_positive_avg_score_rolling_10', 'STOCH_FASTD',
                                                            'HT_PHASOR_quadrature_rolling_10', 'coinbase_total_news_score', 
                                                            'coinbase_neutral_count_rolling_10', 'MINUS_DI', 'TRANGE_rolling_10', 
                                                            'coinbase_positive_avg_score_rolling_30', 'HT_DCPHASE'])
            
            # using the above features to get our signal for tomorrow
            preds = self.signal_model.predict(df)
            self.predicted_15dars.append(preds[0])

            # Implementing main strategy based on predicted signal
            if len(self.predicted_15dars) >= 0:
                last_1_day = self.predicted_15dars[-1]
                buy = 0
                sell = 0
                if last_1_day > 0.01:
                    buy = 1
                    sell = 0
                elif last_1_day < -0.005:
                    buy = 0
                    sell = 1
                
                # Getting importance of prediction from custom importance function
                importance = abs(apply_importance_function(last_1_day))
                importance += 0.4

                if importance < 0.51:
                    importance = 0.51
                elif importance > 1:
                    importance = 0.99
                
                # Getting bet size based on importance and modified kelly kriterion
                kelly_fraction = (1 * abs(importance - (1 - importance))) / 1
                if kelly_fraction > 0.8:
                    kelly_fraction = 0.8
                elif kelly_fraction < 0:
                    kelly_fraction = 0

                # Trading based on strategy
                if (buy == 1):
                    self.SetHoldings("BTCUSD", kelly_fraction * 0.8)
                    self.SetHoldings("ETHUSD", kelly_fraction * 0.2)
                    # self.SetHoldings("SOLUSD", kelly_fraction * 0.2)
                    self.bought_btc = True
                    self.shorted_btc = False
                elif (sell == 1):
                    # We take less risk when going short
                    self.SetHoldings("BTCUSD", -0.5 * kelly_fraction * 0.8)
                    self.SetHoldings("ETHUSD", -0.5 * kelly_fraction * 0.2)
                    # self.SetHoldings("SOLUSD", -0.5 * kelly_fraction * 0.2)
                    self.bought_btc = False
                    self.shorted_btc = True
            
            self.debug(f"time: {self.time}, pred: {preds[0]}, importance: {importance}, kelly_fraction: {kelly_fraction}")

            # Liquidate if portfolio value falls for nine consecutive days - Risk management 2
            if (self.Portfolio.TotalPortfolioValue < self.old_portfolio_value) and not self.first_run:
                self.days_portfolio_decline += 1
            else:
                self.days_portfolio_decline = 0

            if self.days_portfolio_decline >= 9:
                self.Liquidate()

            self.old_portfolio_value = self.Portfolio.TotalPortfolioValue

        current_date = self.Time
        
        # Check if it's the last day of the backtest
        if current_date.date() >= self.EndDate.date():
            self.Liquidate()  # Liquidate all positions

#region imports
from AlgorithmImports import *
from transformers import AutoTokenizer
from transformers import pipeline
import joblib
import talib
#endregion


def get_news_count_d(df, label) -> int:
    """Calculate the count of news articles for a specific label."""
    news_count = (
        df[df["label"] == label].groupby(["date_dt"]).count()["News"].reset_index()
    )
    news_count.columns = ["date_dt", f"{label}_count"]
    
    return news_count


# function to get average sentiment score
def get_avg_sentiment_d(df, label) -> float:
    """Calculate the average sentiment score for a specific label."""
    avg_sentiment = (
        df[df["label"] == label].groupby(["date_dt"])["score"].mean().reset_index()
    )
    avg_sentiment.columns = ["date_dt", f"{label}_avg_score"]
    return avg_sentiment

# making a function to return setiement
def get_bert_sentiment(pipe, news_description) -> pd.Series:
    """Get sentiment using FINBERT"""
    try:
        sent = pipe(news_description) # Get sentiment using BERT pipeline
        label = sent[0]['label'] # Extract sentiment label
        score = sent[0]['score'] # Extract sentiment score

        # attempt at making a consolidated label
        if label == 'positive':
            combined_label = 1
        elif label == 'negative':
            combined_label = -1
        else:
            combined_label = 0

        # Return sentiment information
        return pd.Series([label, score, combined_label])

    except IndexError:
        # Return for bad data
        return pd.Series(['BAD Data', 0, 0])

def process_coinbase(df) -> pd.DataFrame:
    """Process data from Coinbase."""
    coinbase = df
    # Drop the null values
    coinbase = coinbase.dropna(subset=['articleid', 'description'])
    # Drop if News is ' '
    coinbase = coinbase[coinbase['description'] != ' ']


    coinbase = coinbase[['articleid', 'description']]
    coinbase.drop_duplicates(subset=['articleid', 'description'], inplace=True)

    # turn all News to lowercase
    coinbase['description'] = coinbase['description'].str.lower()

    # only keep row if bitcoin or btc is in string
    bitcoin = coinbase[coinbase['description'].str.contains('bitcoin|btc')]

    # Return the dataframe
    return coinbase, bitcoin

def compute_momentum_indicators(df) -> pd.DataFrame:
    """Compute momentum indicators."""
    # Prepare data
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open1 = df['open'].values
    volume = df['volume'].values
    
    # Initialize dictionary to hold results
    results = {}
    
    # Compute momentum indicators
    results['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    results['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    results['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    results['BOP'] = talib.BOP(open1, high, low, close)
    results['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    results['CMO'] = talib.CMO(close, timeperiod=14)
    results['MACD'], results['MACDSIGNAL'], results['MACDHIST'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    results['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    results['MOM'] = talib.MOM(close, timeperiod=10)
    results['RSI'] = talib.RSI(close, timeperiod=14)
    results['STOCH_FASTK'], results['STOCH_FASTD'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    results['STOCHRSI_K'], results['STOCHRSI_D'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    results['TRIX'] = talib.TRIX(close, timeperiod=30)
    results['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    results['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
    
    # Convert dictionary to DataFrame
    results_df = pd.DataFrame(results)
    results_df.index = df.index
    return results_df

def compute_extended_volume_indicators(df) -> pd.DataFrame:
    """Get extended volumne indicators"""
    # Ensure numerical columns are floats
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values
    volume = df['volume'].astype(float).values  # Ensure volume is float for calculations
    open_ = df['open'].astype(float).values  # Add open prices for indicators that might use it
    
    # Initialize dictionary to hold results
    results = {}
    
    # Compute original volume indicators
    results['AD'] = talib.AD(high, low, close, volume)
    results['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    results['OBV'] = talib.OBV(close, volume)
    
    results_df = pd.DataFrame(results)
    
    results_df.index = df.index
    
    return results_df

def compute_trend_indicators(df) -> pd.DataFrame:
    """Compute trend indicators"""
    # Ensure all necessary columns are floats
    open_ = df['open'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values
    volume = df['volume'].astype(float).values  # Necessary for MAMA
    
    # Initialize dictionary to hold results
    results = {}
    
    # Moving Averages
    results['SMA'] = talib.SMA(close, timeperiod=14)
    results['EMA'] = talib.EMA(close, timeperiod=14)
    results['WMA'] = talib.WMA(close, timeperiod=14)
    results['DEMA'] = talib.DEMA(close, timeperiod=14)
    results['TEMA'] = talib.TEMA(close, timeperiod=14)
    results['TRIMA'] = talib.TRIMA(close, timeperiod=14)
    results['KAMA'] = talib.KAMA(close, timeperiod=14)
    mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)  # MAMA returns MAMA and FAMA
    results['MAMA'] = mama
    results['FAMA'] = fama
    results['T3'] = talib.T3(close, timeperiod=14, vfactor=0.7)
    
    # Directional Movement Indicators
    results['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    results['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    results['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    
    # Others
    results['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    results['AROON_DOWN'], results['AROON_UP'] = talib.AROON(high, low, timeperiod=14)
    results['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    # results['VI_PLUS'], results['VI_MINUS'] = talib.VI(high, low, close, timeperiod=14)
    
    # Convert dictionary to DataFrame and ensure it aligns with the original DataFrame's index
    results_df = pd.DataFrame(results, index=df.index)
    
    return results_df

def compute_volatility_indicators(df) -> pd.DataFrame:
    """Compute volatility indicators"""
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values
    
    # Initialize dictionary to hold results
    results = {}
    
    # Bollinger Bands
    results['BBANDS_UPPER'], results['BBANDS_MIDDLE'], results['BBANDS_LOWER'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    
    # Average True Range
    results['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    
    # Normalized Average True Range
    results['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    
    # True Range
    results['TRANGE'] = talib.TRANGE(high, low, close)
    
    # Chandelier Exit (custom calculation, not a direct TA-Lib function)
    # Typically uses a 22-day period and a multiplier of 3 times the ATR
    atr_22 = talib.ATR(high, low, close, timeperiod=22)
    highest_high_22 = talib.MAX(high, timeperiod=22)
    lowest_low_22 = talib.MIN(low, timeperiod=22)
    results['CHANDELIER_EXIT_LONG'] = highest_high_22 - (atr_22 * 3)
    results['CHANDELIER_EXIT_SHORT'] = lowest_low_22 + (atr_22 * 3)
    
    # Convert dictionary to DataFrame and ensure it aligns with the original DataFrame's index
    results_df = pd.DataFrame(results, index=df.index)
    
    return results_df

def compute_price_transform_indicators(df) -> pd.DataFrame:
    """Get price transform indicators"""
    # Ensure all necessary columns are floats
    open_ = df['open'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values
    
    # Initialize dictionary to hold results
    results = {}
    
    # Weighted Close Price
    results['WCLPRICE'] = talib.WCLPRICE(high, low, close)
    
    # Typical Price
    results['TYPPRICE'] = talib.TYPPRICE(high, low, close)
    
    # Median Price
    results['MEDPRICE'] = talib.MEDPRICE(high, low)
    
    # Price Rate of Change
    results['ROC'] = talib.ROC(close, timeperiod=10)
    
    # Average Price
    results['AVGPRICE'] = talib.AVGPRICE(open_, high, low, close) 
    
    # Convert dictionary to DataFrame and ensure it aligns with the original DataFrame's index
    results_df = pd.DataFrame(results, index=df.index)
    
    return results_df



def compute_cycle_indicators(df) -> pd.DataFrame:
    """Gey cycle indicators"""
    # Ensure 'close' column is a float
    close = df['close'].astype(float).values
    
    # Initialize dictionary to hold results
    results = {}
    
    # Hilbert Transform - Dominant Cycle Period
    results['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    
    # Hilbert Transform - Dominant Cycle Phase
    results['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    
    # Hilbert Transform - Phasor Components
    results['HT_PHASOR_inphase'], results['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    
    # Hilbert Transform - SineWave
    results['HT_SINE'], results['HT_LEADSINE'] = talib.HT_SINE(close)
    
    # Hilbert Transform - Trend vs Cycle Mode
    results['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
    
    # Convert dictionary to DataFrame and ensure it aligns with the original DataFrame's index
    results_df = pd.DataFrame(results, index=df.index)
    
    return results_df


def technical_indicators(df) -> pd.DataFrame:
    """Compute various technical indicators."""
    all_results = pd.concat([compute_momentum_indicators(df),
         compute_trend_indicators(df),
         compute_price_transform_indicators(df),
         compute_volatility_indicators(df),
         compute_cycle_indicators(df)
         ], axis=1
    )
    return all_results


def get_sentiment(object_class, bitcoin_news, coinbase_news):
    """Get sentiment assigned by Finbert"""
    if len(bitcoin_news) != 0:
        bitcoin_news = bitcoin_news.rename(columns={'description': 'News'})
        bitcoin_news['News'] = bitcoin_news['News'].apply(lambda x: object_class.tokenizer(x, truncation=True, max_length=512)['input_ids'])
        bitcoin_news['News'] = bitcoin_news['News'].apply(lambda x: object_class.tokenizer.decode(x))
        bitcoin_news[['label', 'score', 'combined_label']] = bitcoin_news['News'].apply(lambda x: get_bert_sentiment(object_class.pipe, x))
        bitcoin_news['Date'] = pd.to_datetime(bitcoin_news.index, format='mixed', utc=True)

        # make a new column 'date' and only keep the day from the 'Date' column
        bitcoin_news["date_dt"] = bitcoin_news['Date'].dt.date
        bitcoin_news["date_hr"] = bitcoin_news['Date'].dt.hour
    
    if len(coinbase_news) != 0:
        coinbase_news = coinbase_news.rename(columns={'description': 'News'})
        coinbase_news['News'] = coinbase_news['News'].apply(lambda x: object_class.tokenizer(x, truncation=True, max_length=512)['input_ids'])
        coinbase_news['News'] = coinbase_news['News'].apply(lambda x: object_class.tokenizer.decode(x))
        coinbase_news[['label', 'score', 'combined_label']] = coinbase_news['News'].apply(lambda x: get_bert_sentiment(object_class.pipe, x))
        coinbase_news['Date'] = pd.to_datetime(coinbase_news.index, format='mixed', utc=True)
        # make a new column 'date' and only keep the day from the 'Date' column
        coinbase_news["date_dt"] = coinbase_news['Date'].dt.date
        coinbase_news["date_hr"] = coinbase_news['Date'].dt.hour

    return bitcoin_news, coinbase_news


def get_sentiment_scores(bitcoin_news, coinbase_news):
    """Get latest day Sentiment Scores"""

    # Get the news count for each label
    pn = get_news_count_d(bitcoin_news, "positive")
    nn = get_news_count_d(bitcoin_news, "negative")
    neun = get_news_count_d(bitcoin_news, "neutral")

    # Get the average sentiment score for each label
    pn_avg = get_avg_sentiment_d(bitcoin_news, "positive")
    nn_avg = get_avg_sentiment_d(bitcoin_news, "negative")
    neun_avg = get_avg_sentiment_d(bitcoin_news, "neutral")

    # # Merge the dataframes
    merged_d = pd.merge(pn, nn, on="date_dt", how="outer")
    merged_d = pd.merge(merged_d, neun, on="date_dt", how="outer")
    merged_d = pd.merge(merged_d, pn_avg, on="date_dt", how="outer")
    merged_d = pd.merge(merged_d, nn_avg, on="date_dt", how="outer")
    merged_d = pd.merge(merged_d, neun_avg, on="date_dt", how="outer")

    # Fill NaN values with 0
    merged_d = merged_d.fillna(0)

    # total news count
    merged_d["total_news_count"] = (
        merged_d["positive_count"] + merged_d["negative_count"] + merged_d["neutral_count"]
    )

    merged_d["total_news_score"] = (
        merged_d["positive_avg_score"] * merged_d["positive_count"]
    ) - (merged_d["negative_avg_score"] * merged_d["negative_count"])

    merged_d["signal"] = (merged_d["positive_count"] - merged_d["negative_count"]) / (
        merged_d["positive_count"] + merged_d["negative_count"]
    )
        
    merged_d.columns = [f"btc_{c}" for c in merged_d.columns]

    # Get the news count for each label
    pn = get_news_count_d(coinbase_news, "positive")
    nn = get_news_count_d(coinbase_news, "negative")
    neun = get_news_count_d(coinbase_news, "neutral")

    # Get the average sentiment score for each label
    pn_avg = get_avg_sentiment_d(coinbase_news, "positive")
    nn_avg = get_avg_sentiment_d(coinbase_news, "negative")
    neun_avg = get_avg_sentiment_d(coinbase_news, "neutral")

    # Merge the dataframes
    merged_dc = pd.merge(pn, nn, on="date_dt", how="outer")
    merged_dc = pd.merge(merged_dc, neun, on="date_dt", how="outer")
    merged_dc = pd.merge(merged_dc, pn_avg, on="date_dt", how="outer")
    merged_dc = pd.merge(merged_dc, nn_avg, on="date_dt", how="outer")
    merged_dc = pd.merge(merged_dc, neun_avg, on="date_dt", how="outer")

    # Fill NaN values with 0
    merged_dc = merged_dc.fillna(0)

    # total news count
    merged_dc["total_news_count"] = (
        merged_dc["positive_count"]
        + merged_dc["negative_count"]
        + merged_dc["neutral_count"]
    )

    merged_dc["total_news_score"] = (
        merged_dc["positive_avg_score"] * merged_dc["positive_count"]
    ) - (merged_dc["negative_avg_score"] * merged_dc["negative_count"])


    merged_dc["signal"] = (merged_dc["positive_count"] - merged_dc["negative_count"]) / (
        merged_dc["positive_count"] + merged_dc["negative_count"]
    )

    merged_dc.columns = [f"coinbase_{c}" for c in merged_dc.columns]

    coinbase_neutral_count_rolling_10 = merged_dc['coinbase_neutral_count'].rolling(10).sum().values[-1]
    coinbase_positive_avg_score_rolling_10 = merged_dc['coinbase_positive_avg_score'].rolling(10).sum().values[-1]
    coinbase_positive_avg_score_rolling_30 = merged_dc['coinbase_positive_avg_score'].rolling(30).sum().values[-1]
    btc_total_news_score_rolling_30 = merged_d['btc_total_news_score'].rolling(30).sum().values[-1]
    coinbase_total_news_score = merged_dc['coinbase_total_news_score'].values[-1]

    return (coinbase_neutral_count_rolling_10, coinbase_positive_avg_score_rolling_10, coinbase_positive_avg_score_rolling_30, 
           btc_total_news_score_rolling_30, coinbase_total_news_score)

def apply_importance_function(x):
    """Applies importance based on piecewise function"""
    if x < -0.05:
        return -1
    elif -0.05 <= x < -0.004:
        return 16.433*(x) - 0.1777
    elif -0.004 <= x < 0:
        return 60 * x
    elif 0 <= x < 0.01:
        return 40 * x
    elif 0.01 <= x <= 0.05:
        return 14.975*(x) + 0.25
    else:
        return 1

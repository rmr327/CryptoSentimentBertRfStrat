# A Novel Algorithmic Cryptocurrency Trading Strategy with Local LLM Sentimental Analysis, Memory Features & Random Forest Models
### by Rakeen Rouf, Tianji Rao, Mike Lund, Ching-Lung Hsu

[![PythonCiCd](https://github.com/rmr327/cicd_python_template/actions/workflows/python_ci_cd.yml/badge.svg)](https://github.com/rmr327/cicd_python_template/actions/workflows/python_ci_cd.yml)

---
## Link to experimentation Repo 

(Rakeen lead this team concurrently for doing the research groound work for our strategy)

https://github.com/nogibjj/Flamingo-ML

## Link to Quant Connect Code 

This project was completed on Quant Connect, this repo just serves as a showcase. Please access orginal environment via the link below.

[Backtest Link Quant Connect](https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_78aa6cba58fcfd7865f5bd0c64928241.html)

## Abstract

Cryptocurrencies are increasingly recognized as a compelling asset class, marked by swiftly growing market capitalization and low barriers to entry for trading. However, volatile and speculative cryptocurrency prices pose significant challenges for return prediction using traditional financial approaches. Our team undertook a novel investigation in this study to delineate the complex relationship between cryptocurrency returns and human sentiment. We developed a supervised machine learning framework that merges fundamental and technical indicators with cutting-edge sentiment analysis from various data streams. Our methodology incorporated a baseline linear regression model, further enhanced by a Random Forest model. These models underwent rigorous testing to confirm their robustness and reliability in making predictions. Our findings represent a substantial advancement in predictive accuracy and model interpretability, offering crucial insights for investment strategies within the cryptocurrency domain.

## Experimentation Data Source

<img width="488" alt="image" src="https://github.com/rmr327/CryptoSentimentBertRfStrat/assets/36940292/f76aeb7f-9369-4e83-be2f-5c7ebd28020c">

## Experiments

<img width="1008" alt="image" src="https://github.com/rmr327/CryptoSentimentBertRfStrat/assets/36940292/3c4365f5-48aa-4816-b00a-096d13e37773">

## Predicted 15 Day moving Average Returns (15DAR) Transformation to Daily Direction and Magnitude Prediction

<img width="567" alt="image" src="https://github.com/rmr327/CryptoSentimentBertRfStrat/assets/36940292/9234b560-769b-487e-b3bb-b55d58b1fc29">

## Result Vs Literature

Note, time period in literature was different from our model due to data availability.

<img width="755" alt="image" src="https://github.com/rmr327/CryptoSentimentBertRfStrat/assets/36940292/39ee51a8-eda4-48f1-b620-8dc8c52b6758">

## Algorithmic Trading Strategy

<img width="522" alt="image" src="https://github.com/rmr327/CryptoSentimentBertRfStrat/assets/36940292/6e3f4364-538a-4334-b974-1d8cc6ef1969">

## Trading Performance
<img width="1179" alt="image" src="https://github.com/rmr327/CryptoSentimentBertRfStrat/assets/36940292/b4c2b4b3-0462-425a-ad17-ed06835dfca7">

[Backtest Link Quant Connect](https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_78aa6cba58fcfd7865f5bd0c64928241.html)

The table below compared performance of the above strategy vs a benchmark buy and hold strategy (80% BTC 20% ETH See report for details).
<img width="1069" alt="image" src="https://github.com/rmr327/CryptoSentimentBertRfStrat/assets/36940292/f95792df-3ba9-4503-afc9-d2528befee0b">

## Key Contributions
**Enhancing Signal Quality**: We tackled the problem within the 15DAR domain, thereby improving the signal to noise ratio, significantly improving prediction accuracy. Daily returns have too much noise.

**Advanced Sentiment Analysis**: Leveraging cutting-edge Large Language Models, we improved on traditional lexicon-based approaches. This advanced methodology enabled us to capture nuanced market sentiment more effectively. Furthermore, we have implemented this model locally, which means our data is not shared with outside organizations.

**Incorporating Memory Features**: We introduced the concept of memory features, specifically working memory, which offers a dynamic perspective on market conditions. By contrast, traditional lagged features provide only a static snapshot of historical data. 

**Original Trading Strategy**: We developed an original trading strategy based on the modified Kelly criterion, tailored to the importance of the predictive signal. This approach enabled us to dynamically adjust risk exposure based on signal significance, leading to more efficient capital allocation and improved performance.


## Conclusion
This research introduces a pioneering approach to predicting cryptocurrency trends, emphasizing the intricate interplay between human sentiment and sophisticated machine-learning techniques. The combination of Bitcoin historical data, technical and fundamental indicators, and sentiment analysis, particularly from financial news, has demonstrated a marked improvement in predicting crypto market movements. Additionally, our successful algorithmic trading strategy utilizes the alpha signal generated from the predictive model, further solidifying the efficacy of our approach.



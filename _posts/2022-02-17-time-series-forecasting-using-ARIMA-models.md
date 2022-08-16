## Time Series Forecasting Using ARIMA models

```
Ever curious about the next-day price of your stock investment? 
Or perhaps you'd like to have a proxy for your next-day sales? 
Did you know that you do not even have to look for factors beyond your historical prices or sales to get this approximate estimate? 
Computing this sounds complex but is, in fact, quite simple. 
In this post, we will investigate how to forecast time series of your choice using such sophisticated models as ARIMA - 
and I will make it as easy as possible to digest & apply!
```

### What is Time Series?

In simplest terms, time series is the sequential data you have over some time period, be it a week, a month, a year or more. Any time series can be decomposed into three components:

1. trend: general upward or downward movement
2. seasonality: any patterns observed at specific times, e.g. in summer
3. noise: random spikes and troughs unrelated to any fundamental factors

### ARIMA - Who Dreamed Up This Schema?

ARIMA stands for AutoRegressive Integrated Moving Average, and is also known as the Box-Jenkins model. Generally, this model assumes you can predict the future values using the past. 

- The autoregressive component captures the relationship between current and past values.
- The integration component assumes that data is stationary, meaning its mean does not change over time.
- The moving average component captures the relationship between a current data point and its residual moving error.

ARIMA lives within `statsmodels` Python library and has three parameters: *p* - the number of autoregressive terms, *d* - differencing order, *q* - the number of moving average terms.

![ARIMA_Function](/assets/arima.png)

[Source of the image](https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7)

### Python Implementation

Below, we will analyze the Disney (DIS) stock price over a time period of 1 year. Let's first import the libraries we will be working with throughout the post.

```python
import pandas as pd
import seaborn as sns
import matplotlib.dates as md
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```

#### Read and Visualize the Data

First, let's read in the data from a saved *.csv* file and set its index to the *Date* column. We are interested in the *Adjusted Close* price, in particular. Note that the historical data can be downloaded from Yahoo!Finance, or alternatively accessed directly using the `yfinance` package (see the [previous blogpost](https://purrwhite.github.io/2022/02/16/accessing-financial-data-with-python.html)).

```python
df = pd.read_csv('../Downloads/DIS.csv')
df = df.set_index('Date')
```
We can check if there are any *NaN* values by using `df.isna().sum()`. In our example, there are none.

Let's proceed to visualize the data trend over the past year using `seaborn`'s lineplot with a grouping by first weekday.

```python
# initialize figure with size
plt.figure(figsize=(15, 7))

# create a lineplot for adjusted closing price
ax = sns.lineplot(data=df, x=df.index, y="Adj Close")

# set a locator for weekdays
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday = 1))

# rotate tick labels by 90 deg
plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90);
```

![ARIMA_Original_Trend](/assets/arima_original_trend.png)

The data trend is quite variable. At first, a decreasing price trend is observed up until July 2022. Then, starting from mid-July 2022, DIS price is going drastically up. The upward trend is driven by recent news about change campaigns within the company.

#### Is the Data Stationary?

Next, the required features are normalized to ensure they follow the same distribution. This is done by deducting the mean of the column from each data point, followed by dividing the result by standard deviation.

```python
df['NormalizedPrice']  = (df['Close'] - df['Close'].mean())/df['Close'].std()
df['NormalizedVolume'] = (df['Volume'] - df['Volume'].mean())/df['Volume'].std()
```

To analyze the volume-price interactions more deeply, a new feature is engineered as a combination of the two.

```python
df['VolumexPrice'] = df['NormalizedVolume'] * df['NormalizedPrice']
```

On the plot generated below, we can see that the true volume-price anomalies occurred at around February 2021 (due to reasons outlined above) and May-June 2021 period, when drastic changes in price were not supported by the same change in trading volumes. This points out the fact that prices were not driven by market forces but rather by external conditions. The external conditions pertain to Tesla's comments about stopping to accept Bitcoin payments, Donald Trump's statements about bitcoin scams and Chinese crypto bans.

![Volume-Price Interaction](/assets/isoforest-volume-price.png?raw=true "Normalized BTC Volume x Price")

#### Instantiate Isolation Forest

To call the Isolation Forest instance, we use the code below, where:

- `n_estimators`: the number of estimators, i.e. decision trees, in the ensemble,
- `max_samples`: the number of sub-samples to take for tree training (`auto` ensures at least 256 sub-samples),
- `contamination`: judgement about the proportion of outliers in data,
- `random_state`: ensures result reproducibility.

We then fit the model on the column we want to compute outliers for - in our scenario, it's the engineered `VolumexPrice` feature.

```python
model=IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=np.random.RandomState(42))
model.fit(df[['VolumexPrice']])
```

#### Compute Anomaly Scores

Next, we use Isolation Forest's `predict` function to compute the anomaly scores on the given feature. By printing the dates where the score was a negative one, we can observe the timeframes during which anomaly was occurring.

 ```python
df['anomaly_scores']=model.predict(df[['VolumexPrice']])
print(df[df['anomaly_scores']==-1].index.values)
```

The outputted timeframes are February-March 2021, May-June 2021, as well as a few other time frames that were not initially visually visible by naked eye - for example, July 2021 or October 2021, when price fluctuations were not supported by underlying volume volatility, indicating the presence of other external factors.


### Caution

Although very useful, it's important to be cautious when it comes to using this model. 

- Firstly, the anomaly scores highly depend on the contamination parameter, which is a judgement made by the scientist. You should have a pretty good understanding of your data to make this assumption correctly.
- Secondly, due to the presence of randomness and the use of straight-angle branching techniques, the model can be biased.

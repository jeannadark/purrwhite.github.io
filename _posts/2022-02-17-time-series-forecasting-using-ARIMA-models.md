## Time Series Forecasting Using ARIMA models

```
Ever curious about the next-day price of your stock investment? 
Or perhaps you'd like to have a proxy for your next-day sales? 
Did you know that you do not even have to look for factors 
beyond your historical prices or sales to get this approximate estimate? 
Computing this sounds complex but is, in fact, quite simple. 
In this post, we will investigate how to forecast time series of your choice 
using such sophisticated models as ARIMA - 
and I will make it as easy as possible to digest & apply!
```

### What is Time Series?

In simplest terms, time series is the sequential data you have over some time period, be it a week, a month, a year or more. Any time series can be decomposed into three components:

1. **trend**: general upward or downward movement
2. **seasonality**: any patterns observed at specific times, e.g. summer heat or December sales
3. **noise**: random spikes and troughs unrelated to any fundamental factors

### ARIMA - Who Dreamed Up This Schema?

ARIMA stands for **AutoRegressive Integrated Moving Average** - phew, it's pretty long & heavy! It is also known as the Box-Jenkins model. Generally, this model assumes you can predict the future values using the past. 

- The **autoregressive component** captures the relationship between current and past values.
- The **integration component** assumes that data is stationary, meaning its mean does not change over time.
- The **moving average component** captures the relationship between a current data point and its residual moving error.

Stationarity is important as an assumption due to the fact that we have to forecast an expected value which shall remain the same regardless of any externalities. 

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

First, let's read in the data from a saved *.csv* file and set its index to the *Date* column. We are interested in the *Adjusted Close* price, in particular. Note that the historical data can be downloaded from Yahoo!Finance, or alternatively accessed directly using Python's `yfinance` package (see the [previous blogpost](https://purrwhite.github.io/2022/02/16/accessing-financial-data-with-python.html)).

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

Next, we have to analyze if the mean of the time series is stable over time. If it is not, modeling the time series becomes harder as changing means have to be accounted for. In other words, we have to make an assumption that the expected value is approximately the same regardless of the time period we study.

We can test stationarity either visually by plotting the 14-day rolling mean, for instance, or alternatively perform the Augmented Dickey-Fuller test. If the rolling mean is stable over time and/or the p-value of the latter test is less than 0.05, the data is stationary. The Dickey-Fuller test is a statistical hypothesis about stationarity that gets rejected if it is not found to be significant. Significance here is measured by the so-called *p-value* statistic, which is simply the probability of an event occurring.

```python
# compute the 14-day rolling mean for adjusted close price
df['Adj Close Rolling Mean (14)'] = df['Adj Close'].rolling(14).mean().fillna(0)
```

![ARIMA_Rolling_Mean](/assets/arima_rolling_mean.png)

The 14-day rolling mean changes over time, increasing in some periods and decreasing in others. Hence, the data is not stationary. Moreover, using `adfuller(df['Adj Close'])` prints out a useful summary with a p-value exceeding the 5% significance threshold. This, in turn, is indicative of a non-stationary time series.

#### Making the Data Stationary

We can make the data stationary using any of the methods below:

1. logging: helps to make the data stationary by linearizing its growth over time
2. time-shifting: helps to de-trend the data by shifting over some time period
3. exponential decaying: smoothes the data trend by delaying the spikes and troughs 
4. first- or second-order differencing: removes changes in the time series and stabilizes the mean

```python
# make data stationary by logging
df['Logged Adj Close'] = np.log(df['Adj Close']) - df['Adj Close Rolling Mean (14)']

# decay logged values over 14-day period to half and find mean
df['Exponential Decay Adj Close'] = df['Logged Adj Close'].ewm(halflife=14, min_periods=0, adjust=True).mean()

# find difference between decay and logged value
df['Exponential Decay Adj Close'] = df['Logged Adj Close'] - df['Exponential Decay Adj Close']

# shift by 1 time period
df['Logged Shifted Adj Close'] = df['Logged Adj Close'] - df['Logged Adj Close'].shift(1)

# second order differencing
df['Adj Close 2nd Order Differencing'] = df['Adj Close'].diff().diff().fillna(0)
```

As observed on the graphs below (which portray the mean of the transformed series over time), exponential decay works best in making the data stationary. This is because its mean is stabler than in other methods across all time periods. We can also judge its performance by performing the Augmented Dickey-Fuller test again, if we wish.

Log-transform                 |  Exponential Decay
:---------------------------: |:---------------------------------:
![log](/assets/arima_log.png) |  ![decay](/assets/arima_decay.png)

Time-shifting                   |  Second-order Differencing
:-----------------------------: |:-------------------------------:
![time](/assets/arima_time.png) |  ![diff](/assets/arima_diff.png)

#### How Do I Pick the Right Parameters for ARIMA?

To pick the right *p* & *q* parameters for ARIMA, autocorrelation and partial autocorrelation plots are a go-to. The autocorrelation plot shows the correlation coefficient of the time series with its own lag, and hence is useful for selecting the window for a moving average. The reason for that is that you may observe over which lag period the correlation of the time series with itself still holds.

In the graph below, we have plotted the autocorrelation plot for the exponentially decayed time series, and based on our observations the correlation is quite significant (represented by big spikes) across all lag-intervals. Hence, we can choose the first 5 most significant ones as our *q* parameter. Here a spike at lag 1 means there's a strong correlation between a value and its preceding data point, while a spike at lag 2 means there's a strong correlation between a value and the one preceding it by two steps back.

The partial autocorrelation plot is similar to the former but explains the correlation between shorter time periods that lags do not necessarily get to explain. To make it simpler, if the autocorrelation plot explains the relationship between points that are *k* distances apart, the partial autocorrelation plot explains the relationship between points that are *k* distances apart but conditional on values in-between. This can help us select the right value for our *p* parameter, which seems to be 2 (one of the biggest spikes). 

```python
# plot the autocorrelation plot of the exponentially decayed adjusted close
plot_acf(df['Exponential Decay Adj Close'].fillna(0))

# plot the partial autocorrelation plot of the exponentially decayed adjusted close
plot_pacf(df['Exponential Decay Adj Close'].fillna(0))
```

Autocorrelation                       |  Partial Autocorrelation
:---------------------------------:   |:-------------------------------------:
![ARIMA_acf](/assets/arima_log.png)   |  ![ARIMA_PACF](/assets/arima_pacf.png)

#### Train the Model

First, we can divide our data into train and test sets using an 80% / 20% split. The important thing to note here is that the test data should always come **after** train data sequentially, since we are dealing with **time**.

```python
# train is 80% of data
train = df.iloc[:int(len(df)*0.8), :]

# test is 20% of data
test = df.iloc[int(len(df)*0.8):, :]
```

Next, we call the ARIMA model with *p=2*, *q=3* and optionally *d=1* to perform first-order differencing on our exponentially decayed time series for more stationarity. This is followed by the `.fit()` method to train the model on our data.

```python
# instantiate ARIMA with given parameters
model = ARIMA(df['Exponential Decay Adj Close'].dropna(), order=(3, 1, 3))

# train the model
model_fit = model.fit()

# view the model summary
print(model_fit.summary())
```

We can visualize the model residuals to judge if we have a modelling bias. Residual plots help to determine whether the modeling error is simply a random error or a repeating pattern, in which case there's a bias. On the graph below, we can see residuals being quite close to zero and displaying no particular pattern - hence, we can say there's no bias.

```python
model_fit.resid.plot()
```
![ARIMA_residuals](/assets/arima_residuals.png)

#### Model Prediction & Performance Measure

Use the `.forecast()` method with specified start and end dates to predict future values of the test dataset, and then compare against the true values using such performance measures, as *MSE* (Mean Squared Error).

```python
# get predictions from June 2 to August 15
predictions = model_fit.predict(start='2022-06-02', end='2022-08-15')

# print out the MSE of the model
mean_squared_error(test['Exponential Decay Adj Close'].values, predictions)
```

The MSE we obtained is around 68, which is quite high but can be decreased if we consider a larger dataset, say over 5 years.

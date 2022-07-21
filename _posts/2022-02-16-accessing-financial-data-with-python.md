## Accessing Financial Data With Python

### Let the Market Come to You!

```
I only hope that we never lose sight of one thing -
That it was all started by a mouse.
```
~ *Walt Disney*

Python makes it really easy to interact with any financial data you need (so long as you have access to it!).

Whenever you find yourself in need of downloading a huge amount of data for a much huger analysis, use this tutorial to easily get around and do your business as efficiently as you can. In this article, we will be accessing the financial data of the Disney Corporation (DIS) using Python.

The libraries enabling easy connections with financial data are `yfinance`, `quandl` and `pandas_datareader`. Note that we are attempting to access only structured data within this article, without reverting to the capture of unstructured data like news and sentiments. 

### Connect to Yahoo!Finance

Yahoo!Finance is entirely free and houses lots of data. Its only disadvantage is in rate-limiting and the lack of more detailed data. You can use either of the two methods below to access DIS stock data from Yahoo. 

Method 1 uses `pandas_datareader`:

```python
import pandas_datareader as dr
data = dr.get_data_yahoo('DIS')
```
The `data` variable now contains a dataframe with Disney's opening, high, low and closing stock prices. 

![DIS data from Yahoo](/assets/yahoo_disney.png)

Method 2 uses `yfinance` to access historical 1 year data for Disney:

```python
import yfinance as yf
data = yf.Ticker('DIS').history(period='1y')
```
The start and end dates can be further specified with `start_date` and `end_date` parameters.

### Connect to Quandl

Quandl offers more details than Yahoo!Finance but at the expense of limited free datasets. You can use either of the two methods below to access Quandl data for DIS stock.

Method 1 uses `quandl` package to retrieve the Wiki dataset for DIS.

```python
import quandl
data = quandl.get('WIKI/DIS')
```
![DIS data from Quandl](/assets/quandl_disney.png)

Method 2 uses `pandas_datareader` to perform the same operation, and once again, `start_date` and `end_date` parameters can be optionally specified.

```python
import pandas_datareader as dr
data = dr.get_data_quandl('DIS', api_key='XXXXX')
```

#### Perform High-Level Analysis

Using the code below, we can visualize the 20-day rolling average for DIS closing price. 

```python
rolling_avg = data.rolling(window=20).mean()
plt.plot(rolling_avg.index, rolling_avg)
plt.show()
```

![20-day DIS Rolling Average](/assets/rolling_average_disney.png)

We can also compute returns on the stock and plot them as a histogram.

```python
data['return'] = (data['Open'] - data['Close']) / data['Close']
data['return'].hist()
plt.title('DIS returns distribution')
plt.show()
```
![DIS Returns Distribution](/assets/distribution_disney.png)

## Anomaly Detection in Financial Data Using Isolation Forest

### There's an Anomaly in Your Data, but It's Healthy!

```
An anomalous bitcoin 
Makes you wish that you could join
The city of clever fools
Where a smarter person rules.
```

Why detect anomalies in general?

Take the cryptocurrency world as an example: any anomalous change in volume and price, for instance, can be an indication of an upcoming price swing that you can utilize to your advantage. Perhaps, market participants are executing a pump-and-dump manipulation, so wouldn't it be great to detect such schemes in real-time so you can trade accordingly?

In this article, we will analyze the Bitcoin price (BTC), volume and their combinations with the help of Isolation Forest to detect any anomalous behaviors.

The library in use is sklearn's `ensemble.IsolationForest`, the inner workings of which are described in the next section.

### Iso-What?

An Isolation Forest is an unsupervised machine learning model that represents an ensemble of decision trees (iTrees). The idea behind it is that those data points that are splitted earlier in the tree are more likely to be anomalies than those which travel further down the tree.

In particular, here is how the forest works:

1. A random sub-sample of data is selected for a binary tree.
2. Within that random sub-sample, branching starts taking place based on a random feature and threshold. If a data point falls below the threshold value, then it's assigned to the left sub-tree. Otherwise, it's assigned to the right sub-tree.
3. This branching workflow continues until either maximum tree depth is reached or each data point is completely isolated. This process is repeated for each tree in the ensemble.
4. Depending on the depth required to reach a data point, anomaly scores are assigned. The more negative the score, the more anomalous it is.

![Isolation Forest Structure](/assets/isoforest.png)

[Source of the image](https://www.researchgate.net/figure/Overview-of-the-isolation-forest-method-Light-green-circles-represent-common-normal_fig3_341629782)

### Python Implementation

Below, we will analyze the cryptocurrency market anomalies over the past 12 months in Python. The dataset is obtained from Yahoo!Finance and spans a year's worth of closing price data and trading volume for BTC.

#### Read and Visualize the Data

Using the code below, we visualize BTC closing price and trading volume over time. Visually, we can see that there was an unusual volume spike in February 2021, which was the record-setting month for BTC driven by Tesla's acceptance of cryptocurrencies. However, a price correction took place at around the same time due to a lot of political talk around the legality of bitcoin payments.

```python
df[['Close', 'Volume']].plot(by=df.index, subplots=True, figsize=(10,10))
```

![Data Distribution](/assets/isoforest-visualize-data.png)

A few other outliers in trading volume re-appeared over the year, as evidenced by the boxplot below. Anomalous fluctuations in price are less visible due to general unpredictability of bitcoin price directions and their susceptibility to external forces.

```python
ax = sns.boxplot(data=df[['Close', 'Volume_Adjusted']], orient="h", palette="Set2")
ax.set_xlabel('Closing Price and Volume Distributions')
```

![Boxplots](/assets/isoforest-boxplot-data.png?raw=true "Boxplots")


#### Perform Some Feature Engineering

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

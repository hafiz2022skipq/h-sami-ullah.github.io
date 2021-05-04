## Deep learning and statistical based daily stock price forecasting and Monitoring

**Project description:** Designing a Machine Learning algorithm to predict stock prices is a subject of interest for economists and machine learning practitioners. Financial modelling is a challenging task, not only from an analytical perspective but also from a psychological perspective. After 2008 financial crisis, many financial companies and investors shifted their interest towards predicting future trends. Most of the existing methods for stock price forecasting are modelled using non-linear methods and evaluated on specific data sets. These models are not able to generalize for diverse datasets. Financial time series data is highly dynamic in nature and makes it difficult to analyze through statistical methods. Recurrent Neural Networks (RNN) based Long Short- Term Memory (LSTM) networks were able to capture the patterns of the sequences data meanwhile statistical methods tried to generalize by memorizing data instead of recognizing patterns. In this work, we examined the performance of LSTM model and statistical models over stock prices of different companies to generalize the model. The experimental results of this study show that, LSTM network outperformed traditional statistical methods like ARIMA, MA and AR models. Furthermore, we have noticed that, LSTM network was able to perform consistently on different data sets while statistical methods showed varied performance. Through this project, we addressed the gaps in current models of stock price prediction in both economic and machine learning perspective.

### 1. Autoregression model 

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

```javascript
from statsmodels.tsa.ar_model import AR
model = AR(train_ar)# train autoregression
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
```

### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

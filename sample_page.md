## Deep learning and statistical based daily stock price forecasting and Monitoring

**Project description:** Designing a Machine Learning algorithm to predict stock prices is a subject of interest for economists and machine learning practitioners. Financial modelling is a challenging task, not only from an analytical perspective but also from a psychological perspective. After 2008 financial crisis, many financial companies and investors shifted their interest towards predicting future trends. Most of the existing methods for stock price forecasting are modelled using non-linear methods and evaluated on specific data sets. These models are not able to generalize for diverse datasets. Financial time series data is highly dynamic in nature and makes it difficult to analyze through statistical methods. Recurrent Neural Networks (RNN) based Long Short- Term Memory (LSTM) networks were able to capture the patterns of the sequences data meanwhile statistical methods tried to generalize by memorizing data instead of recognizing patterns. In this work, we examined the performance of LSTM model and statistical models over stock prices of different companies to generalize the model. The experimental results of this study show that, LSTM network outperformed traditional statistical methods like ARIMA, MA and AR models. Furthermore, we have noticed that, LSTM network was able to perform consistently on different data sets while statistical methods showed varied performance. Through this project, we addressed the gaps in current models of stock price prediction in both economic and machine learning perspective.



### 2. Moving Average
This statistical approach gives the general direction of trends and patterns by exploring historical data. For a window size of K, moving average is calculated by adding most recent data points and divide them by K and iterated until it converges. The MA is commonly used to smoothen the prices by eliminating white noise from the given data set.

### 3. Autoregression (AR) Model 
Auto regression is a method for interpreting time series data which takes input from the past time steps as source and forecasts the future time step values. It is one of the basic time series models which can deliver precise results and also proved to be one the best model to evaluate time series data. An AR is an autoregressive process where it uses previous p values to predict future values. These models are built upon the assumption that, past values don’t change with respect to time. Most of the times it can lead to wrong predictions if the input data is strictly temporal. Yet these models are still used for predicting non-temporal data. For instance, before 20008 financial crises, most shareholders used AR models to forecast US stock prices before investing. Surprisingly, most of them were successful to capture the trends and made profits using this model. 

```javascript
from statsmodels.tsa.ar_model import AR
model = AR(train_ar)# train autoregression
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
```
### 4. ARIMA Model
Auto regressive integrated moving average method, often referred as ARIMA model. A non-seasonal data without white noise can be modeled using this model. This model is distinguished by 3 variables i.e., p, d, q. Where p represents the order of autoregressive term, q is the indicator about the order of moving average term and d is the quantity used when time series is required to be converted into stationary form.

```javascript
from statsmodels.tsa.arima_model import ARIMA
history = [x for x in train_arima]
y = test_arima
# make first prediction
predictions = list()
model = ARIMA(history, order=(1,1,0))
model_fit = model.fit(disp=0)
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0])
```
### 5. LSTM Model
LSTM network is implemented using deep learning framework called Keras, which is built on top of TensorFlow. While processing the stock prices, LSTM networks unwrap the data along the time axis. Cell states from the previous time steps will be saved and used for calculating the current cell state.

```javascript

model = Sequential()

# Adding the first LSTM layer 
# Here return_sequences=True means whether to return the last output in the output sequence, or the full sequence.
# it basically tells us that there is another(or more) LSTM layer ahead in the network.
model.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Dropout regularisation for tackling overfitting
model.add(Dropout(0.2))

model.add(LSTM(units = 80))
model.add(Dropout(0.25))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
# RMSprop is a recommended optimizer as per keras documentation
# check out https://keras.io/optimizers/ for more details
model.compile(optimizer = 'adamax', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 10, batch_size = 150)
```

### 6. Pipeline
Implementing this project can be understood using the below flow chart. 

<img src="images/pipeline.png?raw=true"/>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# 1. Fetch BND stock data from July 1, 2015 to July 31, 2025
df = yf.download('BND', start='2015-07-01', end='2025-07-31', auto_adjust=True)
df = df[['Close']]  # adjusted close prices come under 'Close' when auto_adjust=True
df.dropna(inplace=True)

# 2. Chronological train-test split (train: until 2023-12-31, test: from 2024-01-01)
train = df.loc[:'2023-12-31']
test = df.loc['2024-01-01':]

# ------------------------
# ARIMA MODEL
# ------------------------

arima_order = (5,1,0)  # example order
model_arima = ARIMA(train, order=arima_order)
model_arima_fit = model_arima.fit()

arima_forecast = model_arima_fit.forecast(steps=len(test))
y_test_arima = test['Close']
y_pred_arima = arima_forecast

mae_arima = mean_absolute_error(y_test_arima, y_pred_arima)
rmse_arima = np.sqrt(mean_squared_error(y_test_arima, y_pred_arima))
mape_arima = np.mean(np.abs((y_test_arima - y_pred_arima) / y_test_arima)) * 100

print("ARIMA Model Performance:")
print(f"MAE: {mae_arima:.4f}")
print(f"RMSE: {rmse_arima:.4f}")
print(f"MAPE: {mape_arima:.2f}%")

plt.figure(figsize=(12,6))
plt.plot(y_test_arima.index, y_test_arima, label='Actual')
plt.plot(y_test_arima.index, y_pred_arima, label='ARIMA Forecast')
plt.title('ARIMA Model Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()

# ------------------------
# LSTM MODEL
# ------------------------

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values)

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

split_date = '2024-01-01'
split_idx = np.where(df.index[seq_length:] >= split_date)[0][0]

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length,1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

mae_lstm = mean_absolute_error(actual_prices, predicted_prices)
rmse_lstm = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mape_lstm = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

print("LSTM Model Performance:")
print(f"MAE: {mae_lstm:.4f}")
print(f"RMSE: {rmse_lstm:.4f}")
print(f"MAPE: {mape_lstm:.2f}%")

plt.figure(figsize=(12,6))
plt.plot(actual_prices, label='Actual')
plt.plot(predicted_prices, label='LSTM Forecast')
plt.title('LSTM Model Forecast vs Actuals')
plt.xlabel('Time Steps')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()

# ------------------------
# Task 3: Forecast Future Market Trends using ARIMA on full data
# ------------------------

df_full = yf.download('BND', start='2015-07-01', end='2025-07-31', auto_adjust=True)
df_full = df_full[['Close']].dropna()

arima_order = (5,1,0)
model_full = ARIMA(df_full, order=arima_order)
model_fit_full = model_full.fit()

forecast_steps = 252
forecast_result = model_fit_full.get_forecast(steps=forecast_steps)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

last_date = df_full.index[-1]
forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)

plt.figure(figsize=(14,7))
plt.plot(df_full.index, df_full['Close'], label='Historical Adj Close')
plt.plot(forecast_dates, forecast_mean, label='12-Month Forecast', color='orange')
plt.fill_between(forecast_dates, conf_int.iloc[:,0], conf_int.iloc[:,1], color='orange', alpha=0.3, label='Confidence Interval')
plt.title('BND Adjusted Close Price Forecast - 12 Months')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()

print("Forecast Summary:")
print(f"- Forecast period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
print(f"- Forecast mean price range: {forecast_mean.min():.2f} to {forecast_mean.max():.2f}")
print(f"- Confidence interval width tends to widen over time, indicating increasing uncertainty.")

trend = "upward" if forecast_mean[-1] > forecast_mean[0] else "downward or stable"
print(f"- Overall trend over forecast horizon appears to be {trend}.")

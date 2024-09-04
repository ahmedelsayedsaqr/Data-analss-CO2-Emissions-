import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Example time series data
np.random.seed(0)
date_range = pd.date_range(start='2008-01-01', periods=240, freq='M')
data = 10 + 0.8 * np.sin(2 * np.pi * date_range.month /
                         12) + np.random.normal(0, 0.5, 240)
time_series = pd.Series(data, index=date_range)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series.values.reshape(-1, 1))

# Prepare the dataset for LSTM


def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 10
X, y = create_dataset(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32, verbose=0)

# Forecast and evaluate
X_test = scaled_data[-(time_step + 12):-12].reshape(1, time_step, 1)
lstm_forecast = []
for _ in range(12):
    pred = model.predict(X_test)
    lstm_forecast.append(pred[0, 0])
    X_test = np.append(X_test[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

lstm_forecast = scaler.inverse_transform(
    np.array(lstm_forecast).reshape(-1, 1)).flatten()
lstm_rmse = np.sqrt(mean_squared_error(time_series[-12:], lstm_forecast))
lstm_mae = mean_absolute_error(time_series[-12:], lstm_forecast)
lstm_r2 = r2_score(time_series[-12:], lstm_forecast)

print(f"LSTM RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}, R²: {lstm_r2:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_series[-24:], label='True Values')
plt.plot(time_series.index[-12:], lstm_forecast,
         label='LSTM Forecast', linestyle='--', color='red')
plt.title('LSTM Forecast vs True Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

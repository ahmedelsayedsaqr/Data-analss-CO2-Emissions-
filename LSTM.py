import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example time series data (replace with your actual dataset)
np.random.seed(0)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
data = 10 + 0.8 * np.sin(2 * np.pi * date_range.month /
                        12) + np.random.normal(0, 0.5, 100)
time_series = pd.Series(data, index=date_range)

# Normalize the data to a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series.values.reshape(-1, 1))

# Prepare the dataset for LSTM (e.g., use the last 10 time steps to predict the next value)


def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 10
X, y = create_dataset(scaled_data, time_step)
# Reshape for LSTM input [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=1, verbose=1)

# Predict future values
predicted = model.predict(X)
# Inverse scaling to original values
predicted = scaler.inverse_transform(predicted)

# Plotting the original and predicted time series
plt.figure(figsize=(10, 6))
plt.plot(time_series.index[time_step:], time_series.values[time_step:],
        label='Original Time Series', color='blue')
plt.plot(time_series.index[time_step:], predicted,
        label='LSTM Predicted', color='red', linestyle='--')
plt.title('LSTM Model Predicted Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

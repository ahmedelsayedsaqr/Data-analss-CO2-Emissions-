import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Example data setup
np.random.seed(0)
data = np.sin(np.linspace(0, 100, 1000))
time_step = 10

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

X, y = [], []
for i in range(len(data) - time_step - 1):
    X.append(data[i:(i + time_step), 0])
    y.append(data[i + time_step, 0])
X = np.array(X)
y = np.array(y)

# Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], time_step, 1)
X_test = X_test.reshape(X_test.shape[0], time_step, 1)

# Define LSTM model


def lstm_model(params):
    lstm_units = int(params[2])
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Example parameters
best_params = [0.001, 32, 60]  # Example: learning rate, batch size, lstm units

# Train the model with best parameters
model = lstm_model(best_params)
model.fit(X_train, y_train, epochs=50,
          batch_size=int(best_params[1]), verbose=0)

# Predict and plot
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test_unscaled, label='True Values')
plt.plot(predictions, label='LSTM Predictions', linestyle='--')
plt.title('LSTM Predictions vs True Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

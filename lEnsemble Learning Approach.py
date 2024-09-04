import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Example data setup (you should replace this with your actual data)
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

# Reshape input for LSTM to be [samples, time steps, features]
X_train_lstm = X_train.reshape(X_train.shape[0], time_step, 1)
X_test_lstm = X_test.reshape(X_test.shape[0], time_step, 1)

# LSTM model


def lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


lstm = lstm_model()
lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
lstm_predictions = lstm.predict(X_test_lstm)

# ARIMA model
arima_model = ARIMA(y_train, order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_predictions = arima_fit.forecast(steps=len(y_test))

# SARIMA model
sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
sarima_predictions = sarima_fit.forecast(steps=len(y_test))

# Combine predictions using Random Forest Regressor
combined_predictions = np.column_stack(
    (lstm_predictions.flatten(), arima_predictions, sarima_predictions))
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(combined_predictions, y_test)
ensemble_predictions = rf.predict(combined_predictions)

# Performance metrics
rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
mae = mean_absolute_error(y_test, ensemble_predictions)
r2 = rf.score(combined_predictions, y_test)

print(f"Ensemble Model Performance: RMSE={
      rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

# Plotting the results

plt.figure(figsize=(10, 6))
plt.plot(y_test, label="True Values")
plt.plot(ensemble_predictions, label="Ensemble Predictions", linestyle='--')
plt.title("Ensemble Model Predictions vs True Values")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

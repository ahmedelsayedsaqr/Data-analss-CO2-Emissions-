import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Example CO2 emissions data
np.random.seed(0)
date_range = pd.date_range(start='2020-01-01', periods=240, freq='M')
data = 50 + 0.5 * np.sin(2 * np.pi * date_range.month /
                         12) + np.random.normal(0, 2, 240)
time_series = pd.Series(data, index=date_range)

# Fit ARIMA model
arima_model = ARIMA(time_series, order=(2, 1, 2))
arima_result = arima_model.fit()

# Forecast and evaluate ARIMA
arima_forecast = arima_result.forecast(steps=12)
arima_rmse = np.sqrt(mean_squared_error(time_series[-12:], arima_forecast))
arima_mae = mean_absolute_error(time_series[-12:], arima_forecast)
arima_r2 = r2_score(time_series[-12:], arima_forecast)

print(f"ARIMA RMSE: {arima_rmse:.2f}, MAE: {
      arima_mae:.2f}, R²: {arima_r2:.2f}")

# Fit SARIMA model
sarima_model = SARIMAX(time_series, order=(
    2, 1, 2), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast and evaluate SARIMA
sarima_forecast = sarima_result.forecast(steps=12)
sarima_rmse = np.sqrt(mean_squared_error(time_series[-12:], sarima_forecast))
sarima_mae = mean_absolute_error(time_series[-12:], sarima_forecast)
sarima_r2 = r2_score(time_series[-12:], sarima_forecast)

print(f"SARIMA RMSE: {sarima_rmse:.2f}, MAE: {
      sarima_mae:.2f}, R²: {sarima_r2:.2f}")

# Plotting the results with specified colors
plt.figure(figsize=(14, 7))
plt.plot(time_series, label='Original Series', color='blue', linewidth=2)
plt.plot(time_series.index[-12:], arima_forecast,
         label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
plt.plot(time_series.index[-12:], sarima_forecast,
         label='SARIMA Forecast', color='green', linestyle='-.', linewidth=2)
plt.legend()
plt.title('ARIMA vs SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('CO2 Emissions')
plt.grid(True)
plt.show()
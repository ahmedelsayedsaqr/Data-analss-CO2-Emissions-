import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Example time series data
np.random.seed(0)
date_range = pd.date_range(start='2008-01-01', periods=240, freq='M')
data = 10 + 0.8 * np.sin(2 * np.pi * date_range.month /
                         12) + np.random.normal(0, 0.5, 240)
time_series = pd.Series(data, index=date_range)

# Train/test split
train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

# Optimal ARIMA parameters
optimal_arima_order = (1, 1, 1)

# Optimal SARIMA parameters
optimal_sarima_order = (1, 0, 1, 12)

# Refit ARIMA model with optimal parameters
final_arima_model = ARIMA(train, order=optimal_arima_order)
final_arima_model_fit = final_arima_model.fit()
arima_predictions = final_arima_model_fit.forecast(steps=len(test))

# Refit SARIMA model with optimal parameters
final_sarima_model = SARIMAX(train, order=(
    1, 1, 1), seasonal_order=optimal_sarima_order)
final_sarima_model_fit = final_sarima_model.fit(disp=False)
sarima_predictions = final_sarima_model_fit.forecast(steps=len(test))

# Calculate final RMSE, MAE, and R² for ARIMA
final_arima_rmse = np.sqrt(mean_squared_error(test, arima_predictions))
final_arima_mae = mean_absolute_error(test, arima_predictions)
final_arima_r2 = r2_score(test, arima_predictions)

# Calculate final RMSE, MAE, and R² for SARIMA
final_sarima_rmse = np.sqrt(mean_squared_error(test, sarima_predictions))
final_sarima_mae = mean_absolute_error(test, sarima_predictions)
final_sarima_r2 = r2_score(test, sarima_predictions)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test.index, test.values, label='True Values', color='blue')
plt.plot(test.index, arima_predictions,
         label='ARIMA Forecast', linestyle='--', color='red')
plt.plot(test.index, sarima_predictions,
         label='SARIMA Forecast', linestyle='--', color='green')
plt.title('ARIMA and SARIMA Forecast vs True Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Return results for display
final_arima_rmse, final_arima_mae, final_arima_r2, final_sarima_rmse, final_sarima_mae, final_sarima_r2

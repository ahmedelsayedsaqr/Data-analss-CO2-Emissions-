import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Example time series data with future dates
np.random.seed(0)
# Start date set to 2025
date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
data = 10 + np.sin(2 * np.pi * date_range.month / 12) + \
    np.random.normal(0, 0.5, 100)
time_series = pd.Series(data, index=date_range)

# Plotting the original time series
plt.figure(figsize=(10, 4))
plt.plot(time_series, label='Original Time Series')
plt.title('Original Time Series (with Future Dates)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot ACF and PACF to identify p and q for ARIMA
plot_acf(time_series)
plot_pacf(time_series)
plt.show()

# Fit an ARIMA model (p, d, q)
arima_model = ARIMA(time_series, order=(2, 1, 2))
arima_result = arima_model.fit()

# Summary of the ARIMA model
print(arima_result.summary())

# Plotting the fitted values and forecast
plt.figure(figsize=(10, 4))
plt.plot(time_series, label='Original Time Series', color='blue')
plt.plot(arima_result.fittedvalues, label='ARIMA Fitted Values',
        linestyle='--', color='red')
plt.title('ARIMA Model Fitted Values (with Future Dates)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

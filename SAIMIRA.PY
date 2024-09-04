import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Example time series data with seasonal component and future dates
np.random.seed(0)
# Start date set to 2025
date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
data = 10 + 0.8 * np.sin(2 * np.pi * date_range.month /
                        12) + np.random.normal(0, 0.5, 100)
time_series = pd.Series(data, index=date_range)

# Fit a SARIMA model (p, d, q) x (P, D, Q, s)
sarima_model = SARIMAX(time_series, order=(
    2, 1, 2), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Summary of the SARIMA model
print(sarima_result.summary())

# Plotting the fitted values and forecast
plt.figure(figsize=(10, 4))
plt.plot(time_series, label='Original Time Series', color='blue')
plt.plot(sarima_result.fittedvalues, label='SARIMA Fitted Values',
        linestyle='--', color='red')
plt.title('SARIMA Model Fitted Values (with Future Dates)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

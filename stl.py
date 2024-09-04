import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Generate example time series data with future dates
np.random.seed(0)
# Replace '2025-01-01' with a future start date
date_range = pd.date_range(start='2025-01-01', periods=365, freq='D')
data = 10 + np.sin(2 * np.pi * date_range.dayofyear /
                   365) + np.random.normal(0, 0.5, 365)
time_series = pd.Series(data, index=date_range)

# Perform Seasonal-Trend decomposition using LOESS (STL)
# Specify the seasonal period (e.g., 365 for yearly seasonality)
seasonal_period = 365
stl = STL(time_series, seasonal=seasonal_period)
result = stl.fit()

# Enhanced Plotting of STL Decomposition Components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Plot original time series
ax1.plot(result.observed, label='Original', color='blue')
ax1.set_title('Original Time Series')
ax1.legend(loc='upper left')

# Plot trend component
ax2.plot(result.trend, label='Trend', color='orange')
ax2.set_title('Trend Component')
ax2.legend(loc='upper left')

# Plot seasonal component
ax3.plot(result.seasonal, label='Seasonal', color='green')
ax3.set_title('Seasonal Component')
ax3.legend(loc='upper left')

# Plot residual component
ax4.plot(result.resid, label='Residual', color='red')
ax4.set_title('Residual Component')
ax4.legend(loc='upper left')

# Formatting the plot
plt.tight_layout()
plt.show()

# Save components for further analysis or modeling
trend_component = result.trend
seasonal_component = result.seasonal
residual_component = result.resid

# Optional: Print summary statistics for each component
print("Trend Component Summary:")
print(trend_component.describe())
print("\nSeasonal Component Summary:")
print(seasonal_component.describe())
print("\nResidual Component Summary:")
print(residual_component.describe())

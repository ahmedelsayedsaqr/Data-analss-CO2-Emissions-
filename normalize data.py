import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Example CO2 emissions data with future dates
data = {
    # Future dates starting from 2025-01-01
    'Date': pd.date_range(start='202-01-01', periods=10, freq='M'),
    'CO2_Emissions': [450, 470, 420, 480, 500, 460, 490, 510, 530, 550]
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Display the original data
print("Original CO2 Emissions Data:")
print(df)

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalize the CO2 emissions values to a range of 0-1
df['Normalized_CO2_Emissions'] = scaler.fit_transform(df[['CO2_Emissions']])

# Display the normalized data
print("\nNormalized CO2 Emissions Data (0-1 range):")
print(df)

# Plotting the original and normalized CO2 emissions
plt.figure(figsize=(12, 6))

# Plot original CO2 emissions
plt.subplot(1, 2, 1)
plt.plot(df.index, df['CO2_Emissions'], marker='o',
        linestyle='-', color='blue', label='Original CO2 Emissions')
plt.title('Original CO2 Emissions')
plt.xlabel('Date')
plt.ylabel('CO2 Emissions')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')
plt.gca().xaxis.set_major_formatter(
    mdates.DateFormatter('%b %Y'))  # Format date as Month Year
plt.gca().xaxis.set_major_locator(
    mdates.MonthLocator())  # Set major ticks to months
plt.xticks(rotation=45)  # Rotate dates for better readability

# Plot normalized CO2 emissions
plt.subplot(1, 2, 2)
plt.plot(df.index, df['Normalized_CO2_Emissions'], marker='o',
        linestyle='-', color='green', label='Normalized CO2 Emissions (0-1)')
plt.title('Normalized CO2 Emissions')
plt.xlabel('Date')
plt.ylabel('Normalized CO2 Emissions')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')
plt.gca().xaxis.set_major_formatter(
    mdates.DateFormatter('%b %Y'))  # Format date as Month Year
plt.gca().xaxis.set_major_locator(
    mdates.MonthLocator())  # Set major ticks to months
plt.xticks(rotation=45)  # Rotate dates for better readability

# Improve layout
plt.tight_layout()
plt.show()

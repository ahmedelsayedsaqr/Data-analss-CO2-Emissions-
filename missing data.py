import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example time series data with missing values
data = {
    'Date': pd.date_range(start='2025-01-01', periods=10, freq='M'),
    'CO2_Emissions': [450, np.nan, 420, 480, np.nan, 460, 490, np.nan, 530, 550]
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Display the original data with missing values
print("Original CO2 Emissions Data with Missing Values:")
print(df)

# 1. Filling missing data using linear interpolation
df['Interpolated_CO2_Emissions'] = df['CO2_Emissions'].interpolate(
    method='linear')

# 2. Filling missing data using mean replacement
df['Mean_Replaced_CO2_Emissions'] = df['CO2_Emissions'].fillna(
    df['CO2_Emissions'].mean())

# Display the data after handling missing values
print("\nData After Handling Missing Values (Linear Interpolation and Mean Replacement):")
print(df)

# Plotting the original and filled data
plt.figure(figsize=(12, 6))

# Plot original CO2 emissions with missing values
plt.plot(df.index, df['CO2_Emissions'], marker='o', linestyle='-',
        color='blue', label='Original CO2 Emissions (with Missing Values)')

# Plot interpolated CO2 emissions
plt.plot(df.index, df['Interpolated_CO2_Emissions'], marker='o',
        linestyle='--', color='green', label='Interpolated CO2 Emissions')

# Plot mean-replaced CO2 emissions
plt.plot(df.index, df['Mean_Replaced_CO2_Emissions'], marker='o',
    linestyle='--', color='red', label='Mean Replaced CO2 Emissions')

# Formatting the plot
plt.title('Handling Missing Data in CO2 Emissions Time Series')
plt.xlabel('Date')
plt.ylabel('CO2 Emissions')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')
plt.xticks(rotation=45)

# Improve layout
plt.tight_layout()
plt.show()

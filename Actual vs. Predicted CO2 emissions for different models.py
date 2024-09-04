import numpy as np
import matplotlib.pyplot as plt

# Sample data: Actual vs. Predicted CO2 emissions for different models
years = np.arange(2018, 2028)

# Actual CO2 emissions (in gigatonnes) - hypothetical data for illustration
actual_emissions = np.array(
    [35.5, 36.0, 36.4, 36.9, 37.2, 37.5, 37.9, 38.3, 38.6, 39.0])

# Predicted emissions by different models
arima_pred = np.array([35.6, 36.1, 36.5, 37.0, 37.4,
                      37.7, 38.0, 38.2, 38.5, 38.8])
sarima_pred = np.array([35.5, 36.0, 36.3, 36.8, 37.1,
                       37.4, 37.8, 38.1, 38.4, 38.7])
lstm_pred = np.array([35.4, 36.0, 36.4, 36.9, 37.3,
                     37.6, 37.9, 38.2, 38.6, 39.0])
hybrid_lstm_ga_pred = np.array(
    [35.4, 36.0, 36.5, 37.0, 37.3, 37.7, 38.0, 38.3, 38.7, 39.1])

# Check lengths
print("Length of years:", len(years))
print("Length of actual_emissions:", len(actual_emissions))
print("Length of arima_pred:", len(arima_pred))
print("Length of sarima_pred:", len(sarima_pred))
print("Length of lstm_pred:", len(lstm_pred))
print("Length of hybrid_lstm_ga_pred:", len(hybrid_lstm_ga_pred))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(years, actual_emissions, label="Actual Emissions",
         marker='o', linestyle='-', color='black')
plt.plot(years, arima_pred, label="ARIMA Predictions",
         marker='x', linestyle='--', color='blue')
plt.plot(years, sarima_pred, label="SARIMA Predictions",
         marker='s', linestyle='--', color='green')
plt.plot(years, lstm_pred, label="LSTM Predictions",
         marker='d', linestyle='--', color='orange')
plt.plot(years, hybrid_lstm_ga_pred, label="Hybrid LSTM-GA Predictions",
         marker='^', linestyle='-', color='red')

# Adding titles and labels
plt.title("CO2 Emissions Forecasting: Actual vs. Predicted")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions (Gigatonnes)")
plt.legend()
plt.grid(True)

# Show plot
plt.show()

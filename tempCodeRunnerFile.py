import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from geneticalgorithm import geneticalgorithm as ga
import yfinance as yf

# Fetch data using yfinance
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
data = data['Close'].resample('MS').mean()  # Resample to monthly data

# Split the data into train and test sets
train, test = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]

# Define the objective function for GA


def objective_function(params):
    try:
        p, d, q, P, D, Q, s = int(params[0]), int(params[1]), int(params[2]), int(
            params[3]), int(params[4]), int(params[5]), int(params[6])
        print(f"Testing parameters: p={p}, d={
              d}, q={q}, P={P}, D={D}, Q={Q}, s={s}")

        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(
            start=len(train), end=len(train) + len(test) - 1, dynamic=False)

        rmse = np.sqrt(mean_squared_error(test, predictions))
        print(f"RMSE: {rmse}")
        return rmse
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.inf  # Return a large number if the function fails


# Define the parameter bounds for GA
varbound = np.array([[0, 2], [0, 2], [0, 2], [0, 1], [0, 1], [0, 1], [12, 12]])

# Run the GA optimization
algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 10,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None,
    'funtimeout': 60  # Increased timeout to 60 seconds
}

model = ga(function=objective_function, dimension=7, variable_type='int',
           variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model.run()

# Extract the optimized parameters
optimized_params = model.output_dict['variable']
p, d, q, P, D, Q, s = int(optimized_params[0]), int(optimized_params[1]), int(optimized_params[2]), int(
    optimized_params[3]), int(optimized_params[4]), int(optimized_params[5]), int(optimized_params[6])

# Fit the optimized SARIMA model
optimized_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
optimized_model_fit = optimized_model.fit(disp=False)

# Make predictions
predictions = optimized_model_fit.predict(
    start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test, predictions))
mae = mean_absolute_error(test, predictions)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()

print(f'Optimized Parameters: p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, s={s}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

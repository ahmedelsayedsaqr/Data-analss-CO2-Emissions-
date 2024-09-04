import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from geneticalgorithm import geneticalgorithm as ga

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

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], time_step, 1)
X_test = X_test.reshape(X_test.shape[0], time_step, 1)

# Define the LSTM model


def lstm_model(params):
    learning_rate, lstm_units, dropout_rate = params
    model = Sequential()
    model.add(LSTM(units=int(lstm_units), input_shape=(time_step, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Objective function for GA


def ga_optimize(params):
    model = lstm_model(params)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse


# GA bounds for parameters
varbound = np.array([[0.0001, 0.01],  # learning_rate
                     [10, 100],       # lstm_units
                     [0.1, 0.5]])     # dropout_rate

# GA model
algorithm_param = {'max_num_iteration': 30,
                   'population_size': 10,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': None}

model = ga(function=ga_optimize,
           dimension=3,
           variable_type='real',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

model.run()

# Get the best parameters and model performance
best_params = model.output_dict['variable']
optimized_model = lstm_model(best_params)
optimized_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
predictions = optimized_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Best parameters: {best_params}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Plotting the predictions vs true values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values')
plt.plot(predictions, label='LSTM-GA Predictions', linestyle='--')
plt.title('LSTM-GA Predictions vs True Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

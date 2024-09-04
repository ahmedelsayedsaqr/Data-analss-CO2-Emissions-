import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Example time series data
np.random.seed(0)
date_range = pd.date_range(start='2025-01-01', periods=100, freq='MS')
data = 10 + 0.8 * np.sin(2 * np.pi * date_range.month /
                         12) + np.random.normal(0, 0.5, 100)
time_series = pd.Series(data, index=date_range)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series.values.reshape(-1, 1))

# Prepare the dataset for LSTM


def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 10
X, y = create_dataset(scaled_data, time_step)
# Reshape for LSTM input [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# LSTM Model Architecture


def build_lstm_model(params):
    model = Sequential()
    model.add(
        LSTM(units=params['units'], return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=params['units']))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(
        learning_rate=params['learning_rate']), loss='mean_squared_error')
    return model

# GA Fitness Function


def evaluate_lstm(individual):
    params = {
        'units': int(individual[0]),
        'learning_rate': individual[1]
    }
    model = build_lstm_model(params)
    history = model.fit(X, y, epochs=5, batch_size=1, verbose=0)
    loss = model.evaluate(X, y, verbose=0)
    return loss,


# DEAP GA Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 10, 100)  # units in LSTM
toolbox.register("attr_float", np.random.uniform,
                 0.0001, 0.01)  # learning rate
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int, toolbox.attr_float), n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_lstm)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[
                 10, 0.0001], up=[100, 0.01], eta=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=10)
ngen = 5
cxpb = 0.5
mutpb = 0.2

# Function to plot fitness progress


def plot_fitness(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Override the evaluate function to return history for plotting


def evaluate_lstm(individual):
    params = {
        'units': int(individual[0]),
        'learning_rate': individual[1]
    }
    model = build_lstm_model(params)
    history = model.fit(X, y, epochs=5, batch_size=1,
                        verbose=0, return_dict=True)
    plot_fitness(history.history)
    loss = model.evaluate(X, y, verbose=0)
    return loss,


# Run the Genetic Algorithm
results = algorithms.eaSimple(
    population, toolbox, cxpb, mutpb, ngen, verbose=True)

# Plot final results
best_individual = tools.selBest(population, k=1)[0]
print(f"Best LSTM Units: {int(best_individual[0])}, Best Learning Rate: {
      best_individual[1]}")

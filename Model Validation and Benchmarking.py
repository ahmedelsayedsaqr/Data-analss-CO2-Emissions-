import matplotlib.pyplot as plt
import seaborn as sns

# Sample data for existing models
existing_model_data = [
    {
        "time": [1, 2, 3, 4, 5],
        "emissions": [10, 12, 15, 18, 20]
    },
    {
        "time": [1, 2, 3, 4, 5],
        "emissions": [8, 10, 12, 14, 16]
    }
]

# Sample data for the proposed model
proposed_model_data = {
    "time": [1, 2, 3, 4, 5],
    "emissions": [9, 11, 13, 15, 17]
}

# Performance metrics (replace with actual values)
existing_model_performance = [0.95, 0.92, 0.90, 0.98]
proposed_model_performance = [0.97, 0.94, 0.92, 0.99]

# Set up the figure and subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Plot historical vs. predicted emissions for existing models
for i, model_data in enumerate(existing_model_data):
    model_title = f"Existing Model {i+1}"
    sns.lineplot(x='time', y='emissions', data=model_data, ax=axs[0, i])
    axs[0, i].set_title(model_title)

# Plot historical vs. predicted emissions for the proposed model
sns.lineplot(x='time', y='emissions', data=proposed_model_data, ax=axs[1, 0])
axs[1, 0].set_title("Proposed Model")

# Create a bar chart for performance metrics
metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
sns.barplot(x=metrics, y=existing_model_performance,
            ax=axs[1, 1], label="Existing Models")
sns.barplot(x=metrics, y=proposed_model_performance,
            ax=axs[1, 1], label="Proposed Model")
axs[1, 1].set_title("Performance Comparison")
axs[1, 1].legend()

# Add labels, titles, and legends
plt.tight_layout()
plt.show()

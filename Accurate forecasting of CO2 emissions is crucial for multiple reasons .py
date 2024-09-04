import matplotlib.pyplot as plt  # Importing pyplot from matplotlib for plotting
import numpy as np  # Importing numpy for numerical operations and handling arrays

# Creating example data
years = np.arange(2000, 2031)  # Creating an array of years from 2000 to 2030
actual_emissions = np.array([100, 105, 108, 112, 115, 120, 123, 130, 135, 140, 142, 145, 150, 155, 160, 162,
                            # Example actual emissions data
                             165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235])
forecasted_emissions = np.array([101, 107, 110, 113, 118, 122, 125, 132, 137, 142, 144, 148, 152, 157, 162, 165,
                                # Example forecasted emissions data
                                 168, 172, 177, 182, 188, 193, 198, 203, 208, 213, 218, 223, 228, 233, 238])

# Creating the plot
plt.figure(figsize=(12, 6))  # Setting the size of the figure (plot)
# Plotting actual emissions data with point markers
plt.plot(years, actual_emissions, label='Actual Data', marker='o')
# Plotting forecasted emissions data with dashed line and point markers
plt.plot(years, forecasted_emissions,
        label='Forecasts', linestyle='--', marker='x')
plt.xlabel('Year')  # Label for the x-axis
plt.ylabel('CO2 Emissions')  # Label for the y-axis
plt.title('Actual vs Forecasted CO2 Emissions')  # Title of the plot
plt.legend()  # Displaying the legend to explain what each line represents
plt.grid(True)  # Adding a grid to the plot for better readability
plt.show()  # Displaying the plot

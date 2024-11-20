import numpy as np
import matplotlib.pyplot as plt

# Fixed parameter for boundary runs
b = 2

# Range of r values to test
r_values = [5,10, 20, 30, 40, 50]  # Different r values for the 5 plots

# Loop over each r value, generate the plot, and save it as a file
for r in r_values:
    # Range of u values (number of length-1 runs) to vary for the fixed r
    u_values = np.linspace(b + 1, r, 100)  # u varies from b+1 up to r

    # Calculate weights for both functions with varying u
    weight_fazeli_varying_u = (1 / r) * (1 - (u_values - b) / r**2)
    weight_theorem_1_varying_u = (1 / r) * (1 + (2 * u_values - b - 2) / ((r + 1) * (r + 2)))**-1

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(u_values, weight_fazeli_varying_u, label="Fazeli et al.'s weight", color='blue')
    plt.plot(u_values, weight_theorem_1_varying_u, label="Theorem 1's weight", color='red')
    plt.xlabel('u (Number of Length-1 Runs)')
    plt.ylabel('Weight')
    plt.title(f'Comparison of Weight Functions as Number of Length-1 Runs Varies (r = {r})')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(f'weight_comparison_r_{r}.png')
    plt.close()  # Close the figure to avoid displaying it in the loop

r_values, "Plots saved successfully."

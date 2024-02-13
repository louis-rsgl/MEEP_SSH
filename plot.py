import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('flux0.dat', sep=',', header=None)
df2 = pd.read_csv('flux.dat', sep=',', header=None)

x1 = df1[1] 
y1 = df1[2]  

x2 = df2[1]  
y2 = df2[2] 

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

marker_size = 10

# First subplot with data from the first .dat file
ax1.scatter(x1, y1, color='b', s=marker_size)  # Scatter plot with blue markers
ax1.plot(x1, y1, 'b')  # Plot with blue dots
ax1.set_xlabel('Frequency')  # Set x-axis label for the first subplot
ax1.set_ylabel('Transmission')  # Set y-axis label for the first subplot
ax1.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid for the first subplot
ax1.set_title('Flux 0')  # Title for the first subplot

# Second subplot with data from the second .dat file
ax2.scatter(x2, y2, color='r', s=marker_size)  # Scatter plot with red markers
ax2.plot(x2, y2, 'r')  # Plot with red dots
ax2.set_xlabel('Frequency')  # Set x-axis label for the second subplot
ax2.set_ylabel('Transmission')  # Set y-axis label for the second subplot
ax2.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid for the second subplot
ax2.set_title('Flux')  # Title for the second subplot

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.savefig("Fluxes.png")
plt.close()

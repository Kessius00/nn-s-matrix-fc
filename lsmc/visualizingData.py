

import matplotlib.pyplot as plt
import numpy as np




def errorPlot(errors):
    errors = np.array(errors)

    # Create a plot
    plt.figure()
    plt.plot(errors[:,0], marker='o', linestyle='-', color='b', label='MAE')
    plt.plot(errors[:,1], color='r', label='RMSE')

    # Add labels and title
    plt.title("Error decrease")
    plt.xlabel("Iteration")
    plt.ylabel("Error")

    # Show legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
    
    
def visualizeData(dense_matrix):
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    plt.imshow(dense_matrix, cmap='viridis', aspect='auto', interpolation='none')
    plt.colorbar(label='Intensity')
    plt.title('dense matrix visualization of part of data')
    plt.xlabel('movies')
    plt.ylabel('users')

    # Show the plot
    plt.show()

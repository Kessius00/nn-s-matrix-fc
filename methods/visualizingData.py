import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import random
import scipy.stats as stats
from scipy.stats import norm
from scipy.sparse import *
import os


def errorPlot(errors):
    errors = np.array(errors)

    # Create a plot
    plt.figure()
    plt.plot(errors[:,0], marker='o', linestyle='-', color='b', label='MAE')
    # plt.plot(errors[:,1], color='r', label='RMSE')

    # Add labels and title
    plt.title("Error decrease")
    plt.xlabel("Iteration")
    plt.ylabel("Error")

    # Show legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
    
def newVisualization(new_matrix):
    nans_matrix=np.where(new_matrix==0, np.nan, new_matrix)
    print('nu al klaar met nans')
    visualizeData(nans_matrix)
    
    
def visualizeData(dense_matrix):
    plt.figure(figsize=(10, 6))
    plt.imshow(dense_matrix, cmap='viridis', aspect='auto', interpolation='none')
    plt.colorbar(label='Intensity')
    plt.title('dense matrix visualization of part of data')
    plt.xlabel('movies')
    plt.ylabel('users')

    # Show the plot
    plt.show()


def visualizeMassBin(dense_matrix, massbin,shape):
    picturematrix = dense_matrix[:, massbin].reshape(shape)
    # Plot each matrix with its respective colormap

    fig, ax = plt.subplots()
    cax = ax.imshow(picturematrix, aspect='equal', cmap='cubehelix')  # Set aspect to 'equal' and use same limits as original data
    ax.set_title(f'Massbin {massbin} Colormap')
    fig.colorbar(cax, ax=ax)
    
    # Add overall title and show the plot
    fig.suptitle(f'Data of Massbin {massbin}', fontsize=16)
    plt.show() 




def find_massbins(dense_matrix):
    a = 0
    massbins = []
    for mz_num in range(dense_matrix.shape[1]):
        massbin = dense_matrix[:,mz_num]
        massbin_E_density = np.sum(massbin)
        if massbin_E_density > a:
            a = massbin_E_density
            massbins.append(mz_num)
            
    return massbins
 

    
def plotHistData(data):

    # Plotting a basic histogram
    plt.hist(data, bins='auto', color='skyblue', edgecolor='black')
    plt.title(f'with {data.shape} bins')
    # Adding labels and title
    plt.xlabel('m/z values')
    plt.ylabel('Frequency')

    # Display the plot
    plt.show()



def plotHistogram(data, bins='auto'):
    plt.figure(figsize=(15, 10))
    
    # Plotting the histogram of the input data
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black', density=True, label='Data Histogram')
    
    # Generate and plot the selected distribution
    x_values = np.linspace(min(data), max(data), 1000)

    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate the normal PDF
    normal_pdf = norm.pdf(x_values, mean, std)
    plt.plot(x_values, normal_pdf, label=f'Normal Distribution (mean={mean}, std={std})', color='red')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Probability')
    plt.legend()

    # Display the plot
    plt.show()



def massbinImageComparisonGPT(massbin_of_interest, data, clipped_data, clipped_sampled_data, colormap='cubehelix'):
    # Set the base colormap
    base_cmap = plt.get_cmap(colormap)

    # Create a new colormap with red for zero values
    new_cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))  # Create a new colormap
    new_cmap.set_under('red')  # Set color for values below the minimum (0 in this case)
    
    # Reshape the data to 500x100 for visualization
    data_matrix = data[:, massbin_of_interest].reshape((500, 100))
    clipped_data_matrix = clipped_data[:, massbin_of_interest].reshape((500, 100))
    clipped_sampled_data_matrix = clipped_sampled_data[:, massbin_of_interest].reshape((500, 100))
    
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    # Calculate the global min/max across all matrices for consistent color scaling
    min_val = min(np.min(data_matrix), np.min(clipped_data_matrix), np.min(clipped_sampled_data_matrix))
    max_val = max(np.max(data_matrix), np.max(clipped_data_matrix), np.max(clipped_sampled_data_matrix))
    
    # Set values less than or equal to zero to a specific number for coloring
    data_matrix[data_matrix <= 0] = np.nan  # Use NaN for zeros (or a small value if you want to keep them)
    clipped_data_matrix[clipped_data_matrix <= 0] = np.nan
    clipped_sampled_data_matrix[clipped_sampled_data_matrix <= 0] = np.nan

    # Plot each matrix with the same color scale (vmin and vmax)
    for ax, matrix, cmap, title in zip(axes, 
                                       [data_matrix, clipped_data_matrix, clipped_sampled_data_matrix], 
                                       [new_cmap] * 3, 
                                       ['Original Data', 'Clipped Data', 'Clipped & Sampled Data']):
        # Use global min and max for consistent brightness/color scaling
        cax = ax.imshow(matrix, aspect='equal', cmap=cmap, vmin=min_val, vmax=max_val, interpolation=None)
        
        # Set zero values to be red
        ax.imshow(np.ma.masked_where(matrix != 0, matrix), cmap='Reds', alpha=0.5, vmin=min_val, vmax=max_val)
        
        ax.set_title(f'{title}', fontsize=14)
        fig.colorbar(cax, ax=ax)
    
    # Add an overall title and display the plot
    fig.suptitle(f'Side by Side Comparison of Original, Clipped, and C+Sampled Data for Massbin {massbin_of_interest}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the suptitle
    plt.show()
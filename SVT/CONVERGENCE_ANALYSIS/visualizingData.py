

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def sample_array_with_mask(arr, mask, row_sample_size=10000, col_sample_size=None):
    # Check if row_sample_size exceeds the number of rows
    if row_sample_size >= arr.shape[0]:
        row_indices = np.arange(arr.shape[0])  # Take all rows
    else:
        row_indices = np.random.choice(arr.shape[0], size=row_sample_size, replace=False)
    
    # Check if column sampling is specified
    if col_sample_size is not None:
        if col_sample_size >= arr.shape[1]:
            col_indices = np.arange(arr.shape[1])  # Take all columns
        else:
            col_indices = np.random.choice(arr.shape[1], size=col_sample_size, replace=False)
        
        sampled_arr = arr[row_indices, :][:, col_indices]
        sampled_mask = mask[row_indices, :][:, col_indices]
    else:
        # If no column sampling, take all columns
        sampled_arr = arr[row_indices, :]
        sampled_mask = mask[row_indices, :]
    
    return sampled_arr, sampled_mask


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
    
def relErrorPlot(errors):
    errors = np.array(errors)
    # Create a plot
    plt.figure()
    
    plt.plot(errors[:,0], marker='o', linestyle='-', color='orange', label='in-sampling')
    plt.plot(errors[:,1], color='r', linestyle=':', label='out-of-sampling')
    plt.plot(errors[:,2], color='b', label='general')

    # Add labels and title
    plt.title("Error decrease")
    plt.xlabel("Iteration")
    plt.ylabel("Error")

    # Show legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


def errorPlotDataFrame(lambda_, rho, gamma, df_results):
    specific_errors = df_results[
    (df_results['lambda'] == lambda_) &
    (df_results['rho'] == rho) &
    (df_results['gamma'] == gamma)]
    # Convert 'errors' and 'relative_errors' columns to NumPy arrays
    errors_array = specific_errors['errors'].to_numpy()[0]

    # Create a plot
    plt.figure(figsize=(8, 5))
    
    plt.plot(np.array(errors_array)[:,0], marker='o', linestyle='-', color='b', label='MAE')
    plt.plot(np.array(errors_array)[:,1], color='r', label='RMSE')

    # Add labels and title6
    plt.title(f'Error Convergence (lambda = {lambda_}, rho = {rho}, gamma = {gamma})')

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


def visualizeMassBin(dense_matrix, massbin,shape):
    picturematrix = dense_matrix[:, massbin].reshape(shape)
    # Plot each matrix with its respective colormap
    min_val = np.min(picturematrix)
    max_val = np.max(picturematrix)
    fig, ax = plt.subplots()
    cax = ax.imshow(picturematrix, aspect='equal', cmap='cubehelix', vmin=min_val, vmax=max_val)  # Set aspect to 'equal' and use same limits as original data
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

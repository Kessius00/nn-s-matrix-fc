

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

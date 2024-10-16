import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re
import time
import random
from scipy.stats import poisson, norm
import pandas as pd
from scipy.sparse import csr_matrix



df = pd.read_csv('.\\data\\ratings.csv')


# het liefst heb je hieronder ipv df.tail(..) gewoon df (de hele dataset), maar dan krijg ik problemen.. 
matrix = pd.pivot_table(df.tail(3000000), index='userId', columns='movieId', values='rating', aggfunc='mean')
print(matrix)

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


    
m = matrix.to_numpy()
visualizeData(m)
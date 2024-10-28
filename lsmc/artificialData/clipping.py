import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import random
import scipy.stats as stats
from scipy.stats import poisson, norm




def keep_small_percentage(uniform, small_perc=.1):
    clipping = small_perc / uniform
    return clipping        

def zeroCalculator(data):
    # Testing how many values are non-zero. Quite a lot. But not all. 
    non_zero_indices = len(np.argwhere(data != 0))   
    print(f'So, {100*non_zero_indices/np.prod(data.shape)}% are non-zero indices\n This means that {round(np.prod(data.shape)*(1-non_zero_indices/np.prod(data.shape)))} indices of {np.prod(data.shape)} are zero')
        


def probabilityMatrix(n_rows, n_cols, percentage):
    one_matrix = np.ones((n_rows, n_cols), dtype=int)
    num_cells = n_rows * n_cols
    num_zeros = int(percentage * num_cells)
    
    # Flatten the matrix to choose random indices
    flat_indices = np.random.choice(num_cells, num_zeros, replace=False)
    
    # Set these chosen indices to 0
    one_matrix.flat[flat_indices] = 0
    
    return one_matrix

def randomDiscard(M, ratio_keep):
    P = probabilityMatrix(M.shape[0], M.shape[1], ratio_keep)
    return np.multiply(M, P), P

def select_top_percentage(M, ratio_keep):
    # Flatten the matrix into a 1D array for easier processing
    M_flat = M.flatten()
    
    n_values_to_keep = int(np.ceil(len(M_flat) * ratio_keep))
    
    # Sort the flattened array to find the cutoff value for the top percentage
    sorted_values = np.sort(M_flat)[::-1]  # Sort in descending order
    cutoff_value = sorted_values[n_values_to_keep - 1]
    
    # Create a copy of the original matrix
    M_top = np.where(M >= cutoff_value, M, 0)
    
    return M_top




def partlyDiscardedClipping(data_matrix, uniform_ratio_keep, threshold_ratio_keep):
    # Clip the imperfect matrix
    clipped_matrix = select_top_percentage(data_matrix, threshold_ratio_keep)
    mask_clip = np.where(clipped_matrix!=0, 1, 0)
    
    # Randomly discard values to create an imperfect matrix
    imperfect_matrix, mask_gaussian = randomDiscard(clipped_matrix, 1 - uniform_ratio_keep)
    
    # naive multiplication
    sampled_mask = mask_clip*mask_gaussian
    return imperfect_matrix, sampled_mask




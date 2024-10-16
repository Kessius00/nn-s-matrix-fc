import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd, matrix_rank
import math
import time




# in sampling out of sampling and general (overall info)

# Helper functions
def relative_approx_error(original, approximation):
    """
    Calculate the relative approximation error between the original and approximation matrices.
    """
    error = norm(original - approximation, 'fro')
    fro_M = norm(original, 'fro')
    rel_error = 100 * (error / fro_M) ** 2
    return rel_error


def projection_operator(M_sampled, X):
    """
    Compute the projection operator for matrix completion.
    """
    mask = M_sampled != 0
    projection = np.zeros_like(X)
    projection[mask] = X[mask] - M_sampled[mask]
    return projection

def suggested_stop(X_k, original_sampled, tolerance):
    """
    Check the stopping condition based on the relative difference between X_k and the original sampled matrix.
    """
    rel_error = norm(projection_operator(original_sampled, X_k), 'fro') / norm(original_sampled, 'fro')
    return rel_error <= tolerance

def k_0_finder(tau, step_size, sampled_entries):
    """
    Find the initial value of k_0.
    """
    return math.ceil(tau / (step_size * norm(sampled_entries, 2)))


def plot_metrics(em, r):
    """
    Plot error history and rank history over iterations.
    """
    fig, ax1 = plt.subplots()

    # Plot error history
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Frobenius norm of error', color=color)
    ax1.plot(range(len(em)), em, color=color, label='Frobenius norm of error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Rank of X', color=color)
    ax2.plot(range(len(r)), r, color=color, label='Rank of X')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Adjust layout to avoid overlap
    plt.title('Error and Rank Convergence Over Iterations')
    ax1.legend(loc='upper left')  # Add legend for error plots
    ax2.legend(loc='upper right')  # Add legend for rank plot
    plt.show()

def svt_algorithm(sampled_entries, step_size, tolerance, tau, max_iter=100):
    """
    Singular Value Thresholding (SVT) algorithm for matrix completion.
    """
    n_rows, n_columns = sampled_entries.shape
    # Initialize k and a Y_0
    k = 1
    Y_0 = k_0_finder(tau, step_size, sampled_entries) * step_size * sampled_entries
    
    # First singular value calculation
    u, s, v = svd(Y_0, full_matrices=False)
    print('The biggest singular value on iteration 0:' , s[0])
    
    
    # Get list ready
    iterate = True
    
    times = []
    em_sampled_arr = []
    rank_arr = []
    rel_error_s_arr = []

    while iterate:
        start_time = time.time()
        if k == 1:
            Y = Y_0

        # Singular Value Decomposition
        U, sigma, Vt = svd(Y, full_matrices=False)
        # Thresholding
        
        sigma_thresh = np.maximum(sigma - tau, 0)
        # Check for rank stopping condition
        rank = np.sum(sigma_thresh > 0)
        # Construct the approximation matrix X
        X = U[:, :rank] @ np.diag(sigma_thresh[:rank]) @ Vt[:rank, :]

        # Calculate errors
        error_sampled_matrix = -projection_operator(M_sampled=sampled_entries, X=X)
        rel_error_s = (norm(projection_operator(sampled_entries, X=X), 'fro') / norm(sampled_entries, 'fro')) 

        # make threshold smaller when relative difference is smaller than 1 percent
        if rel_error_s<1e-1:
            tau = 3 * np.sqrt(max(n_rows, n_columns))
            step_size = 1.2
        elif rel_error_s<1e-2:
            tau *= 0.9
            step_size *=.9
        
        # Update Y
        Y = Y + step_size * error_sampled_matrix
        
        # Append results for plotting
        em_sampled_arr.append(norm(error_sampled_matrix, 'fro'))
        rank_arr.append(rank)
        rel_error_s_arr.append(rel_error_s)

        # Check stopping conditions
        if suggested_stop(X, sampled_entries, tolerance):
            iterate = False
        if k >= max_iter:
            iterate = False
        k += 1        
        print(f'Iter {k}; Relative error:', rel_error_s)
        print('rank:',rank)
        iter_time = time.time() - start_time
        times.append(iter_time)
        print(f'iteration time: {iter_time} s')
        print('\n')
        
    print(f'average time of one iteration: {np.sum(times)/(k-1)}')

    return X, em_sampled_arr, rank_arr, rel_error_s_arr



def presentReconstruction(data, data_sampled, reconstruction, relative_sample_error):
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(r'Matrix Completion using SVT, 50% data kept', fontsize=16)
    gs = fig.add_gridspec(2, 3)
    original_img = fig.add_subplot(gs[0, 0])
    sampled_img = fig.add_subplot(gs[0, 1])
    reconstructed_img = fig.add_subplot(gs[0, 2])
    error_plot = fig.add_subplot(gs[1, :])


    original_img.imshow(data, aspect='auto', interpolation='none')
    original_img.set_title('Original Data ')
    original_img.axis('off')
    original_img.set_ylabel('pixel')
    original_img.set_xlabel('m/z')

    sampled_img.imshow(data_sampled, aspect='auto', interpolation='none')
    sampled_img.set_title('Incomplete Data')
    sampled_img.axis('off')

    reconstructed_img.imshow(reconstruction, aspect='auto', interpolation='none')
    reconstructed_img.set_title('Reconstructed Data')
    reconstructed_img.axis('off')


    error_plot.plot(relative_sample_error, label='SVT')
    error_plot.set_xlabel('Iteration')
    error_plot.set_ylabel(r'Relative Error $\frac{||Estimate - Original||_F}{||Original||_F}$')
    error_plot.legend()

    plt.tight_layout()
    plt.show()

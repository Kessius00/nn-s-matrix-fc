import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import random
import scipy.stats as stats

# handige functies voor gaandeweg
def shrinkage_operator(x, phi):
    # Apply the shrinkage (soft-thresholding) operator element-wise
    return np.where(x > phi, x - phi, np.where(x < -phi, x + phi, 0))


def soft_threshold(x, threshold):
    """Soft-thresholding operator"""
    return np.where(x > threshold, x - threshold,
                    np.where(x < -threshold, x + threshold, 0))
    
    
def orthogonal_projection(x, t):
    # In de paper weergegeven als P_K 
    norm_x = np.linalg.norm(x)  # Frobenius norm for the vector x
    if norm_x <= t:
        return np.array([x, t])
    elif -norm_x <= t <= norm_x:
        scale = (norm_x + t) / (2 * norm_x)
        return np.array([scale * x, scale * norm_x])
    else:
        return np.array([0, 0])

def projection_onto_soc(x, t):
    """Project onto the second-order cone K"""
    x_norm = np.linalg.norm(x)

    if x_norm <= t:
        return x,t  # Case 1: x_F ≤ t
    elif -x_norm <= t <= x_norm:
        # Case 2: -x_F ≤ t ≤ x_F
        projected_x = 1/2 * (1 + t / x_norm) * x  # Scale the vector x
        projected_t = (x_norm + t) / 2  # Update t
        return projected_x, projected_t  # Return as separate entities
    else:
        return np.zeros_like(x), 0  # Case 3: t ≤ -x_F



def error(P, R):
    re_er = 0.5 * np.linalg.norm((P-R), 'fro')**2
    data_norm_squared = np.linalg.norm(R, 'fro')**2
    objective_percentage = (re_er / data_norm_squared) * 100
    return objective_percentage

def validationErrors(P, R, num_tested_ratings):
    MAE = np.sum(np.abs(R-P))/num_tested_ratings
    RMSE = np.sqrt(np.sum(np.abs(R-P)**2)/num_tested_ratings)
    return MAE, RMSE


def matrix_rank(M, tol=1e-6):
    """Rank check function"""
    _, Sigma, _ = np.linalg.svd(M, full_matrices=False)
    return np.sum(Sigma > tol)


def generate_matrix_with_rank(m, n, r):
    assert r <= min(m, n), "Rank r must be less than or equal to the smaller of m or n."
    
    # Generate two random matrices of size m x r and r x n
    A = np.random.rand(m, r)
    B = np.random.rand(r, n)
    
    # Multiply A and B to get a matrix of size m x n with rank r
    matrix = np.dot(A, B)
    
    return matrix



def proximal_operator_P(y, R, u, rho):
    """Proximal operator for P"""
    S = y / rho + R - u
    Q, Sigma, W = np.linalg.svd(S, full_matrices=False)
    Sigma_prox = soft_threshold(Sigma, 1/rho)
    rank = np.sum(Sigma_prox > 0)
    P_next = Q[:, :rank] * Sigma_prox[:rank] @ W[:rank, :]
    return P_next, rank


def update_y_s(y, s, P, R, delta, epsilon):
    """Update Lagrange multipliers y and s using projection onto the second-order cone K."""
    # P must be the P_{k+1} and not the P_k
    difference =(R - P)
    y_new, s_new = projection_onto_soc(y + delta * difference, s - delta*epsilon)
    return y_new, s_new



def proximal_operator_Z(P, u, lambda_, rho):
    """Proximal operator for auxiliary variable Z"""
    # P is the new P_k+1
    # u is the u_k 
    Z_next = soft_threshold(P + u, lambda_ / rho)
    return Z_next


def LMSC_optimize(rho, lambda_, R, P_init, u_init, y_init, s_init, delta, epsilon, num_iterations):
    """Main optimization loop."""
    P = P_init
    u = u_init
    y = y_init
    s = s_init
    Z = np.zeros_like(P) 
    errors = []
    

    for k in range(num_iterations):

        P, rank_P = proximal_operator_P(y, R, u, rho)

        y, s = update_y_s(y, s, P, R, delta, epsilon)

        Z = proximal_operator_Z(P, u, lambda_, rho)

        u = u + (P - Z)
        
        if round((k)/10) == (k)/10:
            print(f"Iteration {k+1}: Rank of P = {rank_P}")

        MAE, RMSE = validationErrors(P, R, R.shape[0]*R.shape[1])
        errors.append([MAE, RMSE])

    return P, Z, u, y, s, errors

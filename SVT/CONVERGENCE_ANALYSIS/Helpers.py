import numpy as np

def euclideanDivergence(x,y):
    return .5*(x-y)**2

def KLDivergence(x,y):
    # the Kullback-Leibler divergence function
    return x * np.log(x/y)-x+y

def ISDivergence(x,y):
    # the Itakura-Saito divergence function
    return x/y - np.log(x/y)-1

def svd_threshold(A, threshold):
    """
    Apply the thresholding operator to the singular values of X
    """
    U, s, V = np.linalg.svd(A, full_matrices=False)
    s = np.maximum(s - threshold, 0)
    return np.dot(U, np.dot(np.diag(s), V))

def sample_from_matrix(A, ratio_to_keep, seed=42):
    rng = np.random.default_rng(seed)
    num_samples = int(ratio_to_keep * A.size)
    coords = rng.choice(np.arange(A.size), size=num_samples, replace=False)
    coords_TF = np.zeros(A.shape, dtype=bool)
    coords_TF.flat[coords] = True
    X = np.zeros(A.shape)
    X.flat[coords] = A.flat[coords]
        
    
    return X, coords, coords_TF



def sample_from_matrix_ratio(A, ratio_to_keep, seed=42):
    rng = np.random.default_rng(seed)
    num_samples = int(ratio_to_keep * A.size)
    coords = rng.choice(np.arange(A.size), size=num_samples, replace=False)
    coords_TF = np.zeros(A.shape, dtype=bool)
    coords_TF.flat[coords] = True
    X = np.zeros(A.shape)
    X_nans = np.zeros(A.shape)
    X_nans[:] = np.nan
    X_nans.flat[coords] = A.flat[coords]
    X.flat[coords] = A.flat[coords]
    
        
    
    return X, coords, coords_TF, X_nans


def normalize_rows(X):
    row_max = X.max(axis=1)
    X = X / row_max[:, np.newaxis]
    return X

def normalize_columns(X):
    col_max = X.max(axis=0)
    X = X / col_max[np.newaxis, :]
    return X

def generate_simulated_IMS_matrix(m=50000, n=500, rank=100, seed=42, random_mean=1.0, random_scale=1, batch_contrast=2, batch_abondance=200, remove_batches=True):
    """
    m: number of rows
    n: number of columns
    rank: rank of the matrix, constructed from 2 matrices W and H of size m x rank and rank x n
    seed: seed for the random number generator, for ease of reproducibility
    random_mean: mean of the gamma distribution for the random values
    random_scale: scale of the gamma distribution for the random values
    batch_contrast: contrast of the batch values, during generation, 'batches' are added to the W and H matrices, to simulate the original data
    batch_abondance: number of batches to add to the W and H matrices
    remove: if True, remove some values in the batches, to simulate the original data
    
    Run the function to generate a simulated IMS matrix, with the given parameters
    The function makes first two matrices W and H of size m x rank and rank x n, and then computes the product W*H, which is the simulated IMS matrix
    Run the function with no input to get the default values
    """
    
    rng=np.random.default_rng(seed)
    W_rows, W_cols = m, rank
    H_rows, H_cols = rank, n
    
    W = rng.gamma(shape=random_mean, scale=random_scale, size=(W_rows, W_cols))
    H = rng.gamma(shape=random_mean, scale=random_scale/100, size=(H_rows, H_cols))
    
    def add_batches(A,num_batches, batch_size, high_value, remove=True):
        for _ in range(num_batches):
            row_start = np.random.randint(0, A.shape[0] - batch_size[0])
            col_start = np.random.randint(0, A.shape[1] - batch_size[1])
            A[row_start:row_start + batch_size[0], col_start:col_start + batch_size[1]] += high_value+np.random.normal(0,high_value/2,batch_size)
        # Set some values to zero in the batches
        if remove:
            for _ in range(num_batches):
                row_start = np.random.randint(0, A.shape[0] - batch_size[0])
                col_start = np.random.randint(0, A.shape[1] - batch_size[1])
                A[row_start:row_start + batch_size[0], col_start:col_start + batch_size[1]] = 1e-2
        A = np.clip(A, 1e-2, None)
        return A
    
    W = add_batches(W, num_batches=batch_abondance, batch_size=(np.max([int(m/5),1]), 2), high_value=batch_contrast,remove=remove_batches)
    H = add_batches(H, num_batches=batch_abondance*5, batch_size=(2,np.max([int(n/25),1])), high_value=batch_contrast/40, remove=remove_batches)
    simulated_X = np.dot(W,H)
    simulated_X = normalize_columns(simulated_X)
    return simulated_X

def insampling_error_relative(X_sampled, X_reconstructed, Coords_TF, norm='fro'):
    """
    Compute the relative insampling error between the original matrix X_sampled and the reconstructed matrix X_reconstructed
    X_sampled: the sampled matrix with zeros where we have not sampled
    X_reconstructed: reconstructed/estimated matrix we want to compare to X_sampled
    Coords_TF: boolean matrix of the same size as X_sampled, with True values where we have sampled the matrix X_sampled. If you use the function sample_from_matrix, you can use the third output of the function
    """
    
    X_reconstructed_sampled = np.where(Coords_TF, X_reconstructed, 0)
    error = np.linalg.norm(X_sampled - X_reconstructed_sampled, ord=norm)
    relative_error = error / np.linalg.norm(X_sampled, ord=norm)
    return relative_error
    
def out_of_sample_error_relative(X_original, X_reconstructed, Coords_TF, norm='fro'):
    """
    Compute the relative out-of-sample error between the original matrix X_original and the reconstructed matrix X_reconstructed divided by the complement of the sampled matrix
    X_original: the original matrix
    X_reconstructed: reconstructed/estimated matrix we want to compare to X_original
    Coords_TF: boolean matrix of the same size as X_original, with True values where we have sampled the matrix X_original. If you use the function sample_from_matrix, you can use the third output of the function
    """
    X_original_complement = np.where(~Coords_TF, X_original, 0)
    X_reconstructed_complement = np.where(~Coords_TF, X_reconstructed, 0)
    error = np.linalg.norm(X_original_complement - X_reconstructed_complement, ord=norm)
    relative_error = error / np.linalg.norm(X_original_complement, ord=norm)
    return relative_error

def general_error_relative(X_original, X_reconstructed, norm='fro'):
    """
    Compute the relative error between the original matrix X_original and the reconstructed matrix X_reconstructed
    X_original: the original matrix
    X_reconstructed: reconstructed/estimated matrix we want to compare to X_original
    """
    error = np.linalg.norm(X_original - X_reconstructed, ord=norm)
    relative_error = error / np.linalg.norm(X_original, ord=norm)
    return relative_error

''' Parameter testing template
# Write the name of the algorithm you are testing (To make it easy to find the error file)
algorithm_name = 'Algo1' # <----------------- CHANGE THIS

lower_power = 1 # 10^lower_power, lower bound for the test parameter <----------------- CHANGE THIS
upper_power = 3 # 10^upper_power, upper bound for the test parameter <----------------- CHANGE THIS
manual_added = np.array([]) # if you want to add some manual points <----------------- ADD YOUR MANUAL POINTS HERE

n = upper_power - lower_power + 1 # number of test points
params_to_test = np.hstack((np.logspace(lower_power, upper_power, n), manual_added)) # the parameters to test, to loop over
print(f'Testing parameter values: {params_to_test}')

# Load the errors file 
try:
    df = pd.read_csv(f'Out/errors_{algorithm_name}.csv',sep=',', decimal='.')
except FileNotFoundError:
    df = pd.DataFrame(columns=['Parameter_value', 'Error'])
    
for i in range(len(params_to_test)):
    
    W, H, error = YOUR_FUNCTION(params_to_test[i]) # test with the parameter value <----------------- CHANGE THIS
    df = df.append({'Parameter_value': params_to_test[i], 'Error': error}, ignore_index=True) # 
    df.to_csv(f'Out/errors_{algorithm_name}.csv', index=False, deccimal='.')
'''

#This module defines statistical operations for data analysis

import numpy as np
import scipy.special as sci_esp

def SVD(X):
    """
    This method Whitens data by doing Singular Value Decomposition.

    Parameters:
    -----------
    X: np.ndarray
        2d array with (objects, properties)

    Returns:
    --------
    Xw: np.ndarray
        Data from SVD eigenvectors
    P: np.ndarray
        Covariance of properties
    """
    R = np.dot(X.T,X)
    U,S,V = np.linalg.svd(R)
    P = np.dot(np.diag(1./np.sqrt(S)),U.T)
    Xw = np.dot(X,P.T)

    return Xw,P

def PCA(X):
    """
    This method applies a Principal Component Analysis on the data.

    Parameters:
    -----------
    X: np.ndarray
        2d array with (objects, properties)

    Returns:
    --------
    X_pca: np.ndarray
        Data with PCA components
    """
    # Data matrix X, assumes 0-centered
    X = X - X.mean(axis=0)
    n, m = X.shape
    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    return X_pca

def vec_cov(x, y, normed = False):
    """
    This method computes the covariance between vectors x and y.

    Parameters:
    -----------
    x, y: np.ndarray, np.ndarray
        2d arrays (n,m), with n realizations of a m size vector
    normed: bool
        If True, the normalized covariance is obtained

    Returns:
    --------
    cov_ij: np.ndarray
        Covariance
    """
    cov_ij = np.mean(np.dot((x - np.mean(x)),(y - np.mean(y)).T))
    if normed:
        cov_ii = np.mean(np.dot((x - np.mean(x)),(x - np.mean(x)).T))
        cov_jj = np.mean(np.dot((y - np.mean(y)),(y - np.mean(y)).T))
        return cov_ij/np.sqrt(cov_ii*cov_jj)
    else:
        return cov_ij

def vec_cov_mat(x, y, normed = False):
    """
    This method computes the covariance matrix of vector variables x and y.

    Parameters:
    -----------
    x, y: np.ndarray, np.ndarray
        2d arrays (l,n,m), with l realizations of n variables of m size vector
    normed: bool
        If True, the normalized covariance is obtained

    Returns:
    --------
    cov: np.ndarray
        Covariance matrix
    """
    n_var = x.shape[1]
    cov = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(n_var):
            cov[i,j] = vec_cov(x[:,i], y[:,j], normed = normed)
    return cov

def mat_mean(mat):
    """
    This method computes the mean value of a diagonally
    symmetric matrix, excluding the diagonal
    matrix and one of the symmetric halfs.

    Parameters:
    -----------
    mat: np.ndarray
        Matrix with diagonaly symmetry.

    Returns:
    --------
    mean: float
        Mean value of matrix.
    """
    mat_len = len(mat)
    total = .5*(mat_len**2. - mat_len)
    mean = .0
    for i in range(mat_len):
        for j in range(mat_len):
            if j > i:
                mean += mat[i,j]/total
    return mean

def mat_vals(mat, mask = None, diag = True):
    """
    This method returns the values in a
    diagonally symmetric matrix excluding
    the diagonal terms and the elements masked.

    Parameters:
    -----------
    mat: np.ndarray
        Matrix with diagonaly symmetry.
    mask: np.array
        Mask selecting the elements in mat used.
    diag: bool
        It specifies whether the matrix is diagonally
        symmetric and repeated cases are excluded

    Returns:
    --------
    vals: np.array
        array of all the non-diagonal values
        of the matrix.
    """
    vals = []
    mat_len = mat.shape
    if mask is None:
        mask = np.ones(mat_len[0], dtype = bool)
    if mat_len[0] != mat_len[1] or diag is False:
        diag = False
        mask = np.ones_like(mat, dtype = bool)
    for ii in range(mat_len[0]):
        for jj in range(mat_len[1]):
            if diag is False and mask[ii,jj]:
                vals.append(mat[ii,jj])
            elif jj > ii and mask[ii] and mask[jj]:
                vals.append(mat[ii,jj])
    return np.array(vals)

def mat_mean_err(mat, jk_num = 20, rand_order = True, mask = None, diag = True):
    """
    This method measures the mean and Jack-Knife (JK) error of
    the values in a diagonally symmetric matrix excluding the
    diagonal terms.

    Parameters:
    -----------
    mat: np.ndarray
        Matrix with diagonaly symmetry.
    jk_num: int
        Number of JK subsamples to identify
    rand_order: bool
        If True, the indeces are assigned in a random order
    mask: np.array
        Mask selecting the elements in mat used
    diag: bool
        It specifies whether the matrix is diagonally
        symmetric and repeated cases are excluded

    Returns:
    --------
    mean_val: float
        Mean value of the array
    err: float
        JK error of the mean
    """
    mean_val = np.mean(mat_vals(mat, diag = diag))
    jk_ids = get_jk_indeces_1d(mat[0], jk_num, rand_order)#TODO fix for diag = False
    mean_arr_jk = np.array([np.mean(mat_vals(mat, jk_ids != i, diag = diag)) for i in range(jk_num)])
    err = jack_knife(np.array([mean_val]), mean_arr_jk)
    return mean_val, err

def get_jk_indeces_1d(array, jk_num, rand_order = True):
    """
    This method assigns equally distributed indeces to the elements of an array.

    Parameters:
    -----------
    array: np.array
        Data array
    jk_num: int
        Number of JK subsamples to identify
    rand_order: bool
        If True, the indeces are assigned in a random order

    Returns:
    --------
    jk_indeces: np.array
        Array assigning an index (from 0 to jk_num - 1) to
        each of the data elements
    """
    ratio = len(array)/jk_num + int(len(array)%jk_num > 0)
    jk_indeces = np.arange(len(array), dtype = int)/ratio
    np.random.shuffle(jk_indeces)
    return jk_indeces

def mean_err(array, jk_num = 50, rand_order = True):#TODO test: check stats on random case
    """
    This method measures the mean and Jack-Knife (JK) error of
    the values in an array

    Parameters:
    -----------
    array: np.array
        Data array
    jk_num: int
        Number of JK subsamples to identify
    rand_order: bool
        If True, the indeces are assigned in a random order

    Returns:
    --------
    mean_val: float
        Mean value of the array
    err: float
        JK error of the mean
    """
    mean_val = np.mean(array)
    jk_ids = get_jk_indeces_1d(array, jk_num, rand_order)
    mean_arr_jk = np.array([np.mean(array[jk_ids != i]) for i in range(jk_num)])
    err = jack_knife(np.array([mean_val]), mean_arr_jk)
    return mean_val, err

def jack_knife(var, jk_var):
    """
    This method gives the Jack-Knife error of var from the jk_var subsamples.

    Parameters:
    -----------
    var: float
        The mean value of the variable
    jk_var: np.array
        The variable from the subsamples. The shape of the jk_var must be (jk subsamples, bins)

    Returns:
    --------
    jk_err: float
        The JK error of var.
    """
    if type(var) == np.ndarray:
        jk_dim = jk_var.shape[0]
        err = (jk_dim - 1.)/jk_dim * (jk_var - var)**2.
        jk_err = np.sqrt(np.sum(err, axis = 0))
    else:
        jk_dim = len(jk_var)
        err = 0
        for i in range(jk_dim):
            for j in range(jk_dim):
                err += (jk_dim - 1.)/jk_dim*(jk_var[i] - var)*(jk_var[j] - var)
        jk_err = np.sqrt(err)
    return jk_err


def sig2pow(sig):
    """
    This method returns the confidence interval
    of a distribution from a sigma factor.

    Parameters:
    -----------
    sig: float
        Value indicating how many sigmas away the
        signal is.

    Returns:
    --------
    p: float
        Power indicating interval of confidence.
    """
    return sci_esp.erf(sig/np.sqrt(2.))

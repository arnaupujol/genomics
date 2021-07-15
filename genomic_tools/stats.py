#This module defines statistical operations for data analysis

import numpy as np
import scipy.special as sci_esp
from genomic_tools.relatedness import relatedness_mat
import pandas as pd

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
    if mat_len[0] != mat_len[1]:
        diag = False
    if mask is None:
        if diag:
            mask = np.ones(mat_len[0], dtype = bool)
        else:
            mask = np.ones(mat_len, dtype = bool)
    for ii in range(mat_len[0]):
        for jj in range(mat_len[1]):
            if diag:
                if jj > ii and mask[ii] and mask[jj]:
                    vals.append(mat[ii,jj])
            else:
                if mask[ii,jj]:
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
    ratio = int(len(array)/jk_num)
    res = int(len(array)%jk_num > 0)
    jk_indeces = (np.arange(len(array), dtype = int)/ratio).astype(int)
    jk_indeces[-res:] = np.random.randint(jk_num, size = res)#TODO test
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
    jk_var: np.ndarray
        The variable from the subsamples. The shape of the jk_var must be (jk subsamples, bins)

    Returns:
    --------
    jk_err: float
        The JK error of var.
    """
    jk_dim = np.prod(jk_var.shape)
    err = (jk_dim - 1.)/jk_dim * (jk_var - var)**2.
    jk_err = np.sqrt(np.sum(err, axis = 0))
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

def count_cases(variable, ignore_null = True):
    """This method counts the appearences of each case
    or label in a list of values.

    Parameters:
    -----------
    variable: (pd.Series, np.array)
        List of values or labels in the variable
    ignore_null: bool, default is True
        If True, is does not include the null cases

    Returns:
    --------
    cases: list
        List of appearing cases
    counts: list
        Number of appearences per case
    """
    cases = []
    counts = []
    #If we include the null cases, we add them here
    if not ignore_null:
        cases.append('None')
        counts.append(np.sum(variable.isnull()))
    #We take only the no null values and count their cases
    for case in variable[variable.notnull()].unique():
        cases.append(case)
        counts.append(np.sum(variable.notnull()&(variable == case)))
    return cases, counts


def bootstrap_resample(data):
    """
    This method creates a shuffled version of the data
    with resampling (so repetitions can happen).

    Parameters:
    -----------
    data: np.ndarray
        Data with shape (samples, values)

    Returns:
    --------
    new_data: np.ndarray
        New data resample from the original, resampling
        the samples with their data
    """
    data_len = len(data)
    rand_ints = np.random.randint(0, data_len, data_len)
    new_data = data[rand_ints]
    return new_data

def mat_bootstrap_mean_err(x_data, y_data, nrands = 100, method = 'pcorr', diag = True):
    """
    This method measures the mean and error of the relatedness of two
    populations using Bootstrap.

    Parameters:
    -----------
    x_data: np.ndarray
        Data of the first population with shape (samples, values)
    y_data: np.ndarray
        Data of the second population with shape (samples, values)
    nrands: int
        Number of Bootstrap iterations to calculate the error
    method: str {'L2', 'L1', 'jaccard', 'bin_sharing', 'pcorr'}
        Metrics used to measure the relatedness (default is 'L2')

            'L2':
                L2-norm distance, Euclidean

            'L1':
                L1-norm distance

            'jaccard':
                Jaccard distance

            'bin_sharing':
                Binary sharing

            'pcorr':
                Pearson correlation coefficient
    diag: bool
        It specifies whether the relatedness matrix is diagonally symmetric
        (x_data and y_data are the same) and repeated cases are excluded

    Returns:
    --------
    mean: float
        Mean relatedness of the sample pairs of the two populations
    err: float
        Bootstrap error of the mean
    mean_resamples: float
        Mean relatedness over all the resamples
    """
    means = np.zeros(nrands)
    mat = relatedness_mat(x_data, y_data, method = method)
    mean = np.mean(mat_vals(mat, diag = diag))
    for i in range(nrands):
        x_data_s = bootstrap_resample(x_data)
        y_data_s = bootstrap_resample(y_data)
        mat_s = relatedness_mat(x_data_s, y_data_s, method = method)
        means[i] = np.mean(mat_vals(mat_s, diag = False))#Diagonal sym broken in resampling
    err = np.std(means)
    mean_resamples = np.mean(means)
    return mean, err, mean_resamples

def bootstrap_mean_err(data, nrands = 100, weights = None, ret_resamples = False):
    """
    This method calculates the mean and error of an array of values
    using the Bootstrap method.

    Parameters:
    -----------
    data: np.ndarray
        A 1-d array of values
    nrands: int
        Number of Bootstrap iteration to calculate the error
    weights: np.array
        Weights to apply to the data to calculate the mean.
    ret_resamples: bool
        If True, the means of all the resamples are return

    Returns:
    --------
    mean: float
        Mean value of the data
    err: float
        Bootstrap error of the mean
    mean_resamples: float
        Mean over all the resamples
    means: np.array
        Means of all the resamples
    """
    means = np.zeros(nrands)
    if weights is None:
        weights = np.ones_like(data)
    try:
        mean = np.sum(data*weights)/np.sum(weights)
    except ZeroDivisionError:
        print('Zero division error when calculating bootstram mean')
        mean = np.nan
    for i in range(nrands):
        r_data = bootstrap_resample(np.array([data, weights]).T)
        r_vals, r_weight = r_data[:,0], r_data[:,1]
        try:
            means[i] = np.sum(r_vals*r_weight)/np.sum(r_weight)
        except ZeroDivisionError:
            print('Zero division error when calculating bootstram mean')
            means[i] = np.nan
    err = np.std(means)
    mean_resamples = np.mean(means)
    if ret_resamples:
            return mean, err, mean_resamples, means
    else:
        return mean, err, mean_resamples

def mean_prev_time_bins(dates, positive, data_mask = None, nbins = 10, \
                        nrands = 1000, weights = None, verbose = True, \
                        ret_resamples = False):
    """
    This method calculates the mean positivity of data in time bins.

    Parameters:
    -----------
    dates: pd.DataFrame
        Time dates of visits
    positive: np.array
        Values specifying positivity (0 or 1)
    data_mask: np.array
        Mask to apply to data
    nbins: int or sequence of scalars
        Number of time equally spaced bins (if int) or bin edges (if scalars) used
    nrands: int
        Number of random Bootstrap iterations to calculate the errors
    weights: np.array
        Weights to apply to the data to calculate the mean
    verbose: bool
        It specifies the verbose mode
    ret_resamples: bool
        If True, the measurements of all the resamples are return

    Returns:
    --------
    mean_dates: list
        Mean date per time bin
    mean_prev: np.array
        Mean prevalence per bin
    err_prev: np.array
        Bootstrap error of the mean prevalence per bin
    mean_prevs: np.ndarray
        Mean prevalences per bin for all the resamples
    """
    #Bins defined
    if weights is None:
        weights = np.ones_like(positive)
    if data_mask is None:
        data_mask = np.ones_like(dates, dtype=bool)
    if type(nbins) is int:
        out, bins = pd.cut(dates[data_mask], nbins, retbins = True)
    elif len(nbins) > 1:
        bins = nbins
        nbins = len(bins) - 1
    else:
        if verboe:
            print("Error: incorrect assignment of nbins (int or list): " + str(nbins))
    mean_dates = []
    mean_prev = np.zeros(nbins)
    err_prev = np.zeros(nbins)
    mean_prevs = np.zeros((nbins, nrands))

    #Calculate mean time and prevalence per time bin
    for i in range(nbins):
        mask = (dates >= bins[i])&(dates < bins[i + 1])&data_mask
        mean_dates.append(dates[mask].mean())
        if np.sum(mask) > 0:
            m, e, mb, means = bootstrap_mean_err(np.array(positive[mask]), \
                                        nrands = nrands, weights = weights[mask], \
                                        ret_resamples = True)
        else:
            if verbose:
                print("Warning: bin number ", i, " empty")
            m, e, mb, means = np.nan, np.nan, np.nan, np.nan*np.zeros(nrands)
        mean_prev[i] = m
        err_prev[i] = e
        mean_prevs[i] = means
    if ret_resamples:
        return mean_dates, mean_prev, err_prev, mean_prevs
    else:
        return mean_dates, mean_prev, err_prev

def mean_pos_diff(pos_1, pos_2, nrands = 100, weights_1 = None, weights_2 = None):
    """
    This method calculates the differences in prevalences between two populations.

    Parameters:
    -----------
    pos_1: np.array
        Positivity information about first population
    pos_2: np.array
        Positivity information about second population
    nrands: int
        Number of random subsamples for Bootstrap error
    weights_1: np.array
        Weights of the first population
    weights_2: np.array
        Weights of the second population

    Returns:
    --------
    pos_diff: float
        Positivity difference between the two populations
    diff_err: float
        Error in the measured difference
    mean_boots: float
        Measured value from Bootstrap subsamples
    """
    pos_diffs = np.zeros(nrands)
    if weights_1 is None:
        weights_1 = np.ones_like(pos_1)
    if weights_2 is None:
        weights_2 = np.ones_like(pos_2)
    mean_pos_1 = np.sum(pos_1*weights_1)/np.sum(weights_1)
    mean_pos_2 = np.sum(pos_2*weights_2)/np.sum(weights_2)
    pos_diff = mean_pos_1 - mean_pos_2
    for i in range(nrands):
        pos_1_r = bootstrap_resample(np.array([pos_1, weights_1]).T)
        r_vals_1, r_weight_1 = pos_1_r[:,0], pos_1_r[:,1]
        pos_2_r = bootstrap_resample(np.array([pos_2, weights_2]).T)
        r_vals_2, r_weight_2 = pos_2_r[:,0], pos_2_r[:,1]
        mean_1 = np.sum(r_vals_1*r_weight_1)/np.sum(r_weight_1)
        mean_2 = np.sum(r_vals_2*r_weight_2)/np.sum(r_weight_2)
        pos_diffs[i] = mean_1 - mean_2
    diff_err = np.std(pos_diffs)
    mean_boots = np.mean(pos_diffs)
    return pos_diff, diff_err, mean_boots

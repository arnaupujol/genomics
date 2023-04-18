#This module contains functions to operate on barcodes.

import numpy as np

def diff_count(bar1, bar2, verbose = False):
    """This method counts the fraction of different SNPs between two barcodes.
    X and N elements are excluded from the comparison.

    Parameters:
    -----------
    bar1, bar2: list of str
        Lists of strings representing the SNPs
    verbose: bool
        It specifies whether text is written on the process

    Returns:
    --------
    frac: float
        Fraction of common SNPs (from 0 to 1)
    """
    if len(bar1) != len(bar2):
        if verbose:
            print('Warning: barcodes bar1 and bar2 have different lengths: ' + str(len(bar1)) + ', ' + str(len(bar2)))
        return np.nan
    if  '-' in [bar1, bar2]:
        if verbose:
            print('Warning: at least one barcode is missing')
        return np.nan
    #X and N elements are excluded from the comparison
    N_cases = base_is(bar1, 'N') + base_is(bar2, 'N')
    X_cases = base_is(bar1, 'X') + base_is(bar2, 'X')
    mask = np.logical_not(N_cases + X_cases)
    #calculate fraction of common elements
    bar1, bar2 = barcode_list(bar1), barcode_list(bar2)
    diff = bar1[mask] == bar2[mask]
    frac = np.sum(diff)/float(np.sum(mask))
    return frac

def bar_diff(bar1, bar2, verbose = False):
    """
    This method returns a boolean specifying where the two barcodes are the same.

    Parameters:
    -----------
    bar1, bar2: list of str
        Lists of strings representing the SNPs
    verbose: bool
        It specifies whether text is written on the process

    Returns:
    --------
    diff: np.array
        A boolean of the size of bar1, bar2 specifying whether the bases are the same
    """
    if len(bar1) != len(bar2):
        if verbose:
            print('Warning: barcodes bar1 and bar2 have different lengths: ' + str(len(bar1)) + ', ' + str(len(bar2)))
        return False
    if  '-' in [bar1, bar2]:
        if verbose:
            print('Warning: at least one barcode is missing')
        return False
    bar1, bar2 = barcode_list(bar1), barcode_list(bar2)
    diff = bar1 == bar2
    return diff

def relatedness_matrix(bars):
    """
    This method calculates a matrix where index i,j corresponds to the SNP relatedness in cases i,j

    Parameters:
    -----------
    bars: list
        List of barcodes

    Returns:
    --------
    diff_matrix: np.ndarray
        Array of size (len(bars), len(bars)) with the fractions of common bases
    """
    diff_matrix = np.zeros((len(bars),len(bars)))
    for i, b1 in enumerate(bars):
        for j, b2 in enumerate(bars):
            diff_matrix[i][j] = diff_count(b1,b2)
    return diff_matrix

def base_is(bar, char):
    """
    This method returns a boolean array specifying where the
    SNPs of bar correspond to a character char.

    Parameters:
    -----------
    bar: list of str
        List of strings representing the SNPs
    char: str
        Character, can be A,C,T,P,N,X...

    Returns:
    --------
    ischar: np.array
        Boolean array of the shape of bar
    """
    arr_bar = barcode_list(bar)
    ischar = arr_bar == char
    return ischar

def barcode_list(bar):
    """
    This method converts a barcode into an array of characters.

    Parameters:
    -----------
    bar: list
        List of strings representing the SNPs

    Returns:
    --------
    array_bar: np.array
        Array of the same length as bar
    """
    list_bar = list(bar)
    array_bar = np.array(list_bar, dtype=str)
    return array_bar

def barcode_is_valid(bar):
    """
    This method returns a boolean specifying whether the barcode is valid.

    Parameters:
    -----------
    bar: list
        List of strings representing the SNPs

    Returns:
    --------
    boolean specifying if the barcode is valid
    """
    return (bar != '-')&(bar != 'X'*101)

def barcodes_are_valid(bars):
    """
    This method returns a boolean array specifying what barcodes are valid.

    Parameters:
    -----------
    bars: list
        List of barcodes

    Returns:
    --------
    valids: np.array
        Boolean array of the shape of bars
    """
    valids = np.array([barcode_is_valid(bar) for bar in bars])
    return valids

def barcode_is_polyclonal(bar):
    """
    This method returns a boolean specifying whether the barcode is polyclonal.

    Parameters:
    -----------
    bar: list
        List of strings representing the SNPs

    Returns:
    --------
    boolean specifying if the barcode is polyclonal
    """
    return barcode_is_valid(bar) and 'N' in bar

def barcodes_are_polyclonal(bars):
    """
    This method returns a boolean array specifying what barcodes are polyclonal.

    Parameters:
    -----------
    bars: list
        List of barcodes

    Returns:
    --------
    valids: np.array
        Boolean array of the shape of bars
    """
    valids = np.array([barcode_is_polyclonal(bar) for bar in bars])
    return valids

def barcode_is_monoclonal(bar):
    """
    This method returns a boolean specifying whether the barcode is monoclonal.

    Parameters:
    -----------
    bar: list
        List of strings representing the SNPs

    Returns:
    --------
    boolean specifying if the barcode is monoclonal
    """
    return barcode_is_valid(bar) and 'N' not in bar

def barcodes_are_monoclonal(bars):
    """
    This method returns a boolean array specifying what barcodes are monoclonal.

    Parameters:
    -----------
    bars: list
        List of barcodes

    Returns:
    --------
    valids: np.array
        Boolean array of the shape of bars
    """
    valids = np.array([barcode_is_monoclonal(bar) for bar in bars])
    return valids


def bar2vec(bar, base_dict):
    """
    This method translates the barcodes to vectors.

    Parameters:
    -----------
    bar: list
        List of strings representing the SNPs
    base_dict: dict
        Dictionary specifying vector convertion

    Returns:
    --------
    vect: np.array
        Binary vector representing the barcode
    """
    bar_list = barcode_list(bar)
    vect = np.array([base_dict[i] for i in bar_list])
    vect = vect.reshape(np.prod(vect.shape))
    return vect

def bars2vecs(bars, base_dict):
    """
    This method translates the barcodes to vectors.

    Parameters:
    -----------
    bars: list
        List of barcodes
    base_dict: dict
        Dictionary specifying vector convertion

    Returns:
    --------
    vecs: np.ndarray
        binary vectors
    """
    vecs = np.array([bar2vec(bar, base_dict) for bar in bars])
    return vecs

def count_unique_bases(bases):
    """
    This method counts the number of different bases present in a list.

    Parameters:
    -----------
    bases: list
        List of bases

    Returns:
    count: int
        Number of unique bases present in the list
    """
    uniques = np.unique(bases)
    count = len(uniques)
    if 'X' in uniques:
        count -= 1
    if 'N' in uniques:
        if count > 2:
            count -= 1
    return count

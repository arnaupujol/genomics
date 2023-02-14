#This module contains methods to analyse microhaplotype data

import pandas as pd
import numpy as np


def get_allele_frequencies(dataframe, locus_name = 'locus', \
                           allele_name = 'allele', freq_name = 'freq', \
                           use_presence = True):
    """
    This method calculates the allele frequencies from different loci and
    samples.

    Parameters:
    -----------
    dataframe: pd.DataFrame
        A dataframe with information on the loci, alleles and frequencies for
        the samples.
    locus_name: str
        The name of the column referring to the loci identifiers from dataframe.
    allele_name: str
        The column name of the alleles from dataframe.
    freq_name: str
        The column name of the allele frequencies in each individual sample.
    use_presence: bool
        If true, presence or absence instead of the frequency in each individual
        sample is considered.

    Returns:
    --------
    allele_freq: pd.DataFrame
        A dataframe with the information on loci, alleles, total frequency (or
        presence) of alleles in loci, total frequency (or presence) from each
        locus, and the allele frequencies.
    """
    data = dataframe.copy()
    if use_presence:
        data[freq_name] = np.array(data[freq_name] > 0, dtype = int)
    #Total frequence per locus
    loci_total_freq = data[[locus_name, freq_name]].groupby(locus_name).sum()
    loci_total_freq.columns = ['locus_freq_sum']
    #Dataframe relating loci with alleles
    loci_alleles = data[[locus_name, allele_name]].drop_duplicates()
    #Allele frequencies
    allele_freq = data[[allele_name, freq_name]].groupby(allele_name).sum()
    allele_freq.columns = ['allele_freq_sum']
    allele_freq = pd.merge(allele_freq, loci_alleles, left_index = True, \
                           right_on = allele_name, how = 'left')
    allele_freq = pd.merge(allele_freq, loci_total_freq, left_on = locus_name, \
                           right_index = True, how = 'left')
    allele_freq['allele_freq'] = allele_freq['allele_freq_sum']/allele_freq['locus_freq_sum']
    return allele_freq

def exp_He_locus(p_alleles):
    """
    This method calculates the expected heterozygosity in a locus given its
    allele frequencies.

    Parameters:
    -----------
    p_alleles: np.array
        An array with the allele frequencies or probabilities.

    Returns:
    --------
    he: float
        The expected Heterozygosity of the locus.
    """
    he = 1 - np.sum(p_alleles**2)
    return he

def exp_He(data, locus_name = 'locus', freq_name = 'allele_freq', \
           verbose = False):
    """
    This method calculates the expected heterozygosity for each loci from the
    given allele frequencies.

    Parameters:
    -----------
    data: pd.DataFrame
        A dataframe with information on the allele frequencies in each locus.
        Each row must show the frequency of a different allele and its
        corresponding locus.
    locus_name: str
        Column name of the loci.
    freq_name: str
        Column name showing the allele frequencies.
    verbose: bool
        If True, the percentage of the progress of the calculation over all loci
        is printed.

    Returns:
    --------
    loci_He: pd.DataFrame
        A dataframe with each locus and the corresponding expected
        heterozygosity.
    overall_He: float
        The mean He over all loci.
    """
    #Create DataFrame to store He results
    loci_He = pd.DataFrame({
                        locus_name : data[locus_name].unique(),
                        'He' : pd.Series(np.nan*np.zeros(len(data[locus_name].unique())))
                        })
    for i in loci_He.index:
        if verbose:
            percent_5 = int(len(loci_He)/20)
            if i%percent_5 == 0:
                print(round(i/len(loci_He)*100,1), '% of loci calculated')
        #Locus name
        locus = loci_He[locus_name][i]
        #Mask to select alleles from this locus
        mask = data[locus_name] == locus
        #Allele frequencies
        p_alleles = np.array(data[mask][freq_name])
        #He of the locus
        he = exp_He_locus(p_alleles)
        loci_He.loc[i, 'He'] = he
    overall_He = np.mean(loci_He['He'])
    return loci_He, overall_He

def He_from_samples(data, locus_name = 'p_name', allele_name = 'h_popUID', \
                           freq_name = 'c_AveragedFrac', use_presence = True):
    """
    This method calculates the expected heterozygosity for each loci in a given
    set of samples.

    Parameters:
    -----------
    data: pd.DataFrame
        A dataframe with information on the sample id and allele frequencies in
        for each locus. Each row must show the sample id, the frequency of its
        different alleles and their corresponding loci.
    locus_name: str
        Column name of the loci.
    allele_name: str
        The column name of the alleles from dataframe.
    freq_name: str
        Column name showing the allele frequencies.
    use_presence: bool
        If true, presence or absence instead of the frequency in each individual
        sample is considered.

    Returns:
    --------
    loci_He: pd.DataFrame
        A dataframe with each locus and the corresponding expected
        heterozygosity.
    overall_He: float
        The mean He over all loci.
    """
    loci_alleles = data[[locus_name, allele_name]].drop_duplicates()
    allele_freq = get_allele_frequencies(data, locus_name = locus_name, allele_name = allele_name, \
                           freq_name = freq_name, use_presence = use_presence)
    loci_alleles = pd.merge(loci_alleles, allele_freq[[allele_name, 'allele_freq', 'allele_freq_sum', 'locus_freq_sum']], \
                            on = allele_name, how = 'left')
    loci_He, overall_He = exp_He(loci_alleles, locus_name = locus_name, freq_name = 'allele_freq')
    return loci_He, overall_He

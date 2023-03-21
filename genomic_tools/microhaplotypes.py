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

def fst_all_loci(subsample_data, sample_data):
    """
    This method calculates Fst between two populations 
    for all loci and calculates the mean Fst overall loci. 
    
    Parameters:
    -----------
    subsample_data: pd.DataFrame
        Data containing the loci, allele frequencies and sample id of the 
        subsample data.
    sample_data: pd.DataFrame
        Data containing the loci, allele frequencies and sample id of the 
        total data.
    locus: str
        Name of the locus to analyse. 
        
    Returns:
    --------
    mean_fst: float
        Mean Fst accross all loci.
    fst_loci: np.array
        All Fst measurements for all loci. 
    """
    all_loci = sample_data['p_name'].unique()
    #Calculating Fst for all loci
    fst_loci = []
    for l in all_loci:
        fst_loci.append(fst_locus(subsample_data, sample_data, l))
    fst_loci = np.array(fst_loci)
    mask = np.isfinite(fst_loci)
    mean_fst = np.mean(fst_loci[mask])
    return mean_fst, fst_loci

def fst_locus(subsample_data, sample_data, locus):
    """
    This method calculates the Wright's Fst for a given locus comparing 
    a two populations, one representing a subsample and the other one 
    the total population. 
    
    Parameters:
    -----------
    subsample_data: pd.DataFrame
        Data containing the loci, allele frequencies and sample id of the 
        subsample data.
    sample_data: pd.DataFrame
        Data containing the loci, allele frequencies and sample id of the 
        total data.
    locus: str
        Name of the locus to analyse. 
    
    Returns:
    --------
    fst: float
        Wright's Fst result
    """
    #Allele frequencies
    allele_freq = get_allele_frequencies(sample_data, locus_name = 'p_name', \
                                         allele_name = 'h_popUID', \
                                         freq_name = 'c_AveragedFrac')
    #Allele frequency in locus
    locus_mask = allele_freq['p_name'] == locus
    p_locus = allele_freq[locus_mask]['allele_freq']
    
    #Get allele_freq for subpopulation
    allele_freq_sub = get_allele_frequencies(subsample_data, locus_name = 'p_name', \
                                             allele_name = 'h_popUID', \
                                             freq_name = 'c_AveragedFrac')
    #Mean allele frequency in a locus for subpopulation
    locus_mask_sub = allele_freq_sub['p_name'] == locus
    p_locus_sub = allele_freq_sub[locus_mask_sub]['allele_freq']
    
    #Calculating Fst
    fst = fst_from_p(p_locus_sub, p_locus)
    if fst < -1:
        import pdb; pdb.set_trace()
    return fst

def fst_from_p(p_locus_sub, p_locus):
    """
    This method calculates the Wright's Fst from the allele frequencies 
    of the subpopulation and the total population. 
    
    Parameters:
    -----------
    p_locus_sub: np.array
        Allele frequencies of the sub-population.
    p_locus: np.array
        Allele frequencies of the total population.
    
    Returns:
    --------
    fst: float
        Wright's Fst result
    """
    #Total number of alleles in locus
    allele_num = len(p_locus)
    if allele_num == 1: 
        print("Only one allele present")
        fst = np.nan
    #Overall mean allele frequency
    mean_p = np.mean(p_locus)
    subpop_factor = np.sum(p_locus_sub*(1 - p_locus_sub))/allele_num
    total_factor = mean_p*(1 - mean_p)
    fst = 1 - subpop_factor/total_factor
    return fst

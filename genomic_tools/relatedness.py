"""
This module contains methods to compute metrics of genomic relatedness.
"""

import numpy as np
from scipy import spatial
import scipy.stats as sci_stats
import pandas as pd

def dist(vect_a, vect_b):
    """
    This method calculates the n-dimensional of two vectors.

    Parameters:
    -----------
    vect_a: np.ndarray
        Vector with high-dimensional position
    vect_b: np.ndarray
        Vector with high-dimensional position with the same size as vect_a

    Returns:
    --------
    d: float
        The distance modulus between the two vectors
    """
    d = np.linalg.norm(vect_a-vect_b)
    return d

def L1(vect_a, vect_b):
    """
    This method calculates the L1 norm of the difference between two vectors.

    Parameters:
    -----------
    vect_a: np.ndarray
        Vector with high-dimensional position
    vect_b: np.ndarray
        Vector with high-dimensional position with the same size as vect_a

    Returns:
    --------
    d: float
        The distance norm L1 between the two vectors
    """
    d = np.mean(np.abs(vect_a-vect_b))
    return d

def jaccard_dist(vect_a, vect_b):
    """
    This method calculates the Jaccard distance between two vectors.

    Parameters:
    -----------
    vect_a: np.ndarray
        Vector with high-dimensional position
    vect_b: np.ndarray
        Vector with high-dimensional position with the same size as vect_a

    Returns:
    --------
    d: float
        The Jaccard distance between the two vectors
    """
    PHS = np.sum((vect_a > 0)*(vect_b > 0))/float(np.sum((vect_a > 0)+(vect_b > 0)))
    return 1 - PHS



def relatedness_mat(data_1, data_2, method = 'L2'):
    """
    This method builds a matrix with the relatedness between two
    data samples for each pair comparison.

    Parameters:
    -----------
    data_1: np.ndarray
        Data with shape (samples, properties)
    data_2: np.ndarray
        Data with shape (samples, properties)
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

    Returns:
    --------
    rel_mat: np.ndarray
        Matrix of shape (data_1 samples, data_2 samples) with their
        relatedness
    """
    if method == 'L2': #TODO test
        rel_mat = spatial.distance_matrix(data_1, data_2)
    elif method == 'L1': #TODO test
        rel_mat = spatial.distance_matrix(data_1, data_2, p = 1)
    else:
        rel_mat = np.zeros((data_1.shape[0],data_2.shape[0]))
        for i in range(rel_mat.shape[0]):
            for j in range(rel_mat.shape[1]):
                if method == 'jaccard':
                    rel_mat[i,j] = jaccard_dist(data_1[i], data_2[j])
                if method == 'bin_sharing':
                    rel_mat[i,j] = jaccard_dist(data_1[i], data_2[j]) < 1
                if method == 'pcorr':
                    rel_mat[i,j] = sci_stats.pearsonr(data_1[i], data_2[j])[0]
    return rel_mat

def raw_case_classification(ibd_res_meta, travel_1 = 'travel_prov', travel_2 = 'travel_prov2', \
                            r_origin = 'rel_origin', r_des1 = 'rel_dest1', r_des2 = 'rel_dest2', \
                            class_name = 'classification'):
    """
    This method classifies infections as local or imported by simply comparing the genetic relatedness 
    of the samples with their origin population and the travel destination populations. Cases are 
    classified as imported only when the sample is more genetically related to the population of a travel 
    destination than from the local population, leaving them as unclassified when no data is available or 
    complete, or when relatedness values are equal. 
    
    Parameters:
    -----------
    ibd_res_meta: pd.DataFrame
        Data frame containing the pairwise IBD results and other metadata columns.
    travel_1: pd.Series
        Name of the column in ibd_res_meta for the travel destination.
    travel_2: pd.Series
        Name of the column in ibd_res_meta for the second travel destination.
    r_origin: pd.Series
        Name of the column in ibd_res_meta for the relatedness of samples with their local population. 
    r_des1: pd.Series
        Name of the column in ibd_res_meta for the relatedness of samples with the population of their 
        first reported travel.
    r_des2: pd.Series
        Name of the column in ibd_res_meta for the relatedness of samples with the population of their 
        second reported travel.
    class_name: str
        Name of the variable of the case classification. 
    
    Returns: 
    --------
    ibd_res_meta: pd.DataFrame
        Data frame containing the pairwise IBD results and other metadata columns, 
        including the case classification. 
    """
    #initially setting all as unclassified
    ibd_res_meta[class_name] = 'Unclassified'
    #if no travel, classify as local
    no_travel_mask = ibd_res_meta[travel_1].isnull()&ibd_res_meta[travel_2].isnull()
    ibd_res_meta.loc[no_travel_mask, class_name] = 'Local'
    #If only travel 1, compare dest1 with origin
    trav1_mask = ibd_res_meta[travel_1].notnull()&ibd_res_meta[travel_2].isnull()
    mask_local = ibd_res_meta[r_origin] > ibd_res_meta[r_des1]
    mask_equal = ibd_res_meta[r_origin] == ibd_res_meta[r_des1]
    mask_imported1 = ibd_res_meta[r_des1] > ibd_res_meta[r_origin]
    ibd_res_meta.loc[trav1_mask&mask_local, class_name] = 'Local'
    ibd_res_meta.loc[trav1_mask&mask_equal, class_name] = 'Unclassified'
    ibd_res_meta.loc[trav1_mask&mask_imported1, class_name] = 'Imported'
    #If only travel 2, compare dest2 with origin
    trav2_mask = ibd_res_meta[travel_1].isnull()&ibd_res_meta[travel_2].notnull()
    mask_local = ibd_res_meta[r_origin] > ibd_res_meta[r_des2]
    mask_equal = ibd_res_meta[r_origin] == ibd_res_meta[r_des2]
    mask_imported2 = ibd_res_meta[r_des2] > ibd_res_meta[r_origin]
    ibd_res_meta.loc[trav2_mask&mask_local, class_name] = 'Local'
    ibd_res_meta.loc[trav2_mask&mask_equal, class_name] = 'Unclassified'
    ibd_res_meta.loc[trav2_mask&mask_imported2, class_name] = 'Imported'
    #If both travels not null, compare both with origin
    trav12_mask = ibd_res_meta[travel_1].notnull()&ibd_res_meta[travel_2].notnull()
    mask_local = (ibd_res_meta[r_origin] > ibd_res_meta[r_des1])&(ibd_res_meta[r_origin] > ibd_res_meta[r_des2])
    ibd_res_meta.loc[trav12_mask&mask_local, class_name] = 'Local'
    mask_equal1 = (ibd_res_meta[r_origin] == ibd_res_meta[r_des1])&(ibd_res_meta[r_origin] >= ibd_res_meta[r_des2])
    mask_equal2 = (ibd_res_meta[r_origin] >= ibd_res_meta[r_des1])&(ibd_res_meta[r_origin] == ibd_res_meta[r_des2])
    ibd_res_meta.loc[trav12_mask&mask_equal1&mask_equal2, class_name] = 'Unclassified'
    mask_imported = (ibd_res_meta[r_origin] < ibd_res_meta[r_des1]) | (ibd_res_meta[r_origin] < ibd_res_meta[r_des2])
    ibd_res_meta.loc[trav12_mask&mask_imported, class_name] = 'Imported'
    return ibd_res_meta

def r_importation_prob(ibd_res_meta, travel_1 = 'travel_prov', travel_2 = 'travel_prov2', \
                       r_origin = 'rel_origin', r_des1 = 'rel_dest1', r_des2 = 'rel_dest2', \
                       class_name = 'prob_imported'):
    """
    This method estimates the probability of cases being imported at the individual level from 
    the fraction of the genetic relatedness of the sample between the origin and the destination 
    populations, as P(I) = sum_i (r_dest_i) / (sum_i (r_dest_i) + r_origin), where r_dest_i is 
    the relatedness of the sample with the population from the destination of the travel i, 
    and r_origin the relatedness with the local population. 
    
    Parameters:
    -----------
    ibd_res_meta: pd.DataFrame
        Data frame containing the pairwise IBD results and other metadata columns.
    travel_1: pd.Series
        Name of the column in ibd_res_meta for the travel destination.
    travel_2: pd.Series
        Name of the column in ibd_res_meta for the second travel destination.
    r_origin: pd.Series
        Name of the column in ibd_res_meta for the relatedness of samples with their local population. 
    r_des1: pd.Series
        Name of the column in ibd_res_meta for the relatedness of samples with the population of their 
        first reported travel.
    r_des2: pd.Series
        Name of the column in ibd_res_meta for the relatedness of samples with the population of their 
        second reported travel.
    class_name: str
        Name of the variable of the case classification. 
    
    Returns: 
    --------
    ibd_res_meta: pd.DataFrame
        Data frame containing the pairwise IBD results and other metadata columns, 
        including the case classification. 
    """
    #initially setting all as unclassified
    ibd_res_meta[class_name] = np.nan
    #if no travel, P(I) = 0
    no_travel_mask = ibd_res_meta[travel_1].isnull()&ibd_res_meta[travel_2].isnull()
    ibd_res_meta.loc[no_travel_mask, class_name] = 0
    #If only travel 1, use only r_des1 and r_origin
    trav1_mask = ibd_res_meta[travel_1].notnull()&ibd_res_meta[travel_2].isnull()
    ibd_res_meta.loc[trav1_mask, class_name] = ibd_res_meta[r_des1]/(ibd_res_meta[r_des1] + ibd_res_meta[r_origin])
    #If only travel 2, use only r_des2 and r_origin
    trav2_mask = ibd_res_meta[travel_1].isnull()&ibd_res_meta[travel_2].notnull()
    ibd_res_meta.loc[trav2_mask, class_name] = ibd_res_meta[r_des2]/(ibd_res_meta[r_des2] + ibd_res_meta[r_origin])
    #If both travels not null, use r_des1, r_des2 and r_origin
    trav12_mask = ibd_res_meta[travel_1].notnull()&ibd_res_meta[travel_2].notnull()
    ibd_res_meta.loc[trav12_mask, class_name] = (ibd_res_meta[r_des1] + ibd_res_meta[r_des2])/(ibd_res_meta[r_des1] + ibd_res_meta[r_des2] + ibd_res_meta[r_origin])
    return ibd_res_meta

#old deprecated function:
def individual_case_classification(province_samples, destiny_samples, ibdfrac_per_cat, \
                                   visualise = True, verbose = True):#TODO address when multiple travels are reported
    """
    This method classifies as local or imported infections cases with reported travels. 
    
    Parameters:
    -----------
    province_samples: str
        List with origin location of each sample.
    destiny_samples: str
        List with travel destination of the sample. 
    ibdfrac_per_cat: pd.DataFrame
        2-D matrix with the fraction of related pairs between each sample (columns) and 
        the different locations (rows). 
    visualise: bool
        If true, visualisation plots are produced. 
    verbose: bool
        If true, output text is printed. 
        
    
    Returns: 
    --------
    case_class: pd.DataFrame
        Data frame with information on the samples and their case classification.
    """
    #DataFrame of case classification
    case_class = pd.DataFrame(columns = ['sampleID', 'origin', 'destination', 'rel_origin', \
                            'rel_destination', 'classification'])#TODO address for multiple reported travels
    #Case classification for samples
    for i in range(len(province_samples)):
        if verbose: 
            print("Sample:", ibdfrac_per_cat.columns[i])
            print("Travel:", province_samples[i], "->", destiny_samples[i])
        ibd_orig = ibdfrac_per_cat.loc[province_samples[i], ibdfrac_per_cat.columns[i]]
        if destiny_samples[i] in list(ibdfrac_per_cat.index):
            ibd_dest = ibdfrac_per_cat.loc[destiny_samples[i], ibdfrac_per_cat.columns[i]]
        else: 
            ibd_dest = np.nan
        if verbose: 
            print("IBD at origin:", ibd_orig)
            print("IBD at destiny:", ibd_dest)
        label = None
        #classification
        if ibd_orig > ibd_dest:
            classif = 'local'
            if verbose: 
                print("                ---------")
                print("Classification: | local |")
        elif ibd_orig < ibd_dest:
            classif = 'imported'
            if verbose: 
                print("                ------------")
                print("Classification: | imported |")
        else:
            classif = 'Unclassified'
            if verbose: 
                print("Unclassified:", ibd_orig, "vs", ibd_dest)
        case_class = pd.concat([case_class, pd.DataFrame({'sampleID' : [ibdfrac_per_cat.columns[i]], \
                                            'origin' : [province_samples[i]], \
                                             'destination' : [destiny_samples[i]], \
                                             'rel_origin' : [ibd_orig], \
                                             'rel_destination' : [ibd_dest], \
                                             'classification' : [classif]})])
        if verbose:
            print("----------------------------")
    if visualise:
        for i, case in enumerate(case_class['classification'].unique()):
            mask = case_class['classification'] == case
            plt.scatter(case_class['rel_origin'][mask], case_class['rel_destination'][mask], label = case)
        ibd_max = min(case_class['rel_origin'].max(), case_class['rel_destination'].max())
        plt.plot([0, ibd_max], [0, ibd_max], c = 'tab:grey', label = [r'$R_{origin} = R_{dest}$'])
        plt.xlabel("IBD fraction at origin")
        plt.ylabel("IBD fraction at destiny")
        plt.legend()
        plt.show()
    if verbose: 
        print("Total local:", np.sum(case_class['classification'] == 'local'))
        print("Total imported:", np.sum(case_class['classification'] == 'imported'))
        print("Total unkwnown:", np.sum(case_class['classification'] == 'Unclassified'))
    return case_class

def importation_statistics(ibd_res_meta, rel_origin = 'rel_origin', rel_dest1 = 'rel_dest1', \
                           rel_dest2 = 'rel_dest2', prob_imp1 = 'prob_imp1', \
                            prob_imp2 = 'prob_imp2', prob_imported = 'prob_imported'):
    """
    This method outputs the statistics of imported cases per travel 
    destination. 
    
    Parameters: 
    -----------
    ibd_res_meta: pd.DataFrame
        Data frame containing the pairwise IBD results and other metadata 
        columns including relatedness statistics and case classification. 
    rel_origin: str
        Name of the variable describing the relatedness of the sample with the origin population. 
    rel_dest1: str
        Name of the variable describing the relatedness of the sample with the population of 
        the first travel destination.
    rel_dest2: str
        Name of the variable describing the relatedness of the sample with the population of 
        the second travel destination. 
    prob_imp1: str
        Name to save the probability of importation from the first travel. 
    prob_imp2: str
        Name to save the probability of importation from the second travel. 
    prob_imported: str
        Name to save the total probability of importation. 
        
    Returns: 
    --------
    imported_stats: pd.DataFrame
        Data frame with the statistics on the imported cases. 
    """
    #Specifying importation contributions of each travel when 2 are reported
    mask_2travels = ibd_res_meta[rel_dest1].notnull()&ibd_res_meta[rel_dest2].notnull()
    ibd_res_meta.loc[:,prob_imp1] = pd.Series()
    ibd_res_meta.loc[mask_2travels, prob_imp1] = ibd_res_meta[rel_dest1]/(ibd_res_meta[rel_dest1] + \
                                                                              ibd_res_meta[rel_dest2] + \
                                                                              ibd_res_meta[rel_origin])
    ibd_res_meta.loc[:,prob_imp2] = pd.Series()
    ibd_res_meta.loc[mask_2travels, prob_imp2] = ibd_res_meta[rel_dest2]/(ibd_res_meta[rel_dest1] + \
                                                                              ibd_res_meta[rel_dest2] + \
                                                                              ibd_res_meta[rel_origin])
    
    total_cases = np.sum(ibd_res_meta[prob_imported].notnull())

    #Masking REACT2 cases with metadata
    mask_trav1 = ibd_res_meta[rel_dest1].notnull()&ibd_res_meta[rel_dest2].isnull()
    mask_trav2 = ibd_res_meta[rel_dest1].isnull()&ibd_res_meta[rel_dest2].notnull()
    mask_trav12 = mask_2travels
    #Defining list of all travel destinations
    dest_list_1 = ibd_res_meta['travel_prov'].unique().astype(str)
    dest_list_2 = ibd_res_meta['travel_prov2'].unique().astype(str)
    all_destinations = np.unique(np.concatenate((dest_list_1, dest_list_2)))
    #Summing imported cases
    imported = {}
    for dest in all_destinations:
        mask_dest1 = ibd_res_meta['travel_prov'] == dest
        mask_dest2 = ibd_res_meta['travel_prov2'] == dest
        imported[dest] = ibd_res_meta.loc[mask_trav12&mask_dest1, prob_imp1].sum()
        imported[dest] = imported[dest] + ibd_res_meta.loc[mask_trav12&mask_dest2, prob_imp2].sum()
        imported[dest] = imported[dest] + ibd_res_meta.loc[mask_trav1&mask_dest1, prob_imported].sum()
        imported[dest] = imported[dest] + ibd_res_meta.loc[mask_trav2&mask_dest2, prob_imported].sum()

    imported = pd.Series(imported)
    
    #Calculating imported stats
    imported_stats = pd.DataFrame()
    for i in all_destinations:
        imported_stats[i] = {'N. cases' : imported[i], \
                        '% cases' : imported[i]/total_cases*100}
    del imported_stats['nan']
    imported_stats['Total'] = {'N. cases' : imported_stats.loc['N. cases'].sum(), \
                              '% cases' : imported_stats.loc['% cases'].sum()}
    imported_stats = imported_stats.T
    return imported_stats
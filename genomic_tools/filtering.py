#This file contains methods for filtering genetic data. 
import numpy as np
import pandas as pd

def filter_loci_by_sample_coverage(data, min_samples=100, min_reads=100, verbose=True):
    """
    Filter out loci that don't have sufficient sample coverage.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns 'locus', 'sampleID', and 'reads'
    min_samples : int, default=100
        Minimum number of samples that must have >= min_reads at a locus
    min_reads : int, default=100
        Minimum number of reads per sample at a locus
    verbose : bool, default=True
        Print information about filtered loci
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with loci removed if they don't meet the criteria
    """
    filtered_data = data.copy()
    
    for locus in data['locus'].unique():
        # Mask data from the locus
        mask_locus = filtered_data['locus'] == locus
        
        # Count reads in locus per sample
        reads_per_sample = filtered_data.loc[mask_locus, ['sampleID', 'reads']].groupby('sampleID').sum()
        
        # Count how many samples have >= min_reads
        sample_high_cov = reads_per_sample >= min_reads
        total_samples_high_cov = sample_high_cov.sum()
        
        # Remove locus if < min_samples have >= min_reads
        if total_samples_high_cov.iloc[0] < min_samples:
            if verbose:
                print(f"{locus}: {total_samples_high_cov.iloc[0]} samples")
                print("this locus is being filtered from data")
            filtered_data = filtered_data[np.invert(mask_locus)]
    
    return filtered_data

def filter_samples_by_locus_coverage(data, all_samples, min_loci=123.75, min_reads=100, verbose=True):
    """
    Filter out samples that don't have sufficient locus coverage.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns 'sampleID', 'locus', 'reads', and optionally
        'run_id_resmark', 'run', 'study', 'source' for verbose output
    all_samples : list or array-like
        List of all sample IDs to check
    min_loci : float, default=123.75
        Minimum number of loci that must have >= min_reads per sample
    min_reads : int, default=100
        Minimum number of reads per locus for a sample
    verbose : bool, default=True
        Print information about filtered samples
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with samples removed if they don't meet the criteria
    """
    filtered_data = data.copy()
    
    for sample in all_samples:
        # Mask to select sample
        mask_sample = filtered_data['sampleID'] == sample
        
        # Count reads per locus
        locus_reads = filtered_data.loc[mask_sample, ['locus', 'reads']].groupby('locus').sum()
        
        # Count loci with >= min_reads
        n_good_loci = np.sum(locus_reads['reads'] >= min_reads)
        
        # Remove sample if it has fewer than min_loci covered
        if n_good_loci < min_loci:
            if verbose:
                print(f"Sample {sample} with only {n_good_loci} loci covered, removed")
            
            # Remove sample
            filtered_data = filtered_data[np.invert(mask_sample)]
    
    return filtered_data


#This module contains methods to analyse pairwise measurements such as IBD
#relatedness.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
import geopandas
from stat_tools import errors

def classify_ibd_per_label(category_label, ibd_res_meta):
    """
    This method generates a dictionary where all IBD results are classified
    accoring to the label of its pairs.

    Parameters:
    -----------
    category_label: str
        The name of the column of the IBD data with the classification label.
    ibd_res_meta: pd.DataFrame
        The data including the pairwise IBD results (NxN) and some extra columns
        at the end with metadata labels. The first N columns and rows must
        correspond to the pairwise comparison of samples (same order of samples
        in both rows and columns).

    Returns:
    --------
    ibd_per_cat: dict
        Dictionary where the element [i][j] contains all the IBD results of
        pairs with labels i and j.
    """
    #Create dictionary to store all the IBD pairs per regions
    categories = ibd_res_meta[category_label].unique()
    ibd_per_cat = {}
    for i in categories:
        ibd_per_cat[i] = {}

    #Assigning all IBD results for each pair of categories
    for i in categories:
        for j in categories:
            mask_i = ibd_res_meta[category_label] == i
            mask_j = np.zeros(len(ibd_res_meta.columns), dtype = bool)
            mask_j[:len(ibd_res_meta)] = ibd_res_meta[category_label] == j
            ibd_res_meta_ij = np.array(ibd_res_meta.loc[mask_i, mask_j])
            ibd_res_meta_ij = ibd_res_meta_ij[ibd_res_meta_ij>=0]
            ibd_per_cat[i][j] = ibd_res_meta_ij

    #Adding all pairs to the two compinations of different categories,
    #so that the order does not matter
    for ii, i in enumerate(categories):
        for j in categories[ii+1:]:
            if i != j:
                ibd_per_cat[i][j] = np.concatenate((ibd_per_cat[i][j], \
                                                    ibd_per_cat[j][i]))
                ibd_per_cat[j][i] = ibd_per_cat[i][j]
    return ibd_per_cat

def high_ibd_frac_per_cat(all_ibd_res, ibd_per_cat, all_p_res = None, \
                          p_per_cat = None, min_IBD = .0, max_p = .05, \
                          categories = None, verbose = True):
    """
    This method calculates the fraction of pairwise IBD results higher than a
    threshold with a minimum p-value for the comparisons in different
    categories.

    Parameters:
    -----------
    all_ibd_res: pd.DataFrame, np.ndarray
        NxN dataframe or matrix wih all the IBD results of the whole dataset.
    ibd_per_cat: dict
        Dictionary with all the IBD results for each par of categories
    all_p_res: pd.DataFrame, np.ndarray
        NxN dataframe or matrix wih all the p-values of the IBD results of the
        whole dataset.
    p_per_cat: dict
        Dictionary with all the p-values of the IBD results for each par of
        categories.
    min_IBD: float
        Minimum IBD from which to calculate the fraction
    max_p: float
        Maximum p-value from which to calculate the fraction
    categories: list
        List of values of the categories on which ibd_per_cat stores the
        results.
    verbose: bool
        Verbose mode.

    Returns:
    --------
    ibdfrac_per_cat: pd.DataFrame
        Data frame showing the fraction of IBD results higher than the threshold
        for each population of pairs with their correponding categories.
    ibdfrac_pval_per_cat; pd.DataFrame
        Data frame showing the p-value of the deviation of the high IBD fraction
        with respect to the average (the expected from random pairs)
    overall_high_ibd_frac: float
        Fraction of pairwise IBD above the threshold over all the samples
    """
    if categories is None:
        categories = list(ibd_per_cat.keys())
    ibd_mask = np.array(all_ibd_res)[np.array(all_ibd_res)>=0] >= min_IBD
    if all_p_res is None or p_per_cat is None:
        overall_high_ibd_frac = np.nanmean(ibd_mask)
    else:
        p_mask = np.array(all_p_res)[np.array(all_p_res)>=0] <= max_p
        overall_high_ibd_frac = np.nanmean(ibd_mask&p_mask)
    if verbose:
        print("Overall fraction of pairs with IBD >= "+str(min_IBD)+ \
              " and p<= " + str(max_p) + ": " + str(overall_high_ibd_frac))
    #Create dictionary to get fraction of high IBD pairs
    ibdfrac_per_cat = {}
    ibdfrac_pval_per_cat = {}
    for i in categories:
        ibdfrac_per_cat[i] = {}
        ibdfrac_pval_per_cat[i] = {}
        for j in categories:
            ibd_mask = np.array(ibd_per_cat[i][j]) >= min_IBD
            if all_p_res is None or p_per_cat is None:
                high_ibd_pairs = np.sum(ibd_mask)
            else:
                p_mask = np.array(p_per_cat[i][j]) <= max_p
                high_ibd_pairs = np.sum(ibd_mask&p_mask)
            high_ibd_fraction = high_ibd_pairs/len(ibd_per_cat[i][j])
            ibdfrac_per_cat[i][j] = high_ibd_fraction

            #P-value of deviating (from above or below) from average
            phigher =  1 - stats.binom.cdf(high_ibd_pairs, \
                                           len(ibd_per_cat[i][j]), \
                                           overall_high_ibd_frac)
            pval = 2*min(phigher, 1 - phigher)
            ibdfrac_pval_per_cat[i][j] = pval

    ibdfrac_per_cat = pd.DataFrame(ibdfrac_per_cat)
    ibdfrac_pval_per_cat = pd.DataFrame(ibdfrac_pval_per_cat)
    return ibdfrac_per_cat, ibdfrac_pval_per_cat, overall_high_ibd_frac

def show_ibd_frac_per_cat(ibdfrac_per_cat, overall_high_ibd_frac, \
                          ibdfrac_pval_per_cat = None, cmap = 'bwr', \
                          cmap_p = 'viridis', min_IBD = .0, max_p = .05):
    """
    This method visualises the results of the fraction of IBD above a threshold
    for different categories.

    Parameters:
    -----------
    ibdfrac_per_cat: np.DataFrame:
        Data frame showing the fraction of IBD results higher than the threshold
        for each population of pairs with their correponding categories.
    overall_high_ibd_frac: float
        Fraction of pairwise IBD above the threshold over all the samples
    ibdfrac_pval_per_cat; pd.DataFrame
        Data frame showing the p-value of the deviation of the high IBD fraction
        with respect to the average (the expected from random pairs).
    cmap: str
        Color map of the IBD fraction results.
    cmap_p: str
        Color map of the p-value results.
    min_IBD: float
        Minimum IBD from which the fraction was calculated
    max_p: float
        Maximum p-value from which the fraction was calculated

    Returns:
    --------
    plt.imshow plots with the results
    """
    #Renormalising colormaps to show symmetric colorbar with respect to the
    #average fraction
    max_deviation = np.max(np.abs(np.array(ibdfrac_per_cat).flatten() - \
                                  overall_high_ibd_frac))
    vmin = overall_high_ibd_frac - max_deviation
    vmax = overall_high_ibd_frac + max_deviation
    plt.imshow(np.array(ibdfrac_per_cat), vmin = vmin, vmax = vmax, \
               cmap = 'bwr')
    plt.yticks(np.arange(len(ibdfrac_per_cat)), ibdfrac_per_cat.index, \
               fontsize = 6)
    plt.xticks(np.arange(len(ibdfrac_per_cat)), ibdfrac_per_cat.index, \
               rotation = 90, fontsize = 6)
    plt.colorbar(label = "Fraction of pairs with IBD >= " + str(min_IBD) + \
                 " and p<= " + str(max_p))
    plt.show()

    if ibdfrac_pval_per_cat is not None:
        plt.imshow(np.array(ibdfrac_pval_per_cat), cmap = cmap_p)
        plt.yticks(np.arange(len(ibdfrac_pval_per_cat)), \
                   ibdfrac_pval_per_cat.index, fontsize = 6)
        plt.xticks(np.arange(len(ibdfrac_pval_per_cat)), \
                   ibdfrac_pval_per_cat.index, rotation = 90, fontsize = 6)
        plt.colorbar(label = 'P-value of deviation wrt average')
        plt.show()

def connectivity_map(ibdfrac_per_cat, categories, list_locs, \
                     xlims = [30, 42], ylims = [-28, -10], \
                     figsize = [6,9], color = 'tab:blue', linewidth = 'auto'):
    """
    This method generates a map with connectivities between locations.

    Parameters:
    -----------
    ibdfrac_per_cat: pd.DataFrame
        Matrix showing the connectivity (e.g.fraction of related pairs) between
        locations.
    categories: list
        List of location names.
    list_locs: dictionary
        dictionary describing the location of each category.
    xlims: list
        Limits of x-axis plot.
    ylims: list
        Limits of y-axis plot.
    figsize: list
        Size of figure.

    Returns:
    --------
    Map visualising the connectivity between categories.
    """
    Path = mpath.Path

    ax = provinces.plot(markersize = 0, alpha = 0, figsize = figsize)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, \
                    crs='EPSG:4326')
    zorder = 1
    for ii, i in enumerate(categories):
        l = 0
        for j in categories[ii+1:]:
            x = [list_locs[i][0], list_locs[j][0]]
            y = [list_locs[i][1], list_locs[j][1]]
            if linewidth == 'auto':
                lw = 15*((ibdfrac_per_cat.loc[i,j] - \
                     np.min(np.array(ibdfrac_per_cat)))/np.mean(np.array(ibdfrac_per_cat)))
            else:
                lw = linewidth
            deltax = x[1] - x[0]
            deltay = y[1] - y[0]
            xinter = [x[0] + .2*deltax, x[0] + .8*deltax]
            yinter = [y[0] + .8*deltay, y[0] + .2*deltay]
            if color == 'auto':
                max_deviation = np.max(np.abs(np.array(ibdfrac_per_cat).flatten() - \
                                              overall_high_ibd_frac))
                vmin = overall_high_ibd_frac - max_deviation
                vmax = overall_high_ibd_frac + max_deviation
                col = cm.turbo((ibdfrac_per_cat.loc[i,j] - vmin)/(vmax-vmin))
            else:
                col = color
            pp1 = mpatches.PathPatch(Path([(x[0], y[0]), (xinter[l%2], \
                                          yinter[l%2]), (x[1], y[1])], \
                                          [Path.MOVETO, Path.CURVE3, Path.CURVE3]), \
                                     fc="none", transform=ax.transData, \
                                     color = col, lw = lw, zorder = zorder, \
                                     alpha = .5)
            l+=1
            ax.add_patch(pp1)

            zorder += 1
        size = 120*((ibdfrac_per_cat.loc[i,i] - \
                    np.min(np.array(ibdfrac_per_cat)))/np.mean(np.array(ibdfrac_per_cat)))
        provinces[provinces['location'] == i].plot(ax = ax, markersize = size,
        color = 'k', zorder = zorder)
        zorder += 1
        ax.annotate(i, xy=np.array(list_locs[i]) + np.array([.2,0]))

def mean_high_ibd_frac_vs_dist(ibd_values, dist_values, p_values = None, \
                               min_IBD = .0, max_p = .05, nbins = 10, \
                               min_dist = None, max_dist = None, nrands = 100, \
                               show = True, label = ''):
    """
    This method calculates the fraction of IBD related pairs as a function of
    their geographical distance in distance bins.

    Parameters:
    -----------
    ibd_values: np.array or pd.Series
        Array of IBD values of the pairs.
    dist_values: np.array or pd.Series
        Array of the distance between the pairs.
    p_values: np.array or pd.Series
        Array of p-values of the IBD of the pairs.
    min_IBD: float
        Minimum IBD from which the fraction was calculated.
    max_p: float
        Maximum p-value from which the fraction was calculated.
    nbins: int
        Number of distance bins used.
    min_dist: float
        Minimum distance to include in the bins.
    max_dist: float
        Maximum distance to include in the bins.
    nrands: int
        Number of Bootstrap random resamples to calculate the error bars.
    show: bool
        If True, the plot is shown.
    label: str
        Label of the error bar.

    Returns:
    --------
    Error bar plot showing the fraction of related pairs in distance bins.
    """
    #Define high IBD pairs
    ibd_mask = np.array(ibd_values) >= min_IBD
    if p_values is not None:
        ibd_mask = ibd_mask&(np.array(p_values) <= max_p)
    #Define distance bins
    if min_dist is None:
        min_dist = np.min(dist_values)
    if max_dist is None:
        max_dist = np.max(dist_values)
    dist_bins = np.linspace(min_dist, max_dist, nbins)
    #get mean high IBD fraction per distance bin
    mean_high_ibd_frac = []
    err_high_ibd_frac = []
    mean_dist = []
    for i in range(len(dist_bins)-1):
        mask = (dist_values >= dist_bins[i])&(dist_values <= dist_bins[i+1])
        mean, err, mean_r = errors.boostrap_mean_err(ibd_mask[mask], \
                                                     nrands = nrands)
        mean_high_ibd_frac.append(mean)
        err_high_ibd_frac.append(err)
        mean_dist.append(np.mean(dist_values[mask]))

    plt.errorbar(mean_dist, mean_high_ibd_frac, err_high_ibd_frac, \
                 marker = '+', label = label)
    plt.ylabel("Fraction of high IBD pairs")
    plt.xlabel("Distance (km)")
    if show:
        plt.show()

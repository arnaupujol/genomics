#This module contains methods to analyse pairwise measurements such as IBD relatedness.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import contextily as ctx
import geopandas
from stat_tools import errors, glm
from matplotlib import cm

def classify_ibd_per_label(category_label, ibd_res_meta, category_label2 = None):
    """
    This method generates a dictionary where all IBD results are classified
    accoring to the label of its pairs.

    Parameters:
    -----------
    category_label: str
        The name of the column of the IBD data with the first (or unique)
        classification label.
    category_label2: str
        The name of the column of the IBD data with the second classification label.
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
    categories1 = ibd_res_meta[category_label].unique()
    if category_label2 is None:
        category_label2 = category_label
    categories2 = ibd_res_meta[category_label2].unique()
    ibd_per_cat = {}
    for i in categories1:
        ibd_per_cat[i] = {}

    #Assigning all IBD results for each pair of categories
    for i in categories1:
        for j in categories2:
            mask_i = ibd_res_meta[category_label] == i
            mask_j = np.zeros(len(ibd_res_meta.columns), dtype = bool)
            mask_j[:len(ibd_res_meta)] = ibd_res_meta[category_label2] == j
            ibd_res_meta_ij = np.array(ibd_res_meta.loc[mask_i, mask_j])
            ibd_res_meta_ij = ibd_res_meta_ij[ibd_res_meta_ij>=0]
            ibd_per_cat[i][j] = ibd_res_meta_ij
    #Adding all pairs to the two compinations of different categories,
    #so that the order does not matter when category_label == category_label2
    if category_label == category_label2:
        for ii, i in enumerate(categories1):
            for j in categories2[ii+1:]:
                if i != j:
                    ibd_per_cat[i][j] = np.concatenate((ibd_per_cat[i][j], \
                                                        ibd_per_cat[j][i]))
                    ibd_per_cat[j][i] = ibd_per_cat[i][j]
    else:
        for i in categories2:
            for j in categories1:
                mask_i = ibd_res_meta[category_label2] == i
                mask_j = np.zeros(len(ibd_res_meta.columns), dtype = bool)
                mask_j[:len(ibd_res_meta)] = ibd_res_meta[category_label] == j
                ibd_res_meta_ij = np.array(ibd_res_meta.loc[mask_i, mask_j])
                ibd_res_meta_ij = ibd_res_meta_ij[ibd_res_meta_ij>=0]
                ibd_per_cat[j][i] = np.concatenate((ibd_per_cat[j][i], ibd_res_meta_ij))
    return ibd_per_cat


def high_ibd_frac_per_cat(all_ibd_res, ibd_res_meta, category_label, category_label2, all_p_res = None, \
                          ibd_pval_meta = None, min_IBD = .0, max_p = .05, categories = None, \
                          categories2 = None, verbose = True, perm_pval = True, nrands = 100):
    """
    This method calculates the fraction of pairwise IBD results higher than a
    threshold with a minimum p-value for the comparisons in different
    categories.

    Parameters:
    -----------
    all_ibd_res: pd.DataFrame, np.ndarray
        NxN dataframe or matrix wih all the IBD results of the whole dataset.
    ibd_res_meta: pd.DataFrame
        The data including the pairwise IBD results (NxN) and some extra columns
        at the end with metadata labels. The first N columns and rows must
        correspond to the pairwise comparison of samples (same order of samples
        in both rows and columns).
    category_label: str
        The name of the column of the IBD data with the first (or unique)
        classification label.
    category_label2: str
        The name of the column of the IBD data with the second classification label.
    all_p_res: pd.DataFrame, np.ndarray
        NxN dataframe or matrix wih all the p-values of the IBD results of the
        whole dataset.
    ibd_pval_meta: pd.DataFrame
        The data including the pairwise IBD p-values (NxN) and some extra columns
        at the end with metadata labels. The first N columns and rows must
        correspond to the pairwise comparison of samples (same order of samples
        in both rows and columns).
    min_IBD: float
        Minimum IBD from which to calculate the fraction.
    max_p: float
        Maximum p-value from which to calculate the fraction.
    categories: list
        List of values of the categories on which ibd_per_cat stores the
        results.
    categories2: list
        List of values of the categories on which ibd_per_cat stores the
        results from the second category label (if any).
    verbose: bool
        Verbose mode.
    perm_pval: bool
        If True, it obtains the p-value from sample permutations. If False, it assumes
        a binomial distribution of values. 
    nrands: int
        Number of random permutations used to calculate the p-values. 

    Returns:
    --------
    ibdfrac_per_cat: pd.DataFrame
        Data frame showing the fraction of IBD results higher than the threshold
        for each population of pairs with their correponding categories.
    ibdfrac_pval_per_cat: pd.DataFrame
        Data frame showing the p-value of the deviation of the high IBD fraction
        with respect to the average (the expected from random pairs).
    overall_high_ibd_frac: float
        Fraction of pairwise IBD above the threshold over all the samples
    """
    if category_label2 is None:
        category_label2 = category_label
    ibd_per_cat = classify_ibd_per_label(category_label, ibd_res_meta, category_label2)
    if all_p_res is None:
        p_per_cat = None
    else:
        p_per_cat = classify_ibd_per_label(category_label, ibd_pval_meta, category_label2)
    
    if categories is None:
        categories = list(ibd_per_cat.keys())
    if categories2 is None:
        categories2 = list(ibd_per_cat[categories[0]].keys())
    
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
        for j in categories2:
            ibd_mask = np.array(ibd_per_cat[i][j]) >= min_IBD
            if all_p_res is None or p_per_cat is None:
                high_ibd_pairs = np.sum(ibd_mask)
            else:
                p_mask = np.array(p_per_cat[i][j]) <= max_p
                high_ibd_pairs = np.sum(ibd_mask&p_mask)
            high_ibd_fraction = high_ibd_pairs/len(ibd_per_cat[i][j])
            ibdfrac_per_cat[i][j] = high_ibd_fraction

            #P-value of deviating (above or below) from average assuming binomial distributions
            phigher =  1 - stats.binom.cdf(high_ibd_pairs, \
                                           len(ibd_per_cat[i][j]), \
                                           overall_high_ibd_frac)
            pval = 2*min(phigher, 1 - phigher)
            ibdfrac_pval_per_cat[i][j] = pval
    ibdfrac_per_cat = pd.DataFrame(ibdfrac_per_cat)
    if perm_pval:
        ibdfrac_per_cat_r = get_all_ibdfrac_perms(category_label, all_ibd_res, ibd_res_meta, all_p_res, \
                                                  ibd_pval_meta, min_IBD, max_p, \
                                                  categories, category_label2, categories2, nrands)#TODO TEST
        ibdfrac_pval_per_cat = get_pval_from_permutations(ibdfrac_per_cat, ibdfrac_per_cat_r)
    else:
        ibdfrac_pval_per_cat = pd.DataFrame(ibdfrac_pval_per_cat)
    return ibdfrac_per_cat, ibdfrac_pval_per_cat, overall_high_ibd_frac

def show_ibd_frac_per_cat(ibdfrac_per_cat, overall_high_ibd_frac, \
                          ibdfrac_pval_per_cat = None, cmap = 'bwr', \
                          cmap_p = 'viridis', min_IBD = .0, max_p = .05, \
                            fontsize = 12, xticks = None, xrotation = 45):
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
    fontsize: int
        Font size of labels in plot.
    xticks: list
        The list of names associated to xticks
    xrotation: float
        The rotation angle of the xticks

    Returns:
    --------
    plt.imshow plots with the results
    """
    #Renormalising colormaps to show symmetric colorbar with respect to the
    #average fraction
    max_deviation = np.nanmax(np.abs(np.array(ibdfrac_per_cat).flatten() - \
                                  overall_high_ibd_frac))
    vmin = overall_high_ibd_frac - max_deviation
    vmax = overall_high_ibd_frac + max_deviation
    plt.imshow(np.array(ibdfrac_per_cat), vmin = vmin, vmax = vmax, \
               cmap = 'bwr')
    if xticks is None: 
        xnames = ibdfrac_per_cat.columns
    else: 
        xnames = xticks
    plt.xticks(np.arange(ibdfrac_per_cat.shape[1]), xnames, \
               rotation = xrotation, fontsize = fontsize)
    plt.yticks(np.arange(ibdfrac_per_cat.shape[0]), ibdfrac_per_cat.index, \
               fontsize = fontsize)
    plt.colorbar().set_label(label = "Fraction of pairwise IBD >= " + str(min_IBD) + \
                 " and p<= " + str(max_p), size = fontsize)
    plt.show()

    if ibdfrac_pval_per_cat is not None:
        plt.imshow(np.array(ibdfrac_pval_per_cat), cmap = cmap_p, vmax = .5)
        plt.xticks(np.arange(ibdfrac_pval_per_cat.shape[1]), xnames, \
               rotation = xrotation, fontsize = fontsize)
        plt.yticks(np.arange(ibdfrac_pval_per_cat.shape[0]), ibdfrac_pval_per_cat.index, \
                   fontsize = fontsize)
        plt.colorbar().set_label(label = 'P-value of deviation wrt average', size = fontsize)
        plt.show()

def connectivity_map(ibdfrac_per_cat, categories, locations, \
                     xlims = [30, 42], ylims = [-28, -10], \
                     figsize = [6,9], color = 'tab:blue', linewidth = 'auto', \
                     print_locations = True):
    """
    This method generates a map with connectivities between locations.

    Parameters:
    -----------
    ibdfrac_per_cat: pd.DataFrame
        Matrix showing the connectivity (e.g.fraction of related pairs) between
        locations.
    categories: list
        List of location names.
    locations: geopandas.GeoDataFrame
        Geopandas dataframe describing the locations of each category.
    xlims: list
        Limits of x-axis plot.
    ylims: list
        Limits of y-axis plot.
    figsize: list
        Size of figure.
    color: str
        Colour of connectivity lines. If 'auto', the colour encodes the fraction 
        of IBD related pairs scaled to the range of values. If 'prop', the colour
        encodes the IBD fraction proportionally. 
    linewidth: 'auto' or int
        If 'auto', the line width is rescaled with respect to the 
        minimum and average values. 
    print_locations: bool
        If True, the location names are annotated in the map.
    

    Returns:
    --------
    Map visualising the connectivity between categories.
    """
    Path = mpath.Path

    ax = locations.plot(markersize = 0, alpha = 0, figsize = figsize)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, \
                    crs='EPSG:4326')

    #Define x and y positions of locations
    locations['x'] = locations['geometry'].x
    locations['y'] = locations['geometry'].y
    #Define dictionary of locations and positions
    list_locs = {}
    for i, l in enumerate(locations['location']):
        list_locs[l] = [locations.loc[i]['x'], locations.loc[i]['y']]
    zorder = 1
    for ii, i in enumerate(categories):
        l = 0
        for j in categories[ii+1:]:
            x = [list_locs[i][0], list_locs[j][0]]
            y = [list_locs[i][1], list_locs[j][1]]
            if linewidth == 'auto':
                lw = 15*((ibdfrac_per_cat.loc[i,j] - \
                     np.min(np.array(ibdfrac_per_cat)))/np.mean(np.array(ibdfrac_per_cat)))
            elif linewidth == 'prop':
                lw = 5*ibdfrac_per_cat.loc[i,j]/np.mean(np.array(ibdfrac_per_cat))
            else:
                lw = linewidth
            deltax = x[1] - x[0]
            deltay = y[1] - y[0]
            xinter = [x[0] + .2*deltax, x[0] + .8*deltax]
            yinter = [y[0] + .8*deltay, y[0] + .2*deltay]
            if color == 'auto':
                max_deviation = np.nanmax(np.abs(np.array(ibdfrac_per_cat).flatten() - \
                                              np.mean(np.array(ibdfrac_per_cat))))
                vmin = np.mean(np.array(ibdfrac_per_cat)) - max_deviation
                vmax = np.mean(np.array(ibdfrac_per_cat)) + max_deviation
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
        if linewidth == 'auto':
            size = 120*((ibdfrac_per_cat.loc[i,i] - \
                    np.min(np.array(ibdfrac_per_cat)))/np.mean(np.array(ibdfrac_per_cat)))
        elif linewidth == 'prop':
            size = 40*ibdfrac_per_cat.loc[i,i]/np.mean(np.array(ibdfrac_per_cat))
        else:
            size = 40
        if color == 'auto':
            col = cm.turbo((ibdfrac_per_cat.loc[i,i] - vmin)/(vmax-vmin))
        else:
            col = 'k'
        locations[locations['location'] == i].plot(ax = ax, \
                                                   markersize = size,
                                                   color = col, zorder = zorder)
        zorder += 1
        ax.annotate(i, xy=np.array(list_locs[i]) + np.array([.2,0]))
        
def mean_high_ibd_frac_vs_dist(ibd_values, dist_values, p_values = None, \
                               min_IBD = .0, max_p = .05, nbins = 10, \
                               min_dist = None, max_dist = None, nrands = 100, \
                               show = True, label = '', get_glm = False, \
                               c = None, c2 = None, verbose = False, lw = 2):
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
    get_glm: bool
        If true, a binomial regression is done and added to the plot.
    c: str or RGB value
        The colour of the plot lines.
    c2: str or RGB value
        The colour of the regression lines. If None, c is assigned
    verbose: bool
        It specifies if information is printed out.
    lw: int
        Line width of plot lines.

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
        min_dist = np.nanmin(dist_values)
    if max_dist is None:
        max_dist = np.nanmax(dist_values)
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
                 marker = '+', label = label, color = c, lw = lw)

    if get_glm:
        dist_mask = (dist_values >= min_dist)&(dist_values <= max_dist)
        if c2 is None: 
            c2 = c
        glm.regression(dist_values[dist_mask], ibd_mask[dist_mask], \
                   family = 'binomial', verbose = verbose, show = True, c = c2, \
                   ls = '--', lw = lw)
    plt.ylabel("Fraction of high IBD pairs")
    plt.xlabel("Distance (km)")
    if show:
        plt.show()

def get_label_permutation(dataframe, label, label2 = None):
    """
    This method creates a label permutation in a dataframe. 
    
    Parameters:
    -----------
    dataframe: pd.DataFrame
        Dataframe with label.
    label: str
        Name of the label to permute.
    label2: str
        Name of the 2nd label to permute.
    
    Returns:
    --------
    dataframe: pd.DataFrame
        Dataframe with label and perm_label (the corresponding 
        permutation). 
    """
    indeces = np.array(dataframe.index)
    np.random.shuffle(indeces)
    dataframe['perm_label'] = np.array(dataframe[label].loc[indeces])
    if label2 is not None:
        dataframe['perm_label2'] = np.array(dataframe[label2].loc[indeces])
    return dataframe

def ibd_pval_label_permutation(ibd_res_meta, pval_data, label, label2 = None):
    """
    This method creates a label permutation in ibd and pval 
    data frames. 
    
    Parameters:
    -----------
    ibd_res_meta: pd.DataFrame
        Data frame containing IBD results.
    pval_data: pd.DataFrame
        Data frame containing p-value results.
    label: str
        Name of the label to permute.
    label2: str
        Name of the 2nd label to permute.
    
    Returns:
    --------
    ibd_data: pd.DataFrame
        Data frame containing IBD results, with perm_label column.
    pval_data: pd.DataFrame
        Data frame containing p-value results, with perm_label column.
    """
    ibd_data = get_label_permutation(ibd_res_meta, label, label2)
    pval_data['perm_label'] = ibd_data['perm_label']
    if label2 is not None:
        pval_data['perm_label2'] = ibd_data['perm_label2']
    return ibd_data, pval_data

def get_all_ibdfrac_perms(category_label, ibd_res, ibd_res_meta, ibd_pval, \
                          ibd_pval_meta, min_IBD = .0, max_p = .05, \
                          categories = None, category_label2=None, categories2 = None, nrands = 100):
    """
    This method generates several runs calculating the fraction of related pairs by applying 
    permutations of the samples over the categories selected for the sampling. 
    
    Parameters:
    -----------
    category_label: str
        The name of the column of the IBD data with the first (or unique)
        classification label.
    ibd_res: pd.DataFrame, np.ndarray
        NxN dataframe or matrix wih all the IBD results of the whole dataset.
    ibd_res_meta: pd.DataFrame
        The data including the pairwise IBD p-values (NxN) and some extra columns
        at the end with metadata labels. The first N columns and rows must
        correspond to the pairwise comparison of samples (same order of samples
        in both rows and columns).
    ibd_pval: pd.DataFrame, np.ndarray
        NxN dataframe or matrix wih all the p-values of the IBD results of the
        whole dataset.
    ibd_pval_meta: pd.DataFrame
        The data including the pairwise IBD p-values (NxN) and some extra columns
        at the end with metadata labels. The first N columns and rows must
        correspond to the pairwise comparison of samples (same order of samples
        in both rows and columns).
    min_IBD: float
        Minimum IBD from which to calculate the fraction.
    max_p: float
        Maximum p-value from which to calculate the fraction.
    categories: list
        List of values of the categories on which ibd_per_cat stores the
        results.
    category_label2: str
        The name of the column of the IBD data with the second classification label.
    categories2: list
        List of values of the categories on which ibd_per_cat stores the
        results from the second category label (if any).
    nrands: int
        Number of random permutations used to calculate the IBD-related fraction of pairs.
        
    Returns:
    --------
    ibdfrac_per_cat_r: pd.DataFrame
        Data frame showing the fraction of IBD results higher than the threshold for all
        the permutations for each population of pairs within their correponding categories.
    """
    ibdfrac_per_cats = []
    for r in range(nrands):
        #Get random permutations of the category label
        ibd_res_meta, ibd_pval_meta = ibd_pval_label_permutation(ibd_res_meta, ibd_pval_meta, \
                                                                 category_label, category_label2)
        #Calculate IBD fraction per category from permutation label
        perm_category_label = 'perm_label'
        if category_label2 is None:
            perm_category_label2 = None
        else:
            perm_category_label2 = 'perm_label2' 
        ibdfrac_per_cat, ibdfrac_binomial_pval_per_cat, \
        overall_high_ibd_frac = high_ibd_frac_per_cat(ibd_res, ibd_res_meta, perm_category_label, perm_category_label2, \
                                                      ibd_pval, ibd_pval_meta, min_IBD, max_p, \
                                                      categories, categories2, verbose = False, perm_pval = False)
        #add results from new random run
        ibdfrac_per_cats.append(ibdfrac_per_cat)
    
    #Store results as arrays per categories
    ibdfrac_per_cat_r = {}
    
    for i in list(ibdfrac_per_cats[0].columns):
        ibdfrac_per_cat_r[i] = {}
        for j in list(ibdfrac_per_cats[0].index):
            ibdfrac_per_cat_r[i][j] = [ibdfrac_per_cats[ii][i][j] for ii in range(len(ibdfrac_per_cats))]
    ibdfrac_per_cat_r = pd.DataFrame(ibdfrac_per_cat_r)

    return ibdfrac_per_cat_r

def get_pval_from_permutations(ibdfrac_per_cat, ibdfrac_per_cat_r):
    """
    This method calculated the p-value corresponding to a null-hypothesis test under the 
    assumption that the fraction of related pairs is independent of their categories. 
    
    Parameters:
    -----------
    ibdfrac_per_cat: pd.DataFrame
        Data frame showing the fraction of IBD results higher than the threshold
        for each population of pairs with their correponding categories.
    ibdfrac_per_cat_r: pd.DataFrame
        Data frame showing the fraction of IBD results higher than the threshold for all
        the permutations for each population of pairs within their correponding categories.
    
    Returns:
    --------
    ibdfrac_per_cat_pval: pd.DataFrame
        Data frame showing the p-value of the deviation of the high IBD fraction
        with respect to the average (the expected from random pairs) for the different 
        pairs of categories.
    """
    ibdfrac_per_cat_pval = {}
    for i in list(ibdfrac_per_cat.columns):
        ibdfrac_per_cat_pval[i] = {}
        for j in list(ibdfrac_per_cat.index):
            phigher = np.mean(ibdfrac_per_cat[i][j] >= ibdfrac_per_cat_r[i][j])
            pval = 2*min(phigher, 1 - phigher)
            ibdfrac_per_cat_pval[i][j] = pval
    ibdfrac_per_cat_pval = pd.DataFrame(ibdfrac_per_cat_pval)
    return ibdfrac_per_cat_pval

def travel_map(travel_matrix, origins, destinies, locations, \
                     xlims = [30, 42], ylims = [-28, -10], \
                     figsize = [6,9], color = 'tab:blue', linewidth = 'auto', \
                    categories2 = None, alpha = 0.5, print_locations = True):
    """
    This method generates a map visualising travels between locations.

    Parameters:
    -----------
    travel_matrix: pd.DataFrame
        Matrix showing the connectivity (e.g.number of travels) between
        locations.
    origins: list
        List of travel origins.
    destinies: list
        List of travel destinies.
    locations: geopandas.GeoDataFrame
        Geopandas dataframe describing the locations of each category.
    xlims: list
        Limits of x-axis plot.
    ylims: list
        Limits of y-axis plot.
    figsize: list
        Size of figure.
    color: str
        Colour of connectivity lines. If 'auto', the colour encodes the fraction 
        of IBD related pairs scaled to the range of values. If 'prop', the colour
        encodes the IBD fraction proportionally. 
    linewidth: 'auto' or int
        If 'auto', the line width is rescaled with number of travels. 
    alpha: float
        Transparency of lines.
    print_locations: bool
        If True, the names of the locations are printed in the map. 

    Returns:
    --------
    Map visualising the connectivity between categories.
    """
    Path = mpath.Path

    ax = locations.plot(markersize = 0, alpha = 0, figsize = figsize)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, \
                    crs='EPSG:4326')

    #Define x and y positions of locations
    locations['x'] = locations['geometry'].x
    locations['y'] = locations['geometry'].y
    #Define dictionary of locations and positions
    list_locs = {}
    for i, l in enumerate(locations['location']):
        list_locs[l] = [locations.loc[i]['x'], locations.loc[i]['y']]
    zorder = 1
    for ii, i in enumerate(origins):
        l = 0
        for j in destinies:
            x = [list_locs[i][0], list_locs[j][0]]
            y = [list_locs[i][1], list_locs[j][1]]
            if linewidth in ['auto', 'prop', 'log']:
                if travel_matrix.loc[i,j] == 0:
                    lw = 0
                else:
                    if linewidth == 'log':
                        lw = 10*(np.log(float(travel_matrix.loc[i,j]))/np.log(np.max(np.array(travel_matrix))))
                    else:
                        lw = 10*(float(travel_matrix.loc[i,j])/np.max(np.array(travel_matrix)))
            else:
                lw = linewidth
            deltax = x[1] - x[0]
            deltay = y[1] - y[0]
            xinter = [x[0] + .2*deltax, x[0] + .8*deltax]
            yinter = [y[0] + .8*deltay, y[0] + .2*deltay]
            if color == 'auto':
                col = cm.copper(travel_matrix.loc[i,j]/np.max(np.array(travel_matrix)))
            else:
                col = color
            pp1 = mpatches.PathPatch(Path([(x[0], y[0]), (xinter[l%2], \
                                          yinter[l%2]), (x[1], y[1])], \
                                          [Path.MOVETO, Path.CURVE3, Path.CURVE3]), \
                                     fc="none", transform=ax.transData, \
                                     color = col, lw = lw, zorder = zorder, \
                                     alpha = alpha)
            l=np.random.randint(10)
            ax.add_patch(pp1)

            zorder += 1
        
        if linewidth in ['auto', 'prop', 'log']:
            size = 60*travel_matrix.T.sum()[i]/travel_matrix.T.sum().sum()
        else:
            size = 60
        if color == 'auto':
            if travel_matrix.loc[i,j] == 0:
                col = cm.copper(0)
            else:
                col = cm.copper(travel_matrix.T.sum()[i]/travel_matrix.T.sum().sum())
        else:
            col = 'k'
        locations[locations['location'] == i].plot(ax = ax, \
                                                   markersize = size,
                                                   color = col, zorder = zorder)
        zorder += 1
    if print_locations:
        for ii, i in enumerate(locations['location']):
            ax.annotate(i, xy=np.array(list_locs[i]) + np.array([.2,0]))
   

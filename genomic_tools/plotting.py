#This module contains functions to make plots.

import numpy as np
import matplotlib.pyplot as plt
from genomic_tools.barcodes import bar_diff, barcode_list, base_is
from genomic_tools.stats import count_cases

def plot_barcode(bar, base2color, y = 0, ysize = 4, xlim = None, s = 70):
    """
    This method shows a colour-coded visualisation of the barcode.

    Parameters:
    -----------
    bar: list
        Barcode
    base2color: dict
        A dictionary specifying the colors for each key
    ysize: float
        It defines the y size of the figure
    xlim: list of length 2
        It defines the x limits of the plot
    s: float
        size of the marker

    Returns:
    --------
    Plot with a line of colours corresponding to the bases
    """
    if y == 0:
        plt.figure(figsize=[20,ysize])
    x = np.arange(len(bar))
    y = x*0 + y
    colors = [base2color[i] for i in bar]
    plt.scatter(x, y, c = colors, s = s, marker = 'o')
    if xlim is None:
        plt.xlim(-1, len(bar))
    else:
        plt.xlim(xlim)
    plt.xlabel('SNP')

def plot_barcodes(bars, base2color, xlim = None, s = 70):
    """
    This method shows a list of colour-coded visualisations of barcodes.

    Parameters:
    -----------
    bars: list
        List of barcodes
    base2color: dict
        A dictionary specifying the colors for each key
    xlim: list of length 2
        It defines the x limits of the plot
    s: float
        size of the marker

    Returns:
    --------
    Plot with lines of colours corresponding to the bases
    """
    for y in range(len(bars)):
        plot_barcode(bars[y], base2color, y = y, ysize = 4./15*len(bars), xlim = xlim, s = s)
    plt.ylim(len(bars),-1)
    plt.ylabel('Sample number')

def plot_legend(base2color):
    """This method shows the colour coding of the bases.

    Parameters:
    -----------
    base2color: dict
        A dictionary specifying the colors for each key

    Returns:
    --------
    Plot with the legend describing the colors for each key
    """
    y = 0
    x = 0
    for i in base2color:
        plt.scatter(x, y, c = base2color[i], label = i, s = 50)
        y -= 1
    plt.legend()

def plot_bar_diff(bar1, bars, base2color, keep_N = False, keep_X = False):
    """This method colour codes the barcodes masking in black the
    coinciding bases with the reference bar1.

    Parameters:
    -----------
    bar1: list
        Reference barcode
    bars: list
        List of barcodes to compare with
    base2color: dict
        A dictionary specifying the colors for each key
    keep_N: bool
        If True, N cases are left represented regardless of the reference
    keep_X: bool
        If True, X cases are left represented regardless of the reference

    Returns:
    --------
    Plot with lines of colours corresponding to the bases
    """
    plot_barcode(bar1, base2color, y = 0, ysize = 4./15*(len(bars)+1))
    for y in range(len(bars)):
        mask = bar_diff(bar1, bars[y]) #TODO specify if I keep or not X and N
        bar_plot = barcode_list(bars[y])
        if np.sum(mask) >= 1:
            bar_plot[mask] = np.array(['m']*np.sum(mask))
            if keep_N:
                n_cases = base_is(bars[y], 'N')
                bar_plot[n_cases] = np.array(['N']*np.sum(n_cases))
            if keep_X:
                x_cases = base_is(bars[y], 'X')
                bar_plot[x_cases] = np.array(['X']*np.sum(x_cases))
        plot_barcode(bar_plot, base2color, y = y + 1, ysize = 4./15*(len(bars)+1))
    plt.ylim(len(bars)+1,-1)
    plt.ylabel('Sample number')
    plt.xlabel('SNP')

def plot_FS(bar, c = 'k', title = ''):
    """This method visualizes the scatter distributions between
    the first 4 PCs with a colour code.

    Parameters:
    -----------
    bar: np.ndarray
        Set of barcodes
    c: str
        String or array to specify colorbar
    title: str
        title of the plot if any

    Returns:
    --------
    6 subplots comparing the principal components
    """
    fig = plt.figure(figsize=[17,10])
    plt.subplot(231)
    plt.scatter(bar[:,0], bar[:,1], c = c)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')

    plt.subplot(232)
    plt.scatter(bar[:,0], bar[:,2], c = c)
    plt.xlabel('PC 1')
    plt.ylabel('PC 3')

    plt.subplot(233)
    plt.scatter(bar[:,0], bar[:,3], c = c)
    plt.xlabel('PC 1')
    plt.ylabel('PC 4')

    plt.subplot(234)
    plt.scatter(bar[:,1], bar[:,2], c = c)
    plt.xlabel('PC 2')
    plt.ylabel('PC 3')

    plt.subplot(235)
    plt.scatter(bar[:,1], bar[:,3], c = c)
    plt.xlabel('PC 2')
    plt.ylabel('PC 4')

    plt.subplot(236)
    plt.scatter(bar[:,2], bar[:,3], c = c)
    plt.xlabel('PC 3')
    plt.ylabel('PC 4')

    fig.suptitle(title)
    plt.show()

def make_pie(variable, verbose = False, ignore_null = True):
    """
    This method shows a Pie chart with the percentage of
    appearence of a given variable in the data.

    Parameters:
    -----------
    variable: (pd.Series, np.array)
        List of values or labels in the variable
    verbose: bool, default is False
        If True, it prints the results as text
    ignore_null: bool, default is True
        If True, is does not include the null cases

    Returns:
    --------
    Plot showing the proportion of appearence of each value
    """
    cases, counts = count_cases(variable, ignore_null = ignore_null)
    plt.figure(figsize=[6,6])
    plt.pie(counts, labels = cases, autopct='%1.1f%%')
    plt.show()
    if verbose:
        print("Number and fraction of cases:")
        max_len_str = max([len(i) for i in cases])
        for i in range(len(cases)):
            print(str(cases[i]) + ": " + ' '*(max_len_str - len(cases[i])) + \
            str(counts[i]) + '\t'+ \
            str(round(counts[i]*100./sum(counts), 1)) + "%")

def make_barplot(variable, verbose = False, ignore_null = True):
    """
    This method shows a bar plot with the number of
    appearences of a given variable in the data.

    Parameters:
    -----------
    variable: (pd.Series, np.array)
        List of values or labels in the variable
    verbose: bool, default is False
        If True, it prints the results as text
    ignore_null: bool, default is True
        If True, is does not include the null cases

    Returns:
    --------
    Plot showing the number of appearences of each value
    """
    cases, counts = count_cases(variable, ignore_null = ignore_null)
    plt.bar(cases, counts)
    plt.show()

def make_hist(variable, nbins = 50, range = None, title = "", xlabel = "", ylabel = "Frequency", yscale = "linear", show = True):
    """This method makes a simple histogram of a variable.

    Parameters:
    -----------
    variable: list or np.array
        Array of values
    nbins: int
        Number of bins of the histogram
    title: str
        Title of the histogram
    xlabel: str
        Label in x-axis
    ylabel: str
        Label in y-axis
    yscale: str {'log', 'linear'}
        Scale of y-axis
    show: bool
        if True, it shows the plot

    Returns:
    --------
    Plot with the histogram
    """
    plt.hist(variable[variable.notnull()], nbins, range = range)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    if show:
        plt.show()

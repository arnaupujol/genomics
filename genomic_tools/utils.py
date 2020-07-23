#This module defines a diversity of utility functions.

def dict_translate(var, dic):
    """This method translates an array of values to
    a correspondig value from a dictionary.

    Parameters:
    -----------
    var: np.array
        Array of values
    dic: dict
        Dictionary

    Returns:
    --------
    vals: list
        Values of var corresponding to the dictionary
    """
    vals = [dic[i] for i in var]
    return vals

def dates2days(dates):
    """
    This method outputs the number of days after the earliest date from the data.

    Parameters:
    -----------
    dates: data frame array
        Array of dates

    Returns:
    --------
    days: np.array
        Array corresponding to the days after the earliest of them
    """
    date_min = dates.min()
    diff_dates = dates - date_min
    days = np.array([i.days for i in diff_dates])
    return days

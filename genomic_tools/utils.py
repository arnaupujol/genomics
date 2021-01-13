#This module defines a diversity of utility functions.
import numpy as np

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

@np.vectorize
def control_digit(nida_num):
    """This method outputs the control digits of nida numbers.

    Parameters:
    -----------
    nida_num: int or float
        An integer number with 7 digits. If float, the control
        digit will be ignored

    Returns:
    --------
    control: int
        The control digit of the number
    """
    nida_int = int(nida_num)
    nida_s = str(nida_int)
    x1 = 3*(int(nida_s[6]) + int(nida_s[4]) + int(nida_s[2]) + int(nida_s[0]))
    x2 = int(nida_s[5]) + int(nida_s[3]) + int(nida_s[1])
    x = x1 + x2
    control = np.mod(10 - np.mod(x,10),10)
    return control

@np.vectorize
def validate_nida(nida_num):
    """This method validates the control digits of nida numbers.

    Parameters:
    -----------
    nida_num: float
        An float with 7 digits and 1 decimal

    Returns:
    --------
    valid: bool
        It specifies whether the nida control digit is valid
    cdigit: int
        The control digit in the nida number
    control: int
        The control number the nida should have
    """
    control = control_digit(nida_num)
    cdigit = int(str(nida_num)[-1])
    valid = control == cdigit
    return valid, cdigit, control

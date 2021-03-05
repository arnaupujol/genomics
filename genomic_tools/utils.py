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

#Defining region masks
def get_area_masks(cross210, mipmon, rrs, opd_2to9, cross_areas, mipmon_areas, clinic_areas):
    """This method returns the mask of the corresponding areas for cross, mipmon and clinic data.

    Parameters:
    -----------
    cross210: pd.DataFrame
        Dataframe of cross-sectionals
    mipmon: pd.DataFrame
        Dataframe of MiPMon samples
    rrs: pd.DataFrame
        Dataframe of RRS samples
    opd_2to9: pd.DataFrame
        Dataframe of OPD samples
    cross_areas: list
        List of area names for cross-sectionals
    mipmon_areas: list
        List of area names for MiPMon samples
    clinic_areas: list
        List of area names for clinical cases

    Returns:
    cross_areas_mask: np.array
        Boolean mask of cross-sectionals
    mipmon_areas_mask: np.array
        Boolean mask of MiPMon
    clinic_areas_mask: np.array
        Boolean mask of clinical cases
    """
    #Cross mask
    cross_areas_mask = get_cross_area_mask(cross210, cross_areas)

    #MiPMon mask
    mipmon_areas_mask = get_mipmon_area_mask(mipmon, mipmon_areas)

    #Clinics mask
    clinic_areas_mask = get_clinic_area_mask(rrs, opd_2to9, clinic_areas)
    return cross_areas_mask, mipmon_areas_mask, clinic_areas_mask

def get_cross_area_mask(cross210, cross_areas):
    """This method returns the mask of the corresponding areas for cross-sectional data.

    Parameters:
    -----------
    cross210: pd.DataFrame
        Dataframe of cross-sectionals
    cross_areas: list

    Returns:
    cross_areas_mask: np.array
        Boolean mask of cross-sectionals
    """
    #Cross mask
    cross_areas_mask = cross210['area'].isnull()&cross210['area'].notnull()
    for a in cross_areas:
        cross_areas_mask = cross_areas_mask | (cross210['area']==a)
        if a == 'Panjane':
            cross_areas_mask = cross_areas_mask&(cross210['lat'] < -25)
    return cross_areas_mask

def get_mipmon_area_mask(mipmon, mipmon_areas):
    """This method returns the mask of the corresponding areas for MiPMon data.

    Parameters:
    -----------
    mipmon: pd.DataFrame
        Dataframe of MiPMon samples
    mipmon_areas: list
        List of area names for MiPMon samples

    Returns:
    mipmon_areas_mask: np.array
    """
    #MiPMon mask
    mipmon_areas_mask = mipmon['posto_code'].isnull()&mipmon['posto_code'].notnull()
    for a in mipmon_areas:
        mipmon_areas_mask = mipmon_areas_mask | (mipmon['posto_code'] == a)
        if a == 'Pandjane':
            mipmon_areas_mask = mipmon_areas_mask&(mipmon['latitude'] < -25)
    return mipmon_areas_mask

def get_clinic_area_mask(rrs, opd_2to9, clinic_areas):
    """This method returns the mask of the corresponding areas for clinic data.

    Parameters:
    -----------
    rrs: pd.DataFrame
        Dataframe of RRS samples
    opd_2to9: pd.DataFrame
        Dataframe of OPD samples
    clinic_areas: list
        List of area names for clinical cases

    Returns:
    clinic_areas_mask: np.array
        Boolean mask of clinical cases
    """
    #Clinics mask
    if clinic_areas[0] in rrs['hfca'].unique():
        clinic_areas_mask = rrs['hfca'].isnull()&rrs['hfca'].notnull()
        for a in clinic_areas:
            clinic_areas_mask = clinic_areas_mask | (rrs['hfca'] == a)
    elif clinic_areas[0] in opd_2to9['place'].unique():
        clinic_areas_mask = opd_2to9['place'].isnull()&opd_2to9['place'].notnull()
        for a in clinic_areas:
            clinic_areas_mask = clinic_areas_mask | (opd_2to9['place'] == a)
    else:
        print('Invalid area names for clinic cases: ' + clinic_areas[0])
    return clinic_areas_mask

def clinic_df(clinic_areas, rrs, opd_2to9):
    """This method defines the clinical data to be used for the
    corresponding area.

    Parameters:
    -----------
    clinic_areas: list
        List of area names for clinical cases
    rrs: pd.DataFrame
        Dataframe of RRS samples
    opd_2to9: pd.DataFrame
        Dataframe of OPD samples

    Returns:
    --------
    clinic_data: pd.DataFrame
        The corresponding dataframe to use
    """
    if clinic_areas[0] in rrs['hfca'].unique():
        rrs['visits'] = rrs['tot_test']
        return rrs
    elif clinic_areas[0] in opd_2to9['place'].unique():
        return opd_2to9
    else:
        print('Invalid area names for clinic cases: ' + clinic_areas[0])

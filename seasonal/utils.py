import numpy as np
from seasonal import constants
from climlab.solar.insolation import daily_insolation


def seasonal_ebm(t, Z, F_c, lat):
    """Calculate the increments in the state vector Z, dZ/dt, for the seasonal cycle model

    Parameters
    ----------
    t : numpy.float
        Time step in seconds
    Z : numpy.array
        An array of the state vectors: T_atm, T_land, T_ocean
    F_c : numpy.float
        Prescribed heat flux convergence
    lat : numpy.float
        Prescribed latitude

    Returns
    -------
    dZdt : numpy.array
        The estimated slope in Z
    """

    T_atm, T_land, T_ocean = Z
    F_sw = daily_insolation(lat, t/constants.seconds_per_day % 365)

    dTa_dt = 1/constants.C_atm*(F_c
                                + constants.epsilon*constants.sigma*T_ocean**4
                                + constants.epsilon*constants.sigma*T_land**4
                                - 4*constants.epsilon*constants.sigma*T_atm**4)
    dTl_dt = 1/constants.C_land*((1 - constants.alpha)*F_sw
                                 + constants.epsilon*constants.sigma*T_atm**4
                                 - constants.sigma*T_land**4)
    dTo_dt = 1/constants.C_ocean*((1 - constants.alpha)*F_sw
                                  + constants.epsilon*constants.sigma*T_atm**4
                                  - constants.sigma*T_ocean**4)

    dZdt = np.array([dTa_dt, dTl_dt, dTo_dt])

    return dZdt

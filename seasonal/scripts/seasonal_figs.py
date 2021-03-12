# Analysis
import xarray as xr
import numpy as np
import pandas as pd
from climlab.solar.insolation import daily_insolation
import time

savedir = '/glade/work/mckinnon/seasonal/proc'

# constants
alpha = 0.3
epsilon = 0.8
mwe = 4.2e6
C_land = 2*mwe
C_atm = 2*mwe
C_ocean = 100*mwe
C_ocean_deep = 3900*mwe
kappa = 0
sigma = 5.67e-8  # stefan boltzmann constant, W m^-2 K^-4


def seasonal_ebm(t, Z, F_c, lat):
    """Calculate the increments in the state vector Z, dZ/dt

    Parameters
    ----------
    Z : numpy.array
        An array of the state vectors: T_atm, T_land, T_ocean, T_ocean_deep

    Returns
    -------
    dZdt : numpy.array
        The estimated slope in Z
    """

    T_atm, T_land, T_ocean = Z
    F_sw = daily_insolation(lat, t/seconds_per_day % 365)

    dTa_dt = 1/C_atm*(F_c
                      + epsilon*sigma*T_ocean**4
                      + epsilon*sigma*T_land**4
                      - 4*epsilon*sigma*T_atm**4)
    dTl_dt = 1/C_land*((1 - alpha)*F_sw
                       + epsilon*sigma*T_atm**4
                       - sigma*T_land**4)
    dTo_dt = 1/C_ocean*((1 - alpha)*F_sw
                        + epsilon*sigma*T_atm**4
                        - sigma*T_ocean**4)

    dZdt = np.array([dTa_dt, dTl_dt, dTo_dt])

    return dZdt


nyrs = 50
yrs_avg = 10
dt = 1  # day
ndays_per_year = 365
seconds_per_day = 24*3600
h = dt*seconds_per_day
t = np.arange(365*nyrs*seconds_per_day, step=h)
time_vec = pd.date_range(start='1950-01-01', freq='D', periods=ndays_per_year)

F_c_range = np.arange(-120, 151, 1)
lat_range = np.arange(-87.5, 90, 1)

T_land = np.empty((ndays_per_year, len(lat_range), len(F_c_range)))
T_ocean = np.empty_like(T_land)

# iterate through lat, F_c to create land, ocean time series

for lat_ct, lat in enumerate(lat_range):
    print('%i/%i' % (lat_ct + 1, len(lat_range)))
    for F_ct, F_c in enumerate(F_c_range):
        t1 = time.time()
        F0 = np.mean(daily_insolation(lat, np.arange(365))).values
        T0 = ((F0*(1-alpha) + F_c)/sigma)**(1/4)
        T0 = np.max((T0, 260))
        Tl_0 = T0
        To_0 = T0
        Ta_0 = T0 - 25

        Z = np.empty((len(t), 3))
        Z[0, :] = np.array([Ta_0, Tl_0, To_0])

        # Use RK4 for integration
        for ct in range(1, len(t)):

            t_n = t[ct - 1]
            Z_n = Z[ct - 1, :]

            k1 = seasonal_ebm(t_n, Z_n, F_c, lat)
            k2 = seasonal_ebm(t_n + h/2, Z_n + h/2*k1, F_c, lat)
            k3 = seasonal_ebm(t_n + h/2, Z_n + h/2*k2, F_c, lat)
            k4 = seasonal_ebm(t_n + h, Z_n + h*k3, F_c, lat)

            Z_n1 = Z_n + 1/6*h*(k1 + 2*k2 + 2*k3 + k4)

            Z[ct, :] = Z_n1

        # save and average end
        T_land[:, lat_ct, F_ct] = Z[-dt*ndays_per_year*yrs_avg:, 1].reshape((yrs_avg,
                                                                             ndays_per_year)).mean(axis=0)
        T_ocean[:, lat_ct, F_ct] = Z[-dt*ndays_per_year*yrs_avg:, 2].reshape((yrs_avg,
                                                                             ndays_per_year)).mean(axis=0)

        t2 = time.time()
        print(t2-t1)
ds_ebm = xr.Dataset(data_vars={'T_land': (('time', 'lat', 'F_c'), T_land),
                               'T_ocean': (('time', 'lat', 'F_c'), T_ocean)},
                    coords={'time': time_vec, 'lat': lat_range, 'F_c': F_c_range})

ds_ebm.to_netcdf('%s/ebm_fits_1x1.nc' % savedir)

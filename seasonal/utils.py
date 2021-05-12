import numpy as np
from seasonal import constants
from climlab.solar.insolation import daily_insolation
import os
import xarray as xr
import pandas as pd


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


def find_best_params(da_data, ds_ebm, savedir, dataname, m_opts=np.arange(0, 1.01, 0.01)):
    """Given prospective model fits as a function of m and F_c, find the best fit for each gridbox.

    Parameters
    ----------
    da_data : xarray.DataArray
        Climatology to fit to. Should have dimensions 'month', 'lat, 'lon'
    ds_ebm : xarray.Dataset
        Output from EBM, contains T_land and T_ocean
    savedir : float
        Directory to save best parameter fits to
    dataname : float
        Name of data being fit to (e.g. BEST, CESM1-LE-001, etc)
    m_opts : numpy.array
        Options for mixing valies

    Returns
    -------

    """

    savename = '%s/ebm_fit_%s.nc' % (savedir, dataname)

    if os.path.isfile(savename):
        ds_fit = xr.open_dataset(savename)

    else:

        nlat = len(da_data['lat'])
        nlon = len(da_data['lon'])

        # interpolate EBM output to same grid
        this_ds_ebm = ds_ebm.interp(lat=da_data.lat)

        # create different options for weightings
        T_weighted = []
        for m in m_opts:
            T_weighted.append(m*this_ds_ebm['T_land'] + (1 - m)*this_ds_ebm['T_ocean'])
        T_weighted = xr.concat(T_weighted, dim='m')
        T_weighted = T_weighted.assign_coords({'m': m_opts})

        # Initialize things to save
        m_save = np.nan*np.ones((nlat, nlon))
        F_c_save = np.nan*np.ones((nlat, nlon))
        rho = np.nan*np.ones((nlat, nlon))
        T_model = np.nan*np.ones((nlat, nlon, 12))

        for lat_ct, this_lat in enumerate(da_data['lat']):
            for lon_ct, this_lon in enumerate(da_data['lon']):
                this_clim = da_data.sel({'lat': this_lat, 'lon': this_lon})
                model = T_weighted.sel({'lat': this_lat})
                if np.isnan(model).any():  # case where the interpolation didn't extend to the poles
                    continue

                rmse = (np.mean((model.transpose(..., 'month') - this_clim)**2, axis=-1))
                idx = np.argwhere(rmse.values == np.min(rmse.values))

                m_save[lat_ct, lon_ct] = rmse[idx[0][0], idx[0][1]].m.values
                F_c_save[lat_ct, lon_ct] = rmse[idx[0][0], idx[0][1]].F_c.values

                model_selected = model.sel({'m': m_save[lat_ct, lon_ct], 'F_c': F_c_save[lat_ct, lon_ct]})
                T_model[lat_ct, lon_ct, :] = model_selected
                rho[lat_ct, lon_ct] = np.corrcoef(this_clim.values, model_selected.values)[0, 1]

        # make data arrays for all output
        da_T = xr.DataArray(T_model,
                            dims=['lat', 'lon', 'month'],
                            coords={'lat': da_data['lat'], 'lon': da_data['lon'], 'month': da_data['month']})

        lam = 4*constants.sigma*((da_T.mean('month') + 273.15)**3)

        ds_m = xr.DataArray(m_save,
                            dims=['lat', 'lon'],
                            coords={'lat': da_data['lat'], 'lon': da_data['lon']})

        ds_rho = xr.DataArray(rho**2,
                              dims=['lat', 'lon'],
                              coords={'lat': da_data['lat'], 'lon': da_data['lon']})

        ds_Fc = xr.DataArray(F_c_save,
                             dims=['lat', 'lon'],
                             coords={'lat': da_data['lat'], 'lon': da_data['lon']})

        # combine all into one dataset
        ds_fit = xr.merge([da_T.to_dataset(name='T'),
                           lam.to_dataset(name='lam'),
                           ds_m.to_dataset(name='m'),
                           ds_rho.to_dataset(name='rho2'),
                           ds_Fc.to_dataset(name='Fc')])

        # save
        ds_fit.to_netcdf(savename)

    return ds_fit


def seasonal_solution(m, lam):
    """The gain and lag of the seasonal cycle in temperature from the mixture EBM.

    Parameters
    ----------
    m : numpy.ndarray
        An array of values [0, 1] for the mixing ratio. 0 = ocean, 1 = land
    lam : numpy.ndarray
        An array of values (> 0) for the feedback parameter. Greater values = less sensitive

    Returns
    -------
    da_gain_ebm : xr.DataArray
        DataArray containing predicted gain as a function of m and lambda
    da_lag_ebm : xr.DataArray
        DataArray containing predicted lag as a function of m and lambda

    """

    m = m[np.newaxis, :]
    lam = lam[:, np.newaxis]

    A1 = m*(constants.omega**2*constants.C_land**2 + lam**2)**(-1/2)
    A2 = (1 - m)*(constants.omega**2*constants.C_ocean**2 + lam**2)**(-1/2)
    cos_phi1 = lam*(constants.omega**2*constants.C_land**2 + lam**2)**(-1/2)
    sin_phi1 = constants.omega*constants.C_land*(constants.omega**2*constants.C_land**2 + lam**2)**(-1/2)

    # base of triangle
    X = A1*cos_phi1
    # height of triangle
    Y = A1*sin_phi1 + A2

    gain = (X**2 + Y**2)**(1/2)
    gain *= 1e3  # per kW
    lag = 1/constants.omega*np.arctan(Y/X)
    lag /= constants.seconds_per_day  # in days

    da_gain_ebm = xr.DataArray(gain,
                               dims=('lambda', 'mixing'),
                               coords={'lambda': lam.flatten(), 'mixing': m.flatten()})

    da_lag_ebm = xr.DataArray(lag,
                              dims=('lambda', 'mixing'),
                              coords={'lambda': lam.flatten(), 'mixing': m.flatten()})

    return da_gain_ebm, da_lag_ebm


def get_soln_constants(lam):
    """Calculate the various constants used in the solution to the EBM from Geoffroy et al, 2013

    Parameters
    ----------
    lam : numpy.ndarray
        An array of values (> 0) for the feedback parameter. Greater values = less sensitive

    Returns
    -------
    b, b_star, delta, tau_f, tau_s, phi_f, phi_s, a_f, a_s : floats
        All derived constants for the Geoffroy solution

    """

    b = (lam + constants.gamma)/constants.C_ocean + constants.gamma/constants.C_od
    b_star = (lam + constants.gamma)/constants.C_ocean - constants.gamma/constants.C_od
    delta = b**2 - 4*(lam*constants.gamma)/(constants.C_ocean*constants.C_od)
    tau_f = constants.C_ocean*constants.C_od/(2*constants.gamma*lam)*(b - delta**(1/2))
    tau_s = constants.C_ocean*constants.C_od/(2*constants.gamma*lam)*(b + delta**(1/2))
    phi_f = constants.C_ocean/(2*constants.gamma)*(b_star - delta**(1/2))
    phi_s = constants.C_ocean/(2*constants.gamma)*(b_star + delta**(1/2))
    a_f = phi_s*tau_f/(constants.C_ocean*(phi_s - phi_f))*lam
    a_s = -phi_f*tau_s*lam/(constants.C_ocean*(phi_s - phi_f))

    return b, b_star, delta, tau_f, tau_s, phi_f, phi_s, a_f, a_s


def ramp_solution(m, lam, dF=3.74, yrs_ramp=np.arange(1971, 2051), return_endmembers=False):
    """Temperature anomalies in response to a linear forcing as a functio of m and lambda.

    Parameters
    ----------
    m : numpy.ndarray
        An array of values [0, 1] for the mixing ratio. 0 = ocean, 1 = land
    lam : numpy.ndarray
        An array of values (> 0) for the feedback parameter. Greater values = less sensitive

    Returns
    -------
    da_T_ebm : xr.DataArray
        DataArray containing predicted temperature as a function of m and lambda, and time

    """

    nyrs = len(yrs_ramp)

    # Time vector in seconds, monthly time step
    t = np.linspace(0, nyrs*constants.seconds_per_day*constants.days_per_year, nyrs*12)[:, np.newaxis, np.newaxis]
    m = m[np.newaxis, :]
    lam = lam[:, np.newaxis]

    b, b_star, delta, tau_f, tau_s, phi_f, phi_s, a_f, a_s = get_soln_constants(lam)

    k = dF/(nyrs*constants.days_per_year*constants.seconds_per_day)

    T_anom_ocean_linear = k/lam*(t - tau_f*a_f*(1 - np.exp(-t/tau_f)) -
                                 tau_s*a_s*(1 - np.exp(-t/tau_s)))

    tau_land = constants.C_land/lam
    T_anom_land_linear = k/lam*(t - tau_land*(1 - np.exp(-t/tau_land)))

    T_anom_mix = m*T_anom_land_linear + (1 - m)*T_anom_ocean_linear

    real_time = pd.date_range(start='%i-01' % (yrs_ramp[0]), periods=len(t), freq='M')

    da_T_ebm = xr.DataArray(T_anom_mix,
                            dims=('time', 'lambda', 'mixing'),
                            coords={'time': real_time, 'lambda': lam.flatten(), 'mixing': m.flatten()})

    if return_endmembers:
        return T_anom_ocean_linear, T_anom_land_linear
    else:
        return da_T_ebm


def ramp_stabilize_solution(m, lam, dF=3.74, yrs_ramp=np.arange(1971, 2051), yrs_stable=np.arange(2051, 2200)):
    """Temperature anomalies in response to a linear then stabilized forcing as a function of m and lambda.

    Parameters
    ----------
    m : numpy.ndarray
        An array of values [0, 1] for the mixing ratio. 0 = ocean, 1 = land
    lam : numpy.ndarray
        An array of values (> 0) for the feedback parameter. Greater values = less sensitive

    Returns
    -------
    da_T_ebm : xr.DataArray
        DataArray containing predicted temperature as a function of m and lambda, and time

    """

    # first get ramped solution
    T_anom_ocean_ramp, T_anom_land_ramp = ramp_solution(m, lam, dF, yrs_ramp, return_endmembers=True)

    nyrs_ramp = len(yrs_ramp)
    steps_per_yr = 12
    t1 = np.linspace(0, nyrs_ramp*constants.seconds_per_day*constants.days_per_year, nyrs_ramp*steps_per_yr)

    nyrs_stable = len(yrs_stable)
    t2 = np.linspace(0, nyrs_stable*constants.seconds_per_day*constants.days_per_year, nyrs_stable*steps_per_yr)
    t2 += t1[-1] + constants.seconds_per_day*constants.days_per_year/steps_per_yr
    t_st = t2[0]

    t2 = t2[:, np.newaxis, np.newaxis]

    m = m[np.newaxis, :]
    lam = lam[:, np.newaxis]

    b, b_star, delta, tau_f, tau_s, phi_f, phi_s, a_f, a_s = get_soln_constants(lam)

    k = dF/(nyrs_ramp*constants.days_per_year*constants.seconds_per_day)

    slow_term = tau_s*a_s*(1 - np.exp(-t_st/tau_s))*np.exp(-(t2 - t_st)/tau_s)
    fast_term = tau_f*a_f*(1 - np.exp(-t_st/tau_f))*np.exp(-(t2 - t_st)/tau_f)
    T_anom_ocean_stable = k/lam*(t_st - slow_term - fast_term)

    tau_land = constants.C_land/lam
    T_anom_land_stable = k/lam*(t_st - tau_land*(1 - np.exp(-t_st/tau_land))*np.exp(-(t2 - t_st)/tau_land))

    T_anom_land = np.empty((len(t1) + len(t2), len(lam), 1))
    T_anom_ocean = np.empty((len(t1) + len(t2), len(lam), 1))

    T_anom_land[:len(t1), ...] = T_anom_land_ramp
    T_anom_land[len(t1):, ...] = T_anom_land_stable

    T_anom_ocean[:len(t1), ...] = T_anom_ocean_ramp
    T_anom_ocean[len(t1):, ...] = T_anom_ocean_stable

    T_anom_mix = m*T_anom_land + (1 - m)*T_anom_ocean

    real_time = pd.date_range(start='%i-01' % (yrs_ramp[0]), periods=len(t1) + len(t2), freq='M')

    da_T_ebm = xr.DataArray(T_anom_mix,
                            dims=('time', 'lambda', 'mixing'),
                            coords={'time': real_time, 'lambda': lam.flatten(), 'mixing': m.flatten()})

    return da_T_ebm

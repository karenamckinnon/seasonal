import numpy as np
from seasonal import constants
from climlab.solar.insolation import daily_insolation
import os
import xarray as xr
import pandas as pd
from glob import glob


smile_dir = '/gpfs/fs1/collections/cdg/data/CLIVAR_LE'

# Set interpolation lat/lon, and landmask
lat1x1 = np.arange(-89.5, 90, 1)
lon1x1 = np.arange(0.5, 360, 1)

landfrac_name = 'sftlf'
f_landfrac = sorted(glob('%s/%s/%s/%s/*.nc' % (smile_dir, 'cesm_lens', 'fx', landfrac_name)))
da_landfrac = xr.open_dataset(f_landfrac[0])[landfrac_name]
da_landfrac = da_landfrac.interp({'lat': lat1x1, 'lon': lon1x1})
is_land = da_landfrac > 50


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


def calc_amp_phase(da):
    """Calculate the amplitude and phase of an annual-period sinusoid of monthly data

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the climatology of interest. The time variable should be "month"

    Returns
    -------
    ds_1yr : xr.Dataset
        Contains amplitude, phase, and variance explained (R2) for the annual-period sinusoid.
    da_rec : xr.DataArray
        Contains the reconsructed data with one annual-period sinusoid.

    """

    # Get time vector for monthly data
    doy_da = xr.DataArray(np.arange(365), dims=['time'],
                          coords={'time': pd.date_range(start='1950/01/01', freq='D', periods=365)})

    t_basis = (doy_da.groupby('time.month').mean()/365).values

    nt = len(t_basis)

    # 1/yr sinusoid
    basis = np.exp(2*np.pi*1j*t_basis)

    # Project data onto basis
    data = da.copy().values
    mu = np.mean(data, axis=0)
    data -= mu[np.newaxis, ...]

    if len(da.shape) == 3:
        coeff = 2/nt*(np.sum(basis[..., np.newaxis, np.newaxis]*da.transpose('month', ...).values,
                             axis=0))
    elif len(da.shape) == 2:  # only latitude
        coeff = 2/nt*(np.sum(basis[..., np.newaxis]*da.transpose('month', ...).values, axis=0))

    amp_1yr = np.abs(coeff)
    phase_1yr = (np.angle(coeff))*365/(2*np.pi)

    if len(da.shape) == 3:
        rec = np.real(np.conj(coeff[np.newaxis, ...])*basis[..., np.newaxis, np.newaxis])
    elif len(da.shape) == 2:  # only latitude
        rec = np.real(np.conj(coeff[np.newaxis, ...])*basis[..., np.newaxis])

    da_rec = da.copy(data=rec)
    rho = xr.corr(da_rec, da, dim='month')

    if len(da.shape) == 3:
        ds_1yr = xr.Dataset(data_vars={'A': (('lat', 'lon'), amp_1yr),
                                       'phi': (('lat', 'lon'), phase_1yr),
                                       'R2': (('lat', 'lon'), rho**2)},
                            coords={'lat': da.lat, 'lon': da.lon})
    elif len(da.shape) == 2:  # only latitude
        ds_1yr = xr.Dataset(data_vars={'A': (('lat'), amp_1yr),
                                       'phi': (('lat'), phase_1yr),
                                       'R2': (('lat'), rho**2)},
                            coords={'lat': da.lat})

    return ds_1yr, da_rec


def change_lon_180(da):
    """Returns the xr.DataArray with longitude changed from 0, 360 to -180, 180"""
    da = da.assign_coords({'lon': (((da.lon + 180) % 360) - 180)})
    return da.sortby('lon')


def calc_trend_season(da, trend_years, this_season):
    """Calculate the trend in a dataarray over a given time period

    Parameters
    ----------
    da : xr.DataArray
        Contains data for trend calculation. Must have standard time dimension.
    trend_years : tuple
        (Start year, end year) for trend to be calculated over
    this_season : str
        Either 'ann' or 'MMM' (i.e. 'DJF') of traditional season.
        Data will be average over the year, or for the season, before trend calculation.

    Returns
    -------
    da_beta : xr.DataArray
        The slope of the OLS trend in the data, in units of per year.

    """
    if this_season != 'ann':

        da = da.resample(time='QS-DEC').mean()
        da = da.sel({'time': da['time.season'] == this_season})

    da = da.sel({'time': (da['time.year'] >= trend_years[0]) & (da['time.year'] <= trend_years[1])})

    da = da.groupby('time.year').mean().load()

    # Calculate linear trend
    year_orig = da['year']
    da['year'] = year_orig - year_orig.mean()
    da_beta = da.to_dataset(name='data').polyfit(dim='year', deg=1)

    da_beta = da_beta['data_polyfit_coefficients'].sel({'degree': 1})

    return da_beta


def calc_load_SMILE_trends(models, trend_years, seasonal_years, this_season, forcing, savedir):
    """Calculate or load pre-calculated seasonal cycle and trend metrics from each SMILE.

    Parameters
    ----------
    models : list
        Names (matching paths) of each SMILE
    trend_years : tuple
        The start and end year of the trend to calculate
    seasonal_years : tuple
        The start and end year of the data to be used to calculate the seasonal cycle
    this_season : str
        Standard season e.g. 'DJF' or 'ann' (annual mean) over which averages are taken before calculating trend
    forcing : str
        Type of heating for seasonal cycle: 'ERA5_div', 'CERES_div', 'ERA5', 'CERES'
        ERA5: SW at the surface, CERES: SW at TOA
        div: to include heat flux divergence from ERA5 or not
    savedir : str
        Where to save the netcdfs with the seasonal cycle and trend

    Returns
    -------
    da_gain : xr.DataArray
        Gain of the seasonal cycle (based on member 1) for each SMILE
    da_lag : xr.DataArray
        Lag of the seasonal cycle (in days, based on member 1) for each SMILE
    da_trend : xr.DataArray
        Trend over the specified trend years in each SMILE

    """

    # monthly temperature
    freq = 'Amon'
    varname = 'tas'

    # Collect all the amplitude, phases, and trends
    da_amp = []
    da_phase = []
    da_trend = []

    for m in models:

        amp_savename = '%s/%s_amplitude.nc' % (savedir, m)
        phase_savename = '%s/%s_phase.nc' % (savedir, m)
        R2_savename = '%s/%s_R2.nc' % (savedir, m)
        trend_savename = '%s/%s_trend_%s_%i-%i.nc' % (savedir, m, this_season, trend_years[0], trend_years[-1])
        files = sorted(glob('%s/%s/%s/%s/*historical_rcp85*nc' % (smile_dir, m, freq, varname)))

        if (os.path.isfile(amp_savename) & os.path.isfile(trend_savename)):
            this_amp = xr.open_dataarray(amp_savename)
            this_phase = xr.open_dataarray(phase_savename)
            this_R2 = xr.open_dataarray(R2_savename)
            this_trend = xr.open_dataarray(trend_savename)
        else:
            print(m)
            this_amp = []
            this_phase = []
            this_R2 = []
            this_trend = []

            for counter, f in enumerate(files):
                print(counter)

                if m == 'ec_earth_lens':
                    ds = xr.open_dataset(f, decode_times=False)
                    new_time = pd.date_range(start='1860-01-01', periods=len(ds.time), freq='M')
                    ds = ds.assign_coords({'time': new_time})
                else:
                    ds = xr.open_dataset(f)

                da = ds[varname].load()

                da = da.interp({'lat': lat1x1, 'lon': lon1x1})

                da_seasonal = da.copy().sel({'time': (da['time.year'] >= seasonal_years[0]) &
                                             (da['time.year'] <= seasonal_years[1])}).groupby('time.month').mean()

                ds_1yr_T, _ = calc_amp_phase(da_seasonal)

                this_amp.append(ds_1yr_T['A'])
                this_phase.append(ds_1yr_T['phi'])
                this_R2.append(ds_1yr_T['R2'])

                this_trend.append(calc_trend_season(da.copy(), trend_years, this_season))

            this_amp = xr.concat(this_amp, dim='member')
            this_phase = xr.concat(this_phase, dim='member')
            this_R2 = xr.concat(this_R2, dim='member')
            this_trend = xr.concat(this_trend, dim='member')

            this_amp.attrs = {'sc_years': seasonal_years}
            this_phase.attrs = {'sc_years': seasonal_years}
            this_phase.attrs = {'sc_years': seasonal_years}
            this_trend.attrs = {'trend_years': trend_years}

            this_amp.to_netcdf(amp_savename)
            this_phase.to_netcdf(phase_savename)
            this_R2.to_netcdf(R2_savename)
            this_trend.to_netcdf(trend_savename)

        this_amp = this_amp.isel({'member': 0})  # signal to noise is large for seasonal cycle, only need 1 member
        this_phase = this_phase.isel({'member': 0})
        this_R2 = this_R2.isel({'member': 0})
        this_trend = this_trend.mean('member')

        is_greenland = (this_amp.lat > 60) & (this_amp.lon > 300)
        this_mask = is_land & (np.abs(this_amp.lat) > 20) & ~is_greenland

        this_amp = this_amp.sel({'lat': slice(30, 90)})
        this_amp = this_amp.where(this_mask == 1)

        this_phase = this_phase.sel({'lat': slice(30, 90)})
        this_phase = this_phase.where(this_mask == 1)

        this_trend = this_trend.sel({'lat': slice(30, 90)})
        this_trend = this_trend.where(this_mask == 1)

        da_amp.append(this_amp)
        da_phase.append(this_phase)
        da_trend.append(this_trend)

    da_amp = xr.concat(da_amp, dim='model', coords='minimal')
    da_amp = da_amp.assign_coords({'model': list(models)})

    da_phase = xr.concat(da_phase, dim='model', coords='minimal')
    da_phase = da_phase.assign_coords({'model': list(models)})

    tmp = da_phase.values
    tmp[tmp < 0] += 365
    da_phase.values = tmp

    da_trend = xr.concat(da_trend, dim='model', coords='minimal')
    da_trend = da_trend.assign_coords({'model': list(models)})

    ds_1yr_F = get_heating(forcing)

    da_gain = da_amp/ds_1yr_F['A']
    da_lag = da_phase - ds_1yr_F['phi']

    tmp = da_lag.values
    tmp[tmp < 0] += 365
    tmp[tmp > 100] = np.nan
    da_lag.values = tmp

    return da_gain, da_lag, da_trend


def get_heating(forcing, return_components=False, era5_sw_fname='/glade/work/mckinnon/ERA5/month/ssr/era5_ssr.nc',
                heatdiv_fname='/glade/work/mckinnon/seasonal/data/AnnualCycle-1979-2020-TEDIV-CSCALE-ERA5-LL90.nc',
                ceres_sw_fname='/glade/work/mckinnon/CERES/CERES_EBAF-TOA_Ed4.1_Subset_CLIM01-CLIM12.nc'):
    """Calculate the latitudinal variations in the amplitude and phase of heating at the surface

    Parameters
    ----------
    forcing : str
        Type of forcing to use: 'ERA5_div', 'CERES_div', 'ERA5', 'CERES'
        ERA5: SW at the surface, CERES: SW at TOA
        div: to include heat flux divergence from ERA5 or not
    return_components : bool
        Indicator of whether to also return the sw_down and div fields
    era5_sw_fname : str
        Path and filename for SW from ERA5
    heatdiv_fname : str
        Path and filename for divergence from ERA5
    ceres_sw_fname : str
        Path and filename for SW from CERES

    Returns
    -------
    ds_1yr_F : xr.Dataset
        Amplitude, phase, and variance explained for first annual sinusoid of zonal-average forcing

    """

    # Solar at surface from ERA5
    if 'ERA5' in forcing:
        sw_down = xr.open_dataarray(era5_sw_fname)  # J/m2
        sw_down /= constants.seconds_per_day

        sw_down = sw_down.groupby('time.month').mean()
        sw_down = sw_down.rename({'latitude': 'lat', 'longitude': 'lon'})
        sw_down = sw_down.sortby('lat')
        sw_down = sw_down.interp({'lat': lat1x1, 'lon': lon1x1})
        sw_down = change_lon_180(sw_down)

    elif 'CERES' in forcing:
        ds_ceres = xr.open_dataset(ceres_sw_fname)
        ds_ceres = ds_ceres.rename({'ctime': 'month'})
        sw_down = ds_ceres['solar_clim'] - ds_ceres['toa_sw_all_clim']

    # Heat flux divergence
    ds_heatdiv = xr.open_dataset(heatdiv_fname)

    da_heatdiv = xr.DataArray(ds_heatdiv['AC_TEDIV'].values, dims=('month', 'lat', 'lon'),
                              coords={'month': ds_heatdiv['time'].values, 'lat': ds_heatdiv['lat'].values,
                                      'lon': ds_heatdiv['lon'].values})
    da_heatdiv = da_heatdiv.sortby('lat')
    da_heatdiv = da_heatdiv.interp({'lat': lat1x1, 'lon': lon1x1})
    da_heatdiv = change_lon_180(da_heatdiv)

    if 'div' in forcing:
        all_heating = sw_down - da_heatdiv
    else:
        all_heating = sw_down

    # only consider latitudinal variations in forcing
    ds_1yr_F, _ = calc_amp_phase(all_heating.mean('lon'))

    if return_components & ('div' in forcing):
        return ds_1yr_F, sw_down, da_heatdiv
    elif return_components & ('div' not in forcing):
        return ds_1yr_F, sw_down
    else:
        return ds_1yr_F


def predict_with_ebm(da_gain, da_lag, da_gain_ebm, da_lag_ebm, da_trend_ebm, dataname, forcing, savedir):
    """Use gain and lag from obs or model + EBM prediction to map from seasonal cycle to temperature trends.

    Parameters
    ----------
    da_gain : xr.DataArray
        Gain from observations or model
    da_lag : xr.DataArray
        Lag from observations or model
    da_gain_ebm : xr.DataArray
        Gain in the EBM
    da_lag_ebm : xr.DataArray
        Lag in EBM
    da_trend_ebm : xr.DataArray
        Temperature trend in EBM. Will be a function of the forcing and trend years, specified earlier.
    dataname : str
        Name of data or model for the seasonal cycle
    forcing : str
        Type of heating for seasonal cycle: 'ERA5_div', 'CERES_div', 'ERA5', 'CERES'
        ERA5: SW at the surface, CERES: SW at TOA
        div: to include heat flux divergence from ERA5 or not
    savedir : str
        Where to save the netcdfs with the seasonal cycle and trend

    Returns
    -------
    da_T_pred : xr.DataArray
        Temperature trend prediction at each location from the EBM.

    """

    savename = '%s/ebm_predicted_T_trend_%s_F-%s.nc' % (savedir, dataname, forcing)
    if os.path.isfile(savename):
        da_T_pred = xr.open_dataarray(savename)
    else:
        da_T_pred = []
        for this_lat in da_gain.lat:
            if this_lat < 30:  # only do predictions in the NH extratropics
                continue
            this_gain = da_gain.sel({'lat': this_lat})
            this_lag = da_lag.sel({'lat': this_lat})

            # At every gridbox, find the closest gain and lag from the EBM.
            T_pred = []
            for this_lon in this_gain.lon:

                if np.isnan(this_lag.sel({'lon': this_lon}).values):  # over ocean
                    T_pred.append(np.nan)
                else:
                    delta_gain = da_gain_ebm - this_gain.sel({'lon': this_lon}).values
                    delta_lag = da_lag_ebm - this_lag.sel({'lon': this_lon}).values

                    # normalize for fair comparison in "distance"
                    delta_gain = delta_gain/this_gain.std('lon')
                    delta_lag = delta_lag/this_lag.std('lon')

                    d = np.sqrt(delta_gain**2 + delta_lag**2)
                    idx_match = np.where(d == d.min())
                    T_pred.append(da_trend_ebm[idx_match].values.flatten()[0])

            da_T_pred.append(xr.DataArray(np.array(T_pred), dims=('lon'), coords={'lon': this_gain.lon}))

        da_T_pred = xr.concat(da_T_pred, dim='lat')
        da_T_pred['lat'] = da_gain.lat[da_gain.lat >= 30]

        da_T_pred.to_netcdf(savename)

    return da_T_pred

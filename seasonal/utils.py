import numpy as np
from scipy.spatial import distance_matrix
from seasonal import constants
from climlab.solar.insolation import daily_insolation
import os
import xarray as xr
import pandas as pd
from glob import glob
import geopandas
from helpful_utilities import ncutils
import geocat.viz as gv
from scipy.ndimage.morphology import binary_dilation


smile_dir = '/gpfs/fs1/collections/cdg/data/CLIVAR_LE'

# Set interpolation lat/lon, and landmask
lat1x1 = np.arange(-89.5, 90, 1)
lon1x1 = np.arange(0.5, 360, 1)

landfrac_name = 'sftlf'
f_landfrac = sorted(glob('%s/%s/%s/%s/*.nc' % (smile_dir, 'cesm_lens', 'fx', landfrac_name)))
da_landfrac = xr.open_dataset(f_landfrac[0])[landfrac_name]
da_landfrac = gv.xr_add_cyclic_longitudes(da_landfrac, 'lon')
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
    A2 = (1 - m)*(constants.omega**2*constants.C_ocean**2 + constants.lam_ocean**2)**(-1/2)
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

    k = dF/(nyrs*constants.days_per_year*constants.seconds_per_day)

    # Use a single lambda for the ocean
    b, b_star, delta, tau_f, tau_s, phi_f, phi_s, a_f, a_s = get_soln_constants(constants.lam_ocean)
    T_anom_ocean_linear = k/lam*(t - tau_f*a_f*(1 - np.exp(-t/tau_f)) -
                                 tau_s*a_s*(1 - np.exp(-t/tau_s)))

    # Use variable values of lambda for the land
    b, b_star, delta, tau_f, tau_s, phi_f, phi_s, a_f, a_s = get_soln_constants(lam)
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

    k = dF/(nyrs_ramp*constants.days_per_year*constants.seconds_per_day)

    # Use a single lambda for the ocean
    b, b_star, delta, tau_f, tau_s, phi_f, phi_s, a_f, a_s = get_soln_constants(constants.lam_ocean)

    slow_term = tau_s*a_s*(1 - np.exp(-t_st/tau_s))*np.exp(-(t2 - t_st)/tau_s)
    fast_term = tau_f*a_f*(1 - np.exp(-t_st/tau_f))*np.exp(-(t2 - t_st)/tau_f)
    T_anom_ocean_stable = k/lam*(t_st - slow_term - fast_term)

    # but use variable lambda for land
    b, b_star, delta, tau_f, tau_s, phi_f, phi_s, a_f, a_s = get_soln_constants(lam)
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
    data = da.copy()
    # make sure month is first dimension
    data = data.transpose('month', ...)
    vals = data.values
    mu = np.mean(vals, axis=0)
    vals -= mu[np.newaxis, ...]

    if len(da.shape) == 3:
        coeff = 2/nt*(np.sum(basis[..., np.newaxis, np.newaxis]*vals, axis=0))
    elif len(da.shape) == 2:  # only latitude
        coeff = 2/nt*(np.sum(basis[..., np.newaxis]*vals, axis=0))

    amp_1yr = np.abs(coeff)
    phase_1yr = (np.angle(coeff))*365/(2*np.pi)

    if len(da.shape) == 3:
        rec = np.real(np.conj(coeff[np.newaxis, ...])*basis[..., np.newaxis, np.newaxis])
    elif len(da.shape) == 2:  # only latitude
        rec = np.real(np.conj(coeff[np.newaxis, ...])*basis[..., np.newaxis])

    da_rec = data.copy(data=rec)
    rho = xr.corr(da_rec, data, dim='month')

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


def calc_load_SMILE_seasonal_cycle(models, seasonal_years, nboot, savedir, varname='tas'):
    """Calculate or load pre-calculated seasonal cycle in temperature, with bootstrapping, from each SMILE.

    Parameters
    ----------
    models : list
        Names (matching paths) of each SMILE
    seasonal_years : tuple
        The start and end year of the data to be used to calculate the seasonal cycle
    nboot : int
        How many bootstrap resamples to perform
    savedir : str
        Where to save the netcdfs with the seasonal cycle metrics
    varname : str
        CMOR-style variable name

    Returns
    -------
    ds_seasonal : xr.DataSet
        Dataset containing amplitude and phase of the seasonal cycle, across bootstrap samples

    """

    # monthly temperature
    freq = 'Amon'
    ds_seasonal = []
    for m in models:

        savename = '%s/%s_seasonal_cycle_%s_%03i-samples.nc' % (savedir, m, varname, nboot)

        if os.path.isfile(savename):
            ds_1yr_T_boot = xr.open_dataset(savename)
        else:

            files = sorted(glob('%s/%s/%s/%s/*historical_rcp85*nc' % (smile_dir, m, freq, varname)))
            f = files[0]  # using first member of each ensemble for seasonal cycle

            if m == 'ec_earth_lens':
                ds = xr.open_dataset(f, decode_times=False)
                new_time = pd.date_range(start='1860-01-01', periods=len(ds.time), freq='M')
                ds = ds.assign_coords({'time': new_time})
            else:
                ds = xr.open_dataset(f)

            da = ds[varname].load()
            da = gv.xr_add_cyclic_longitudes(da, 'lon')
            da = da.interp({'lat': lat1x1, 'lon': lon1x1})

            da_seasonal = da.copy().sel({'time': (da['time.year'] >= seasonal_years[0]) &
                                                 (da['time.year'] <= seasonal_years[1])})

            # da_seasonal = do_mask(da_seasonal)
            years = da_seasonal['time.year'].values
            unique_years = np.unique(years)
            # resample years to get uncertainty in seasonal cycle
            ds_1yr_T_boot = []
            for kk in range(nboot):
                if kk == 0:
                    da_boot = da_seasonal.groupby('time.month').mean()
                else:
                    boot_years = np.random.choice(unique_years, len(unique_years))
                    new_idx = [np.where(years == by)[0] for by in boot_years]
                    new_idx = np.array(new_idx).flatten()
                    da_boot = da_seasonal.isel(time=new_idx).groupby('time.month').mean()

                ds_1yr_T, _ = calc_amp_phase(da_boot)
                tmp = ds_1yr_T['phi'].values
                tmp[tmp < 0] += 365
                ds_1yr_T['phi'].values = tmp
                ds_1yr_T_boot.append(ds_1yr_T)

            ds_1yr_T_boot = xr.concat(ds_1yr_T_boot, dim='sample')
            ds_1yr_T_boot.attrs = {'sc_years': seasonal_years}
            ds_1yr_T_boot.to_netcdf(savename)
        ds_seasonal.append(ds_1yr_T_boot)

    ds_seasonal = xr.concat(ds_seasonal, dim='model', coords='minimal')
    ds_seasonal = ds_seasonal.assign_coords({'model': list(models)})

    return ds_seasonal


def calc_load_SMILE_trends(models, trend_years, this_season, savedir):
    """Calculate or load pre-calculated trends from each SMILE.

    Parameters
    ----------
    models : list
        Names (matching paths) of each SMILE
    trend_years : tuple
        The start and end year of the trend to calculate
    this_season : str
        Standard season e.g. 'DJF' or 'ann' (annual mean) over which averages are taken before calculating trend
    savedir : str
        Where to save the netcdfs with the trends of each ensemble member in each model

    Returns
    -------
    da_trend : xr.DataArray
        Ensemble-mean trend over the specified trend years in each SMILE

    """

    # monthly temperature
    freq = 'Amon'
    varname = 'tas'

    # Collect EM trends across models
    da_trend = []

    for m in models:

        trend_savename = '%s/%s_trend_%s_%i-%i.nc' % (savedir, m, this_season, trend_years[0], trend_years[-1])
        files = sorted(glob('%s/%s/%s/%s/*historical_rcp85*nc' % (smile_dir, m, freq, varname)))

        if os.path.isfile(trend_savename):
            this_trend = xr.open_dataarray(trend_savename)
        else:
            print(m)
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
                da = gv.xr_add_cyclic_longitudes(da, 'lon')
                da = da.interp({'lat': lat1x1, 'lon': lon1x1})
                # mask out ocean, greenland, and remove south of 30N
                # da = do_mask(da)

                this_trend.append(calc_trend_season(da, trend_years, this_season))

            this_trend = xr.concat(this_trend, dim='member')
            this_trend.attrs = {'trend_years': trend_years}
            this_trend.to_netcdf(trend_savename)

        this_trend = this_trend.mean('member')
        da_trend.append(this_trend)

    da_trend = xr.concat(da_trend, dim='model', coords='minimal')
    da_trend = da_trend.assign_coords({'model': list(models)})

    return da_trend


def do_mask(da):
    """Mask out Greenland and ocean, and remove south of 30N and north of 80N for 1x1 data"""

    countries = geopandas.read_file('/glade/work/mckinnon/seasonal/geom/ne_110m_admin_0_countries/')
    greenland = countries.query("ADMIN == 'Greenland'").reset_index(drop=True)
    ds = xr.Dataset(coords={'lon': np.linspace(-179.5, 180, 1000), 'lat': np.linspace(-90, 90, 1000)})
    da_greenland = ncutils.rasterize(greenland['geometry'], ds.coords)
    da_greenland = ncutils.lon_to_360(da_greenland)
    da_greenland = da_greenland.fillna(0)
    da_greenland = da_greenland.interp({'lat': lat1x1, 'lon': lon1x1})
    expanded_greenland = binary_dilation(binary_dilation((da_greenland > 0).values))  # regridded mask misses boundary
    da_greenland = da_greenland.copy(data=expanded_greenland)
    this_mask = is_land & ~(da_greenland)
    da = da.where(this_mask == 1)
    da = da.sel({'lat': slice(30, 80)})

    return da


def get_heating(forcing, nboot, seasonal_years,
                return_components=False, era5_sw_fname='/glade/work/mckinnon/ERA5/month/ssr/era5_ssr.nc',
                heatdiv_fname='/glade/work/mckinnon/seasonal/data/ZonalMean-1979-2020-TEDIV-CSCALE-ERA5-LL90.nc',
                ceres_sw_fname='/glade/work/mckinnon/CERES/CERES_EBAF-TOA_Ed4.1_Subset_CLIM01-CLIM12.nc'):
    """Calculate the latitudinal variations in the amplitude and phase of heating at the surface

    Parameters
    ----------
    forcing : str
        Type of forcing to use: 'ERA5_div', 'CERES_div', 'ERA5', 'CERES', CMIP model name
        ERA5: SW at the surface, CERES: SW at TOA
        div: to include heat flux divergence from ERA5 or not
    nboot : int
        Number of times to bootstrap resample years
    seasonal_years : tuple
         The start and end year of the data to be used to calculate the seasonal cycle.
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

    if 'CERES' in forcing:
        raise ValueError('CERES TOA not currently included')
    elif 'ERA5' in forcing:

        sw_net = xr.open_dataarray(era5_sw_fname)  # J/m2
        sw_net /= constants.seconds_per_day  # monthly data, accumulation period is one day

        sw_net = sw_net.rename({'latitude': 'lat', 'longitude': 'lon'})
        sw_net = sw_net.sortby('lat')
        sw_net = sw_net.mean('lon')

    else:  # climate model based forcing
        freq = 'Amon'
        var1 = 'rsds'
        var2 = 'rsus'
        files1 = sorted(glob('%s/%s/%s/%s/*historical_rcp85*nc' % (smile_dir, forcing, freq, var1)))
        files2 = sorted(glob('%s/%s/%s/%s/*historical_rcp85*nc' % (smile_dir, forcing, freq, var2)))

        if (len(files1) == 0) | (len(files2) == 0):
            return 0

        f1 = files1[0]
        f2 = files2[0]
        if forcing == 'ec_earth_lens':
            ds1 = xr.open_dataset(f1, decode_times=False)
            new_time = pd.date_range(start='1860-01-01', periods=len(ds1.time), freq='M')
            ds1 = ds1.assign_coords({'time': new_time})
            ds2 = xr.open_dataset(f2, decode_times=False)
            new_time = pd.date_range(start='1860-01-01', periods=len(ds2.time), freq='M')
            ds2 = ds2.assign_coords({'time': new_time})
        else:
            ds1 = xr.open_dataset(f1)
            ds2 = xr.open_dataset(f2)

        if (ds1.time.values != ds2.time.values).any():  # non-matching saved times, happens in MPI
            return 0
        da = ds1[var1].load() - ds2[var2].load()
        da = da.rename('rsns')
        sw_net = da.mean('lon')

    # interpolate to 1x1 and select only specified years for seasonal cycle
    sw_net = sw_net.interp({'lat': lat1x1})
    sw_net = sw_net.sel({'time': (sw_net['time.year'] >= seasonal_years[0]) &
                                 (sw_net['time.year'] <= seasonal_years[1])})
    if 'div' in forcing:

        # Heat flux divergence
        ds_heatdiv = xr.open_dataset(heatdiv_fname)
        da_heatdiv = xr.DataArray(ds_heatdiv['ZM_TEDIV'].values, dims=('time', 'lat'),
                                  coords={'time': sw_net['time'].values, 'lat': ds_heatdiv['lat'].values})
        da_heatdiv = da_heatdiv.sortby('lat')
        all_heating = sw_net - da_heatdiv
    else:
        all_heating = sw_net

    years = all_heating['time.year'].values
    unique_years = np.unique(years)
    # resample years to get uncertainty in seasonal cycle
    ds_1yr_F_boot = []
    for kk in range(nboot):
        if kk == 0:
            da_boot = all_heating.groupby('time.month').mean()
        else:
            boot_years = np.random.choice(unique_years, len(unique_years))
            new_idx = [np.where(years == by)[0] for by in boot_years]
            new_idx = np.array(new_idx).flatten()
            da_boot = all_heating.isel(time=new_idx).groupby('time.month').mean()

        ds_1yr_F, _ = calc_amp_phase(da_boot)
        tmp = ds_1yr_F['phi'].values
        tmp[tmp < 0] += 365
        ds_1yr_F['phi'].values = tmp
        ds_1yr_F_boot.append(ds_1yr_F)

    ds_1yr_F_boot = xr.concat(ds_1yr_F_boot, dim='sample')

    if return_components & ('div' in forcing):
        return ds_1yr_F_boot, sw_net, da_heatdiv
    elif return_components & ('div' not in forcing):
        return ds_1yr_F_boot, sw_net
    else:
        return ds_1yr_F_boot


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
    savename_lam = '%s/ebm_inferred_lam_%s_F-%s.nc' % (savedir, dataname, forcing)
    savename_mix = '%s/ebm_inferred_mix_%s_F-%s.nc' % (savedir, dataname, forcing)

    if os.path.isfile(savename) & os.path.isfile(savename_lam) & os.path.isfile(savename_mix):
        da_T_pred = xr.open_dataarray(savename)
        da_lam_inferred = xr.open_dataarray(savename_lam)
        da_mix_inferred = xr.open_dataarray(savename_mix)
    else:

        nlat = len(da_gain.lat)
        nlon = len(da_gain.lon)
        nx = nlat*nlon

        d1 = np.vstack((da_gain.values.flatten(), da_lag.values.flatten())).T
        has_data = ~np.isnan(d1[:, 0])
        d1 = d1[has_data, :]

        d2 = np.vstack((da_gain_ebm.values.flatten(), da_lag_ebm.values.flatten())).T

        norm1 = np.std(d1[:, 0])
        norm2 = np.std(d1[:, 1])
        d1[:, 0] /= norm1
        d1[:, 1] /= norm2
        d2[:, 0] /= norm1
        d2[:, 1] /= norm2

        d = distance_matrix(d1, d2)

        min_loc = np.argmin(d, axis=-1)

        T_pred = da_trend_ebm.values.flatten()[min_loc]

        mix_mat, lam_mat = np.meshgrid(da_trend_ebm['mixing'], da_trend_ebm['lambda'])
        lam_value = lam_mat.flatten()[min_loc]
        mixing_value = mix_mat.flatten()[min_loc]

        da_T_pred = np.nan*np.ones((nx, ))
        da_T_pred[has_data] = T_pred
        da_T_pred = da_T_pred.reshape((nlat, nlon))
        da_T_pred = da_gain.copy(data=da_T_pred)

        da_lam_inferred = np.nan*np.ones((nx, ))
        da_lam_inferred[has_data] = lam_value
        da_lam_inferred = da_lam_inferred.reshape((nlat, nlon))
        da_lam_inferred = da_gain.copy(data=da_lam_inferred)

        da_mix_inferred = np.nan*np.ones((nx, ))
        da_mix_inferred[has_data] = mixing_value
        da_mix_inferred = da_mix_inferred.reshape((nlat, nlon))
        da_mix_inferred = da_gain.copy(data=da_mix_inferred)

    return da_T_pred, da_lam_inferred, da_mix_inferred


def get_SMILE_forcing(models, savedir, nboot, seasonal_years):
    """Get SW forcing from SMILEs. Not all models have saved output; fill in with EM for other models."""

    savename_amp_phase = '%s/sw_net_amp_phase_SMILEs_%i-samples.nc' % (savedir, nboot)
    savename_sw_ts = '%s/sw_net_ts_SMILEs_%i-samples.nc' % (savedir, nboot)

    if os.path.isfile(savename_amp_phase) & os.path.isfile(savename_sw_ts):
        ds_seasonal_F_SMILES = xr.open_dataset(savename_amp_phase)
        sw_net_ts_SMILES = xr.open_dataarray(savename_sw_ts)
    else:

        missing_models = []
        ds_seasonal_F_SMILES = []
        sw_net_ts_SMILES = []
        for m in models:
            out = get_heating(m, nboot, seasonal_years, return_components=True)
            if (type(out) == int):
                missing_models.append(m)
            else:
                ds_seasonal_F_SMILES.append(out[0])
                # make sure all times match for sw_net
                # can have issue where different models use different day/time timestamps for the same
                # monthly average
                this_sw = out[1]
                start_year = this_sw['time.year'][0]
                start_month = this_sw['time.month'][0]
                new_time = pd.date_range(start='%04i-%02i-01' % (start_year, start_month),
                                         periods=len(this_sw.time), freq='M')
                this_sw = this_sw.assign_coords({'time': new_time})
                sw_net_ts_SMILES.append(this_sw)

        print('Models w/o SW up and SW down in MMLEA:')
        print(missing_models)
        ds_seasonal_F_SMILES = xr.concat(ds_seasonal_F_SMILES, dim='model')
        sw_net_ts_SMILES = xr.concat(sw_net_ts_SMILES, dim='model')
        ds_seasonal_F_SMILES['model'] = np.array(models)[~np.isin(models, missing_models)]
        sw_net_ts_SMILES['model'] = np.array(models)[~np.isin(models, missing_models)]

        EM_F = ds_seasonal_F_SMILES.mean('model')
        EM_SW = sw_net_ts_SMILES.mean('model')

        ds_EM = []
        sw_net_ts_EM = []
        for m in missing_models:
            ds_EM.append(EM_F)
            sw_net_ts_EM.append(EM_SW)

        ds_EM = xr.concat(ds_EM, dim='model')
        ds_EM['model'] = missing_models

        sw_net_ts_EM = xr.concat(sw_net_ts_EM, dim='model')
        sw_net_ts_EM['model'] = missing_models

        ds_seasonal_F_SMILES = xr.concat((ds_seasonal_F_SMILES, ds_EM), dim='model')
        sw_net_ts_SMILES = xr.concat((sw_net_ts_SMILES, sw_net_ts_EM), dim='model')

        ds_seasonal_F_SMILES.to_netcdf(savename_amp_phase)
        sw_net_ts_SMILES.to_netcdf(savename_sw_ts)

    return ds_seasonal_F_SMILES, sw_net_ts_SMILES

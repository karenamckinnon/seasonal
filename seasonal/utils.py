import numpy as np
from seasonal import constants
from climlab.solar.insolation import daily_insolation
import os
import xarray as xr


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
        # standardize ordering of dimensions
        da_data = da_data.transpose('month', 'lat', ' lon')

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

        ds_m.coords['mask'] = (('lat', 'lon'), ds_rho > 0.5)

        # combine all into one dataset
        ds_fit = xr.merge([da_T.to_dataset(name='T'),
                           lam.to_dataset(name='lam'),
                           ds_m.to_dataset(name='m'),
                           ds_rho.to_dataset(name='rho')])

        # save
        ds_fit.to_netcdf(savename)

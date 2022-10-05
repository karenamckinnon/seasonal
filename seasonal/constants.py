import numpy as np

# from CMIP5 mean estimates in Geoffroy et al 2013
rho = 1030  # kg/m3
cp = 4180  # J/kg/K
mwe = rho*cp
C_ocean = 20*mwe
C_od = 673*mwe
gamma = 0.74
lam_ocean = 1.13  # W/m2/K

# inferred from standard parameters for soil / standard diffusivity model
C_land = 0.75*mwe
seconds_per_day = 60*60*24
days_per_year = 365.25
omega = 2*np.pi/(days_per_year*seconds_per_day)

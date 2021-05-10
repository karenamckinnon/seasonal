import numpy as np

# from CMIP5 mean estimates in Geoffroy et al 2013
rho = 1030  # kg/m3
cp = 4180  # J/kg/K
mwe = rho*cp
C_land = 1*mwe
C_ocean = 54*mwe
C_od = 774*mwe
gamma = 0.74
seconds_per_day = 60*60*24
days_per_year = 365.25
omega = 2*np.pi/(days_per_year*seconds_per_day)

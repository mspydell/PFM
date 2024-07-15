# %%
# scratch code to get ocn forcing .nc files

#from datetime import datetime

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import matplotlib.pyplot as plt
#import numpy as np
#import scipy.ndimage as ndimage
#import xarray as xr
#import netCDF4 as nc
#from scipy.interpolate import RegularGridInterpolator


# %%


sys.path.append('../sdpm_py_util')

import atm_functions as atmfuns
import ocn_functions as ocnfuns
import grid_functions as grdfuns
import util_functions as utlfuns 
from util_functions import s_coordinate_4
from get_PFM_info import get_PFM_info
# row after setting suitable values for theta_b, theta_s, Tcline, Nz, hraw, eta, we could probably run the line:

PFM=get_PFM_info()

# %%
run_type = 'forecast'
# the year, month, day of the 
yyyymmdd='20240714'
# the hour in Z of the forecast, hycom has forecasts once per day starting at 1200Z
hhmm='1200'
# we will use hycom for IC and BC
ocn_mod = 'hycom'
# we will use nam_nest for the atm forcing
atm_mod = 'nam_nest'
# we will use opendap, and netcdf to grab ocn, and atm data
get_method = 'open_dap_nc'

# get the ROMS grid as a dict
#fngr = '/Users/mspydell/research/FF2024/models/SDPM_mss/PFM_user/grids/GRID_SDTJRE_LV1.nc'
RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

#import ipdb; ipdb.set_trace()

# %%
# make a switch to see if this file exists. If it exists, we don't need to run the code in this block
# first the atm data
# get the data as a dict
# need to specify hhmm because nam forecast are produced at 6 hr increments
ATM = atmfuns.get_atm_data_as_dict(yyyymmdd,hhmm,run_type,atm_mod,get_method)
# put in a function to check to make sure that all the data is good
# put in a function to plot the raw atm data if we want to
# put the atm data on the roms grid, and rotate the velocities
# everything in this dict turn into the atm.nc file
ATM_R  = atmfuns.get_atm_data_on_roms_grid(ATM,RMG)

# output a netcdf file of ATM_R
# make the atm .nc file here.
# fn_out is the name of the atm.nc file used by roms
fn_out = PFM['lv1_forc_dir'] + '/ATM_FORCING.nc'    #'/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/atm_test_file_v2.nc'
print(fn_out)
atmfuns.atm_roms_dict_to_netcdf(ATM_R,fn_out)
# put in a function to plot the atm.nc file if we want to

# %%
# make the ocn IC and BC .nc files here
# fn*_out are the names of the the IC.nc and BC.nc roms files
lv1_forc_dir = PFM['lv1_forc_dir']   #'/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/ocn_test_IC_file.nc'


# note, this function is hard wired to return 2.5 days of data
# also note that the first time of this data is yyyymmdd 12:00Z
# so we grab nam atm forecast data starting at this hour too.
OCN = ocnfuns.get_ocn_data_as_dict(yyyymmdd,run_type,ocn_mod,get_method)
# note this takes 24.5 minutes to run on my laptop
# 3 times this timed out
# will likely need to use a wget method and directly download .nc files (arh)
# maybe downloading the netcdf file would be quicker? 


# %%
# put the ocn data on the roms grid
OCN_R  = ocnfuns.hycom_to_roms_latlon(OCN,RMG)

# %%
# get the OCN_IC dictionary
OCN_IC = ocnfuns.ocn_r_2_ICdict(OCN_R,RMG)

# %%
# get the OCN_BC dictionary
#OCN_BC = ocnfuns.ocn_r_2_BCdict(OCN_R,RMG)

# %%
make_LV1_and_SLURM_dotin(PFM) 

#run_slurm_script



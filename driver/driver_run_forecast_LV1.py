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


sys.path.append('/Users/mspydell/research/FF2024/models/SDPM_mss/PFM/sdpm_py_util')

import atm_functions as atmfuns
import ocn_functions as ocnfuns
import grid_functions as grdfuns
import util_functions as utlfuns 
from util_functions import s_coordinate_4

# row after setting suitable values for theta_b, theta_s, Tcline, Nz, hraw, eta, we could probably run the line:



# %%
run_type = 'forecast'
# the year, month, day of the 
yyyymmdd='20240706'
# the hour in Z of the forecast, hycom has forecasts once per day starting at 1200Z
hhmm='1200'
# we will use hycom for IC and BC
ocn_mod = 'hycom'
# we will use nam_nest for the atm forcing
atm_mod = 'nam_nest'
# we will use opendap, and netcdf to grab ocn, and atm data
get_method = 'open_dap_nc'

# get the ROMS grid as a dict
fngr = '/Users/mspydell/research/FF2024/models/SDPM_mss/PFM_user/grids/GRID_SDTJRE_LV1.nc'
RMG = grdfuns.roms_grid_to_dict(fngr)



# %%

hraw = None
h = RMG['h']
eta = 0 * h
zrom = s_coordinate_4(h, 3.0 , 8.0 , 50.0 , 40, hraw=hraw, zeta=eta)


# %%
zr = zrom.z_r[0,:,:,:]

# %%
# make the atm .nc file here.
# fn_out is the name of the atm.nc file used by roms
fn_out = '/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/atm_test_file_v2.nc'
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
atmfuns.atm_roms_dict_to_netcdf(ATM_R,fn_out)
# put in a function to plot the atm.nc file if we want to

# %%
# make the ocn IC and BC .nc files here
# fn*_out are the names of the the IC.nc and BC.nc roms files
fn_ic_out = '/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/ocn_test_IC_file.nc'
fn_bc_out = '/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/ocn_test_BC_file.nc'

# note, this function is hard wired to return 2.5 days of data
# also note that the first time of this data is yyyymmdd 12:00Z
# so we grab nam atm forecast data starting at this hour too.
OCN = ocnfuns.get_ocn_data_as_dict(yyyymmdd,run_type,ocn_mod,get_method)
# note this takes 24.5 minutes to run on my laptop
# 3 times this timed out
# will likely need to use a wget method and directly download .nc files (arh)
# maybe downloading the netcdf file would be quicker? 


### should work to here! ####


# %%
# save the OCN dict so that we can restart the python session
# and not have to worry about opendap timing out
with open(fnout,'wb') as fp:
    pickle.dump(OCN,fp)
    print('OCN dict saved with pickle')

# %%
fnout='/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/ocn_hycom_dict_file.pkl'

with open(fnout,'rb') as fp:
    OCN = pickle.load(fp)


# %%
# put the ocn data on the roms grid
OCN_R  = ocnfuns.hycom_to_roms_latlon(OCN,RMG)


# %%
print(OCN_R.keys())

# %%
# get the OCN_IC dictionary
OCN_IC = ocnfuns.ocn_r_2_ICdict(OCN_R,RMG)

# %%
# get the OCN_BC dictionary
OCN_BC = ocnfuns.ocn_r_2_BCdict(OCN_R,RMG)

# %%
ilat = 100
ilon = 145
#print(OCN_R.keys())
print(np.shape(OCN_R['urm']))
dum = OCN_R['urm'][0,:,ilat,ilon]

#dum = OCN_R['ubar'][0,:,:]
dum2 = 0*dum
dum2[np.isnan(dum)==1] = 1
print(np.sum(dum2))
print(np.prod(dum.shape))

# %%
fig, ax = plt.subplots(nrows=1, ncols=2)
yy = OCN_R['depth'][:]
ilat = 100
ilon = 145

plevs=np.arange(0,4500,1)
cmap=plt.get_cmap('turbo')
plt.set_cmap(cmap)
cset1=ax[0].contourf(RMG['lon_rho'],RMG['lat_rho'],RMG['h'],plevs)
ax[0].plot(RMG['lon_rho'][ilat,ilon],RMG['lat_rho'][ilat,ilon],'wx')

xx = OCN_R['urm'][0,:,ilat,ilon]
ib = np.argwhere(np.isnan(xx))
ig = np.argwhere(np.isfinite(xx))
print(len(ib))
print(len(ig))

ax[1].plot(xx,-yy,'-o')
hrm = RMG['h'][ilat,ilon]
#hrm2 = np.max(xx)
#hrm1 = np.min(xx)
ax[1].plot([0,.25],[-hrm,-hrm],'--')
xx2 = OCN_R['ubar'][0,ilat,ilon]
ax[1].plot([0,0],[-yy[0],-yy[-1]])
ax[1].plot([xx2,xx2],[-yy[0],-yy[-1]],'-.')

hyz = OCN_R['depth']
igu = np.argwhere(hyz <= RMG['h'][ilat,ilon])
ax[1].plot(xx[igu],-yy[igu],'--')

dz = yy[1:]-yy[0:-1]
um = 0.5*( xx[1:] + xx[0:-1] )
ubar2 = np.sum(xx[igu]*dz[igu]) / yy[igu[-1]]
ubar3 = np.sum(xx[igu]*dz[igu]) / hrm
ubar4 = np.sum(um[igu]*dz[igu]) / hrm
print('the raw mean of the good vels is:')
print(np.mean(xx[igu]))
print('the depth average velocity calculated in ofun is:')
print(xx2)
print('simple depth average velocity is:')
print(ubar3)
print('midpoint formula depth avg vel is:')
print(ubar4)
print('looks like the ubar from ocnfun is good!!!')

#ib = np.argwhere(np.isnan(xx))
#ig = np.argwhere(np.isfinite(xx))

#print(np.shape(OCN_R['temp']))
#print(RMG['lat_rho'][ilat,ilon])
#print(RMG['lon_rho'][ilat,ilon])
#print(ib)
#print(yy[ib])
#print(RMG['h'][ilat,ilon])

#print(ig)



# %%
fig, ax = plt.subplots()
plevs=np.arange(-.2,.2,.001)
cmap=plt.get_cmap('turbo')
plt.set_cmap(cmap)
cset1=ax.contourf(RMG['lon_v'],RMG['lat_v'],OCN_R['vrm'][0,0,:,:],plevs)
cbar=fig.colorbar(cset1,ax=ax,orientation='vertical')




# %%
OCN_Rz = ocnfuns.ocn_r_hycomz_2_romsz(OCN_R,RMG)

# output OCN_R dict to roms IC
ocn_roms_dict_to_IC_netcdf
# ouput OCN_R dict to roms BC
ocn_roms_dict_to_BC_netcdf



# %%
make_roms_dotin
make_slurm_script
run_slurm_script



from datetime import datetime

import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
import pygrib
from pydap.client import open_url

sys.path.append('/opt/homebrew/Cellar/nceplibs-g2c/1.9.0/lib')
#base_url = 'https://www.ncei.noaa.gov/thredds/dodsC/namanl/'
#dt = datetime(2016, 4, 16, 18)
#data = xr.open_dataset('{}{dt:%Y%m}/{dt:%Y%m%d}/namanl_218_{dt:%Y%m%d}_'
#                       '{dt:%H}00_000.grb'.format(base_url, dt=dt),
#                       decode_times=True)

#url='nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.20240614/'
#fn_nam5km='nam.t00z.conusnest.hiresf00.tm00.grib2'
#fn_nomad = '/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/nam.t00z.awphys00.tm00-2.grib2'
#fn_nam5k = '/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/nam.t00z.conusnest.hiresf00.tm00.grib2'

# this file uses the grib filter at nomad to get a chunk of the data over our region of interest
# this file has: surface Pres [1], Temp [2], Precip [3], 
# swd, lwd, swu, lwu (average [4,5,6,7] and instant [8,9,10,11])
# need to get 10 m winds separately.
# the url is
# https://nomads.ncep.noaa.gov/cgi-bin/filter_nam_conusnest.pl?dir=%2Fnam.20240614&file=nam.t00z.conusnest.hiresf00.tm00.grib2&var_DLWRF=on&var_DSWRF=on&var_PRATE=on&var_PRES=on&var_RH=on&var_TMP=on&var_ULWRF=on&var_USWRF=on&lev_surface=on&subregion=&toplat=37.5&leftlon=-124.5&rightlon=-115.5&bottomlat=27.5
fn_nam5k = '/Users/mspydell/research/FF2024/models/SDPM_mss/atm_stuff/nam.t00z.conusnest.hiresf00.tm00-2.grib2'
fn_nomad=fn_nam5k
#print(fn_nomad)
#fn_nomad=url+fn_nam5km

# here is another way to get the data using pydap
dataset = open_url("https://nomads.ncep.noaa.gov/dods/nam/nam20240614/nam_conusnest_00z")
pres = dataset['pressfc']


#data = xr.open_dataset(fn_nomad, engine='cfgrib')
#data = xr.load_dataset(fn_nomad, engine='cfgrib')

sys.exit()
# one way to get data
data = pygrib.open(fn_nomad)
data.seek(0)
for mm in data:
    print(mm)
data.seek(0)

pres = data[1].values
temp = data[2].values
#uw   = data[657].values
#vw   = data[658].values
rain = data[3].values
swd_a = data[4].values
lwd_a = data[5].values
swu_a = data[6].values
lwu_a = data[7].values
swd_i = data[8].values
lwd_i = data[9].values
swu_i = data[10].values
lwu_i = data[11].values


Lt,Ln = data[1].latlons()
#print(data.t2m.values)
#print(np.shape(data.latitude.values))
#print(np.shape(data.r2.values))
#Ln=data.longitude.values
#Lt=data.latitude.values
#temp=data.t2m.values

# using xarray, but only gets 'instant' variables...
data=xr.open_dataset(fn_nomad, engine='cfgrib',filter_by_keys={'stepType': 'instant'})
# this might be good as you might not need to download the file?
# cfgrib is slow
# https://stackoverflow.com/questions/75314858/grib2-data-extraction-with-xarray-and-cfgrib-very-slow-how-to-improve-the-code
# maybe best to convert to netcdf first?


z=temp
zlevels = np.arange(np.floor(z.min()),np.ceil(z.max()), 1)
fig, axs = plt.subplots()
cmap = plt.get_cmap('bwr')
plt.set_cmap(cmap)
cset1 = axs.contourf(Ln,Lt,z,zlevels,cmap=plt.cm.turbo)
axs.set_title('nomads 5 km temp [K]')
axs.set_xlabel('lon')
axs.set_ylabel('lat')
fig.colorbar(cset1, ax=axs)

z=pres
zlevels = np.arange(np.floor(z.min()),np.ceil(z.max()), 1)
fig, axs = plt.subplots()
cmap = plt.get_cmap('bwr')
plt.set_cmap(cmap)
cset1 = axs.contourf(Ln,Lt,z,zlevels,cmap=plt.cm.turbo)
axs.set_title('nomads 5 km pres [Pa]')
axs.set_xlabel('lon')
axs.set_ylabel('lat')
fig.colorbar(cset1, ax=axs)

z=lwu_a
zlevels = np.arange(np.floor(z.min()),np.ceil(z.max()), 1)
fig, axs = plt.subplots()
cmap = plt.get_cmap('bwr')
plt.set_cmap(cmap)
cset1 = axs.contourf(Ln,Lt,z,zlevels,cmap=plt.cm.turbo)
axs.set_title('nomads 5 km long wave up avg [W/m2]')
axs.set_xlabel('lon')
axs.set_ylabel('lat')
fig.colorbar(cset1, ax=axs)

z=lwu_i
zlevels = np.arange(np.floor(z.min()),np.ceil(z.max()), 1)
fig, axs = plt.subplots()
cmap = plt.get_cmap('bwr')
plt.set_cmap(cmap)
cset1 = axs.contourf(Ln,Lt,z,zlevels,cmap=plt.cm.turbo)
axs.set_title('nomads 5 km long wave up instant [W/m2]')
axs.set_xlabel('lon')
axs.set_ylabel('lat')
fig.colorbar(cset1, ax=axs)


sys.exit()

z=rain
zlevels = np.arange(np.floor(z.min()),np.ceil(z.max()), 1)
fig, axs = plt.subplots()
cmap = plt.get_cmap('bwr')
plt.set_cmap(cmap)
cset1 = axs.contourf(Ln,Lt,z,zlevels,cmap=plt.cm.turbo)
axs.set_title('nomads 5 km rain [kg/m2/s]')
axs.set_xlabel('lon')
axs.set_ylabel('lat')
fig.colorbar(cset1, ax=axs)


g = grib2io.open(fn_nomad)
print(g)
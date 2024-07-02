"""
This is the one place where you set the path structure for the PFM.
The info is stored in the dict SDP.
All? paths are pathlib.Path objects.
This program is meant to be loaded as a module by Lfun which then adds more
entries to the Ldir dict based on which model run you are working on.
Users should copy this to PFM_user/get_sdpm_info.py and edit it appropriately.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# defaults that should work on all machines
parent = Path(__file__).absolute().parent.parent
LO = parent / 'PFM'
# this is where this set up script lives
LOu = parent / 'PFM_user'
# data is where the input files, atm, ocn IC, ocn BC, etc are found
data = parent / 'PFM_data'
# LOo is the location where his.nc files etc will go
LOo = parent / 'PFM_output'
# where are the grids??? We keep them in LO so that they come with
# git clone.
gridl1 = str(LOu) + '/grids/GRID_SDTJRE_LV1.nc'
gridl2 = str(LOu) + '/grids/GRID_SDTJRE_LV2.nc'
gridl3 = str(LOu) + '/grids/GRID_SDTJRE_LV3.nc'
gridl4 = str(LOu) + '/grids/GRID_SDTJRE_LV4.nc'
# what is the ocean / atm model used to force?
ocn_model = 'hycom'
atm_model = 'wrf'
# what is the time resolution of the models (in days)
daystep_ocn = 3/24
daystep_atm = 3/24
# this is the box that covers LV1 so we get only a rectangle of hycom data
# needs to be floats
latlonbox = [27.75, 37.25, -124.5+360, -115.5+360]

# roms will be run with this time step (in days)
daystep = 1

# add to the path
LOlo = str(LO)  + '/sdpm_py_util'
sys.path.append(LOlo)

testing = 0
if testing == 1:
    print('runs up to here in get_sdpm_info.py')
    sys.exit(0)

HOME = Path.home()
try:
    HOSTNAME = os.environ['HOSTNAME']
except KeyError:
    HOSTNAME = 'BLANK'
    
if str(HOME) == '/Users/mspydell':
    lo_env = 'mss_mac'
    roms_bin_lv1 = '/Users/mspydell/research/FF2024/ROMS_v0/Projects/Upwelling/romsS'
    roms_bin_lv2 = roms_bin_lv1
    roms_bin_lv3 = roms_bin_lv2
    roms_bin_lv4 = roms_bin_lv3

elif (str(HOME) == '/home/mspydell') & ('swell' in HOSTNAME):
    lo_env = 'mss_swell'
    roms_bin_lv1 = 'path_to_roms_binary_on_swell'
    roms_bin_lv2 = roms_bin_lv1
    roms_bin_lv3 = roms_bin_lv2
    roms_bin_lv4 = roms_bin_lv3

# this it the one place where the model time reference is set
modtime0 = datetime(1970,1,1,0,0)
# correct string for time units in ROMS forcing files
# see notes in Evernote, Run Log 9, 2020.10.06
roms_time_units = 'seconds since 1970-01-01 00:00:00'
# format used for naming day folders
ds_fmt = '%Y.%m.%d'
# number of forecast days
fdays = 3

# vertical stretching info for each grid are embedded in a dict
SS=dict()
SS['L1','Vtransform']=2                       # transformation equation
SS['L1','Vstretching'] = 4                    # stretching function
SS['L1','THETA_S'] = 8.0                      # surface stretching parameter
SS['L1','THETA_B'] = 3.0                      # bottom  stretching parameter
SS['L1','TCLINE'] = 50.0                      # critical depth (m)

SDP = dict()
SDP['lo_env'] = lo_env
SDP['parent'] = parent
SDP['LO'] = LO
SDP['LOo'] = LOo
SDP['LOu'] = LOu
SDP['data'] = data
SDP['rombinl1'] = roms_bin_lv1
SDP['rombinl2'] = roms_bin_lv2
SDP['rombinl3'] = roms_bin_lv3
SDP['rombinl4'] = roms_bin_lv4
SDP['modtime0'] = modtime0
SDP['roms_time_units'] = roms_time_units
SDP['ds_fmt'] = ds_fmt
SDP['forecast_days'] = fdays
SDP['gridl1'] = gridl1
SDP['gridl2'] = gridl2
SDP['gridl3'] = gridl3
SDP['gridl4'] = gridl4
SDP['daystep'] = daystep
SDP['daystep_ocn'] = daystep_ocn
SDP['daystep_atm'] = daystep_atm
SDP['ocn_model'] = ocn_model
SDP['atm_model'] = atm_model
SDP['latlonbox'] = latlonbox
SDP['stretching'] = SS
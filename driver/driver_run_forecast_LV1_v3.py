# -- driver_run_forecast_LV1.py  --
# master python script to do a full LV1 forecast simulation

import sys
import pickle
#import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timezone, timedelta
import gc
import resource
import subprocess
#import xarray as xr
#import netCDF4 as nc

import glob
import netCDF4
from netCDF4 import num2date, date2num, date2index
import matplotlib.dates as mdates

##############

sys.path.append('../sdpm_py_util')

import atm_functions as atmfuns
import ocn_functions as ocnfuns
import grid_functions as grdfuns
import util_functions as utlfuns 
import plotting_functions as pltfuns
from util_functions import s_coordinate_4
from get_PFM_info import get_PFM_info
from make_LV1_dotin_and_SLURM import make_LV1_dotin_and_SLURM
from run_slurm_LV1 import run_slurm_LV1

#import warnings
#warnings.filterwarnings("ignore")

print('\nStarting the LV1 simulation, Current time ', datetime.now())
print('\n')



run_type = 'forecast'
# we will use hycom for IC and BC
# we will use opendap, and netcdf to grab ocn, and atm data
get_method = 'open_dap_nc'
## figure out what the time is local and UTC
#start_time = datetime.now()
#utc_time = datetime.now(timezone.utc)
#year_utc = utc_time.year
#month_utc = utc_time.month
#day_utc = utc_time.day
#hour_utc = utc_time.hour

#fetch_time = datetime.now(timezone.utc) - timedelta(days=1)

#if hour_utc < 12:
#    fetch_time = datetime(fetch_time.year,fetch_time.month, fetch_time.day, 12) - timedelta(days=1)
#else:
#    fetch_time = datetime(fetch_time.year,fetch_time.month, fetch_time.day, 12)    
    




#$yyyymmdd = "%d%02d%02d" % (fetch_time.year, fetch_time.month, fetch_time.day)
    
#yyyymmdd = '20240717'
# the hour in Z of the forecast, hycom has forecasts once per day starting at 1200Z
#hhmm='1200'
#forecastZdatestr = yyyymmdd+hhmm+'Z'   # this could be used for model output to indicate when model was initialized.

## this is the hard coded 
#yyyymmdd = '20240817'


PFM=get_PFM_info()
print("Starting: driver_run_forecast_LV1: Current local Time =", PFM['start_time'], "UTC = ", PFM['utc_time'], ' Fetch time = ', PFM['fetch_time'])
yyyymmdd = PFM['yyyymmdd']
hhmm = PFM['hhmm']

print("Preparing forecast starting on",yyyymmdd,"at ",hhmm)
ocn_mod = PFM['ocn_model']
print('ocean boundary and initial conditions will be from:')
print(ocn_mod)
# we will use nam_nest for the atm forcing
atm_mod = PFM['atm_model']
print('atm forcing will be from:')
print(atm_mod)

# get the ROMS grid as a dict
RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
fn_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocn_tmp_pckl_file']
ocnIC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnIC_tmp_pckl_file']


print('before getting OCN, using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')

use_ncks = 1 # flag to get data using ncks. if =0, then a pre saved pickle file is loaded.
use_pckl_sav = 1
if use_ncks == 1:
    if use_pckl_sav == 0: # the original way that breaks when going to OCN_R
        OCN = ocnfuns.get_ocn_data_as_dict(yyyymmdd,run_type,ocn_mod,'ncks_para')
        print('driver_run_forecast_LV1: done with get_ocn_data_as_dict: Current time ',datetime.now() )
        print('after getting OCN with ncks_para, using:')
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('kilobytes')
    else:
        print('going to use subprocess, and save a pickle file.')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','ocn_functions.py','get_ocn_data_as_dict_pckl',yyyymmdd,run_type,ocn_mod,'ncks_para']
        ret1 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('hycom data saved with pickle, correctly?')
        print(ret1)
        print('0=yes,1=no')

        #with open(fn_pckl,'rb') as fp:
        #    OCN = pickle.load(fp)
        #    print('OCN dict now loaded with pickle')
        #    print('after getting OCN from file, using:')
        #    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        #    print('kilobytes')

else:
    save_ocn = 0 # if 0, this loads the pickle file. if 1, it saves the pickle file
    import pickle
    # save the OCN dict so that we can restart the python session
    # and not have to worry about opendap timing out
    fnout='/scratch/PFM_Simulations/LV1_Forecast/Forc/ocn_dict_file_2024-07-29T12:00.pkl'
    if save_ocn == 1:
        with open(fnout,'wb') as fp:
            pickle.dump(OCN,fp)
            print('OCN dict saved with pickle')
    else:
        fn_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocn_tmp_pckl_file']
        with open(fn_pckl,'rb') as fp:
            #OCN = pickle.load(fp)
            print('OCN dict loaded with pickle')
            print('after getting OCN from file, using:')
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            print('kilobytes')


## plot OCN
plot_ocn = 1
if plot_ocn ==1:
    pltfuns.plot_ocn_fields_from_dict_pckl(fn_pckl)




print('before gc.collect and getting OCN_R, using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')
gc.collect()
print('after gc.collect and before OCN_R, using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')
# put the ocn data on the roms grid
print('starting: ocnfuns.hycom_to_roms_latlon(OCN,RMG)')
sv_ocnR_pkl_file=1
if sv_ocnR_pkl_file==0:
    print('returning OCN_R')
    OCN_R  = ocnfuns.hycom_to_roms_latlon(OCN,RMG)
else:
    #print('going to save OCN_R to temp pickle files.')
    os.chdir('../sdpm_py_util')
    #cmd_list = ['python','ocn_functions.py','hycom_to_roms_latlon_pckl',fn_pckl]
    print('\nputting the hycom data in ' + fn_pckl + ' on the roms grid')
    ocnfuns.make_all_tmp_pckl_ocnR_files(fn_pckl)

    #print(cmd_list)
    #ret2 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    #print('OCN_R data saved to pickle files, correctly?')
    #print(ret2)
    #print('0=yes,1=no')

print('\ndriver_run_forecast_LV1: done with hycom_to_roms_latlon')
# add OCN + OCN_R plotting function here !!!
print('using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')


## plot OCN_R
plot_ocnr = 1
if plot_ocnr == 1:
    pltfuns.plot_ocn_R_fields_pckl()


print('using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')
# %%
# get the OCN_IC dictionary

# make the depth pickle file
print('making the depth pickle file...')
fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']
cmd_list = ['python','ocn_functions.py','make_rom_depths',fname_depths]
os.chdir('../sdpm_py_util')
ret6 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('subprocess return code? ' + str(ret6.returncode) +  ' (0=good)')
print('...done.')


fr_ocnR_pkl_file=1
if fr_ocnR_pkl_file==0:
    OCN_IC = ocnfuns.ocn_r_2_ICdict(OCN_R,RMG,PFM)
else:
    print('going to save OCN_IC to a pickle file to:')
    print(ocnIC_pckl) 
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','ocn_functions.py','ocn_r_2_ICdict_pckl',ocnIC_pckl]
    ret3 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN IC data saved with pickle, correctly?')
    print(ret3)
    print('0=yes,1=no')



print('driver_run_forecast_LV1: done with ocn_r_2_ICdict')
# add OCN_IC.nc plotting function here !!!!
print('using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')

print('using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')

ic_file_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ini_file']

frm_ICpkl_file = 1
if frm_ICpkl_file == 0:
    print('making IC file: '+ ic_file_out)
    ocnfuns.ocn_roms_IC_dict_to_netcdf(OCN_IC, ic_file_out)
    print('done makeing IC file.')
else:
    print('making IC file from pickled IC: '+ ic_file_out)
    cmd_list = ['python','ocn_functions.py','ocn_roms_IC_dict_to_netcdf_pckl',ocnIC_pckl,ic_file_out]
    os.chdir('../sdpm_py_util')
    ret4 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN IC nc data saved, correctly?')
    print(ret4)
    print('0=yes,1=no')
    print('done makeing IC .nc file.')

print('using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')

## plot OCN_R 
# this code plots the bottom not top.
plot_ocn_icnc= 1
if plot_ocn_icnc == 1:
    pltfuns.plot_ocn_ic_fields(ic_file_out,RMG,PFM)


print('using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')
# %%
# get the OCN_BC dictionary

fr_ocnR_pkl_file=1
if fr_ocnR_pkl_file==0:
    OCN_BC = ocnfuns.ocn_r_2_BCdict(OCN_R,RMG,PFM)
else:
    print('going to save OCN_BC to a pickle file to:')
    ocnBC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']
    print(ocnBC_pckl) 
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','ocn_functions.py','ocn_r_2_BCdict_pckl_new',ocnBC_pckl]
    ret4 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN BC data saved with pickle, correctly?')
    print(ret4.returncode)
    print('0=yes,1=no')
    
print('using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')

bc_file_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_bc_file']
ocnBC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']

frm_BCpkl_file = 1
if frm_BCpkl_file == 0:
    print('making BC file: '+ bc_file_out)
    ocnfuns.ocn_roms_BC_dict_to_netcdf(OCN_BC, bc_file_out)
    print('done makeing BC nc file.')
else:
    print('making BC nc file from pickled BC: '+ bc_file_out)
    cmd_list = ['python','ocn_functions.py','ocn_roms_BC_dict_to_netcdf_pckl',ocnBC_pckl,bc_file_out]
    os.chdir('../sdpm_py_util')
    ret5 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN BC nc data saved, correctly?')
    print(ret5)
    print('0=yes,1=no')
    print('done makeing IC .nc file.')

print('using:')
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print('kilobytes')

# get the OCN_BC dictionary

print('driver_run_forecast_LV1: done with ocn_r_2_BCdict')

# make a switch to see if this file exists. If it exists, we don't need to run the code in this block
# first the atm data
# get the data as a dict
# need to specify hhmm because nam forecast are produced at 6 hr increments
ATM = atmfuns.get_atm_data_as_dict(yyyymmdd,hhmm,run_type,atm_mod,'open_dap_nc',PFM)
# put in a function to check to make sure that all the data is good
# put in a function to plot the raw atm data if we want to


# plot some stuff
pltfuns.plot_atm_fields(ATM, RMG, PFM)
print('done with plotting ATM fields')


# put the atm data on the roms grid, and rotate the velocities
# everything in this dict turn into the atm.nc file

ATM_R  = atmfuns.get_atm_data_on_roms_grid(ATM,RMG)
print('done with: atmfuns.get_atm_data_on_roms_grid(ATM,RMG)')
# all the fields plotted with the data on roms grid


pltfuns.plot_all_fields_in_one(ATM, ATM_R, RMG, PFM)
print('done with: pltfuns.plot_all_fields_in_one(ATM, ATM_R, RMG, PFM)')

# output a netcdf file of ATM_R
# make the atm .nc file here.
# fn_out is the name of the atm.nc file used by roms
fn_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_atm_file'] # LV1 atm forcing filename
print('driver_run_forcast_LV1: saving ATM file to ' + fn_out)
atmfuns.atm_roms_dict_to_netcdf(ATM_R,fn_out)
print('driver_run_forecast_LV1:  done with writing ATM file, Current time ', datetime.now())
# put in a function to plot the atm.nc file if we want to
pltfuns.load_and_plot_atm(RMG, PFM)
print('done with pltfuns.load_and_plot_atm(PFM)')


print('driver_run_forecast_LV1:  now make .in and .sb files')

pfm_driver_src_dir = os.getcwd()

os.chdir('../sdpm_py_util')
make_LV1_dotin_and_SLURM( PFM , yyyymmdd + hhmm )
print('...done')

# run command will be
print('now running roms with slurm')
run_slurm_LV1(PFM)

os.chdir('../driver')

print('\nFinished the LV1 simulation, Current time ', datetime.now())

# now making history file plots
print('now making LV1 history file plots')

fn = PFM['lv1_his_dir'] + '/LV1_ocean_his.nc'
Ix = np.array([175,240])
Iy = np.array([175,170])
It = 0
sv_fig = 1

pltfuns.plot_roms_LV1_bathy_and_locs(fn,Ix,Iy,sv_fig)
pltfuns.plot_ssh_his_tseries(fn,Ix,Iy,sv_fig)
pltfuns.plot_his_temps_wuv(fn,It,-1,sv_fig)
pltfuns.plot_his_temps_wuv(fn,It+24,-1,sv_fig)  # 24h forecast
pltfuns.plot_his_temps_wuv(fn,It+48,-1,sv_fig)
pltfuns.plot_his_temps_wuv(fn,It+60,-1,sv_fig)  # 60 h forecast

print('\n')
#######################


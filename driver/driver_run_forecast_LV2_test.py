# -- driver_run_forecast_LV1.py  --
# master python script to do a full LV1 forecast simulation

import sys
import pickle
import numpy as np
import os
from datetime import datetime
import gc
import resource
import subprocess

##############

sys.path.append('../sdpm_py_util')

import atm_functions as atmfuns
import ocn_functions as ocnfuns
import grid_functions as grdfuns
import plotting_functions as pltfuns
from get_PFM_info import get_PFM_info
from make_LV2_dotin_and_SLURM import make_LV2_dotin_and_SLURM
from run_slurm_LV2 import run_slurm_LV2
from init_funs import initialize_simulation

print('\n Testing  LV2 simulation, Current time ', datetime.now())

# we are going to make a forecast
run_type = 'forecast'

# PFM has all of the information needed to run the model
clean_start = True
initialize_simulation(clean_start) # this removes the PFM_info.pkl file if clean_start = True
PFM=get_PFM_info()
RMG = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])

print("Starting: driver_run_forecast_LV2")
print("Current local Time =", PFM['start_time'], "UTC = ", PFM['utc_time'], ' Fetch time = ', PFM['fetch_time'])
yyyymmdd = PFM['yyyymmdd']
hhmm = PFM['hhmm']

print("\nPreparing forecast starting on",yyyymmdd,"at ",hhmm)
ocn_mod = PFM['ocn_model']
#print('ocean boundary and initial conditions will be from:')
#print(ocn_mod)

#atm_mod = PFM['atm_model']
#print('atm forcing will be from:')
#print(atm_mod)

print('driver_run_forecast_LV2:  now make .in and .sb files')

pfm_driver_src_dir = os.getcwd()

os.chdir('../sdpm_py_util')
make_LV2_dotin_and_SLURM( PFM , yyyymmdd + hhmm )
print('...done')

# run command will be
#print('now running roms with slurm')
#run_slurm_LV1(PFM)

os.chdir('../driver')

t12 = datetime.now()
print('this took:')
print(t12-t11)
print('\n')

# now making history file plots
#print('\nFinished the LV1 simulation')
#print('now making LV1 history file plots')
#pltfuns.make_all_his_figures('LV1')
#print('Current time: ', datetime.now())

#t13 = datetime.now()
#print('total time to run script was:')
#print(t13-t0)
#print('\n')

#######################


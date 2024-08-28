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
from make_LV1_dotin_and_SLURM import make_LV1_dotin_and_SLURM
from run_slurm_LV1 import run_slurm_LV1
from init_funs import initialize_simulation

print('\nStarting the LV1 simulation, Current time ', datetime.now())

# we are going to make a forecast
run_type = 'forecast'

# PFM has all of the information needed to run the model
clean_start = True
initialize_simulation(clean_start) # this removes the PFM_info.pkl file if clean_start = True
PFM=get_PFM_info()
RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

print("Starting: driver_run_forecast_LV1")
print("Current local Time =", PFM['start_time'], "UTC = ", PFM['utc_time'], ' Fetch time = ', PFM['fetch_time'])
yyyymmdd = PFM['yyyymmdd']
hhmm = PFM['hhmm']

print("\nPreparing forecast starting on",yyyymmdd,"at ",hhmm)
ocn_mod = PFM['ocn_model']
print('ocean boundary and initial conditions will be from:')
print(ocn_mod)

atm_mod = PFM['atm_model']
print('atm forcing will be from:')
print(atm_mod)

# we are going to use ncks and save pickle files, set flags to run
use_ncks = 1 # flag to get data using ncks. if =0, then a pre saved pickle file is loaded.
use_pckl_sav = 1
sv_ocnR_pkl_file=1
fr_ocnR_pkl_file=1
frm_ICpkl_file = 1
frm_BCpkl_file = 1
fr_ocnR_pkl_file=1

# what are we going to plot?
plot_ocn = 1
plot_ocnr = 1
plot_atm = 1
plot_all_atm = 1
plot_ocn_icnc= 1
load_plot_atm= 1

# need the file names and locations of the pickle and .nc files we will save
fn_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocn_tmp_pckl_file']
ocnIC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnIC_tmp_pckl_file']
ic_file_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ini_file']
bc_file_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_bc_file']
ocnBC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']
fn_atm_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_atm_file'] # LV1 atm forcing filename

#print('before getting OCN, using:')
#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#print('kilobytes')

t0 = datetime.now()
if use_ncks == 1:
    if use_pckl_sav == 0: # the original way that breaks when going to OCN_R
        OCN = ocnfuns.get_ocn_data_as_dict(yyyymmdd,run_type,ocn_mod,'ncks_para')
        print('driver_run_forecast_LV1: done with get_ocn_data_as_dict: Current time ',datetime.now() )
    else:
        print('\nGetting OCN forecast data. Going to use subprocess, and save a pickle file of ocn data.')
        os.chdir('../sdpm_py_util')
        #cmd_list = ['python','ocn_functions.py','get_ocn_data_as_dict_pckl',yyyymmdd,run_type,ocn_mod,'ncks_para']
     #-W "ignore"
        cmd_list = ['python','-W','ignore','ocn_functions.py','get_ocn_data_as_dict_pckl',yyyymmdd,run_type,ocn_mod,'ncks_para']
        ret1 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('did subprocess run correctly? ' + str(ret1.returncode) + ' (0=yes,1=no)')
else: # this was used for testing and loading a previous pickle file 
    save_ocn = 0 # if 0, this loads the pickle file. if 1, it saves the pickle file
    import pickle
    fnout='/scratch/PFM_Simulations/LV1_Forecast/Forc/ocn_dict_file_2024-07-29T12:00.pkl'
    if save_ocn == 1:
        with open(fnout,'wb') as fp:
            pickle.dump(OCN,fp)
            print('OCN dict saved with pickle')
    else:
        with open(fn_pckl,'rb') as fp:
            OCN = pickle.load(fp)
            #print('OCN dict loaded with pickle')

## plot OCN

if plot_ocn ==1:
    pltfuns.plot_ocn_fields_from_dict_pckl(fn_pckl)

gc.collect()

t1= datetime.now()
print('this took:')
print(t1-t0)
print('\n')


# put the ocn data on the roms grid
print('starting: ocnfuns.hycom_to_roms_latlon(OCN,RMG)')
if sv_ocnR_pkl_file==0:
    print('returning OCN_R')
    OCN_R  = ocnfuns.hycom_to_roms_latlon(OCN,RMG)
else:
    os.chdir('../sdpm_py_util')
    print('putting the hycom data in ' + fn_pckl + ' on the roms grid...')
    ocnfuns.make_all_tmp_pckl_ocnR_files(fn_pckl)
    os.chdir('../driver')

ocnfuns.print_maxmin_HYrm_pickles()
print('driver_run_forecast_LV1: done with hycom_to_roms_latlon')
# add OCN + OCN_R plotting function here !!!

## plot OCN_R
if plot_ocnr == 1:
    pltfuns.plot_ocn_R_fields_pckl()

t2 = datetime.now()
print('this took:')
print(t2-t1)
print('\n')


# make the depth pickle file
print('making the depth pickle file...')
fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']
cmd_list = ['python','-W','ignore','ocn_functions.py','make_rom_depths',fname_depths]
os.chdir('../sdpm_py_util')
ret6 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('subprocess return code? ' + str(ret6.returncode) +  ' (0=good)')
t3 = datetime.now()
print('this took:')
print(t3-t2)
print('\n')



if fr_ocnR_pkl_file==0:
    OCN_IC = ocnfuns.ocn_r_2_ICdict(OCN_R,RMG,PFM)
else:
    print('going to save OCN_IC to a pickle file: ' + ocnIC_pckl)
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_r_2_ICdict_pckl',ocnIC_pckl]
    ret3 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN IC data saved with pickle, correctly? ' + str(ret3.returncode) + ' (0=yes,1=no)')

print('driver_run_forecast_LV1: done with ocn_r_2_ICdict')
# add OCN_IC.nc plotting function here !!!!
t4 = datetime.now()
print('this took:')
print(t4-t3)
print('\n')


if frm_ICpkl_file == 0:
    print('making IC file: '+ ic_file_out)
    ocnfuns.ocn_roms_IC_dict_to_netcdf(OCN_IC, ic_file_out)
else:
    print('making IC file from pickled IC: '+ ic_file_out)
    cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_IC_dict_to_netcdf_pckl',ocnIC_pckl,ic_file_out]
    os.chdir('../sdpm_py_util')
    ret4 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN IC nc data saved, correctly? ' + str(ret4.returncode) + ' (0=yes)')

print('done makeing IC file.')

if plot_ocn_icnc == 1:
    pltfuns.plot_ocn_ic_fields(ic_file_out)

t5 = datetime.now()
print('this took:')
print(t5-t4)
print('\n')

# get the OCN_BC dictionary
if fr_ocnR_pkl_file==0:
    OCN_BC = ocnfuns.ocn_r_2_BCdict(OCN_R,RMG,PFM)
else:
    print('going to save OCN_BC to a pickle file to:')
    ocnBC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']
    print(ocnBC_pckl) 
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_r_2_BCdict_pckl_new',ocnBC_pckl]
    ret4 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN BC data saved with pickle, correctly? ' + str(ret4.returncode) + ' (0=yes)')
    
t6 = datetime.now()
print('this took:')
print(t6-t5)
print('\n')


if frm_BCpkl_file == 0:
    print('making BC file: '+ bc_file_out)
    ocnfuns.ocn_roms_BC_dict_to_netcdf(OCN_BC, bc_file_out)
else:
    print('making BC nc file from pickled BC: '+ bc_file_out)
    cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_BC_dict_to_netcdf_pckl',ocnBC_pckl,bc_file_out]
    os.chdir('../sdpm_py_util')
    ret5 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN BC nc data saved, correctly? ' + str(ret5.returncode) + ' (0=yes)')

print('done makeing BC nc file.')
t7 = datetime.now()
print('this took:')
print(t7-t6)
print('\n')


# get atm data as a dict
# need to specify hhmm because nam forecast are produced at 6 hr increments
print('getting the atm data now...')
#ATM = atmfuns.get_atm_data_as_dict(yyyymmdd,hhmm,run_type,atm_mod,'open_dap_nc',PFM)
cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_as_dict']
os.chdir('../sdpm_py_util')
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('...done.')
t8 = datetime.now()
print('this took:')
print(t8-t7)
print('\n')


# plot some stuff
if plot_atm == 1:
    pltfuns.plot_atm_fields()
    print('done with plotting ATM fields')

t9 = datetime.now()
print('this took:')
print(t9-t8)
print('\n')

level = 1
# put the atm data on the roms grid, and rotate the velocities
# everything in this dict turn into the atm.nc file
print('in atmfuns.get_atm_data_on_roms_grid(ATM,RMG)')
#ATM_R  = atmfuns.get_atm_data_on_roms_grid(ATM,RMG)
cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_on_roms_grid',str(level)]
os.chdir('../sdpm_py_util')
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('done with: atmfuns.get_atm_data_on_roms_grid(ATM,RMG)')
# all the fields plotted with the data on roms grid

if plot_all_atm == 1:
#    pltfuns.plot_all_fields_in_one(ATM, ATM_R, RMG, PFM)
    pltfuns.plot_all_fields_in_one(str(level))
    print('done with: pltfuns.plot_all_fields_in_one(ATM, ATM_R, RMG, PFM)')

t10 = datetime.now()
print('this took:')
print(t10-t9)
print('\n')

# fn_out is the name of the atm.nc file used by roms
print('driver_run_forcast_LV1: saving ATM file to ' + fn_atm_out)
#atmfuns.atm_roms_dict_to_netcdf(ATM_R,fn_atm_out)
cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level)]
os.chdir('../sdpm_py_util')
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_LV1:  done with writing ATM.nc file.') 
# put in a function to plot the atm.nc file if we want to

if load_plot_atm == 1:
    pltfuns.load_and_plot_atm(str(level))
    print('done with pltfuns.load_and_plot_atm(PFM)')

t11 = datetime.now()
print('this took:')
print(t11-t10)
print('\n')

print('driver_run_forecast_LV1:  now make .in and .sb files')

pfm_driver_src_dir = os.getcwd()

os.chdir('../sdpm_py_util')
make_LV1_dotin_and_SLURM( PFM , yyyymmdd + hhmm )
print('...done')

# run command will be
print('now running roms with slurm')
run_slurm_LV1(PFM)

os.chdir('../driver')

t12 = datetime.now()
print('this took:')
print(t12-t11)
print('\n')

# now making history file plots
print('\nFinished the LV1 simulation')
print('now making LV1 history file plots')
pltfuns.make_all_his_figures('LV1')
print('Current time: ', datetime.now())

t13 = datetime.now()
print('total time to run script was:')
print(t13-t0)
print('\n')

#######################


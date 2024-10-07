# -- driver_run_forecast_LV1.py  --
# master python script to do a full LV1 forecast simulation

import sys
import os
from datetime import datetime, timedelta
import subprocess

##############

sys.path.append('../sdpm_py_util')

import ocn_functions as ocnfuns
import plotting_functions as pltfuns
from get_PFM_info import get_PFM_info
import init_funs as infuns
from make_LV1_dotin_and_SLURM import make_LV1_dotin_and_SLURM
from run_slurm_LV1 import run_slurm_LV1

print('\nStarting the LV1 simulation, Current time ', datetime.now())

t00 = datetime.now() # for keeping track of timing of the simulation
# we are going to make a forecast
run_type = 'forecast'

# PFM has all of the information needed to run the model
clean_start = True
infuns.initialize_simulation(clean_start)
PFM=get_PFM_info()

# return the hycom forecast date based on the PFM simulation times
#yyyymmdd_hy = infuns.determine_hycom_foretime() 

t1  = PFM['fetch_time']                              # this is the first time of the PFM forecast
# now a string of the time to start ROMS (and the 1st atm time too)
yyyymmddhhmm_pfm = "%d%02d%02d%02d%02d" % (t1.year, t1.month, t1.day, t1.hour, t1.minute)
t2  = t1 + PFM['forecast_days'] * timedelta(days=1)  # this is the last time of the PFM forecast

print('\nGoing to do a PFM forecast from')
print(t1)
print('to')
print(t2)
print('\n')

t2str = "%d%02d%02d%02d%02d" % (t2.year, t2.month, t2.day, t2.hour, t2.minute)
# get_hycom_foretime is a function that returns the lates hycom forecast date with data to span the PFM forecast from 
# this functions cleans, then refreshes the hycom directory.
print('getting the hycom forecast time...')
print('this will clean, download new hycom files, then find the forecast date covering the PFM times.')
t01 = datetime.now()
yyyymmdd_hy = ocnfuns.get_hycom_foretime(yyyymmddhhmm_pfm,t2str)

# now make the catted hycom data .nc file
#ocnfuns.cat_hycom_to_onenc(yyyymmdd_hy,[t1,t2])
print('\n\nwe will use the hycom forecast from')
print(yyyymmdd_hy)
print('for the PFM forecast.')
print('...done.')
t02 = datetime.now() # for keeping track of timing of the simulation
print('this took:')
print(t02-t01)
print('\n')

""" print('catting the hycom.nc files into two (1hr and 3hr) nc files ...')
t01 = datetime.now() # for keeping track of timing of the simulation
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','cat_hycom_to_twonc_1hr',yyyymmdd_hy,yyyymmddhhmm_pfm,t2str]
ret1 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('did subprocess run correctly? ' + str(ret1.returncode) + ' (0=yes,1=no)')
#ocnfuns.cat_hycom_to_onenc(yyyymmdd_hy,yyyymmddhhmm_pfm,t2str)
print('...done.')
t02 = datetime.now() # for keeping track of timing of the simulation
print('this took:')
print(t02-t01)
print('\n')
 """

""" print('making the hycom pickle file from 1hr and 3hr hycom.nc ...')
t01 = datetime.now() # for keeping track of timing of the simulation
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','hycom_cats_to_pickle',yyyymmdd_hy]
ret1 = subprocess.run(cmd_list)     
#ocnfuns.hycom_cat_to_pickles(yyyymmdd_hy)
os.chdir('../driver')
print('did subprocess run correctly? ' + str(ret1.returncode) + ' (0=yes,1=no)')
#ocnfuns.cat_hycom_to_onenc(yyyymmdd_hy,yyyymmddhhmm_pfm,t2str)
print('...done.')
t02 = datetime.now() # for keeping track of timing of the simulation
print('this took:')
print(t02-t01)
print('\n') """

print('making the hycom pickle file from all hycom.nc files ...')
t01 = datetime.now() 
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','hycom_ncfiles_to_pickle',yyyymmdd_hy]
ret1 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('did subprocess run correctly? ' + str(ret1.returncode) + ' (0=yes,1=no)')
print('...done.')
t02 = datetime.now() 
print('this took:')
print(t02-t01)
print('\n')


# now we are back to the previous code...
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


t01 = datetime.now()
if plot_ocn ==1:
    print('making some plots from: ' + fn_pckl)
    cmd_list = ['python','-W','ignore','plotting_functions.py','plot_ocn_fields_from_dict_pckl',fn_pckl]
    os.chdir('../sdpm_py_util')
    ret1 = subprocess.run(cmd_list)     
    #pltfuns.plot_ocn_fields_from_dict_pckl(fn_pckl)
    print('subprocess return code? ' + str(ret1.returncode) +  ' (0=good)')
    print('...done')
    os.chdir('../driver')

t02= datetime.now()
print('this took:')
print(t02-t01)
print('\n')



# put the ocn data on the roms grid
print('starting: ocnfuns.hycom_to_roms_latlon(OCN,RMG)')
t01 = datetime.now()

os.chdir('../sdpm_py_util')
print('putting the hycom data in ' + fn_pckl + ' on the roms grid...')
cmd_list = ['python','-W','ignore','ocn_functions.py','make_all_tmp_pckl_ocnR_files_1hrzeta',fn_pckl]
os.chdir('../sdpm_py_util')
ret1 = subprocess.run(cmd_list)     
#ocnfuns.make_all_tmp_pckl_ocnR_files(fn_pckl)
os.chdir('../driver')
print('subprocess return code? ' + str(ret1.returncode) +  ' (0=good)')


cmd_list = ['python','-W','ignore','ocn_functions.py','print_maxmin_HYrm_pickles']
os.chdir('../sdpm_py_util')
ret1 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('driver_run_forecast_LV1: done with hycom_to_roms_latlon')

if plot_ocnr == 1:
    print('plotting LV1 ocn_R_fields...')
    cmd_list = ['python','-W','ignore','plotting_functions.py','plot_ocn_R_fields_pckl']
    os.chdir('../sdpm_py_util')
    ret1 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('subprocess return code? ' + str(ret1.returncode) +  ' (0=good)')

t02 = datetime.now()
print('...done with LV1 ocn_R')
print('this took:')
print(t02-t01)
print('\n')



# make the depth pickle file
print('making the depth pickle file...')
fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']
cmd_list = ['python','-W','ignore','ocn_functions.py','make_rom_depths_1hrzeta',fname_depths]
os.chdir('../sdpm_py_util')
ret6 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('subprocess return code? ' + str(ret6.returncode) +  ' (0=good)')
print('\n')


print('going to save OCN_IC to a pickle file: ' + ocnIC_pckl)
t01 = datetime.now()
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','ocnr_2_ICdict_from_tmppkls',ocnIC_pckl]
ret3 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('OCN IC data saved with pickle, correctly? ' + str(ret3.returncode) + ' (0=yes,1=no)')

print('driver_run_forecast_LV1: done with ocn_r_2_ICdict')
t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')


print('making IC file from pickled IC: '+ ic_file_out)
t01 = datetime.now()
cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_IC_dict_to_netcdf_pckl',ocnIC_pckl,ic_file_out]
os.chdir('../sdpm_py_util')
ret4 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('OCN IC nc data saved, correctly? ' + str(ret4.returncode) + ' (0=yes)')

print('done makeing IC file.')

if plot_ocn_icnc == 1:
    pltfuns.plot_ocn_ic_fields(ic_file_out)

t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')

# get the OCN_BC dictionary
print('going to save OCN_BC to a pickle file to:')
t01 = datetime.now()
ocnBC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']
print(ocnBC_pckl) 
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','ocnr_2_BCdict_1hrzeta_from_tmppkls',ocnBC_pckl]
ret4 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('OCN BC data saved with pickle, correctly? ' + str(ret4.returncode) + ' (0=yes)')
    
t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')


print('making BC nc file from pickled BC: '+ bc_file_out)
t01 = datetime.now()
cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_BC_dict_to_netcdf_pckl_1hrzeta',ocnBC_pckl,bc_file_out]
os.chdir('../sdpm_py_util')
ret5 = subprocess.run(cmd_list)     
os.chdir('../driver')
print('OCN BC nc data saved, correctly? ' + str(ret5.returncode) + ' (0=yes)')

print('done makeing BC nc file.')
t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')


# now for the atm part...
print('we are now getting the atm data and saving as a dict...')
t01 = datetime.now()
cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_as_dict']
os.chdir('../sdpm_py_util')
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('...done.')
t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')

# plot some stuff
if plot_atm == 1:
    print('we are now plotting the atm data...')
    t01 = datetime.now()
    pltfuns.plot_atm_fields()
    print('...done with plotting ATM fields')

t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')

level = 1
# put the atm data on the roms grid, and rotate the velocities
# everything in this dict turn into the atm.nc file
print('we are now putting the atm data on the roms LV1 grid...')
t01 = datetime.now()
cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_on_roms_grid',str(level)]
os.chdir('../sdpm_py_util')
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('...done.')
# all the fields plotted with the data on roms grid
t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')

if plot_all_atm == 1:
    t01 = datetime.now()
    print('we are now plotting the atm data on roms grid...')
    pltfuns.plot_all_fields_in_one(str(level))
    print('...done.')

t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')

# fn_out is the name of the atm.nc file used by roms
print('we are now saving ATM LV1 to ' + fn_atm_out + ' ...')
t01 = datetime.now()
cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level)]
os.chdir('../sdpm_py_util')
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('...done.') 
# put in a function to plot the atm.nc file if we want to
t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')

if load_plot_atm == 1:
    t01 = datetime.now()
    print('we are now plotting the atm data...')
    pltfuns.load_and_plot_atm(str(level))
    print('...done.')

t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')


print('driver_run_forecast_LV1:  now make .in and .sb files...')
pfm_driver_src_dir = os.getcwd()
yyyymmdd = PFM['yyyymmdd']
hhmm = PFM['hhmm']
os.chdir('../sdpm_py_util')
make_LV1_dotin_and_SLURM( PFM , yyyymmdd + hhmm )
print('...done.\n')

# run command will be
print('now running roms LV1 with slurm.')
print('using ' + str(PFM['gridinfo']['L1','nnodes']) + ' nodes.')
print('Ni = ' + str(PFM['gridinfo']['L1','ntilei']) + ', NJ = ' + str(PFM['gridinfo']['L1','ntilej']))
print('working...')
t01 = datetime.now()
run_slurm_LV1(PFM)
print('...done.')
os.chdir('../driver')
t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')

# now making history file plots
print('now making LV1 history file plots...')
t01 = datetime.now()
pltfuns.make_all_his_figures('LV1')
print('...done.')
t02 = datetime.now()
print('this took:')
print(t02-t01)
print('\n')

print('\nFinished the LV1 simulation')
print('this took:')
print(t02-t00)
print('\n')



#######################



# -- driver_run_forecast_LV2_v1.py  --
# master python script to do a full LV2 forecast simulation

import sys
import os
from datetime import datetime
import subprocess

##############

sys.path.append('../sdpm_py_util')
#from init_funs import remake_PFM_pkl_file
from get_PFM_info import get_PFM_info
#from init_funs import initialize_simulation
import plotting_functions as pltfuns
from make_LV2_dotin_and_SLURM import make_LV2_dotin_and_SLURM
from run_slurm_LV2 import run_slurm_LV2

# used for various calls later
print('now starting the LV2 simulation...')
print("Current local Time =")
print(datetime.now())
level = 2
#remake_PFM_pkl_file() # this function will remake the PFM.pkl file. Good for developing when get_PFM_info.py changes, but you don't want to re run from the start.
PFM=get_PFM_info()

t1 = datetime.now()
# this is the same function as for LV1, only difference is providing LV2 grid to interpolate to
print('in atmfuns.get_atm_data_on_roms_grid(ATM,RMG)')
print('doing level: ' + str(level))
cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_on_roms_grid',str(level)]
os.chdir('../sdpm_py_util')
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('done with: atmfuns.get_atm_data_on_roms_grid(ATM,RMG)')
t2 = datetime.now()
print('this took:')
print(t2-t1)
print('\n')


# plot both raw and LV2 atm fields
t1 = datetime.now()
plot_all_atm = 1
if plot_all_atm == 1:
    cmd_list = ['python','-W','ignore','plotting_functions.py','plot_all_fields_in_one',str(level)]
    print('plotting atm and atm on roms grid...')
    os.chdir('../sdpm_py_util')
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    print('driver_run_forecast_L21:  done with writing LV2_ATM.nc file.') #pltfuns.plot_all_fields_in_one(str(level))
    print('...done with: pltfuns.plot_all_fields_in_one')

t2 = datetime.now()
print('this took:')
print(t2-t1)
print('\n')

# save the atm data into LV2_atm.nc
t1 = datetime.now()
print('driver_run_forcast_LV2: saving LV2_ATM.nc file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level)]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_L21:  done with writing LV2_ATM.nc file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')

# make the LV2_OCN_BC.pkl file
t1 = datetime.now()
print('driver_run_forcast_LV2: saving LV2_OCN_BC pickle file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_BC_dict',str(level)]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_L21:  done with writing LV2_OCN_BC.pkl file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')

# convert LV2_BC.pkl to LV2_BC.nc
t1 = datetime.now()
lv2_ocnBC_pckl = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnBC_tmp_pckl_file']
lv2_bc_file_out = PFM['lv2_forc_dir'] + '/' + PFM['lv2_bc_file']
print('driver_run_forcast_LV2: saving LV2_OCN_BC netcdf file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_BC_dict_to_netcdf_pckl',lv2_ocnBC_pckl,lv2_bc_file_out]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_L21:  done with writing LV2_OCN_BC.nc file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')

# make and save the LV2_IC.pkl file
t1=datetime.now()
print('driver_run_forcast_LV2: making and saving LV2_OCN_IC pickle file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_IC_dict',str(level)]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_L21:  done with writing LV2_OCN_IC.pkl file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')

# convert the LV2_IC.pkl file to LV2_IC.nc
t1=datetime.now()
lv2_ocnIC_pckl = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnIC_tmp_pckl_file']
lv2_ic_file_out = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ini_file']
print('driver_run_forcast_LV2: saving LV2_OCN_IC netcdf file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_IC_dict_to_netcdf_pckl',lv2_ocnIC_pckl,lv2_ic_file_out]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_L21:  done with writing LV2_OCN_IC.nc file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')

# now make .in and .sb for roms, and run roms...
print('making .in and .sb...')
yyyymmdd = PFM['yyyymmdd']
hhmm = PFM['hhmm']
os.chdir('../sdpm_py_util')
make_LV2_dotin_and_SLURM( PFM , yyyymmdd + hhmm ) # we could remove the arguments?
print('...done')
# run command will be
print('now running roms LV2 with slurm.')
print('using ' + str(PFM['gridinfo']['L2','nnodes']) + ' nodes.')
print('Ni = ' + str(PFM['gridinfo']['L2','ntilei']) + ', NJ = ' + str(PFM['gridinfo']['L2','ntilej']))
print('working...')
t1=datetime.now()
run_slurm_LV2(PFM)
os.chdir('../driver')
print('running ROMS took:')
t2 = datetime.now()
print(t2-t1)
print('\n')

# plot his.nc from LV2 
print('now making LV2 history file plots')
pltfuns.make_all_his_figures('LV2')
print('done with all of LV2.')

# -- driver_run_forecast_LV2_v1.py  --
# master python script to do a full LV2 forecast simulation

import sys
import os
from datetime import datetime
import subprocess
import pickle

##############

sys.path.append('../sdpm_py_util')
from init_funs import remake_PFM_pkl_file
from get_PFM_info import get_PFM_info
from make_LV3_dotin_and_SLURM import make_LV3_dotin_and_SLURM
from run_slurm_LV3 import run_slurm_LV3
import plotting_functions as pltfuns

##############

level = 3

##############
print('now starting the LV3 simulation, setting up...')
print("Current local Time =")
print(datetime.now())
PFM=get_PFM_info()

##############
# putting atm raw data on to the roms LV3 grid
t1 = datetime.now()
# put the atm data on the roms LV3 grid, and rotate the velocities
# everything in this dict turn into the atm.nc file
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
dt_atm = []
dt_atm.append(t2-t1)

##############
# plot both raw and LV3 atm fields
t1 = datetime.now()
plot_all_atm = 1
if plot_all_atm == 1:
    cmd_list = ['python','-W','ignore','plotting_functions.py','plot_all_fields_in_one',str(level)]
    print('plotting atm and atm on roms grid...')
    os.chdir('../sdpm_py_util')
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    print('...done with: pltfuns.plot_all_fields_in_one')
    t2 = datetime.now()
    print('this took:')
    print(t2-t1)
    print('\n')
    dt_plotting = []
    dt_plotting.append(t2-t1)



##############
# save the atm data into LV2_atm.nc
t1 = datetime.now()
print('driver_run_forcast_LV3: saving LV3_ATM.nc file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level)]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_LV3:  done with writing LV3_ATM.nc file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')
dt_atm.append(t2-t1)

##############
t1 = datetime.now()
t01 = datetime.now()
print('driver_run_forcast_LV3: saving LV'+str(level)+'_OCN_BC pickle file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_BC_dict',str(level)]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('done with writing LV'+str(level)+'_OCN_BC.pkl file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')

##############
t1 = datetime.now()
lv3_ocnBC_pckl = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnBC_tmp_pckl_file']
lv3_bc_file_out = PFM['lv3_forc_dir'] + '/' + PFM['lv3_bc_file']
print('driver_run_forcast_LV3: saving LV3_OCN_BC netcdf file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_BC_dict_to_netcdf_pckl',lv3_ocnBC_pckl,lv3_bc_file_out]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_LV3:  done with writing LV3_OCN_BC.nc file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')
dt_bc = []
dt_bc.append(t2-t01)

##############
t1=datetime.now()
t01 = datetime.now()
print('driver_run_forcast_LV3: saving LV'+str(level)+'_OCN_IC pickle file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_IC_dict',str(level)]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_LV3:  done with writing LV3_OCN_IC.pkl file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')

##############
t1=datetime.now()
lv3_ocnIC_pckl = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnIC_tmp_pckl_file']
lv3_ic_file_out = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ini_file']
print('driver_run_forcast_LV3: saving LV3_OCN_IC netcdf file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_IC_dict_to_netcdf_pckl',lv3_ocnIC_pckl,lv3_ic_file_out]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_L3:  done with writing LV3_OCN_IC.nc file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')
dt_ic = []
dt_ic.append(t2-t01)

##############
yyyymmdd = PFM['yyyymmdd']
hhmm = PFM['hhmm']

t1=datetime.now()
print('making LV3 .in and .sb...')
os.chdir('../sdpm_py_util')
make_LV3_dotin_and_SLURM( PFM , yyyymmdd + hhmm )
print('...done')
# run command will be
print('now running roms LV3 with slurm.')
print('using ' + str(PFM['gridinfo']['L3','nnodes']) + ' nodes.')
print('Ni = ' + str(PFM['gridinfo']['L3','ntilei']) + ', NJ = ' + str(PFM['gridinfo']['L3','ntilej']))
print('working...')
run_slurm_LV3(PFM)

os.chdir('../driver')
print('...done.')
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')
#print(t2-t00)
dt_roms = []
dt_roms.append(t2-t1)

print('now making LV3 history file plots...')
t01=datetime.now()
pltfuns.make_all_his_figures('LV3')
print('...done.')
dt_plotting.append(datetime.now()-t01)

dt_LV3 = {}
dt_LV3['roms'] = dt_roms
dt_LV3['ic'] = dt_ic
dt_LV3['bc'] = dt_bc
dt_LV3['atm'] = dt_atm
dt_LV3['plotting'] = dt_plotting

fn_timing = PFM['lv3_run_dir'] + '/LV3_timing_info.pkl'
with open(fn_timing,'wb') as fout:
    pickle.dump(dt_LV3,fout)
    print('OCN_LV3 timing info dict saved with pickle to: ',fn_timing)

print('\n\n----------------------')
print('Finished the LV3 simulation\n')




# -- driver_run_forecast_LV4_v1.py  --
# master python script to do a full LV4 forecast simulation

import sys
import os
from datetime import datetime
import subprocess
import pickle

##############

sys.path.append('../sdpm_py_util')
from init_funs import remake_PFM_pkl_file
from get_PFM_info import get_PFM_info
from make_LV4_coawst_dotins_dotsb import make_LV4_coawst_dotins_dotsb
from run_slurm_LV4 import run_slurm_LV4
import plotting_functions as pltfuns

##############

level = 4

##############
print('now starting the LV4 simulation, setting up...')
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
# plot both raw and LV4 atm fields
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
# save the atm data into LV4_atm.nc
t1 = datetime.now()
print('driver_run_forcast_LV4: saving LV4_ATM.nc file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level)]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_LV4:  done with writing LV4_ATM.nc file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')
dt_atm.append(t2-t1)


##############
t1 = datetime.now()
t01 = datetime.now()
print('driver_run_forcast_LV4: saving LV'+str(level)+'_OCN_BC pickle file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_BC_dict_edges',str(level)]
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
lv4_ocnBC_pckl = PFM['lv4_forc_dir'] + '/' + PFM['lv4_ocnBC_tmp_pckl_file']
lv4_bc_file_out = PFM['lv4_forc_dir'] + '/' + PFM['lv4_bc_file']
print('driver_run_forcast_LV4: saving LV4_OCN_BC netcdf file')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_BC_dict_to_netcdf_pckl',lv4_ocnBC_pckl,lv4_bc_file_out]
ret5 = subprocess.run(cmd_list)   
print('return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_LV4:  done with writing LV4_OCN_BC.nc file.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')
dt_bc = []
dt_bc.append(t2-t01)

##############
dt_ic = []
t1=datetime.now()
t01 = datetime.now()
if PFM['lv4_use_restart']==0:
    print('driver_run_forcast_LV4: saving LV'+str(level)+'_OCN_IC pickle file')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_IC_dict',str(level)]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_LV4:  done with writing LV4_OCN_IC.pkl file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')

    ##############
    t1=datetime.now()
    lv4_ocnIC_pckl = PFM['lv4_forc_dir'] + '/' + PFM['lv4_ocnIC_tmp_pckl_file']
    lv4_ic_file_out = PFM['lv4_forc_dir'] + '/' + PFM['lv4_ini_file']
    print('driver_run_forcast_LV4: saving LV4_OCN_IC netcdf file')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_IC_dict_to_netcdf_pckl',lv4_ocnIC_pckl,lv4_ic_file_out]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_L3:  done with writing LV4_OCN_IC.nc file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')
    dt_ic.append(t2-t01)
else:
    print('going to use a restart file for the LV4 IC. Setting this up...')
    cmd_list = ['python','-W','ignore','init_funs.py','restart_setup','LV4']
    os.chdir('../sdpm_py_util')
    ret4 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('...done setting up for restart.')
    print('this took:')
    dt_ic = []
    t05 = datetime.now()
    print(t05-t01)
    dt_ic.append(t05-t01)
    PFM = get_PFM_info()
    print('\nGoing to use the file ' + PFM['lv4_ini_file'] + ' to restart the simulation')
    print('with time index ' + str(PFM['lv4_nrrec']))
    print('\n')
    if PFM['lv4_nrrec'] < 0:
        print('WARNING RESTARTING LV3 WILL NOT WORK!!!')



##############
# make clm, nud, river .nc files...

t1=datetime.now()
print('driver_run_forcast_LV4: making clm.nc, nud.nc, and river.nc files...')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','ocn_functions.py','mk_lv4_clm_nc']
ret5 = subprocess.run(cmd_list)   
print('clm return code: ' + str(ret5.returncode) + ' (0=good)')  
cmd_list = ['python','-W','ignore','ocn_functions.py','mk_lv4_nud_nc']
ret5 = subprocess.run(cmd_list)   
print('nud return code: ' + str(ret5.returncode) + ' (0=good)')  
cmd_list = ['python','-W','ignore','ocn_functions.py','mk_lv4_river_nc']
ret5 = subprocess.run(cmd_list)   
print('river return code: ' + str(ret5.returncode) + ' (0=good)')  
os.chdir('../sdpm_py_util')
print('driver_run_forecast_L4:  done making clm, nud, and river.nc files.') 
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')
dt_ic.append(t2-t1)

##############
# make swan files
t1=datetime.now()
print('driver_run_forcast_LV4: making swan bnd and wnd files...')
os.chdir('../sdpm_py_util')
cmd_list = ['python','-W','ignore','swan_functions.py','cdip_ncs_to_dict','refresh']
ret5 = subprocess.run(cmd_list)   
print('cdip to dictionary return code: ' + str(ret5.returncode) + ' (0=good)')  
fout = PFM['lv4_forc_dir'] + '/' + PFM['lv4_swan_bnd_file']
print('making swan .bnd file...')
cmd_list = ['python','-W','ignore','swan_functions.py','mk_swan_bnd_file',fout]
ret5 = subprocess.run(cmd_list)   
print('...done. swan bnd file return code: ' + str(ret5.returncode) + ' (0=good)')  
fout = PFM['lv4_forc_dir'] + '/' + PFM['lv4_swan_wnd_file']
print('making swan wnd file...')
cmd_list = ['python','-W','ignore','swan_functions.py','mk_swan_wnd_file',fout]
ret5 = subprocess.run(cmd_list)   
print('...done. swan wnd file return code: ' + str(ret5.returncode) + ' (0=good)')  
t2 = datetime.now()
print('...done. making swan .bnd and .wnd files took:')
print(t2-t1)
dt_sw = []
dt_sw.append(t2-t1)

##############
# make all of the dotins
t1=datetime.now()
print('making LV4 ocean, swan, coupling .in and .sb...')
os.chdir('../sdpm_py_util')
make_LV4_coawst_dotins_dotsb()
print('...done')

################
# run coawst LV4
print('now running Coawst LV4 with slurm.')
print('using ' + str(PFM['gridinfo']['L4','nnodes']) + ' nodes (= ' + str(36 * PFM['gridinfo']['L4','nnodes']) + ' CPUs.)')
print('using ' + str( PFM['gridinfo']['L4','ntilei']*PFM['gridinfo']['L4','ntilej'] ) + ' CPUs for ROMS, with tiling:')
print('Ni = ' + str(PFM['gridinfo']['L4','ntilei']) + ', NJ = ' + str(PFM['gridinfo']['L4','ntilej']))
print('and using ' + str(PFM['gridinfo']['L4','np_swan']) + ' CPUs for SWAN.')
print('working...')

run_slurm_LV4(PFM)

os.chdir('../driver')
print('...done.')
print('this took:')
t2 = datetime.now()
print(t2-t1)
print('\n')
#print(t2-t00)
dt_roms = []
dt_roms.append(t2-t1)

print('now making LV4 history file plots...')
t01=datetime.now()
cmd_list = ['python','-W','ignore','plotting_functions.py','make_all_his_figures','LV4']
os.chdir('../sdpm_py_util')
ret6 = subprocess.run(cmd_list)   
print('...done plotting LV4: ' + str(ret6.returncode) + ' (0=good)')  
os.chdir('../driver')
print('this took:')
print(datetime.now()-t01)

#pltfuns.make_all_his_figures('LV4')
dt_plotting.append(datetime.now()-t01)

dt_LV4 = {}
dt_LV4['roms'] = dt_roms
dt_LV4['ic'] = dt_ic
dt_LV4['bc'] = dt_bc
dt_LV4['atm'] = dt_atm
dt_LV4['plotting'] = dt_plotting
dt_LV4['swan'] = dt_sw

fn_timing = PFM['lv4_run_dir'] + '/LV4_timing_info.pkl'
with open(fn_timing,'wb') as fout:
    pickle.dump(dt_LV4,fout)
    print('OCN_LV4 timing info dict saved with pickle to: ',fn_timing)

print('\n\n----------------------')
print('Finished the LV4 simulation\n')




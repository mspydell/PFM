# -- driver_hindcast_LV1.py  --
# master python script to do a full LV1 forecast simulation

import sys
import os
from datetime import datetime, timedelta
import subprocess
import pickle

##############

sys.path.append('../sdpm_py_util')

import ocn_functions as ocnfuns
import plotting_functions as pltfuns
from get_PFM_info import get_PFM_info
import init_funs as infuns
from make_LV1_dotin_and_SLURM import make_LV1_dotin_and_SLURM
from run_slurm_LV1 import run_slurm_LV1

print('\nStarting the LV1 hindcast, Current time ', datetime.now())

# PFM has all of the information needed to run the model
clean_start = True
infuns.initialize_simulation(clean_start)
PFM=get_PFM_info()

if PFM['run_type'] == 'hindcast':
    print('the hindcast start time is:')
    tbeg = PFM['sim_start_time']
    print(tbeg)
    print('hindcasting will be done in 1 day chunks.')
    print('the hindcast end time is:')
    tend = PFM['sim_end_time']
    print(tend)
else:
    print('PFM run_type is:')
    print(PFM['run_type'])
    print('aborting!')
    sys.exit()
    

t00 = datetime.now() # for keeping track of timing of the simulation
print('this hindcast simulation is starting at (local time)')
print(t00)

tsim = PFM['fetch_time']

start_type = PFM['lv1_use_restart'] # 0=new solution. 1=from a restart file
lv1_use_restart = start_type

level = 1


while tsim < tend:  
    # now a string of the time to start ROMS (and the 1st atm time too)
    PFM = get_PFM_info()
 
    print('\nGoing to do a PHM daily hindcast starting at (UTC)')
    print(tsim)
    print('and ending at (UTC)')
    t2 = tsim + timedelta(days=1)
    t2str = t2.strftime("%Y%m%d%H")
    print(t2)

    # getting the hycom hindcast data for this simulation
    yyyymmddhh = tsim.strftime("%Y%m%d%H")
    t1str = yyyymmddhh
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','get_hycom_hind_data',yyyymmddhh]
    ret1 = subprocess.run(cmd_list)     
    

    print('making the hycom pickle file from all hycom.nc files ...')
    t01 = datetime.now() 
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','hycom_hind_ncfiles_to_pickle']
    ret1 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('did subprocess run correctly? ' + str(ret1.returncode) + ' (0=yes,1=no)')
    print('...done.')
    t02 = datetime.now() 
    print('this took:')
    print(t02-t01)
    print('\n')
    dt_process =[]
    dt_process.append(t02-t01)


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
    dt_plotting = []

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
        dt_plotting.append(t02-t01)


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
    t02 = datetime.now()
    dt_process.append(t02-t01)

#    if plot_ocnr == 1:
#        print('plotting LV1 ocn_R_fields...')
#        cmd_list = ['python','-W','ignore','plotting_functions.py','plot_ocn_R_fields_pckl']
#        os.chdir('../sdpm_py_util')
#        ret1 = subprocess.run(cmd_list)     
#        os.chdir('../driver')
#        print('subprocess return code? ' + str(ret1.returncode) +  ' (0=good)')
#        t03 = datetime.now()
#        dt_plotting.append(t03-t02)
        
    print('...done with LV1 ocn_R')
    print('this took:')
    print(t02-t01)
    print('\n')
    t04 = datetime.now()

    # make the depth pickle file
    print('making the depth pickle file...')
    fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']
    cmd_list = ['python','-W','ignore','ocn_functions.py','make_rom_depths_1hrzeta',fname_depths]
    os.chdir('../sdpm_py_util')
    ret6 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('subprocess return code? ' + str(ret6.returncode) +  ' (0=good)')
    print('\n')
    dt_process.append(datetime.now()-t04)
        
    t01 = datetime.now()
    lv1_use_restart = PFM['lv1_use_restart']

    if lv1_use_restart==0:
        print('going to save OCN_IC to a pickle file: ' + ocnIC_pckl)
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
        t03 = datetime.now()
        cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_IC_dict_to_netcdf_pckl',ocnIC_pckl,ic_file_out]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('OCN IC nc data saved, correctly? ' + str(ret4.returncode) + ' (0=yes)')

        print('done makeing IC file.')
        dt_ic = []
        t05 = datetime.now()
        dt_ic.append(t05-t01)
        if plot_ocn_icnc == 1:
            print('plotting...')
            pltfuns.plot_ocn_ic_fields(ic_file_out)
            t04 = datetime.now()
            print('...done. this took:')
            print(t04-t03)
            print('\n')
            dt_plotting.append(t04-t05)
    else:
        print('going to use a restart file for the LV1 IC. Setting this up...')
        print('first need to update the PFM pickle file...')
        infuns.update_PFM_pkl(t1str,'1')
        cmd_list = ['python','-W','ignore','init_funs.py','restart_setup','LV1']
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
        print('\nGoing to use the file ' + PFM['lv1_ini_file'] + ' to restart the simulation')
        print('with time index ' + str(PFM['lv1_nrrec']))
        print('\n')
        if PFM['lv1_nrrec'] < 0:
            print('WARNING RESTARTING LV1 WILL NOT WORK!!!')

    PFM = get_PFM_info() # refresh this
    # get the OCN_BC dictionary
    print('going to save OCN_BC to a pickle file to:')
    t01 = datetime.now()
    t04 = datetime.now()
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
    dt_bc = []
    dt_bc.append(t02-t04)

    # now for the atm part... this function is different from PFM
    dt_download_atm = []
    print('we are now getting the atm data...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','hind_functions.py','get_nam_hindcast_grb2s_v2',t1str,t2str]
    os.chdir('../sdpm_py_util')
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('...done.')
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')
    dt_download_atm.append(t02-t01)

    print('we are now converting the atm grb2 files to pickles...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','hind_functions.py','grb2s_to_pickles',t1str,t2str]
    os.chdir('../sdpm_py_util')
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('...done.')
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')

    # put the atm data on the roms grid, and rotate the velocities
    # everything in this dict turn into the atm.nc file
    print('we are now putting the hind atm data on the roms LV1 grid...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','hind_functions.py','nam_pkls_2_romsatm_pkl',t1str,t2str,str(level)]
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
    dt_atm = []
    dt_atm.append(t02-t01)

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
    dt_atm.append(t02-t01)

    print('driver_run_forecast_LV1:  making .in and .sb files...')
    t01 = datetime.now()
    pfm_driver_src_dir = os.getcwd()
    #yyyymmdd = PFM['yyyymmdd']
    #hhmm = PFM['hhmm']
    yyyymmddhhmm = PFM['fetch_time'].strftime('%Y%m%d%H%M')
    os.chdir('../sdpm_py_util')
    make_LV1_dotin_and_SLURM( PFM , yyyymmddhhmm )
    print('...done.\n')

    # run command will be
    print('now running roms LV1 with slurm.')
    print('using ' + str(PFM['gridinfo']['L1','nnodes']) + ' nodes.')
    print('Ni = ' + str(PFM['gridinfo']['L1','ntilei']) + ', NJ = ' + str(PFM['gridinfo']['L1','ntilej']))
    print('working...')
    run_slurm_LV1(PFM)
    print('...done.')
    os.chdir('../driver')
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')
    dt_roms = []
    dt_roms.append(t02-t01)

    # go to the next day
    tsim = tsim + timedelta(days=1)
    PFM2 = {}
    PFM2['lv1_use_restart'] = 1
    PFM2['fetch_time'] = tsim
    infuns.edit_and_save_PFM(PFM2)
    print('done with a 1 day LV1 hindcast, going to the next day.\n')
    #sys.exit("exiting for now.")

import numpy as np
import os
import sys
import subprocess
sys.path.append('../sdpm_py_util')
import init_funs as initfuns
from datetime import datetime, timedelta

# the functions in here are used to make the 

def run_hind_LV1(t1str,pkl_fnm):

    MI = initfuns.get_model_info(pkl_fnm)

    t1 = datetime.strptime(t1str,'%Y%m%d%H')
    t2 = t1 + MI['forecast_days']*timedelta(days=1)
    t2str = t2.strftime("%Y%m%d%H")

    start_type = MI['lv1_use_restart'] # 0=new solution. 1=from a restart file
    lv1_use_restart = start_type
    level = 1

    print('getting the hycom data for this simulation...')
    os.chdir('../sdpm_py_util')
    # this will only work for 1 day chunks. need to fix?
    cmd_list = ['python','-W','ignore','ocn_functions.py','get_hycom_hind_data',t1str,t2str,pkl_fnm]
    ret1 = subprocess.run(cmd_list)
    print('...done')     

    print('making the hycom pickle file from all hycom.nc files ...')
    t01 = datetime.now() 
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','hycom_hind_ncfiles_to_pickle',pkl_fnm]
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
    t01 = datetime.now()
    dt_plotting = []

    # need the file names and locations of the pickle and .nc files we will save
    ocnBC_pckl = MI['lv1_forc_dir'] + '/' + MI['lv1_ocnBC_tmp_pckl_file']
    fn_atm_out = MI['lv1_forc_dir'] + '/' + MI['lv1_atm_file'] # LV1 atm forcing filename

    # fix this for new pkl file usage !!!
    #if plot_ocn == 1:
    #    print('making some plots from: ' + hy_pckl)
    #    cmd_list = ['python','-W','ignore','plotting_functions.py','plot_ocn_fields_from_dict_pckl',hy_pckl]
    #    os.chdir('../sdpm_py_util')
    #    ret1 = subprocess.run(cmd_list)     
        #pltfuns.plot_ocn_fields_from_dict_pckl(fn_pckl)
    #    print('subprocess return code? ' + str(ret1.returncode) +  ' (0=good)')
    #    print('...done')
    #    os.chdir('../driver')
    #    t02= datetime.now()
    #    print('this took:')
    #    print(t02-t01)
    #    print('\n')
    #    dt_plotting.append(t02-t01)

    # put the ocn data on the roms grid
    print('starting: ocnfuns.hycom_to_roms_latlon(OCN,RMG)')
    t01 = datetime.now()
    os.chdir('../sdpm_py_util')
    hy_pckl = MI['lv1_forc_dir'] + '/' + MI['lv1_ocn_tmp_pckl_file']
    print('putting the hycom data in ' + hy_pckl + ' on the roms grid...')
    cmd_list = ['python','-W','ignore','ocn_functions.py','make_all_tmp_pckl_ocnR_files_1hrzeta',pkl_fnm]
    os.chdir('../sdpm_py_util')
    ret1 = subprocess.run(cmd_list)     
    #ocnfuns.make_all_tmp_pckl_ocnR_files(fn_pckl)
    os.chdir('../driver')
    print('subprocess return code? ' + str(ret1.returncode) +  ' (0=good)')

    cmd_list = ['python','-W','ignore','ocn_functions.py','print_maxmin_HYrm_pickles',pkl_fnm]
    os.chdir('../sdpm_py_util')
    ret1 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('driver_run_forecast_LV1: done with hycom_to_roms_latlon')
    t02 = datetime.now()
    dt_process.append(t02-t01)
        
    print('...done with LV1 ocn_R')
    print('this took:')
    print(t02-t01)
    print('\n')
    t04 = datetime.now()

    # make the depth pickle file
    print('making the depth pickle file...')
    fname_depths = MI['lv1_forc_dir'] + '/' + MI['lv1_depth_file']
    cmd_list = ['python','-W','ignore','ocn_functions.py','make_rom_depths_1hrzeta',fname_depths,pkl_fnm]
    os.chdir('../sdpm_py_util')
    ret6 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('subprocess return code? ' + str(ret6.returncode) +  ' (0=good)')
    print('\n')
    dt_process.append(datetime.now()-t04)
        
    t01 = datetime.now()
    lv1_use_restart = MI['lv1_use_restart']

    ocnIC_pckl = MI['lv1_forc_dir'] + '/' + MI['lv1_ocnIC_tmp_pckl_file']
    if lv1_use_restart==0:
        print('going to save OCN_IC to a pickle file: ' + ocnIC_pckl)
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_functions.py','ocnr_2_ICdict_from_tmppkls',ocnIC_pckl,pkl_fnm]
        ret3 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('OCN IC data saved with pickle, correctly? ' + str(ret3.returncode) + ' (0=yes,1=no)')

        print('driver_run_forecast_LV1: done with ocn_r_2_ICdict')
        t02 = datetime.now()
        print('this took:')
        print(t02-t01)
        print('\n')

        print('making IC file from pickled IC: '+ ic_file_out)
        ic_file_out = MI['lv1_forc_dir'] + '/' + MI['lv1_ini_file']
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
        #if plot_ocn_icnc == 1:
        #    print('plotting...')
        #    pltfuns.plot_ocn_ic_fields(ic_file_out)
        #    t04 = datetime.now()
        #    print('...done. this took:')
        #    print(t04-t03)
        #    print('\n')
        #    dt_plotting.append(t04-t05)
    else:
        print('going to use a restart file for the LV1 IC. Setting this up...')
        print('first need to update the MI pickle file...')
        initfuns.update_MI_pkl(t1str,'1')
        cmd_list = ['python','-W','ignore','init_funs.py','restart_setup','LV1',pkl_fnm]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('...done setting up for restart.')
        print('this took:')
        dt_ic = []
        t05 = datetime.now()
        print(t05-t01)
        dt_ic.append(t05-t01)
        MI = get_model_info()
        print('\nGoing to use the file ' + MI['lv1_ini_file'] + ' to restart the simulation')
        print('with time index ' + str(MI['lv1_nrrec']))
        print('\n')
        if MI['lv1_nrrec'] < 0:
            print('WARNING RESTARTING LV1 WILL NOT WORK!!!')

    MI = initfuns.get_model_info(pkl_fnm) # refresh this
    # get the OCN_BC dictionary
    print('going to save OCN_BC to a pickle file to:')
    t01 = datetime.now()
    t04 = datetime.now()
    ocnBC_pckl = MI['lv1_forc_dir'] + '/' + MI['lv1_ocnBC_tmp_pckl_file']
    print(ocnBC_pckl) 
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','ocnr_2_BCdict_1hrzeta_from_tmppkls',ocnBC_pckl,pkl_fnm]
    ret4 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('OCN BC data saved with pickle, correctly? ' + str(ret4.returncode) + ' (0=yes)')
        
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')

    bc_file_out = MI['lv1_forc_dir'] + '/' + MI['lv1_bc_file']
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

    # now for the atm part... this function is different from MI
    dt_download_atm = []
    print('we are now getting the atm data...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','hind_functions.py','get_nam_hindcast_grb2s_v2',t1str,t2str,pkl_fnm]
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
    #yyyymmdd = MI['yyyymmdd']
    #hhmm = MI['hhmm']
    yyyymmddhhmm = MI['fetch_time'].strftime('%Y%m%d%H%M')
    os.chdir('../sdpm_py_util')
    make_LV1_dotin_and_SLURM( MI , yyyymmddhhmm )
    print('...done.\n')

    # run command will be
    print('now running roms LV1 with slurm.')
    print('using ' + str(MI['gridinfo']['L1','nnodes']) + ' nodes.')
    print('Ni = ' + str(MI['gridinfo']['L1','ntilei']) + ', NJ = ' + str(MI['gridinfo']['L1','ntilej']))
    print('working...')
    run_slurm_LV1(MI)
    print('...done.')
    os.chdir('../driver')
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')
    dt_roms = []
    dt_roms.append(t02-t01)


def run_hind_simulation(t1str,lvl,pkl_fnm):
    if lvl == 'LV1':
        run_hind_LV1(t1str,pkl_fnm)
    if lvl == 'LV2':
        run_hind_LV2(t1str,pkl_fnm)
    if lvl == 'LV3':
        run_hind_LV3(t1str,pkl_fnm)
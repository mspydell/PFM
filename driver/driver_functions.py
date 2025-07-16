import os
import sys
import subprocess
import pickle
sys.path.append('../sdpm_py_util')
from datetime import datetime, timedelta
import run_funs as runfuns 

#from make_LV1_dotin_and_SLURM import make_LV1_dotin_and_SLURM
#from run_slurm_LV1 import run_slurm_LV1



# the functions in here are used to make the 

def run_hind_LV1(t1str,pkl_fnm):
    import init_funs as initfuns

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

        ic_file_out = MI['lv1_forc_dir'] + '/' + MI['lv1_ini_file']
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
        # update the model information as we are now using restarts
        MI = initfuns.get_model_info(pkl_fnm)
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
    cmd_list = ['python','-W','ignore','hind_functions.py','grb2s_to_pickles',t1str,t2str,pkl_fnm]
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
    cmd_list = ['python','-W','ignore','hind_functions.py','nam_pkls_2_romsatm_pkl',t1str,t2str,str(level),pkl_fnm]
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
    fn_atm_out = MI['lv1_forc_dir'] + '/' + MI['lv1_atm_file'] # LV1 atm forcing filename
    print('we are now saving ATM LV1 to ' + fn_atm_out + ' ...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level),pkl_fnm,'hind']
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
    #yyyymmdd = MI['yyyymmdd']
    #hhmm = MI['hhmm']
    os.chdir('../sdpm_py_util')
    runfuns.make_LV1_dotin_and_SLURM( pkl_fnm , 'hind' )
    print('...done.\n')

    # run command will be
    print('now running roms LV1 with slurm.')
    print('using ' + str(MI['gridinfo']['L1','nnodes']) + ' nodes.')
    print('Ni = ' + str(MI['gridinfo']['L1','ntilei']) + ', NJ = ' + str(MI['gridinfo']['L1','ntilej']))
    runfuns.run_slurm_LV1( pkl_fnm , 'hind')
    print('...done.')
    os.chdir('../driver')
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')
    dt_roms = []
    dt_roms.append(t02-t01)

def run_hind_LV2(t1str,pkl_fnm):
    import init_funs as initfuns

    level = 2
    MI = initfuns.get_model_info(pkl_fnm)
    t1 = datetime.strptime(t1str,'%Y%m%d%H')
    t2 = t1 + MI['forecast_days']*timedelta(days=1)
    t2str = t2.strftime('%Y%m%d%H')
    print('we are now putting the hind atm data on the roms LV2 grid...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','hind_functions.py','nam_pkls_2_romsatm_pkl',t1str,t2str,str(level),pkl_fnm]
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
    fn_atm_out = MI['lv2_forc_dir'] + '/' + MI['lv2_atm_file'] # LV1 atm forcing filename
    print('we are now saving ATM LV2 to ' + fn_atm_out + ' ...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level),pkl_fnm,'hind']
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

    # make the LV2_OCN_BC.pkl file
    t1 = datetime.now()
    t01 = datetime.now()
    print('driver_run_forcast_LV2: saving LV2_OCN_BC pickle file')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_BC_dict_edges',str(level),pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_LV2:  done with writing LV2_OCN_BC.pkl file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')

    # convert LV2_BC.pkl to LV2_BC.nc
    t1 = datetime.now()
    lv2_ocnBC_pckl = MI['lv2_forc_dir'] + '/' + MI['lv2_ocnBC_tmp_pckl_file']
    lv2_bc_file_out = MI['lv2_forc_dir'] + '/' + MI['lv2_bc_file']
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
    dt_bc = []
    dt_bc.append(t2-t01)

    # make and save the LV2_IC.pkl file
    dt_ic = []
    t1=datetime.now()
    t01 = datetime.now()
    if MI['lv2_use_restart']==0:
        print('driver_run_forcast_LV2: making and saving LV2_OCN_IC pickle file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_IC_dict',str(level),pkl_fnm]
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
        lv2_ocnIC_pckl = MI['lv2_forc_dir'] + '/' + MI['lv2_ocnIC_tmp_pckl_file']
        lv2_ic_file_out = MI['lv2_forc_dir'] + '/' + MI['lv2_ini_file']
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
        dt_ic.append(t2-t01)
    else:
        print('going to use a restart file for the LV2 IC. Setting this up...')
        cmd_list = ['python','-W','ignore','init_funs.py','restart_setup','LV2',pkl_fnm]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('...done setting up for restart.')
        print('this took:')
        dt_ic = []
        t05 = datetime.now()
        print(t05-t01)
        dt_ic.append(t05-t01)
        MI = initfuns.get_model_info(pkl_fnm)
        print('\nGoing to use the file ' + MI['lv2_ini_file'] + ' to restart the simulation')
        print('with time index ' + str(MI['lv2_nrrec']))
        print('\n')
        if MI['lv2_nrrec'] < 0:
            print('WARNING RESTARTING LV2 WILL NOT WORK!!!')

    # now make .in and .sb for roms, and run roms...
    print('making .in and .sb...')
    t1=datetime.now()
    os.chdir('../sdpm_py_util')
    runfuns.make_LV2_dotin_and_SLURM( pkl_fnm ,'hind')
    print('...done')
    # run command will be
    print('now running roms LV2 with slurm.')
    print('using ' + str(MI['gridinfo']['L2','nnodes']) + ' nodes.')
    print('Ni = ' + str(MI['gridinfo']['L2','ntilei']) + ', NJ = ' + str(MI['gridinfo']['L2','ntilej']))
    print('working...')
    runfuns.run_slurm_LV2( pkl_fnm ,'hind')
    os.chdir('../driver')
    print('running ROMS took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')
    dt_roms = []
    dt_roms.append(t2-t1)

def run_hind_LV3(t1str,pkl_fnm):
    import init_funs as initfuns

    level = 3
    MI = initfuns.get_model_info(pkl_fnm)
    t1 = datetime.strptime(t1str,'%Y%m%d%H')
    t2 = t1 + MI['forecast_days']*timedelta(days=1)
    t2str = t2.strftime('%Y%m%d%H')
    print('we are now putting the hind atm data on the roms LV3 grid...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','hind_functions.py','nam_pkls_2_romsatm_pkl',t1str,t2str,str(level),pkl_fnm]
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
    fn_atm_out = MI['lv3_forc_dir'] + '/' + MI['lv3_atm_file'] # LV1 atm forcing filename
    print('we are now saving ATM LV2 to ' + fn_atm_out + ' ...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level),pkl_fnm,'hind']
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

    # make the LV3_OCN_BC.pkl file
    t1 = datetime.now()
    t01 = datetime.now()
    print('driver_run_forcast_LV3: saving LV3_OCN_BC pickle file')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_BC_dict_edges',str(level),pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_LV3:  done with writing LV3_OCN_BC.pkl file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')

    # convert LV3_BC.pkl to LV3_BC.nc
    t1 = datetime.now()
    lv3_ocnBC_pckl = MI['lv3_forc_dir'] + '/' + MI['lv3_ocnBC_tmp_pckl_file']
    lv3_bc_file_out = MI['lv3_forc_dir'] + '/' + MI['lv3_bc_file']
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

    # make and save the LV2_IC.pkl file
    dt_ic = []
    t1=datetime.now()
    t01 = datetime.now()
    if MI['lv3_use_restart']==0:
        print('driver_run_forcast_LV3: making and saving LV3_OCN_IC pickle file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_functions.py','mk_LV2_IC_dict',str(level),pkl_fnm]
        ret5 = subprocess.run(cmd_list)   
        print('return code: ' + str(ret5.returncode) + ' (0=good)')  
        os.chdir('../sdpm_py_util')
        print('driver_run_forecast_LV3:  done with writing LV3_OCN_IC.pkl file.') 
        print('this took:')
        t2 = datetime.now()
        print(t2-t1)
        print('\n')

        # convert the LV2_IC.pkl file to LV2_IC.nc
        t1=datetime.now()
        lv3_ocnIC_pckl = MI['lv3_forc_dir'] + '/' + MI['lv3_ocnIC_tmp_pckl_file']
        lv3_ic_file_out = MI['lv3_forc_dir'] + '/' + MI['lv3_ini_file']
        print('driver_run_forcast_LV3: saving LV3_OCN_IC netcdf file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_functions.py','ocn_roms_IC_dict_to_netcdf_pckl',lv3_ocnIC_pckl,lv3_ic_file_out]
        ret5 = subprocess.run(cmd_list)   
        print('return code: ' + str(ret5.returncode) + ' (0=good)')  
        os.chdir('../sdpm_py_util')
        print('driver_run_forecast_LV3:  done with writing LV3_OCN_IC.nc file.') 
        print('this took:')
        t2 = datetime.now()
        print(t2-t1)
        print('\n')
        dt_ic.append(t2-t01)
    else:
        print('going to use a restart file for the LV3 IC. Setting this up...')
        cmd_list = ['python','-W','ignore','init_funs.py','restart_setup','LV3',pkl_fnm]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('...done setting up for restart.')
        print('this took:')
        dt_ic = []
        t05 = datetime.now()
        print(t05-t01)
        dt_ic.append(t05-t01)
        MI = initfuns.get_model_info(pkl_fnm)
        print('\nGoing to use the file ' + MI['lv3_ini_file'] + ' to restart the simulation')
        print('with time index ' + str(MI['lv3_nrrec']))
        print('\n')
        if MI['lv3_nrrec'] < 0:
            print('WARNING RESTARTING LV2 WILL NOT WORK!!!')

    # now make .in and .sb for roms, and run roms...
    print('making .in and .sb...')
    t1=datetime.now()
    os.chdir('../sdpm_py_util')
    runfuns.make_LV3_dotin_and_SLURM( pkl_fnm ,'hind')
    print('...done')
    # run command will be
    print('now running roms LV3 with slurm.')
    print('using ' + str(MI['gridinfo']['L3','nnodes']) + ' nodes.')
    print('Ni = ' + str(MI['gridinfo']['L3','ntilei']) + ', NJ = ' + str(MI['gridinfo']['L3','ntilej']))
    print('working...')
    runfuns.run_slurm_LV3( pkl_fnm ,'hind')
    os.chdir('../driver')
    print('running ROMS took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')
    dt_roms = []
    dt_roms.append(t2-t1)

def run_fore_LV1(pkl_fnm):
    import init_funs_forecast as initfuns_fore
    import ocn_funs_forecast as ocnfuns_fore
    t01 = datetime.now()

    PFM = initfuns_fore.get_model_info(pkl_fnm)

    initfuns_fore.initialize_simulation(pkl_fnm)

    t1  = PFM['fetch_time']    # this is the first time of the PFM forecast
    t1str = t1.strftime('%Y%m%d%H%M')
    print(t1str)
    # now a string of the time to start ROMS (and the 1st atm time too)
    yyyymmddhhmm_pfm = "%d%02d%02d%02d%02d" % (t1.year, t1.month, t1.day, t1.hour, t1.minute)
    t2  = t1 + PFM['forecast_days'] * timedelta(days=1)  # this is the last time of the PFM forecast

    print('\nGoing to do a PFM forecast from')
    print(t1)
    print('to')
    print(t2)

    t2str = "%d%02d%02d%02d%02d" % (t2.year, t2.month, t2.day, t2.hour, t2.minute)
    # get_hycom_foretime is a function that returns the lates hycom forecast date with data to span the PFM forecast from 
    # this functions cleans, then refreshes the hycom directory.
    print('getting the hycom forecast time...')
    print('this will clean, download new hycom files, then find the forecast date covering the PFM times.')
    t01 = datetime.now()
    yyyymmdd_hy = ocnfuns_fore.get_hycom_foretime_v2(yyyymmddhhmm_pfm,t2str,pkl_fnm)
    #print(yyyymmdd_hy)
 
    dt_download_ocn = []
    dt_download_ocn.append(datetime.now()-t01)
    
    start_type = PFM['lv1_use_restart'] # 0=new solution. 1=from a restart file
    lv1_use_restart = start_type
    level = 1

    print('\n\nwe will use the hycom forecast from')
    print(yyyymmdd_hy)
    print('for the PFM forecast.')
    print('...done.')
    t02 = datetime.now() # for keeping track of timing of the simulation
    print('this took:')
    print(t02-t01)
    print('\n')

    print('making the hycom pickle file from all hycom.nc files ...')
    t01 = datetime.now() 
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','hycom_ncfiles_to_pickle',yyyymmdd_hy,pkl_fnm]
    ret1 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    if ret1.returncode != 0:
        print('FATAL')
        print('hycom_ncfiles_to_pickle did not work!')
        print('exiting driver_LV1 now!')
        sys.exit(1)

    print('did subprocess run correctly? ' + str(ret1.returncode) + ' (0=yes,1=no)')
    print('...done.')
    t02 = datetime.now() 
    print('this took:')
    print(t02-t01)
    print('\n')
    dt_process =[]
    dt_process.append(t02-t01)

    # now we are back to the previous code...
    # now we are back to the previous code...
    use_ncks = 1 # flag to get data using ncks. if =0, then a pre saved pickle file is loaded.
    use_pckl_sav = 1
    sv_ocnR_pkl_file=1
    fr_ocnR_pkl_file=1
    frm_ICpkl_file = 1
    frm_BCpkl_file = 1
    fr_ocnR_pkl_file=1

    # what are we going to plot?
    plot_ocn = 0
    plot_ocnr = 0
    plot_atm = 0
    plot_all_atm = 0
    plot_ocn_icnc= 0
    load_plot_atm= 0

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
        cmd_list = ['python','-W','ignore','plotting_functions.py','plot_ocn_fields_from_dict_pckl',fn_pckl,pkl_fnm]
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

    t02 = datetime.now()    
    dt_plotting.append(t02-t01)

    # put the ocn data on the roms grid
    t01 = datetime.now()
    os.chdir('../sdpm_py_util')
    hy_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocn_tmp_pckl_file']
    print('putting the hycom data in ' + hy_pckl + ' on the roms grid...')
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','make_all_tmp_pckl_ocnR_files_1hrzeta',pkl_fnm]
    os.chdir('../sdpm_py_util')
    ret1 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    if ret1.returncode != 0:
        print('FATAL')
        print('make_all_tmp_pckl_ocnR_files_1hrzeta did not run correctly')
        print('exiting driver_LV1 now!')
        sys.exit(1)

    print('subprocess return code? ' + str(ret1.returncode) +  ' (0=good)')

    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','print_maxmin_HYrm_pickles',pkl_fnm]
    os.chdir('../sdpm_py_util')
    ret1 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('driver_run_forecast_LV1: done with hycom_to_roms_latlon')
    t02 = datetime.now()
    dt_process.append(t02-t01)
        
    if plot_ocnr == 1:
        print('plotting LV1 ocn_R_fields...')
        cmd_list = ['python','-W','ignore','plotting_functions.py','plot_ocn_R_fields_pckl',pkl_fnm]
        os.chdir('../sdpm_py_util')
        ret1 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('subprocess return code? ' + str(ret1.returncode) +  ' (0=good)')

    t03 = datetime.now()    
    dt_plotting.append(t03-t02)

    t04 = datetime.now()

    # make the depth pickle file
    print('making the depth pickle file...')
    fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']
    print(fname_depths)
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','make_rom_depths_1hrzeta',fname_depths,pkl_fnm]
    os.chdir('../sdpm_py_util')
    ret6 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    print('...done makeing depth pickle file.')
    print('subprocess return code? ' + str(ret6.returncode) +  ' (0=good)')
    print('\n')
    dt_process.append(datetime.now()-t04)

    t01 = datetime.now()
    lv1_use_restart = PFM['lv1_use_restart']

    ocnIC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnIC_tmp_pckl_file']
    if lv1_use_restart==0:
        print('going to save OCN_IC to a pickle file: ' + ocnIC_pckl)
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocnr_2_ICdict_from_tmppkls',ocnIC_pckl,pkl_fnm]
        ret3 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        if ret3.returncode != 0:
            print('FATAL')
            print('ocnr_2_ICdict_from_tmppkls did not run correctly')
            print('exiting driver_LV1 now!')
            sys.exit(1)
        
        print('OCN IC data saved with pickle, correctly? ' + str(ret3.returncode) + ' (0=yes,1=no)')

        print('driver_run_forecast_LV1: done with ocn_r_2_ICdict')
        t02 = datetime.now()
        print('this took:')
        print(t02-t01)
        print('\n')

        ic_file_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ini_file']
        print('making IC file from pickled IC: '+ ic_file_out)
        t03 = datetime.now()
        cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocn_roms_IC_dict_to_netcdf_pckl',ocnIC_pckl,ic_file_out]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        if ret4.returncode != 0:
            print('FATAL')
            print('cn_roms_IC_dict_to_netcdf_pckl did not run correctly')
            print('exiting driver_LV1 now!')
            sys.exit(1)
        print('OCN IC nc data saved, correctly? ' + str(ret4.returncode) + ' (0=yes)')

        print('done makeing IC file.')
        dt_ic = []
        t05 = datetime.now()
        dt_ic.append(t05-t01)
    else:
        print('going to use a restart file for the LV1 IC. Setting this up...')
        cmd_list = ['python','-W','ignore','init_funs_forecast.py','restart_setup','LV1',pkl_fnm]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('...done setting up for restart.')
        print('this took:')
        dt_ic = []
        t05 = datetime.now()
        print(t05-t01)
        dt_ic.append(t05-t01)
        # update the model information as we are now using restarts
        PFM = initfuns_fore.get_model_info(pkl_fnm)
        print('\nGoing to use the file ' + PFM['lv1_ini_file'] + ' to restart the simulation')
        print('with time index ' + str(PFM['lv1_nrrec']))
        print('\n')
        if PFM['lv1_nrrec'] < 0:
            print('WARNING RESTARTING LV1 WILL NOT WORK!!!')

    print('reloading PFM pkl info!')
    PFM = initfuns_fore.get_model_info(pkl_fnm) # refresh this
    
    # get the OCN_BC dictionary
    print('going to save OCN_BC to a pickle file to:')
    t01 = datetime.now()
    t04 = datetime.now()
    ocnBC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']
    print(ocnBC_pckl) 
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocnr_2_BCdict_1hrzeta_from_tmppkls',ocnBC_pckl,pkl_fnm]
    ret4 = subprocess.run(cmd_list)     
    os.chdir('../driver')
    if ret4.returncode != 0:
        print('FATAL')
        print('ocnr_2_BCdict_1hrzeta_from_tmppkls did not run correctly')
        print('exiting driver_LV1 now!')
        sys.exit(1)
    print('OCN BC data saved with pickle, correctly? ' + str(ret4.returncode) + ' (0=yes)')
        
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')

    bc_file_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_bc_file']
    print('making BC nc file from pickled BC: '+ bc_file_out)
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocn_roms_BC_dict_to_netcdf_pckl_1hrzeta',ocnBC_pckl,bc_file_out]
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

    # now for the atm part...
    print('we are now getting the atm data and saving as a dict...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_as_dict',pkl_fnm]
    os.chdir('../sdpm_py_util')
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('...done.')
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')

    dt_download_atm = []
    dt_download_atm.append(t02-t01)
 
    # plot atm here, if we want.
    #if plot_atm == 1:
    #    print('we are now plotting the atm data...')
    #    t01 = datetime.now()
    #    pltfuns.plot_atm_fields()
    #    print('...done with plotting ATM fields')
    #    t02 = datetime.now()
    #    print('this took:')
    #    print(t02-t01)
    #    print('\n')
    #    dt_plotting.append(t02-t01)

    # put the atm data on the roms grid, and rotate the velocities
    # everything in this dict turn into the atm.nc file
    print('we are now putting the atm data on the roms LV1 grid...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_on_roms_grid',str(level),pkl_fnm]
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

    # more atm plotting could be done here

    # fn_out is the name of the atm.nc file used by roms
    fn_atm_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_atm_file'] # LV1 atm forcing filename
    print('we are now saving ATM LV1 to ' + fn_atm_out + ' ...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level),pkl_fnm,'fore']
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
    #yyyymmdd = MI['yyyymmdd']
    #hhmm = MI['hhmm']
    os.chdir('../sdpm_py_util')
    runfuns.make_LV1_dotin_and_SLURM( pkl_fnm , 'fore' )
    print('...done.\n')

    # run command will be
    print('now running roms LV1 with slurm.')
    print('using ' + str(PFM['gridinfo']['L1','nnodes']) + ' nodes.')
    print('Ni = ' + str(PFM['gridinfo']['L1','ntilei']) + ', NJ = ' + str(PFM['gridinfo']['L1','ntilej']))
    runfuns.run_slurm_LV1( pkl_fnm , 'fore')
    print('...done.')
    os.chdir('../driver')
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    print('\n')
    dt_roms = []
    dt_roms.append(t02-t01)

    t01=datetime.now()
    dt_plotting.append(datetime.now()-t01)

    dt_LV1 = {}
    dt_LV1['roms'] = dt_roms
    dt_LV1['ic'] = dt_ic
    dt_LV1['bc'] = dt_bc
    dt_LV1['atm'] = dt_atm
    dt_LV1['plotting'] = dt_plotting
    dt_LV1['process'] = dt_process
    dt_LV1['download_atm'] = dt_download_atm
    dt_LV1['download_ocn'] = dt_download_ocn
    dt_LV1['hycom_t0'] = yyyymmdd_hy

    fn_timing = PFM['lv1_run_dir'] + '/LV1_timing_info.pkl'
    with open(fn_timing,'wb') as fout:
        pickle.dump(dt_LV1,fout)
        print('OCN_LV1 timing info dict saved with pickle to: ',fn_timing)


def run_fore_LV2(pkl_fnm):
    import init_funs_forecast as initfuns
    MI = initfuns.get_model_info(pkl_fnm)

    level = 2
    print('\nwe are now putting the fore atm data on the roms LV2 grid...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_on_roms_grid',str(level),pkl_fnm]
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
    fn_atm_out = MI['lv2_forc_dir'] + '/' + MI['lv2_atm_file'] # LV1 atm forcing filename
    print('we are now saving ATM LV2 to ' + fn_atm_out + ' ...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level),pkl_fnm,'fore']
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

    # make the LV2_OCN_BC.pkl file
    t1 = datetime.now()
    t01 = datetime.now()
    print('driver_run_forcast_LV2: saving LV2_OCN_BC pickle file')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','mk_LV2_BC_dict_edges',str(level),pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_LV2:  done with writing LV2_OCN_BC.pkl file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')

    # convert LV2_BC.pkl to LV2_BC.nc
    t1 = datetime.now()
    lv2_ocnBC_pckl = MI['lv2_forc_dir'] + '/' + MI['lv2_ocnBC_tmp_pckl_file']
    lv2_bc_file_out = MI['lv2_forc_dir'] + '/' + MI['lv2_bc_file']
    print('driver_run_forcast_LV2: saving LV2_OCN_BC netcdf file')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocn_roms_BC_dict_to_netcdf_pckl',lv2_ocnBC_pckl,lv2_bc_file_out]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_LV2:  done with writing LV2_OCN_BC.nc file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')
    dt_bc = []
    dt_bc.append(t2-t01)

    # make and save the LV2_IC.pkl file
    dt_ic = []
    t1=datetime.now()
    t01 = datetime.now()
    if MI['lv2_use_restart']==0:
        print('driver_run_forcast_LV2: making and saving LV2_OCN_IC pickle file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','mk_LV2_IC_dict',str(level),pkl_fnm]
        ret5 = subprocess.run(cmd_list)   
        print('return code: ' + str(ret5.returncode) + ' (0=good)')  
        os.chdir('../sdpm_py_util')
        print('driver_run_forecast_LV2:  done with writing LV2_OCN_IC.pkl file.') 
        print('this took:')
        t2 = datetime.now()
        print(t2-t1)
        print('\n')

        # convert the LV2_IC.pkl file to LV2_IC.nc
        t1=datetime.now()
        lv2_ocnIC_pckl = MI['lv2_forc_dir'] + '/' + MI['lv2_ocnIC_tmp_pckl_file']
        lv2_ic_file_out = MI['lv2_forc_dir'] + '/' + MI['lv2_ini_file']
        print('driver_run_forcast_LV2: saving LV2_OCN_IC netcdf file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocn_roms_IC_dict_to_netcdf_pckl',lv2_ocnIC_pckl,lv2_ic_file_out]
        ret5 = subprocess.run(cmd_list)   
        print('return code: ' + str(ret5.returncode) + ' (0=good)')  
        os.chdir('../sdpm_py_util')
        print('driver_run_forecast_L21:  done with writing LV2_OCN_IC.nc file.') 
        print('this took:')
        t2 = datetime.now()
        print(t2-t1)
        print('\n')
        dt_ic.append(t2-t01)
    else:
        print('going to use a restart file for the LV2 IC. Setting this up...')
        cmd_list = ['python','-W','ignore','init_funs_forecast.py','restart_setup','LV2',pkl_fnm]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('...done setting up for restart.')
        print('this took:')
        dt_ic = []
        t05 = datetime.now()
        print(t05-t01)
        dt_ic.append(t05-t01)
        MI = initfuns.get_model_info(pkl_fnm)
        print('\nGoing to use the file ' + MI['lv2_ini_file'] + ' to restart the simulation')
        print('with time index ' + str(MI['lv2_nrrec']))
        print('\n')
        if MI['lv2_nrrec'] < 0:
            print('WARNING RESTARTING LV2 WILL NOT WORK!!!')

    # now make .in and .sb for roms, and run roms...
    print('making .in and .sb...')
    t1=datetime.now()
    os.chdir('../sdpm_py_util')
    runfuns.make_LV2_dotin_and_SLURM( pkl_fnm , 'fore')
    print('...done')
    # run command will be
    print('now running roms LV2 with slurm.')
    print('using ' + str(MI['gridinfo']['L2','nnodes']) + ' nodes.')
    print('Ni = ' + str(MI['gridinfo']['L2','ntilei']) + ', NJ = ' + str(MI['gridinfo']['L2','ntilej']))
    print('working...')
    runfuns.run_slurm_LV2( pkl_fnm , 'fore')
    os.chdir('../driver')
    print('running ROMS took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')
    dt_roms = []
    dt_roms.append(t2-t1)

    dt_plotting=[]
    t11=datetime.now()
    t22=datetime.now()
    dt_plotting.append(t22-t11)



    dt_LV2 = {}
    dt_LV2['roms'] = dt_roms
    dt_LV2['ic'] = dt_ic
    dt_LV2['bc'] = dt_bc
    dt_LV2['atm'] = dt_atm
    dt_LV2['plotting'] = dt_plotting

    fn_timing = MI['lv2_run_dir'] + '/LV2_timing_info.pkl'
    with open(fn_timing,'wb') as fout:
        pickle.dump(dt_LV2,fout)
        print('OCN_LV2 timing info dict saved with pickle to: ',fn_timing)


def run_fore_LV3(pkl_fnm):
    import init_funs_forecast as initfuns
    MI = initfuns.get_model_info(pkl_fnm)

    level = 3
    print('\nwe are now putting the fore atm data on the roms LV3 grid...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','get_atm_data_on_roms_grid',str(level),pkl_fnm]
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
    fn_atm_out = MI['lv3_forc_dir'] + '/' + MI['lv3_atm_file'] # LV1 atm forcing filename
    print('we are now saving ATM LV3 to ' + fn_atm_out + ' ...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level),pkl_fnm,'fore']
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

    # make the LV2_OCN_BC.pkl file
    t1 = datetime.now()
    t01 = datetime.now()
    print('driver_run_forcast_LV3: saving LV3_OCN_BC pickle file')
    os.chdir('../sdpm_py_util')
    # note mk_LV2_BC_dict_edges works for LV2,3,4 !!! bad function name
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','mk_LV2_BC_dict_edges',str(level),pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_LV3:  done with writing LV3_OCN_BC.pkl file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')

    # convert LV2_BC.pkl to LV2_BC.nc
    t1 = datetime.now()
    lv3_ocnBC_pckl = MI['lv3_forc_dir'] + '/' + MI['lv3_ocnBC_tmp_pckl_file']
    lv3_bc_file_out = MI['lv3_forc_dir'] + '/' + MI['lv3_bc_file']
    print('driver_run_forcast_LV3: saving LV3_OCN_BC netcdf file')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocn_roms_BC_dict_to_netcdf_pckl',lv3_ocnBC_pckl,lv3_bc_file_out]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_LV3:  done with writing LV2_OCN_BC.nc file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')
    dt_bc = []
    dt_bc.append(t2-t01)

    # make and save the LV3_IC.pkl file
    dt_ic = []
    t1=datetime.now()
    t01 = datetime.now()
    if MI['lv3_use_restart']==0:
        print('driver_run_forcast_LV3: making and saving LV3_OCN_IC pickle file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','mk_LV2_IC_dict',str(level),pkl_fnm]
        ret5 = subprocess.run(cmd_list)   
        print('return code: ' + str(ret5.returncode) + ' (0=good)')  
        os.chdir('../sdpm_py_util')
        print('driver_run_forecast_LV3:  done with writing LV3_OCN_IC.pkl file.') 
        print('this took:')
        t2 = datetime.now()
        print(t2-t1)
        print('\n')

        # convert the LV2_IC.pkl file to LV2_IC.nc
        t1=datetime.now()
        lv3_ocnIC_pckl = MI['lv3_forc_dir'] + '/' + MI['lv3_ocnIC_tmp_pckl_file']
        lv3_ic_file_out = MI['lv3_forc_dir'] + '/' + MI['lv3_ini_file']
        print('driver_run_forcast_LV3: saving LV3_OCN_IC netcdf file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocn_roms_IC_dict_to_netcdf_pckl',lv3_ocnIC_pckl,lv3_ic_file_out]
        ret5 = subprocess.run(cmd_list)   
        print('return code: ' + str(ret5.returncode) + ' (0=good)')  
        os.chdir('../sdpm_py_util')
        print('driver_run_forecast_LV3:  done with writing LV3_OCN_IC.nc file.') 
        print('this took:')
        t2 = datetime.now()
        print(t2-t1)
        print('\n')
        dt_ic.append(t2-t01)
    else:
        print('going to use a restart file for the LV3 IC. Setting this up...')
        cmd_list = ['python','-W','ignore','init_funs_forecast.py','restart_setup','LV3',pkl_fnm]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('...done setting up for restart.')
        print('this took:')
        dt_ic = []
        t05 = datetime.now()
        print(t05-t01)
        dt_ic.append(t05-t01)
        MI = initfuns.get_model_info(pkl_fnm)
        print('\nGoing to use the file ' + MI['lv3_ini_file'] + ' to restart the simulation')
        print('with time index ' + str(MI['lv3_nrrec']))
        print('\n')
        if MI['lv3_nrrec'] < 0:
            print('WARNING RESTARTING LV3 WILL NOT WORK!!!')

    # now make .in and .sb for roms, and run roms...
    print('making .in and .sb...')
    t1=datetime.now()
    os.chdir('../sdpm_py_util')
    runfuns.make_LV3_dotin_and_SLURM( pkl_fnm , 'fore')
    print('...done')
    # run command will be
    print('now running roms LV3 with slurm.')
    print('using ' + str(MI['gridinfo']['L3','nnodes']) + ' nodes.')
    print('Ni = ' + str(MI['gridinfo']['L3','ntilei']) + ', NJ = ' + str(MI['gridinfo']['L3','ntilej']))
    print('working...')
    runfuns.run_slurm_LV3( pkl_fnm , 'fore')
    os.chdir('../driver')
    print('running ROMS took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')
    dt_roms = []
    dt_roms.append(t2-t1)

    dt_plotting = []
    t11=datetime.now()
    t22=datetime.now()
    dt_plotting.append(t22-t11)

    dt_LV3 = {}
    dt_LV3['roms'] = dt_roms
    dt_LV3['ic'] = dt_ic
    dt_LV3['bc'] = dt_bc
    dt_LV3['atm'] = dt_atm
    dt_LV3['plotting'] = dt_plotting

    fn_timing = MI['lv3_run_dir'] + '/LV3_timing_info.pkl'
    with open(fn_timing,'wb') as fout:
        pickle.dump(dt_LV3,fout)
        print('OCN_LV3 timing info dict saved with pickle to: ',fn_timing)


def run_fore_LV4(pkl_fnm):
    import init_funs_forecast as initfuns
    from util_functions import copy_mv_nc_file
    MI = initfuns.get_model_info(pkl_fnm)

    level = 4
    print('\nwe are now putting the fore atm data on the roms LV4 grid...')
    t01 = datetime.now()
    cmd_list = ['python','atm_functions.py','get_atm_data_on_roms_grid',str(level),pkl_fnm]
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
    fn_atm_out = MI['lv4_forc_dir'] + '/' + MI['lv4_atm_file'] # LV1 atm forcing filename
    print('we are now saving ATM LV4 to ' + fn_atm_out + ' ...')
    t01 = datetime.now()
    cmd_list = ['python','-W','ignore','atm_functions.py','atm_roms_dict_to_netcdf',str(level),pkl_fnm,'fore']
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

    # make the LV2_OCN_BC.pkl file
    t1 = datetime.now()
    t01 = datetime.now()
    print('driver_run_forcast_LV4: saving LV4_OCN_BC pickle file')
    os.chdir('../sdpm_py_util')
    # note mk_LV2_BC_dict_edges works for LV2,3,4 !!! bad function name
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','mk_LV2_BC_dict_edges',str(level),pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('return code: ' + str(ret5.returncode) + ' (0=good)')  
    os.chdir('../sdpm_py_util')
    print('driver_run_forecast_LV4:  done with writing LV4_OCN_BC.pkl file.') 
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')

    # convert LV4_BC.pkl to LV4_BC.nc
    t1 = datetime.now()
    lv4_ocnBC_pckl = MI['lv4_forc_dir'] + '/' + MI['lv4_ocnBC_tmp_pckl_file']
    lv4_bc_file_out = MI['lv4_forc_dir'] + '/' + MI['lv4_bc_file']
    print('driver_run_forcast_LV4: saving LV4_OCN_BC netcdf file')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','ocn_roms_BC_dict_to_netcdf_pckl',lv4_ocnBC_pckl,lv4_bc_file_out]
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

    # make and save the LV3_IC.pkl file
    dt_ic = []
    t1=datetime.now()
    t01 = datetime.now()
    if MI['lv4_use_restart']==0:
        print('driver_run_forcast_LV4: making and saving LV4_OCN_IC pickle file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-W','ignore','ocn_funs_forecast.py','mk_LV2_IC_dict',str(level),pkl_fnm]
        ret5 = subprocess.run(cmd_list)   
        print('return code: ' + str(ret5.returncode) + ' (0=good)')  
        os.chdir('../sdpm_py_util')
        print('driver_run_forecast_LV4:  done with writing LV4_OCN_IC.pkl file.') 
        print('this took:')
        t2 = datetime.now()
        print(t2-t1)
        print('\n')

        # convert the LV2_IC.pkl file to LV2_IC.nc
        t1=datetime.now()
        lv4_ocnIC_pckl = MI['lv4_forc_dir'] + '/' + MI['lv4_ocnIC_tmp_pckl_file']
        lv4_ic_file_out = MI['lv4_forc_dir'] + '/' + MI['lv4_ini_file']
        print('driver_run_forcast_LV4: saving LV4_OCN_IC netcdf file')
        os.chdir('../sdpm_py_util')
        cmd_list = ['python','-u','-W','ignore','ocn_funs_forecast.py','ocn_roms_IC_dict_to_netcdf_pckl',lv4_ocnIC_pckl,lv4_ic_file_out]
        ret5 = subprocess.run(cmd_list)   
        print('return code: ' + str(ret5.returncode) + ' (0=good)')  
        os.chdir('../sdpm_py_util')
        print('driver_run_forecast_LV4:  done with writing LV4_OCN_IC.nc file.') 
        print('this took:')
        t2 = datetime.now()
        print(t2-t1)
        print('\n')
        dt_ic.append(t2-t01)
    else:
        print('going to use a restart file for the LV4 IC. Setting this up...')
        cmd_list = ['python','-u','-W','ignore','init_funs_forecast.py','restart_setup','LV4',pkl_fnm]
        os.chdir('../sdpm_py_util')
        ret4 = subprocess.run(cmd_list)     
        os.chdir('../driver')
        print('...done setting up for restart.')
        print('this took:')
        dt_ic = []
        t05 = datetime.now()
        print(t05-t01)
        dt_ic.append(t05-t01)
        MI = initfuns.get_model_info(pkl_fnm)
        print('\nGoing to use the file ' + MI['lv4_ini_file'] + ' to restart the simulation')
        print('with time index ' + str(MI['lv4_nrrec']))
        print('\n')
        if MI['lv4_nrrec'] < 0:
            print('WARNING RESTARTING LV4 WILL NOT WORK!!!')

    t1=datetime.now()
    print('driver_run_forcast_LV4: making clm.nc, nud.nc, and river.nc files...')
    os.chdir('../sdpm_py_util')
    cmd_list = ['python','-u','-W','ignore','ocn_funs_forecast.py','mk_lv4_clm_nc',pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('clm return code: ' + str(ret5.returncode) + ' (0=good)')  
    cmd_list = ['python','-u','-W','ignore','ocn_funs_forecast.py','mk_lv4_nud_nc',pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('nud return code: ' + str(ret5.returncode) + ' (0=good)')  
    cmd_list = ['python','-u','-W','ignore','ocn_funs_forecast.py','mk_lv4_river_nc',pkl_fnm]
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
    cmd_list = ['python','-u','-W','ignore','swan_functions.py','cdip_ncs_to_dict','refresh',pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('cdip to dictionary return code: ' + str(ret5.returncode) + ' (0=good)')  
    fout = MI['lv4_forc_dir'] + '/' + MI['lv4_swan_bnd_file']
    print('making swan .bnd file...')
    cmd_list = ['python','-u','-W','ignore','swan_functions.py','mk_swan_bnd_file',fout,pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('...done. swan bnd file return code: ' + str(ret5.returncode) + ' (0=good)')  
    fout = MI['lv4_forc_dir'] + '/' + MI['lv4_swan_wnd_file']
    print('making swan wnd file...')
    cmd_list = ['python','-u','-W','ignore','swan_functions.py','mk_swan_wnd_file',fout,pkl_fnm]
    ret5 = subprocess.run(cmd_list)   
    print('...done. swan wnd file return code: ' + str(ret5.returncode) + ' (0=good)')  
    t2 = datetime.now()
    print('...done. making swan .bnd and .wnd files took:')
    print(t2-t1)
    dt_sw = []
    dt_sw.append(t2-t1)


    # make all of the dotins
    t1=datetime.now()
    print('making LV4 ocean, swan, coupling .in and .sb...')
    os.chdir('../sdpm_py_util')
    runfuns.make_LV4_coawst_dotins_dotsb(pkl_fnm,'fore')
    print('...done')

    ################
    # run coawst LV4
    print('now running Coawst LV4 with slurm.')
    print('using ' + str(MI['gridinfo']['L4','nnodes']) + ' nodes (= ' + str(36 * MI['gridinfo']['L4','nnodes']) + ' CPUs.)')
    print('using ' + str( MI['gridinfo']['L4','ntilei']*MI['gridinfo']['L4','ntilej'] ) + ' CPUs for ROMS, with tiling:')
    print('Ni = ' + str(MI['gridinfo']['L4','ntilei']) + ', NJ = ' + str(MI['gridinfo']['L4','ntilej']))
    print('and using ' + str(MI['gridinfo']['L4','np_swan']) + ' CPUs for SWAN.')
    print('working...')

    runfuns.run_slurm_LV4(pkl_fnm, 'fore')

    os.chdir('../driver')
    print('...done.')
    print('this took:')
    t2 = datetime.now()
    print(t2-t1)
    print('\n')
    dt_roms = []
    dt_roms.append(t2-t1)

    print('copying and moving files around ...')   
    print('moving LV4 atm and river files to Archive...')
    copy_mv_nc_file('atm','lv4',pkl_fnm)
    copy_mv_nc_file('river','lv4',pkl_fnm)
    print('...done')

    print('making the web netcdf file...')
    cmd_list = ['python','-W','ignore','web_functions.py','full_his_to_essential',fn_hs,fn_gr,pkl_fnm]
    os.chdir('../web_util')
    ret6 = subprocess.run(cmd_list)   
    print('...done making web nc file: ' + str(ret6.returncode) + ' (0=good)')  
    os.chdir('../driver')
    t02 = datetime.now()
    print('this took:')
    print(t02-t01)
    dt_web = []
    dt_web.append(t02-t01)

    dt_plotting = []
    print('now making all of the plots for the entire simulation...')
    t01 = datetime.now()
    
    print('making LV1 history file plots and moving on (Popen)...')
    cmd_list = ['python','-W','ignore','plotting_functions.py','make_all_his_figures','LV1',pkl_fnm]
    os.chdir('../sdpm_py_util')
    pr1 = subprocess.Popen(cmd_list)   
    
    print('making LV2 history file plots and moving on (Popen)...')
    cmd_list = ['python','-W','ignore','plotting_functions.py','make_all_his_figures','LV2',pkl_fnm]
    pr2 = subprocess.Popen(cmd_list)   
    
    print('making LV3 history file plots and moving on (Popen)...')
    cmd_list = ['python','-W','ignore','plotting_functions.py','make_all_his_figures','LV3',pkl_fnm]
    pr3 = subprocess.Popen(cmd_list)   
   
    print('making LV4 history file plots and moving on (Popen)...')
    t01=datetime.now()
    cmd_list = ['python','-W','ignore','plotting_functions.py','make_all_his_figures','LV4',pkl_fnm]
    pr4 = subprocess.run(cmd_list)   
    
    print('doing LV4 dye plots and waiting (Popen)...')
    fn_gr = MI['lv4_grid_file']
    fn_hs = MI['lv4_his_name_full']
    cmd_list = ['python','-W','ignore','plotting_functions.py','make_dye_plots',fn_gr,fn_hs,pkl_fnm]
    pr5 = subprocess.Popen(cmd_list)   
    
    print('waiting for plotting to finish...')
    exit_codes = []
    for pr in [pr1,pr2,pr3,pr4,pr5]:
        exit_codes.append(pr.wait()) # this waits for pr1...pr5 to finish
    
    t02 = datetime.now()
    dt_plotting.append(t02-t01)
    print('...done waiting.')
    print('plotting took:')
    print(t02-t01)
    print('LV1 history plots made correctly: ',exit_codes[0],' (0=yes)')
    print('LV2 history plots made correctly: ',exit_codes[1],' (0=yes)')
    print('LV3 history plots made correctly: ',exit_codes[2],' (0=yes)')
    print('LV4 history plots made correctly: ',exit_codes[3],' (0=yes)')
    print('LV4 dye     plots made correctly: ',exit_codes[4],' (0=yes)')

    os.chdir('../driver')

    #  this replace Falks copy_forecast_to_dataSIO.sh file
    # implement later!
    #t01=datetime.now()
    #print('moving files to Archive and website and making history gifs...')
    #cmd_list = ['python','-W','ignore','util_functions.py','end_of_sim_housekeeping',pkl_fnm]
    #os.chdir('../web_util')
    #ret6 = subprocess.run(cmd_list)   
    #print('...done making web nc file: ' + str(ret6.returncode) + ' (0=good)')  
    #os.chdir('../driver')
    #t02 = datetime.now()
    #print('this took:')
    #print(t02-t01)

    #print('...done')
    #dt_plotting.append(datetime.now()-t01)

    dt_LV4 = {}
    dt_LV4['roms'] = dt_roms
    dt_LV4['ic'] = dt_ic
    dt_LV4['bc'] = dt_bc
    dt_LV4['atm'] = dt_atm
    dt_LV4['plotting'] = dt_plotting
    dt_LV4['swan'] = dt_sw
    dt_LV4['web'] = dt_web

    fn_timing = MI['lv4_run_dir'] + '/LV4_timing_info.pkl'
    with open(fn_timing,'wb') as fout:
        pickle.dump(dt_LV4,fout)
        print('OCN_LV4 timing info dict saved with pickle to: ',fn_timing)

def run_hind_simulation(t1str,lvl,pkl_fnm):
    if lvl == 'LV1':
        os.chdir('../driver')
        print('starting LV1 hindcast with subprocess...')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_hind_LV1',t1str,pkl_fnm]
        ret1 = subprocess.run(cmd_list)   
        print('...finished LV1 hindcast.')
        print('return code: ' + str(ret1.returncode) + ' (0=good)')  
        #run_hind_LV1(t1str,pkl_fnm)
    if lvl == 'LV2':
        os.chdir('../driver')
        print('starting LV2 hindcast with subprocess...')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_hind_LV2',t1str,pkl_fnm]
        ret1 = subprocess.run(cmd_list)   
        print('...finished LV2 hindcast.')
        print('return code: ' + str(ret1.returncode) + ' (0=good)')  
        #run_hind_LV2(t1str,pkl_fnm)
    if lvl == 'LV3':
        os.chdir('../driver')
        print('starting LV3 hindcast with subprocess...')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_hind_LV3',t1str,pkl_fnm]
        ret1 = subprocess.run(cmd_list)   
        print('...finished LV3 hindcast.')
        print('return code: ' + str(ret1.returncode) + ' (0=good)')  
        #run_hind_LV3(t1str,pkl_fnm)

def run_fore_simulation(lvl,pkl_fnm):
    if lvl == 'LV1':
        os.chdir('../driver')
        print('starting LV1 forecast with subprocess...')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_fore_LV1',pkl_fnm]
        ret1 = subprocess.run(cmd_list)   
        print('...finished LV1 forecast.')
        print('return code: ' + str(ret1.returncode) + ' (0=good)')  
        #run_fore_LV1(pkl_fnm)
    if lvl == 'LV2':
        os.chdir('../driver')
        print('starting LV2 forecast with subprocess...')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_fore_LV2',pkl_fnm]
        ret1 = subprocess.run(cmd_list)   
        print('...finished LV2 forecast.')
        print('return code: ' + str(ret1.returncode) + ' (0=good)')  
        #run_fore_LV2(pkl_fnm)
    if lvl == 'LV3':
        os.chdir('../driver')
        print('starting LV3 forecast with subprocess...')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_fore_LV3',pkl_fnm]
        ret1 = subprocess.run(cmd_list)   
        print('...finished LV3 forecast.')
        print('return code: ' + str(ret1.returncode) + ' (0=good)')  
        #run_fore_LV3(pkl_fnm)
    if lvl == 'LV4':
        os.chdir('../driver')
        print('starting LV4 forecast with subprocess...')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_fore_LV4',pkl_fnm]
        ret1 = subprocess.run(cmd_list)   
        print('...finished LV4 forecast.')
        print('return code: ' + str(ret1.returncode) + ' (0=good)')  
        #run_fore_LV4(pkl_fnm)


if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])

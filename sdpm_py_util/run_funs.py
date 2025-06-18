"""
This is designed to work with the BLANK.in file, replacing things like
$whatever$ with meaningful values.

!!!            everything is read in from PFM_info.                   !!!!!!
!!!            the values in PFM_info are set in get_PFM_info.py      !!!!!!

--- to reload a module ---
import sys, importlib
importlib.reload(sys.modules['foo'])
from foo import bar
"""

import sys
from datetime import timedelta
sys.path.append('../sdpm_py_util')
import os
import subprocess

##############

def  make_LV1_dotin_and_SLURM( pkl_fnm , mod_type ):
    print(' --- making dot_in and dot_sb --- ')  # + Ldir['date_string'])

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    print('loading model info...')
    PFM = initfuns.get_model_info(pkl_fnm)
    print('...done')
    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    # this is where the varinfo.yaml file is located.
    # the original location was 
    # VARNAME = /home/matt/models/roms/ROMS/External/varinfo.yaml
    # but now it is
    D['roms_varinfo_dir'] = PFM['lv1_run_dir'] +  '/LV1_varinfo.yaml'  ## FIX!

    # grid info
    D['ncols']  = PFM['gridinfo']['L1','Lm']
    D['nrows']  = PFM['gridinfo']['L1','Mm']
    D['nz']     = PFM['stretching']['L1','Nz']
    D['ntilei'] = PFM['gridinfo']['L1','ntilei']
    D['ntilej'] = PFM['gridinfo']['L1','ntilej']
    D['np']     = PFM['gridinfo']['L1','np']
    D['nnodes'] = PFM['gridinfo']['L1','nnodes']

    # timing info
    dtsec         = PFM['tinfo']['L1','dtsec']
    D['ndtfast']  = PFM['tinfo']['L1','ndtfast']
    forecast_days = PFM['tinfo']['L1','forecast_days']  #   forecast_days=2;
    days_to_run   = forecast_days  #float(Ldir['forecast_days'])
    his_interval  = PFM['outputinfo']['L1','his_interval']
    rst_interval  = PFM['outputinfo']['L1','rst_interval']

    # a string version of dtsec, for the .in file
    if dtsec == int(dtsec):  ## should this be a floating point number?  it could be 2.5 etc.
        dt = str(dtsec) + '.0d0'
        D['dt']=dt
    else:
        dt = str(dtsec) + 'd0'
        D['dt'] = dt

    # this is the number of time steps to run runs for.  dtsec is BC timestep
    D['ntimes']  = int(days_to_run*86400/dtsec)
    D['ninfo']   = 1 # how often to write info to the log file (# of time steps)
    D['nhis']    = int(his_interval/dtsec) # how often to write to the history files
    D['ndefhis'] = PFM['ndefhis'] # how often, in time steps, to create new history files, =0 only one his.nc file
    D['nrst']    = int(rst_interval*86400/dtsec)
    D['ndia']    = 60 # write to log file every time step.
    D['ndefdia'] = 60

    #date_string_yesterday = fdt_yesterday.strftime(Lfun.ds_fmt)
    t0          = PFM['modtime0']
    t2          = PFM['sim_time_1']
    dt          = (t2-t0)/timedelta(days=1) # days since 19990101
    D['dstart'] = str(dt) + 'd0' # this is in the form xxxxx.5 due to 12:00 start time for hycom

        # Paths to forcing various file locations
    D['lv1_grid_dir']      = PFM['lv1_grid_dir']
    D['lv1_run_dir']       = PFM['lv1_run_dir']     
    D['lv1_forc_dir']      = PFM['lv1_forc_dir']   
    D['lv1_his_dir']       = PFM['lv1_his_dir']
    D['lv1_his_name_full'] = PFM['lv1_his_name_full']
    D['lv1_rst_name_full'] = PFM['lv1_rst_name_full'] 
    D['lv1_max_time'] = PFM['lv1_max_time_str']

    start_type = PFM['lv1_use_restart'] # 0=new solution. 1=from a restart file
    if start_type == 1:
        nrrec        = str(PFM['lv1_nrrec']) # 
        lv1_ini_dir  = PFM['restart_files_dir']
    else: 
        nrrec        = '0' # '0' for a new solution
        lv1_ini_dir  = PFM['lv1_forc_dir']

    ininame     = PFM['lv1_ini_file']  # start from ini file
    D['nrrec']        = nrrec
    D['lv1_ini_file'] = lv1_ini_dir + '/' + ininame
    D['lv1_bc_file']  = PFM['lv1_forc_dir'] + '/' + PFM['lv1_bc_file']    

 
    D['vtransform']  = PFM['stretching']['L1','Vtransform']
    D['vstretching'] = PFM['stretching']['L1','Vstretching']
    D['theta_s']     = str( PFM['stretching']['L1','THETA_S'] ) + 'd0' #'8.0d0'
    D['theta_b']     = str( PFM['stretching']['L1','THETA_B'] ) + 'd0' #'3.0d0'
    D['tcline']      = str( PFM['stretching']['L1','TCLINE'] )  + 'd0' #'50.0d0'

    lv1_infile_local  = 'LV1_forecast_run.in'
    lv1_logfile_local = 'LV1_forecast.log'
    lv1_sbfile_local  = 'LV1_SLURM.sb'
    D['lv1_infile_local']  = lv1_infile_local
    D['lv1_logfile_local'] = lv1_logfile_local
    D['lv1_tides_file']    = PFM['lv1_tides_file']

    D['lv1_executable'] = PFM['executable_dir'] + PFM['lv1_executable']
 
    # we assume that we are in the PFM/sdpm_py_util/ directory.
    # might want a check to see if you are here. If not, then
    # cd here?
    dot_in_dir   = '.'
    blank_infile = dot_in_dir +'/' +  'LV1_BLANK.in'
    if D['lv1_executable'] == '/scratch/PFM_Simulations/executables/LV3_romsM_INTEL':
        blank_sbfile = dot_in_dir +'/' +  'LV1_SLURM_intel_BLANK.sb'
    else:        
        blank_sbfile = dot_in_dir +'/' +  'LV1_SLURM_BLANK.sb'
    
    print('for this LV1 simulation')
    print('history file made will be:')
    print(D['lv1_his_name_full'])
    print('restart file made will be:')
    print(D['lv1_rst_name_full'])
    print('the ini file used is:')
    print(D['lv1_ini_file'])
    print('we are using')
    print(D['lv1_executable'])
    
    lv1_infile   = D['lv1_run_dir'] + '/' + lv1_infile_local
    lv1_sbfile   = D['lv1_run_dir'] + '/' + lv1_sbfile_local

    ## create lv1_infile_local .in ##########################
    f  = open( blank_infile,'r')
    f2 = open( lv1_infile,'w')   # change this name to be LV1_forecast_yyyymmddd_HHMMZ.in
    for line in f:
        for var in D.keys():
            if '$'+var+'$' in line:
                line2 = line.replace('$'+var+'$', str(D[var]))
                line = line2 # needed because we loop over all "var" for line
            else:
                line2 = line
        f2.write(line2)

    f.close()
    f2.close()

    ## create slurb lv1 .sb file  ##########################
    f  = open( blank_sbfile,'r')
    f2 = open( lv1_sbfile,'w')   # change this name to be LV1_forecast_yyyymmddd_HHMMZ.in
    for line in f:
        for var in D.keys():
            if '$'+var+'$' in line:
                line2 = line.replace('$'+var+'$', str(D[var]))
                line = line2 # needed because we loop over all "var" for line
            else:
                line2 = line
        f2.write(line2)

    f.close()
    f2.close()




def run_slurm_LV1( pkl_fnm , mod_type):

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    PFM = initfuns.get_model_info( pkl_fnm )
    cwd = os.getcwd()
    os.chdir(PFM['lv1_run_dir'])
    print('run_slurm_LV1: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV1_SLURM.sb']
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(proc)
    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')
    print('run_slurm_LV1: run command: ', cmd_list )

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc

from pathlib import Path
import sys
from datetime import datetime, timedelta
from get_PFM_info import get_PFM_info


def  make_LV2_dotin_and_SLURM( pkl_fnm ):

    PFM = initfuns.get_model_info(pkl_fnm)

    print(' --- making dot_in and dot_sb --- ')  # + Ldir['date_string'])

    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    # this is where the varinfo.yaml file is located.
    # the original location was 
    # VARNAME = /home/matt/models/roms/ROMS/External/varinfo.yaml
    # but now it is
    D['roms_varinfo_dir'] = PFM['lv2_run_dir'] +  '/LV2_varinfo.yaml'  ## FIX!

    # grid info
    D['ncols']  = PFM['gridinfo']['L2','Lm']
    D['nrows']  = PFM['gridinfo']['L2','Mm']
    D['nz']     = PFM['stretching']['L2','Nz']     # number of vertical levels: 40
    D['ntilei'] = PFM['gridinfo']['L2','ntilei']
    D['ntilej'] = PFM['gridinfo']['L2','ntilej']
    D['np']     = PFM['gridinfo']['L2','np']
    D['nnodes'] = PFM['gridinfo']['L2','nnodes']

    # timing info
    dtsec         = PFM['tinfo']['L2','dtsec']
    D['ndtfast']  = PFM['tinfo']['L2','ndtfast']
    forecast_days = PFM['tinfo']['L2','forecast_days']  #   forecast_days=2;
    days_to_run   = forecast_days  #float(Ldir['forecast_days'])
    his_interval  = PFM['outputinfo']['L2','his_interval']
    rst_interval  = PFM['outputinfo']['L2','rst_interval']

    # a string version of dtsec, for the .in file
    if dtsec == int(dtsec):  ## should this be a floating point number?  it could be 2.5 etc.
        dt = str(dtsec) + '.0d0'
        D['dt']=dt
    else:
        dt = str(dtsec) + 'd0'
        D['dt'] = dt

    # this is the number of time steps to run runs for.  dtsec is BC timestep
    D['ntimes']  = int(days_to_run*86400/dtsec)
    D['ninfo']   = 1 # how often to write info to the log file (# of time steps)
    D['nhis']    = int(his_interval/dtsec) # how often to write to the history files
    D['ndefhis'] = PFM['ndefhis'] # how often, in time steps, to create new history files, =0 only one his.nc file
    D['nrst']    = int(rst_interval*86400/dtsec)
    D['ndia']    = 60 # write to log file every time step.
    D['ndefdia'] = 60

    #date_string_yesterday = fdt_yesterday.strftime(Lfun.ds_fmt)
    t0          = PFM['modtime0']
    t2          = PFM['sim_time_1']
    dt          = (t2-t0)/timedelta(days=1) # days since 19990101
    D['dstart'] = str(dt) + 'd0' # this is in the form xxxxx.5 due to 12:00 start time for hycom

    # Paths to forcing various file locations
    D['lv2_grid_dir'] = PFM['lv2_grid_dir']
    D['lv2_run_dir']  = PFM['lv2_run_dir']     
    D['lv2_forc_dir'] = PFM['lv2_forc_dir']   
    D['lv2_his_dir']  = PFM['lv2_his_dir']
    D['lv2_max_time'] = PFM['lv2_max_time_str']

    D['lv2_his_name_full'] = PFM['lv2_his_name_full']
    D['lv2_rst_name_full'] = PFM['lv2_rst_name_full'] 
    
    start_type = PFM['lv2_use_restart'] # 0=new solution. 1=from a restart file
    if start_type == 1:
        nrrec        = str(PFM['lv2_nrrec']) # 
        lv2_ini_dir  = PFM['restart_files_dir']
    else: 
        nrrec        = '0' # '0' for a new solution
        lv2_ini_dir  = PFM['lv2_forc_dir']

    ininame     = PFM['lv2_ini_file']  # start from ini file
    D['nrrec']        = nrrec
    D['lv2_ini_file'] = lv2_ini_dir + '/' + ininame
    D['lv2_bc_file']  = PFM['lv2_forc_dir'] + '/' + PFM['lv2_bc_file']    

    D['vtransform']  = PFM['stretching']['L2','Vtransform']
    D['vstretching'] = PFM['stretching']['L2','Vstretching']
    D['theta_s']     = str( PFM['stretching']['L2','THETA_S'] ) + 'd0' #'8.0d0'
    D['theta_b']     = str( PFM['stretching']['L2','THETA_B'] ) + 'd0' #'3.0d0'
    D['tcline']      = str( PFM['stretching']['L2','TCLINE'] ) + 'd0' #'50.0d0'


    lv2_infile_local       = 'LV2_forecast_run.in'
    lv2_logfile_local      = 'LV2_forecast.log'
    lv2_sbfile_local       = 'LV2_SLURM.sb'
    D['lv2_infile_local']  = lv2_infile_local
    D['lv2_logfile_local'] = lv2_logfile_local

    #lv2_executable      = 'LV2_oceanM'
    #D['lv2_executable'] = PFM['lv2_run_dir'] + '/' + lv2_executable
    D['lv2_executable'] = PFM['executable_dir']  + PFM['lv2_executable']
    
    print('for this LV2 simulation')
    print('history file made will be:')
    print(D['lv2_his_name_full'])
    print('restart file made will be:')
    print(D['lv2_rst_name_full'])
    print('the ini file used is:')
    print(D['lv2_ini_file'])
    print('we are using')
    print(D['lv2_executable'])

    dot_in_dir   = '.'
    blank_infile = dot_in_dir +'/' +  'LV2_BLANK.in'
    if D['lv2_executable'] == '/scratch/PFM_Simulations/executables/LV3_romsM_INTEL':
        blank_sbfile = dot_in_dir +'/' +  'LV2_SLURM_intel_BLANK.sb'
    else:        
        blank_sbfile = dot_in_dir +'/' +  'LV2_SLURM_BLANK.sb'

    lv2_infile   = D['lv2_run_dir'] + '/' + lv2_infile_local
    lv2_sbfile   = D['lv2_run_dir'] + '/' + lv2_sbfile_local

    ## create lv2_infile_local .in ##########################
    f  = open( blank_infile,'r')
    f2 = open( lv2_infile,'w')   # change this name to be LV2_forecast_yyyymmddd_HHMMZ.in
    for line in f:
        for var in D.keys():
            if '$'+var+'$' in line:
                line2 = line.replace('$'+var+'$', str(D[var]))
                line = line2 # needed because we loop over all "var" for line
            else:
                line2 = line
        f2.write(line2)

    f.close()
    f2.close()


    ## create slurb lv2 .sb file  ##########################
    f  = open( blank_sbfile,'r')
    f2 = open( lv2_sbfile,'w')   # change this name to be LV2_forecast_yyyymmddd_HHMMZ.in
    for line in f:
        for var in D.keys():
            if '$'+var+'$' in line:
                line2 = line.replace('$'+var+'$', str(D[var]))
                line = line2 # needed because we loop over all "var" for line
            else:
                line2 = line
        f2.write(line2)

    f.close()
    f2.close()

def run_slurm_LV2( pkl_fnm ):
    PFM = initfuns.get_model_info( pkl_fnm )
    cwd = os.getcwd()
    os.chdir(PFM['lv2_run_dir'])
    print('run_slurm_LV2: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV2_SLURM.sb']
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #proc = subprocess.run("/cm/shared/apps/slurm/current/bin/sbatch  --wait LV1_SLURM.sb")

    print(proc)
    print('run_slurm_LV2: run command: ', cmd_list )
    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc

def  make_LV3_dotin_and_SLURM( pkl_fnm ):
    PFM = initfuns.get_model_info( pkl_fnm )

    print(' --- making dot_in and dot_sb --- ')  # + Ldir['date_string'])

    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    # this is where the varinfo.yaml file is located.
    # the original location was 
    # VARNAME = /home/matt/models/roms/ROMS/External/varinfo.yaml
    # but now it is
    D['roms_varinfo_dir'] = PFM['lv3_run_dir'] +  '/LV3_varinfo.yaml'  ## FIX!

    # grid info
    D['ncols']  = PFM['gridinfo']['L3','Lm']
    D['nrows']  = PFM['gridinfo']['L3','Mm']
    D['nz']     = PFM['stretching']['L3','Nz']     # number of vertical levels: 40
    D['ntilei'] = PFM['gridinfo']['L3','ntilei']
    D['ntilej'] = PFM['gridinfo']['L3','ntilej']
    D['np']     = PFM['gridinfo']['L3','np']
    D['nnodes'] = PFM['gridinfo']['L3','nnodes']

    # timing info
    dtsec         = PFM['tinfo']['L3','dtsec']
    D['ndtfast']  = PFM['tinfo']['L3','ndtfast']
    forecast_days = PFM['tinfo']['L3','forecast_days']  #   forecast_days=2;
    days_to_run   = forecast_days  #float(Ldir['forecast_days'])
    his_interval  = PFM['outputinfo']['L3','his_interval']
    rst_interval  = PFM['outputinfo']['L3','rst_interval']

    # a string version of dtsec, for the .in file
    if dtsec == int(dtsec):  ## should this be a floating point number?  it could be 2.5 etc.
        dt = str(dtsec) + '.0d0'
        D['dt']=dt
    else:
        dt = str(dtsec) + 'd0'
        D['dt'] = dt

    # this is the number of time steps to run runs for.  dtsec is BC timestep
    D['ntimes']  = int(days_to_run*86400/dtsec)
    D['ninfo']   = 1 # how often to write info to the log file (# of time steps)
    D['nhis']    = int(his_interval/dtsec) # how often to write to the history files
    D['ndefhis'] = PFM['ndefhis'] # how often, in time steps, to create new history files, =0 only one his.nc file
    D['nrst']    = int(rst_interval*86400/dtsec)
    D['ndia']    = 60 # write to log file every time step.
    D['ndefdia'] = 60

    #date_string_yesterday = fdt_yesterday.strftime(Lfun.ds_fmt)
    t0          = PFM['modtime0']
    t2          = PFM['sim_time_1']
    dt          = (t2-t0)/timedelta(days=1) # days since 19990101
    D['dstart'] = str(dt) + 'd0' # this is in the form xxxxx.5 due to 12:00 start time for hycom

        # Paths to forcing various file locations
    D['lv3_grid_dir'] = PFM['lv3_grid_dir']
    D['lv3_run_dir']  = PFM['lv3_run_dir']     
    D['lv3_forc_dir'] = PFM['lv3_forc_dir']   
    D['lv3_his_dir']  = PFM['lv3_his_dir']
    D['lv3_max_time'] = PFM['lv3_max_time_str']

    D['lv3_his_name_full'] = PFM['lv3_his_name_full']
    D['lv3_rst_name_full'] = PFM['lv3_rst_name_full'] 
    
    start_type = PFM['lv3_use_restart'] # 0=new solution. 1=from a restart file
    
    if start_type == 1:
        nrrec        = str(PFM['lv3_nrrec']) # 
        lv3_ini_dir  = PFM['restart_files_dir']
    else: 
        nrrec        = '0' # '0' for a new solution
        lv3_ini_dir  = PFM['lv3_forc_dir']

    ininame     = PFM['lv3_ini_file']  # start from ini file
    D['nrrec']        = nrrec
    D['lv3_ini_file'] = lv3_ini_dir + '/' + ininame
    D['lv3_bc_file']  = PFM['lv3_forc_dir'] + '/' + PFM['lv3_bc_file']    

    D['vtransform']  = PFM['stretching']['L3','Vtransform']
    D['vstretching'] = PFM['stretching']['L3','Vstretching']
    D['theta_s']     = str( PFM['stretching']['L3','THETA_S'] ) + 'd0' #'8.0d0'
    D['theta_b']     = str( PFM['stretching']['L3','THETA_B'] ) + 'd0' #'3.0d0'
    D['tcline']      = str( PFM['stretching']['L3','TCLINE'] ) + 'd0' #'50.0d0'


    lv3_infile_local       = 'LV3_forecast_run.in'
    lv3_logfile_local      = 'LV3_forecast.log'
    lv3_sbfile_local       = 'LV3_SLURM.sb'
    D['lv3_infile_local']  = lv3_infile_local
    D['lv3_logfile_local'] = lv3_logfile_local

    D['lv3_executable'] = PFM['executable_dir']  + PFM['lv3_executable']

    print('for this LV3 simulation')
    print('history file made will be:')
    print(D['lv3_his_name_full'])
    print('restart file made will be:')
    print(D['lv3_rst_name_full'])
    print('the ini file used is:')
    print(D['lv3_ini_file'])
    print('we are using')
    print(D['lv3_executable'])

    dot_in_dir   = '.'
    blank_infile = dot_in_dir +'/' +  'LV3_BLANK.in'
    if D['lv3_executable'] == '/scratch/PFM_Simulations/executables/LV3_romsM_INTEL':
        blank_sbfile = dot_in_dir +'/' +  'LV3_SLURM_intel_BLANK.sb'
    else:        
        blank_sbfile = dot_in_dir +'/' +  'LV3_SLURM_BLANK.sb'
    
    lv3_infile   = D['lv3_run_dir'] + '/' + lv3_infile_local
    lv3_sbfile   = D['lv3_run_dir'] + '/' + lv3_sbfile_local


    ## create lv3_infile_local .in ##########################
    f  = open( blank_infile,'r')
    f2 = open( lv3_infile,'w')   # change this name to be LV3_forecast_yyyymmddd_HHMMZ.in
    for line in f:
        for var in D.keys():
            if '$'+var+'$' in line:
                line2 = line.replace('$'+var+'$', str(D[var]))
                line = line2 # needed because we loop over all "var" for line
            else:
                line2 = line
        f2.write(line2)

    f.close()
    f2.close()

    ## create slurb lv3 .sb file  ##########################
    f  = open( blank_sbfile,'r')
    f2 = open( lv3_sbfile,'w')   # change this name to be LV3_forecast_yyyymmddd_HHMMZ.in
    for line in f:
        for var in D.keys():
            if '$'+var+'$' in line:
                line2 = line.replace('$'+var+'$', str(D[var]))
                line = line2 # needed because we loop over all "var" for line
            else:
                line2 = line
        f2.write(line2)

    f.close()
    f2.close()



def run_slurm_LV3( pkl_fnm ):

    PFM = initfuns.get_model_info( pkl_fnm )
    cwd = os.getcwd()
    os.chdir(PFM['lv3_run_dir'])
    print('run_slurm_LV3: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV3_SLURM.sb']
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #proc = subprocess.run("/cm/shared/apps/slurm/current/bin/sbatch  --wait LV1_SLURM.sb")

    print(proc)
    print('run_slurm_LV3: run command: ', cmd_list )
    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc


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

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    print('loading model info...')
    PFM = initfuns.get_model_info(pkl_fnm)
    print('...done')
    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    if mod_type == 'hind':
        D['mod_type']='HINDCAST'
    else:
        D['mod_type']='FORECAST'

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
    if "INTEL" in D['lv1_executable']:
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


def  make_LV2_dotin_and_SLURM( pkl_fnm , mod_type ):

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    PFM = initfuns.get_model_info(pkl_fnm)

    print(' --- making dot_in and dot_sb --- ')  # + Ldir['date_string'])

    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    if mod_type == 'hind':
        D['mod_type']='HINDCAST'
    else:
        D['mod_type']='FORECAST'

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
    if "INTEL" in D['lv2_executable']:
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

def run_slurm_LV2( pkl_fnm , mod_type ):
    print(' --- running LV2 --- ')  # + Ldir['date_string'])

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    PFM = initfuns.get_model_info( pkl_fnm )
    cwd = os.getcwd()
    os.chdir(PFM['lv2_run_dir'])
    print('run_slurm_LV2: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV2_SLURM.sb']
    print('run_slurm_LV2: run command: ', cmd_list )
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #proc = subprocess.run("/cm/shared/apps/slurm/current/bin/sbatch  --wait LV1_SLURM.sb")

    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc

def  make_LV3_dotin_and_SLURM( pkl_fnm , mod_type ):

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    PFM = initfuns.get_model_info( pkl_fnm )

    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    if mod_type == 'hind':
        D['mod_type']='HINDCAST'
    else:
        D['mod_type']='FORECAST'

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
    if "INTEL" in D['lv3_executable']:
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


def run_slurm_LV3( pkl_fnm , mod_type ):
    print(' --- running LV3 --- ')  # + Ldir['date_string'])

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    PFM = initfuns.get_model_info( pkl_fnm )
    cwd = os.getcwd()
    os.chdir(PFM['lv3_run_dir'])
    print('run_slurm_LV3: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV3_SLURM.sb']
    print('run_slurm_LV3: run command: ', cmd_list )
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #proc = subprocess.run("/cm/shared/apps/slurm/current/bin/sbatch  --wait LV1_SLURM.sb")

    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc


def  make_LV4_coawst_dotins_dotsb(pkl_fnm,mod_type):

    import pickle
    import numpy as np

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    PFM = initfuns.get_model_info(pkl_fnm)
    yyyymmddhhmm = PFM['yyyymmdd'] + PFM['hhmm']

    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    if mod_type == 'hind':
        D['mod_type']='HINDCAST'
    else:
        D['mod_type']='FORECAST'

    # this is where the varinfo.yaml file is located.
    # the original location was 
    # VARNAME = /home/matt/models/roms/ROMS/External/varinfo.yaml
    # but now it is
    D['varinfo_full'] = PFM['lv4_coawst_varinfo_full'] 

    # grid info
    D['ncols']  = PFM['gridinfo']['L4','Lm']   # number in x
    D['nrows']  = PFM['gridinfo']['L4','Mm']   # number in y
    D['ncols_swan'] = D['ncols'] + 1
    D['nrows_swan'] = D['nrows'] + 1
    D['nz']     = PFM['stretching']['L4','Nz']     # number of vertical levels: 40
    D['ntilei'] = PFM['gridinfo']['L4','ntilei']
    D['ntilej'] = PFM['gridinfo']['L4','ntilej']
    D['np_roms'] = PFM['gridinfo']['L4','np_roms']
    D['np']     = PFM['gridinfo']['L4','np_tot']
    D['nnodes'] = PFM['gridinfo']['L4','nnodes']
    D['nd']     = PFM['lv4_nwave_dirs']
    D['np_swan'] = PFM['gridinfo']['L4','np_swan']

    D['lv4_clm_file'] = PFM['lv4_forc_dir'] + '/' + PFM['lv4_clm_file']
    D['lv4_nud_file'] = PFM['lv4_forc_dir'] + '/' + PFM['lv4_nud_file']
    D['lv4_river_file'] = PFM['lv4_forc_dir'] +'/' + PFM['lv4_river_file']

    # timing info
    dtsec         = PFM['tinfo']['L4','dtsec']
    D['ndtfast']  = PFM['tinfo']['L4','ndtfast']
#    forecast_days = PFM['tinfo']['L4','forecast_days']  #   forecast_days=2;
    forecast_days = PFM['forecast_days']  #   forecast_days=2;
    days_to_run   = forecast_days  #float(Ldir['forecast_days'])
    his_interval  = PFM['outputinfo']['L4','his_interval']
    rst_interval  = PFM['outputinfo']['L4','rst_interval']

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
    t2          = datetime.strptime( yyyymmddhhmm, '%Y%m%d%H%M')
    dt          = (t2-t0)/timedelta(days=1) # days since 19990101
    D['dstart'] = str(dt) + 'd0' # this is in the form xxxxx.5 due to 12:00 start time for hycom

        # Paths to forcing various file locations
    D['lv4_grid_dir'] = PFM['lv4_grid_dir']
    D['lv4_run_dir']  = PFM['lv4_run_dir']     
    D['lv4_forc_dir'] = PFM['lv4_forc_dir']   
    D['lv4_his_dir']  = PFM['lv4_his_dir']
    D['lv4_grid_file'] = PFM['lv4_grid_file']

    D['lv4_his_name_full'] = PFM['lv4_his_name_full']
    D['lv4_rst_name_full'] = PFM['lv4_rst_name_full'] 
    
    D['swan_grd_full'] = "'" + PFM['lv4_grid_dir'] + '/' + PFM['lv4_swan_grd_file'] + "'"
    D['swan_bot_full'] = "'" + PFM['lv4_grid_dir'] + '/' + PFM['lv4_swan_bot_file'] + "'"      
    D['swan_bnd_full'] = "'" + PFM['lv4_forc_dir'] + '/' +PFM['lv4_swan_bnd_file'] + "'"      
    D['swan_wnd_full'] = "'" + PFM['lv4_forc_dir'] + '/' +PFM['lv4_swan_wnd_file'] + "'"     
    D['atm_dt_hr'] = PFM['atm_dt_hr']

    t1 = PFM['fetch_time']
    #t2 = PFM['fore_end_time']
    t2 = t1 + PFM['forecast_days'] * timedelta(days=1)
    t1_swan_str = t1.strftime("%Y%m%d.%H") + '0000'
    t2_swan_str = t2.strftime("%Y%m%d.%H") + '0000'
    D['swan_t1_str'] = t1_swan_str
    D['swan_t2_str'] = t2_swan_str
    D['swan_dt_sec'] = PFM['lv4_swan_dt_sec']
    D['swan_rst_int_hr'] = PFM['lv4_swan_rst_int_hr']
    D['swan_rst_name_full'] = PFM['lv4_swan_rst_name_full']
    
    start_type = PFM['lv4_use_restart'] # 0=new solution. 1=from a restart file
    if start_type == 1:
        if "lv4_nrrec" in PFM:
            nrrec        = str(PFM['lv4_nrrec']) # 
        else:
            nrrec        = '1'
        lv4_ini_dir  = PFM['restart_files_dir']
    else: 
        nrrec        = '0' # '0' for a new solution
        lv4_ini_dir  = PFM['lv4_forc_dir']

    ininame     = PFM['lv4_ini_file']  # start from ini file
    D['nrrec']        = nrrec
    D['lv4_ini_file'] = lv4_ini_dir + '/' + ininame
    D['lv4_bc_file']  = PFM['lv4_forc_dir'] + '/' + PFM['lv4_bc_file']    

    D['vtransform']  = PFM['stretching']['L4','Vtransform']
    D['vstretching'] = PFM['stretching']['L4','Vstretching']
    D['theta_s']     = str( PFM['stretching']['L4','THETA_S'] ) + 'd0' #'8.0d0'
    D['theta_b']     = str( PFM['stretching']['L4','THETA_B'] ) + 'd0' #'3.0d0'
    D['tcline']      = str( PFM['stretching']['L4','TCLINE'] ) + 'd0' #'50.0d0'
    D['lv4_max_time'] = PFM['lv4_max_time_str']

    if PFM['lv4_model'] == 'ROMS':
        lv4_infile_local   = 'LV4_forecast_run.in'
    if PFM['lv4_model'] == 'COAWST':
        lv4_infile_coupled   = 'LV4_forecast_coupled.in'
        lv4_infile_roms    = 'LV4_forecast_run.in'
        lv4_infile_swan    = 'LV4_forecast_swan.in'

    lv4_logfile_local      = 'LV4_forecast.log'
    lv4_sbfile_local       = 'LV4_SLURM.sb'
    D['lv4_infile_local']  = lv4_infile_coupled
    D['lv4_logfile_local'] = lv4_logfile_local
#    D['lv4_executable']    = PFM['lv4_run_dir'] + '/' + PFM['lv4_exe_name']
    D['lv4_executable'] = PFM['executable_dir']  + PFM['lv4_executable']
    print('we are using')
    print(D['lv4_executable'])
    print('for LV4')


    dot_in_dir   = '.'
    blank_infile = dot_in_dir +'/' +  PFM['lv4_blank_name'] 
    #blank_sbfile = dot_in_dir +'/' +  'LV4_SLURM_BLANK.sb'
    if D['lv4_executable'] == '/scratch/PFM_Simulations/executables/coawstM_intel':
        blank_sbfile = dot_in_dir +'/' +  'LV4_SLURM_intel_BLANK.sb'
    else:        
        blank_sbfile = dot_in_dir +'/' +  'LV4_SLURM_BLANK.sb'
    
    
    blank_coupling = dot_in_dir + '/' + 'LV4_COUPLING_BLANK.in'
    blank_swan     = dot_in_dir + '/' + 'LV4_SWAN_BLANK.in'
    lv4_couple_infile   = D['lv4_run_dir'] + '/' + lv4_infile_coupled
    lv4_swan_infile     =  D['lv4_run_dir'] + '/' + lv4_infile_swan
    lv4_infile = D['lv4_run_dir'] + '/' + lv4_infile_roms
    lv4_sbfile   = D['lv4_run_dir'] + '/' + lv4_sbfile_local
    D['swan_to_roms'] = PFM['swan_to_roms']
    D['lv4_roms_infile'] = lv4_infile
    D['lv4_swan_infile'] = lv4_swan_infile

    D['swan_init_txt'] = PFM['swan_init_txt_full'] # 'ZERO'
                                                    # or 'HOTSTART PFM['restart_files_dir']+swan_file_name

   # need to get wave spectra information for swan.in 
    fn_in = PFM['lv4_forc_dir'] + '/' + PFM['lv4_swan_pckl_file']
    with open(fn_in,'rb') as fp:
        cdip = pickle.load(fp)

    D['freq_min'] = np.min(cdip['f'])    
    D['freq_max'] = np.max(cdip['f'])    
    D['freq_num'] = len(cdip['f'])

    D['angle_min'] = np.min(cdip['dir'])    
    D['angle_max'] = np.max(cdip['dir'])    
    D['angle_num'] = len(cdip['dir'])
    D['angle_dangle'] = cdip['dir'][1] - cdip['dir'][0]    


    ## here is ocean.in 
    ## create lv4_infile_local .in ##########################
    f  = open( blank_infile,'r')
    f2 = open( lv4_infile,'w')   # change this name to be LV3_forecast_yyyymmddd_HHMMZ.in
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

# do the coupling .in 
    f  = open( blank_coupling,'r')
    f2 = open( lv4_couple_infile,'w')   # change this name to be LV3_forecast_yyyymmddd_HHMMZ.in
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

# do the swan .in 
    f  = open( blank_swan,'r')
    f2 = open( lv4_swan_infile,'w')   # change this name to be LV3_forecast_yyyymmddd_HHMMZ.in
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


    ## create slurb lv4 .sb file  ##########################
    f  = open( blank_sbfile,'r')
    f2 = open( lv4_sbfile,'w')   # change this name to be LV3_forecast_yyyymmddd_HHMMZ.in
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

def run_slurm_LV4( pkl_fnm , mod_type):

    import time

    if mod_type == 'hind':
        import init_funs as initfuns
    else: 
        import init_funs_forecast as initfuns

    PFM = initfuns.get_model_info(pkl_fnm)
    cwd = os.getcwd()
    # start the python program that checks for new swan files...
    fname = PFM['lv4_swan_rst_name'] # this is the root name of the swan file
                                     # LV4_swan_rst_202412010600.dat-001 etc
    nfiles = str( PFM['gridinfo']['L4','np_swan'] )
    # how long one pauses to check for swan restart writes, needs to be 1/4 of how often it writes.
    swan_check_freq = str( PFM['lv4_swan_check_freq_sec'] )
#    cmd_list = ['python','-W','ignore','swan_functions.py','check_and_move',fname,swan_check_freq,nfiles,pkl_fnm]
    cmd_list = ['python','swan_functions.py','check_and_move',fname,swan_check_freq,nfiles,pkl_fnm]
    os.chdir('../sdpm_py_util')
    print('starting the background process that looks for modified swan restart files')
    print('we check for modified files every '+ swan_check_freq + ' s (wall time)')
    checking = subprocess.Popen(cmd_list)   
    
    os.chdir(PFM['lv4_run_dir'])
    print('run_slurm_LV4: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV4_SLURM.sb']
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(int(swan_check_freq))  # Check every dt_sec second

    checking.terminate() # turn off the function that moves swan rst files
    print('terminated the swan restart file check and move subprocess.')

    print(proc)
    print('run_slurm_LV4: run command: ', cmd_list )
    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc

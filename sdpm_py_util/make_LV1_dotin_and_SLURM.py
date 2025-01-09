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

from pathlib import Path
import sys
from datetime import datetime, timedelta
from get_PFM_info import get_PFM_info


def  make_LV1_dotin_and_SLURM( PFM , yyyymmddhhmm):

    print(' --- making dot_in and dot_sb --- ')  # + Ldir['date_string'])

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
    t2          = datetime.strptime( yyyymmddhhmm, '%Y%m%d%H%M')
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

    D['lv1_executable'] = PFM['lv1_run_dir'] + '/' + PFM['lv1_executable']
    #D['lv1_executable'] = PFM['lv1_run_dir'] + '/' + lv1_executable

    # we assume that we are in the PFM/sdpm_py_util/ directory.
    # might want a check to see if you are here. If not, then
    # cd here?
    dot_in_dir   = '.'
    blank_infile = dot_in_dir +'/' +  'LV1_BLANK.in'
    blank_sbfile = dot_in_dir +'/' +  'LV1_SLURM_BLANK.sb'
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


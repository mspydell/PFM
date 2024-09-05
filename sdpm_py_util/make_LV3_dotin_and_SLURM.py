"""
This is designed to work with the BLANK.in file, replacing things like
$whatever$ with meaningful values.

!!!            everything is read in from PFM_info.                   !!!!!!
!!!            the values in PFM_info are set in get_PFM_info.py      !!!!!!

"""


from pathlib import Path
import sys
from datetime import datetime, timedelta
from get_PFM_info import get_PFM_info


def  make_LV3_dotin_and_SLURM( PFM , yyyymmddhhmm):

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
    t2          = datetime.strptime( yyyymmddhhmm, '%Y%m%d%H%M')
    dt          = (t2-t0)/timedelta(days=1) # days since 19990101
    D['dstart'] = str(dt) + 'd0' # this is in the form xxxxx.5 due to 12:00 start time for hycom

        # Paths to forcing various file locations
    D['lv3_grid_dir'] = PFM['lv3_grid_dir']
    D['lv3_run_dir']  = PFM['lv3_run_dir']     
    D['lv3_forc_dir'] = PFM['lv3_forc_dir']   
    D['lv3_his_dir']  = PFM['lv3_his_dir']

    D['lv3_his_name_full'] = PFM['lv3_his_name_full']
    D['lv3_rst_name_full'] = PFM['lv3_rst_name_full'] 
    
    start_type = 'new'

    if start_type == 'perfect':
        nrrec       = '-1' # '-1' for a perfect restart
        ininame     = 'ocean_rst.nc' # start from restart file
        lv3_ini_dir = PFM['lv3_run_dir']
    else: 
        nrrec       = '0' # '0' for a history or ini file
        ininame     = PFM['lv3_ini_file']  # start from ini file
        lv3_ini_dir = PFM['lv3_forc_dir']

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

    lv3_executable      = 'LV3_oceanM'
    D['lv3_executable'] = PFM['lv3_run_dir'] + '/' + lv3_executable

    dot_in_dir   = '.'
    blank_infile = dot_in_dir +'/' +  'LV3_BLANK.in'
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


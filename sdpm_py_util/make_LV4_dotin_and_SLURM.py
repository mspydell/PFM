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


def  make_LV4_dotin_and_SLURM( PFM , yyyymmddhhmm):

    print(' --- making dot_in and dot_sb --- ')  # + Ldir['date_string'])

    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    # this is where the varinfo.yaml file is located.
    # the original location was 
    # VARNAME = /home/matt/models/roms/ROMS/External/varinfo.yaml
    # but now it is
    D['roms_varinfo_dir'] = PFM['lv4_run_dir'] +  '/' + PFM['lv4_yaml_file']  ## FIX!

    # grid info
    D['ncols']  = PFM['gridinfo']['L4','Lm']
    D['nrows']  = PFM['gridinfo']['L4','Mm']
    D['nz']     = PFM['stretching']['L4','Nz']     # number of vertical levels: 40
    D['ntilei'] = PFM['gridinfo']['L4','ntilei']
    D['ntilej'] = PFM['gridinfo']['L4','ntilej']
    D['np']     = PFM['gridinfo']['L4','np']
    D['nnodes'] = PFM['gridinfo']['L4','nnodes']

    # timing info
    dtsec         = PFM['tinfo']['L4','dtsec']
    D['ndtfast']  = PFM['tinfo']['L4','ndtfast']
    forecast_days = PFM['tinfo']['L4','forecast_days']  #   forecast_days=2;
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
    
    start_type = 'new'

    if start_type == 'perfect':
        nrrec       = '-1' # '-1' for a perfect restart
        ininame     = 'ocean_rst.nc' # start from restart file
        lv4_ini_dir = PFM['lv4_run_dir']
    else: 
        nrrec       = '0' # '0' for a history or ini file
        ininame     = PFM['lv4_ini_file']  # start from ini file
        lv4_ini_dir = PFM['lv4_forc_dir']

    D['nrrec']        = nrrec
    D['lv4_ini_file'] = lv4_ini_dir + '/' + ininame
    D['lv4_bc_file']  = PFM['lv4_forc_dir'] + '/' + PFM['lv4_bc_file']    

    D['vtransform']  = PFM['stretching']['L4','Vtransform']
    D['vstretching'] = PFM['stretching']['L4','Vstretching']
    D['theta_s']     = str( PFM['stretching']['L4','THETA_S'] ) + 'd0' #'8.0d0'
    D['theta_b']     = str( PFM['stretching']['L4','THETA_B'] ) + 'd0' #'3.0d0'
    D['tcline']      = str( PFM['stretching']['L4','TCLINE'] ) + 'd0' #'50.0d0'


    lv4_infile_local       = 'LV4_forecast_run.in'
    lv4_logfile_local      = 'LV4_forecast.log'
    lv4_sbfile_local       = 'LV4_SLURM.sb'
    D['lv4_infile_local']  = lv4_infile_local
    D['lv4_logfile_local'] = lv4_logfile_local
#    D['lv4_executable']    = PFM['lv4_run_dir'] + '/' + PFM['lv4_exe_name']
    D['lv4_executable'] = PFM['executable_dir']  + PFM['lv4_executable']
    print('we are using')
    print(D['lv4_executable'])
    print('for LV4')

    dot_in_dir   = '.'
    blank_infile = dot_in_dir +'/' +  PFM['lv4_blank_name']
    blank_sbfile = dot_in_dir +'/' +  'LV4_SLURM_BLANK.sb'
    lv4_infile   = D['lv4_run_dir'] + '/' + lv4_infile_local
    lv4_sbfile   = D['lv4_run_dir'] + '/' + lv4_sbfile_local

 



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


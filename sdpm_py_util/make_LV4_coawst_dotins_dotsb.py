"""
This is designed to work with the BLANK.in file, replacing things like
$whatever$ with meaningful values.

!!!            everything is read in from PFM_info.                   !!!!!!
!!!            the values in PFM_info are set in get_PFM_info.py      !!!!!!

"""


from pathlib import Path
import sys
import pickle
import numpy as np
from datetime import datetime, timedelta
from get_PFM_info import get_PFM_info


def  make_LV4_coawst_dotins_dotsb():

    print(' --- making dot_in and dot_sb --- ')  # + Ldir['date_string'])
    PFM = get_PFM_info()
    yyyymmddhhmm = PFM['yyyymmdd'] + PFM['hhmm']

    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

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
    
    D['swan_grd_full'] = "'" + PFM['lv4_grid_dir'] + '/' + PFM['lv4_swan_grd_file'] + "'"
    D['swan_bot_full'] = "'" + PFM['lv4_grid_dir'] + '/' + PFM['lv4_swan_bot_file'] + "'"      
    D['swan_bnd_full'] = "'" + PFM['lv4_forc_dir'] + '/' +PFM['lv4_swan_bnd_file'] + "'"      
    D['swan_wnd_full'] = "'" + PFM['lv4_forc_dir'] + '/' +PFM['lv4_swan_wnd_file'] + "'"     
    D['atm_dt_hr'] = PFM['atm_dt_hr']

    t1 = PFM['fetch_time']
    t2 = PFM['fore_end_time']
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
        print('\nCDIP pickle file loaded')

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

    

"""
This creates and populates directories for ROMS runs on mox or similar.
Its main product is the .in file used for a ROMS run.

It is designed to work with the BLANK.in file, replacing things like
$whatever$ with meaningful values.

To test from ipython on mac:
run make_dot_in -g cas6 -t v00 -x uu0m -r backfill -s continuation -d 2019.07.04 -bu 0 -np 400

If you call with -short_roms True it will create a .in that runs for a shorter time or
writes history files more frequently (exact behavior is in the code below).  This can
be really useful for debugging.

--- to reload a module ---
import sys, importlib
importlib.reload(sys.modules['foo'])
from foo import bar

"""


# NOTE: we limit the imports to modules that exist in python3 on mox
from pathlib import Path
import sys
from datetime import datetime, timedelta
from get_PFM_info import get_PFM_info


def  make_LV2_dotin_and_SLURM( PFM , yyyymmddhhmm):


    pth = Path(__file__).absolute().parent.parent.parent / 'lo_tools' / 'lo_tools'
    if str(pth) not in sys.path:
        sys.path.append(str(pth))

    dot_in_dir = Path(__file__).absolute().parent

    #Ldir = dfun.intro() # this handles all the argument passing


    #fdt = datetime.strptime(Ldir['date_string'], Lfun.ds_fmt)
    #fdt_yesterday = fdt - timedelta(days=1)

    ## This stuff above is all about directories.  The important thing below
    ## is that the run_dir needs to be set and the 

    print(' --- making dot_in for ')  # + Ldir['date_string'])

    # initialize dict to hold values that we will substitute into the dot_in file.
    D = dict()

    D['roms_varinfo_dir'] = PFM['lv2_run_dir'] +  '/LV2_varinfo.yaml'  ## FIX!

    #### USER DEFINED VALUES ####

    #Ldir['run_type'] == 'forecast'

    ## number of grid points in each direction

    #ncols = 251    # Lm in input file
    #nrows = 388   # Mm in input file
    ncols = PFM['gridinfo']['L2','Lm']
    nrows = PFM['gridinfo']['L2','Mm']
    nz   = PFM['stretching']['L2','Nz']     # number of vertical levels: 40

    D['ncols'] = ncols
    D['nrows'] = nrows
    D['nz'] = nz

    # The LV1 TILING shoudl be hard coded
#    ntilei = 6  # number of tiles in I-direction
#    ntilej = 18 # number of tiles in J-direction
#    np = ntilei*ntilej  # total number of processors
#    nnodes = 3  # number of nodes to be used.  not for .in file but for slurm!


    D['ntilei'] = PFM['gridinfo']['L2','ntilei']
    D['ntilej'] = PFM['gridinfo']['L2','ntilej']
    D['np'] = PFM['gridinfo']['L2','np']
    D['nnodes'] = PFM['gridinfo']['L2','nnodes']

    ### time stuff

#    dtsec = 60
    dtsec = PFM['tinfo']['L2','dtsec']
    # time step in seconds (should fit evenly into 3600 sec)
    #if Ldir['blow_ups'] == 0:
    #    dtsec = 60
    #elif Ldir['blow_ups'] == 1:
    #    dtsec = 45
    #elif Ldir['blow_ups'] == 2:
    #    dtsec = 30
    #else:
    #    print('Unsupported number of blow ups: %d' % (Ldir['blow_ups']))

#    D['ndtfast'] = 15
    D['ndtfast'] = PFM['tinfo']['L2','ndtfast']
    forecast_days = PFM['tinfo']['L2','forecast_days']  #   forecast_days=2;
    days_to_run = forecast_days  #float(Ldir['forecast_days'])

    #his_interval = 3600 # seconds to define and write to history files
    #rst_interval = 1 # days between writing to the restart file (e.g. 5)
    his_interval = PFM['outputinfo']['L2','his_interval']
    rst_interval = PFM['outputinfo']['L2','rst_interval']


    #### END USER DEFINED VALUES ####


    # a string version of dtsec, for the .in file
    if dtsec == int(dtsec):  ## should this be a floating point number?  it could be 2.5 etc.
        dt = str(dtsec) + '.0d0'
        D['dt']=dt
    else:
        dt = str(dtsec) + 'd0'
        D['dt'] = dt

        # this is the number of time steps to run runs for.  dtsec is BC timestep
    D['ntimes'] = int(days_to_run*86400/dtsec)
    D['ninfo'] = 1 # how often to write info to the log file (# of time steps)
    D['nhis'] = int(his_interval/dtsec) # how often to write to the history files
    D['ndefhis'] = PFM['ndefhis'] # how often, in time steps, to create new history files, =0 only one his.nc file
    D['nrst'] = int(rst_interval*86400/dtsec)
    D['ndia'] = 60 # write to log file every time step.
    D['ndefdia'] = 60


    #date_string_yesterday = fdt_yesterday.strftime(Lfun.ds_fmt)
    t0 = PFM['modtime0']
    t2 = datetime.strptime( yyyymmddhhmm, '%Y%m%d%H%M')
    dt = (t2-t0)/timedelta(days=1) # days since 19990101
    D['dstart'] = str(dt) + 'd0' # this is in the form xxxxx.5 due to 12:00 start time for hycom

        # Paths to forcing various file locations
    D['lv2_grid_dir'] = PFM['lv2_grid_dir']
    D['lv2_run_dir'] = PFM['lv2_run_dir']     
    D['lv2_forc_dir'] = PFM['lv2_forc_dir']   
    D['lv2_his_dir'] = PFM['lv2_his_dir']

    D['lv2_his_name_full'] = PFM['lv2_his_name_full']
    D['lv2_rst_name_full'] = PFM['lv2_rst_name_full'] 
    
    start_type = 'new'

    if start_type == 'perfect':
        nrrec = '-1' # '-1' for a perfect restart
        ininame = 'ocean_rst.nc' # start from restart file
        lv2_ini_dir = PFM['lv2_run_dir']
    else: 
        nrrec = '0' # '0' for a history or ini file
        ininame = PFM['lv2_ini_file']  # start from ini file
        lv2_ini_dir = PFM['lv2_forc_dir']

    D['nrrec'] = nrrec
    D['lv2_ini_file'] = lv2_ini_dir + '/' + ininame
    D['lv2_bc_file'] = PFM['lv2_forc_dir'] + '/' + PFM['lv2_bc_file']    

 #   vtransform = 2
 #   vstretching = 4
 #   D['vtransform']=vtransform
 #   D['vstretching']=vstretching
 #   D['theta_s']='8.0d0'
 #   D['theta_b']='3.0d0'
 #   D['tcline']='50.0d0'

    D['vtransform'] = PFM['stretching']['L2','Vtransform']
    D['vstretching']= PFM['stretching']['L2','Vstretching']
    D['theta_s']= str( PFM['stretching']['L2','THETA_S'] ) + 'd0' #'8.0d0'
    D['theta_b']= str( PFM['stretching']['L2','THETA_B'] ) + 'd0' #'3.0d0'
    D['tcline']= str( PFM['stretching']['L2','TCLINE'] ) + 'd0' #'50.0d0'


    lv2_infile_local = 'LV2_forecast_run.in'
    lv2_logfile_local = 'LV2_forecast.log'
    lv2_sbfile_local = 'LV2_SLURM.sb'
    D['lv2_infile_local'] = lv2_infile_local
    D['lv2_logfile_local'] = lv2_logfile_local
    D['lv2_tides_file'] = PFM['lv2_tide_dir'] + '/' + PFM['lv2_tide_fname']

    lv2_executable = 'LV2_oceanM'
    D['lv2_executable'] = PFM['lv2_run_dir'] + '/' + lv2_executable

    #/scratch/PFM_Simulations/LV2_Forecast/Run/LV2_oceanM
    #/scratch/PFM_Simulations/LV2_Forecast/Run/LV2_oceanM
    # END DERIVED VALUES
    dot_in_dir = '.'
    blank_infile = dot_in_dir +'/' +  'LV2_BLANK.in'
    blank_sbfile = dot_in_dir +'/' +  'LV2_SLURM_BLANK.sb'

    lv2_infile = D['lv2_run_dir'] + '/' + lv2_infile_local
    lv2_sbfile = D['lv2_run_dir'] + '/' + lv2_sbfile_local


    ## create lv2_infile_local .in ##########################
    f = open( blank_infile,'r')
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
    f = open( blank_sbfile,'r')
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


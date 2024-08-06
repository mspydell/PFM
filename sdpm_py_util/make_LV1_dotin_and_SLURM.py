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


def  make_LV1_dotin_and_SLURM( PFM ):


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

    #D['EX_NAME'] = Ldir['ex_name'].upper()

    # this is where the varinfo.yaml file is located.

    #D['roms_varinfo_dir'] = Ldir['parent'] / 'LO_roms_source_alt' / 'varinfo'  ## FIX!
    #### USER DEFINED VALUES ####

    #Ldir['run_type'] == 'forecast'

    ## number of grid points in each direction

    ncols = 251    # Lm in input file
    nrows = 388   # Mm in input file
    nz =  40     # number of vertical levels
    D['ncols'] = ncols
    D['nrows'] = nrows
    D['nz'] = nz

    # The LV1 TILING shoudl be hard coded
    ntilei = 6  # number of tiles in I-direction
    ntilej = 18 # number of tiles in J-direction
    np = ntilei*ntilej  # total number of processors
    nnodes = 3  # number of nodes to be used.  not for .in file but for slurm!

    D['ntilei'] = ntilei
    D['ntilej'] = ntilej
    D['np'] = np
    D['nnodes'] = nnodes

    ### time stuff

    dtsec = 60

    # time step in seconds (should fit evenly into 3600 sec)
    #if Ldir['blow_ups'] == 0:
    #    dtsec = 60
    #elif Ldir['blow_ups'] == 1:
    #    dtsec = 45
    #elif Ldir['blow_ups'] == 2:
    #    dtsec = 30
    #else:
    #    print('Unsupported number of blow ups: %d' % (Ldir['blow_ups']))

    D['ndtfast'] = 15
    forecast_days=2;
    days_to_run = forecast_days  #float(Ldir['forecast_days'])

    his_interval = 3600 # seconds to define and write to history files
    rst_interval = 1 # days between writing to the restart file (e.g. 5)

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

    D['ninfo'] = int(his_interval/dtsec) # how often to write info to the log file (# of time steps)
    D['nhis'] = int(his_interval/dtsec) # how often to write to the history files
    D['ndefhis'] = D['nhis'] # how often to create new history files
    D['nrst'] = int(rst_interval*86400/dtsec)


        #date_string_yesterday = fdt_yesterday.strftime(Lfun.ds_fmt)
        #D['dstart'] = int(Lfun.datetime_to_modtime(fdt) / 86400.)

        # Paths to forcing various file locations
    D['lv1_grid_dir'] = PFM['lv1_grid_dir']
    D['lv1_run_dir'] = PFM['lv1_run_dir']     
    D['lv1_forc_dir'] = PFM['lv1_forc_dir']   
    D['lv1_his_dir'] = PFM['lv1_his_dir']


    start_type = 'new'

    if start_type == 'perfect':
        nrrec = '-1' # '-1' for a perfect restart
        ininame = 'ocean_rst.nc' # start from restart file
    else: 
        nrrec = '0' # '0' for a history or ini file
        ininame = PFM['ininame']  # start from ini file


    D['nrrec'] = nrrec
    D['lv1_ini_file'] = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ini_file']
    D['lv1_bc_file'] = PFM['lv1_forc_dir'] + '/' + PFM['lv1_bc_file']    

    vtransform = 2
    vstretching = 4
    D['vtransform']=vtransform
    D['vstretching']=vstretching
    D['theta_s']='8.0d0'
    D['theta_b']='3.0d0'
    D['tcline']='50.0d0'

    # the output directory and the one from the day before
    #out_dir = Ldir['roms_out'] / Ldir['gtagex'] / ('f' + Ldir['date_string'])
    #D['out_dir'] = out_dir
    #out_dir_yesterday = Ldir['roms_out'] / Ldir['gtagex'] / ('f' + date_string_yesterday)
    #Lfun.make_dir(out_dir, clean=True) # make sure it exists and is empty

    lv1_infile_local = 'LV1_forecast_run.in'
    lv1_logfile_local = 'LV1_forecast.log'
    lv1_sbfile_local = 'LV1_SLURM.sb'
    D['lv1_infile_local'] = lv1_infile_local
    D['lv1_logfile_local'] = lv1_logfile_local

    lv1_executable = 'lv1_oceanM'
    D['lv1_executable'] = lv1_executable

    # END DERIVED VALUES
    dot_in_dir = '.'
    blank_infile = dot_in_dir +'/' +  'LV1_BLANK.in'
    blank_sbfile = dot_in_dir +'/' +  'LV1_SLURM_BLANK.sb'

    lv1_infile = D['lv1_run_dir'] + '/' + lv1_infile_local
    lv1_sbfile = D['lv1_run_dir'] + '/' + lv1_sbfile_local


    ## create lv1_infile_local .in ##########################
    f = open( blank_infile,'r')
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
    f = open( blank_sbfile,'r')
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


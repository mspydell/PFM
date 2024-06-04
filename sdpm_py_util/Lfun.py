"""
Module of functions for LO version of LiveOcean.

This is purposely kept to a minimum of imports so that it will run with
whatever python3 exists on the large clusters we use for ROMS, e.g. mox and klone.

Re-coded 2022.04.19 to first look for LO_user/get_lo_info.py, and if not found
then look for LO/get_user_info.py.  The goal is to clean out LO_user of all of the code
that the LO developer (MacCready) would edit.  Then we add "hooks" to look for
user versions in LO_user at strategic places, as is done below.
"""
import os, sys, shutil
from pathlib import Path 
from datetime import datetime, timedelta
import importlib.util
from importlib import reload

testing = 1
# get initial version of Ldir when this module is loaded
import get_sdpm_info as gsd

if testing == 1:
    reload(gsd)

# initialize Ldir for this module
#print('here in Lfun.py')
Ldir = gsd.SDP.copy()

ds_fmt = '%Y.%m.%d'
#ds_fmt = '%Y%m%d'

def parserfun(parser):
    import argparse
    parser.add_argument('-g', '--gridname', type=str)   # e.g. cas6
    parser.add_argument('-t', '--tag', type=str)        # e.g. v00
    parser.add_argument('-x', '--ex_name', type=str)    # e.g. uu0mb
    # the ocean model could become a command line argument when we have more than 1 to choose from
    #parser.add_argument('-om', '--ocean_model', type=str, default='hycom') # LS ocean model
    #parser.add_argument('-am', '--atm_model', type=str, default='wrf') # the atmosphere model
    parser.add_argument('-r', '--run_type', type=str, default='backfill')   # forecast or backfill
    parser.add_argument('-s', '--start_type', type=str, default='perfect') # new, perfect, or continuation
    # -0 and -1 only required for -r backfill
    parser.add_argument('-0', '--ds0', type=str)        # e.g. 2019.07.04
    parser.add_argument('-1', '--ds1', type=str, default='') # is set to ds0 if omitted
    parser.add_argument('-np', '--np_num', type=int) # e.g. 200, number of cores
    parser.add_argument('-N', '--cores_per_node', type=int) # 40 on klone, 28 or 32 on mox
    # various flags to facilitate testing
    parser.add_argument('-v', '--verbose', default=False, type=boolean_string)
    parser.add_argument('--get_forcing', default=True, type=boolean_string)
    parser.add_argument('--short_roms', default=False, type=boolean_string)
    parser.add_argument('--run_dot_in', default=True, type=boolean_string)
    parser.add_argument('--run_roms', default=True, type=boolean_string)
    parser.add_argument('--move_his', default=True, type=boolean_string)

    # flag to sleep during the forecast window, False means it will run continuously
    # parser.add_argument('-sfw', '--sleep_forecast_window', default=False, type=Lfun.boolean_string)
    args = parser.parse_args()
    return args

def Lstart(gridname='BLANK', tag='BLANK', ex_name='BLANK'):
    """
    This adds more run-specific entries to Ldir.
    """
    # put top level information from input into a dict
    Ldir['gridname'] = gridname
    Ldir['tag'] = tag
    Ldir['ex_name'] = ex_name
    # and add a few more things
    Ldir['gtag'] = gridname + '_' + tag
    Ldir['gtagex'] = gridname + '_' + tag + '_' + ex_name
    return Ldir.copy()
    # the use of copy() means different calls to Lstart (e.g. when importing
    # plotting_functions) to not overwrite each other

# set time range 
def time_fun(args,Ldd):
    if args.run_type == 'forecast':
        ds0 = datetime.now().strftime(ds_fmt)
        dt0 = datetime.strptime(ds0, ds_fmt)
        # NOTE: to run as separate days we need Ldir['forecast_days'] to be an integer
        dt1 = dt0 + timedelta(days = (float(Ldd['forecast_days']) - 1))
        ds1 = dt1.strftime(ds_fmt)
        # override for short_roms
        if args.short_roms == True:
            ds1 = ds0
            dt1 = dt0
        # Check to see if the forecast has already been run, and if so, exit!
        done_fn = Ldd['LO'] / 'driver' / ('forecast3_done_' + ds0 + '.txt')
        if done_fn.is_file():
            print('Forecast has already run successfully - exiting')
            print(str(done_fn))
            sys.exit()
    elif args.run_type == 'backfill': # you have to provide at least ds0 for backfill
        ds0 = args.ds0
        if len(args.ds1) == 0:
            ds1 = ds0
        else:
            ds1 = args.ds1
        dt0 = datetime.strptime(ds0, ds_fmt)
        dt1 = datetime.strptime(ds1, ds_fmt) 
    else:
        print('Error: Unknown run_type')
        sys.exit()

    sys.stdout.flush()

    return dt0, ds0, dt1, ds1

def get_input_nc_file_names(ags,Ldd,ddlist):
    daystep=Ldd['daystep']
    daystep_ocn=Ldd['daystep_ocn']
    daystep_atm=Ldd['daystep_atm']
    dt0 = ddlist['dt0']
    dt1 = ddlist['dt1']

    # setting up for both hindcast input data and forecast input data
    # raw ocean and atmospher data do not have different levels do those 1st
    # the "raw" files are in the native format that is grabbed from the web
    
    # do ocean first
    ocndhr = str(Ldd['data']) + '/ocn/' + Ldd['ocn_model'] + '/hind/raw'
    ocndfr = str(Ldd['data']) + '/ocn/' + Ldd['ocn_model'] + '/fore/raw'
    # a temporary directory for raw files. This is where ncks downloads to
    ncksotmph = str(Ldd['data']) + '/ocn/' + Ldd['ocn_model'] + '/hind/raw/tmp'
    ncksotmpf = str(Ldd['data']) + '/ocn/' + Ldd['ocn_model'] + '/fore/raw/tmp'
    # objects to contain the path and name of ocn hindcast and forecast raw files
    ocnhr=[]
    ocnfr=[]
    dt = dt0
    hr = 0
    while dt <= dt1: 
        dum = dt.strftime(ds_fmt)
        yr = dum[0:4]
        mo = dum[5:7]
        dy = dum[8:10]
        hh = str(hr).zfill(2)
        fnm1 = ocndhr + '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['ocn_model'] + '_hind_' + yr + mo + dy + hh + '.nc'
        fnm2 = ocndfr + '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['ocn_model'] + '_fore_' + yr + mo + dy + hh + '.nc'
        ocnhr.append(fnm1)
        ocnfr.append(fnm2)
        dt = dt + timedelta(days=daystep_ocn)
        hr = hr + int(24*daystep_ocn)
        if hr==24:
            hr=0

    # do atm 2nd
    atmdhr = str(Ldd['data']) + '/atm/' + Ldd['atm_model'] + '/hind/raw'
    atmdfr = str(Ldd['data']) + '/atm/' + Ldd['atm_model'] + '/fore/raw'
    # a temporary directory for raw files. This is where ncks downloads to
    ncksatmph = str(Ldd['data']) + '/atm/' + Ldd['atm_model'] + '/hind/raw/tmp'
    ncksatmpf = str(Ldd['data']) + '/atm/' + Ldd['atm_model'] + '/fore/raw/tmp'
    atmhr=[]
    atmfr=[]
    dt = dt0
    hr = 0
    while dt <= dt1: 
        dum = dt.strftime(ds_fmt)
        yr = dum[0:4]
        mo = dum[5:7]
        dy = dum[8:10]
        hh = str(hr).zfill(2)
        fnm1 = atmdhr + '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['atm_model'] + '_hind_' + yr + mo + dy + hh + '.nc'
        fnm2 = atmdfr + '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['atm_model'] + '_fore_' + yr + mo + dy + hh + '.nc'
        atmhr.append(fnm1)
        atmfr.append(fnm2)
        dt = dt + timedelta(days=daystep_atm)
        hr = hr + int(24*daystep_atm)
        if hr==24:
            hr=0

    # we will only return the needed .nc files for the specific model LV.
    # there will only be 1 file per day as that is what will be needed to run
    # roms
    ocndhnc = str(Ldd['data']) + '/ocn/' + Ldd['ocn_model'] + '/hind/nc/' + Ldd['gridname']
    atmdhnc = str(Ldd['data']) + '/atm/' + Ldd['atm_model'] + '/hind/nc/' + Ldd['gridname']
    ocndfnc = str(Ldd['data']) + '/ocn/' + Ldd['ocn_model'] + '/fore/nc/' + Ldd['gridname']
    atmdfnc = str(Ldd['data']) + '/atm/' + Ldd['atm_model'] + '/fore/nc/' + Ldd['gridname']
    ocnhicnc = []
    ocnhbcnc = []
    ocnficnc = []
    ocnfbcnc = []
    atmhnc = []
    atmfnc = []
    dt = dt0
    hr = 0
    while dt < dt1: 
        dum = dt.strftime(ds_fmt)
        yr = dum[0:4]
        mo = dum[5:7]
        dy = dum[8:10]
        fnm1bc = '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['ocn_model'] + '_hind_bc_' + yr + mo + dy + '.nc'
        fnm1ic = '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['ocn_model'] + '_hind_ic_' + yr + mo + dy + '.nc'
        fnm2bc = '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['ocn_model'] + '_fore_bc_' + yr + mo + dy + '.nc'
        fnm2ic = '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['ocn_model'] + '_fore_ic_' + yr + mo + dy + '.nc'
        fnm3 = '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['atm_model'] + '_hind_' + yr + mo + dy + '.nc'
        fnm4 = '/' + yr + '/' + mo + '/' + dy + '/' + Ldd['atm_model'] + '_fore_' + yr + mo + dy + '.nc'
        ocnhicnc.append(ocndhnc+fnm1ic)
        ocnhbcnc.append(ocndhnc+fnm1bc)
        ocnficnc.append(ocndfnc+fnm2ic)
        ocnfbcnc.append(ocndfnc+fnm2bc)
        atmhnc.append(atmdhnc+fnm3)
        atmfnc.append(atmdfnc+fnm4)
        dt = dt + timedelta(days=daystep)

    # set up dict of files needed to run roms
    ifiles={}
    
    if ags.run_type == 'backfill':    
        ifiles['ocean_raw'] = ocnhr
        ifiles['atm_raw'] = atmhr
        ifiles['atm_nc'] = atmhnc
        ifiles['ocn_nc_ic'] = ocnhicnc
        ifiles['ocn_nc_bc'] = ocnhbcnc
        ifiles['ocn_raw_ncks'] = ncksotmph
        ifiles['atm_raw_ncks'] = ncksatmph
    elif ags.run_type == 'forecast':
        ifiles['ocean_raw'] = ocnfr
        ifiles['atm_raw'] = atmfr
        ifiles['atm_nc'] = atmfnc
        ifiles['ocn_nc_ic'] = ocnficnc
        ifiles['ocn_nc_bc'] = ocnfbcnc
        ifiles['ocn_raw_ncks'] = ncksotmpf
        ifiles['atm_raw_ncks'] = ncksatmpf
    else:
        print(str(ags.run_type) + ' is not a valid run type. Exiting.')
        sys.exit()

    return ifiles

def got_input_nc_files(infiles):
    # there are 3 required nc files, check and see if we got them...
    dgot_atm_nc=[]
    dgot_ocn_nc_ic=[]
    dgot_ocn_nc_bc=[]

    for i in infiles['atm_nc']:
        dum2 = os.path.isfile(i)
        dgot_atm_nc.append(dum2)
    for i in infiles['ocn_nc_ic']:
        dum2 = os.path.isfile(i)
        dgot_ocn_nc_ic.append(dum2)
    for i in infiles['ocn_nc_bc']:
        dum2 = os.path.isfile(i)
        dgot_ocn_nc_bc.append(dum2)

    gotinncfiles={}
    gotinncfiles['atm_nc'] = dgot_atm_nc
    gotinncfiles['ocn_nc_ic'] = dgot_ocn_nc_ic
    gotinncfiles['ocn_nc_bc'] = dgot_ocn_nc_bc
    
    return gotinncfiles

    
def got_input_raw_files(infiles):
    # there are 2 required raw files. 
    dgot_ocean_raw=[]
    dgot_atm_raw=[]

    for i in infiles['ocean_raw']:
        dum2 = os.path.isfile(i)
        dgot_ocean_raw.append(dum2)
    for i in infiles['atm_raw']:
        dum2 = os.path.isfile(i)
        dgot_atm_raw.append(dum2)

    gotifiles={}
    gotifiles['atm_raw'] = dgot_atm_raw
    gotifiles['ocean_raw'] = dgot_ocean_raw
    
    return gotifiles

def make_get_needed_input_files(dlist,ags,Ldd,infiles,gotinncfiles,gotinrawfiles):
    # first check for the atmospheric forcing files
    cnt=0
    print('checking for atmospheric forcing nc files...')
    gotem=bool(True)
    for i in gotinncfiles['atm_nc']:
        if i == False:
            gotem=bool(False)
            fnm = infiles['atm_nc'][cnt]
            print('the atmospheric forcing nc file:')
            print(fnm)
            print('does not exit. Need to make.')
        cnt=cnt+1
    if gotem == False:
        print('need to make atm.nc file(s).')
    else:
        print('atm.nc file(s) exist in the right place. atm forcing...check!')    
    print('...done')
    
    # if the .nc files do not exist, must check for the raw files
    if gotem == False:
        cnt=0
        print('checking for raw atm forcing files...')
        gotem2=bool(True)
        for i in gotinrawfiles['atm_raw']:
            if i == False:
                gotem2=bool(False)
                fnm = infiles['atm_raw'][cnt]
                print('the raw atm file:')
                print(fnm)
                print('does not exist. Need to get.')
            cnt=cnt+1    
        if gotem2 == False:
            print('need to get atm_raw file(s). Getting them...')
            # ncks? function to get files from the web and put in right spot
            print('raw atm files were retrieved and are in the right spot')
        else:
            print('atm_raw file(s) exist.')
        # now we go from raw to .nc here    
        print('Making atm.nc file(s) from atm raw files...')
        # function to go from raw atm to atm.nc files
        print('...done.')
        print('atm.nc file(s) exist in the right place. atm forcing...check!')

    # now we do the same but for the ocn_ic.nc files
    # the IC file could be made as "new", ie. hycom (or other)
    # Or it could be "perfect" and would require the 
    # previous days model run rst.nc file

    if ags.start_type == 'new':    
        # this will look for IC.nc files made from hycom (or)
        # there will be one file per day
        cnt=0
        print('checking for new ocean IC .nc files...')
        gotem=bool(True)
        for i in gotinncfiles['ocn_nc_ic']:
            if i == False:
                gotem=bool(False)
                fnm = infiles['ocn_nc_ic'][cnt]
                print('the ocean IC nc file:')
                print(fnm)
                print('does not exit. Need to make.')
            cnt=cnt+1
        if gotem == False:
            print('need to make ocn_IC.nc file(s).')
        else:
            print('ocn_IC.nc file(s) exist in the right place. new_ocn_IC...check!')    
        print('...done')
    
        # if the .nc files do not exist, must check for the raw files
        if gotem == False:
            cnt=0
            print('checking for raw ocn files...')
            gotem2=bool(True)
            for i in gotinrawfiles['ocean_raw']:
                if i == False:
                    gotem2=bool(False)
                    fnm = infiles['ocean_raw'][cnt]
                    print('the raw ocn file:')
                    print(fnm)
                    print('does not exist. Need to get.')
                cnt=cnt+1    
            if gotem2 == False:
                print('need to get ocn_raw file(s). Getting them...')
                # ncks? function to get files from the web and put in right spot
                emsg = get_raw_ocean_files(dlist,ags,Ldd,infiles)
                if (emsg[1] == True) & (emsg[2] == True):
                    print('the raw ocean file was retrieved and split up')
                elif (emsg[1] == True) & (emsg[2] == False):
                    print('the raw ocean files was retrieved but not split up ')
                else:
                    print('the raw ocean file was not retrieved so it couldnt be split up')
            else:
                print('ocn_raw file(s) exist.')

            # now we go from raw to .nc here    
            print('Making ocn_IC.nc file(s) from ocn raw files...')
            # function that makes IC.nc files
            icbc = raw_to_roms_ic_bc(dlist,ags,Ldd,infiles)
            print('...done.')
            print('ocean IC nc files was made.')
    
    # now look for the right restart file
    elif ags.start_type == 'perfect':
        print('looking for the restart file')
    else:
        print(str(ags.start_type) + ' is not a valid start type. Exiting')
        sys.exit()
    
    file_msg = 'working in this function!'
    return file_msg

def raw_to_roms_ic_bc(dlist,ags,Ldd,o_files):
    print('if we are here it is because the ic.nc and bc.nc files do not exist')
    print('but the raw.nc files do. so we will do some interpolating and making .nc files')

    icbc = 'none made yet'
    return icbc

def get_raw_ocean_files(dlist,ags,Ldd,o_files):
    import os, sys
    import time
    import subprocess

# the following command (in a terminal) gets 1 full day of data, dt=3hr from hycom, in our region of interest
# ncks -d time,2024-01-01T00:00,2024-01-02T00:00 -d lon,236.,244. -d lat,28.,37. -v surf_el,water_temp,salinity,water_u,water_v,depth  https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0  -4 -O /Users/mspydell/hycom_out.nc

    ocn_mod=Ldd['ocn_model']
    mod_type=ags.run_type
    south = Ldd['latlonbox'][0]
    north = Ldd['latlonbox'][1]
    west  = Ldd['latlonbox'][2]
    east  = Ldd['latlonbox'][3]

    # time limits
    dstr0 = dlist['dt0'].strftime('%Y-%m-%dT00:00') 
    dstr1 = dlist['dt1'].strftime('%Y-%m-%dT00:00')
    # use subprocess.call() to execute the ncks command
    vstr = 'surf_el,water_temp,salinity,water_u,water_v,depth'
    full_fn_out=o_files['ocn_raw_ncks']+'/hycom_out.nc'
    dum = os.path.isdir(o_files['ocn_raw_ncks'])
    if dum == False:
        make_dir(o_files['ocn_raw_ncks'],clean=False)

    cmd_list = ['ncks',
        '-d', 'time,'+dstr0+','+dstr1,
        '-d', 'lon,'+str(west)+','+str(east),
        '-d', 'lat,'+str(south)+','+str(north),
        '-v', vstr,
        'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0',
        '-4', '-O', full_fn_out]

    print(cmd_list)

    # run ncks
    tt0 = time.time()
    ret1 = subprocess.call(cmd_list)
    #ret1 = 1
    print('Time to get full file using ncks = %0.2f sec' % (time.time()-tt0))
    print('Return code = ' + str(ret1) + ' (0=success, 1=skipped ncks)')

    # time to split into individual files for each time. This is already set up in the raw
    # file names. We need one file for each. 
    # But might need to make directories based on these file names...
    # and we need the times of the files, get that from the file names too
    dstr=[]
    for i in o_files['ocean_raw']:
        ie = i.rfind('/')
        dum = os.path.isdir(i[0:ie])
        if dum == False:
            make_dir(i[0:ie],clean=False)
        ie = i.rfind('.')
        yr = i[ie-10:ie-6]
        mo = i[ie-6:ie-4]
        dy = i[ie-4:ie-2]
        hr = i[ie-2:ie]
        # this is the format ncks wants
        ddstr = yr + '-' + mo + '-' + dy + 'T' + hr + ':00'
        dstr.append(ddstr)
        #print(yr + mo + dy + hr)
        #2024-01-01T00:00

    # now we should be set up to split the .nc file from ncks into separate raw files
    # and put them in the right spot        
    
    cnt = 0
    for fnout in o_files['ocean_raw']:
        iis = dstr[cnt]
        cmd_list = ['ncks',
            '-d', 'time,'+iis+','+iis,
            '-O', full_fn_out, fnout]
        ret2 = subprocess.call(cmd_list)
        cnt = cnt+1
        
    if (ret1 == 0) & (ret2 == 0):
        got_ncks = [True,True]
    elif (ret1 ==0) & (ret2 == 1):
        got_ncks = [True,False]
    else:
        got_ncks = [False,False]
    
    return got_ncks

# format the messages
def messages(stdout, stderr, mtitle, verbose):
    Ncenter = 30
    # utility function for displaying subprocess info
    if verbose:
        print((' ' + mtitle + ' ').center(Ncenter,'='))
        if len(stdout) > 0:
            print(' sdtout '.center(Ncenter,'-'))
            print(stdout.decode())
    if len(stderr) > 0:
        print((' ' + mtitle + ' ').center(Ncenter,'='))
        # always print errors
        print(' stderr '.center(Ncenter,'-'))
        print(stderr.decode())
    sys.stdout.flush()

def make_dir(pth, clean=False):
    """
    >>> WARNING: Be careful! This can delete whole directory trees. <<<
    
    Make a directory from the path "pth" which can:
    - be a string or a pathlib.Path object
    - be a relative path
    - have a trailing / or not
    
    Use clean=True to clobber the existing directory (the last one in pth).
    
    This function will create all required intermediate directories in pth.
    """
    if clean == True:
        shutil.rmtree(str(pth), ignore_errors=True)
    Path(pth).mkdir(parents=True, exist_ok=True)

def datetime_to_modtime(dt):
    """
    This is where we define how time will be treated
    in all the model forcing files.

    INPUT: dt is a single datetime value
    OUTPUT: dt as seconds since modtime0 (float)
    """
    t = (dt - modtime0).total_seconds()
    return t

def modtime_to_datetime(t):
    """
    INPUT: seconds since modtime0 (single number)
    OUTPUT: datetime version
    """
    dt = modtime0 + timedelta(seconds=t)
    return dt

def modtime_to_mdate_vec(mt_vec):
    """ 
    INPUT: numpy vector of seconds since modtime0
    - mt stands for model time
    OUTPUT: a vector of mdates
    """ 
    import matplotlib.dates as mdates
    #first make a list of datetimes
    dt_list = []
    for mt in mt_vec:
        dt_list.append(modtime0 + timedelta(seconds=mt))
    md_vec = mdates.date2num(dt_list)
    return md_vec
    
def boolean_string(s):
    # used by argparse (also present in zfun, redundant but useful)
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True' # note use of ==

# Functions used by postprocessing code like pan_plot or the various extractors

def date_list_utility(dt0, dt1, daystep):
    """
    INPUT: start and end datetimes
    OUTPUT: list of LiveOcean formatted dates
    """
    date_list0 = {}
    date_list = []
    yr = []
    mon = []
    day = []

    dt = dt0
    while dt <= dt1:
        dum = dt.strftime(ds_fmt)
        date_list.append(dum)
        yr.append(dum[0:4])
        mon.append(dum[5:7])
        day.append(dum[8:10])
        dt = dt + timedelta(days=daystep)

    date_list0['strs'] = date_list
    date_list0['daystep'] = daystep
    date_list0['dt0'] = dt0
    date_list0['dt1'] = dt1

    return date_list0

def fn_list_utility(dt0, dt1, Ldir, hourmax=24, his_num=2):
    """
    INPUT: start and end datetimes
    OUTPUT: list of all history files expected to span the dates
    - list items are Path objects
    """
    dir0 = Ldir['roms_out'] / Ldir['gtagex']
    fn_list = []
    date_list = date_list_utility(dt0, dt1)
    if his_num == 1:
        # New scheme 2023.10.05 to work with new or continuation start_type,
        # by assuming we want to start with ocean_his_0001.nc of dt0
        fn_list.append(dir0 / ('f'+dt0.strftime(ds_fmt)) / 'ocean_his_0001.nc')
        for dl in date_list:
            f_string = 'f' + dl
            hourmin = 1
            for nhis in range(hourmin+1, hourmax+2):
                nhiss = ('0000' + str(nhis))[-4:]
                fn = dir0 / f_string / ('ocean_his_' + nhiss + '.nc')
                fn_list.append(fn)
    else:
        # For any other value of his_num we assume this is a perfect start_type
        # and so there is no ocean_his_0001.nc on any day and we start with
        # ocean_his_0025.nc of the day before.
        dt00 = (dt0 - timedelta(days=1))
        fn_list.append(dir0 / ('f'+dt00.strftime(ds_fmt)) / 'ocean_his_0025.nc')
        for dl in date_list:
            f_string = 'f' + dl
            hourmin = 1
            for nhis in range(hourmin+1, hourmax+2):
                nhiss = ('0000' + str(nhis))[-4:]
                fn = dir0 / f_string / ('ocean_his_' + nhiss + '.nc')
                fn_list.append(fn)
    return fn_list
    
def get_fn_list(list_type, Ldir, ds0, ds1, his_num=2):
    """
    INPUT:
    A function for getting lists of history files.
    List items are Path objects
    
    NEW 2023.10.05: for list_type = 'hourly', if you pass his_num = 1
    it will start with ocean_his_0001.nc on instead of the default which
    is to start with ocean_his_0025.nc on the day before.
    """
    dt0 = datetime.strptime(ds0, ds_fmt)
    dt1 = datetime.strptime(ds1, ds_fmt)
    dir0 = Ldir['roms_out'] / Ldir['gtagex']
    if list_type == 'snapshot':
        # a single file name in a list
        his_string = ('0000' + str(his_num))[-4:]
        fn_list = [dir0 / ('f' + ds0) / ('ocean_his_' + his_string + '.nc')]
    elif list_type == 'hourly':
        # list of hourly files over a date range
        fn_list = fn_list_utility(dt0,dt1,Ldir,his_num=his_num)
    elif list_type == 'daily':
        # list of history file 21 (Noon PST) over a date range
        fn_list = []
        date_list = date_list_utility(dt0, dt1)
        for dl in date_list:
            f_string = 'f' + dl
            fn = dir0 / f_string / 'ocean_his_0021.nc'
            fn_list.append(fn)
    elif list_type == 'weekly':
        # like "daily" but at 7-day intervals
        fn_list = []
        date_list = date_list_utility(dt0, dt1, daystep=7)
        for dl in date_list:
            f_string = 'f' + dl
            fn = dir0 / f_string / 'ocean_his_0021.nc'
            fn_list.append(fn)
    elif list_type == 'allhours':
        # a list of all the history files in a directory
        # (this is the only list_type that actually finds files)
        in_dir = dir0 / ('f' + ds0)
        fn_list = [ff for ff in in_dir.glob('ocean_his*nc')]
        fn_list.sort()

    return fn_list
    
def choose_item(in_dir, tag='', exclude_tag='',
    itext='** Choose item from list **', last=False):
    """
    INPUT: in_dir = Path object of a directory
    OUTPUT: just the name you chose, a string, not the full path.
    
    You can set strings to search for (tag), strings to exclude (exclude_tag),
    and the prompt text (itext).
    
    Use last=True to have the "return" choice be the last one.
    """
    print('\n%s\n' % (itext))
    ilist_raw = [item.name for item in in_dir.glob('*')]
    ilist_raw.sort()
    if len(tag) == 0:
        ilist = [item for item in ilist_raw if item[0] != '.']
    else:
        ilist = [item for item in ilist_raw if tag in item]
        
    if len(exclude_tag) == 0:
        pass
    else:
        ilist = [item for item in ilist if exclude_tag not in item]
    
    Nitem = len(ilist)
    idict = dict(zip(range(Nitem), ilist))
    for ii in range(Nitem):
        print(str(ii) + ': ' + ilist[ii])
    if last == False:
        my_choice = input('-- Input number -- (return=0) ')
        if len(my_choice)==0:
            my_choice = 0
    elif last == True:
        my_choice = input('-- Input number -- (return=last) ')
        if len(my_choice)==0:
            my_choice = Nitem-1
        
    my_item = idict[int(my_choice)]
    return my_item
    
def dict_to_csv(in_dict, out_fn):
    """
    Writes a dict to a csv file.
    
    out_fn should be a Path object.
    """
    out_fn.unlink(missing_ok=True)
    with open(out_fn, 'w') as f:
        for k in in_dict.keys():
            f.write(k + ',' + str(in_dict[k]) + '\n')
            
def csv_to_dict(in_fn):
    """
    Reads a csv file into a dict and returns it.
    
    We should add some error checking to make sure the input is as expected.
    """
    out_dict = dict()
    with open(in_fn, 'r') as f:
        for line in f:
            k,v = line.split(',')
            out_dict[k] = str(v).replace('\n','')
    return out_dict
    
def module_from_file(module_name, file_path):
    """
    This is used for the hook ot LO_user.  It allows you to import a module from a
    specific path, even if a module of the same name exists in the current directory.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == '__main__':
    # TESTING: run Lfun will execute these (don't import Lfun first)
    
    if False:
        print(' TESTING Lstart() '.center(60,'-'))
        Ldir = Lstart(gridname='cas6', tag='v3', ex_name='lo8b')
        print(' Ldir seen by make_forcing_main '.center(60,'+'))
        for k in Ldir.keys():
            print('%20s : %s' % (k, Ldir[k]))
            
    if False:
        print(' TESTING datetime_to_modtime() '.center(60,'-'))
        dt = datetime(2019,7,4)
        print(dt)
        t = datetime_to_modtime(dt)
        print(t)
        print(' TESTING modtime_to_datetime() '.center(60,'-'))
        dt_new = modtime_to_datetime(t)
        print(dt_new)
    
    if False:
        print(' TESTING copy_to_azure() '.center(60,'-'))
        input_filename = Ldir['data'] / 'accounts' / 'az_testfile.txt'
        output_filename = input_filename.name
        container_name = 'pm-share'
        az_dict = copy_to_azure(input_filename, output_filename, container_name, Ldir)
        if az_dict['result'] =='success':
            print('USE THIS URL TO ACCESS THE FILE')
            print(az_dict['az_url'])
        elif az_dict['result'] =='fail':
            print('EXCEPTION')
            print(az_dict['exception'])
            
    if True:
        print(' TESTING get_fn_list() '.center(60,'-'))
        Ldir = Lstart(gridname='cas6', tag='v0', ex_name='live')
        ds0 = '2019.07.04'
        ds1 = '2019.07.05'
        for list_type in ['daily','snapshot', 'allhours', 'hourly']:
            print(list_type.center(60,'.'))
            fn_list = get_fn_list(list_type, Ldir, ds0, ds1, his_num=1)
            for fn in fn_list:
                print(fn)
                
    if False:
        print(' TESTING choose_item() '.center(60,'-'))
        Ldir = Lstart(gridname='cas6', tag='v3', ex_name='lo8b')
        in_dir = Ldir['roms_out1'] / Ldir['gtagex'] / 'f2019.07.04'
        my_item = choose_item(in_dir, tag='.nc', exclude_tag='')
        print(my_item)
    
    




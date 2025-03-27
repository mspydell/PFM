import os
import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from get_PFM_info import get_PFM_info
import glob
import shutil
import netCDF4
import cftime

sys.path.append('../sdpm_py_util')
import ocn_functions as ocnfuns



def evaluate_function_from_file(file_path, function_name, *args):
    """
    Evaluates a function defined in a file, given its name and arguments.

    Args:
        file_path (str): The path to the Python file containing the function definition.
        function_name (str): The name of the function to evaluate.
        *args: Positional arguments to pass to the function.

    Returns:
        The result of the function call, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            code_string = file.read()
        
        # Create a dictionary to hold the function definition
        func_dict = {}
        
        # Execute the code string in the dictionary's namespace
        exec(code_string, func_dict)
        
        # Get the function from the dictionary
        func = func_dict[function_name]
        
        # Call the function with the provided arguments
        result = func(*args)
        return result
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except KeyError:
        print(f"Error: Function '{function_name}' not found in file.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_hindcast_days(t1,tend,dt):
    # this function returns a list of strings for the range of hindcasts.
    t1s = []
    t2s = []    

    while t1<tend:
        t2 = t1 + dt * timedelta(days=1)
        t1s.append(t1.strftime('%Y%m%d%H'))
        t2s.append(t2.strftime('%Y%m%d%H'))
        t1 = t2
    
    # make sure the last t2s is tend...
    tendstr = tend.strftime('%Y%m%d%H')
    if t2s[-1] != tendstr:
        print('the last t2 is not tend, something is wrong with dt and or tend. not returning lists of dates.')
        t1s = []
        t2s = []

    return t1s, t2s


def initialize_model(input_py_full,modinfo_pkl_full):
    # this makes the pickle file from the model_input_dictionary
    # first return the dictionary of model info
    print('first, we need all the model information.')
    print('from ' + input_py_full)
    print('we create a dictionary of model info')
    MINFO = evaluate_function_from_file( input_py_full , 'create_model_info_dict')
    # now we save the MINFO dict to a pickle file

    if MINFO['run_type'] == 'hindcast':
        # get list of strings start and end days for the simulation
        t_starts, t_ends = get_hindcast_days(MINFO['sim_start_time'],MINFO['sim_end_time'],MINFO['forecast_days'])
        MINFO['start_times_str'] = t_starts
        MINFO['end_times_str'] = t_ends
        MINFO['levels_to_run'] = ['LV1']

    with open(modinfo_pkl_full,'wb') as fout:
        pickle.dump(MINFO,fout)
        print('the model_info dictionary is saved to ' + modinfo_pkl_full)
        print('this file dictates how the model is run, where files are saved, etc.')


def get_model_info(pkl_fnm):
    # this returns the model information stored in pkl_fnm
    # as a dictionary
    with open(pkl_fnm,'rb') as fp:
        MI = pickle.load(fp)

    return MI


def add_file_names_2MI(yyyymmddhh,pkl_fnm):

    yyyymmddhhmm = yyyymmddhh + '00'
    t1 = datetime.strptime(yyyymmddhh,'%Y%m%d%H')
    PFM = get_model_info( pkl_fnm )
    t2 = t1 + PFM['forecast_days'] * timedelta(days=1)
    end_str = t2.strftime('%Y%m%d%H%M')

    PFM['lv1_his_name'] = 'LV1_ocean_his_' + yyyymmddhhmm + '.nc'
    PFM['lv1_rst_name'] = 'LV1_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc' 
    PFM['lv1_his_name_full'] = PFM['lv1_his_dir'] + '/' + PFM['lv1_his_name']
    PFM['lv1_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv1_rst_name']

    PFM['lv2_his_name'] = 'LV2_ocean_his_' + yyyymmddhhmm + '.nc'
    PFM['lv2_rst_name'] = 'LV2_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc' 
    PFM['lv2_his_name_full'] = PFM['lv2_his_dir'] + '/' + PFM['lv2_his_name']
    PFM['lv2_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv2_rst_name']

    PFM['lv3_his_name'] = 'LV3_ocean_his_' + yyyymmddhhmm + '.nc'
    PFM['lv3_rst_name'] = 'LV3_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc' 
    PFM['lv3_his_name_full'] = PFM['lv3_his_dir'] + '/'  + PFM['lv3_his_name']
    PFM['lv3_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv3_rst_name']

    PFM['lv4_his_name'] = 'LV4_ocean_his_' + yyyymmddhhmm + '.nc'
    PFM['lv4_rst_name'] = 'LV4_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc' 
    PFM['lv4_swan_rst_name']  = 'LV4_swan_rst_' + yyyymmddhhmm + '.dat' 
    PFM['lv4_his_name_full'] = PFM['lv4_his_dir'] + '/'  + PFM['lv4_his_name']
    PFM['lv4_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv4_rst_name']
    PFM['lv4_swan_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv4_swan_rst_name']

    with open(pkl_fnm,'wb') as fout:
        pickle.dump(PFM,fout)
        print('the model_info dictionary was resaved to ' + pkl_fnm)
        print('with properly time stamped output file names')
    

def print_initial_model_info(pkl_fnm):
          
    MI = get_model_info(pkl_fnm)    

    run_type = MI['run_type']
    if run_type == 'hindcast':
        t1 = MI['sim_start_time']
        t2 = MI['sim_end_time']
    else:
        t1 = MI['fetch_time'] 
        t2 = t1 + MI['forecast_days']*timedelta(days=1)
    t1str = t1.strftime('%Y-%m-%d %H:%M') 
    t2str = t2.strftime('%Y-%m-%d %H:%M') 
    dtstr = str(MI['forecast_days'])
    nsims = len(MI['start_times_str'])
    lvlstr = ', '.join(MI['levels_to_run'])

    nn = 60 # total spaces
    print(f"{'-'*nn}")
    print(f"{'Model Set Up' : ^60}")
    print(f"{'-'*nn}")
    print(f"{run_type : <60}")
    print(f"{'from           : ' + t1str : <60}")
    print(f"{'to             : ' + t2str : <60}")
    print(f"{'in             : ' + dtstr + ' day chunks (if hindcast)' : <60}")
    if run_type == 'hindcast':
        print(f"{'we will do     : ' + str(nsims) + ' separate ' + dtstr + ' day sims' : <60}")
    print(f"{'atm model      : ' + MI['atm_model'] : <60}")
    print(f"{'ocean_model    : ' + MI['ocn_model'] : <60}")
    print(f"{'running levels : ' + lvlstr : <60}")

def determine_hycom_foretime():
    PFM=get_PFM_info()
    # return the hycom forecast date based on the PFM simulation times

    fetch_time = PFM['fetch_time'] # the start of the PFM forecast
    hy_time = datetime(fetch_time.year,fetch_time.month,fetch_time.day)
    hy_time = hy_time - timedelta(days=1)
    yyyymmdd = "%d%02d%02d" % (hy_time.year, hy_time.month, hy_time.day)

    # these times do not change!!!
    t1  = fetch_time                                     # this is the first time of the PFM forecast
    t2  = t1 + PFM['forecast_days'] * timedelta(days=1)  # this is the last time of the PFM forecast

    print('making a PFM simulation starting from')
    print(t1)
    print('and ending at')
    print(t2)
    print('getting the right hycom data...')
    # make stuff below this into a function...
    # n0 is the number of files we should have
    # num_missing is the number of files we are missing
    num_missing = 10
    days_off = 1
    while num_missing > 0:
        n0, num_missing = ocnfuns.check_hycom_data(yyyymmdd,[t1,t2])
        print('there should be ' + str(n0) + ' files.')
        print('there are ' + str(num_missing) + ' missing files.')

        if num_missing > 0:
            # try downloading all of the data again
            print('we did not have the files from the ' + yyyymmdd + ' hycom forecast')
            print('but we will try getting that data again. Maybe hycom filled in their .nc files?')
            ocnfuns.get_hycom_data(yyyymmdd)
            n02, num_missing2 = ocnfuns.check_hycom_data(yyyymmdd,[t1,t2])
            if num_missing2 == 0:
                num_missing = 0
            else:
                print('we need the change the date of the hycom forecast (but not change the dates of the PFM forecast)')
                hy_time = hy_time - timedelta(days=1)
                yyyymmdd = "%d%02d%02d" % (hy_time.year, hy_time.month, hy_time.day)
                days_off = days_off + 1
                if days_off > 4:
                    num_missing = -10
                    return num_missing

    return yyyymmdd


def initialize_simulation(args):
    if isinstance(args,bool) == True:
        clean_start = args
    else:
        clean_start = args[0]

    if clean_start:
        print('we are going to start clean...')
        print('getting PFM info...')
        PFM=get_PFM_info()

        dirs=[PFM['lv1_his_dir'],PFM['lv1_plot_dir'],PFM['lv1_forc_dir'],
              PFM['lv2_his_dir'],PFM['lv2_plot_dir'],PFM['lv2_forc_dir'],
              PFM['lv3_his_dir'],PFM['lv3_plot_dir'],PFM['lv3_forc_dir'],
              PFM['lv4_his_dir'],PFM['lv4_plot_dir'],PFM['lv4_forc_dir']] 

        runl = ['/*.out','/*.sb','/*.log','/*.in','*_timing_info.pkl']

        print('removing PFM info file...')
        os.remove(PFM['info_file'])
        print('now removing all previous input and png files...')

        for dd in dirs: # for everything but run dir
            if 'Forc' in dd or 'His' in dd:
                ddd = dd + '/*.*'
            elif 'Plots' in dd:
                ddd = dd + '/*.png'
            for f in glob.glob(ddd):
                os.remove(f)        
        for pp in ['lv1_run_dir','lv2_run_dir','lv3_run_dir','lv4_run_dir']:
            for dd in runl:
                ddd = PFM[pp] + dd
                for f in glob.glob(ddd):
                    os.remove(f)
        for f in glob.glob(PFM['lv4_run_dir'] + '/Err*'):
            os.remove(f)

        print('now making a new PFM.pkl file.')
        PFM = get_PFM_info()
        if isinstance(args,bool) == False:
            yyyymmdd = args[1]
            print('now changing the start date to: ', yyyymmdd)
            start_time  = datetime.now()
            utc_time    = datetime.now(timezone.utc)
            yyyymmdd    = args[1]
            fetch_time  = date(int(yyyymmdd[0:4]),int(yyyymmdd[4:6]),int(yyyymmdd[6:8]))
            PFM['yyyymmdd']   = yyyymmdd
            PFM['hhmm']       = '1200'
            PFM['fetch_time'] = fetch_time
            PFM['start_time'] = start_time
            PFM['utc_time']   = utc_time
            print('fetch_time is now: ', fetch_time)
            print('resaving the PFM.pkl file with specified start time.')
            with open(PFM['info_file'],'wb') as fout:
                pickle.dump(PFM,fout)
                print('PFM info was resaved as ' + PFM['info_file'])

    else:
        print('we are NOT starting clean.')
        print('NO files are being deleted.')        


def remake_PFM_pkl_file(args):
    print('we are remaking the PFM.pkl file...')
    print('getting PFM info...')
    PFM=get_PFM_info()

    print('removing PFM info file...')
    os.remove(PFM['info_file'])

    PFM=get_PFM_info()
 
    if args != 0:
        yyyymmdd = args
        print('now changing the start date to: ', yyyymmdd)
        start_time  = datetime.now()
        utc_time    = datetime.now(timezone.utc)
        fetch_time  = date(int(yyyymmdd[0:4]),int(yyyymmdd[4:6]),int(yyyymmdd[6:8]))
        PFM['yyyymmdd']   = yyyymmdd
        PFM['hhmm']       = '1200'
        PFM['fetch_time'] = fetch_time
        PFM['start_time'] = start_time
        PFM['utc_time']   = utc_time
        print('fetch_time is now: ', fetch_time)
        print('resaving the PFM.pkl file with specified start time.')
        with open(PFM['info_file'],'wb') as fout:
            pickle.dump(PFM,fout)
            print('PFM info was resaved as ' + PFM['info_file'])

def move_restart_ncs():
    PFM = get_PFM_info()
    PFM['restart_file_dir'] = '/scratch/PFM_Simulations/restart_data'
    rst_dirs = [PFM['lv1_forc_dir'],PFM['lv2_forc_dir'],PFM['lv3_forc_dir'],PFM['lv4_forc_dir']]
    for pp in rst_dirs:
        ddd = pp + '/*rst*.nc'
        fall = glob.glob(ddd)
        if not fall:
            print('there are no restart files in ' + pp + ' to move.')
        else:
            for f in fall:
                head, tail = os.path.split(f)
                fnew = PFM['restart_file_dir'] + '/' + tail
                shutil.move(f,fnew)

def remove_old_restart_ncs():
    PFM = get_PFM_info()
    #PFM['restart_files_dir'] = '/scratch/PFM_Simulations/restart_data'
    for lvl in ['LV1','LV2','LV3','LV4']:
        rst_files = glob.glob(PFM['restart_files_dir'] + '/' + lvl + '*.nc')
        for rf in rst_files:
            head, tail = os.path.split(rf)
            yyyymmddhh = tail[14:24]
            #print(yyyymmddhh)
            tnow = datetime.now()
            told = tnow - timedelta(days=7) # removing files older than 1 week from now
            tf = datetime.strptime(yyyymmddhh,"%Y%m%d%H")
            if tf<told:
                print('removing old restart file:' + rf)


def convert_cftime_to_datetime(cftime_list):
    return [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in cftime_list]

def find_restart_index(datetime_list, target_datetime):
    ind = 0
    for dt in datetime_list:
        if dt == target_datetime:
            return ind # note, the index returned here is python indexing from 0. need to add 1 to it for ROMS
        ind = ind+1

    ind = -99    
    return ind

def get_swan_restart_file_name():

    fnm_swan = None
    PFM = get_PFM_info()
    t_fore = PFM['fetch_time'] 
    print('going to restart swan, forecast start time is')
    print(t_fore)
    rst_files = glob.glob(PFM['restart_files_dir'] + '/*_???.dat-001') # only examine cpu 1 files for timing
 
    #print(rst_files)

    dtfor = []
    if len(rst_files) > 0:
        for rf in rst_files:
            head, tail = os.path.split(rf)
            yyyymmddhh = tail[13:23]
            tnc = datetime.strptime(yyyymmddhh,"%Y%m%d%H")
            dtfor.append(t_fore - tnc)
    else:
        print('there were no swan restart files to use. We must use IC=ZERO for swan. Going to force that')
        return fnm_swan

    isort = np.argsort(dtfor)
    cnt = 0
    while cnt < len(isort):
        head,tail = os.path.split( rst_files[isort[cnt]] )
        yyyymmddhh = tail[13:23]
        tnc = datetime.strptime(yyyymmddhh,"%Y%m%d%H")
        hr = int( tail[26:29] )
        t_tot = tnc + hr*timedelta(hours = 1)
        #print(cnt)
        #print(t_fore)
        #print(tnc)
        #print(hr)
        #print(t_tot)
        #print(t_fore - t_tot)
        dt = np.round( (t_fore - t_tot).total_seconds() )
        if dt == 0: # if we get here we found the previous forecast time and hour 
            fnm_swan = rst_files[isort[cnt]][0:-4]
            break
        cnt=cnt+1

    return fnm_swan

def get_restart_file_and_index(lvl,pkl_fnm):
    PFM = get_model_info(pkl_fnm)
    # get the start time of the simulation
    if PFM['run_type'] == 'forecast':
        t_fore = PFM['fetch_time']
    else:
        t_fore = PFM['sim_time_1']

    print('going to restart ' + lvl + ' from')
    print(t_fore)
    rst_files = glob.glob(PFM['restart_files_dir'] + '/' + lvl + '*.nc')
    print(rst_files)
    dts = []
    for rf in rst_files:
        head, tail = os.path.split(rf)
        yyyymmddhh = tail[14:24]
        tnc = datetime.strptime(yyyymmddhh,"%Y%m%d%H")
        dts.append(t_fore - tnc)
    
    isort = np.argsort(dts)
    cnt = 0
    found = 0
    while cnt < len(isort):
        fname = rst_files[isort[cnt]]
        print('looking in ' + fname + ' for the right restart time...')
        ds = netCDF4.Dataset(fname)
        t_var = ds['ocean_time']
        t_units = t_var.units
        t = netCDF4.num2date(t_var[:],t_units)
        t = convert_cftime_to_datetime(t)
        index = find_restart_index(t, t_fore)
        if index != -99:
            found = 1
            print('found the time index!')
            break

        print('didnt find the right time in ' + fname)
        if cnt<len(isort)-1:
            print('going to look at a previous forecast restart file...')
        else:
            print('WARNIG the time stamp wasnt found in any restart file. CANT USE RESTART')
            fname = 'none'
            index = -99
        cnt = cnt+1

    print('going to restart using ' + fname + ' and python-index ' + str(index))    
        
    head, fname0 = os.path.split(fname)
    return fname0, index

def edit_and_save_MI(dict_in,pkl_fnm):
    PFM = get_model_info(pkl_fnm)
    kys = dict_in.keys()
    for ky in kys:
        PFM[ky] = dict_in[ky]
    
    with open(pkl_fnm,'wb') as fout:
        pickle.dump(PFM,fout)
        print('PFM info was edited and resaved')
    
def remove_old_swan_rst():
    PFM = get_PFM_info()
    PFM['restart_file_dir'] = '/scratch/PFM_Simulations/restart_data'
    rst_files = glob.glob(PFM['restart_file_dir'] + '/LV4*dat*')
    if len(rst_files)>0:
        for rf in rst_files:
            head, tail = os.path.split(rf)
            yyyymmddhh = tail[13:23]
            tnow = datetime.now()
            told = tnow - timedelta(days=7) # removing files older than 1 week from now
            tf = datetime.strptime(yyyymmddhh,"%Y%m%d%H")
            if tf<told:
                print('removing old swan restart file:' + rf)

def get_old_restart_files_list(ftype,older_than_days):
    # this function returns a list of files for ftype = 'ocean' or 'swan'
    # that are older than now-older_than_days
    PFM = get_PFM_info()
    
    dt_now = datetime.now()
    dt_test = dt_now - older_than_days * timedelta(days=1)

    if ftype == 'ocean':
        rst_files = glob.glob(PFM['restart_files_dir'] + '/LV*.nc')
        #itail = np.arange(14,25)
        #itail = list(range(14,25))
        i1 = 14
        i2 = 24
    elif ftype == 'swan':
        rst_files = glob.glob(PFM['restart_files_dir'] + '/LV4*dat*')
        #itail = np.arange(13,24)
        #itail = list(range(13,24))
        i1 = 13
        i2 = 23

    file_list = []

    if len(rst_files)>0:
        for rf in rst_files:
            head, tail = os.path.split(rf)
            tstmp = datetime.strptime(tail[i1:i2],'%Y%m%d%H')
            #print(tail[i1:i2])
            if tstmp < dt_test:
                file_list.append(rf) # this is the list of files

    return file_list

def remove_old_restart_files(ftype,older_than_days):
    flist = get_old_restart_files_list(ftype,older_than_days)
    if len(flist)>0:
        print('there are ' + str(len(flist)) + ' ' + ftype + ' restart files to remove. Deleting them...')
        for fn in flist:
            os.remove(fn)            
        print('...done')
    else:
        print('there were no ' + ftype + ' restart files older than now minus ' + str(older_than_days) + ' days old')

def remove_swan_restarts_eq_foretime():
    PFM = get_PFM_info()
    t_fore = PFM['fetch_time']    
    rst_files = glob.glob(PFM['restart_files_dir'] + '/LV4*dat*')
    i1 = 13
    i2 = 23
    rmd = 0
    if len(rst_files)>0:
       for rf in rst_files:
            head, tail = os.path.split(rf)
            tstmp = datetime.strptime(tail[i1:i2],'%Y%m%d%H')
            dt_test = np.abs ( np.round( (t_fore - tstmp).total_seconds() ) )
            #print(tail[i1:i2])
            if dt_test < 3600:
                os.remove(rf)
                rmd = 1            
                
    if rmd == 0:
        print('there was not a previous forecast with this forecast time. No swan restart files were deleted.')
    else:
        print('a previous forecast was run from this start time. Swan restart files from this forecast were deleted.')


def remove_swan_rst_nohour():
    # this removes the files swan writes to that do not have hr time stamps, just keeps the restart directory clean.
    # we will call at the beginning of a PFM run.
    PFM = get_PFM_info()
    fns = glob.glob(PFM['restart_files_dir'] + '/LV4_swan_rst_????????????.dat-*')
    print('going to deleting swan base rst files...')
    if len(fns)>0:
        for fn in fns:
            print('...deleting: ' + fn)
            os.remove(fn)
    else:
        print('...no swan base rst files to delete.')

def remove_swan_rst_incomplete():
    # we no longer call this function. Not needed.
    # this will delete swan restart files that were made if the simulation didn't complete 
    PFM = get_PFM_info()
    fns = glob.glob(PFM['restart_files_dir'] + '/LV4_swan_rst_????????????_000.dat-001')
    fores = [] # a list with all of the forecast times
    print('possibly deleting incomplete PFM simulation swan rst files...')
    if len(fns)>0:
        for fn in fns:
            head, tail = os.path.split(fn)
            yyyymmddhhmm = tail[13:23]
            fores.append(yyyymmddhhmm)

        unique_fores = list(set(fores))
        #print(unique_fores)
        for fn in unique_fores:
            ftxt = PFM['restart_files_dir'] + '/LV4_swan_rst_' + fn + '00_*.dat-*'
            all_files = glob.glob(ftxt)
            num_files = PFM['gridinfo']['L4','np_swan'] * int( (PFM['forecast_days'] / PFM['outputinfo']['L4','rst_interval']) + 1 )
            #print(num_files)
            #print(len(all_files))
            # below breaks things when changing 2.5 to 5.0 day forecasts, vice versa, etc. 
            if len(all_files) != num_files:
                print('...the simulation from ' + fn + ' wasnt finsished correctly, need to delete swan rst files.')
                for rf in all_files:
                    print('deleting ' + rf)
                    os.remove(rf)
            else:
                print('...there were no incomplete sets of swan rst files from the ' + fn + ' PFM simulation.')

def restart_setup(lvl,pkl_fnm):

    PFM = get_model_info(pkl_fnm)

    older_than_days = 7.0

    PFM_edit = dict()
    fname1,tindex1 = get_restart_file_and_index(lvl,pkl_fnm)

    if lvl == 'LV1':
        if PFM['run_type']=='forecast':
            print('removing ocean restart files older than now - ' + str(older_than_days) + ' days old...')
            remove_old_restart_files('ocean',older_than_days)
        key_rec = 'lv1_nrrec'
        key_file = 'lv1_ini_file'
    if lvl == 'LV2':
        key_rec = 'lv2_nrrec'
        key_file = 'lv2_ini_file'
    if lvl == 'LV3':
        key_rec = 'lv3_nrrec'
        key_file = 'lv3_ini_file'
    if lvl == 'LV4':
        key_rec = 'lv4_nrrec'
        key_file = 'lv4_ini_file'
        remove_swan_rst_nohour()
        #remove_swan_rst_incomplete()
        print('removing swan restart files older than now - ' + str(older_than_days) + ' days old...')
        remove_old_restart_files('swan',older_than_days)
        # below ensure that swan looks for a previous forecast to find the correct restart data!
        remove_swan_restarts_eq_foretime()
        if PFM['lv4_swan_use_rst'] == 1:
            fn0 = PFM['lv4_swan_rst_name'][0:13]
            fnm_swan = get_swan_restart_file_name()
            #yyyymmdd_rm = fname1[14:26]
            #t_sw = tindex1 * PFM['lv4_swan_rst_int_hr']
            #t_sw_str = str(t_sw).zfill(3)
            if fnm_swan == None:
                print('although a swan restart was requested, a restart file could not be found')
                print('and swan will start swan with IC=ZERO and with the line ...')
                print(PFM['swan_init_txt_full'])
            else:    
                swan_txt = 'HOTSTART ' + "'" + fnm_swan + "'"
                #fn0 + yyyymmdd_rm + '_' + t_sw_str + '.dat'
                PFM_edit['swan_init_txt_full'] = swan_txt
                print('we are going to restart swan with the line ...')
                print(PFM_edit['swan_init_txt_full'])

    # these are now the index and file name used to restart the simulation
    PFM_edit[key_rec] = tindex1+1 # need to add 1 to get non-python indexing    
    PFM_edit[key_file] = fname1
    edit_and_save_MI(PFM_edit,pkl_fnm)

def update_PFM_pkl(t1str,lvl,pkl_fnm):
    PFM = get_model_info(pkl_fnm)
    tstart = datetime.strptime(t1str,'%Y%m%d%H')
    tend = tstart + timedelta(days=1)
    yyyymmddhhmm = tstart.strftime('%Y%m%d%H%M')
    end_str = tend.strftime('%Y%m%d%H%M')

    keystr1 = 'lv'+lvl+'_his_name'
    fname1 = 'LV'+lvl+'_ocean_his_' + yyyymmddhhmm + '.nc'
    keystr2 = 'lv'+lvl+'_rst_name'
    fname2 = 'LV'+lvl+'_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc'
    keystr3 = 'lv1_his_name_full'
    str3 = PFM['lv1_his_dir']+ '/' + fname1
    keystr4 = 'lv1_rst_name_full'
    str4 = PFM['restart_files_dir'] + '/' + fname2

    PFM2 = {}
    PFM2[keystr1] = fname1
    PFM2[keystr2] = fname2 
    PFM2[keystr3] = str3
    PFM2[keystr4] = str4
    edit_and_save_PFM(PFM2,pkl_fnm)
   
if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])


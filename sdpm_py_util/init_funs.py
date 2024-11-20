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

        runl = ['/*.out','/*.sb','/*.log','/*.in']

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
    PFM['restart_file_dir'] = '/scratch/PFM_Simulations/restart_data'
    for lvl in ['LV1','LV2','LV3','LV4']:
        rst_files = glob.glob(PFM['restart_file_dir'] + '/' + lvl + '*.nc')
        for rf in rst_files:
            head, tail = os.path.split(rf)
            yyyymmddhh = tail[14:24]
            tnow = datetime.now()
            told = tnow - timedelta(days=7) # removing files older than 1 week from now
            tf = datetime.strptime(yyyymmddhh,"%Y%m%d%H")
            if tf<told:
                print('removing old ' + rf)
            else:
                print('nothing to remove, keeping ' + rf)


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

def get_restart_file_and_index(lvl):
    PFM = get_PFM_info()
    PFM['restart_file_dir'] = '/scratch/PFM_Simulations/restart_data'
    t_fore = PFM['fetch_time'] + 0 * timedelta(days = 1)
    print('going to restart ' + lvl + ' from')
    print(t_fore)
    rst_files = glob.glob(PFM['restart_file_dir'] + '/' + lvl + '*.nc')
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
        print(t)
        index = find_restart_index(t, t_fore)
        if index != -99:
            found = 1
            print('found the time!')
            break

        print('didnt find the right time in ' + fname)
        if cnt<len(isort)-1:
            print('going to look at a previous forecast restart file...')
        else:
            print('the time stamp wasnt found in any restart file.')
            fname = 'none'
            index = -99
        cnt = cnt+1

    print('going to restart using ' + fname + ' and index ' + str(index))    
    return fname, index

def edit_and_save_PFM(dict_in):
    PFM = get_PFM_info()
    kys = dict_in.keys()
    for ky in kys:
        PFM[ky] = dict_in[ky]
    
    with open(PFM['info_file'],'wb') as fout:
        pickle.dump(PFM,fout)
        print('PFM info was edited and resaved')
    

import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from get_PFM_info import get_PFM_info
import glob

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


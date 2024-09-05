import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from get_PFM_info import get_PFM_info
import glob

sys.path.append('../sdpm_py_util')

def initialize_simulation(args):
    if isinstance(args,bool) == True:
        clean_start = args
    else:
        clean_start = args[0]

    if clean_start:
        print('we are going to start clean...')
        print('getting PFM info...')
        PFM=get_PFM_info()

        dirs=[PFM['lv1_his_dir'],PFM['lv1_plot_dir'],PFM['lv1_forc_dir'],PFM['lv2_his_dir'],PFM['lv2_plot_dir'],PFM['lv2_forc_dir']] 

        runl = ['/*.out','/*.sb','/*.log','/*.in']

        print('removing PFM info file...')
        os.remove(PFM['info_file'])
        print('now removing all input files...')

        for dd in dirs: # for everything but run dir
            if 'Forc' in dd or 'His' in dd:
                ddd = dd + '/*.*'
            elif 'Plots' in dd:
                ddd = dd + '/*.png'
            for f in glob.glob(ddd):
                os.remove(f)        
        for pp in ['lv1_run_dir','lv2_run_dir']:
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
            with open(PFM['pfm_info_full'],'wb') as fout:
                pickle.dump(PFM,fout)
                print('PFM info was resaved as ' + PFM['pfm_info_full'])

    else:
        print('we are NOT starting clean.')
        print('NO files are being deleted.')        


def remake_PFM_pkl_file():
        print('we are remaking the PFM.pkl file...')
        print('getting PFM info...')
        PFM=get_PFM_info()

        print('removing PFM info file...')
        os.remove(PFM['info_file'])
        print('now removing all input files...')

        PFM=get_PFM_info()
 

import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
from get_PFM_info import get_PFM_info
import glob

sys.path.append('../sdpm_py_util')

def initialize_simulation(clean_start):
    if clean_start:
        print('we are going to start clean...')
        print('getting PFM info...')
        PFM=get_PFM_info()

        dirs=[PFM['lv1_his_dir'],PFM['lv1_plot_dir'],PFM['lv1_forc_dir']] 

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
                #print('removing:')
                #print(f)
                os.remove(f)        
        for dd in runl:
            ddd = PFM['lv1_run_dir'] + dd
            for f in glob.glob(ddd):
                #print('removing')
                #print(f)
                os.remove(f)

    else:
        print('we are NOT starting clean.')
        print('NO files are being deleted.')        


import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
from get_PFM_info import get_PFM_info

sys.path.append('../sdpm_py_util')

def initialize_simulation(clean_start):
    if clean_start:
        print('we are going to start clean...')
        print('getting PFM info...')
        PFM=get_PFM_info()
        print('removing info file...')
        os.remove(PFM['info_file'])
    else:
        print('we are NOT starting clean.')        



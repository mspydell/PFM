# -- driver_run_forecast_LV2_v1.py  --
# master python script to do a full LV2 forecast simulation

import sys
import os
from datetime import datetime
import subprocess

##############

sys.path.append('../sdpm_py_util')
from init_funs import remake_PFM_pkl_file
from get_PFM_info import get_PFM_info
from make_LV1_dotin_and_SLURM import make_LV1_dotin_and_SLURM
from make_LV2_dotin_and_SLURM import make_LV2_dotin_and_SLURM
from run_slurm_LV1 import run_slurm_LV1
from run_slurm_LV2 import run_slurm_LV2

##############
# Run LV1...
t00 = datetime.now()
cmd_list = ['python','-W','ignore','driver_run_forecast_LV1_v4.py'] # this is in the .../PFM/driver/ directory
ret1 = subprocess.run(cmd_list)
print('return code for level 1:' + str(ret1.returncode) + ' (0=good)')  
print('LV1 took:')
t2 = datetime.now()
print(t2-t00)
print('\n')

###############
# Run LV2...
t0 = datetime.now()
cmd_list = ['python','-W','ignore','driver_run_LV2_v1.py'] # this is in the .../PFM/driver/ directory
ret1 = subprocess.run(cmd_list)
print('return code for level 2:' + str(ret1.returncode) + ' (0=good)')  
print('LV2 took:')
t2 = datetime.now()
print(t2-t0)
print('\n')






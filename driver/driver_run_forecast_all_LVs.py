# -- driver_run_forecast_LV2_v1.py  --
# master python script to do a full LV2 forecast simulation

import sys
import os
from datetime import datetime
import subprocess

##############
sys.path.append('../sdpm_py_util')
from util_functions import display_timing_info
from get_PFM_info import get_PFM_info

##############
# Run LV1...
t00 = datetime.now()

print('=======================================================')
print('driver_LV1.py is now running...')


#cmd_list = ['python','-u','-W','ignore','driver_run_forecast_LV1_v4.py'] # this is in the .../PFM/driver/ directory
#cmd_list = ['python','-u','-W','ignore','driver_run_forecast_LV1_v4.py','20240915'] # this is in the .../PFM/driver/ directory
#cmd_list = ['python','-u','-W','ignore','driver_forecast_LV1_newtiming.py'] # this is in the .../PFM/driver/ directory
cmd_list = ['python','-u','-W','ignore','driver_LV1_1hrzetaBC.py'] # this is in the .../PFM/driver/ directory
ret1 = subprocess.run(cmd_list)
print('...done.')
print('return code for level 1:' + str(ret1.returncode) + ' (0=good)')  
if ret1.returncode != 0:
    print('need to terminate driver_run_forecast_all_LVs.py as LV1 subprocess didnt run correctly...')
    sys.exit("...exiting!")

print('LV1 took:')
t2 = datetime.now()
print(t2-t00)
print('=======================================================')
print('\n\n\n')

###############
# Run LV2...
print('=======================================================')
print('driver_LV2.py is now running...')
t0 = datetime.now()
cmd_list = ['python','-u','-W','ignore','driver_run_LV2_v1.py'] # this is in the .../PFM/driver/ directory
ret1 = subprocess.run(cmd_list)
print('...done.')
print('return code for level 2:' + str(ret1.returncode) + ' (0=good)')  
if ret1.returncode != 0:
    print('need to terminate driver_run_forecast_all_LVs.py as LV2 subprocess didnt run correctly...')
    sys.exit("...exiting!")

print('LV2 took:')
t2 = datetime.now()
print(t2-t0)
print('=======================================================')
print('\n\n\n')

###############
# Run LV3...
print('=======================================================')
print('driver_LV3.py is now running...')
t0 = datetime.now()
cmd_list = ['python','-u','-W','ignore','driver_run_LV3_v1.py'] # this is in the .../PFM/driver/ directory
ret1 = subprocess.run(cmd_list)
print('...done.')
print('return code for level 3:' + str(ret1.returncode) + ' (0=good)')  
if ret1.returncode != 0:
    print('need to terminate driver_run_forecast_all_LVs.py as LV3 subprocess didnt run correctly...')
    sys.exit("...exiting!")

print('LV3 took:')
t2 = datetime.now()
print(t2-t0)

testing = 0
if testing == 1:
    print('exiting the simulation for testing LV1-3.')
    sys.exit()

###############
# Run LV4...
PFM=get_PFM_info()
if PFM['lv4_model'] == 'ROMS':
    print('=======================================================')
    print('LV4 is ROMS only. driver_LV4.py is now running...')
    t0 = datetime.now()
    cmd_list = ['python','-u','-W','ignore','driver_run_LV4_v1.py'] # this is in the .../PFM/driver/ directory
    ret1 = subprocess.run(cmd_list)
    print('...done.')
    print('return code for level 4:' + str(ret1.returncode) + ' (0=good)')  
    if ret1.returncode != 0:
        print('need to terminate driver_run_forecast_all_LVs.py as LV4 subprocess didnt run correctly...')
        sys.exit("...exiting!")

    print('LV4 took:')
    t2 = datetime.now()
    print(t2-t0)
if PFM['lv4_model'] == 'COAWST':
    print('=======================================================')
    print('Using COAWST for LV4. driver_LV4.py is now running...')
    t0 = datetime.now()
    cmd_list = ['python','-u','-W','ignore','driver_run_LV4_coawst_v1.py'] # this is in the .../PFM/driver/ directory
    ret1 = subprocess.run(cmd_list)
    print('...done.')
    print('return code for level 4:' + str(ret1.returncode) + ' (0=good)')  
    if ret1.returncode != 0:
        print('need to terminate driver_run_forecast_all_LVs.py as LV4 subprocess didnt run correctly...')
        sys.exit("...exiting!")

    print('LV4 took:')
    t2 = datetime.now()
    print(t2-t0)

print('=======================================================')
print('\n\n\n')
print('=======================================================')
print('total time to do LV1, LV2, LV3, and LV4:')
print(t2-t00)
print('=======================================================')

display_timing_info()


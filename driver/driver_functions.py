import numpy as np
import os
import sys
import subprocess
sys.path.append('../sdpm_py_util')
import init_funs as initfuns
from datetime import datetime, timedelta

# the functions in here are used to make the 

def run_hind_LV1(t1str,pkl_fnm):

    MI = initfun.get_model_info(pkl_fnm)

    t1 = datetime.strptime(t1str,'%Y%m%d%H')
    t2 = t1 + MI['forecast_days']*timedelta(days=1)
    t2str = t2.strftime("%Y%m%d%H")

    start_type = MI['lv1_use_restart'] # 0=new solution. 1=from a restart file
    lv1_use_restart = start_type
    level = 1

    os.chdir('../sdpm_py_util')
    # this will only work for 1 day chunks. need to fix?
    cmd_list = ['python','-W','ignore','ocn_functions.py','get_hycom_hind_data',t1str]
    ret1 = subprocess.run(cmd_list)     




def run_hind_simulation(t1str,lvl,pkl_fnm):
    if lvl == 'LV1':
        run_hind_LV1(t1str,pkl_fnm)
    if lvl == 'LV2':
        run_hind_LV2(t1str,pkl_fnm)
    if lvl == 'LV3':
        run_hind_LV3(t1str,pkl_fnm)
import sys
import os
import subprocess
import numpy as np
from datetime import datetime

sys.path.append('../sdpm_py_util')
import init_funs as initfuns

def driver_run_hind_LV123( input_py_full, pkl_fnm ):
    t00 = datetime.now()
    # upon initialization, make the model_info.pkl file
    initfuns.initialize_model( input_py_full, pkl_fnm )
    # print info from pickle file
    initfuns.print_initial_model_info( pkl_fnm )
    # get model information
    MI = initfuns.get_model_info( pkl_fnm )
    nt  = len(MI['start_times_str'])
    nlv = len(MI['levels_to_run'])
    time_tots = [0]*nt
    time_lvls = [[0] * nt for i in range(nlv)]



    # this si the loop over the different separate simulations
    cnt_t = 0
    for time1 in MI['start_times_str']:
        t0_t = datetime.now()
        print(f"{'\n'}")
        print(f"{'='*60}")
        time2 = MI['end_times_str'][cnt_t]
        print('doing a simulation from ' + time1 + ' to ' + time2)
        print(f"{'-'*60}")
        cnt_l = 0
        # this is the loop over the levels to run
        for lvl in MI['levels_to_run']:
            t0_lvl = datetime.now()
            print('starting ' + lvl)
            cmd_list = ['python','-W','ignore','driver_functions.py','run_hind_simulation',time1,lvl,pkl_fnm]
            os.chdir('../sdpm_py_util')
            ret1 = subprocess.run(cmd_list)     
            os.chdir('../driver')
            print('done with ' + lvl)
            print('this took ')
            t2_lvl = datetime.now()
            time_lvls[cnt_l][cnt_t] = t2_lvl - t0_lvl
            print(t2_lvl - t0_lvl)
            print(f"{'-'*60}")
            cnt_l = cnt_l + 1

        print('done with the ' + time1 + ' to ' + MI['end_times_str'][cnt_t] + ' simulation')
        print('this took')
        t2_t = datetime.now()
        print(t2_t - t0_t)
        time_tots[cnt_t] = t2_t - t0_t
        print(f"{'='*60}")
        cnt_t = cnt_t + 1

    print(f"{'\n'}")
    print(f"{'='*60}")
    print('done with full simulation. this took')
    print(datetime.now() - t00)


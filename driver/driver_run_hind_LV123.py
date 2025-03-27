import sys
import os
import subprocess
import numpy as np
from datetime import datetime, timedelta

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
        # get 1st and last times as datetime objects
        t1 = datetime.strptime(time1,'%Y%m%d%H')
        t2 = t1 + MI['forecast_days']*timedelta(days=1)
        # put the current sub simulation 1st and last times into MI.pkl file
        MI2 = dict()
        MI2['sim_time_1'] = t1
        MI2['sim_time_2'] = t2
        initfuns.edit_and_save_MI(MI2,pkl_fnm)
        
        # add / change file names to model info
        # this uses the start (and end) times in pkl_fnm to construct
        # his and restart file names
        initfuns.add_file_names_2MI( time1, pkl_fnm )

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
#            cmd_list = ['python','-W','ignore','driver_functions.py','run_hind_simulation',time1,lvl,pkl_fnm]
            cmd_list = ['python','driver_functions.py','run_hind_simulation',time1,lvl,pkl_fnm]
            ret1 = subprocess.run(cmd_list)     
            print('done with ' + lvl)
            print('LV1 hind ran correctly? ' + str(ret1.returncode) + ' (0=yes)')
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

        # after the 1st simulation, change to using restarts
        MI2 = {}
        MI2['lv1_use_restart'] = 1
        MI2['lv2_use_restart'] = 1
        MI2['lv3_use_restart'] = 1
        initfuns.edit_and_save_MI(MI2,pkl_fnm)
        print('done with a 1 day LV1 hindcast, going to the next day.\n')
        #sys.exit("exiting for now.")
        cnt_t = cnt_t + 1



    print(f"{'\n'}")
    print(f"{'='*60}")
    print('done with full simulation. this took')
    print(datetime.now() - t00)


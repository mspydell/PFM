import sys
import os
import subprocess
from datetime import datetime

sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
import util_functions as utilfuns
sys.path.append('../driver')

def driver_run_forecast_LV4_only( pkl_fnm ):
    t00 = datetime.now()
    # upon initialization, make the model_info.pkl file
    #initfuns.initialize_model( input_py_full, pkl_fnm )
    
    # print info from pickle file
    #initfuns.print_initial_model_info( pkl_fnm )
    
    # get model information
    MI = initfuns.get_model_info( pkl_fnm )

    # this is the loop over the levels to run
    print('Running only LV4')
    for lvl in ['LV4']:
        t1 = datetime.now()
        print('\n--------------------------')
        print('starting ' + lvl)
        os.chdir('../driver')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_fore_LV4_dotin_and_run',pkl_fnm]
        print('now doing:')
        print(cmd_list)
        ret1 = subprocess.run(cmd_list)     
        print('done with ' + lvl)
        print(lvl, ' forecast ran correctly? ' + str(ret1.returncode) + ' (0=yes)')
        t2 = datetime.now()
        print('this took:')
        print(t2-t1)
        print('\n')
        
 
if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    if len(sys.argv) == 2:
        arg1 = sys.argv[1]
        driver_run_forecast_LV4_only(arg1)
    else:
        print("Error! Wrong number of arguments in driver_run_forecast_LV4_only")
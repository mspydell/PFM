import sys
import os
import subprocess
from datetime import datetime

sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
import util_functions as utilfuns
sys.path.append('../driver')

def driver_run_forecast_LV4_fillin_apr26( input_py_full, pkl_fnm ):
    # upon initialization, make the model_info.pkl file
    #initfuns.initialize_model( input_py_full, pkl_fnm )

    print('the input .py file is')
    print(input_py_full)
    print('the pkl file is going to be')
    print(pkl_fnm)

    print('!!!starting the model!!!')
    print('current time is: ', datetime.now())
    print('initializing model and making the info.pkl file.')
    initfuns.initialize_model( input_py_full, pkl_fnm )


    # print info from pickle file
    #initfuns.print_initial_model_info( pkl_fnm )
    
    # get model information
    MI = initfuns.get_model_info( pkl_fnm )

    # this is the loop over the levels to run
    print('Running only LV4')
    print('\n--------------------------')
    print('starting')
    os.chdir('../driver')
    cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_fore_lv4_fillin',pkl_fnm]
    ret1 = subprocess.run(cmd_list)     
    print('done')
        

if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    if len(sys.argv) == 3:
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        driver_run_forecast_LV4_fillin_apr26(arg1,arg2)
    else:
        print("Error! Wrong number of arguments in driver_run_forecast_LV4_only")
import sys
import os
import subprocess
from datetime import datetime
sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
import util_functions as utilfuns
sys.path.append('../driver')

def driver_run_pfm_phm( input_py_full, pkl_fnm ):
    print('initializing model and making the info.pkl file.')
    initfuns.initialize_model( input_py_full, pkl_fnm )
    
    # print info from pickle file
    initfuns.print_initial_model_info( pkl_fnm )
    
    # get model information
    MI = initfuns.get_model_info( pkl_fnm )

    # determine if we are doing pfm or phm
    run_type = MI['run_type']
    if run_type == 'hindcast':
        os.chdir('../driver')
        print('\n===========================')
        print('!!!starting the hindcast with subprocess!!!')
        print('===========================')
        cmd_list = ['python','-u','-W','ignore','driver_run_hind_LV123.py','driver_run_hind_LV123', pkl_fnm]
        ret1 = subprocess.run(cmd_list)
        print('\n!Finished the hindcast subprocess!\n')
        print('return code:',str(ret1.returncode),' (0=good)')     
    else:
        os.chdir('../driver')
        print('\n===========================')
        print('!!!starting the forecast with subprocess!!!')
        print('current time is: ',datetime.now())
        print('===========================')
        cmd_list = ['python','-u','-W','ignore','driver_run_forecast_LV1234.py','driver_run_forecast_LV1234',pkl_fnm]
        ret1 = subprocess.run(cmd_list)     
        print('\n!Finished the forecast subprocess!\n')
        print('return code:',str(ret1.returncode),' (0=good)')     
        utilfuns.display_timing_info(pkl_fnm)


if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
#    globals()[args[1]](*args[2:])
    if len(sys.argv) == 3:
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        driver_run_pfm_phm(arg1, arg2)
    else:
        print("Error! Wrong number of arguments in driver_run_pfm_phm.py")
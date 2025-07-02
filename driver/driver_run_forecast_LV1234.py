import sys
import os
import subprocess
from datetime import datetime

sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
sys.path.append('../driver')

def driver_run_forecast_LV1234( input_py_full, pkl_fnm ):
    t00 = datetime.now()
    # upon initialization, make the model_info.pkl file
    # initfuns.initialize_model( input_py_full, pkl_fnm )
    
    # print info from pickle file
    # initfuns.print_initial_model_info( pkl_fnm )
    
    # get model information
    MI = initfuns.get_model_info( pkl_fnm )

    # this is the loop over the levels to run
    for lvl in MI['levels_to_run']:
        print('\nstarting ' + lvl)
        os.chdir('../driver')
        cmd_list = ['python','driver_functions.py','run_fore_simulation',lvl,pkl_fnm]
        ret1 = subprocess.run(cmd_list)     
        print('done with ' + lvl)
        print(lvl, ' forecast ran correctly? ' + str(ret1.returncode) + ' (0=yes)')
        print('this took ')
        
if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])
#    if len(sys.argv) == 3:
#        arg1 = sys.argv[1]
#        arg2 = sys.argv[2]
#    else:
#        print("something went wrong!")
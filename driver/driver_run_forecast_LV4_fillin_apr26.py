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

    print('removing the Boundary condition file...')
    BC_nc_fname = MI['lv4_forc_dir'] + '/' + MI['lv4_bc_file']
    if os.path.isfile(BC_nc_fname):
        print('the file exists, removing it.')
        cmd_lst = ['rm',BC_nc_fname]
        subprocess.run(cmd_lst)
        print('done')
    else:
        print('there was no BC file to remove:')
        print(BC_nc_fname)


    if MI['fetch_time'] <= datetime(2025,4,6,0,0,0):
        move_his = 0
    else:
        move_his = 1
    
    if move_his == 1:
        print('moving the history file to the archive')
        his_name = MI['lv4_his_name_full']
        his_basename = os.path.basename(his_name)
        his_archive_name = '/dataSIO/PFM_Simulations/Archive/LV4_His/' + his_basename
        if os.path.isfile(his_name):
            cmd_list = ['mv',his_name,his_archive_name]
            subprocess.run(cmd_list)
            print('done moving it.')
        else:
            print('couldnt find the file:')
            print(his_name)
    else:
        print('we are not moving this history file!')
        

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
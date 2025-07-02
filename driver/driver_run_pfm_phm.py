import sys
import os
import subprocess
sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
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
        #from driver_run_hind_LV123 import driver_run_hind_LV123
        os.chdir('../driver')
        print('starting hindcast with subprocess...')
        cmd_list = ['python','driver_run_hind_LV123.py','driver_run_hind_LV123',input_py_full,pkl_fnm]
        ret1 = subprocess.run(cmd_list)
        print('...finished hindcast subprocess.')
        print('return code:',str(ret1.returncode),' (0=good)')     
        #driver_run_hind_LV123( input_py_full, pkl_fnm )
    else:
        #from driver_run_forecast_LV1234 import driver_run_fore_LV1234
        os.chdir('../driver')
        print('starting forecast with subprocess...')
        cmd_list = ['python','driver_run_forecast_LV1234.py','driver_run_forecast_LV1234',input_py_full,pkl_fnm]
        ret1 = subprocess.run(cmd_list)     
        print('...finished forecast subprocess.')
        print('return code:',str(ret1.returncode),' (0=good)')     
        #driver_run_fore_LV1234( input_py_full, pkl_fnm )


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
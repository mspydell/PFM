import sys

sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
sys.path.append('../driver')

def driver_run_pfm_phm( input_py_full, pkl_fnm ):
    initfuns.initialize_model( input_py_full, pkl_fnm )
    
    # print info from pickle file
    initfuns.print_initial_model_info( pkl_fnm )
    
    # get model information
    MI = initfuns.get_model_info( pkl_fnm )

    # determine if we are doing pfm or phm
    run_type = MI['run_type']
    if run_type == 'hindcast':
        from driver_run_hind_LV123 import driver_run_hind_LV123
        driver_run_hind_LV123( input_py_full, pkl_fnm )
    else:
        from driver_run_forecast_LV1234 import driver_run_fore_LV1234
        driver_run_fore_LV1234( input_py_full, pkl_fnm )


if __name__ == "__main__":
    if len(sys.argv) == 3:
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        driver_run_pfm_phm(arg1, arg2)
    else:
        print("something went wrong!")
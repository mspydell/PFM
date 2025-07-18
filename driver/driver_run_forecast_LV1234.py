import sys
import os
import subprocess
from datetime import datetime

sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
import util_functions as utilfuns
sys.path.append('../driver')

def driver_run_forecast_LV1234( pkl_fnm ):
    t00 = datetime.now()
    # upon initialization, make the model_info.pkl file
    #initfuns.initialize_model( input_py_full, pkl_fnm )
    
    # print info from pickle file
    #initfuns.print_initial_model_info( pkl_fnm )
    
    # get model information
    MI = initfuns.get_model_info( pkl_fnm )

    initfuns.remove_zero_size_files( pkl_fnm )

    # this is the loop over the levels to run
    print('Starting the loop over levels to run!')
    for lvl in MI['levels_to_run']:
        t1 = datetime.now()
        print('\n--------------------------')
        print('starting ' + lvl)
        os.chdir('../driver')
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_fore_simulation',lvl,pkl_fnm]
        ret1 = subprocess.run(cmd_list)     
        print('done with ' + lvl)
        print(lvl, ' forecast ran correctly? ' + str(ret1.returncode) + ' (0=yes)')
        t2 = datetime.now()
        print('this took:')
        print(t2-t1)
        print('\n')
        
        if lvl == 'LV4':
            print('making web.nc file...')
            t01 = datetime.now()
            ret = utilfuns.make_web_nc_file(pkl_fnm)
            t02 = datetime.now()
            print('...done making web nc file: ' + str(ret.returncode) + ' (0=good)')  
            print('this took:')
            print(t02-t01)

            print('copying and moving LV4 atm and river nc files to Archive...')
            utilfuns.copy_mv_nc_file('atm','lv4',pkl_fnm)
            utilfuns.copy_mv_nc_file('river','lv4',pkl_fnm)
            print('...done')


    lvs_to_plt = ['LV1','LV2','LV3','LV4','LV4dye']
    print('making history and dye plots for levels...')
    print(lvs_to_plt)
    t01 = datetime.now()
    utilfuns.make_simulation_plots(lvs_to_plt,pkl_fnm)
    t02 = datetime.now()
    print('...done. plotting took:')
    print(t02-t01)

    print('moving files around (FFs .sh file stuff)...')
    use_FF = 1
    if use_FF == 1:
        print('using FFs shell script!')
    else:
        print('using python the function utilfuns.end_of_sim_housekeeping...')
        utilfuns.end_of_sim_housekeeping(pkl_fnm)
        print('...done')

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
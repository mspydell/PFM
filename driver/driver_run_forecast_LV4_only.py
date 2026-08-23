import sys
import os
import subprocess
from datetime import datetime

sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
import util_functions as utilfuns
sys.path.append('../driver')

# ---------------------------------------------------------------------------
# Post-processing memory guard.
#
# make_web_nc_file reads the ~33 GB LV4 history file and make_simulation_plots
# is nearly as hungry. Each user is capped at 8 GiB on the login node, so both
# get SIGKILLed there -- web.nc returns 137 and the LV4 plot subprocess returns
# -9, and because make_web_nc_file calls sys.exit(1) on a bad return code that
# also kills the rest of this driver (archive copies and plots never run).
#
# So hand those two steps to a small slurm allocation. They run after the LV4
# sbatch has finished, so the nodes are already free and there is no risk of
# waiting on resources we are ourselves holding.
#
# If this driver is already running inside an allocation, call them in-process:
# we have the memory, and nesting srun inside srun invites trouble.
# ---------------------------------------------------------------------------
SRUN_PARTITION = 'fast-hiprio'
SRUN_MEM       = '96G'
SRUN_CPUS      = '4'          # make_simulation_plots runs 2 subprocesses in parallel
SRUN_TIME      = '02:00:00'

def in_slurm_allocation():
    return bool(os.environ.get('SLURM_JOB_ID'))

def run_utilfun_under_srun(call_src, job_name):
    # run utilfuns.<call_src> in a one-node slurm allocation. srun inherits the
    # cwd (.../PFM/driver) and the environment (so the active conda env carries
    # over). returns the CompletedProcess so callers read .returncode as before.
    py_src = ('import sys; sys.path.append("../sdpm_py_util"); '
              'import util_functions as utilfuns; ' + call_src)
    cmd = ['srun',
           '--partition=' + SRUN_PARTITION,
           '--mem=' + SRUN_MEM,
           '--cpus-per-task=' + SRUN_CPUS,
           '--time=' + SRUN_TIME,
           '--job-name=' + job_name,
           'python', '-u', '-W', 'ignore', '-c', py_src]
    print('  -> ' + job_name + ' via srun: --partition=' + SRUN_PARTITION +
          ' --mem=' + SRUN_MEM + ' --cpus-per-task=' + SRUN_CPUS +
          ' --time=' + SRUN_TIME)
    return subprocess.run(cmd)


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
        cmd_list = ['python','-u','-W','ignore','driver_functions.py','run_fore_lv4_fillin',pkl_fnm]
        print('now doing:')
        print(cmd_list)
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
            print('current time is:')
            print(t01)
            if in_slurm_allocation():
                ret = utilfuns.make_web_nc_file(pkl_fnm)
            else:
                ret = run_utilfun_under_srun(
                    'utilfuns.make_web_nc_file(' + repr(pkl_fnm) + ')',
                    'PFM_webnc')
            t02 = datetime.now()
            print('...done making web nc file: ' +
                  str(getattr(ret, 'returncode', 'n/a')) + ' (0=good)')  
            print('this took:')
            print(t02-t01)
            print('current time is:')
            print(t02)

            print('copying and moving LV4 atm to Archive with Popen...')
            utilfuns.copy_mv_nc_file_v2('atm','lv4',pkl_fnm)
            print('copying and moving LV4 river to Archive with Popen...')
            utilfuns.copy_mv_nc_file_v2('river','lv4',pkl_fnm)
            print('moving on...')


    #lvs_to_plt = ['LV1','LV2','LV3','LV4','LV4dye']
    lvs_to_plt = ['LV4','LV4dye']
    print('making history and dye plots for levels...')
    print(lvs_to_plt)
    t01 = datetime.now()
    print('current time is:')
    print(t01)

    if in_slurm_allocation():
        utilfuns.make_simulation_plots(lvs_to_plt,pkl_fnm)
    else:
        run_utilfun_under_srun(
            'utilfuns.make_simulation_plots(' + repr(lvs_to_plt) + ', ' +
            repr(pkl_fnm) + ')',
            'PFM_lv4plots')
    t02 = datetime.now()
    print('...done. plotting took:')
    print(t02-t01)
    print('current time is:')
    print(t02)


    print('moving files around (FFs .sh file stuff)...')
    use_FF = 1
    if use_FF == 1:
        print('going to use FFs shell script after python finishes!!! ')
        print('current time is:')
        print(datetime.now())
    else:
        print('using python the function utilfuns.end_of_sim_housekeeping...')
        utilfuns.end_of_sim_housekeeping(pkl_fnm)
        print('...done')

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
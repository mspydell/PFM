import sys
#import matplotlib.pyplot as plt
#import numpy as np
import os
import subprocess
import time

##############

sys.path.append('../sdpm_py_util')

import util_functions as utlfuns 
import plotting_functions as pltfuns


def run_slurm_LV4( PFM ):

    cwd = os.getcwd()
    # start the python program that checks for new swan files...
    fname = PFM['lv4_swan_rst_name'] # this is the root name of the swan file
                                     # LV4_swan_rst_202412010600.dat-001 etc
    nfiles = str( PFM['gridinfo']['L4','np_swan'] )
    # how long one pauses to check for swan restart writes, needs to be 1/4 of how often it writes.
    swan_check_freq = str( PFM['lv4_swan_check_freq_sec'] )
    cmd_list = ['python','-W','ignore','swan_functions.py','check_and_move',fname,swan_check_freq,nfiles]
    os.chdir('../sdpm_py_util')
    print('starting the background process that looks for swan restart files every')
    print(swan_check_freq + ' seconds...')
    checking = subprocess.Popen(cmd_list)   
    
    os.chdir(PFM['lv4_run_dir'])
    print('run_slurm_LV4: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV4_SLURM.sb']
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(int(swan_check_freq))  # Check every dt_sec second

    checking.terminate() # turn off the function that moves swan rst files
    print('terminated the swan restart file check and move subprocess.')

    print(proc)
    print('run_slurm_LV4: run command: ', cmd_list )
    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc


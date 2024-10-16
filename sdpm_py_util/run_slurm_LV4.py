import sys
#import matplotlib.pyplot as plt
#import numpy as np
import os
import subprocess

##############

sys.path.append('../sdpm_py_util')

import util_functions as utlfuns 
import plotting_functions as pltfuns


def run_slurm_LV4( PFM ):

    cwd = os.getcwd()
    os.chdir(PFM['lv4_run_dir'])
    print('run_slurm_LV4: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV4_SLURM.sb']
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(proc)
    print('run_slurm_LV4: run command: ', cmd_list )
    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc


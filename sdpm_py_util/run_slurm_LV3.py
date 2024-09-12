import sys
#import matplotlib.pyplot as plt
#import numpy as np
import os
import subprocess

##############

sys.path.append('../sdpm_py_util')

import util_functions as utlfuns 
import plotting_functions as pltfuns


def run_slurm_LV3( PFM ):

    cwd = os.getcwd()
    os.chdir(PFM['lv3_run_dir'])
    print('run_slurm_LV3: current directory is now: ', os.getcwd() )
    
    cmd_list = ['sbatch', '--wait' ,'LV3_SLURM.sb']
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #proc = subprocess.run("/cm/shared/apps/slurm/current/bin/sbatch  --wait LV1_SLURM.sb")

    print(proc)
    print('run_slurm_LV3: run command: ', cmd_list )
    print('subprocess slurm ran correctly? ' + str(proc.returncode) + ' (0=yes)')

    # change directory back to what it was before
    os.chdir(cwd)
 
    return proc

#       if args.run_roms:
#            tt0 = time()
#            # Run ROMS using the batch script.                                                                                                                                                            
#            if 'klone' in Ldir['lo_env']:
#                cmd_list = ['sbatch', '-p', 'compute', '-A', 'macc',
#                    str(roms_out_dir / 'klone_batch.sh')]
#            elif 'mox' in Ldir['lo_env']:
#                cmd_list = ['sbatch', '-p', 'macc', '-A', 'macc',
#                    str(roms_out_dir / 'mox_batch.sh')]
#           proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # now we need code to wait until the run has completed                                                                                                                                 

            # these are for checking on the run using squeue                                                                                                                                              
#           if 'mox' in Ldir['lo_env']:
#                cmd_list = ['squeue', '-p', 'macc']
#            elif 'klone' in Ldir['lo_env']:
#                cmd_list = ['squeue', '-A', 'macc']

            # first figure out if it has started                                                                                                                                                          
#            for rrr in range(10):
#                if rrr == 9:
#                    print('Took too long for job to start: quitting')
#                    sys.exit()
#                proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#                stdout, stderr = proc.communicate()
#                if jobname not in stdout.decode():
#                    if args.verbose:
#                        print('still waiting for run to start ' + str(rrr))#
#			sys.stdout.flush()#
#	        elif jobname in stdout.decode():
#                    if args.verbose:
#                        print('run started ' + str(rrr))
#                        sys.stdout.flush()
#                    break
#                sleep(10)



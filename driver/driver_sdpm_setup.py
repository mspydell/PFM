"""
This is the driver for initially setting up the SDPM 

This code:
- runs creates all the folders that are at the same level as path_to_SDPM/SDPM/
eg. path_to_SDPM/SDPM_user, path_to_SDPM/SDPM_data, path_to_SDPM/SDPM_output
- it also creates a copy of path_to_SDPM/SDPM/get_sdpm_info.py and places it in
path_to_SDPM_user/ as that is the file that is read by the program for model runs

testing on mac:
python3 driver_roms3.py -g cas6 -t v00 -x uu0mb -s continuation -0 2021.01.01 -np 196 -N 28 --get_forcing False --run_roms False --move_his False

testing on mox:
python3 driver_roms3.py -g cas6 -t v00 -x uu0mb -s continuation -0 2021.01.01 -np 196 -N 28 --short_roms True < /dev/null > uu0mb_test.log &

production run on mox:
python3 driver_roms3.py -g cas6 -t v00 -x uu0mb -s continuation -0 2021.01.01 -1 2021.01.02 -np 196 -N 28 < /dev/null > uu0mb_a.log &
python3 driver_roms3.py -g cas6 -t v00 -x uu0mb -0 2021.01.03 -1 2021.12.31 -np 196 -N 28 < /dev/null > uu0mb_b.log &
"""

import os
import sys
from pathlib import Path
# from datetime import datetime

testing = 1
if testing ==1:
    from importlib import reload

def make_dir(pth, clean=False):
    """
    >>> WARNING: Be careful! This can delete whole directory trees. <<<
    Make a directory from the path "pth" which can:
    - be a string or a pathlib.Path object
    - be a relative path
    - have a trailing / or not
    Use clean=True to clobber the existing directory (the last one in pth).
    This function will create all required intermediate directories in pth.
    """
    if clean == True:
        shutil.rmtree(str(pth), ignore_errors=True)
    Path(pth).mkdir(parents=True, exist_ok=True)


# get the path to the directory where all SDPM folders should be
parent = Path(__file__).absolute().parent.parent.parent

LO = parent / 'PFM'
# the location of user some user specified grid etc
LOu = parent / 'PFM_user'
# data is where the input files, atm, ocn IC, ocn BC, etc are found
data = parent / 'PFM_data'
# LOo is the location where his.nc files etc will go
LOo = parent / 'PFM_output'
LOtest = parent / 'PFM_test'

for ii in [LOu,data,LOo]:
    dum = os.path.isdir(ii)
    if dum == False:
        print('need to make ' + str(ii))
        print('making this directory...')
        make_dir(ii,clean=False)
        print('...done')
    else:
        print(str(ii) + ' is already a directory')

# this file comes with git clone and is needed in ...
fnm0 = str(LO) + '/driver/get_sdpm_info.py'
# this is the file that the model needs
fnm = str(LOu) + '/get_sdpm_info.py'

dum = os.path.isfile(fnm)
# move the standard get_sdpm_info.py file to the user directory.
if dum == False:
    print('need to copy ' + fnm0)
    print('to '+ str(LOu) + ' ...')
    os.system('scp ' + str(fnm0) + ' ' + str(fnm))
    print('... done')  
else:
    print(fnm + ' already exists in the right spot')


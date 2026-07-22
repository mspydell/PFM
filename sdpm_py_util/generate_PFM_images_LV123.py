import sys
import os
import subprocess
from datetime import datetime

sys.path.append('../sdpm_py_util')
import init_funs_forecast as initfuns
import util_functions as utilfuns
sys.path.append('../driver')


pkl_fnm = '/scratch/PFM_Simulations/forecast_info.pkl'

#lvs_to_plt = ['LV4','LV4dye']
lvs_to_plt = ['LV1','LV2','LV3']
print('making history and dye plots for levels...')
print(lvs_to_plt)
t01 = datetime.now()
print('current time is:')
print(t01)

utilfuns.make_simulation_plots(lvs_to_plt,pkl_fnm)
t02 = datetime.now()
print('...done. plotting took:')
print(t02-t01)
print('current time is:')
print(t02)

import numpy as np
import os
import sys
sys.path.append('../sdpm_py_util')
import init_funs as initfuns

# the functions in here are used to make the 

def run_hind_LV1(t1str,pkl_fnm):
    MI = initfun.get_model_info(pkl_fnm)

def run_hind_simulation(t1str,lvl,pkl_fnm):
    if lvl == 'LV1':
        run_hind_LV1(t1str,pkl_fnm)
    if lvl == 'LV2':
        run_hind_LV2(t1str,pkl_fnm)
    if lvl == 'LV3':
        run_hind_LV3(t1str,pkl_fnm)
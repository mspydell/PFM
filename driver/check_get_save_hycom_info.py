from datetime import datetime, timedelta
import numpy as np
import sys
import glob
sys.path.append('/home/mspydell/models/PFM_root/PFM/sdpm_py_util')
import ocn_functions as ocnfuns

import pickle
import os


# ========================
# first some functions
# ========================

def get_full_hy_file_list(fore_date):
    var_names = ['_ssh','_s3z','_t3z','_u3z','_v3z']
    dstr0 = fore_date + 'T12:00' # this is the forecast time
    
    t0 = datetime.strptime(dstr0,'%Y-%m-%dT%H:%M') # the first forecast time
    t2 = t0 + 8*timedelta(days=1) # the last forecast time

    file_names = []    

    for vnm in var_names:
        dhr = 3
        dtff = t0
        if vnm == '_ssh':
            dhr = 1
        while dtff <= t2:
            ffname = 'hy'+ vnm + '_' + dstr0 + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'
            file_names.append(ffname)
            dtff = dtff + dhr * timedelta(hours=1)

    return file_names


def get_fore_dates(hydir):
    # this retuns a list of forecast dates in the hycom data directory hydir
    fore_dates = []
    try:
        #filenames = glob.glob(hydir+'*.nc')
        filenames = [os.path.basename(x) for x in glob.glob(hydir+'*.nc')]
    except FileNotFoundError:
        print('strange, the hycom directory seems to be empty...')
        fore_dates = []
        return fore_dates

    ff = []
    for fn in filenames:
        ff.append(fn[7:17])

    fore_dates = list(dict.fromkeys(ff))

    return fore_dates


def check_present_hycom_data_v2():
    t_now = datetime.utcnow()
    t_check = t_now.strftime("%Y%m%d%H")
    HY_info = dict()
    hydir = '/scratch/PFM_Simulations/hycom_data/'
    fore_dates = get_fore_dates(hydir)
    
    got_them = dict()
    tots = dict()

    for vrs in ['ssh','s3z','t3z','u3z','v3z']:
        if vrs == 'ssh':
            tots[vrs]=193
        else:
            tots[vrs]=65

    for fd in fore_dates:
        full_file_list = get_full_hy_file_list(fd)
        for vrs in ['ssh','s3z','t3z','u3z','v3z']:
            got_them[vrs] = 0

        for fn in full_file_list:
            vrs = fn[3:6]
            if os.path.exists(hydir+fn):
                got_it = 1
            else:
                got_it = 0
            got_them[vrs] = got_them[vrs]+got_it
    
        for vrs in ['ssh','s3z','t3z','u3z','v3z']:
            HY_info[t_check,fd,vrs] = got_them[vrs] / tots[vrs]

    dir_out = '/scratch/PFM_Simulations/hycom_data/'        
    fn_out = 'hycom_info_' + t_check + '.pkl'
    with open(dir_out+fn_out, 'wb') as file:
        # Use pickle.dump() to serialize and save the dictionary
        pickle.dump(HY_info, file)

    return HY_info

# ========================
# the main script
# ========================

t0s = ocnfuns.stored_hycom_dates()
miss_dict = {}
total_missing = []
for tt in t0s:
    t1 = datetime.strptime(tt,'%Y-%m-%d') + 0.5* timedelta(days=1)
    yyyymmdd = t1.strftime('%Y%m%d')
    t2 = t1+8.0*timedelta(days=1)
    times = [t1,t2]
    n0, num_missing, miss_dict[tt] = ocnfuns.check_hycom_data(yyyymmdd,times)
    for mm in miss_dict[tt]:
        total_missing.append(mm)

tcnt = datetime.strptime(t0s[-1],'%Y-%m-%d') + timedelta(days=1) 
tnow = datetime.utcnow()
tend = tnow.strftime('%Y%m%d')
tend2 = datetime.strptime(tend,'%Y%m%d')

while tcnt <= tend2 - timedelta(days=1): # these are the new dates to get hycomdata for...
    t1 = tcnt + 0.5* timedelta(days=1)
    yyyymmdd = t1.strftime('%Y%m%d')
    t2 = t1+8.0*timedelta(days=1)
    times = [t1,t2]
    n0, num_missing, miss_dict[tcnt] = ocnfuns.check_hycom_data(yyyymmdd,times)
    for mm in miss_dict[tcnt]:
        total_missing.append(mm)
    tcnt = tcnt + timedelta(days=1)


# lets try and get these missing files...
ocnfuns.get_hycom_data_fnames_v3(total_missing)
HY_info = check_present_hycom_data_v2()



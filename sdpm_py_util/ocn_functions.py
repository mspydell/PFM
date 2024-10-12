# the ocean functions will be here

from datetime import datetime
from datetime import timedelta
import time
import gc
import resource
import pickle
import grid_functions as grdfuns
import os
import os.path
import pickle
from scipy.spatial import cKDTree
import glob

#sys.path.append('../sdpm_py_util')
from get_PFM_info import get_PFM_info

import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.interpolate import interp1d

import seawater
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

#import util_functions as utlfuns 
from util_functions import s_coordinate_4
#from pydap.client import open_url
import sys

import warnings
warnings.filterwarnings("ignore")


def para_loop(url,dtff,aa,PFM,dstr_ft):
    # this is the function that is parallelized
    # the input is url: the hycom url to get data from
    # dtff: the time stamp of the forecast, ie, the file we want
    # aa: a box that defines the region of interest
    # PFM: used to set the path of where the .nc files go
    # dstr_ft: the date string of the forecast model run. ie. the first
    #          time stamp of the forecast
    north = aa[0]
    south = aa[1]
    west = aa[2]
    east = aa[3]

    # time limits
    dtff_adv = dtff+timedelta(hours=2) # hours=2 makes it only get 1 time.
    dstr0 = dtff.strftime('%Y-%m-%dT%H:%M')
    dstr1 = dtff_adv.strftime('%Y-%m-%dT%H:%M')
    # use subprocess.call() to execute the ncks command
    vstr = 'surf_el,water_temp,salinity,water_u,water_v'
    #where to save the data
    
    ffname = 'hy_' + dstr_ft + '_' + dtff.strftime("%Y-%m-%dT%H:%M") + '.nc'
    full_fn_out = PFM['lv1_forc_dir'] +'/' + ffname

    cmd_list = ['ncks',
        '-d', 'time,'+dstr0+','+dstr1,
        '-d', 'lon,'+str(west)+','+str(east),
        '-d', 'lat,'+str(south)+','+str(north),
        '-v', vstr,
        url ,
                '-4', '-O', full_fn_out]

    # run ncks
    ret1 = subprocess.call(cmd_list, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return ret1

def para_loop_new(url,dtff,aa,bb,PFM,dstr_ft):
    # this is the function that is parallelized
    # the input is url: the hycom url to get data from
    # dtff: the time stamp of the forecast, ie, the file we want
    # aa: a box that defines the region of interest
    # PFM: used to set the path of where the .nc files go
    # dstr_ft: the date string of the forecast model run. ie. the first
    #          time stamp of the forecast
    north = aa[0]
    south = aa[1]
    west = aa[2]
    east = aa[3]

    # time limits
    dtff_adv = dtff+timedelta(hours=0.5) # hours=2 makes it only get 1 time.
    dstr0 = dtff.strftime('%Y-%m-%dT%H:%M')
    dstr1 = dtff_adv.strftime('%Y-%m-%dT%H:%M')
    # use subprocess.call() to execute the ncks command
    if bb == '_ssh':
        vstr = 'surf_el' 
    if bb == '_t3z':
        vstr = 'water_temp'
    if bb == '_s3z':
        vstr = 'salinity'
    if bb == '_u3z':
        vstr = 'water_u'
    if bb == '_v3z':
        vstr = 'water_v'
    #where to save the data
    
    ffname = 'hy'+ bb + '_' + dstr_ft + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'

    full_fn_out = PFM['lv1_forc_dir'] +'/' + ffname

    cmd_list = ['ncks',
        '-d', 'time,'+dstr0+','+dstr1,
        '-d', 'lon,'+str(west)+','+str(east),
        '-d', 'lat,'+str(south)+','+str(north),
        '-v', vstr,
        url ,
                '-4', '-O', full_fn_out]

    # run ncks
    hide_out = 1
    if hide_out == 1:
        print('trying to hide output... does not work')
        ret1 = subprocess.call(cmd_list, stdout=subprocess.STDOUT, stderr=subprocess.DEVNULL)
    else:
        ret1 = subprocess.call(cmd_list, stderr=subprocess.DEVNULL)
    return ret1

def hycom_grabber(url,dtff,vnm,dstr_ft):
    # this is the function that is parallelized
    # the input is url: the hycom url to get data from
    # dtff: the time stamp of the forecast, ie, the file we want
    # aa: a box that defines the region of interest
    # PFM: used to set the path of where the .nc files go
    # dstr_ft: the date string of the forecast model run. ie. the first
    #          time stamp of the forecast
    
    PFM = get_PFM_info()

    south = PFM['latlonbox']['L1'][0]
    north = PFM['latlonbox']['L1'][1]
    west = PFM['latlonbox']['L1'][2]+360.0
    east = PFM['latlonbox']['L1'][3]+360.0

    # time limits
    dtff_adv = dtff+timedelta(hours=0.5) # hours=2 makes it only get 1 time.
    dstr0 = dtff.strftime('%Y-%m-%dT%H:%M')
    dstr1 = dtff_adv.strftime('%Y-%m-%dT%H:%M')
    # use subprocess.call() to execute the ncks command
    if vnm == '_ssh':
        vstr = 'surf_el' 
    if vnm == '_t3z':
        vstr = 'water_temp'
    if vnm == '_s3z':
        vstr = 'salinity'
    if vnm == '_u3z':
        vstr = 'water_u'
    if vnm == '_v3z':
        vstr = 'water_v'
    #where to save the data
    
    ffname = 'hy'+ vnm + '_' + dstr_ft + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'

    #full_fn_out = PFM['hycom_data_dir'] + ffname
    full_fn_out = '/scratch/PFM_Simulations/hycom_data/' + ffname

    tst_err = 1
    if tst_err == 1: # this didn't do it. got an error. should fix so that ncks doesn't make a ton of errors...
        cmd_list = ['ncks',
            '-q',
            '-D', '0',
            '-d', 'time,'+dstr0+','+dstr1,
            '-d', 'lon,'+str(west)+','+str(east),
            '-d', 'lat,'+str(south)+','+str(north),
            '-v', vstr,
            url ,
                    '-4', '-O', full_fn_out]
    else:
        cmd_list = ['ncks',
            '-d', 'time,'+dstr0+','+dstr1,
            '-d', 'lon,'+str(west)+','+str(east),
            '-d', 'lat,'+str(south)+','+str(north),
            '-v', vstr,
            url ,
            '-4', '-O', full_fn_out ]

    # run ncks
    ret1 = subprocess.call(cmd_list, stderr=subprocess.DEVNULL)
    return ret1

def hycom_grabber_v2(url,dtff,vnm,fn_out):
    # this is the function that is parallelized
    # the input is url: the hycom url to get data from
    # dtff: the time stamp of the forecast, ie, the file we want
    # aa: a box that defines the region of interest
    # PFM: used to set the path of where the .nc files go
    # dstr_ft: the date string of the forecast model run. ie. the first
    #          time stamp of the forecast
    
    PFM = get_PFM_info()

    south = PFM['latlonbox']['L1'][0]
    north = PFM['latlonbox']['L1'][1]
    west = PFM['latlonbox']['L1'][2]+360.0
    east = PFM['latlonbox']['L1'][3]+360.0

    # time limits
    dtff_adv = dtff+timedelta(hours=0.5) # hours=2 makes it only get 1 time.
    dstr0 = dtff.strftime('%Y-%m-%dT%H:%M')
    dstr1 = dtff_adv.strftime('%Y-%m-%dT%H:%M')
    # use subprocess.call() to execute the ncks command
    if vnm == '_ssh':
        vstr = 'surf_el' 
    if vnm == '_t3z':
        vstr = 'water_temp'
    if vnm == '_s3z':
        vstr = 'salinity'
    if vnm == '_u3z':
        vstr = 'water_u'
    if vnm == '_v3z':
        vstr = 'water_v'
    

    tst_err = 1
    if tst_err == 1: # this didn't do it. got an error. should fix so that ncks doesn't make a ton of errors...
        cmd_list = ['ncks',
            '-q',
            '-D', '0',
            '-d', 'time,'+dstr0+','+dstr1,
            '-d', 'lon,'+str(west)+','+str(east),
            '-d', 'lat,'+str(south)+','+str(north),
            '-v', vstr,
            url ,
            '-4', '-O', fn_out]
    else:
        cmd_list = ['ncks',
            '-d', 'time,'+dstr0+','+dstr1,
            '-d', 'lon,'+str(west)+','+str(east),
            '-d', 'lat,'+str(south)+','+str(north),
            '-v', vstr,
            url ,
            '-4', '-O', fn_out]

    # run ncks
    ret1 = subprocess.call(cmd_list,stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
    return ret1


def check_hycom_data(yyyymmdd,times):
    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]
    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00' # this is the forecast time

    var_names = ['_ssh','_s3z','_t3z','_u3z','_v3z']
    nc_out_names = [] # this will be the list of all of the hycom.nc files that we are checking to see if they are there
    
    hycom_dir = '/scratch/PFM_Simulations/hycom_data/'

    dtff = times[0]
    t2 = times[1]

    num_missing = 0
    num_files = 0
    zeta_1hr = 1
    miss_list = []

    for vnm in var_names:
        dhr = 3
        dtff = times[0]
        if vnm == '_ssh' and zeta_1hr == 1:
            dhr = 1
        while dtff <= t2:
            ffname = 'hy'+ vnm + '_' + dstr0 + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'
            full_fn_out = hycom_dir + ffname
            nc_out_names.append(ffname)
            num_files = num_files + 1
            dtff = dtff + dhr * timedelta(hours=1)
            if os.path.isfile(full_fn_out) == False:
                #print(full_fn_out)
                miss_list.append(ffname)
                num_missing = num_missing + 1

    return num_files, num_missing, miss_list

def stored_hycom_dates():
    # this function returns the list 'yyyy-mm-dd' of the hycom data we have downloaded and stored
    hycom_dir = '/scratch/PFM_Simulations/hycom_data/'
    fns = glob.glob(hycom_dir + '*.nc')

    fns3 = []
    for fns2 in fns:
        fns3.append(fns2[43:53])

    set_res = set(fns3)
    list_res = list(set_res)
    list_res2 = sorted(list_res) # 
    return list_res2

def get_missing_file_num_list(hy_dates):
    # this function returns a list of the number of files missing for each string in hy_dates

    missing = []
    for dts in hy_dates:
        dstr0 = dts + 'T12:00' # this is the forecast time
        t1 =  datetime.strptime(dstr0,"%Y-%m-%dT%H:%M")
        t2 = t1 + 8.0 * timedelta(days=1)
        times = [t1,t2]
        yyyymmdd = dts[0:4] + dts[5:7] + dts[8:10]
        n0, num_missing = check_hycom_data(yyyymmdd,times)
        missing.append(num_missing)

    return missing

def clean_hycom_dir():

    #hycom_dir = '/scratch/PFM_Simulations/hycom_data/'
    hycom_dir = PFM['hycom_data_dir']
    hy_dates = stored_hycom_dates()
    if len(hy_dates) == 1:
        print('there is only one hycom date in the directoy')
        print('we are not deleting any files.')
    else:
        fn_miss_list = get_missing_file_num_list(hy_dates)

        fn2 = np.array(fn_miss_list)
        ind0 = np.where(fn2 == 0)[0]
        keeper = np.max(ind0)

        cnt = 0
        for dt in hy_dates:
            if cnt != keeper:
                print('the date ' + dt + ' will be deleted')
                fnsd = hycom_dir + '*' + dt + 'T12:00_2*.nc'
                fls2d = glob.glob(fnsd)
                #print(fls2d)
                #print('will be deleted')
                for ff in fls2d:
                    os.remove(ff)
            else:
                print('the date ' + dt + ' will not be deleted')
            cnt=cnt+1


def refresh_hycom_dir():

    t0s = stored_hycom_dates()
    if len(t0s) > 1: # first clean the directory if there is more than 1 date in it...
        clean_hycom_dir()
        t0s = stored_hycom_dates() # now len(t0s) will be 1!

    dstr0 = t0s[0] + 'T12:00' # this is the forecast time
    t0 =  datetime.strptime(dstr0,"%Y-%m-%dT%H:%M")
    tnow = datetime.now()
    tget = []

    if t0>tnow - 1 * timedelta(days=1):
        print('the directory is up to date and there are no possible hycom files to download')
    else:   # with 1 below, we try to get yesterdays forecast. Not often there but we try!
        while t0<tnow - 1 * timedelta(days=1):  
            t0 = t0 + timedelta(days=1)
            yyyymmdd = "%d%02d%02d" % (t0.year, t0.month, t0.day)
            tget.append(yyyymmdd)

        print('we will download, hopefully, the hycom with these forecast dates:')
        print(tget)
        for tt in tget:
            print('downloading hycom ' + tt + ' forecast...')
            get_hycom_data_1hr(tt)
            print('...done')


def get_hycom_foretime(t1str,t2str):

    refresh_hycom_dir() # this function adds hycom files to the directory. There will always be a forecast with 8 days of data
    t0s = stored_hycom_dates()
    t1 =  datetime.strptime(t1str,"%Y%m%d%H%M")
    t2 =  datetime.strptime(t2str,"%Y%m%d%H%M")
    times = [t1,t2]

    missing = []
    for dts in t0s:
        yyyymmdd = dts[0:4] + dts[5:7] + dts[8:10]
        n0, num_missing = check_hycom_data(yyyymmdd,times)
        missing.append(num_missing)

    fn2 = np.array(missing)
    ind0 = np.where(fn2 == 0)[0]
    keeper = np.max(ind0)
    tkeep = t0s[keeper]
    yyyymmdd = tkeep[0:4] + tkeep[5:7] + tkeep[8:10]

    return yyyymmdd


def get_hycom_foretime_v2(t1str,t2str):

    PFM = get_PFM_info()
    hycom_dir = PFM['hycom_data_dir'] # this has the trailing /
    t0s = stored_hycom_dates()
    tnow = datetime.now()
    tend = t0s[-1]
    t0 =  datetime.strptime(tend,"%Y-%m-%d")
    t0 = t0 + timedelta(days=1)
    while t0 < tnow - timedelta(days = 2):
        t0str = t0.strftime('%Y%m%d')
        print('we are trying ll of the hycom ', t0str, ' forecast')
        get_hycom_data_1hr(t0str)
        t0 = t0 + timedelta(days=1)

    t0s = stored_hycom_dates()
    miss_dict = {}
    nms_dict = {}
    print('how many total files are we currently missing in the hycom directory?')
    for dts in t0s:
        yyyymmdd = dts[0:4] + dts[5:7] + dts[8:10]
        yyyymmddhhmm = yyyymmdd + '1200'
        t1 = datetime.strptime(yyyymmddhhmm,'%Y%m%d%H%M')
        t2 = t1+8.0*timedelta(days=1)
        times = [t1,t2]
        n0, num_missing, miss_dict[yyyymmdd] = check_hycom_data(yyyymmdd,times)
        nms_dict[yyyymmdd] = num_missing
        print('the ', yyyymmdd, ' hycom forecast is missing')
        print(str(num_missing), ' files out of ', str(n0))

    print('\nwe will try and get these missing files...')

    for dts in t0s:
        yyyymmdd = dts[0:4] + dts[5:7] + dts[8:10]
        if nms_dict[yyyymmdd] > 0:
            print('getting hycom ', yyyymmdd, ' files...')
            get_hycom_data_fnames(yyyymmdd,miss_dict[yyyymmdd])
            print('...done')

    print('now how many total files are we missing?')
    t0s = stored_hycom_dates()
    nms = []
    for dts in t0s:
        yyyymmdd = dts[0:4] + dts[5:7] + dts[8:10]
        yyyymmddhhmm = yyyymmdd + '1200'
        t1 = datetime.strptime(yyyymmddhhmm,'%Y%m%d%H%M')
        t2 = t1+8.0*timedelta(days=1)
        times = [t1,t2]
        n0, num_missing, miss_dict[yyyymmdd] = check_hycom_data(yyyymmdd,times)
        nms.append(num_missing)
        print('the ', yyyymmdd, ' hycom forecast is missing')
        print(str(num_missing), ' files out of ', str(n0))

    fn2 = np.array(nms)
    ind0 = np.where(fn2 == 0)[0]
    keeper_i = np.max(ind0)
    if keeper_i>0:
        print('removing old unneeded full forecasts...')
        ii = 0
        while ii < keeper_i:
            dt = t0s[ii]
            print('the date ' + dt + ' will be deleted')
            fnsd = hycom_dir + '*' + dt + 'T12:00_2*.nc'
            fls2d = glob.glob(fnsd)
            for ff in fls2d:
                os.remove(ff)
            ii+=1
        print('...done')
    else:
        print('the oldest forecast is the only full forecast!')
   
    print('which hycom forecast has the times we need for the PFM simulation?')

    t0s = stored_hycom_dates()
    t1 =  datetime.strptime(t1str,"%Y%m%d%H%M")
    t2 =  datetime.strptime(t2str,"%Y%m%d%H%M")
    times = [t1,t2]

    missing = []
    print('for the PFM simulation starting from ', t1)
    print('and ending at ', t2)
    for dts in t0s:
        yyyymmdd = dts[0:4] + dts[5:7] + dts[8:10]
        n0, num_missing, dum = check_hycom_data(yyyymmdd,times)
        missing.append(num_missing)
        print('for the ', yyyymmdd, ' hycom simulation, we are missing')
        print(num_missing, ' files.')

    fn2 = np.array(missing)
    ind0 = np.where(fn2 == 0)[0]
    keeper = np.max(ind0)
    tkeep = t0s[keeper]
    print('so we will use the')
    yyyymmdd = tkeep[0:4] + tkeep[5:7] + tkeep[8:10]
    print(yyyymmdd, ' hycom simulation for this PFM forecast\n')
    return yyyymmdd




def get_hycom_data(yyyymmdd):
    # this function gets all of the new hycom data as separate files for each field (ssh,temp,salt,u,v) and each time
    # and puts each .nc file in the directory for hycom data
    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]

    PFM=get_PFM_info()
    ocn_name = ['https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02','runs/FMRC_ESPC-D-V02','_RUN_'+ yyyy + '-' + mm + '-' + dd + 'T12:00:00Z']
    var_names = ['_ssh','_s3z','_t3z','_u3z','_v3z']

 
    # time limits
    Tfor = 8.0 # hycom should go out 8 days.
    # the first time to get
    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
    t00 = datetime.strptime(dstr0,"%Y-%m-%dT%H:%M")
    # the last time to get
    t10 = t00 + Tfor * timedelta(days=1)

    # form list of days to get, datetimes
    dt0 = t00
    dt1 = t10 
    dt_list_full = []
    dtff = dt0
    ncfiles = [] # this is the list with paths of all of the .nc files made with ncks_para
            #timestamps
    while dtff <= dt1:
        for bb in var_names:
            ffn = PFM['lv1_forc_dir'] + '/hy'+ bb + '_' + dstr0 + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'
            ncfiles.append(ffn)
        
        dt_list_full.append(dtff) # these are the times to get forecast for...
        dtff = dtff + timedelta(hours=3)

        
    #this is where we para
    tt0 = time.time()

    # create parallel executor
    with ThreadPoolExecutor() as executor:
        threads = []
        for bb in var_names:
            for dtff in dt_list_full: # datetimes of forecast time
                fn = hycom_grabber #define function
                hycom = ocn_name[0]+bb+ocn_name[1]+bb+ocn_name[2]
                args = [hycom,dtff,bb,dstr0] #define args to function
                kwargs = {} #
                # start thread by submitting it to the executor
                threads.append(executor.submit(fn, *args, **kwargs))
            
        for future in as_completed(threads):
            # retrieve the result
            result = future.result()
            # report the result
    
    #result = tt0
    #print('Time to get all files using parallel ncks = %0.2f sec' % (time.time()-tt0))
    #print('Return code = ' + str(result) + ' (0=success, 1=skipped ncks)')


def get_hycom_data_fnames(yyyymmdd,fnames):
    # this function gets all of the new hycom data as separate files for each field (ssh,temp,salt,u,v) and each time
    # and puts each .nc file in the directory for hycom data
    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]

    PFM=get_PFM_info()
    ocn_name = ['https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02','runs/FMRC_ESPC-D-V02','_RUN_'+ yyyy + '-' + mm + '-' + dd + 'T12:00:00Z']
 
    with ThreadPoolExecutor() as executor:
        threads = []
        for file_name in fnames:
            #print('getting: ', file_name)
            fn = hycom_grabber_v2
            ffn = PFM['hycom_data_dir'] + '/'+ file_name
            #print('putting here: ',ffn)
            bb = file_name[2:6]
            hycom = ocn_name[0] + bb + ocn_name[1] + bb + ocn_name[2]
            dtff = datetime.strptime(file_name[24:40],'%Y-%m-%dT%H:%M')
            args = [hycom,dtff,bb,ffn]
            kwargs = {} #
            # start thread by submitting it to the executor
            threads.append(executor.submit(fn, *args, **kwargs))
            
        for future in as_completed(threads):
                # retrieve the result
            result = future.result()
            #print(result)
                # report the result        


def get_hycom_data_1hr(yyyymmdd):
    # this function gets all of the new hycom data as separate files for each field (ssh,temp,salt,u,v) and each time
    # and puts each .nc file in the directory for hycom data

    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]
    ocn_name = ['https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02','runs/FMRC_ESPC-D-V02','_RUN_'+ yyyy + '-' + mm + '-' + dd + 'T12:00:00Z']
    
    # time limits
    Tfor = 8.0 # hycom should go out 8 days.
    # the first time to get
    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
    t00 = datetime.strptime(dstr0,"%Y-%m-%dT%H:%M")
    # the last time to get
    t10 = t00 + Tfor * timedelta(days=1)
    
    t1str = "%d%02d%02d%02d%02d" % (t00.year, t00.month, t00.day, t00.hour, t00.minute)
    t2str = "%d%02d%02d%02d%02d" % (t10.year, t10.month, t10.day, t10.hour, t10.minute)
    ncfiles = get_hycom_nc_file_names(yyyymmdd,t1str,t2str)

    # create parallel executor
    with ThreadPoolExecutor() as executor:
        threads = []
        for fname in ncfiles:
            dtffstr = fname[-19:-3] # this is the forecast time we want
            dtff    = datetime.strptime(dtffstr,"%Y-%m-%dT%H:%M")
            bb = fname[-41:-37] # this is the field, eg. _ssh, _s3z, etc
            fun = hycom_grabber #define function
            hycom = ocn_name[0]+bb+ocn_name[1]+bb+ocn_name[2]
            args = [hycom,dtff,bb,dstr0] #define args to function
            kwargs = {} #
            # start thread by submitting it to the executor
            threads.append(executor.submit(fun, *args, **kwargs))
            
        for future in as_completed(threads):
            # retrieve the result
            result = future.result()
            # report the result

def get_hycom_nc_file_names(yyyymmdd,t1str,t2str):

    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]
    # the time of the hycom forecast
    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'


    var_names = ['_s3z','_t3z','_u3z','_v3z','_ssh']
    PFM=get_PFM_info()

    t00 =  datetime.strptime(t1str,"%Y%m%d%H%M")
    t10 =  datetime.strptime(t2str,"%Y%m%d%H%M")

    # form list of days to get, datetimes
    dt0 = t00
    dt1 = t10 
    ncfiles = [] # this is the list of all of the nc files we will get from hycom
    one_hr = timedelta(hours=1)

    for bb in var_names:
        dtff = dt0
        dhr = 3
        if bb=='_ssh': # hook to get 1hr resolution zeta
            dhr = 1
        while dtff <= dt1:
            ffn = PFM['hycom_data_dir'] + 'hy'+ bb + '_' + dstr0 + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'
            ncfiles.append(ffn)
            dtff = dtff + dhr * one_hr

    return ncfiles


def cat_hycom_to_onenc(yyyymmdd,t1str,t2str):

    t1 =  datetime.strptime(t1str,"%Y%m%d%H%M")
    t2 =  datetime.strptime(t2str,"%Y%m%d%H%M")
    times = [t1,t2]

    PFM=get_PFM_info()
    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]
    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00' # this is the forecast time

    var_names = ['_ssh','_s3z','_t3z','_u3z','_v3z']
    nc_in_names = [] # this will be the list of all of the hycom.nc files that we will cat!
    
    hycom_dir = '/scratch/PFM_Simulations/hycom_data/'

    dtff = times[0]
    t2 = times[1]

    while dtff <= t2:
        for vnm in var_names:
            ffname = 'hy'+ vnm + '_' + dstr0 + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'
            full_fn_out = hycom_dir + ffname
            nc_in_names.append(full_fn_out)
        dtff = dtff + timedelta(hours=3)

    cat_fname = PFM['lv1_forc_dir'] + '/' + 'hy_cat_' + dstr0 + '.nc'

    ds = xr.open_mfdataset(nc_in_names,combine = 'by_coords',data_vars='all',coords='all')
    enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
    Enc_dict = {vn:enc_dict for vn in ds.data_vars}
    print('writing to .nc ...')
    ds.to_netcdf(cat_fname,encoding=Enc_dict)
    ds.close()
    del ds

def cat_hycom_to_twonc_1hr(yyyymmdd,t1str,t2str):

    nc_in_names = get_hycom_nc_file_names(yyyymmdd,t1str,t2str)

    nc_1hr = []
    nc_3hr = []
    
    for fn in nc_in_names:
        vnm = fn[-41:-37]
        if vnm == '_ssh':
            nc_1hr.append(fn)
        else:
            nc_3hr.append(fn)

    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]
    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00' # this is the forecast time

    PFM=get_PFM_info()

    cat_1hr_fn = PFM['lv1_forc_dir'] + '/' + 'hy_cat_1hr_' + dstr0 + '.nc'
    cat_3hr_fn = PFM['lv1_forc_dir'] + '/' + 'hy_cat_3hr_' + dstr0 + '.nc'

    ds = xr.open_mfdataset(nc_1hr,combine = 'by_coords',data_vars='all',coords='all')
    enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
    Enc_dict = {vn:enc_dict for vn in ds.data_vars}
    print('writing to .nc ...')
    ds.to_netcdf(cat_1hr_fn,encoding=Enc_dict)
    ds.close()
    del ds

    ds = xr.open_mfdataset(nc_3hr,combine = 'by_coords',data_vars='all',coords='all')
    enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
    Enc_dict = {vn:enc_dict for vn in ds.data_vars}
    print('writing to .nc ...')
    ds.to_netcdf(cat_3hr_fn,encoding=Enc_dict)
    ds.close()
    del ds


    
def hycom_cat_to_pickles(yyyymmdd):

    PFM=get_PFM_info()

    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]    
    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
    cat_fname = PFM['lv1_forc_dir'] + '/' + 'hy_cat_' + dstr0 + '.nc'
    dss = xr.open_dataset(cat_fname)
            
    lat = dss.lat.values
    lon = dss.lon.values
    z   = dss.depth.values
    t   = dss.time.values # these are strings?
    eta = dss.surf_el.values
    temp= dss.water_temp.values
    sal = dss.salinity.values
    u   = dss.water_u.values
    v   = dss.water_v.values
    t_ref = PFM['modtime0']
    dt = (dss.time - np.datetime64(t_ref))  / np.timedelta64(1,'D') # this gets time in days from t_ref
    t_rom = dt.values
    dss.close()
    del dss
    gc.collect()

    # set up dict and fill in
    OCN = dict()
    OCN['vinfo'] = dict()

    # this is the complete list of variables that need to be in the netcdf file
    vlist = ['lon','lat','ocean_time','surf_el','water_u','water_v','temp','sal']

    for aa in vlist:
        OCN['vinfo'][aa] = dict()

    OCN['lon']=lon - 360 # make the lons negative consistent with most 
    OCN['lat']=lat

    OCN['ocean_time'] = t_rom
    OCN['ocean_time_ref'] = t_ref
    
    OCN['u'] = u
    del u
    OCN['v'] = v
    del v
    OCN['temp'] = temp
    del temp
    OCN['salt'] = sal
    del sal
    OCN['zeta'] = eta
    del eta
    OCN['depth'] = z

    print('\nmax and min raw hycom data (iz is top [0] to bottom [39]):')
    vlist = ['zeta','u','v','temp','salt']
    ulist = ['m','m/s','m/s','C','psu']
    ulist2 = dict(zip(vlist,ulist))
    print_var_max_mins(OCN,vlist,ulist2)

    # put the units in OCN...
    OCN['vinfo']['lon'] = {'long_name':'longitude',
                    'units':'degrees_east'}
    OCN['vinfo']['lat'] = {'long_name':'latitude',
                    'units':'degrees_north'}
    OCN['vinfo']['ocean_time'] = {'long_name':'time since initialization',
                    'units':'days',
                    'coordinates':'temp_time',
                    'field':'ocean_time, scalar, series'}
    OCN['vinfo']['ocean_time_ref'] = {'long_name': 'the reference date tref (initialization time)'}
    OCN['vinfo']['depth'] = {'long_name':'ocean depth',
                        'units':'m'}
    OCN['vinfo']['temp'] = {'long_name':'ocean temperature',
                    'units':'degrees C',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['salt'] = {'long_name':'ocean salinity',
                    'units':'psu',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['u'] = {'long_name':'ocean east west velocity',
                    'units':'m/s',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['v'] = {'long_name':'ocean north south velocity',
                    'units':'m/s',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                    'units':'m',
                    'coordinates':'lat,lon',
                    'time':'ocean_time'}

    gc.collect()
    fn_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocn_tmp_pckl_file']
    with open(fn_out,'wb') as fp:
        pickle.dump(OCN,fp)
        print('\nHycom OCN dict saved with pickle')

def hycom_cats_to_pickle(yyyymmdd):

    PFM=get_PFM_info()

    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]    
    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
    
    cat_1hr_fn = PFM['lv1_forc_dir'] + '/' + 'hy_cat_1hr_' + dstr0 + '.nc'
    cat_3hr_fn = PFM['lv1_forc_dir'] + '/' + 'hy_cat_3hr_' + dstr0 + '.nc'
        
    dss = xr.open_dataset(cat_3hr_fn)
            
    lat = dss.lat.values
    lon = dss.lon.values
    z   = dss.depth.values
    temp= dss.water_temp.values
    sal = dss.salinity.values
    u   = dss.water_u.values
    v   = dss.water_v.values
    t_ref = PFM['modtime0']
    dt = (dss.time - np.datetime64(t_ref))  / np.timedelta64(1,'D') # this gets time in days from t_ref
    t_rom = dt.values
    dss.close()
    del dss
    gc.collect()

    dss = xr.open_dataset(cat_1hr_fn)
    eta = dss.surf_el.values
    dt = (dss.time - np.datetime64(t_ref))  / np.timedelta64(1,'D') # this gets time in days from t_ref
    t_rom2 = dt.values

    # set up dict and fill in
    OCN = dict()
    OCN['vinfo'] = dict()

    # this is the complete list of variables that need to be in the netcdf file
    vlist = ['lon','lat','ocean_time','surf_el','water_u','water_v','temp','sal','surf_el_time']

    for aa in vlist:
        OCN['vinfo'][aa] = dict()

    OCN['lon']=lon - 360 # make the lons negative consistent with most 
    OCN['lat']=lat

    OCN['ocean_time'] = t_rom
    OCN['ocean_time_ref'] = t_ref
    OCN['zeta_time'] = t_rom2
    
    OCN['u'] = u
    del u
    OCN['v'] = v
    del v
    OCN['temp'] = temp
    del temp
    OCN['salt'] = sal
    del sal
    OCN['zeta'] = eta
    del eta
    OCN['depth'] = z

    print('\nmax and min raw hycom data (iz is top [0] to bottom [39]):')
    vlist = ['zeta','u','v','temp','salt']
    ulist = ['m','m/s','m/s','C','psu']
    ulist2 = dict(zip(vlist,ulist))
    print_var_max_mins(OCN,vlist,ulist2)

    # put the units in OCN...
    OCN['vinfo']['lon'] = {'long_name':'longitude',
                    'units':'degrees_east'}
    OCN['vinfo']['lat'] = {'long_name':'latitude',
                    'units':'degrees_north'}
    OCN['vinfo']['ocean_time'] = {'long_name':'time since initialization',
                    'units':'days',
                    'coordinates':'temp_time',
                    'field':'ocean_time, scalar, series'}
    OCN['vinfo']['zeta_time'] = {'long_name':'time since initialization for zeta',
                    'units':'days',
                    'coordinates':'zeta_time',
                    'field':'ocean_time, scalar, series'}
    OCN['vinfo']['ocean_time_ref'] = {'long_name': 'the reference date tref (initialization time)'}
    OCN['vinfo']['depth'] = {'long_name':'ocean depth',
                        'units':'m'}
    OCN['vinfo']['temp'] = {'long_name':'ocean temperature',
                    'units':'degrees C',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['salt'] = {'long_name':'ocean salinity',
                    'units':'psu',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['u'] = {'long_name':'ocean east west velocity',
                    'units':'m/s',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['v'] = {'long_name':'ocean north south velocity',
                    'units':'m/s',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                    'units':'m',
                    'coordinates':'lat,lon',
                    'time':'ocean_time'}

    gc.collect()
    fn_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocn_tmp_pckl_file']
    with open(fn_out,'wb') as fp:
        pickle.dump(OCN,fp)
        print('\nHycom OCN dict saved with pickle')

def hycom_ncfiles_to_pickle(yyyymmdd):


    # set up dict and fill in
    OCN = dict()
    OCN['vinfo'] = dict()
    # this is the complete list of variables that need to be in the netcdf file
    vlist = ['lon','lat','ocean_time','surf_el','water_u','water_v','temp','sal','surf_el_time']

    for aa in vlist:
        OCN['vinfo'][aa] = dict()


    # yyyymmdd is the start day of the hycom forecast
    PFM=get_PFM_info()
    t_ref = PFM['modtime0']
    OCN['ocean_time_ref'] = t_ref

    t1  = PFM['fetch_time']       # this is the start time of the PFM forecast
    # now a string of the time to start ROMS (and the 1st atm time too)
    t1str = "%d%02d%02d%02d%02d" % (t1.year, t1.month, t1.day, t1.hour, t1.minute)
    t2  = t1 + PFM['forecast_days'] * timedelta(days=1)  # this is the last time of the PFM forecast

    t2str = "%d%02d%02d%02d%02d" % (t2.year, t2.month, t2.day, t2.hour, t2.minute)

    nc_in_names = get_hycom_nc_file_names(yyyymmdd,t1str,t2str)

    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]    
#    dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
    
#    var3d_1 = ['t3z','s3z','u3z','v3z']
#    var3d_2 = ['water_temp','salinity','water_u','water_v']
#    var2d_1 = ['ssh']
#    var2d_2 = ['surf_el']
    
    # get lists of file names for each variable. I think they are sorted?
    fn_ssh = [s for s in nc_in_names if "_ssh_" in s]
    fn_t3z = [s for s in nc_in_names if "_t3z_" in s]
    fn_s3z = [s for s in nc_in_names if "_s3z_" in s]
    fn_u3z = [s for s in nc_in_names if "_u3z_" in s]
    fn_v3z = [s for s in nc_in_names if "_v3z_" in s]

    ntz = len(fn_ssh) # how many times for ssh
    nt  = len(fn_t3z) # how many times for 3d vars

    dss = xr.open_dataset(fn_ssh[0])
    lat = dss.lat.values
    lon = dss.lon.values
    dss.close()

    OCN['lon']=lon - 360 # make the lons negative consistent with most 
    OCN['lat']=lat

    nln = len(lon)
    nlt = len(lat)
    eta = np.zeros((ntz,nlt,nln))
    t_rom2 = np.zeros((ntz))
    cnt=0
    for fn in fn_ssh:
        dss = xr.open_dataset(fn)
        dt = (dss.time - np.datetime64(t_ref))  / np.timedelta64(1,'D') # this gets time in days from t_ref
        t_rom2[cnt] = dt.values
        eta[cnt,:,:] = dss.surf_el.values
        dss.close()
        cnt=cnt+1

    OCN['zeta_time'] = t_rom2
    OCN['zeta'] = eta
    del eta
   

    dss = xr.open_dataset(fn_t3z[0])
    z   = dss.depth.values
    dss.close()
    OCN['depth'] = z
    nz = len(z)
    temp = np.zeros((nt,nz,nlt,nln))
    t_rom = np.zeros((nt))

    cnt = 0
    for fn in fn_t3z:
        dss = xr.open_dataset(fn)
        dt = (dss.time - np.datetime64(t_ref))  / np.timedelta64(1,'D') # this gets time in days from t_ref
        t_rom[cnt] = dt.values
        temp[cnt,:,:,:] = dss.water_temp.values
        dss.close()
        cnt = cnt + 1

    OCN['ocean_time'] = t_rom
    OCN['temp'] = temp
    del temp

    cnt = 0
    sal = np.zeros((nt,nz,nlt,nln))
    for fn in fn_s3z:
        dss = xr.open_dataset(fn)
        sal[cnt,:,:,:] = dss.salinity.values
        dss.close()
        cnt = cnt + 1

    OCN['salt'] = sal
    del sal

    cnt = 0
    u = np.zeros((nt,nz,nlt,nln))
    for fn in fn_u3z:
        dss = xr.open_dataset(fn)
        u[cnt,:,:,:] = dss.water_u.values
        dss.close()
        cnt = cnt + 1

    OCN['u'] = u
    del u
    v = np.zeros((nt,nz,nlt,nln))
    cnt = 0
    for fn in fn_v3z:
        dss = xr.open_dataset(fn)
        v[cnt,:,:,:] = dss.water_v.values
        dss.close()
        cnt = cnt + 1
 
    OCN['v'] = v
    del v

    print('\nmax and min raw hycom data (iz is top [0] to bottom [39]):')
    vlist = ['zeta','u','v','temp','salt']
    ulist = ['m','m/s','m/s','C','psu']
    ulist2 = dict(zip(vlist,ulist))
    print_var_max_mins(OCN,vlist,ulist2)

    # put the units in OCN...
    OCN['vinfo']['lon'] = {'long_name':'longitude',
                    'units':'degrees_east'}
    OCN['vinfo']['lat'] = {'long_name':'latitude',
                    'units':'degrees_north'}
    OCN['vinfo']['ocean_time'] = {'long_name':'time since initialization',
                    'units':'days',
                    'coordinates':'temp_time',
                    'field':'ocean_time, scalar, series'}
    OCN['vinfo']['zeta_time'] = {'long_name':'time since initialization for zeta',
                    'units':'days',
                    'coordinates':'zeta_time',
                    'field':'ocean_time, scalar, series'}
    OCN['vinfo']['ocean_time_ref'] = {'long_name': 'the reference date tref (initialization time)'}
    OCN['vinfo']['depth'] = {'long_name':'ocean depth',
                        'units':'m'}
    OCN['vinfo']['temp'] = {'long_name':'ocean temperature',
                    'units':'degrees C',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['salt'] = {'long_name':'ocean salinity',
                    'units':'psu',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['u'] = {'long_name':'ocean east west velocity',
                    'units':'m/s',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['v'] = {'long_name':'ocean north south velocity',
                    'units':'m/s',
                    'coordinates':'z,lat,lon',
                    'time':'ocean_time'}
    OCN['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                    'units':'m',
                    'coordinates':'lat,lon',
                    'time':'ocean_time'}

    gc.collect()
    fn_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocn_tmp_pckl_file']
    with open(fn_out,'wb') as fp:
        pickle.dump(OCN,fp)
        print('\nHycom OCN dict saved with pickle')



def print_var_max_mins(OCN,vlist,ulist2):

    for vnm in vlist:
        ind_mx = np.unravel_index(np.nanargmax(OCN[vnm], axis=None), OCN[vnm].shape)
        ind_mn = np.unravel_index(np.nanargmin(OCN[vnm], axis=None), OCN[vnm].shape)
        mxx = OCN[vnm][ind_mx]
        mnn = OCN[vnm][ind_mn]

        if vnm == 'zeta':
            print(f'max {vnm:6} = {mxx:6.3f} {ulist2[vnm]:5}      at  ( it, ilat, ilon)     =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')
            print(f'min {vnm:6} = {mnn:6.3f} {ulist2[vnm]:5}      at  ( it, ilat, ilon)     =  ({ind_mn[0]:3},{ind_mn[1]:4},{ind_mn[2]:4})')
        else:
            print(f'max {vnm:6} = {mxx:6.3f} {ulist2[vnm]:5}      at  ( it, iz, ilat, ilon) =  ({ind_mx[0]:3},{ind_mx[1]:3},{ind_mx[2]:4},{ind_mx[3]:4})')
            print(f'min {vnm:6} = {mnn:6.3f} {ulist2[vnm]:5}      at  ( it, iz, ilat, ilon) =  ({ind_mn[0]:3},{ind_mn[1]:3},{ind_mn[2]:4},{ind_mn[3]:4})')
        if vnm == 'temp' or vnm == 'salt':
            if vnm == 'salt':
                vvv = 'dS/dz'
                uvv = 'psu/m'
            else:
                vvv = 'dT/dz'
                uvv = 'C/m'
            dz = OCN['depth'][1:] - OCN['depth'][0:-1]
            dT = OCN[vnm][:,0:-1,:,:] - OCN[vnm][:,1:,:,:]
            dTdz = dT / dz[None,:,None,None]
            ind_mx = np.unravel_index(np.nanargmax(dTdz, axis=None), dTdz.shape)
            ind_mn = np.unravel_index(np.nanargmin(dTdz, axis=None), dTdz.shape)
            mxx = dTdz[ind_mx]
            mnn = dTdz[ind_mn]
            print(f'max {vvv:6} = {mxx:6.3f} {uvv:5}      at  ( it, iz, ilat, ilon) =  ({ind_mx[0]:3},{ind_mx[1]:3},{ind_mx[2]:4},{ind_mx[3]:4})')
            print(f'min {vvv:6} = {mnn:6.3f} {uvv:5}      at  ( it, iz, ilat, ilon) =  ({ind_mn[0]:3},{ind_mn[1]:3},{ind_mn[2]:4},{ind_mn[3]:4})')


def get_ocn_data_as_dict_pckl(yyyymmdd,run_type,ocn_mod,get_method):
#    import pygrib
    PFM=get_PFM_info()
    
    yyyy = yyyymmdd[0:4]
    mm = yyyymmdd[4:6]
    dd = yyyymmdd[6:8]

    lt_min = PFM['latlonbox']['L1'][0]
    lt_max = PFM['latlonbox']['L1'][1]
    ln_min = PFM['latlonbox']['L1'][2]+360.0
    ln_max = PFM['latlonbox']['L1'][3]+360.0
    # this function will returm OCN, a dict of all ocean fields ROMS requires
    # keys will be the ROMS .nc variable names.
    # they will be on the ocn grid (not roms grid)
    # the forecast data will start at yyyymmdd and 1200z. All times of the forecast will
    # be returned. ocn_mod is the type of OCN models used, one of:
    # 'hycom', 'wcofs', etc
    # get_method, is the type of method used, either 'open_dap' or 'grib_download'

    # the code in here goes from the start date to all forecast dates

    def get_roms_times_from_hycom(fore_date,t,t_ref):
        # this funtion returns times past t_ref in days
        # consistent with how ROMS likes it

        d1=fore_date
        t2 = t # an ndarray of days, t is from atm import
        t3 = d1 + t2 * timedelta(days=1/24)
        print('ocn forecast data is being grabbed for:')
        print(t3)
        # t3 looks good and is the correct time stamps of the forecast.
        # But for ROMS we need ocean_time which is relative to 1970,1,1. 
        # in seconds. So...
        tr = t3 - t_ref
        tr_days = tr.astype("timedelta64[ms]").astype(float) / 1000 / 3600 / 24
        # tr_sec is now an ndarray of days past the reference day
        return tr_days


    if get_method == 'open_dap_pydap' or get_method == 'open_dap_nc' or get_method == 'ncks' or get_method == 'ncks_para' and run_type == 'forecast':
        # with this method the data is not downloaded directly, initially
        # and the data is rectilinear, lon and lat are both vectors

        # the hycom forecast is 7.5 days long and 
        # is at 3 hr resolution, we will get all of the data
        # 0.08 deg horizontal resolution
        # Note, hycom forecasts are start from 1200Z !!!! 
        

        hycom = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/runs/GLBy0.08_930_FMRC_RUN_' + yyyy + '-' + mm + '-' + dd + 'T12:00:00Z'
        #        https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0.html
        
        #        https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_ssh/runs/FMRC_ESPC-D-V02_ssh_RUN_2024-09-11T12:00:00Z.html
        #        https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_s3z/runs/FMRC_ESPC-D-V02_s3z_RUN_2024-09-12T12:00:00Z.html
        #        https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_t3z/runs/FMRC_ESPC-D-V02_t3z_RUN_2024-09-12T12:00:00Z.html
        #        https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_u3z/runs/FMRC_ESPC-D-V02_u3z_RUN_2024-09-12T12:00:00Z.html
        #        https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_v3z/runs/FMRC_ESPC-D-V02_v3z_RUN_2024-09-12T12:00:00Z.html
        
        # define the box to get data in, hycom uses 0-360 longitude.
        #ln_min = -124.5 + 360
        #ln_max = -115 + 360
        #lt_min = 28
        #lt_max = 37


        if ocn_mod == 'hycom':
            ocn_name = hycom
            it1 = 0 # hard wiring here to get 2.5 days of data
            it2 = 20 # this is 2.5 days at 3 hrs
            # it2 = 60 # for the entire 7.5 day long forecast
        if ocn_mod == 'hycom_new': # THIS ONLY WORKS with ncks_para method
            ocn_name = ['https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02','runs/FMRC_ESPC-D-V02','_RUN_'+ yyyy + '-' + mm + '-' + dd + 'T12:00:00Z']
            var_names = ['_ssh','_s3z','_t3z','_u3z','_v3z']
            #print('in here a')
            #print(ocn_name)
    
        if get_method == 'open_dap_pydap': # DEPRECATED METHOD W ADDITION OF hycom_new
            # open a connection to the opendap server. This could be made more robust? 
            # by trying repeatedly?
            # open_url is sometimes slow, and this block of code can be fast (1s), med (6s), or slow (>15s)
            dataset = open_url(ocn_name)

            times = dataset['time']         # ???
            ln   = dataset['lon']          # deg
            lt   = dataset['lat']          # deg
            Eta  = dataset['surface_el']     # surface elevations, m
            Temp = dataset['water_temp']       # at surface, C
            U    = dataset['water_u']      # water E/W velocity, m/s
            V    = dataset['water_v']      # water N/S velocity, m/s
            Sal  = dataset['salinity']     # upward long-wave at surface, W/m2
            Z    = dataset['depth']

            # here we find the indices of the data we want for LV1
            Ln0    = ln[:].data
            Lt0    = lt[:].data
            iln    = np.where( (Ln0>=ln_min)*(Ln0<=ln_max) ) # the lon indices where we want data
            ilt    = np.where( (Lt0>=lt_min)*(Lt0<=lt_max) ) # the lat indices where we want data

            # return the roms times past tref in days
            t=times.data[it1:it2] # this is hrs past t0
            t0 = times.attributes['units'] # this the hycom reference time
            t0 = t0[12:31] # now we get just the date and 12:00
            t0 = datetime.fromisoformat(t0) # put it in datetime format
            #t_ref = datetime(1999,1,1)
            t_ref = PFM['modtime0']
            t_rom = get_roms_times_from_hycom(t0,t,t_ref)

            lon   = ln[iln[0][0]:iln[0][-1]].data
            lat   = lt[ilt[0][0]:ilt[0][-1]].data
            z     = z[:].data

            # we will get the other data directly
            eta  =  Eta.array[it1:it2,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            u    =    U.array[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            v    =    V.array[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            temp = Temp.array[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            sal  =  Sal.array[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data

        
        if get_method == 'open_dap_nc': # DEPRECATED METHOD W ADDITION OF hycom_new
            # open a connection to the opendap server. This could be made more robust? 
            # by trying repeatedly?
            # open_url is sometimes slow, and this block of code can be fast (1s), med (6s), or slow (>15s)
            
            dataset = nc.Dataset(ocn_name)
            #ds2 = xr.open_dataset(ocn_name,use_cftime=False, decode_coords=False, decode_times=False)


            times = dataset['time']         # ???
            ln   = dataset['lon']          # deg
            lt   = dataset['lat']          # deg
            Eta  = dataset['surf_el']     # surface elevations, m
            Temp = dataset['water_temp']       # 3d temp, C
            U    = dataset['water_u']      # water E/W velocity, m/s
            V    = dataset['water_v']      # water N/S velocity, m/s
            Sal  = dataset['salinity']     # salinity psu
            Z    = dataset['depth']

            # here we find the indices of the data we want for LV1
            Ln0    = ln[:]
            Lt0    = lt[:]
            iln    = np.where( (Ln0>=ln_min)*(Ln0<=ln_max) ) # the lon indices where we want data
            ilt    = np.where( (Lt0>=lt_min)*(Lt0<=lt_max) ) # the lat indices where we want data

            lat = lt[ ilt[0][0]:ilt[0][-1] ].data
            lon = ln[ iln[0][0]:iln[0][-1] ].data
            z   =  Z[:].data
            t   = times[it1:it2].data
            eta = Eta[it1:it2,ilt[0][0]:ilt[0][-1] , iln[0][0]:iln[0][-1] ].data
            # we will get the other data directly
            temp = Temp[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            sal  = Sal[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            u = U[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            v = V[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
                # return the roms times past tref in days
            t0 = times.units # this is the hycom reference time
            t0 = t0[12:31] # now we get just the date and 12:00
            t0 = datetime.fromisoformat(t0) # put it in datetime format
            t_ref = PFM['modtime0']
            t_rom = get_roms_times_from_hycom(t0,t,t_ref)

            # I think everything is an np.ndarray ?
            dataset.close()

        if get_method == 'ncks': # DEPRECATED METHOD W ADDITION OF hycom_new

            west =  ln_min
            east =  ln_max
            south = lt_min
            north = lt_max

            # time limits
            Tfor = 1 # the forecast length in days
            # hard wired here. read in from PFM !!!!
            dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
            t00 = datetime.strptime(dstr0,"%Y-%m-%dT%H:%M")
            t10 = t00 + Tfor * timedelta(days=1)
            dstr1 = t10.strftime("%Y-%m-%dT%H:%M")

            #dstr0 = dlist['dt0'].strftime('%Y-%m-%dT00:00') 
            #dstr1 = dlist['dt1'].strftime('%Y-%m-%dT00:00')
            # use subprocess.call() to execute the ncks command
            vstr = 'surf_el,water_temp,salinity,water_u,water_v,depth'

            url3 = hycom

            full_fn_out = PFM['lv1_forc_dir'] +'/' + PFM['lv1_nck_temp_file'] 

            cmd_list = ['ncks',
                '-d', 'time,'+dstr0+','+dstr1,
                '-d', 'lon,'+str(west)+','+str(east),
                '-d', 'lat,'+str(south)+','+str(north),
                '-v', vstr,
                url3 ,
                        '-4', '-O', full_fn_out]
    
            # run ncks
            tt0 = time.time()
            ret1 = subprocess.call(cmd_list)
            print('Time to get full file using ncks = %0.2f sec' % (time.time()-tt0))
            print('Return code = ' + str(ret1) + ' (0=success, 1=skipped ncks)')

            # now get the variables...
            ds = xr.open_dataset(full_fn_out)
            lat = ds.lat.values
            lon = ds.lon.values
            z   = ds.depth.values
            t   = ds.time.values # these are strings?
            eta = ds.surf_el.values
            temp= ds.water_temp.values
            sal = ds.salinity.values
            u   = ds.water_u.values
            v   = ds.water_v.values
            t_ref = PFM['modtime0']
            dt = (ds.time - np.datetime64(t_ref))  / np.timedelta64(1,'D') # this gets time in days from t_ref
            t_rom = dt.values
            ds.close()

        if get_method == 'ncks_para' and ocn_mod == 'hycom':

            print('in the parallel ncks switch')

            west =  ln_min
            east =  ln_max
            south = lt_min
            north = lt_max
            aa = [north,south,west,east]

            # time limits
            Tfor = 2.5 # the forecast length in days
            # hard wired here. read in from PFM !!!!
            # the first time to get
            dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
            t00 = datetime.strptime(dstr0,"%Y-%m-%dT%H:%M")
            # the last time to get
            t10 = t00 + Tfor * timedelta(days=1)
            dstr1 = t10.strftime("%Y-%m-%dT%H:%M")

            #dstr0 = dlist['dt0'].strftime('%Y-%m-%dT00:00') 
            #dstr1 = dlist['dt1'].strftime('%Y-%m-%dT00:00')
            # use subprocess.call() to execute the ncks command

            # form list of days to get, datetimes
            dt0 = t00
            dt1 = t10 
            dt_list_full = []
            dtff = dt0
            ncfiles = [] # this is the list with paths of all of the .nc files made with ncks_para
            #timestamps
            while dtff <= dt1:
                dt_list_full.append(dtff)
                ffn = PFM['lv1_forc_dir'] + '/hy_' + dstr0 + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'
                ncfiles.append(ffn)
                dtff = dtff + timedelta(hours=3)
                
            #this is where we para
            tt0 = time.time()

            # create parallel executor
            with ThreadPoolExecutor() as executor:
                threads = []
                for dtff in dt_list_full:
                    fn = para_loop #define function
                    args = [hycom,dtff,aa,PFM,dstr0] #define args to function
                    kwargs = {} #
                    # start thread by submitting it to the executor
                    threads.append(executor.submit(fn, *args, **kwargs))
                for future in as_completed(threads):
                        # retrieve the result
                        result = future.result()
                        # report the result
            
            #result = tt0
            print('Time to get full file using parallel ncks = %0.2f sec' % (time.time()-tt0))
            print('Return code = ' + str(result) + ' (0=success, 1=skipped ncks)')

            cat_fname = PFM['lv1_forc_dir'] + '/' + 'hy_cat_' + dstr0 + '.nc'

            ds = xr.open_mfdataset(ncfiles,combine = 'by_coords',data_vars='minimal')

            enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
            Enc_dict = {vn:enc_dict for vn in ds.data_vars}
            ds.to_netcdf(cat_fname,encoding=Enc_dict)
            ds.close()
            del ds
            # ncrcat didn't work
            # cmd_list = 'ncrcat ' + full_fns_out + ' ' + cat_fname
            # ret2 = subprocess.call(cmd_list)
            # print('files were catted:')
            # print('code = ' + str(ret2) + ' (0=success, other=failed)')

            # now get the variables...
            # use a dummy .nc file for testing...
            dss = xr.open_dataset(cat_fname)
            lat = dss.lat.values
            lon = dss.lon.values
            z   = dss.depth.values
            t   = dss.time.values # these are strings?
            eta = dss.surf_el.values
            temp= dss.water_temp.values
            sal = dss.salinity.values
            u   = dss.water_u.values
            v   = dss.water_v.values
            t_ref = PFM['modtime0']
            dt = (dss.time - np.datetime64(t_ref))  / np.timedelta64(1,'D') # this gets time in days from t_ref
            t_rom = dt.values
            dss.close()
            del dss
            gc.collect()


        if get_method == 'ncks_para' and ocn_mod == 'hycom_new':

            print('in the parallel ncks switch using new hycom data')

            west =  ln_min
            east =  ln_max
            south = lt_min
            north = lt_max
            aa = [north,south,west,east]

            # time limits
            Tfor = 2.5 # the forecast length in days
            # hard wired here. read in from PFM !!!!
            # the first time to get
            dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
            t00 = datetime.strptime(dstr0,"%Y-%m-%dT%H:%M")
            # the last time to get
            t10 = t00 + Tfor * timedelta(days=1)
            dstr1 = t10.strftime("%Y-%m-%dT%H:%M")

            #dstr0 = dlist['dt0'].strftime('%Y-%m-%dT00:00') 
            #dstr1 = dlist['dt1'].strftime('%Y-%m-%dT00:00')
            # use subprocess.call() to execute the ncks command

            # form list of days to get, datetimes
            dt0 = t00
            dt1 = t10 
            dt_list_full = []
            dtff = dt0
            ncfiles = [] # this is the list with paths of all of the .nc files made with ncks_para
            #timestamps
            while dtff <= dt1:
                for bb in var_names:
                    ffn = PFM['lv1_forc_dir'] + '/hy'+ bb + '_' + dstr0 + '_' + dtff.strftime("%Y-%m-%dT%H:%M") +'.nc'
                    ncfiles.append(ffn)
                
                dt_list_full.append(dtff) # these are the times to get forecast for...
                dtff = dtff + timedelta(hours=3)

                
            #this is where we para
            tt0 = time.time()

            # create parallel executor
            with ThreadPoolExecutor() as executor:
                threads = []
                for bb in var_names:
                    for dtff in dt_list_full:
                        fn = para_loop_new #define function
                        hycom = ocn_name[0]+bb+ocn_name[1]+bb+ocn_name[2]
                        args = [hycom,dtff,aa,bb,PFM,dstr0] #define args to function
                        kwargs = {} #
                        # start thread by submitting it to the executor
                        threads.append(executor.submit(fn, *args, **kwargs))
                    
                for future in as_completed(threads):
                    # retrieve the result
                    result = future.result()
                    # report the result
            
            #result = tt0
            print('Time to get all files using parallel ncks = %0.2f sec' % (time.time()-tt0))
            print('Return code = ' + str(result) + ' (0=success, 1=skipped ncks)')

            cat_fname = PFM['lv1_forc_dir'] + '/' + 'hy_cat_' + dstr0 + '.nc'

            #print(ncfiles)
            ds = xr.open_mfdataset(ncfiles,combine = 'by_coords',data_vars='all',coords='all')

            #print(ds.data_vars)

            enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
            Enc_dict = {vn:enc_dict for vn in ds.data_vars}
            ds.to_netcdf(cat_fname,encoding=Enc_dict)
            ds.close()
            del ds
            # ncrcat didn't work
            # cmd_list = 'ncrcat ' + full_fns_out + ' ' + cat_fname
            # ret2 = subprocess.call(cmd_list)
            # print('files were catted:')
            # print('code = ' + str(ret2) + ' (0=success, other=failed)')

            # now get the variables...
            # use a dummy .nc file for testing...
            dss = xr.open_dataset(cat_fname)
            lat = dss.lat.values
            lon = dss.lon.values
            z   = dss.depth.values
            t   = dss.time.values # these are strings?
            eta = dss.surf_el.values
            temp= dss.water_temp.values
            sal = dss.salinity.values
            u   = dss.water_u.values
            v   = dss.water_v.values
            t_ref = PFM['modtime0']
            dt = (dss.time - np.datetime64(t_ref))  / np.timedelta64(1,'D') # this gets time in days from t_ref
            t_rom = dt.values
            dss.close()
            del dss
            gc.collect()





        # set up dict and fill in
        OCN = dict()
        OCN['vinfo'] = dict()

        # this is the complete list of variables that need to be in the netcdf file
        vlist = ['lon','lat','ocean_time','surf_el','water_u','water_v','temp','sal']

        for aa in vlist:
            OCN['vinfo'][aa] = dict()

        OCN['lon']=lon - 360 # make the lons negative consistent with most 
        OCN['lat']=lat

        OCN['ocean_time'] = t_rom
        OCN['ocean_time_ref'] = t_ref
        
        OCN['u'] = u
        del u
        OCN['v'] = v
        del v
        OCN['temp'] = temp
        del temp
        OCN['salt'] = sal
        del sal
        OCN['zeta'] = eta
        del eta
        OCN['depth'] = z

        print('\nmax and min raw hycom data (iz is top [0] to bottom [39]):')
        vlist = ['zeta','u','v','temp','salt']
        ulist = ['m','m/s','m/s','C','psu']
        ulist2 = dict(zip(vlist,ulist))
        print_var_max_mins(OCN,vlist,ulist2)

        # put the units in OCN...
        OCN['vinfo']['lon'] = {'long_name':'longitude',
                        'units':'degrees_east'}
        OCN['vinfo']['lat'] = {'long_name':'latitude',
                        'units':'degrees_north'}
        OCN['vinfo']['ocean_time'] = {'long_name':'time since initialization',
                        'units':'days',
                        'coordinates':'temp_time',
                        'field':'ocean_time, scalar, series'}
        OCN['vinfo']['ocean_time_ref'] = {'long_name': 'the reference date tref (initialization time)'}
        OCN['vinfo']['depth'] = {'long_name':'ocean depth',
                            'units':'m'}
        OCN['vinfo']['temp'] = {'long_name':'ocean temperature',
                        'units':'degrees C',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['u'] = {'long_name':'ocean east west velocity',
                        'units':'m/s',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['v'] = {'long_name':'ocean north south velocity',
                        'units':'m/s',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'lat,lon',
                        'time':'ocean_time'}
    
    gc.collect()
    fn_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocn_tmp_pckl_file']
    with open(fn_out,'wb') as fp:
        pickle.dump(OCN,fp)
        print('\nHycom OCN dict saved with pickle')


def earth_rad(lat_deg):
    """
    Calculate the Earth radius (m) at a latitude
    (from http://en.wikipedia.org/wiki/Earth_radius) for oblate spheroid

    INPUT: latitude in degrees

    OUTPUT: Earth radius (m) at that latitute
    """
    a = 6378.137 * 1000; # equatorial radius (m)
    b = 6356.7523 * 1000; # polar radius (m)
    cl = np.cos(np.pi*lat_deg/180)
    sl = np.sin(np.pi*lat_deg/180)
    RE = np.sqrt(((a*a*cl)**2 + (b*b*sl)**2) / ((a*cl)**2 + (b*sl)**2))
    return RE

def ll2xy(lon, lat, lon0, lat0):
    """
    This converts lon, lat into meters relative to lon0, lat0.
    It should work for lon, lat scalars or arrays.
    NOTE: lat and lon are in degrees!!
    """
    R = earth_rad(lat0)
    clat = np.cos(np.pi*lat0/180)
    x = R * clat * np.pi * (lon - lon0) / 180
    y = R * np.pi * (lat - lat0) / 180
    return x, y


def extrap_nearest_to_masked(X, Y, fld, fld0=0):
    """
    INPUT: fld is a 2D array (np.ndarray or np.ma.MaskedArray) on spatial grid X, Y
    OUTPUT: a numpy array of the same size with no mask
    and no missing values.        
    If input is a masked array:        
        * If it is ALL masked then return an array filled with fld0.         
        * If it is PARTLY masked use nearest neighbor interpolation to
        fill missing values, and then return data.        
        * If it is all unmasked then return the data.    
    If input is not a masked array:        
        * Return the array.    
    """
    
    # first make sure nans are masked
    if np.ma.is_masked(fld) == False:
        fld = np.ma.masked_where(np.isnan(fld), fld)
        
    if fld.all() is np.ma.masked:
        #print('  filling with ' + str(fld0))
        fldf = fld0 * np.ones(fld.data.shape)
        fldd = fldf.data
        checknan(fldd)
        return fldd
    else:
        # do the extrapolation using nearest neighbor
        fldf = fld.copy() # initialize the "filled" field
        xyorig = np.array((X[~fld.mask],Y[~fld.mask])).T
        xynew = np.array((X[fld.mask],Y[fld.mask])).T
        a = cKDTree(xyorig).query(xynew)
        aa = a[1]
        fldf[fld.mask] = fld[~fld.mask][aa]
        fldd = fldf.data
        checknan(fldd)
        return fldd

def checknan(fld):
    """
    A utility function that issues a working if there are nans in fld.
    """
    if np.isnan(fld).sum() > 0:
        print('WARNING: nans in data field')    


def check_all_nans(z):
    # this assumes z is NOT a masked array...

    nfin = np.count_nonzero(~np.isnan(z))
    if nfin == 0:
        allnan = True
    else:
        allnan = False

    return allnan


def interp_hycom_to_roms(ln_h,lt_h,zz_h,Ln_r,Lt_r,msk_r,Fz):
    # ln_h and lt_h are hycom lon and lat, vectors
    # zz_h is the hycom field that is getting interpolated
    # Ln_r and Lt_r are the roms lon and lat, matrices
    # msk_r is the roms mask, to NaN values
    # mf is the refinement of the hycom  grid before going 
    # nearest neighbor interpolating to ROMS

    # 1st check if field is all NaNs
    allnan = check_all_nans(zz_h)

    if allnan == True:
        # if all NaNs, just return all NaNs, no need to interpolate
        zz_r = np.nan * Ln_r
    else:
        Ln_h, Lt_h = np.meshgrid(ln_h, lt_h, indexing='xy')
        X, Y = ll2xy(Ln_h, Lt_h, ln_h.mean(), lt_h.mean())
        # fill in the hycom NaNs with nearest neighbor values
        # ie. land is now filled with numbers
        zz_hf =  extrap_nearest_to_masked(X, Y, zz_h) 

        # interpolate the filled hycom values to the refined grid
        #Fz = RegularGridInterpolator((lt_h, ln_h), zz_hf)
        # change the z values of the interpolator here
        setattr(Fz,'values',zz_hf)

        lininterp=1
        if lininterp==1:
            zz_rf = Fz((Lt_r,Ln_r),method='linear')
        else:
            # refine the hycom grid
            #lnhi = np.linspace(ln_h[0],ln_h[-1],mf*len(ln_h))
            #lthi = np.linspace(lt_h[0],lt_h[-1],mf*len(lt_h))
            #Lnhi, Lthi = np.meshgrid(lnhi,lthi, indexing='xy')
            #zz_hfi=Fz((Lthi,Lnhi))

            # now nearest neighbor interpolate to the ROMS grid
            #XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
            #XYr = np.array((Ln_r.flatten(), Lt_r.flatten())).T
            # nearest neighbor interpolation from XYin to XYr is done below...
            #IMr = cKDTree(XYin).query(XYr)[1]    
            #zz_rf = zz_hf.flatten()[IMr].reshape(Ln_r.shape)
            print('somethin went wrong!') 

        # now mask zz_r
        zz_r = zz_rf.copy()
        zz_r[msk_r==0] = np.nan

    return zz_r


def hycom_to_roms_latlon(HY,RMG):
    # HYcom and RoMsGrid come in as dicts with ROMS variable names    
    # The output of this, HYrm, is a dict with 
    # hycom fields on roms horizontal grid points
    # but hycom z levels.
    # velocity will be on both (lat_u,lon_u)
    # and (lat_v,lon_v).
    
    # set up the interpolator now and pass to function

    Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])

    # the names of the variables that need to go on the ROMS grid
    vnames = ['zeta', 'temp', 'salt', 'u', 'v']
    lnhy = HY['lon']
    lthy = HY['lat']
    NR,NC = np.shape(RMG['lon_rho'])
    NZ = len(HY['depth'])
    NT = len(HY['ocean_time'])

    #print('before HYrm setup')
    HYrm = dict()
    Tmpu = dict()
    Tmpv = dict()
    HYrm['zeta'] = np.zeros((NT,NR, NC))
    HYrm['salt'] = np.zeros((NT,NZ, NR, NC))
    HYrm['temp'] = np.zeros((NT,NZ, NR, NC))
    Tmpu['u_on_u'] = np.zeros((NT,NZ, NR, NC-1))
    Tmpu['v_on_u'] = np.zeros((NT,NZ, NR, NC-1))
    Tmpv['u_on_v'] = np.zeros((NT,NZ, NR-1, NC))
    Tmpv['v_on_v'] = np.zeros((NT,NZ, NR-1, NC))
    HYrm['ubar']   = np.zeros((NT,NR,NC-1))
    HYrm['vbar']   = np.zeros((NT,NR-1,NC))
    HYrm['lat_rho'] = RMG['lat_rho'][:]
    HYrm['lon_rho'] = RMG['lon_rho'][:]
    HYrm['lat_u'] = RMG['lat_u'][:]
    HYrm['lon_u'] = RMG['lon_u'][:]
    HYrm['lat_v'] = RMG['lat_v'][:]
    HYrm['lon_v'] = RMG['lon_v'][:]
    HYrm['depth'] = HY['depth'][:] # depths are from hycom
    HYrm['ocean_time'] = HY['ocean_time'][:]
    HYrm['ocean_time_ref'] = HY['ocean_time_ref']

    HYrm['vinfo']=dict()
    HYrm['vinfo']['depth'] = HY['vinfo']['depth']
    HYrm['vinfo']['ocean_time'] = HY['vinfo']['ocean_time']
    HYrm['vinfo']['ocean_time_ref'] = HY['vinfo']['ocean_time_ref']
    HYrm['vinfo']['lon_rho'] = {'long_name':'rho point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_rho'] = {'long_name':'rho point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['lon_u'] = {'long_name':'rho point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_u'] = {'long_name': 'u point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['lon_v'] = {'long_name':'v point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_v'] = {'long_name':'v point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['temp'] = {'long_name':'ocean temperature',
                        'units':'degrees C',
                        'coordinates':'z,lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'z,lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['urm'] = {'long_name':'ocean xi velocity',
                        'units':'m/s',
                        'coordinates':'z,lat_u,lon_u',
                        'time':'ocean_time',
                        'note':'this is now rotated in the roms xi direction'}
    HYrm['vinfo']['vrm'] = {'long_name':'ocean eta velocity',
                        'units':'m/s',
                        'coordinates':'z,lat_v,lon_v',
                        'time':'ocean_time',
                        'note':'this is now rotated in the roms eta direction'}
    HYrm['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                        'units':'m/s',
                        'coordinates':'lat_u,lon_u',
                        'time':'ocean_time',
                        'note':'uses hycom depths and this is now rotated in the roms xi direction'}
    HYrm['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                        'units':'m/s',
                        'coordinates':'lat_v,lon_v',
                        'time':'ocean_time',
                        'note':'uses hycom depths and this is now rotated in the roms eta direction'}

    #print('after HYrm setup. before Tmp')

    print('before interp to roms grid, using:')
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('kilobytes')


    for aa in vnames:
        zhy  = HY[aa]
        #print('doing:')
        #print(aa)
        for cc in range(NT):
            if aa=='zeta':
                zhy2 = zhy[cc,:,:]
                HYrm[aa][cc,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            
            elif aa=='temp' or aa=='salt':
                for bb in range(NZ):
                    zhy2 = zhy[cc,bb,:,:]
                    HYrm[aa][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)
            elif aa=='u':
                for bb in range(NZ):
                    zhy2= zhy[cc,bb,:,:]
                    #u_on_u = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],Fz)
                    #u_on_v = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],Fz)
                    Tmpu['u_on_u'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],Fz)
                    Tmpv['u_on_v'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],Fz)
            elif aa=='v':
                for bb in range(NZ):
                    zhy2= zhy[cc,bb,:,:]
                    Tmpu['v_on_u'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],Fz)
                    Tmpv['v_on_v'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],Fz)

    #print('after Tmp. before rotation')
    del HY

    gc.collect()
    # rotate the velocities so that the velocities are in roms eta,xi coordinates
    angr = RMG['angle_u']
    cosang = np.cos(angr)
    sinang = np.sin(angr)
    #Cosang = np.tile(cosang,(NT,NZ,1,1))
    #Sinang = np.tile(sinang,(NT,NZ,1,1))
    #urm = Cosang * Tmp['u_on_u'] + Sinang * Tmp['v_on_u']

    print('before rotating urm, using:')
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('kilobytes')

    #print('before giant multiplication')
    urm = cosang[None,None,:,:] * Tmpu['u_on_u'][:,:,:,:] + sinang[None,None,:,:] * Tmpu['v_on_u'][:,:,:,:]
    #print('just after multiplication. before naning')
    urm[np.isnan(Tmpu['u_on_u'])==1] = np.nan
    #print('after nanning. before filling HYrm')
    HYrm['urm'] = urm
    
    del Tmpu
    #print('after filling HYrm with u rotation. before v rotation')

    angr = RMG['angle_v']
    cosang = np.cos(angr)
    sinang = np.sin(angr)
    #Cosang = np.tile(cosang,(NT,NZ,1,1))
    #Sinang = np.tile(sinang,(NT,NZ,1,1))
    #vrm = Cosang * HYrm['v_on_v'] - Sinang * HYrm['u_on_v']
    print('before rotating vrm, using:')
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('kilobytes')


    vrm = cosang[None,None,:,:] * Tmpv['v_on_v'][:,:,:,:] - sinang[None,None,:,:] * Tmpv['u_on_v'][:,:,:,:]
    vrm[np.isnan(Tmpv['u_on_v'])==1] = np.nan
    HYrm['vrm'] = vrm
    del Tmpv
        

    # we need the roms depths on roms u and v grids
    Hru = 0.5 * (RMG['h'][:,0:-1] + RMG['h'][:,1:])
    Hrv = 0.5 * (RMG['h'][0:-1,:] + RMG['h'][1:,:])
    # get the locations in z of the hycom output
    hyz = HYrm['depth'].copy()
    #print(np.shape(Hru))
    #print(np.shape(HYrm['urm']))

    # do ubar, the depth average velocity in roms (eta,xi) coordinates
    # so ubar and vbar are calculated from hycom depths before interpolating to roms depths

    print('after v rotation. before ubar')


    use_for = 0 
    if use_for == 1: # we will not use this to get ubar. could be removed 
        # for looping this is VERY slow. Need to make faster
        for aa in range(NR): # lat loop
            for bb in range(NC-1): # lon loop
                # get indices of non nan u based depths
                # this is where the hycom depths are less than the roms depth 
                igu = np.argwhere(hyz <= Hru[aa,bb])
                for cc in range(NT): # time loop
                    uonhyz = HYrm['urm'][cc,:,aa,bb]
                    ug = uonhyz[igu]
                    hgu = hyz[igu]
                    HYrm['ubar'][cc,aa,bb] = ( np.sum( 0.5 * (ug[0:-1]+ug[1:])*(hgu[1:]-hgu[0:-1])) + ug[-1]*(Hru[aa,bb]-hgu[-1]) ) / Hru[aa,bb]
        # do vbar
        for aa in range(NR-1): # lat loop
            for bb in range(NC): # lon loop
                # get indices of non nan v based depths 
                igv = np.argwhere(hyz <= Hrv[aa,bb])
                for cc in range(NT): # time loop
                    vonhyz = HYrm['vrm'][cc,:,aa,bb]
                    vg = vonhyz[igv]
                    hgv = hyz[igv]
                    HYrm['vbar'][cc,aa,bb] = ( np.sum( 0.5 * (vg[0:-1]+vg[1:])*(hgv[1:]-hgv[0:-1])) + vg[-1]*(Hrv[aa,bb]-hgv[-1]) ) / Hrv[aa,bb]
    else:
        # set up mask

        # use python "broadcasting" to mask velocity
        # and put zeros in the right place
        utst = Hru[None,:,:] - hyz[:,None,None]
        vtst = Hrv[None,:,:] - hyz[:,None,None]
    
        #dz = hyz[1:]-hyz[0:-1]

        umsk = 0*utst
        vmsk = 0*vtst
        umsk[utst>=0] = 1 # this should put zeros at all depths below the bottom
        vmsk[vtst>=0] = 1 # and ones at all depths above the bottom

        # put zeros at hycom depths below the bottom
        HYrm['urm'] = HYrm['urm']*umsk[None,:,:,:]
        HYrm['vrm'] = HYrm['vrm']*vmsk[None,:,:,:]

        # make copies to make ubar
#        uu = HYrm['urm'].copy()
#        vv = HYrm['vrm'].copy()

#        uu[np.isnan(uu)==1]=0
#        vv[np.isnan(vv)==1]=0

#        ubar = np.squeeze( np.sum( 0.5 * (uu[:,0:-1,:,:]+uu[:,1:,:,:]) * dz[None,:,None,None], axis=1 ) ) / Hru[None,:,:]
#        vbar = np.squeeze( np.sum( 0.5 * (vv[:,0:-1,:,:]+vv[:,1:,:,:]) * dz[None,:,None,None], axis=1 ) ) / Hrv[None,:,:]

        # reapply the land mask...
#        umsk2 = np.squeeze( np.isnan( urm[:,0,:,:]))
#        vmsk2 = np.squeeze( np.isnan( vrm[:,0,:,:]))
#        ubar[umsk2==1] = np.nan
#        vbar[vmsk2==1] = np.nan

        # this ubar isn't used but is returned for reference
#        HYrm['ubar'] = ubar
#        HYrm['vbar'] = vbar
         
    return HYrm


def hycom_to_roms_latlon_pckl(fname_in):
    # HYcom and RoMsGrid come in as dicts with ROMS variable names    
    # The output of this, HYrm, is a dict with 
    # hycom fields on roms horizontal grid points
    # but hycom z levels.
    # velocity will be on both (lat_u,lon_u)
    # and (lat_v,lon_v).

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])


    with open(fname_in,'rb') as fp:
        HY = pickle.load(fp)
        #print('OCN dict loaded with pickle')

    
    # set up the interpolator now and pass to function
    Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])

    # the names of the variables that need to go on the ROMS grid
    vnames = ['zeta', 'temp', 'salt', 'u', 'v']
    lnhy = HY['lon']
    lthy = HY['lat']
    NR,NC = np.shape(RMG['lon_rho'])
    NZ = len(HY['depth'])
    NT = len(HY['ocean_time'])

    HYrm = dict()

    HYrm['zeta'] = np.zeros((NT,NR, NC))
    HYrm['salt'] = np.zeros((NT,NZ, NR, NC))
    HYrm['temp'] = np.zeros((NT,NZ, NR, NC))
    HYrm['ubar']   = np.zeros((NT,NR,NC-1))
    HYrm['vbar']   = np.zeros((NT,NR-1,NC))
    HYrm['lat_rho'] = RMG['lat_rho'][:]
    HYrm['lon_rho'] = RMG['lon_rho'][:]
    HYrm['lat_u'] = RMG['lat_u'][:]
    HYrm['lon_u'] = RMG['lon_u'][:]
    HYrm['lat_v'] = RMG['lat_v'][:]
    HYrm['lon_v'] = RMG['lon_v'][:]
    HYrm['depth'] = HY['depth'][:] # depths are from hycom
    HYrm['ocean_time'] = HY['ocean_time'][:]
    HYrm['ocean_time_ref'] = HY['ocean_time_ref']

    HYrm['vinfo']=dict()
    HYrm['vinfo']['depth'] = HY['vinfo']['depth']
    HYrm['vinfo']['ocean_time'] = HY['vinfo']['ocean_time']
    HYrm['vinfo']['ocean_time_ref'] = HY['vinfo']['ocean_time_ref']
    HYrm['vinfo']['lon_rho'] = {'long_name':'rho point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_rho'] = {'long_name':'rho point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['lon_u'] = {'long_name':'rho point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_u'] = {'long_name': 'u point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['lon_v'] = {'long_name':'v point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_v'] = {'long_name':'v point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['temp'] = {'long_name':'ocean temperature',
                        'units':'degrees C',
                        'coordinates':'z,lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'z,lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['urm'] = {'long_name':'ocean xi velocity',
                        'units':'m/s',
                        'coordinates':'z,lat_u,lon_u',
                        'time':'ocean_time',
                        'note':'this is now rotated in the roms xi direction'}
    HYrm['vinfo']['vrm'] = {'long_name':'ocean eta velocity',
                        'units':'m/s',
                        'coordinates':'z,lat_v,lon_v',
                        'time':'ocean_time',
                        'note':'this is now rotated in the roms eta direction'}
    HYrm['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                        'units':'m/s',
                        'coordinates':'lat_u,lon_u',
                        'time':'ocean_time',
                        'note':'uses hycom depths and this is now rotated in the roms xi direction'}
    HYrm['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                        'units':'m/s',
                        'coordinates':'lat_v,lon_v',
                        'time':'ocean_time',
                        'note':'uses hycom depths and this is now rotated in the roms eta direction'}

    #print('after HYrm setup. before Tmp')

    print('before interp to roms grid, using:')
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('kilobytes')

    reg_u_way = 1

    if reg_u_way == 0:
        Tmpu = dict()
        Tmpv = dict()
        Tmpu['u_on_u'] = np.zeros((NT,NZ, NR, NC-1))
        Tmpu['v_on_u'] = np.zeros((NT,NZ, NR, NC-1))
        Tmpv['u_on_v'] = np.zeros((NT,NZ, NR-1, NC))
        Tmpv['v_on_v'] = np.zeros((NT,NZ, NR-1, NC))

        for aa in vnames:
            zhy  = HY[aa]
            #print('doing:')
            #print(aa)
            for cc in range(NT):
                if aa=='zeta':
                    zhy2 = zhy[cc,:,:]
                    HYrm[aa][cc,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            
                elif aa=='temp' or aa=='salt':
                    for bb in range(NZ):
                        zhy2 = zhy[cc,bb,:,:]
                        HYrm[aa][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)
                elif aa=='u':
                    for bb in range(NZ):
                        zhy2= zhy[cc,bb,:,:]
                        #u_on_u = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],Fz)
                        #u_on_v = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],Fz)
                        Tmpu['u_on_u'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],Fz)
                        Tmpv['u_on_v'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],Fz)
                elif aa=='v':
                    for bb in range(NZ):
                        zhy2= zhy[cc,bb,:,:]
                        Tmpu['v_on_u'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],Fz)
                        Tmpv['v_on_v'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],Fz)
        gc.collect()
        # rotate the velocities so that the velocities are in roms eta,xi coordinates
        angr = RMG['angle_u']
        cosang = np.cos(angr)
        sinang = np.sin(angr)

        print('before rotating urm, using:')
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('kilobytes')

        urm = cosang[None,None,:,:] * Tmpu['u_on_u'][:,:,:,:] + sinang[None,None,:,:] * Tmpu['v_on_u'][:,:,:,:]
        urm[np.isnan(Tmpu['u_on_u'])==1] = np.nan
        HYrm['urm'] = urm
        del Tmpu

        angr = RMG['angle_v']
        cosang = np.cos(angr)
        sinang = np.sin(angr)
        print('before rotating vrm, using:')
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('kilobytes')

        vrm = cosang[None,None,:,:] * Tmpv['v_on_v'][:,:,:,:] - sinang[None,None,:,:] * Tmpv['u_on_v'][:,:,:,:]
        vrm[np.isnan(Tmpv['v_on_v'])==1] = np.nan
        HYrm['vrm'] = vrm
        del Tmpv

    else:
        Tmpu = dict()
        Tmpu['u'] = np.zeros((NT,NZ, NR, NC))
        Tmpu['v'] = np.zeros((NT,NZ, NR, NC))
        for aa in vnames:
            zhy  = HY[aa]
            #print('doing:')
            #print(aa)
            for cc in range(NT):
                if aa=='zeta':
                    zhy2 = zhy[cc,:,:]
                    HYrm[aa][cc,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            
                elif aa=='temp' or aa=='salt':
                    for bb in range(NZ):
                        zhy2 = zhy[cc,bb,:,:]
                        HYrm[aa][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)
                elif aa=='u' or aa=='v':
                    for bb in range(NZ):
                        zhy2= zhy[cc,bb,:,:]
                        Tmpu[aa][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)

        gc.collect()
        # rotate the velocities so that the velocities are in roms eta,xi coordinates
        angr = RMG['angle']
        cosang = np.cos(angr)
        sinang = np.sin(angr)

        print('before rotating urm, using:')
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('kilobytes')

        urm = cosang[None,None,:,:] * Tmpu['u'][:,:,:,:] + sinang[None,None,:,:] * Tmpu['v'][:,:,:,:]
        urm[np.isnan(Tmpu['u'])==1] = np.nan
        HYrm['urm'] = .5*( urm[:,:,:,0:-1] + urm[:,:,:,1:] )
        del urm
        gc.collect()

        print('before rotating vrm, using:')
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('kilobytes')

        vrm = cosang[None,None,:,:] * Tmpu['v'][:,:,:,:] - sinang[None,None,:,:] * Tmpu['u'][:,:,:,:]
        vrm[np.isnan(Tmpu['v'])==1] = np.nan
        HYrm['vrm'] = .5*( vrm[:,:,0:-1,:] + vrm[:,:,1:,:] )
        del Tmpu
        print('after rotating vrm, using:')
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('kilobytes')


    #print('after Tmp. before rotation')
    del HY

    # we need the roms depths on roms u and v grids
    Hru = 0.5 * (RMG['h'][:,0:-1] + RMG['h'][:,1:])
    Hrv = 0.5 * (RMG['h'][0:-1,:] + RMG['h'][1:,:])
    # get the locations in z of the hycom output
    hyz = HYrm['depth'].copy()
    #print(np.shape(Hru))
    #print(np.shape(HYrm['urm']))

    # do ubar, the depth average velocity in roms (eta,xi) coordinates
    # so ubar and vbar are calculated from hycom depths before interpolating to roms depths

    utst = Hru[None,:,:] - hyz[:,None,None]
    vtst = Hrv[None,:,:] - hyz[:,None,None]

    #dz = hyz[1:]-hyz[0:-1]

    umsk = 0*utst
    vmsk = 0*vtst
    umsk[utst>=0] = 1 # this should put zeros at all depths below the bottom
    vmsk[vtst>=0] = 1 # and ones at all depths above the bottom

    # put zeros at hycom depths below the bottom
    HYrm['urm'] = HYrm['urm']*umsk[None,:,:,:]
    HYrm['vrm'] = HYrm['vrm']*vmsk[None,:,:,:]

    print('about to save OCN_R to multiple pickle files...')

    gc.collect()
    print('after gc.collect and before pickling OCN_R, using:')
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('kilobytes')

    ork = HYrm.keys()

    for nm in ork:
        fn_temp = PFM['lv1_forc_dir'] + '/tmp_' + nm + '.pkl'
        with open(fn_temp,'wb') as fp:
            pickle.dump(HYrm[nm],fp)
            print('saved pickle file: ' + fn_temp)

#    return HYrm

def make_all_tmp_pckl_ocnR_files(fname_in):
    
    print('and saving 17 pickle files...')
    
    ork = ['depth','lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v','ocean_time','ocean_time_ref','salt','temp','ubar','urm','vbar','vrm','zeta','vinfo']
    
    os.chdir('../sdpm_py_util')
    rctot = 0

    for aa in ork:
        #print('doing ' + aa + ' using subprocess')
        cmd_list = ['python','-W','ignore','ocn_functions.py','make_tmp_hy_on_rom_pckl_files',fname_in,aa]
        ret1 = subprocess.run(cmd_list)     
        rctot = rctot + ret1.returncode
        if ret1.returncode != 0:
            print('the ' + aa + ' pickle file was not made correctly')

    if rctot == 0: 
        print('...done. \nall 17 ocnR pickle files were made correctly')
    else:
        print('...done. \nat least one of the ocnR pickle files were not made correctly')
    
    os.chdir('../driver')


def make_all_tmp_pckl_ocnR_files_1hrzeta(fname_in):
    
    print('and saving 18 pickle files...')
    
    ork = ['depth','lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v','ocean_time','zeta_time','ocean_time_ref','zeta','salt','temp','ubar','urm','vbar','vrm','vinfo']
    
    os.chdir('../sdpm_py_util')
    rctot = 0

    for aa in ork:
        cmd_list = ['python','-W','ignore','ocn_functions.py','make_tmp_hy_on_rom_pckl_files_1hrzeta',fname_in,aa]
        ret1 = subprocess.run(cmd_list )     
        rctot = rctot + ret1.returncode
        if ret1.returncode != 0:
            print('the ' + aa + ' pickle file was not made correctly')

    if rctot == 0: 
        print('...done. \nall 18 ocnR pickle files were made correctly')
    else:
        print('...done. \nat least one of the ocnR pickle files were not made correctly')
    
    os.chdir('../driver')


def mk_pick_files(cmd_list):

    ret1 = subprocess.run(cmd_list)
    return ret1

def make_all_tmp_pckl_ocnR_files_1hrzeta_para(fname_in):
    
    print('and saving 18 pickle files...')
    
    ork = ['depth','lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v','ocean_time','zeta_time','ocean_time_ref','zeta','salt','temp','ubar','urm','vbar','vrm','vinfo']
    
    os.chdir('../sdpm_py_util')
    rctot = 0

    threads = []
                
    with ThreadPoolExecutor() as executor:
        for aa in ork:
            cmd_list = ['python','-W','ignore','ocn_functions.py','make_tmp_hy_on_rom_pckl_files_1hrzeta',fname_in,aa]
            fn = mk_pick_files #define function
            #args = cmd_list
            kwargs = {} #
            # start thread by submitting it to the executor
            threads.append(executor.submit(fn, cmd_list, **kwargs))

            for future in as_completed(threads):
                # retrieve the result
                ret1 = future.result()
                # report the result
                rctot = rctot + ret1.returncode

    if rctot == 0: 
        print('...done. \nall 18 ocnR pickle files were made correctly')
    else:
        print('...done. \nat least one of the ocnR pickle files was not made correctly')
    
    os.chdir('../driver')



def make_tmp_hy_on_rom_pckl_files_1hrzeta(fname_in,var_name):
    # HYcom and RoMsGrid come in as dicts with ROMS variable names    
    # The output of this, HYrm, is a dict with 
    # hycom fields on roms horizontal grid points
    # but hycom z levels.
    # velocity will be on both (lat_u,lon_u)
    # and (lat_v,lon_v).

    # load the hycom data
    with open(fname_in,'rb') as fp:
        HY = pickle.load(fp)
        #print('OCN dict loaded with pickle')

    lnhy = HY['lon']
    lthy = HY['lat']

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

    NR,NC = np.shape(RMG['lon_rho'])
    NZ = len(HY['depth'])
    NT = len(HY['ocean_time'])
    NTz = len(HY['zeta_time'])

    fn_temp = PFM['lv1_forc_dir'] + '/tmp_' + var_name + '.pkl'

    HYrm=dict()
    if var_name == 'vinfo':
        HYrm['vinfo']=dict()
        HYrm['vinfo']['depth'] = HY['vinfo']['depth']
        HYrm['vinfo']['ocean_time'] = HY['vinfo']['ocean_time']
        HYrm['vinfo']['zeta_time'] = HY['vinfo']['zeta_time']

        HYrm['vinfo']['ocean_time_ref'] = HY['vinfo']['ocean_time_ref']
        HYrm['vinfo']['lon_rho'] = {'long_name':'rho point longitude',
                            'units':'degrees_east'}
        HYrm['vinfo']['lat_rho'] = {'long_name':'rho point latitude',
                            'units':'degrees_north'}
        HYrm['vinfo']['lon_u'] = {'long_name':'rho point longitude',
                            'units':'degrees_east'}
        HYrm['vinfo']['lat_u'] = {'long_name': 'u point latitude',
                            'units':'degrees_north'}
        HYrm['vinfo']['lon_v'] = {'long_name':'v point longitude',
                            'units':'degrees_east'}
        HYrm['vinfo']['lat_v'] = {'long_name':'v point latitude',
                            'units':'degrees_north'}
        HYrm['vinfo']['temp'] = {'long_name':'ocean temperature',
                            'units':'degrees C',
                            'coordinates':'z,lat_rho,lon_rho',
                            'time':'ocean_time'}
        HYrm['vinfo']['salt'] = {'long_name':'ocean salinity',
                            'units':'psu',
                            'coordinates':'z,lat_rho,lon_rho',
                            'time':'ocean_time'}
        HYrm['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                            'units':'m',
                            'coordinates':'lat_rho,lon_rho',
                            'time':'ocean_time'}
        HYrm['vinfo']['urm'] = {'long_name':'ocean xi velocity',
                            'units':'m/s',
                            'coordinates':'z,lat_u,lon_u',
                            'time':'ocean_time',
                            'note':'this is now rotated in the roms xi direction'}
        HYrm['vinfo']['vrm'] = {'long_name':'ocean eta velocity',
                            'units':'m/s',
                            'coordinates':'z,lat_v,lon_v',
                            'time':'ocean_time',
                            'note':'this is now rotated in the roms eta direction'}
        HYrm['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                            'units':'m/s',
                            'coordinates':'lat_u,lon_u',
                            'time':'ocean_time',
                            'note':'uses hycom depths and this is now rotated in the roms xi direction'}
        HYrm['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                            'units':'m/s',
                            'coordinates':'lat_v,lon_v',
                            'time':'ocean_time',
                            'note':'uses hycom depths and this is now rotated in the roms eta direction'}
            
    elif var_name == 'depth' or var_name == 'ocean_time' or var_name == 'zeta_time': 
        HYrm[var_name] = HY[var_name][:] # depths are from hycom
      
    elif var_name == 'ocean_time_ref': 
        HYrm[var_name] = HY[var_name] 
    
    elif var_name == 'lat_rho' or var_name == 'lon_rho' or var_name == 'lat_u' or var_name == 'lon_u' or var_name == 'lat_v' or var_name == 'lon_v':
        HYrm[var_name] = RMG[var_name][:]
   
    elif var_name == 'zeta':
        Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])
        HYrm[var_name] = np.zeros((NTz,NR,NC))
        for cc in range(NTz):
            zhy2 = HY[var_name][cc,:,:]
            HYrm[var_name][cc,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            

    elif var_name == 'salt' or var_name == 'temp':
        Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])
        HYrm[var_name] = np.zeros((NT,NZ,NR,NC))
        for cc in range(NT):
            for bb in range(NZ):
                zhy2 = HY[var_name][cc,bb,:,:]
                HYrm[var_name][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            

    elif var_name == 'urm' or var_name == 'vrm':
        Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])

        hyz = HY['depth']
        angr = RMG['angle']
        cosang = np.cos(angr)
        sinang = np.sin(angr)

        if var_name == 'urm':
            HYrm['urm'] = np.zeros((NT,NZ,NR,NC-1))
        else:
            HYrm['vrm'] = np.zeros((NT,NZ,NR-1,NC)) 
        
        for cc in range(NT):
            for bb in range(NZ):
                uhy = HY['u'][cc,bb,:,:]
                vhy = HY['v'][cc,bb,:,:]
                u1 = interp_hycom_to_roms(lnhy,lthy,uhy,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)
                v1 = interp_hycom_to_roms(lnhy,lthy,vhy,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            
                if var_name == 'urm':
                    u2 = cosang * u1 + sinang * v1
                    u3 = .5*( u2[:,0:-1]+u2[:,1:] )
                    HYrm['urm'][cc,bb,:,:] = u3
                else:
                    v2 = cosang * v1 - sinang * u1
                    v3 = .5*( v2[0:-1,:]+v2[1:,:] )
                    HYrm['vrm'][cc,bb,:,:] = v3

        if var_name == 'urm':
            Hru = 0.5 * (RMG['h'][:,0:-1] + RMG['h'][:,1:])
            utst = Hru[None,:,:] - hyz[:,None,None]
            umsk = 0*utst
            umsk[utst>=0] = 1 # this should put zeros at all depths below the bottom
            HYrm['urm'] = HYrm['urm']*umsk[None,:,:,:]
        else:
            Hrv = 0.5 * (RMG['h'][0:-1,:] + RMG['h'][1:,:])
            vtst = Hrv[None,:,:] - hyz[:,None,None]
            vmsk = 0*vtst
            vmsk[vtst>=0] = 1 # and ones at all depths above the bottom
            HYrm['vrm'] = HYrm['vrm']*vmsk[None,:,:,:]

    elif var_name == 'vbar': # zeros is just a place holder as depth avg velocities are calculated later 
        HYrm[var_name] = np.zeros((NT,NR-1,NC))
    elif var_name == 'ubar':
        HYrm[var_name] = np.zeros((NT,NR,NC-1))    

    with open(fn_temp,'wb') as fp:
        pickle.dump(HYrm[var_name],fp)




def make_tmp_hy_on_rom_pckl_files(fname_in,var_name):
    # HYcom and RoMsGrid come in as dicts with ROMS variable names    
    # The output of this, HYrm, is a dict with 
    # hycom fields on roms horizontal grid points
    # but hycom z levels.
    # velocity will be on both (lat_u,lon_u)
    # and (lat_v,lon_v).

    # load the hycom data
    with open(fname_in,'rb') as fp:
        HY = pickle.load(fp)
        #print('OCN dict loaded with pickle')

    lnhy = HY['lon']
    lthy = HY['lat']

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

    NR,NC = np.shape(RMG['lon_rho'])
    NZ = len(HY['depth'])
    NT = len(HY['ocean_time'])

    fn_temp = PFM['lv1_forc_dir'] + '/tmp_' + var_name + '.pkl'

    HYrm=dict()
    if var_name == 'vinfo':
        HYrm['vinfo']=dict()
        HYrm['vinfo']['depth'] = HY['vinfo']['depth']
        HYrm['vinfo']['ocean_time'] = HY['vinfo']['ocean_time']
        #HYrm['vinfo']['zeta_time'] = HY['vinfo']['zeta_time']

        HYrm['vinfo']['ocean_time_ref'] = HY['vinfo']['ocean_time_ref']
        HYrm['vinfo']['lon_rho'] = {'long_name':'rho point longitude',
                            'units':'degrees_east'}
        HYrm['vinfo']['lat_rho'] = {'long_name':'rho point latitude',
                            'units':'degrees_north'}
        HYrm['vinfo']['lon_u'] = {'long_name':'rho point longitude',
                            'units':'degrees_east'}
        HYrm['vinfo']['lat_u'] = {'long_name': 'u point latitude',
                            'units':'degrees_north'}
        HYrm['vinfo']['lon_v'] = {'long_name':'v point longitude',
                            'units':'degrees_east'}
        HYrm['vinfo']['lat_v'] = {'long_name':'v point latitude',
                            'units':'degrees_north'}
        HYrm['vinfo']['temp'] = {'long_name':'ocean temperature',
                            'units':'degrees C',
                            'coordinates':'z,lat_rho,lon_rho',
                            'time':'ocean_time'}
        HYrm['vinfo']['salt'] = {'long_name':'ocean salinity',
                            'units':'psu',
                            'coordinates':'z,lat_rho,lon_rho',
                            'time':'ocean_time'}
        HYrm['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                            'units':'m',
                            'coordinates':'lat_rho,lon_rho',
                            'time':'ocean_time'}
        HYrm['vinfo']['urm'] = {'long_name':'ocean xi velocity',
                            'units':'m/s',
                            'coordinates':'z,lat_u,lon_u',
                            'time':'ocean_time',
                            'note':'this is now rotated in the roms xi direction'}
        HYrm['vinfo']['vrm'] = {'long_name':'ocean eta velocity',
                            'units':'m/s',
                            'coordinates':'z,lat_v,lon_v',
                            'time':'ocean_time',
                            'note':'this is now rotated in the roms eta direction'}
        HYrm['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                            'units':'m/s',
                            'coordinates':'lat_u,lon_u',
                            'time':'ocean_time',
                            'note':'uses hycom depths and this is now rotated in the roms xi direction'}
        HYrm['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                            'units':'m/s',
                            'coordinates':'lat_v,lon_v',
                            'time':'ocean_time',
                            'note':'uses hycom depths and this is now rotated in the roms eta direction'}
            
    elif var_name == 'depth' or var_name == 'ocean_time': 
        HYrm[var_name] = HY[var_name][:] # depths are from hycom
      
    elif var_name == 'ocean_time_ref': 
        HYrm[var_name] = HY[var_name] 
    
    elif var_name == 'lat_rho' or var_name == 'lon_rho' or var_name == 'lat_u' or var_name == 'lon_u' or var_name == 'lat_v' or var_name == 'lon_v':
        HYrm[var_name] = RMG[var_name][:]
   
    elif var_name == 'zeta':
        Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])
        HYrm[var_name] = np.zeros((NT,NR,NC))
        for cc in range(NT):
            zhy2 = HY[var_name][cc,:,:]
            HYrm[var_name][cc,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            

    elif var_name == 'salt' or var_name == 'temp':
        Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])
        HYrm[var_name] = np.zeros((NT,NZ,NR,NC))
        for cc in range(NT):
            for bb in range(NZ):
                zhy2 = HY[var_name][cc,bb,:,:]
                HYrm[var_name][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            

    elif var_name == 'urm' or var_name == 'vrm':
        Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])

        hyz = HY['depth']
        angr = RMG['angle']
        cosang = np.cos(angr)
        sinang = np.sin(angr)

        if var_name == 'urm':
            HYrm['urm'] = np.zeros((NT,NZ,NR,NC-1))
        else:
            HYrm['vrm'] = np.zeros((NT,NZ,NR-1,NC)) 
        
        for cc in range(NT):
            for bb in range(NZ):
                uhy = HY['u'][cc,bb,:,:]
                vhy = HY['v'][cc,bb,:,:]
                u1 = interp_hycom_to_roms(lnhy,lthy,uhy,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)
                v1 = interp_hycom_to_roms(lnhy,lthy,vhy,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            
                if var_name == 'urm':
                    u2 = cosang * u1 + sinang * v1
                    u3 = .5*( u2[:,0:-1]+u2[:,1:] )
                    HYrm['urm'][cc,bb,:,:] = u3
                else:
                    v2 = cosang * v1 - sinang * u1
                    v3 = .5*( v2[0:-1,:]+v2[1:,:] )
                    HYrm['vrm'][cc,bb,:,:] = v3

        if var_name == 'urm':
            Hru = 0.5 * (RMG['h'][:,0:-1] + RMG['h'][:,1:])
            utst = Hru[None,:,:] - hyz[:,None,None]
            umsk = 0*utst
            umsk[utst>=0] = 1 # this should put zeros at all depths below the bottom
            HYrm['urm'] = HYrm['urm']*umsk[None,:,:,:]
        else:
            Hrv = 0.5 * (RMG['h'][0:-1,:] + RMG['h'][1:,:])
            vtst = Hrv[None,:,:] - hyz[:,None,None]
            vmsk = 0*vtst
            vmsk[vtst>=0] = 1 # and ones at all depths above the bottom
            HYrm['vrm'] = HYrm['vrm']*vmsk[None,:,:,:]

    elif var_name == 'vbar': # zeros is just a place holder as depth avg velocities are calculated later 
        HYrm[var_name] = np.zeros((NT,NR-1,NC))
    elif var_name == 'ubar':
        HYrm[var_name] = np.zeros((NT,NR,NC-1))    

    with open(fn_temp,'wb') as fp:
        pickle.dump(HYrm[var_name],fp)




def print_maxmin_HYrm_pickles():

    HYrm = load_ocnR_from_pckl_files()
    vlist = ['zeta','urm','vrm','temp','salt']
    ulist = ['m','m/s','m/s','C','psu']
    ulist2 = dict(zip(vlist,ulist))
    
    print('\nmax and min of hycom data on ROMS grid (iz is top [0] to bottom [39]):')

    for var_name in vlist:
        ind_mx = np.unravel_index(np.nanargmax(HYrm[var_name], axis=None), HYrm[var_name].shape)
        ind_mn = np.unravel_index(np.nanargmin(HYrm[var_name], axis=None), HYrm[var_name].shape)
        mxx = HYrm[var_name][ind_mx]
        mnn = HYrm[var_name][ind_mn]
        uvnm = ulist2[var_name]
        if var_name == 'zeta':
            print(f'max {var_name:6} = {mxx:6.3f} {uvnm:5}      at  ( it, ilat, ilon)     =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')
            print(f'min {var_name:6} = {mnn:6.3f} {uvnm:5}      at  ( it, ilat, ilon)     =  ({ind_mn[0]:3},{ind_mn[1]:4},{ind_mn[2]:4})')
        else:
            print(f'max {var_name:6} = {mxx:6.3f} {uvnm:5}      at  ( it, iz, ilat, ilon) =  ({ind_mx[0]:3},{ind_mx[1]:3},{ind_mx[2]:4},{ind_mx[3]:4})')
            print(f'min {var_name:6} = {mnn:6.3f} {uvnm:5}      at  ( it, iz, ilat, ilon) =  ({ind_mn[0]:3},{ind_mn[1]:3},{ind_mn[2]:4},{ind_mn[3]:4})')
        if var_name == 'temp' or var_name == 'salt':
            if var_name == 'salt':
                vvv = 'dS/dz'
                uvv = 'psu/m'
            else:
                vvv = 'dT/dz'
                uvv = 'C/m'
            dz = HYrm['depth'][1:] - HYrm['depth'][0:-1]
            dT = HYrm[var_name][:,0:-1,:,:] - HYrm[var_name][:,1:,:,:]
            dTdz = dT / dz[None,:,None,None]
            ind_mx = np.unravel_index(np.nanargmax(dTdz, axis=None), dTdz.shape)
            ind_mn = np.unravel_index(np.nanargmin(dTdz, axis=None), dTdz.shape)
            mxx = dTdz[ind_mx]
            mnn = dTdz[ind_mn]
            print(f'max {vvv:6} = {mxx:6.3f} {uvv:5}      at  ( it, iz, ilat, ilon) =  ({ind_mx[0]:3},{ind_mx[1]:3},{ind_mx[2]:4},{ind_mx[3]:4})')
            print(f'min {vvv:6} = {mnn:6.3f} {uvv:5}      at  ( it, iz, ilat, ilon) =  ({ind_mn[0]:3},{ind_mn[1]:3},{ind_mn[2]:4},{ind_mn[3]:4})')


def load_ocnR_from_pckl_files():

    PFM=get_PFM_info()
    onehr_zeta = 1
    if onehr_zeta == 0:
        ork = ['depth','lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v','ocean_time','ocean_time_ref','salt','temp','ubar','urm','vbar','vrm','zeta','vinfo']
    else:
        ork = ['depth','lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v','ocean_time','zeta_time','ocean_time_ref','salt','temp','ubar','urm','vbar','vrm','zeta','vinfo']

    OCN_R = dict()
    for nm in ork:
        fn_temp = PFM['lv1_forc_dir'] + '/tmp_' + nm + '.pkl'
        with open(fn_temp,'rb') as fp:
            OCN_R[nm] = pickle.load(fp)

    return OCN_R



def ocn_r_2_ICdict_pckl(fname_out):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the initial condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_IC that has all needed fields 
    # for making the .nc file

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

    OCN_R = load_ocnR_from_pckl_files()

    #with open(fname_in,'rb') as fp:
    #    OCN_R = pickle.load(fp)
    #    print('OCN_R dict loaded with pickle')


    i0 = 0 # we will use the first time as the initial condition
    
    OCN_IC = dict()
    # fill in the dict with slicing
    OCN_IC['ocean_time'] = np.zeros((1))
    OCN_IC['ocean_time'][0] = OCN_R['ocean_time'][i0]
    #OCN_IC['ubar'] = np.squeeze(OCN_R['ubar'][i0,:,:])
    #OCN_IC['vbar'] = np.squeeze(OCN_R['vbar'][i0,:,:])

    # the variables that are the same
    var_same = ['lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v'] 
    for vn in var_same:
        OCN_IC[vn] = OCN_R[vn][:]

    OCN_IC['ocean_time_ref'] = OCN_R['ocean_time_ref']


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = OCN_R['depth'] # these are the hycom depths
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

#    Nz   = RMG['Nz']                              # number of vertical levels: 40
#    Vtr  = RMG['Vtransform']                       # transformation equation: 2
#    Vst  = RMG['Vstretching']                    # stretching function: 4 
#    th_s = RMG['THETA_S']                      # surface stretching parameter: 8
#    th_b = RMG['THETA_B']                      # bottom  stretching parameter: 3
#    Tcl  = RMG['TCLINE']                      # critical depth (m): 50

    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    OCN_IC['zeta'] = np.zeros((1,nlt,nln))
    OCN_IC['zeta'][0,:,:] = OCN_R['zeta'][i0,:,:]

    eta = np.squeeze(OCN_IC['zeta'][0,:,:])
    eta_u = 0.5 * (eta[:,0:-1]+eta[:,1:])
    eta_v = 0.5 * (eta[0:-1,:]+eta[1:,:])

    OCN_IC['Nz'] = np.squeeze(Nz)
    OCN_IC['Vtr'] = np.squeeze(Vtr)
    OCN_IC['Vst'] = np.squeeze(Vst)
    OCN_IC['th_s'] = np.squeeze(th_s)
    OCN_IC['th_b'] = np.squeeze(th_b)
    OCN_IC['Tcl'] = np.squeeze(Tcl)
    OCN_IC['hc'] = np.squeeze(hc)

    TMP = dict()
    TMP['temp'] = np.zeros((1,Nz,nlt,nln)) # a helper becasue we convert to potential temp below
    OCN_IC['temp'] = np.zeros((1,Nz,nlt,nln))
    OCN_IC['salt'] = np.zeros((1,Nz,nlt,nln))
    OCN_IC['u'] = np.zeros((1,Nz,nlt,nln-1))
    OCN_IC['v'] = np.zeros((1,Nz,nlt-1,nln))
    OCN_IC['ubar'] = np.zeros((1,nlt,nln-1))
    OCN_IC['vbar'] = np.zeros((1,nlt-1,nln))

    OCN_IC['vinfo']=dict()
    OCN_IC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_IC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_IC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_IC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_IC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_IC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_IC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_IC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']

    OCN_IC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_IC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_IC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_IC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_IC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_IC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_IC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    OCN_IC['vinfo']['temp'] = {'long_name':'ocean potential temperature',
                        'units':'degrees C',
                        'coordinates':'time,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'tiem,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'time,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['u'] = {'long_name':'ocean xi velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_u,lon_u',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['v'] = {'long_name':'ocean eta velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_v,lon_v',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_u,lon_u',
                        'note':'uses roms depths'}
    OCN_IC['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_v,lon_v',
                        'note':'uses roms depths'}

    get_depth_file = 1

    if get_depth_file == 0:
    # get the roms z's
        hraw = None
        if Vst == 4:
            zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta)
            zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_u)
            zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_v)
    
        OCN_IC['Cs_r'] = np.squeeze(zrom.Cs_r)
        zr=np.squeeze(zrom.z_r[0,:,:,:])    
        zr_u=np.squeeze(zrom_u.z_r[0,:,:,:])    
        zr_v=np.squeeze(zrom_v.z_r[0,:,:,:])
    else:
        fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']
        Zrm = load_rom_depths(fname_depths)
        zr=Zrm['zr_ic']
        zr_u=Zrm['zu_ic']
        zr_v=Zrm['zv_ic']
        OCN_IC['Cs_r'] = Zrm['Cs_r']

    OCN_IC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}

    for aa in range(nlt):
        for bb in range(nln):    
            TMP['temp'][0,:,aa,bb]    = interp_to_roms_z(-zhy,OCN_R['temp'][0,:,aa,bb],zr[:,aa,bb],-hb[aa,bb])
            OCN_IC['salt'][0,:,aa,bb] = interp_to_roms_z(-zhy,OCN_R['salt'][0,:,aa,bb],zr[:,aa,bb],-hb[aa,bb])
            
            if aa < nlt-1:            
                OCN_IC['v'][0,:,aa,bb]    = interp_to_roms_z(-zhy,OCN_R['vrm'][0,:,aa,bb],zr_v[:,aa,bb],-hb_v[aa,bb])
                OCN_IC['vbar'][0,aa,bb]    = get_depth_avg_v(OCN_IC['v'][0,:,aa,bb],zr_v[:,aa,bb],eta_v[aa,bb],hb_v[aa,bb])
 
            if bb < nln-1:
                OCN_IC['u'][0,:,aa,bb]    = interp_to_roms_z(-zhy,OCN_R['urm'][0,:,aa,bb],zr_u[:,aa,bb],-hb_u[aa,bb])
                OCN_IC['ubar'][0,aa,bb]    = get_depth_avg_v(OCN_IC['u'][0,:,aa,bb],zr_u[:,aa,bb],eta_u[aa,bb],hb_u[aa,bb])

    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr # pressure in dbar
    OCN_IC['temp'][0,:,:,:] = seawater.ptmp(np.squeeze(OCN_IC['salt']), np.squeeze(TMP['temp']),np.squeeze(pdb))  


    with open(fname_out,'wb') as fout:
        pickle.dump(OCN_IC,fout)
        print('OCN_IC dict saved with pickle')

#    return OCN_IC

def load_tmp_pkl(var_name):

    PFM=get_PFM_info()
    fn_temp = PFM['lv1_forc_dir'] + '/tmp_' + var_name + '.pkl'
    with open(fn_temp,'rb') as fp:
        tmp_dat = pickle.load(fp)

    return tmp_dat

def ocnr_2_ICdict_from_tmppkls(fname_out):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the initial condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_IC that has all needed fields 
    # for making the .nc file

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

    #OCN_R = load_ocnR_from_pckl_files()

    #with open(fname_in,'rb') as fp:
    #    OCN_R = pickle.load(fp)
    #    print('OCN_R dict loaded with pickle')


    i0 = 0 # we will use the first time as the initial condition
    
    OCN_IC = dict()
    # fill in the dict with slicing
    OCN_IC['ocean_time'] = np.zeros((1))

    tmp_dat = load_tmp_pkl('ocean_time')
    OCN_IC['ocean_time'][0] = tmp_dat[i0]
    del tmp_dat
    #OCN_IC['ubar'] = np.squeeze(OCN_R['ubar'][i0,:,:])
    #OCN_IC['vbar'] = np.squeeze(OCN_R['vbar'][i0,:,:])

    # the variables that are the same
    var_same = ['lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v'] 
    for vn in var_same:
        OCN_IC[vn] = load_tmp_pkl(vn)

    OCN_IC['ocean_time_ref'] = load_tmp_pkl('ocean_time_ref')


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = load_tmp_pkl('depth') # these are the hycom depths
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

#    Nz   = RMG['Nz']                              # number of vertical levels: 40
#    Vtr  = RMG['Vtransform']                       # transformation equation: 2
#    Vst  = RMG['Vstretching']                    # stretching function: 4 
#    th_s = RMG['THETA_S']                      # surface stretching parameter: 8
#    th_b = RMG['THETA_B']                      # bottom  stretching parameter: 3
#    Tcl  = RMG['TCLINE']                      # critical depth (m): 50

    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    OCN_IC['zeta'] = np.zeros((1,nlt,nln))
    tmp_dat = load_tmp_pkl('zeta')
    OCN_IC['zeta'][0,:,:] = tmp_dat[i0,:,:]
    del tmp_dat

    eta = np.squeeze(OCN_IC['zeta'][0,:,:])
    eta_u = 0.5 * (eta[:,0:-1]+eta[:,1:])
    eta_v = 0.5 * (eta[0:-1,:]+eta[1:,:])

    OCN_IC['Nz'] = np.squeeze(Nz)
    OCN_IC['Vtr'] = np.squeeze(Vtr)
    OCN_IC['Vst'] = np.squeeze(Vst)
    OCN_IC['th_s'] = np.squeeze(th_s)
    OCN_IC['th_b'] = np.squeeze(th_b)
    OCN_IC['Tcl'] = np.squeeze(Tcl)
    OCN_IC['hc'] = np.squeeze(hc)

    TMP = dict()
    TMP['temp'] = np.zeros((1,Nz,nlt,nln)) # a helper becasue we convert to potential temp below
    OCN_IC['temp'] = np.zeros((1,Nz,nlt,nln))
    OCN_IC['salt'] = np.zeros((1,Nz,nlt,nln))
    OCN_IC['u'] = np.zeros((1,Nz,nlt,nln-1))
    OCN_IC['v'] = np.zeros((1,Nz,nlt-1,nln))
    OCN_IC['ubar'] = np.zeros((1,nlt,nln-1))
    OCN_IC['vbar'] = np.zeros((1,nlt-1,nln))

    OCN_IC['vinfo']=dict()
    
    tmp_dat = load_tmp_pkl('vinfo')

    OCN_IC['vinfo']['ocean_time'] = tmp_dat['ocean_time']
    OCN_IC['vinfo']['ocean_time_ref'] = tmp_dat['ocean_time_ref']
    OCN_IC['vinfo']['lat_rho'] = tmp_dat['lat_rho']
    OCN_IC['vinfo']['lon_rho'] = tmp_dat['lat_rho']
    OCN_IC['vinfo']['lat_u'] = tmp_dat['lat_u']
    OCN_IC['vinfo']['lon_u'] = tmp_dat['lon_u']
    OCN_IC['vinfo']['lat_v'] = tmp_dat['lat_v']
    OCN_IC['vinfo']['lon_v'] = tmp_dat['lon_v']
    del tmp_dat

    OCN_IC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_IC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_IC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_IC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_IC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_IC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_IC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    OCN_IC['vinfo']['temp'] = {'long_name':'ocean potential temperature',
                        'units':'degrees C',
                        'coordinates':'time,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'tiem,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'time,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['u'] = {'long_name':'ocean xi velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_u,lon_u',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['v'] = {'long_name':'ocean eta velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_v,lon_v',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_u,lon_u',
                        'note':'uses roms depths'}
    OCN_IC['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_v,lon_v',
                        'note':'uses roms depths'}

    get_depth_file = 1

    if get_depth_file == 0:
    # get the roms z's
        hraw = None
        if Vst == 4:
            zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta)
            zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_u)
            zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_v)
    
        OCN_IC['Cs_r'] = np.squeeze(zrom.Cs_r)
        zr=np.squeeze(zrom.z_r[0,:,:,:])    
        zr_u=np.squeeze(zrom_u.z_r[0,:,:,:])    
        zr_v=np.squeeze(zrom_v.z_r[0,:,:,:])
    else:
        fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']
        Zrm = load_rom_depths(fname_depths)
        zr=Zrm['zr_ic']
        zr_u=Zrm['zu_ic']
        zr_v=Zrm['zv_ic']
        OCN_IC['Cs_r'] = Zrm['Cs_r']

    OCN_IC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}

    tmp_dat = load_tmp_pkl('temp')

    for aa in range(nlt):
        for bb in range(nln):    
            TMP['temp'][0,:,aa,bb]    = interp_to_roms_z(-zhy,tmp_dat[0,:,aa,bb],zr[:,aa,bb],-hb[aa,bb])
 
    del tmp_dat
    tmp_dat = load_tmp_pkl('salt')
    for aa in range(nlt):
        for bb in range(nln):    
            OCN_IC['salt'][0,:,aa,bb] = interp_to_roms_z(-zhy,tmp_dat[0,:,aa,bb],zr[:,aa,bb],-hb[aa,bb])
            
            
    del tmp_dat
    tmp_dat = load_tmp_pkl('vrm')
    for aa in range(nlt-1):
        for bb in range(nln):    
            OCN_IC['v'][0,:,aa,bb]    = interp_to_roms_z(-zhy,tmp_dat[0,:,aa,bb],zr_v[:,aa,bb],-hb_v[aa,bb])
            OCN_IC['vbar'][0,aa,bb]    = get_depth_avg_v(OCN_IC['v'][0,:,aa,bb],zr_v[:,aa,bb],eta_v[aa,bb],hb_v[aa,bb])
 
    del tmp_dat
    tmp_dat = load_tmp_pkl('urm')
    for aa in range(nlt):
        for bb in range(nln-1):    
            OCN_IC['u'][0,:,aa,bb]    = interp_to_roms_z(-zhy,tmp_dat[0,:,aa,bb],zr_u[:,aa,bb],-hb_u[aa,bb])
            OCN_IC['ubar'][0,aa,bb]    = get_depth_avg_v(OCN_IC['u'][0,:,aa,bb],zr_u[:,aa,bb],eta_u[aa,bb],hb_u[aa,bb])

    del tmp_dat
    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr # pressure in dbar
    OCN_IC['temp'][0,:,:,:] = seawater.ptmp(np.squeeze(OCN_IC['salt']), np.squeeze(TMP['temp']),np.squeeze(pdb))  


    with open(fname_out,'wb') as fout:
        pickle.dump(OCN_IC,fout)
        print('OCN_IC dict saved with pickle')

#    return OCN_IC




def ocn_r_2_BCdict(OCN_R,RMG,PFM):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the boundary condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_BC that has all needed fields 
    # for making the BC.nc file

    OCN_BC = dict()
    # fill in the dict with slicing
    OCN_BC['ocean_time'] = OCN_R['ocean_time']
    Nt = len( OCN_BC['ocean_time'] )
    OCN_BC['ocean_time_ref'] = OCN_R['ocean_time_ref']

#    OCN_BC['Nz'] = np.squeeze(RMG['Nz'])
#    OCN_BC['Vtr'] = np.squeeze(RMG['Vtransform'])
#    OCN_BC['Vst'] = np.squeeze(RMG['Vstretching'])
#    OCN_BC['th_s'] = np.squeeze(RMG['THETA_S'])
#    OCN_BC['th_b'] = np.squeeze(RMG['THETA_B'])
#    OCN_BC['Tcl'] = np.squeeze(RMG['TCLINE'])
#    OCN_BC['hc'] = np.squeeze(RMG['hc'])


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = OCN_R['depth'] # these are the hycom depths
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

#    Nz   = RMG['Nz']                              # number of vertical levels: 40
#    Vtr  = RMG['Vtransform']                       # transformation equation: 2
#    Vst  = RMG['Vstretching']                    # stretching function: 4 
#    th_s = RMG['THETA_S']                      # surface stretching parameter: 8
#    th_b = RMG['THETA_B']                      # bottom  stretching parameter: 3
#    Tcl  = RMG['TCLINE']                      # critical depth (m): 50

    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    OCN_BC['Nz'] = np.squeeze(Nz)
    OCN_BC['Vtr'] = np.squeeze(Vtr)
    OCN_BC['Vst'] = np.squeeze(Vst)
    OCN_BC['th_s'] = np.squeeze(th_s)
    OCN_BC['th_b'] = np.squeeze(th_b)
    OCN_BC['Tcl'] = np.squeeze(Tcl)
    OCN_BC['hc'] = np.squeeze(hc)


    #eta = np.squeeze(OCN_R['zeta'].copy())
    #eta_u = 0.5 * (eta[:,0:-1]+eta[:,1:])
    #eta_v = 0.5 * (eta[0:-1,:]+eta[1:,:])

    OCN_BC['temp_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_south'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Nt,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_north'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Nt,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['u_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['v_west'] = np.zeros((Nt,Nz,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Nt,nlt))

    OCN_BC['zeta_south'] = np.squeeze(OCN_R['zeta'][:,0,:])
    OCN_BC['zeta_north'] = np.squeeze(OCN_R['zeta'][:,-1,:])
    OCN_BC['zeta_west'] = np.squeeze(OCN_R['zeta'][:,:,0])
    OCN_BC['ubar_south'] = np.squeeze(OCN_R['ubar'][:,0,:])
    OCN_BC['ubar_north'] = np.squeeze(OCN_R['ubar'][:,-1,:])
    OCN_BC['ubar_west'] = np.squeeze(OCN_R['ubar'][:,:,0])
    OCN_BC['vbar_south'] = np.squeeze(OCN_R['vbar'][:,0,:])
    OCN_BC['vbar_north'] = np.squeeze(OCN_R['vbar'][:,-1,:])
    OCN_BC['vbar_west'] = np.squeeze(OCN_R['vbar'][:,:,0])
     
    TMP = dict()
    TMP['temp_north'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_south'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_west'] = np.zeros((Nt,Nz,nlt)) # a helper becasue we convert to potential temp below
    
 
    OCN_BC['vinfo']=dict()
    OCN_BC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_BC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_BC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_BC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_BC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_BC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']

    OCN_BC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_BC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_BC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_BC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_BC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_BC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_BC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    
    OCN_BC['vinfo']['temp_south'] = {'long_name':'ocean potential temperature southern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_south'] = {'long_name':'ocean salinity southern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_south'] = {'long_name':'ocean sea surface height southern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_south'] = {'long_name':'ocean xi velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_south'] = {'long_name':'ocean eta velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_south'] = {'long_name':'ocean xi depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_south'] = {'long_name':'ocean eta depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_north'] = {'long_name':'ocean potential temperature northern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_north'] = {'long_name':'ocean salinity northern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_north'] = {'long_name':'ocean sea surface height northern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_north'] = {'long_name':'ocean xi velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_north'] = {'long_name':'ocean eta velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_north'] = {'long_name':'ocean xi depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_north'] = {'long_name':'ocean eta depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_west'] = {'long_name':'ocean potential temperature western boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_west'] = {'long_name':'ocean salinity western boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_west'] = {'long_name':'ocean sea surface height western boundary',
                        'units':'m',
                        'coordinates':'time,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_west'] = {'long_name':'ocean xi velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_west'] = {'long_name':'ocean eta velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_west'] = {'long_name':'ocean xi depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_west'] = {'long_name':'ocean eta depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_v',
                        'note':'uses roms depths'}

    eta = np.squeeze(OCN_R['zeta'])
    eta_u = 0.5 * (eta[:,:,0:-1]+eta[:,:,1:])
    eta_v = 0.5 * (eta[:,0:-1,:]+eta[:,1:,:])

    # get the roms z's
    hraw = None
    if Vst == 4:
        zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta))
        zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta_u))
        zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta_v))
    
    OCN_BC['Cs_r'] = np.squeeze(zrom.Cs_r)
    OCN_BC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    
    # do we need sc_r ???
    #OCN_IC['sc_r'] = []
    #OCN_IC['vinfo']['sc_r'] = {'long_name':'S-coordinate at RHO-points',
    #                    'units':'nondimensional',
    #                    'valid min':'-1',
    #                    'valid max':'0',
    #                    'field':'sc_r, scalar, series'}

    zr_s=np.squeeze(zrom.z_r[:,:,0,:])    
    zr_us=np.squeeze(zrom_u.z_r[:,:,0,:])    
    zr_vs=np.squeeze(zrom_v.z_r[:,:,0,:])
    zr_n=np.squeeze(zrom.z_r[:,:,-1,:])    
    zr_un=np.squeeze(zrom_u.z_r[:,:,-1,:])    
    zr_vn=np.squeeze(zrom_v.z_r[:,:,-1,:])
    zr_w=np.squeeze(zrom.z_r[:,:,:,0])    
    zr_uw=np.squeeze(zrom_u.z_r[:,:,:,0])    
    zr_vw=np.squeeze(zrom_v.z_r[:,:,:,0])

    del zrom, zrom_u, zrom_v

    for aa in range(Nt):
        for bb in range(nln):
            fofz = np.squeeze(OCN_R['temp'][aa,:,0,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_south'][aa,:,bb] = np.squeeze(np.nan*zr_s[aa,:,bb])
                OCN_BC['salt_south'][aa,:,bb] = np.squeeze(np.nan*zr_s[aa,:,bb])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                #print(np.shape(zr_s))
                #print(np.shape(TMP['temp_south']))
                TMP['temp_south'][aa,:,bb] = np.squeeze(Fz(zr_s[aa,:,bb]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,0,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_south'][aa,:,bb] = np.squeeze(Fz(zr_s[aa,:,bb]))
                
            fofz = np.squeeze(OCN_R['vrm'][aa,:,0,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['v_south'][aa,:,bb] = np.squeeze(np.nan*zr_vs[aa,:,bb])
                OCN_BC['vbar_south'][aa,bb] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                vv =  np.squeeze(Fz(zr_vs[aa,:,bb]))                
                OCN_BC['v_south'][aa,:,bb] = vv
                z2 = np.squeeze(zr_vs[aa,:,bb])
                z3 = np.append(z2,eta_v[aa,0,bb])
                dz = np.diff(z3)
                OCN_BC['vbar_south'][aa,bb] = np.sum(vv*dz) / hb_v[0,bb]

            if bb < nln-1:
                fofz = np.squeeze(OCN_R['urm'][aa,:,0,bb])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['u_south'][aa,:,bb] = np.squeeze(np.nan*zr_us[aa,:,bb])
                    OCN_BC['ubar_south'][aa,bb] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    uu =  np.squeeze(Fz(zr_us[aa,:,bb]))                
                    OCN_BC['u_south'][aa,:,bb] = uu
                    z2 = np.squeeze(zr_us[aa,:,bb])
                    z3 = np.append(z2,eta_u[aa,0,bb])
                    dz = np.diff(z3)
                    OCN_BC['ubar_south'][aa,bb] = np.sum(uu*dz) / hb_u[0,bb]

            fofz = np.squeeze(OCN_R['temp'][aa,:,-1,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_north'][aa,:,bb] = np.squeeze(np.nan*zr_n[aa,:,bb])
                OCN_BC['salt_north'][aa,:,bb] = np.squeeze(np.nan*zr_n[aa,:,bb])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp_north'][aa,:,bb] = np.squeeze(Fz(zr_n[aa,:,bb]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,-1,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_north'][aa,:,bb] = np.squeeze(Fz(zr_n[aa,:,bb]))
                
            fofz = np.squeeze(OCN_R['vrm'][aa,:,-1,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['v_north'][aa,:,bb] = np.squeeze(np.nan*zr_vn[aa,:,bb])
                OCN_BC['vbar_north'][aa,bb] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                vv =  np.squeeze(Fz(zr_vn[aa,:,bb]))                
                OCN_BC['v_north'][aa,:,bb] = vv
                z2 = np.squeeze(zr_vs[aa,:,bb])
                z3 = np.append(z2,eta_v[aa,-1,bb])
                dz = np.diff(z3)
                OCN_BC['vbar_north'][aa,bb] = np.sum(vv*dz) / hb_v[-1,bb]

            if bb < nln-1:
                fofz = np.squeeze(OCN_R['urm'][aa,:,-1,bb])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['u_north'][aa,:,bb] = np.squeeze(np.nan*zr_un[aa,:,bb])
                    OCN_BC['ubar_north'][aa,bb] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    uu =  np.squeeze(Fz(zr_un[aa,:,bb]))                
                    OCN_BC['u_north'][aa,:,bb] = uu
                    z2 = np.squeeze(zr_un[aa,:,bb])
                    z3 = np.append(z2,eta_u[aa,0,bb])
                    dz = np.diff(z3)
                    OCN_BC['ubar_north'][aa,bb] = np.sum(uu*dz) / hb_u[-1,bb]

        for cc in range(nlt):
            fofz = np.squeeze(OCN_R['temp'][aa,:,cc,0])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_west'][aa,:,cc] = np.squeeze(np.nan*zr_w[aa,:,cc])
                OCN_BC['salt_west'][aa,:,cc] = np.squeeze(np.nan*zr_w[aa,:,cc])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp_west'][aa,:,cc] = np.squeeze(Fz(zr_w[aa,:,cc]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,cc,0])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_west'][aa,:,cc] = np.squeeze(Fz(zr_w[aa,:,cc]))

            if cc < nlt-1:    
                fofz = np.squeeze(OCN_R['vrm'][aa,:,cc,0])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['v_west'][aa,:,cc] = np.squeeze(np.nan*zr_vw[aa,:,cc])
                    OCN_BC['vbar_west'][aa,cc] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    vv =  np.squeeze(Fz(zr_vw[aa,:,cc]))                
                    OCN_BC['v_west'][aa,:,cc] = vv
                    z2 = np.squeeze(zr_vw[aa,:,cc])
                    z3 = np.append(z2,eta_v[aa,cc,0])
                    dz = np.diff(z3)
                    OCN_BC['vbar_west'][aa,cc] = np.sum(vv*dz) / hb_v[cc,0]

            fofz = np.squeeze(OCN_R['urm'][aa,:,cc,0])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['u_west'][aa,:,cc] = np.squeeze(np.nan*zr_uw[aa,:,cc])
                OCN_BC['ubar_west'][aa,cc] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                uu =  np.squeeze(Fz(zr_uw[aa,:,cc]))                
                OCN_BC['u_west'][aa,:,cc] = uu
                z2 = np.squeeze(zr_uw[aa,:,cc])
                z3 = np.append(z2,eta_u[aa,cc,0])
                dz = np.diff(z3)
                OCN_BC['ubar_west'][aa,cc] = np.sum(uu*dz) / hb_u[cc,0]


    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr_s # pressure in dbar
    OCN_BC['temp_south'] = seawater.ptmp(np.squeeze(OCN_BC['salt_south']), np.squeeze(TMP['temp_south']),np.squeeze(pdb))  
    pdb = -zr_n # pressure in dbar
    OCN_BC['temp_north'] = seawater.ptmp(np.squeeze(OCN_BC['salt_north']), np.squeeze(TMP['temp_north']),np.squeeze(pdb))  
    pdb = -zr_w # pressure in dbar
    OCN_BC['temp_west'] = seawater.ptmp(np.squeeze(OCN_BC['salt_west']), np.squeeze(TMP['temp_west']),np.squeeze(pdb))  

    return OCN_BC

def make_rom_depths(fname_depths_pickle):

    if os.path.isfile(fname_depths_pickle) == False: # might need to make the file if it doesn't exist...
        print('making roms depth pickle file ' + fname_depths_pickle + '...')
        make_temp_rom_depth_files(fname_depths_pickle)
        print('...done makeing depth pickle file.')
    else:
        print('depth pickle file ' + fname_depths_pickle + 'already exists.')

def make_rom_depths_1hrzeta(fname_depths_pickle):

    if os.path.isfile(fname_depths_pickle) == False: # might need to make the file if it doesn't exist...
        print('making roms depth pickle file ' + fname_depths_pickle + '...')
        make_temp_rom_depth_files_1hrzeta(fname_depths_pickle)
        print('...done makeing depth pickle file.')
    else:
        print('depth pickle file ' + fname_depths_pickle + 'already exists.')



def load_rom_depths(fname_depths_pickle):

    # if the file doesn't exit, this function makes the file as a subprocess
    # this is a function that makes the roms depths pickle file for the IC / BC routines 


    with open(fname_depths_pickle,'rb') as fp:
            ROMdepths = pickle.load(fp)

    return ROMdepths    

    
def make_temp_rom_depth_files(fname_out):
    
    PFM=get_PFM_info()
    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    del RMG

    fn_temp = PFM['lv1_forc_dir'] + '/tmp_' + 'zeta' + '.pkl'
    with open(fn_temp,'rb') as fp:
        zeta = pickle.load(fp)

    Nt,Nlt,Nln = np.shape(zeta)
    hraw = None
    Zrm=dict()
    Zrm['zr_ic']   = np.zeros((Nz,Nlt,Nln))
    Zrm['zu_ic']   = np.zeros((Nz,Nlt,Nln-1))
    Zrm['zv_ic']   = np.zeros((Nz,Nlt-1,Nln))
    Zrm['zr_bc_n'] = np.zeros((Nt,Nz,Nln))
    Zrm['zr_bc_s'] = np.zeros((Nt,Nz,Nln))
    Zrm['zr_bc_w'] = np.zeros((Nt,Nz,Nlt))
    Zrm['zu_bc_n'] = np.zeros((Nt,Nz,Nln-1))
    Zrm['zu_bc_s'] = np.zeros((Nt,Nz,Nln-1))
    Zrm['zu_bc_w'] = np.zeros((Nt,Nz,Nlt))
    Zrm['zv_bc_n'] = np.zeros((Nt,Nz,Nln))
    Zrm['zv_bc_s'] = np.zeros((Nt,Nz,Nln))
    Zrm['zv_bc_w'] = np.zeros((Nt,Nz,Nlt-1))

    for aa in range(Nt):
        zeta_r = np.squeeze(zeta[aa,:,:])
        zeta_u = np.squeeze( 0.5 * (zeta[aa,:,0:-1]+zeta[aa,:,1:]) )
        zeta_v = np.squeeze( 0.5 * (zeta[aa,0:-1,:]+zeta[aa,1:,:]) )

        if Vst == 4:
            zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=zeta_r)
            zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=zeta_u)    
            zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=zeta_v)

        Zr = zrom.z_r
        Zu = zrom_u.z_r
        Zv = zrom_v.z_r

        if aa==0:
            Zrm['zr_ic'] = np.squeeze(Zr[0,:,:,:])
            Zrm['zu_ic'] = np.squeeze(Zu[0,:,:,:])
            Zrm['zv_ic'] = np.squeeze(Zv[0,:,:,:])

        Zrm['zr_bc_n'][aa,:,:] = np.squeeze(Zr[0,:,-1,:])
        Zrm['zr_bc_s'][aa,:,:] = np.squeeze(Zr[0,:,0,:])
        Zrm['zr_bc_w'][aa,:,:] = np.squeeze(Zr[0,:,:,0])
        Zrm['zu_bc_n'][aa,:,:] = np.squeeze(Zu[0,:,-1,:])
        Zrm['zu_bc_s'][aa,:,:] = np.squeeze(Zu[0,:,0,:])
        Zrm['zu_bc_w'][aa,:,:] = np.squeeze(Zu[0,:,:,0])
        Zrm['zv_bc_n'][aa,:,:] = np.squeeze(Zv[0,:,-1,:])
        Zrm['zv_bc_s'][aa,:,:] = np.squeeze(Zv[0,:,0,:])
        Zrm['zv_bc_w'][aa,:,:] = np.squeeze(Zv[0,:,:,0])
    
    
    Zrm['Cs_r'] = np.squeeze(zrom.Cs_r)


    with open(fname_out,'wb') as fout:
        pickle.dump(Zrm,fout)
        print('ROMS depths for IC/BC saved with pickle to ' + fname_out)

def make_temp_rom_depth_files_1hrzeta(fname_out):
    
    PFM=get_PFM_info()
    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    del RMG

    fn_temp = PFM['lv1_forc_dir'] + '/tmp_' + 'zeta' + '.pkl'
    with open(fn_temp,'rb') as fp:
        zeta = pickle.load(fp)

    zeta = np.squeeze(zeta[0::3,:,:])

    Nt,Nlt,Nln = np.shape(zeta)
    hraw = None
    Zrm=dict()
    Zrm['zr_ic']   = np.zeros((Nz,Nlt,Nln))
    Zrm['zu_ic']   = np.zeros((Nz,Nlt,Nln-1))
    Zrm['zv_ic']   = np.zeros((Nz,Nlt-1,Nln))
    Zrm['zr_bc_n'] = np.zeros((Nt,Nz,Nln))
    Zrm['zr_bc_s'] = np.zeros((Nt,Nz,Nln))
    Zrm['zr_bc_w'] = np.zeros((Nt,Nz,Nlt))
    Zrm['zu_bc_n'] = np.zeros((Nt,Nz,Nln-1))
    Zrm['zu_bc_s'] = np.zeros((Nt,Nz,Nln-1))
    Zrm['zu_bc_w'] = np.zeros((Nt,Nz,Nlt))
    Zrm['zv_bc_n'] = np.zeros((Nt,Nz,Nln))
    Zrm['zv_bc_s'] = np.zeros((Nt,Nz,Nln))
    Zrm['zv_bc_w'] = np.zeros((Nt,Nz,Nlt-1))

    for aa in range(Nt):
        zeta_r = np.squeeze(zeta[aa,:,:])
        zeta_u = np.squeeze( 0.5 * (zeta[aa,:,0:-1]+zeta[aa,:,1:]) )
        zeta_v = np.squeeze( 0.5 * (zeta[aa,0:-1,:]+zeta[aa,1:,:]) )

        if Vst == 4:
            zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=zeta_r)
            zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=zeta_u)    
            zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=zeta_v)

        Zr = zrom.z_r
        Zu = zrom_u.z_r
        Zv = zrom_v.z_r

        if aa==0:
            Zrm['zr_ic'] = np.squeeze(Zr[0,:,:,:])
            Zrm['zu_ic'] = np.squeeze(Zu[0,:,:,:])
            Zrm['zv_ic'] = np.squeeze(Zv[0,:,:,:])

        Zrm['zr_bc_n'][aa,:,:] = np.squeeze(Zr[0,:,-1,:])
        Zrm['zr_bc_s'][aa,:,:] = np.squeeze(Zr[0,:,0,:])
        Zrm['zr_bc_w'][aa,:,:] = np.squeeze(Zr[0,:,:,0])
        Zrm['zu_bc_n'][aa,:,:] = np.squeeze(Zu[0,:,-1,:])
        Zrm['zu_bc_s'][aa,:,:] = np.squeeze(Zu[0,:,0,:])
        Zrm['zu_bc_w'][aa,:,:] = np.squeeze(Zu[0,:,:,0])
        Zrm['zv_bc_n'][aa,:,:] = np.squeeze(Zv[0,:,-1,:])
        Zrm['zv_bc_s'][aa,:,:] = np.squeeze(Zv[0,:,0,:])
        Zrm['zv_bc_w'][aa,:,:] = np.squeeze(Zv[0,:,:,0])
    
    
    Zrm['Cs_r'] = np.squeeze(zrom.Cs_r)


    with open(fname_out,'wb') as fout:
        pickle.dump(Zrm,fout)
        print('ROMS depths for IC/BC saved with pickle to ' + fname_out)


def get_depth_avg_v(v,z,eta,hb):

    z2 = .5 *( z[0:-1]+z[1:] )
    z2 = np.append(z2,eta)
    z2 = np.insert(z2,0,-hb)
    dz = np.diff(z2)

    vbar = np.sum(v*dz) / (eta+hb)
    return vbar

                #z2 = np.squeeze(zr_vs[aa,:,bb])
                #z3 = np.append(z2,eta_v[aa,0,bb])
                #dz = np.diff(z3)
                #OCN_BC['vbar_south'][aa,bb] = np.sum(vv*dz) / hb_v[0,bb]
                #OCN_BC['vbar_south'][aa,bb] = get_depth_avg_v(vv,eta_v[aa,0,bb],hb_v[0,bb])

def interp_to_roms_z(zh,fofz,zr,hb):

    ig = np.nonzero(zh>=hb)

    # I don't think we ever get into these 1st two conditions, 
    if len(zh[ig]) < 1: # you get in here if all f(z) is nan, ie. we are in land
        # we also make sure that if there is only 1 good value, we also return nans
        fofzrm = np.squeeze(np.nan*zr)
    elif len(zh[ig]) == 1:
        fofzrm = np.squeeze(fofz[ig] * np.ones(np.shape(zr)))
    else:
        if hb > np.min(zh):   # this fills a pesky NaN in OCN_R at the bottom.
            fofz2 = fofz[ig]
            Fz = interp1d(np.squeeze(zh[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[-1],fofz2[0]))
            #print(np.shape(zr_s))
            #print(np.shape(TMP['temp_south']))
            fofzrm = np.squeeze(Fz(zr))
        else:
            fofz2 = fofz
            fofz2[39] = fofz2[38] # this removes the bottom NaN
            Fz = interp1d(np.squeeze(zh[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[-1],fofz2[0]))
            fofzrm = np.squeeze(Fz(zr))


    return fofzrm



def ocn_r_2_BCdict_pckl_new(fname_out):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the boundary condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_BC that has all needed fields 
    # for making the BC.nc file

    PFM=get_PFM_info()   
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
    OCN_R = load_ocnR_from_pckl_files()
    
    fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']

    print('loading ' + fname_depths)
    Zrm = load_rom_depths(fname_depths)

    #print(Zrm.keys())

    OCN_BC = dict()
    # fill in the dict with slicing
    OCN_BC['ocean_time'] = OCN_R['ocean_time']
    Nt = len( OCN_BC['ocean_time'] )
    #OCN_BC['zeta_time'] = OCN_R['zeta_time']
    #Ntz = len( OCN_BC['zeta_time'] )
    OCN_BC['ocean_time_ref'] = OCN_R['ocean_time_ref']


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = OCN_R['depth'] # these are the hycom depths
    zhy = np.squeeze(zhy)
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    OCN_BC['Nz'] = np.squeeze(Nz)
    OCN_BC['Vtr'] = np.squeeze(Vtr)
    OCN_BC['Vst'] = np.squeeze(Vst)
    OCN_BC['th_s'] = np.squeeze(th_s)
    OCN_BC['th_b'] = np.squeeze(th_b)
    OCN_BC['Tcl'] = np.squeeze(Tcl)
    OCN_BC['hc'] = np.squeeze(hc)

    OCN_BC['temp_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_south'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Nt,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_north'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Nt,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['u_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['v_west'] = np.zeros((Nt,Nz,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Nt,nlt))

    OCN_BC['zeta_south'] = np.squeeze(OCN_R['zeta'][:,0,:])
    OCN_BC['zeta_north'] = np.squeeze(OCN_R['zeta'][:,-1,:])
    OCN_BC['zeta_west'] = np.squeeze(OCN_R['zeta'][:,:,0])
    OCN_BC['ubar_south'] = np.squeeze(OCN_R['ubar'][:,0,:])
    OCN_BC['ubar_north'] = np.squeeze(OCN_R['ubar'][:,-1,:])
    OCN_BC['ubar_west'] = np.squeeze(OCN_R['ubar'][:,:,0])
    OCN_BC['vbar_south'] = np.squeeze(OCN_R['vbar'][:,0,:])
    OCN_BC['vbar_north'] = np.squeeze(OCN_R['vbar'][:,-1,:])
    OCN_BC['vbar_west'] = np.squeeze(OCN_R['vbar'][:,:,0])
     
    TMP = dict()
    TMP['temp_north'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_south'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_west'] = np.zeros((Nt,Nz,nlt)) # a helper becasue we convert to potential temp below
    
 
    OCN_BC['vinfo']=dict()
    OCN_BC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_BC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_BC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_BC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_BC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_BC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']

    OCN_BC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_BC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_BC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_BC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_BC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_BC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_BC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    
    OCN_BC['vinfo']['temp_south'] = {'long_name':'ocean potential temperature southern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_south'] = {'long_name':'ocean salinity southern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_south'] = {'long_name':'ocean sea surface height southern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_south'] = {'long_name':'ocean xi velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_south'] = {'long_name':'ocean eta velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_south'] = {'long_name':'ocean xi depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_south'] = {'long_name':'ocean eta depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_north'] = {'long_name':'ocean potential temperature northern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_north'] = {'long_name':'ocean salinity northern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_north'] = {'long_name':'ocean sea surface height northern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_north'] = {'long_name':'ocean xi velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_north'] = {'long_name':'ocean eta velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_north'] = {'long_name':'ocean xi depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_north'] = {'long_name':'ocean eta depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_west'] = {'long_name':'ocean potential temperature western boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_west'] = {'long_name':'ocean salinity western boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_west'] = {'long_name':'ocean sea surface height western boundary',
                        'units':'m',
                        'coordinates':'time,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_west'] = {'long_name':'ocean xi velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_west'] = {'long_name':'ocean eta velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_west'] = {'long_name':'ocean xi depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_west'] = {'long_name':'ocean eta depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_v',
                        'note':'uses roms depths'}

    eta = np.squeeze(OCN_R['zeta'])
    eta_u = 0.5 * (eta[:,:,0:-1]+eta[:,:,1:])
    eta_v = 0.5 * (eta[:,0:-1,:]+eta[:,1:,:])
    
    OCN_BC['Cs_r'] = Zrm['Cs_r']
    OCN_BC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    
    zr_s = Zrm['zr_bc_s']
    zr_n = Zrm['zr_bc_n']
    zr_w = Zrm['zr_bc_w']

    zr_us = Zrm['zu_bc_s']
    zr_un = Zrm['zu_bc_n']
    zr_uw = Zrm['zu_bc_w']

    zr_vs = Zrm['zv_bc_s']
    zr_vn = Zrm['zv_bc_n']
    zr_vw = Zrm['zv_bc_w']
 
    for aa in range(Nt):
        for bb in range(nln):
            TMP['temp_south'][aa,:,bb]    = interp_to_roms_z(-zhy,OCN_R['temp'][aa,:,0,bb],zr_s[aa,:,bb],-hb[0,bb])
            OCN_BC['salt_south'][aa,:,bb] = interp_to_roms_z(-zhy,OCN_R['salt'][aa,:,0,bb],zr_s[aa,:,bb],-hb[0,bb])
            OCN_BC['v_south'][aa,:,bb]    = interp_to_roms_z(-zhy,OCN_R['vrm'][aa,:,0,bb],zr_vs[aa,:,bb],-hb_v[0,bb])
            OCN_BC['vbar_south'][aa,bb]    = get_depth_avg_v(OCN_BC['v_south'][aa,:,bb],zr_vs[aa,:,bb],eta_v[aa,0,bb],hb_v[0,bb])
            
            TMP['temp_north'][aa,:,bb]    = interp_to_roms_z(-zhy,OCN_R['temp'][aa,:,-1,bb],zr_n[aa,:,bb],-hb[-1,bb])
            OCN_BC['salt_north'][aa,:,bb] = interp_to_roms_z(-zhy,OCN_R['salt'][aa,:,-1,bb],zr_n[aa,:,bb],-hb[-1,bb])
            OCN_BC['v_north'][aa,:,bb]    = interp_to_roms_z(-zhy,OCN_R['vrm'][aa,:,-1,bb],zr_vn[aa,:,bb],-hb_v[-1,bb])
            OCN_BC['vbar_north'][aa,bb]    = get_depth_avg_v(OCN_BC['v_north'][aa,:,bb],zr_vn[aa,:,bb],eta_v[aa,-1,bb],hb_v[-1,bb])
                
            if bb < nln-1:
                OCN_BC['u_south'][aa,:,bb]  = interp_to_roms_z(-zhy,OCN_R['urm'][aa,:,0,bb],zr_us[aa,:,bb],-hb_u[0,bb])
                OCN_BC['ubar_south'][aa,bb]  = get_depth_avg_v(OCN_BC['u_south'][aa,:,bb],zr_us[aa,:,bb],eta_u[aa,0,bb],hb_u[0,bb])
                OCN_BC['u_north'][aa,:,bb]  = interp_to_roms_z(-zhy,OCN_R['urm'][aa,:,-1,bb],zr_un[aa,:,bb],-hb_u[-1,bb])
                OCN_BC['ubar_north'][aa,bb]  = get_depth_avg_v(OCN_BC['u_south'][aa,:,bb],zr_un[aa,:,bb],eta_u[aa,-1,bb],hb_u[-1,bb])

        for bb in range(nlt):             
            TMP['temp_west'][aa,:,bb]      = interp_to_roms_z(-zhy,OCN_R['temp'][aa,:,bb,0],zr_w[aa,:,bb],-hb[bb,0])
            OCN_BC['salt_west'][aa,:,bb]   = interp_to_roms_z(-zhy,OCN_R['salt'][aa,:,bb,0],zr_w[aa,:,bb],-hb[bb,0])
            OCN_BC['u_west'][aa,:,bb]      = interp_to_roms_z(-zhy,OCN_R['urm'][aa,:,bb,0],zr_uw[aa,:,bb],-hb_u[bb,0])
            OCN_BC['ubar_west'][aa,bb]      = get_depth_avg_v(OCN_BC['u_west'][aa,:,bb],zr_uw[aa,:,bb],eta_u[aa,bb,0],hb_u[bb,0])
            
            if bb < nlt-1:    
                OCN_BC['v_west'][aa,:,bb]  = interp_to_roms_z(-zhy,OCN_R['vrm'][aa,:,bb,0],zr_vw[aa,:,bb],-hb_v[bb,0])
                OCN_BC['vbar_west'][aa,bb]  = get_depth_avg_v(OCN_BC['v_west'][aa,:,bb],zr_vw[aa,:,bb],eta_v[aa,bb,0],hb_v[bb,0])

    # ROMS wants potential temperature, not temperature, Parker does this in LO.
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr_s # pressure in dbar
    OCN_BC['temp_south'] = seawater.ptmp(np.squeeze(OCN_BC['salt_south']), np.squeeze(TMP['temp_south']),np.squeeze(pdb))  
    pdb = -zr_n # pressure in dbar
    OCN_BC['temp_north'] = seawater.ptmp(np.squeeze(OCN_BC['salt_north']), np.squeeze(TMP['temp_north']),np.squeeze(pdb))  
    pdb = -zr_w # pressure in dbar
    OCN_BC['temp_west'] = seawater.ptmp(np.squeeze(OCN_BC['salt_west']), np.squeeze(TMP['temp_west']),np.squeeze(pdb))  

    with open(fname_out,'wb') as fout:
        pickle.dump(OCN_BC,fout)
        print('OCN_BC dict saved with pickle')


def ocn_r_2_BCdict_pckl_new_1hrzeta(fname_out):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the boundary condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_BC that has all needed fields 
    # for making the BC.nc file

    PFM=get_PFM_info()   
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
    OCN_R = load_ocnR_from_pckl_files()
    
    fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']

    print('loading ' + fname_depths)
    Zrm = load_rom_depths(fname_depths)

    #print(Zrm.keys())

    OCN_BC = dict()
    # fill in the dict with slicing
    OCN_BC['ocean_time'] = OCN_R['ocean_time']
    Nt = len( OCN_BC['ocean_time'] )
    OCN_BC['zeta_time'] = OCN_R['zeta_time']
    Ntz = len( OCN_BC['zeta_time'] )
    OCN_BC['ocean_time_ref'] = OCN_R['ocean_time_ref']


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = OCN_R['depth'] # these are the hycom depths
    zhy = np.squeeze(zhy)
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    OCN_BC['Nz'] = np.squeeze(Nz)
    OCN_BC['Vtr'] = np.squeeze(Vtr)
    OCN_BC['Vst'] = np.squeeze(Vst)
    OCN_BC['th_s'] = np.squeeze(th_s)
    OCN_BC['th_b'] = np.squeeze(th_b)
    OCN_BC['Tcl'] = np.squeeze(Tcl)
    OCN_BC['hc'] = np.squeeze(hc)

    OCN_BC['temp_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_south'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Ntz,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_north'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Ntz,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['u_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['v_west'] = np.zeros((Nt,Nz,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Ntz,nlt))

    OCN_BC['zeta_south'] = np.squeeze(OCN_R['zeta'][:,0,:])
    OCN_BC['zeta_north'] = np.squeeze(OCN_R['zeta'][:,-1,:])
    OCN_BC['zeta_west'] = np.squeeze(OCN_R['zeta'][:,:,0])
    OCN_BC['ubar_south'] = np.squeeze(OCN_R['ubar'][:,0,:])
    OCN_BC['ubar_north'] = np.squeeze(OCN_R['ubar'][:,-1,:])
    OCN_BC['ubar_west'] = np.squeeze(OCN_R['ubar'][:,:,0])
    OCN_BC['vbar_south'] = np.squeeze(OCN_R['vbar'][:,0,:])
    OCN_BC['vbar_north'] = np.squeeze(OCN_R['vbar'][:,-1,:])
    OCN_BC['vbar_west'] = np.squeeze(OCN_R['vbar'][:,:,0])
     
    TMP = dict()
    TMP['temp_north'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_south'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_west'] = np.zeros((Nt,Nz,nlt)) # a helper becasue we convert to potential temp below
    
 
    OCN_BC['vinfo']=dict()
    OCN_BC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_BC['vinfo']['zeta_time'] = OCN_R['vinfo']['zeta_time']
    OCN_BC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_BC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_BC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_BC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_BC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']

    OCN_BC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_BC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_BC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_BC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_BC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_BC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_BC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    
    OCN_BC['vinfo']['temp_south'] = {'long_name':'ocean potential temperature southern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_south'] = {'long_name':'ocean salinity southern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_south'] = {'long_name':'ocean sea surface height southern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_south'] = {'long_name':'ocean xi velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_south'] = {'long_name':'ocean eta velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_south'] = {'long_name':'ocean xi depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_south'] = {'long_name':'ocean eta depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_north'] = {'long_name':'ocean potential temperature northern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_north'] = {'long_name':'ocean salinity northern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_north'] = {'long_name':'ocean sea surface height northern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_north'] = {'long_name':'ocean xi velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_north'] = {'long_name':'ocean eta velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_north'] = {'long_name':'ocean xi depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_north'] = {'long_name':'ocean eta depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_west'] = {'long_name':'ocean potential temperature western boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_west'] = {'long_name':'ocean salinity western boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_west'] = {'long_name':'ocean sea surface height western boundary',
                        'units':'m',
                        'coordinates':'time,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_west'] = {'long_name':'ocean xi velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_west'] = {'long_name':'ocean eta velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_west'] = {'long_name':'ocean xi depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_west'] = {'long_name':'ocean eta depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_v',
                        'note':'uses roms depths'}

    eta = np.squeeze(OCN_R['zeta'][0::3,:,:])
    eta_u = 0.5 * (eta[:,:,0:-1]+eta[:,:,1:])
    eta_v = 0.5 * (eta[:,0:-1,:]+eta[:,1:,:])
    
    OCN_BC['Cs_r'] = Zrm['Cs_r']
    OCN_BC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    
    zr_s = Zrm['zr_bc_s']
    zr_n = Zrm['zr_bc_n']
    zr_w = Zrm['zr_bc_w']

    zr_us = Zrm['zu_bc_s']
    zr_un = Zrm['zu_bc_n']
    zr_uw = Zrm['zu_bc_w']

    zr_vs = Zrm['zv_bc_s']
    zr_vn = Zrm['zv_bc_n']
    zr_vw = Zrm['zv_bc_w']
 
    for aa in range(Nt):
        for bb in range(nln):
            TMP['temp_south'][aa,:,bb]    = interp_to_roms_z(-zhy,OCN_R['temp'][aa,:,0,bb],zr_s[aa,:,bb],-hb[0,bb])
            OCN_BC['salt_south'][aa,:,bb] = interp_to_roms_z(-zhy,OCN_R['salt'][aa,:,0,bb],zr_s[aa,:,bb],-hb[0,bb])
            OCN_BC['v_south'][aa,:,bb]    = interp_to_roms_z(-zhy,OCN_R['vrm'][aa,:,0,bb],zr_vs[aa,:,bb],-hb_v[0,bb])
            OCN_BC['vbar_south'][aa,bb]    = get_depth_avg_v(OCN_BC['v_south'][aa,:,bb],zr_vs[aa,:,bb],eta_v[aa,0,bb],hb_v[0,bb])
            
            TMP['temp_north'][aa,:,bb]    = interp_to_roms_z(-zhy,OCN_R['temp'][aa,:,-1,bb],zr_n[aa,:,bb],-hb[-1,bb])
            OCN_BC['salt_north'][aa,:,bb] = interp_to_roms_z(-zhy,OCN_R['salt'][aa,:,-1,bb],zr_n[aa,:,bb],-hb[-1,bb])
            OCN_BC['v_north'][aa,:,bb]    = interp_to_roms_z(-zhy,OCN_R['vrm'][aa,:,-1,bb],zr_vn[aa,:,bb],-hb_v[-1,bb])
            OCN_BC['vbar_north'][aa,bb]    = get_depth_avg_v(OCN_BC['v_north'][aa,:,bb],zr_vn[aa,:,bb],eta_v[aa,-1,bb],hb_v[-1,bb])
                
            if bb < nln-1:
                OCN_BC['u_south'][aa,:,bb]  = interp_to_roms_z(-zhy,OCN_R['urm'][aa,:,0,bb],zr_us[aa,:,bb],-hb_u[0,bb])
                OCN_BC['ubar_south'][aa,bb]  = get_depth_avg_v(OCN_BC['u_south'][aa,:,bb],zr_us[aa,:,bb],eta_u[aa,0,bb],hb_u[0,bb])
                OCN_BC['u_north'][aa,:,bb]  = interp_to_roms_z(-zhy,OCN_R['urm'][aa,:,-1,bb],zr_un[aa,:,bb],-hb_u[-1,bb])
                OCN_BC['ubar_north'][aa,bb]  = get_depth_avg_v(OCN_BC['u_south'][aa,:,bb],zr_un[aa,:,bb],eta_u[aa,-1,bb],hb_u[-1,bb])

        for bb in range(nlt):             
            TMP['temp_west'][aa,:,bb]      = interp_to_roms_z(-zhy,OCN_R['temp'][aa,:,bb,0],zr_w[aa,:,bb],-hb[bb,0])
            OCN_BC['salt_west'][aa,:,bb]   = interp_to_roms_z(-zhy,OCN_R['salt'][aa,:,bb,0],zr_w[aa,:,bb],-hb[bb,0])
            OCN_BC['u_west'][aa,:,bb]      = interp_to_roms_z(-zhy,OCN_R['urm'][aa,:,bb,0],zr_uw[aa,:,bb],-hb_u[bb,0])
            OCN_BC['ubar_west'][aa,bb]      = get_depth_avg_v(OCN_BC['u_west'][aa,:,bb],zr_uw[aa,:,bb],eta_u[aa,bb,0],hb_u[bb,0])
            
            if bb < nlt-1:    
                OCN_BC['v_west'][aa,:,bb]  = interp_to_roms_z(-zhy,OCN_R['vrm'][aa,:,bb,0],zr_vw[aa,:,bb],-hb_v[bb,0])
                OCN_BC['vbar_west'][aa,bb]  = get_depth_avg_v(OCN_BC['v_west'][aa,:,bb],zr_vw[aa,:,bb],eta_v[aa,bb,0],hb_v[bb,0])

    # ROMS wants potential temperature, not temperature, Parker does this in LO.
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr_s # pressure in dbar
    OCN_BC['temp_south'] = seawater.ptmp(np.squeeze(OCN_BC['salt_south']), np.squeeze(TMP['temp_south']),np.squeeze(pdb))  
    pdb = -zr_n # pressure in dbar
    OCN_BC['temp_north'] = seawater.ptmp(np.squeeze(OCN_BC['salt_north']), np.squeeze(TMP['temp_north']),np.squeeze(pdb))  
    pdb = -zr_w # pressure in dbar
    OCN_BC['temp_west'] = seawater.ptmp(np.squeeze(OCN_BC['salt_west']), np.squeeze(TMP['temp_west']),np.squeeze(pdb))  

    with open(fname_out,'wb') as fout:
        pickle.dump(OCN_BC,fout)
        print('OCN_BC dict saved with pickle')


def ocnr_2_BCdict_1hrzeta_from_tmppkls(fname_out):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the boundary condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_BC that has all needed fields 
    # for making the BC.nc file

    PFM=get_PFM_info()   
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
#    OCN_R = load_ocnR_from_pckl_files()
    
    fname_depths = PFM['lv1_forc_dir'] + '/' + PFM['lv1_depth_file']

    print('loading ' + fname_depths)
    Zrm = load_rom_depths(fname_depths)

    #print(Zrm.keys())

    OCN_BC = dict()
    # fill in the dict with slicing
    OCN_BC['ocean_time'] = load_tmp_pkl('ocean_time')
    Nt = len( OCN_BC['ocean_time'] )
    OCN_BC['zeta_time'] = load_tmp_pkl('zeta_time')
    Ntz = len( OCN_BC['zeta_time'] )
    #OCN_BC['ocean_time_ref'] = OCN_R['ocean_time_ref']
    OCN_BC['ocean_time_ref'] = load_tmp_pkl('ocean_time_ref')


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = load_tmp_pkl('depth') # these are the hycom depths
    zhy = np.squeeze(zhy)
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    OCN_BC['Nz'] = np.squeeze(Nz)
    OCN_BC['Vtr'] = np.squeeze(Vtr)
    OCN_BC['Vst'] = np.squeeze(Vst)
    OCN_BC['th_s'] = np.squeeze(th_s)
    OCN_BC['th_b'] = np.squeeze(th_b)
    OCN_BC['Tcl'] = np.squeeze(Tcl)
    OCN_BC['hc'] = np.squeeze(hc)

    OCN_BC['temp_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_south'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Ntz,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_north'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Ntz,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['u_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['v_west'] = np.zeros((Nt,Nz,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Ntz,nlt))

    OCN_BC['zeta_south'] = np.squeeze(load_tmp_pkl('zeta')[:,0,:])
    OCN_BC['zeta_north'] = np.squeeze(load_tmp_pkl('zeta')[:,-1,:])
    OCN_BC['zeta_west'] = np.squeeze(load_tmp_pkl('zeta')[:,:,0])
    OCN_BC['ubar_south'] = np.squeeze(load_tmp_pkl('ubar')[:,0,:])
    OCN_BC['ubar_north'] = np.squeeze(load_tmp_pkl('ubar')[:,-1,:])
    OCN_BC['ubar_west'] = np.squeeze(load_tmp_pkl('ubar')[:,:,0])
    OCN_BC['vbar_south'] = np.squeeze(load_tmp_pkl('vbar')[:,0,:])
    OCN_BC['vbar_north'] = np.squeeze(load_tmp_pkl('vbar')[:,-1,:])
    OCN_BC['vbar_west'] = np.squeeze(load_tmp_pkl('vbar')[:,:,0])
     
    TMP = dict()
    TMP['temp_north'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_south'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_west'] = np.zeros((Nt,Nz,nlt)) # a helper becasue we convert to potential temp below
    
 
    OCN_BC['vinfo']=dict()
    tmp = load_tmp_pkl('vinfo') 
    OCN_BC['vinfo']['ocean_time'] = tmp['ocean_time']
    OCN_BC['vinfo']['zeta_time'] = tmp['zeta_time']
    OCN_BC['vinfo']['ocean_time_ref'] = tmp['ocean_time_ref']
    OCN_BC['vinfo']['lat_rho'] = tmp['lat_rho']
    OCN_BC['vinfo']['lon_rho'] = tmp['lat_rho']
    OCN_BC['vinfo']['lat_u'] = tmp['lat_u']
    OCN_BC['vinfo']['lon_u'] = tmp['lon_u']
    OCN_BC['vinfo']['lat_v'] = tmp['lat_v']
    OCN_BC['vinfo']['lon_v'] = tmp['lon_v']

    OCN_BC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_BC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_BC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_BC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_BC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_BC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_BC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    
    OCN_BC['vinfo']['temp_south'] = {'long_name':'ocean potential temperature southern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_south'] = {'long_name':'ocean salinity southern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_south'] = {'long_name':'ocean sea surface height southern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_south'] = {'long_name':'ocean xi velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_south'] = {'long_name':'ocean eta velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_south'] = {'long_name':'ocean xi depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_south'] = {'long_name':'ocean eta depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_north'] = {'long_name':'ocean potential temperature northern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_north'] = {'long_name':'ocean salinity northern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_north'] = {'long_name':'ocean sea surface height northern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_north'] = {'long_name':'ocean xi velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_north'] = {'long_name':'ocean eta velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_north'] = {'long_name':'ocean xi depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_north'] = {'long_name':'ocean eta depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_west'] = {'long_name':'ocean potential temperature western boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_west'] = {'long_name':'ocean salinity western boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_west'] = {'long_name':'ocean sea surface height western boundary',
                        'units':'m',
                        'coordinates':'time,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_west'] = {'long_name':'ocean xi velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_west'] = {'long_name':'ocean eta velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_west'] = {'long_name':'ocean xi depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_west'] = {'long_name':'ocean eta depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_v',
                        'note':'uses roms depths'}

    eta = np.squeeze(load_tmp_pkl('zeta')[0::3,:,:])
    eta_u = 0.5 * (eta[:,:,0:-1]+eta[:,:,1:])
    eta_v = 0.5 * (eta[:,0:-1,:]+eta[:,1:,:])
    
    OCN_BC['Cs_r'] = Zrm['Cs_r']
    OCN_BC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    
    zr_s = Zrm['zr_bc_s']
    zr_n = Zrm['zr_bc_n']
    zr_w = Zrm['zr_bc_w']

    zr_us = Zrm['zu_bc_s']
    zr_un = Zrm['zu_bc_n']
    zr_uw = Zrm['zu_bc_w']

    zr_vs = Zrm['zv_bc_s']
    zr_vn = Zrm['zv_bc_n']
    zr_vw = Zrm['zv_bc_w']
 
    tmp = load_tmp_pkl('temp')
    for aa in range(Nt):
        for bb in range(nln):
            TMP['temp_south'][aa,:,bb]    = interp_to_roms_z(-zhy,tmp[aa,:,0,bb],zr_s[aa,:,bb],-hb[0,bb])
            TMP['temp_north'][aa,:,bb]    = interp_to_roms_z(-zhy,tmp[aa,:,-1,bb],zr_n[aa,:,bb],-hb[-1,bb])
        
        for bb in range(nlt):             
            TMP['temp_west'][aa,:,bb]      = interp_to_roms_z(-zhy,tmp[aa,:,bb,0],zr_w[aa,:,bb],-hb[bb,0])

    del tmp    
    tmp = load_tmp_pkl('salt')
    for aa in range(Nt):
        for bb in range(nln):
            OCN_BC['salt_south'][aa,:,bb] = interp_to_roms_z(-zhy,tmp[aa,:,0,bb],zr_s[aa,:,bb],-hb[0,bb])
            OCN_BC['salt_north'][aa,:,bb] = interp_to_roms_z(-zhy,tmp[aa,:,-1,bb],zr_n[aa,:,bb],-hb[-1,bb])

        for bb in range(nlt):             
            OCN_BC['salt_west'][aa,:,bb]   = interp_to_roms_z(-zhy,tmp[aa,:,bb,0],zr_w[aa,:,bb],-hb[bb,0])

    del tmp
    tmp = load_tmp_pkl('vrm')
    for aa in range(Nt):
        for bb in range(nln):
            OCN_BC['v_south'][aa,:,bb]    = interp_to_roms_z(-zhy,tmp[aa,:,0,bb],zr_vs[aa,:,bb],-hb_v[0,bb])
            OCN_BC['vbar_south'][aa,bb]    = get_depth_avg_v(OCN_BC['v_south'][aa,:,bb],zr_vs[aa,:,bb],eta_v[aa,0,bb],hb_v[0,bb])       
            OCN_BC['v_north'][aa,:,bb]    = interp_to_roms_z(-zhy,tmp[aa,:,-1,bb],zr_vn[aa,:,bb],-hb_v[-1,bb])
            OCN_BC['vbar_north'][aa,bb]    = get_depth_avg_v(OCN_BC['v_north'][aa,:,bb],zr_vn[aa,:,bb],eta_v[aa,-1,bb],hb_v[-1,bb])

        for bb in range(nlt-1):             
            OCN_BC['v_west'][aa,:,bb]  = interp_to_roms_z(-zhy,tmp[aa,:,bb,0],zr_vw[aa,:,bb],-hb_v[bb,0])
            OCN_BC['vbar_west'][aa,bb]  = get_depth_avg_v(OCN_BC['v_west'][aa,:,bb],zr_vw[aa,:,bb],eta_v[aa,bb,0],hb_v[bb,0])

    del tmp
    tmp = load_tmp_pkl('urm')
    for aa in range(Nt):
        for bb in range(nln-1):
            OCN_BC['u_south'][aa,:,bb]  = interp_to_roms_z(-zhy,tmp[aa,:,0,bb],zr_us[aa,:,bb],-hb_u[0,bb])
            OCN_BC['ubar_south'][aa,bb]  = get_depth_avg_v(OCN_BC['u_south'][aa,:,bb],zr_us[aa,:,bb],eta_u[aa,0,bb],hb_u[0,bb])
            OCN_BC['u_north'][aa,:,bb]  = interp_to_roms_z(-zhy,tmp[aa,:,-1,bb],zr_un[aa,:,bb],-hb_u[-1,bb])
            OCN_BC['ubar_north'][aa,bb]  = get_depth_avg_v(OCN_BC['u_south'][aa,:,bb],zr_un[aa,:,bb],eta_u[aa,-1,bb],hb_u[-1,bb])

        for bb in range(nlt):             
            OCN_BC['u_west'][aa,:,bb]      = interp_to_roms_z(-zhy,tmp[aa,:,bb,0],zr_uw[aa,:,bb],-hb_u[bb,0])
            OCN_BC['ubar_west'][aa,bb]      = get_depth_avg_v(OCN_BC['u_west'][aa,:,bb],zr_uw[aa,:,bb],eta_u[aa,bb,0],hb_u[bb,0])
            
    del tmp
    
    # ROMS wants potential temperature, not temperature, Parker does this in LO.
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr_s # pressure in dbar
    OCN_BC['temp_south'] = seawater.ptmp(np.squeeze(OCN_BC['salt_south']), np.squeeze(TMP['temp_south']),np.squeeze(pdb))  
    pdb = -zr_n # pressure in dbar
    OCN_BC['temp_north'] = seawater.ptmp(np.squeeze(OCN_BC['salt_north']), np.squeeze(TMP['temp_north']),np.squeeze(pdb))  
    pdb = -zr_w # pressure in dbar
    OCN_BC['temp_west'] = seawater.ptmp(np.squeeze(OCN_BC['salt_west']), np.squeeze(TMP['temp_west']),np.squeeze(pdb))  

    with open(fname_out,'wb') as fout:
        pickle.dump(OCN_BC,fout)
        print('OCN_BC dict saved with pickle')


def ocn_r_2_BCdict_pckl(fname_out):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the boundary condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_BC that has all needed fields 
    # for making the BC.nc file

    PFM=get_PFM_info()    
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
    OCN_R = load_ocnR_from_pckl_files()


    OCN_BC = dict()
    # fill in the dict with slicing
    OCN_BC['ocean_time'] = OCN_R['ocean_time']
    Nt = len( OCN_BC['ocean_time'] )
    OCN_BC['ocean_time_ref'] = OCN_R['ocean_time_ref']

#    OCN_BC['Nz'] = np.squeeze(RMG['Nz'])
#    OCN_BC['Vtr'] = np.squeeze(RMG['Vtransform'])
#    OCN_BC['Vst'] = np.squeeze(RMG['Vstretching'])
#    OCN_BC['th_s'] = np.squeeze(RMG['THETA_S'])
#    OCN_BC['th_b'] = np.squeeze(RMG['THETA_B'])
#    OCN_BC['Tcl'] = np.squeeze(RMG['TCLINE'])
#    OCN_BC['hc'] = np.squeeze(RMG['hc'])


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = OCN_R['depth'] # these are the hycom depths
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

#    Nz   = RMG['Nz']                              # number of vertical levels: 40
#    Vtr  = RMG['Vtransform']                       # transformation equation: 2
#    Vst  = RMG['Vstretching']                    # stretching function: 4 
#    th_s = RMG['THETA_S']                      # surface stretching parameter: 8
#    th_b = RMG['THETA_B']                      # bottom  stretching parameter: 3
#    Tcl  = RMG['TCLINE']                      # critical depth (m): 50

    Nz   = PFM['stretching']['L1','Nz']                              # number of vertical levels: 40
    Vtr  = PFM['stretching']['L1','Vtransform']                       # transformation equation: 2
    Vst  = PFM['stretching']['L1','Vstretching']                    # stretching function: 4 
    th_s = PFM['stretching']['L1','THETA_S']                      # surface stretching parameter: 8
    th_b = PFM['stretching']['L1','THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = PFM['stretching']['L1','TCLINE']                      # critical depth (m): 50
    hc   = PFM['stretching']['L1','hc']

    OCN_BC['Nz'] = np.squeeze(Nz)
    OCN_BC['Vtr'] = np.squeeze(Vtr)
    OCN_BC['Vst'] = np.squeeze(Vst)
    OCN_BC['th_s'] = np.squeeze(th_s)
    OCN_BC['th_b'] = np.squeeze(th_b)
    OCN_BC['Tcl'] = np.squeeze(Tcl)
    OCN_BC['hc'] = np.squeeze(hc)


    eta = np.squeeze(OCN_R['zeta'].copy())
    eta_u = 0.5 * (eta[:,0:-1]+eta[:,1:])
    eta_v = 0.5 * (eta[0:-1,:]+eta[1:,:])

    OCN_BC['temp_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_south'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Nt,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_north'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Nt,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['u_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['v_west'] = np.zeros((Nt,Nz,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Nt,nlt))

    OCN_BC['zeta_south'] = np.squeeze(OCN_R['zeta'][:,0,:])
    OCN_BC['zeta_north'] = np.squeeze(OCN_R['zeta'][:,-1,:])
    OCN_BC['zeta_west'] = np.squeeze(OCN_R['zeta'][:,:,0])
    OCN_BC['ubar_south'] = np.squeeze(OCN_R['ubar'][:,0,:])
    OCN_BC['ubar_north'] = np.squeeze(OCN_R['ubar'][:,-1,:])
    OCN_BC['ubar_west'] = np.squeeze(OCN_R['ubar'][:,:,0])
    OCN_BC['vbar_south'] = np.squeeze(OCN_R['vbar'][:,0,:])
    OCN_BC['vbar_north'] = np.squeeze(OCN_R['vbar'][:,-1,:])
    OCN_BC['vbar_west'] = np.squeeze(OCN_R['vbar'][:,:,0])
     
    TMP = dict()
    TMP['temp_north'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_south'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_west'] = np.zeros((Nt,Nz,nlt)) # a helper becasue we convert to potential temp below
    
 
    OCN_BC['vinfo']=dict()
    OCN_BC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_BC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_BC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_BC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_BC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_BC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']

    OCN_BC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_BC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_BC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_BC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_BC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_BC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_BC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    
    OCN_BC['vinfo']['temp_south'] = {'long_name':'ocean potential temperature southern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_south'] = {'long_name':'ocean salinity southern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_south'] = {'long_name':'ocean sea surface height southern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_south'] = {'long_name':'ocean xi velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_south'] = {'long_name':'ocean eta velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_south'] = {'long_name':'ocean xi depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_south'] = {'long_name':'ocean eta depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_north'] = {'long_name':'ocean potential temperature northern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_north'] = {'long_name':'ocean salinity northern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_north'] = {'long_name':'ocean sea surface height northern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_north'] = {'long_name':'ocean xi velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_north'] = {'long_name':'ocean eta velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_north'] = {'long_name':'ocean xi depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_north'] = {'long_name':'ocean eta depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_west'] = {'long_name':'ocean potential temperature western boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_west'] = {'long_name':'ocean salinity western boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_west'] = {'long_name':'ocean sea surface height western boundary',
                        'units':'m',
                        'coordinates':'time,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_west'] = {'long_name':'ocean xi velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_west'] = {'long_name':'ocean eta velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_west'] = {'long_name':'ocean xi depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_west'] = {'long_name':'ocean eta depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_v',
                        'note':'uses roms depths'}

    eta = np.squeeze(OCN_R['zeta'])
    eta_u = 0.5 * (eta[:,:,0:-1]+eta[:,:,1:])
    eta_v = 0.5 * (eta[:,0:-1,:]+eta[:,1:,:])

    print('got here 1')
    # get the roms z's
    hraw = None
    if Vst == 4:
        zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta))
        #zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta_u))
        #zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta_v))
    
    OCN_BC['Cs_r'] = np.squeeze(zrom.Cs_r)
    OCN_BC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    
    # do we need sc_r ???
    #OCN_IC['sc_r'] = []
    #OCN_IC['vinfo']['sc_r'] = {'long_name':'S-coordinate at RHO-points',
    #                    'units':'nondimensional',
    #                    'valid min':'-1',
    #                    'valid max':'0',
    #                    'field':'sc_r, scalar, series'}


    zr_s=np.squeeze(zrom.z_r[:,:,0,:])    
    zr_n=np.squeeze(zrom.z_r[:,:,-1,:])    
    zr_w=np.squeeze(zrom.z_r[:,:,:,0])    

    del zrom
    gc.collect()


    zr_us = .5 * (zr_s[:,:,0:-1]+zr_s[:,:,1:])
    zr_un = .5 * (zr_n[:,:,0:-1]+zr_n[:,:,1:])
    zr_uw = zr_w

    #zr_us=np.squeeze(zrom_u.z_r[:,:,0,:])    
    #zr_un=np.squeeze(zrom_u.z_r[:,:,-1,:])    
    #zr_uw=np.squeeze(zrom_u.z_r[:,:,:,0])    

    #del zrom_u
    #gc.collect()


    zr_vs = zr_w
    zr_vn = zr_n
    zr_vw = .5 * (zr_w[:,:,0:-1]+zr_w[:,:,1:])

    #zr_vs=np.squeeze(zrom_v.z_r[:,:,0,:])
    #zr_vn=np.squeeze(zrom_v.z_r[:,:,-1,:])
    #zr_vw=np.squeeze(zrom_v.z_r[:,:,:,0])

    #del zrom_v
    #gc.collect()






    for aa in range(Nt):
        for bb in range(nln):
            fofz = np.squeeze(OCN_R['temp'][aa,:,0,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_south'][aa,:,bb] = np.squeeze(np.nan*zr_s[aa,:,bb])
                OCN_BC['salt_south'][aa,:,bb] = np.squeeze(np.nan*zr_s[aa,:,bb])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                #print(np.shape(zr_s))
                #print(np.shape(TMP['temp_south']))
                TMP['temp_south'][aa,:,bb] = np.squeeze(Fz(zr_s[aa,:,bb]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,0,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_south'][aa,:,bb] = np.squeeze(Fz(zr_s[aa,:,bb]))
                
            fofz = np.squeeze(OCN_R['vrm'][aa,:,0,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['v_south'][aa,:,bb] = np.squeeze(np.nan*zr_vs[aa,:,bb])
                OCN_BC['vbar_south'][aa,bb] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                vv =  np.squeeze(Fz(zr_vs[aa,:,bb]))                
                OCN_BC['v_south'][aa,:,bb] = vv
                z2 = np.squeeze(zr_vs[aa,:,bb])
                z3 = np.append(z2,eta_v[aa,0,bb])
                dz = np.diff(z3)
                OCN_BC['vbar_south'][aa,bb] = np.sum(vv*dz) / hb_v[0,bb]

            if bb < nln-1:
                fofz = np.squeeze(OCN_R['urm'][aa,:,0,bb])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['u_south'][aa,:,bb] = np.squeeze(np.nan*zr_us[aa,:,bb])
                    OCN_BC['ubar_south'][aa,bb] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    uu =  np.squeeze(Fz(zr_us[aa,:,bb]))                
                    OCN_BC['u_south'][aa,:,bb] = uu
                    z2 = np.squeeze(zr_us[aa,:,bb])
                    z3 = np.append(z2,eta_u[aa,0,bb])
                    dz = np.diff(z3)
                    OCN_BC['ubar_south'][aa,bb] = np.sum(uu*dz) / hb_u[0,bb]

            fofz = np.squeeze(OCN_R['temp'][aa,:,-1,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_north'][aa,:,bb] = np.squeeze(np.nan*zr_n[aa,:,bb])
                OCN_BC['salt_north'][aa,:,bb] = np.squeeze(np.nan*zr_n[aa,:,bb])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp_north'][aa,:,bb] = np.squeeze(Fz(zr_n[aa,:,bb]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,-1,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_north'][aa,:,bb] = np.squeeze(Fz(zr_n[aa,:,bb]))
                
            fofz = np.squeeze(OCN_R['vrm'][aa,:,-1,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['v_north'][aa,:,bb] = np.squeeze(np.nan*zr_vn[aa,:,bb])
                OCN_BC['vbar_north'][aa,bb] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                vv =  np.squeeze(Fz(zr_vn[aa,:,bb]))                
                OCN_BC['v_north'][aa,:,bb] = vv
                z2 = np.squeeze(zr_vn[aa,:,bb])
                z3 = np.append(z2,eta_v[aa,-1,bb])
                dz = np.diff(z3)
                OCN_BC['vbar_north'][aa,bb] = np.sum(vv*dz) / hb_v[-1,bb]

            if bb < nln-1:
                fofz = np.squeeze(OCN_R['urm'][aa,:,-1,bb])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['u_north'][aa,:,bb] = np.squeeze(np.nan*zr_un[aa,:,bb])
                    OCN_BC['ubar_north'][aa,bb] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    uu =  np.squeeze(Fz(zr_un[aa,:,bb]))                
                    OCN_BC['u_north'][aa,:,bb] = uu
                    z2 = np.squeeze(zr_un[aa,:,bb])
                    z3 = np.append(z2,eta_u[aa,0,bb])
                    dz = np.diff(z3)
                    OCN_BC['ubar_north'][aa,bb] = np.sum(uu*dz) / hb_u[-1,bb]

        for cc in range(nlt):
            fofz = np.squeeze(OCN_R['temp'][aa,:,cc,0])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_west'][aa,:,cc] = np.squeeze(np.nan*zr_w[aa,:,cc])
                OCN_BC['salt_west'][aa,:,cc] = np.squeeze(np.nan*zr_w[aa,:,cc])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp_west'][aa,:,cc] = np.squeeze(Fz(zr_w[aa,:,cc]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,cc,0])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_west'][aa,:,cc] = np.squeeze(Fz(zr_w[aa,:,cc]))

            if cc < nlt-1:    
                fofz = np.squeeze(OCN_R['vrm'][aa,:,cc,0])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['v_west'][aa,:,cc] = np.squeeze(np.nan*zr_vw[aa,:,cc])
                    OCN_BC['vbar_west'][aa,cc] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    vv =  np.squeeze(Fz(zr_vw[aa,:,cc]))                
                    OCN_BC['v_west'][aa,:,cc] = vv
                    z2 = np.squeeze(zr_vw[aa,:,cc])
                    z3 = np.append(z2,eta_v[aa,cc,0])
                    dz = np.diff(z3)
                    OCN_BC['vbar_west'][aa,cc] = np.sum(vv*dz) / hb_v[cc,0]

            fofz = np.squeeze(OCN_R['urm'][aa,:,cc,0])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['u_west'][aa,:,cc] = np.squeeze(np.nan*zr_uw[aa,:,cc])
                OCN_BC['ubar_west'][aa,cc] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                uu =  np.squeeze(Fz(zr_uw[aa,:,cc]))                
                OCN_BC['u_west'][aa,:,cc] = uu
                z2 = np.squeeze(zr_uw[aa,:,cc])
                z3 = np.append(z2,eta_u[aa,cc,0])
                dz = np.diff(z3)
                OCN_BC['ubar_west'][aa,cc] = np.sum(uu*dz) / hb_u[cc,0]

    print('got here 3')


    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr_s # pressure in dbar
    OCN_BC['temp_south'] = seawater.ptmp(np.squeeze(OCN_BC['salt_south']), np.squeeze(TMP['temp_south']),np.squeeze(pdb))  
    pdb = -zr_n # pressure in dbar
    OCN_BC['temp_north'] = seawater.ptmp(np.squeeze(OCN_BC['salt_north']), np.squeeze(TMP['temp_north']),np.squeeze(pdb))  
    pdb = -zr_w # pressure in dbar
    OCN_BC['temp_west'] = seawater.ptmp(np.squeeze(OCN_BC['salt_west']), np.squeeze(TMP['temp_west']),np.squeeze(pdb))  

    with open(fname_out,'wb') as fout:
        pickle.dump(OCN_BC,fout)
        print('OCN_BC dict saved with pickle')

def get_child_xi_eta_interp(ln1,lt1,ln2,lt2):
    # this function returns the index values of (lat,lon) in grid 2 on grid 1
    # this way we can ruse egular grid interpolator
    # this function also returns the regular grid interpolator object
    M, L = np.shape(ln1) 
    Xind, Yind = np.meshgrid(np.arange(L),np.arange(M))
    points1 = np.zeros( (M*L, 2) )
    points1[:,1] = ln1.flatten()
    points1[:,0] = lt1.flatten()
    scat_interp_xi  = LinearNDInterpolator(points1,Xind.flatten())
    scat_interp_eta = LinearNDInterpolator(points1,Yind.flatten())
    xi_2  = scat_interp_xi(lt2,ln2)
    eta_2 = scat_interp_eta(lt2,ln2)

    interper = RegularGridInterpolator( (np.arange(M), np.arange(L)), lt1 , bounds_error=False, fill_value=None)

    return xi_2, eta_2, interper

def get_indices_to_fill(gr_msk):
        M, L = np.shape(gr_msk)
        X, Y = np.meshgrid(np.arange(L),np.arange(M))

        xyorig = np.array((X[gr_msk==1],Y[gr_msk==1])).T
        xynew = np.array((X[gr_msk==0],Y[gr_msk==0])).T
        a = cKDTree(xyorig).query(xynew)
        aa = a[1]

        return aa 

def mk_LV2_BC_dict(lvl):

    PFM=get_PFM_info()  
    if lvl == '2':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        fn = PFM['lv1_his_name_full']
        LV1_BC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']
        lv1 = 'L1'
        lv2 = 'L2'
        fn_out = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnBC_tmp_pckl_file']
    elif lvl == '3':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        fn = PFM['lv2_his_name_full']
        LV1_BC_pckl = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnBC_tmp_pckl_file']
        lv1 = 'L2'
        lv2 = 'L3'
        fn_out = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnBC_tmp_pckl_file']
    elif lvl == '4':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
        fn = PFM['lv3_his_name_full']
        LV1_BC_pckl = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnBC_tmp_pckl_file']
        lv1 = 'L3'
        lv2 = 'L4'
        fn_out = PFM['lv4_forc_dir'] + '/' + PFM['lv4_ocnBC_tmp_pckl_file']
     
    # parent vertical stretching info 
    Nz1   = PFM['stretching'][lv1,'Nz']                              # number of vertical levels: 40
    Vtr1  = PFM['stretching'][lv1,'Vtransform']                       # transformation equation: 2
    Vst1  = PFM['stretching'][lv1,'Vstretching']                    # stretching function: 4 
    th_s1 = PFM['stretching'][lv1,'THETA_S']                      # surface stretching parameter: 8
    th_b1 = PFM['stretching'][lv1,'THETA_B']                      # bottom  stretching parameter: 3
    Tcl1  = PFM['stretching'][lv1,'TCLINE']                      # critical depth (m): 50
    hc1   = PFM['stretching'][lv1,'hc']
    
    # child vertical stretching info
    Nz2   = PFM['stretching'][lv2,'Nz']                              # number of vertical levels: 40
    Vtr2  = PFM['stretching'][lv2,'Vtransform']                       # transformation equation: 2
    Vst2  = PFM['stretching'][lv2,'Vstretching']                    # stretching function: 4 
    th_s2 = PFM['stretching'][lv2,'THETA_S']                      # surface stretching parameter: 8
    th_b2 = PFM['stretching'][lv2,'THETA_B']                      # bottom  stretching parameter: 3
    Tcl2  = PFM['stretching'][lv2,'TCLINE']                      # critical depth (m): 50
    hc2   = PFM['stretching'][lv2,'hc']

    ltr1 = G1['lat_rho']
    lnr1 = G1['lon_rho']
    ltr2 = G2['lat_rho']
    lnr2 = G2['lon_rho']
    ltu1 = G1['lat_u']
    lnu1 = G1['lon_u']
    ltu2 = G2['lat_u']
    lnu2 = G2['lon_u']
    ltv1 = G1['lat_v']
    lnv1 = G1['lon_v']
    ltv2 = G2['lat_v']
    lnv2 = G2['lon_v']

    his_ds = nc.Dataset(fn)
    
    with open(LV1_BC_pckl,'rb') as fout:
        BC1=pickle.load(fout)
        print('OCN_LV' + str(int(lvl)-1) + '_BC dict loaded with pickle')

    
    OCN_BC_0 = dict() # dict of data with the origian Nz for 3d vars

    OCN_BC = dict() # dict of data with final Nz
    OCN_BC['vinfo'] = dict()
    OCN_BC['vinfo'] = BC1['vinfo']

    OCN_BC['Nz'] = np.squeeze(Nz2)
    OCN_BC['Vtr'] = np.squeeze(Vtr2)
    OCN_BC['Vst'] = np.squeeze(Vst2)
    OCN_BC['th_s'] = np.squeeze(th_s2)
    OCN_BC['th_b'] = np.squeeze(th_b2)
    OCN_BC['Tcl'] = np.squeeze(Tcl2)
    OCN_BC['hc'] = np.squeeze(hc2)

    hraw = None
    if Vst1 == 4:
        zrom1 = s_coordinate_4(G1['h'], th_b1 , th_s1 , Tcl1 , Nz1, hraw=hraw, zeta=np.squeeze(his_ds.variables['zeta'][:,:,:]))
        

    OCN_BC['ocean_time'] = his_ds.variables['ocean_time'][:] / (3600.0 * 24) # his.nc has time in sec past reference time.
    Nt = len( OCN_BC['ocean_time'] )                                           # need in days past.
    OCN_BC['ocean_time_ref'] = BC1['ocean_time_ref']

    nlt, nln = np.shape(ltr2)

    OCN_BC['temp_south'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['u_south']    = np.zeros((Nt,Nz2,nln-1))
    OCN_BC['v_south']    = np.zeros((Nt,Nz2,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Nt,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['u_north']    = np.zeros((Nt,Nz2,nln-1))
    OCN_BC['v_north']    = np.zeros((Nt,Nz2,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Nt,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz2,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz2,nlt))
    OCN_BC['u_west']    = np.zeros((Nt,Nz2,nlt))
    OCN_BC['v_west']    = np.zeros((Nt,Nz2,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Nt,nlt))

    OCN_BC_0['temp_south'] = np.zeros((Nt,Nz1,nln))
    OCN_BC_0['salt_south'] = np.zeros((Nt,Nz1,nln))
    OCN_BC_0['u_south']    = np.zeros((Nt,Nz1,nln-1))
    OCN_BC_0['v_south']    = np.zeros((Nt,Nz1,nln))

    OCN_BC_0['temp_north'] = np.zeros((Nt,Nz1,nln))
    OCN_BC_0['salt_north'] = np.zeros((Nt,Nz1,nln))
    OCN_BC_0['u_north']    = np.zeros((Nt,Nz1,nln-1))
    OCN_BC_0['v_north']    = np.zeros((Nt,Nz1,nln))
 
    OCN_BC_0['temp_west'] = np.zeros((Nt,Nz1,nlt))
    OCN_BC_0['salt_west'] = np.zeros((Nt,Nz1,nlt))
    OCN_BC_0['u_west']    = np.zeros((Nt,Nz1,nlt))
    OCN_BC_0['v_west']    = np.zeros((Nt,Nz1,nlt-1))
    

    ZZ = dict() # this is a dict of the depths based on horizontal interpolation
    ZZ['rho_west'] = np.zeros((Nt,Nz1,nlt))
    ZZ['rho_north'] = np.zeros((Nt,Nz1,nln))
    ZZ['rho_south'] = np.zeros((Nt,Nz1,nln))
    ZZ['u_west'] = np.zeros((Nt,Nz1,nlt))
    ZZ['u_north'] = np.zeros((Nt,Nz1,nln-1))
    ZZ['u_south'] = np.zeros((Nt,Nz1,nln-1))
    ZZ['v_west'] = np.zeros((Nt,Nz1,nlt-1))
    ZZ['v_north'] = np.zeros((Nt,Nz1,nln))
    ZZ['v_south'] = np.zeros((Nt,Nz1,nln))

    ZTA = dict() # this is a dict of zeta along the edges of the child boundary for each variable
    ZTA['v_north'] = np.zeros((Nt,nln))
    ZTA['v_south'] = np.zeros((Nt,nln))
    ZTA['v_west'] = np.zeros((Nt,nlt-1))
    ZTA['u_north'] = np.zeros((Nt,nln-1))
    ZTA['u_south'] = np.zeros((Nt,nln-1))
    ZTA['u_west'] = np.zeros((Nt,nlt))
    ZTA['rho_north'] = np.zeros((Nt,nln))
    ZTA['rho_south'] = np.zeros((Nt,nln))
    ZTA['rho_west'] = np.zeros((Nt,nlt))

    Z2 = dict() # this is a dict of depths along the boundaries on the child grid
    Z2['v_north'] = np.zeros((Nt,Nz2,nln))
    Z2['v_south'] = np.zeros((Nt,Nz2,nln))
    Z2['v_west'] = np.zeros((Nt,Nz2,nlt-1))
    Z2['u_north'] = np.zeros((Nt,Nz2,nln-1))
    Z2['u_south'] = np.zeros((Nt,Nz2,nln-1))
    Z2['u_west'] = np.zeros((Nt,Nz2,nlt))
    Z2['rho_north'] = np.zeros((Nt,Nz2,nln))
    Z2['rho_south'] = np.zeros((Nt,Nz2,nln))
    Z2['rho_west'] = np.zeros((Nt,Nz2,nlt))



    # get x,y on LV1 grid.
    # x1,y1 = ll2xy(lnr1, ltr1, np.mean(lnr1), np.mean(ltr1))

    # get (x,y) grids, note zi = interp_r( (eta,xi) )
    xi_r2, eta_r2, interp_r = get_child_xi_eta_interp(lnr1,ltr1,lnr2,ltr2)
    xi_u2, eta_u2, interp_u = get_child_xi_eta_interp(lnu1,ltu1,lnu2,ltu2)
    xi_v2, eta_v2, interp_v = get_child_xi_eta_interp(lnv1,ltv1,lnv2,ltv2)

    # get nearest indices, from bad indices, so that land can be filled
    indr = get_indices_to_fill(G1['mask_rho'])
    indu = get_indices_to_fill(G1['mask_u'])
    indv = get_indices_to_fill(G1['mask_v'])

    # bookkeeping so that everything needed for each variable is associated with that variable
    v_list1 = ['zeta','ubar','vbar']
    v_list2 = ['temp','salt','u','v']

    msk_d1 = dict()
    msk_d1['zeta'] = G1['mask_rho']
    msk_d1['ubar'] = G1['mask_u']
    msk_d1['vbar'] = G1['mask_v']
    msk_d2 = dict()
    msk_d2['temp'] = G1['mask_rho']
    msk_d2['salt'] = G1['mask_rho']
    msk_d2['u']    = G1['mask_u']
    msk_d2['v']    = G1['mask_v']

    msk2_d1 = dict()
    msk2_d1['zeta'] = G2['mask_rho']
    msk2_d1['ubar'] = G2['mask_u']
    msk2_d1['vbar'] = G2['mask_v']
    msk2_d2 = dict()
    msk2_d2['temp'] = G2['mask_rho']
    msk2_d2['salt'] = G2['mask_rho']
    msk2_d2['u']    = G2['mask_u']
    msk2_d2['v']    = G2['mask_v']


    ind_d1 = dict()
    ind_d1['zeta'] = indr
    ind_d1['ubar'] = indu
    ind_d1['vbar'] = indv
    ind_d2 = dict()
    ind_d2['temp'] = indr
    ind_d2['salt'] = indr
    ind_d2['u']    = indu
    ind_d2['v']    = indv

    lat_d1 = dict()
    lat_d1['zeta'] = eta_r2
    lat_d1['ubar'] = eta_u2
    lat_d1['vbar'] = eta_v2
    lat_d2 = dict()
    lat_d2['temp'] = eta_r2
    lat_d2['salt'] = eta_r2
    lat_d2['u']    = eta_u2
    lat_d2['v']    = eta_v2

    lon_d1 = dict()
    lon_d1['zeta'] = xi_r2
    lon_d1['ubar'] = xi_u2
    lon_d1['vbar'] = xi_v2
    lon_d2 = dict()
    lon_d2['temp'] = xi_r2
    lon_d2['salt'] = xi_r2
    lon_d2['u']    = xi_u2
    lon_d2['v']    = xi_v2

    intf_d1 = dict()
    intf_d1['zeta'] = interp_r
    intf_d1['ubar'] = interp_u
    intf_d1['vbar'] = interp_v
    intf_d2 = dict()
    intf_d2['temp'] = interp_r
    intf_d2['salt'] = interp_r
    intf_d2['u']    = interp_u
    intf_d2['v']    = interp_v

    
    for vn in v_list1: # loop through all 2d variables
        msk = msk_d1[vn] # get mask on LV1
        msk2 = msk2_d1[vn] # get mask on LV2
        ind = ind_d1[vn] # get indices so that land can be filled with nearest neighbor
        xx2 = lon_d1[vn] # get xi_LV2 on LV1
        yy2 = lat_d1[vn] # get eta_LV2 on LV1, to use with the interpolator
        interpfun = intf_d1[vn]
        for tind in np.arange(Nt): # loop through times
            z0 = np.squeeze( his_ds.variables[vn][tind,:,:] )
            z0[msk==0] = z0[msk==1][ind] # fill the mask with nearest neighbor
            setattr(interpfun,'values',z0) # change the interpolator z values
            z2 = interpfun((yy2,xx2)) # perhaps change here to directly interpolate to (xi,eta) on the edges?
            #z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the land
            OCN_BC[vn+'_south'][tind,:] = z2[0,:] # fill correctly
            OCN_BC[vn+'_north'][tind,:] = z2[-1,:]
            OCN_BC[vn+'_west'][tind,:]  = z2[:,0]
            if vn == 'zeta': # this is zeta at child grid edges, for interpolating in z later
                ZTA['rho_south'][tind,:] = z2[0,:]
                ZTA['rho_north'][tind,:] = z2[-1,:]
                ZTA['rho_west'][tind,:] = z2[:,0]
                ZTA['u_south'][tind,:] = 0.5*( z2[0,0:-1]+z2[0,1:] )
                ZTA['u_north'][tind,:] = 0.5*( z2[-1,0:-1]+z2[-1,1:] )
                ZTA['u_west'][tind,:]  = 0.5*( z2[:,0]+z2[:,1] )
                ZTA['v_south'][tind,:] = 0.5*( z2[0,:]+z2[1,:] )
                ZTA['v_north'][tind,:] = 0.5*( z2[-2,:]+z2[-1,:] )
                ZTA['v_west'][tind,:]  = 0.5*( z2[0:-1,0]+z2[1:,0] )


    for vn in v_list2:
        msk = msk_d2[vn]
        msk2 = msk2_d2[vn]
        ind = ind_d2[vn]
        xx2 = lon_d2[vn]
        yy2 = lat_d2[vn]
        interpfun = intf_d2[vn]
        for tind in np.arange(Nt):
            for zind in np.arange(Nz1):
                z0 = np.squeeze( his_ds.variables[vn][tind,zind,:,:] )
                z0[msk==0] = z0[msk==1][ind]
                setattr(interpfun,'values',z0)
                z2 = interpfun((yy2,xx2))
                z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask
                OCN_BC_0[vn+'_south'][tind,zind,:] = z2[0,:]
                OCN_BC_0[vn+'_north'][tind,zind,:] = z2[-1,:]
                OCN_BC_0[vn+'_west'][tind,zind,:]  = z2[:,0]

                if vn == 'temp':
                    z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                    z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                    setattr(interpfun,'values',z0)
                    z2 = interpfun((yy2,xx2))
                    ZZ['rho_north'][tind,zind,:] = z2[-1,:] # we need the depths that the horizontal interpolation thinks it is
                    ZZ['rho_south'][tind,zind,:] = z2[0,:]
                    ZZ['rho_west'][tind,zind,:] = z2[:,0]
                if vn == 'u':
                    z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                    z0 = 0.5 * ( z0[:,0:-1] + z0[:,1:] )
                    z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                    setattr(interpfun,'values',z0)
                    z2 = interpfun((yy2,xx2))
                    ZZ['u_north'][tind,zind,:] = z2[-1,:] # we need the depths that the horizontal interpolation thinks it is
                    ZZ['u_south'][tind,zind,:] = z2[0,:]
                    ZZ['u_west'][tind,zind,:] = z2[:,0]
                if vn == 'v':
                    z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                    z0 = 0.5 * ( z0[0:-1,:] + z0[1:,:] )
                    z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                    setattr(interpfun,'values',z0)
                    z2 = interpfun((yy2,xx2))
                    ZZ['v_north'][tind,zind,:] = z2[-1,:] # we need the depths that the horizontal interpolation thinks it is
                    ZZ['v_south'][tind,zind,:] = z2[0,:]
                    ZZ['v_west'][tind,zind,:] = z2[:,0]

    h = G2['h']
    hu = .5*( h[:,0:-1] + h[:,1:] )
    hv = .5*( h[0:-1,:] + h[1:,:] )

    zr2 = s_coordinate_4(h[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_north'][:,:])
    Z2['rho_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(h[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_south'][:,:])
    Z2['rho_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(h[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_west'][:,:])
    Z2['rho_west'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_north'][:,:])
    Z2['u_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_south'][:,:])
    Z2['u_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_west'][:,:])
    Z2['u_west'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_north'][:,:])
    Z2['v_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_south'][:,:])
    Z2['v_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_west'][:,:])
    Z2['v_west'][:,:,:] = zr2.z_r[:,:,:]

    OCN_BC['Cs_r'] = np.squeeze(zr2.Cs_r)

    # now loop through all 3d variables and vertically interpolate to the correct z levels.
    pp = 0
    if pp == 1: # this didn't work. and this isn't the slow part anyway...
        print('going to interpolate BC z with ThreadPoolExecutor...')
        for vn in v_list2:
            if vn in ['temp','salt']:
                zt = 'rho'
            else:
                zt = vn
            for bnd in ['_north','_south','_west']:
                v1 = OCN_BC[vn+bnd][:,:,:] # the horizontally interpolated field
                nnt,nnz,nnp = np.shape(v1)
                for cc in np.arange(nnt):
                    futures = []
                    with ThreadPoolExecutor() as exe:
                        for aa in np.arange(nnp):
                            vp = np.squeeze( v1[cc,:,aa]) # the vertical data
                            zp = np.squeeze( ZZ[zt+bnd][cc,:,aa] ) # z locations data thinks it is at
                            zf = np.squeeze( Z2[zt+bnd][cc,:,aa]) # z locations where the data should be
                            futures.append(exe.submit(zzinterp,vp,zp,zf))
                            for future in as_completed(futures):                       
                                OCN_BC[vn+bnd][cc,:,aa] = future.result()   
        print('...done')            
    else:
        print('interpolating z with normal loops.')
        for vn in v_list2:
            if vn in ['temp','salt']:
                zt = 'rho'
            else:
                zt = vn
            for bnd in ['_north','_south','_west']:
                v1 = OCN_BC_0[vn+bnd][:,:,:] # the horizontally interpolated field
                nnt,nnz,nnp = np.shape(v1)
                for cc in np.arange(nnt):
                    for aa in np.arange(nnp):
                        vp = np.squeeze( v1[cc,:,aa]) # the vertical data
                        zp = np.squeeze( ZZ[zt+bnd][cc,:,aa] ) # z locations data thinks it is at
                        zf = np.squeeze( Z2[zt+bnd][cc,:,aa]) # z locations where the data should be
                        Fz = interp1d(zp,vp,bounds_error=False,kind='linear',fill_value = 'extrapolate') 
                        vf =  np.squeeze(Fz(zf))
                        OCN_BC[vn+bnd][cc,:,aa] = vf                

    #fn_out = '/scratch/PFM_Simulations/LV3_Forecast/Forc/test_BC_LV3.pkl'

    with open(fn_out,'wb') as fout:
        pickle.dump(OCN_BC,fout)
        print('OCN_LV',lvl,'_BC dict saved with pickle to: ',fn_out)

    #return OCN_BC
    #return xi_r2, eta_r2, interp_r

def mk_LV2_BC_dict_edges(lvl):

    PFM=get_PFM_info()  
    if lvl == '2':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        fn = PFM['lv1_his_name_full']
        LV1_BC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']
        lv1 = 'L1'
        lv2 = 'L2'
        fn_out = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnBC_tmp_pckl_file']
    elif lvl == '3':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        fn = PFM['lv2_his_name_full']
        LV1_BC_pckl = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnBC_tmp_pckl_file']
        lv1 = 'L2'
        lv2 = 'L3'
        fn_out = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnBC_tmp_pckl_file']
    elif lvl == '4':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
        fn = PFM['lv3_his_name_full']
        LV1_BC_pckl = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnBC_tmp_pckl_file']
        lv1 = 'L3'
        lv2 = 'L4'
        fn_out = PFM['lv4_forc_dir'] + '/' + PFM['lv4_ocnBC_tmp_pckl_file']
     
    # parent vertical stretching info 
    Nz1   = PFM['stretching'][lv1,'Nz']                              # number of vertical levels: 40
    Vtr1  = PFM['stretching'][lv1,'Vtransform']                       # transformation equation: 2
    Vst1  = PFM['stretching'][lv1,'Vstretching']                    # stretching function: 4 
    th_s1 = PFM['stretching'][lv1,'THETA_S']                      # surface stretching parameter: 8
    th_b1 = PFM['stretching'][lv1,'THETA_B']                      # bottom  stretching parameter: 3
    Tcl1  = PFM['stretching'][lv1,'TCLINE']                      # critical depth (m): 50
    hc1   = PFM['stretching'][lv1,'hc']
    
    # child vertical stretching info
    Nz2   = PFM['stretching'][lv2,'Nz']                              # number of vertical levels: 40
    Vtr2  = PFM['stretching'][lv2,'Vtransform']                       # transformation equation: 2
    Vst2  = PFM['stretching'][lv2,'Vstretching']                    # stretching function: 4 
    th_s2 = PFM['stretching'][lv2,'THETA_S']                      # surface stretching parameter: 8
    th_b2 = PFM['stretching'][lv2,'THETA_B']                      # bottom  stretching parameter: 3
    Tcl2  = PFM['stretching'][lv2,'TCLINE']                      # critical depth (m): 50
    hc2   = PFM['stretching'][lv2,'hc']

    ltr1 = G1['lat_rho']
    lnr1 = G1['lon_rho']
    ltr2 = G2['lat_rho']
    lnr2 = G2['lon_rho']
    ltu1 = G1['lat_u']
    lnu1 = G1['lon_u']
    ang1 = G1['angle']
    
    ltu2 = G2['lat_u']
    lnu2 = G2['lon_u']
    ltv1 = G1['lat_v']
    lnv1 = G1['lon_v']
    ltv2 = G2['lat_v']
    lnv2 = G2['lon_v']
    ang2 = G2['angle']

    his_ds = nc.Dataset(fn)
    
    with open(LV1_BC_pckl,'rb') as fout:
        BC1=pickle.load(fout)
        print('OCN_LV' + str(int(lvl)-1) + '_BC dict loaded with pickle')

    
    OCN_BC_0 = dict() # dict of data with the origian Nz for 3d vars

    OCN_BC = dict() # dict of data with final Nz
    OCN_BC['vinfo'] = dict()
    OCN_BC['vinfo'] = BC1['vinfo']

    OCN_BC['Nz'] = np.squeeze(Nz2)
    OCN_BC['Vtr'] = np.squeeze(Vtr2)
    OCN_BC['Vst'] = np.squeeze(Vst2)
    OCN_BC['th_s'] = np.squeeze(th_s2)
    OCN_BC['th_b'] = np.squeeze(th_b2)
    OCN_BC['Tcl'] = np.squeeze(Tcl2)
    OCN_BC['hc'] = np.squeeze(hc2)

    hraw = None
    if Vst1 == 4:
        zrom1 = s_coordinate_4(G1['h'], th_b1 , th_s1 , Tcl1 , Nz1, hraw=hraw, zeta=np.squeeze(his_ds.variables['zeta'][:,:,:]))
        

    OCN_BC['ocean_time'] = his_ds.variables['ocean_time'][:] / (3600.0 * 24) # his.nc has time in sec past reference time.
    Nt = len( OCN_BC['ocean_time'] )                                           # need in days past.
    OCN_BC['ocean_time_ref'] = BC1['ocean_time_ref']

    nlt, nln = np.shape(ltr2)

    OCN_BC['temp_south'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['u_south']    = np.zeros((Nt,Nz2,nln-1))
    OCN_BC['v_south']    = np.zeros((Nt,Nz2,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Nt,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['u_north']    = np.zeros((Nt,Nz2,nln-1))
    OCN_BC['v_north']    = np.zeros((Nt,Nz2,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Nt,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz2,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz2,nlt))
    OCN_BC['u_west']    = np.zeros((Nt,Nz2,nlt))
    OCN_BC['v_west']    = np.zeros((Nt,Nz2,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Nt,nlt))

    OCN_BC_0['temp_south'] = np.zeros((Nt,Nz1,nln))
    OCN_BC_0['salt_south'] = np.zeros((Nt,Nz1,nln))
    OCN_BC_0['u_south']    = np.zeros((Nt,Nz1,nln-1))
    OCN_BC_0['v_south']    = np.zeros((Nt,Nz1,nln))

    OCN_BC_0['temp_north'] = np.zeros((Nt,Nz1,nln))
    OCN_BC_0['salt_north'] = np.zeros((Nt,Nz1,nln))
    OCN_BC_0['u_north']    = np.zeros((Nt,Nz1,nln-1))
    OCN_BC_0['v_north']    = np.zeros((Nt,Nz1,nln))
 
    OCN_BC_0['temp_west'] = np.zeros((Nt,Nz1,nlt))
    OCN_BC_0['salt_west'] = np.zeros((Nt,Nz1,nlt))
    OCN_BC_0['u_west']    = np.zeros((Nt,Nz1,nlt))
    OCN_BC_0['v_west']    = np.zeros((Nt,Nz1,nlt-1))
    

    ZZ = dict() # this is a dict of the depths based on horizontal interpolation
    ZZ['rho_west'] = np.zeros((Nt,Nz1,nlt))
    ZZ['rho_north'] = np.zeros((Nt,Nz1,nln))
    ZZ['rho_south'] = np.zeros((Nt,Nz1,nln))
    ZZ['u_west'] = np.zeros((Nt,Nz1,nlt))
    ZZ['u_north'] = np.zeros((Nt,Nz1,nln-1))
    ZZ['u_south'] = np.zeros((Nt,Nz1,nln-1))
    ZZ['v_west'] = np.zeros((Nt,Nz1,nlt-1))
    ZZ['v_north'] = np.zeros((Nt,Nz1,nln))
    ZZ['v_south'] = np.zeros((Nt,Nz1,nln))

    ZTA = dict() # this is a dict of zeta along the edges of the child boundary for each variable
    ZTA['v_north'] = np.zeros((Nt,nln))
    ZTA['v_south'] = np.zeros((Nt,nln))
    ZTA['v_west'] = np.zeros((Nt,nlt-1))
    ZTA['u_north'] = np.zeros((Nt,nln-1))
    ZTA['u_south'] = np.zeros((Nt,nln-1))
    ZTA['u_west'] = np.zeros((Nt,nlt))
    ZTA['rho_north'] = np.zeros((Nt,nln))
    ZTA['rho_south'] = np.zeros((Nt,nln))
    ZTA['rho_west'] = np.zeros((Nt,nlt))

    Z2 = dict() # this is a dict of depths along the boundaries on the child grid
    Z2['v_north'] = np.zeros((Nt,Nz2,nln))
    Z2['v_south'] = np.zeros((Nt,Nz2,nln))
    Z2['v_west'] = np.zeros((Nt,Nz2,nlt-1))
    Z2['u_north'] = np.zeros((Nt,Nz2,nln-1))
    Z2['u_south'] = np.zeros((Nt,Nz2,nln-1))
    Z2['u_west'] = np.zeros((Nt,Nz2,nlt))
    Z2['rho_north'] = np.zeros((Nt,Nz2,nln))
    Z2['rho_south'] = np.zeros((Nt,Nz2,nln))
    Z2['rho_west'] = np.zeros((Nt,Nz2,nlt))


    bnds = ['_north','_south','_west']

    # get x,y on LV1 grid.
    # x1,y1 = ll2xy(lnr1, ltr1, np.mean(lnr1), np.mean(ltr1))

    # get (x,y) grids, note zi = interp_r( (eta,xi) )
    xi_r2, eta_r2, interp_r = get_child_xi_eta_interp(lnr1,ltr1,lnr2,ltr2)
    xi_u2, eta_u2, interp_u = get_child_xi_eta_interp(lnu1,ltu1,lnu2,ltu2)
    xi_v2, eta_v2, interp_v = get_child_xi_eta_interp(lnv1,ltv1,lnv2,ltv2)

    

    XX = dict()
    YY = dict()
    XX['u','_north'] = xi_u2[-1,:]
    YY['u','_north'] = eta_u2[-1,:]
    XX['u','_south'] = xi_u2[0,:]
    YY['u','_south'] = eta_u2[0,:]
    XX['u','_west']  = xi_u2[:,0]
    YY['u','_west']  = eta_u2[:,0]

    XX['v','_north'] = xi_v2[-1,:]
    YY['v','_north'] = eta_v2[-1,:]
    XX['v','_south'] = xi_v2[0,:]
    YY['v','_south'] = eta_v2[0,:]
    XX['v','_west']  = xi_v2[:,0]
    YY['v','_west']  = eta_v2[:,0]

    XX['rho','_north'] = xi_r2[-1,:]
    YY['rho','_north'] = eta_r2[-1,:]
    XX['rho','_south'] = xi_r2[0,:]
    YY['rho','_south'] = eta_r2[0,:]
    XX['rho','_west']  = xi_r2[:,0]
    YY['rho','_west']  = eta_r2[:,0]

    # get nearest indices, from bad indices, so that land can be filled
    indr = get_indices_to_fill(G1['mask_rho'])
    indu = get_indices_to_fill(G1['mask_u'])
    indv = get_indices_to_fill(G1['mask_v'])

    # bookkeeping so that everything needed for each variable is associated with that variable
    v_list1 = ['zeta','ubar','vbar']
    v1_2_g = dict()
    v1_2_g['zeta'] = 'rho'
    v1_2_g['ubar'] = 'u'
    v1_2_g['vbar'] = 'v'

    v_list2 = ['temp','salt','u','v']
    v2_2_g = dict()
    v2_2_g['temp'] = 'rho'
    v2_2_g['salt'] = 'rho'
    v2_2_g['u'] = 'u'
    v2_2_g['v'] = 'v'

    msk_d1 = dict()
    msk_d1['zeta'] = G1['mask_rho']
    msk_d1['ubar'] = G1['mask_u']
    msk_d1['vbar'] = G1['mask_v']
    msk_d2 = dict()
    msk_d2['temp'] = G1['mask_rho']
    msk_d2['salt'] = G1['mask_rho']
    msk_d2['u']    = G1['mask_u']
    msk_d2['v']    = G1['mask_v']

    msk2_d1 = dict()
    msk2_d1['zeta','_north'] = G2['mask_rho'][-1,:]
    msk2_d1['zeta','_south'] = G2['mask_rho'][0,:]
    msk2_d1['zeta','_west']  = G2['mask_rho'][:,0]
    msk2_d1['ubar','_north'] = G2['mask_u'][-1,:]
    msk2_d1['ubar','_south'] = G2['mask_u'][0,:]
    msk2_d1['ubar','_west']  = G2['mask_u'][:,0]
    msk2_d1['vbar','_north'] = G2['mask_v'][-1,:]
    msk2_d1['vbar','_south'] = G2['mask_v'][0,:]
    msk2_d1['vbar','_west']  = G2['mask_v'][:,0]

    msk2_d2 = dict()
    msk2_d2['temp','_north'] = msk2_d1['zeta','_north']
    msk2_d2['salt','_north'] = msk2_d1['zeta','_north']
    msk2_d2['u','_north']    = msk2_d1['ubar','_north']
    msk2_d2['v','_north']    = msk2_d1['vbar','_north']
    msk2_d2['temp','_south'] = msk2_d1['zeta','_south']
    msk2_d2['salt','_south'] = msk2_d1['zeta','_south']
    msk2_d2['u','_south']    = msk2_d1['ubar','_south']
    msk2_d2['v','_south']    = msk2_d1['vbar','_south']
    msk2_d2['temp','_west']  = msk2_d1['zeta','_west']
    msk2_d2['salt','_west']  = msk2_d1['zeta','_west']
    msk2_d2['u','_west']     = msk2_d1['ubar','_west']
    msk2_d2['v','_west']     = msk2_d1['vbar','_west']
 
 
    ind_d1 = dict()
    ind_d1['zeta'] = indr
    ind_d1['ubar'] = indu
    ind_d1['vbar'] = indv
    ind_d2 = dict()
    ind_d2['temp'] = indr
    ind_d2['salt'] = indr
    ind_d2['u']    = indu
    ind_d2['v']    = indv

    lat_d1 = dict()
    lat_d1['zeta'] = eta_r2
    lat_d1['ubar'] = eta_u2
    lat_d1['vbar'] = eta_v2
    lat_d2 = dict()
    lat_d2['temp'] = eta_r2
    lat_d2['salt'] = eta_r2
    lat_d2['u']    = eta_u2
    lat_d2['v']    = eta_v2

    lon_d1 = dict()
    lon_d1['zeta'] = xi_r2
    lon_d1['ubar'] = xi_u2
    lon_d1['vbar'] = xi_v2
    lon_d2 = dict()
    lon_d2['temp'] = xi_r2
    lon_d2['salt'] = xi_r2
    lon_d2['u']    = xi_u2
    lon_d2['v']    = xi_v2

    intf_d1 = dict()
    intf_d1['zeta'] = interp_r
    intf_d1['ubar'] = interp_u
    intf_d1['vbar'] = interp_v
    intf_d2 = dict()
    intf_d2['temp'] = interp_r
    intf_d2['salt'] = interp_r
    intf_d2['u']    = interp_u
    intf_d2['v']    = interp_v

    angle_on_2 = dict()
    ang2u = 0.5 * ( ang2[:,1:] + ang2[:,0:-1] )
    ang2v = 0.5 * ( ang2[0:-1,:] + ang2[1:,:] )
    angle_on_2['ubar','_north'] = ang2u[-1,:]
    angle_on_2['ubar','_south'] = ang2u[0,:]
    angle_on_2['ubar','_west']  = ang2u[:,0]
    angle_on_2['vbar','_north'] = ang2v[-1,:]
    angle_on_2['vbar','_south'] = ang2v[0,:]
    angle_on_2['vbar','_west'] =  ang2v[:,0]

    angle_on_1 = dict()
    ang_2m1 = dict()
    setattr(interp_r,'values',ang1) # change the interpolator z values
    for vn in ['ubar','vbar']:
        for bb in bnds:
            xx2 = XX[v1_2_g[vn],bb]
            yy2 = YY[v1_2_g[vn],bb]
            z2 = interp_r((yy2,xx2))    
            angle_on_1[vn,bb] = z2
            ang_2m1[vn,bb] = angle_on_2[vn,bb] - angle_on_1[vn,bb]

    for vn in v_list1: # loop through all 2d variables
        msk = msk_d1[vn] # get mask on LV1
        ind = ind_d1[vn] # get indices so that land can be filled with nearest neighbor
        interpfun = intf_d1[vn]
        for tind in np.arange(Nt): # loop through times
            z0 = np.squeeze( his_ds.variables[vn][tind,:,:] )
            z0[msk==0] = z0[msk==1][ind] # fill the mask with nearest neighbor
            setattr(interpfun,'values',z0) # change the interpolator z values
            for bb in bnds:
                xx2 = XX[v1_2_g[vn],bb]
                yy2 = YY[v1_2_g[vn],bb]
                z2 = interpfun((yy2,xx2)) # perhaps change here to directly interpolate to (xi,eta) on the edges?
                msk2 = msk2_d1[vn,bb]
                #print(vn)
                #print(tind)
                #print(bb)
                #print(np.shape(msk2))
                #print(np.shape(z2))
                
                z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask
                OCN_BC[vn+bb][tind,:] = z2 # fill correctly
                if vn == 'zeta': # this is zeta at child grid edges, for interpolating in z later
                    ZTA['rho'+bb][tind,:] = z2
                    xx2 = XX['u',bb]
                    yy2 = YY['u',bb]
                    z2 = interpfun((yy2,xx2)) 
                    ZTA['u'+bb][tind,:] = z2
                    xx2 = XX['v',bb]
                    yy2 = YY['v',bb]
                    z2 = interpfun((yy2,xx2)) 
                    ZTA['v'+bb][tind,:] = z2


    for vn in v_list2:
        msk = msk_d2[vn]
        ind = ind_d2[vn]
        interpfun = intf_d2[vn]
        for tind in np.arange(Nt):
            for zind in np.arange(Nz1):
                z0 = np.squeeze( his_ds.variables[vn][tind,zind,:,:] )
                z0[msk==0] = z0[msk==1][ind]
                setattr(interpfun,'values',z0)
                for bb in bnds:
                    xx2 = XX[v2_2_g[vn],bb]
                    yy2 = YY[v2_2_g[vn],bb]
                    z2 = interpfun((yy2,xx2)) # perhaps change here to directly interpolate to (xi,eta) on the edges?
                    msk2 = msk2_d2[vn,bb]
                    z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask
                    OCN_BC_0[vn+bb][tind,zind,:] = z2 # fill correctly
                    if vn == 'temp':
                        z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                        z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                        setattr(interpfun,'values',z0)
                        z2 = interpfun((yy2,xx2))
                        ZZ['rho'+bb][tind,zind,:] = z2 # we need the depths that the horizontal interpolation thinks it is
                    if vn == 'u':
                        z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                        z0 = 0.5 * ( z0[:,0:-1] + z0[:,1:] )
                        z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                        setattr(interpfun,'values',z0)
                        z2 = interpfun((yy2,xx2))
                        ZZ['u'+bb][tind,zind,:] = z2 # we need the depths that the horizontal interpolation thinks it is
                    if vn == 'v':
                        z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                        z0 = 0.5 * ( z0[0:-1,:] + z0[1:,:] )
                        z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                        setattr(interpfun,'values',z0)
                        z2 = interpfun((yy2,xx2))
                        ZZ['v'+bb][tind,zind,:] = z2 # we need the depths that the horizontal interpolation thinks it is
 
    if lvl == '4': # need to rotate the velocities!
        msk = msk_d1['vbar'] # get mask on LV1
        ind = ind_d1['vbar'] # get indices so that land can be filled with nearest neighbor
        interpfun = intf_d1['vbar']
        for tind in np.arange(Nt): # loop through times
            z0 = np.squeeze( his_ds.variables['vbar'][tind,:,:] )
            z0[msk==0] = z0[msk==1][ind] # fill the mask with nearest neighbor
            setattr(interpfun,'values',z0) # change the interpolator z values
            for bb in bnds:
                xx2 = XX[v1_2_g['ubar'],bb]
                yy2 = YY[v1_2_g['ubar'],bb]
                z2 = interpfun((yy2,xx2)) 
                msk2 = msk2_d1['ubar',bb]
                z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask, this is vbar on u
                cosa = np.cos(ang_2m1['ubar',bb])
                sina = np.sin(ang_2m1['ubar',bb])
                OCN_BC['ubar'+bb][tind,:] = cosa * OCN_BC['ubar'+bb][tind,:] + sina * z2
                
                    
        msk = msk_d1['ubar'] # get mask on LV1
        ind = ind_d1['ubar'] # get indices so that land can be filled with nearest neighbor
        interpfun = intf_d1['ubar']
        for tind in np.arange(Nt): # loop through times
            z0 = np.squeeze( his_ds.variables['ubar'][tind,:,:] )
            z0[msk==0] = z0[msk==1][ind] # fill the mask with nearest neighbor
            setattr(interpfun,'values',z0) # change the interpolator z values
            for bb in bnds:
                xx2 = XX[v1_2_g['vbar'],bb]
                yy2 = YY[v1_2_g['vbar'],bb]
                z2 = interpfun((yy2,xx2)) # perhaps change here to directly interpolate to (xi,eta) on the edges?
                msk2 = msk2_d1['vbar',bb]
                z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask
                cosa = np.cos(ang_2m1['vbar',bb])
                sina = np.sin(ang_2m1['vbar',bb])
                OCN_BC['vbar'+bb][tind,:] = cosa * OCN_BC['vbar'+bb][tind,:] - sina * z2

        msk = msk_d1['ubar'] # get mask on LV1
        ind = ind_d1['ubar'] # get indices so that land can be filled with nearest neighbor
        interpfun = intf_d1['ubar']
        for tind in np.arange(Nt): # loop through times
            for zind in np.arange(Nz1):
                z0 = np.squeeze( his_ds.variables['u'][tind,zind,:,:] )
                z0[msk==0] = z0[msk==1][ind] # fill the mask with nearest neighbor
                setattr(interpfun,'values',z0) # change the interpolator z values
                for bb in bnds:
                    xx2 = XX[v1_2_g['vbar'],bb]
                    yy2 = YY[v1_2_g['vbar'],bb]
                    z2 = interpfun((yy2,xx2)) 
                    msk2 = msk2_d1['vbar',bb]
                    z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask, this is vbar on u
                    cosa = np.cos(ang_2m1['vbar',bb])
                    sina = np.sin(ang_2m1['vbar',bb])
                    OCN_BC_0['v'+bb][tind,zind,:] = cosa * OCN_BC_0['v'+bb][tind,zind,:] - sina * z2

        msk = msk_d1['vbar'] # get mask on LV1
        ind = ind_d1['vbar'] # get indices so that land can be filled with nearest neighbor
        interpfun = intf_d1['vbar']
        for tind in np.arange(Nt): # loop through times
            for zind in np.arange(Nz1):
                z0 = np.squeeze( his_ds.variables['v'][tind,zind,:,:] )
                z0[msk==0] = z0[msk==1][ind] # fill the mask with nearest neighbor
                setattr(interpfun,'values',z0) # change the interpolator z values
                for bb in bnds:
                    xx2 = XX[v1_2_g['ubar'],bb]
                    yy2 = YY[v1_2_g['ubar'],bb]
                    z2 = interpfun((yy2,xx2)) 
                    msk2 = msk2_d1['ubar',bb]
                    z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask, this is vbar on u
                    cosa = np.cos(ang_2m1['ubar',bb])
                    sina = np.sin(ang_2m1['ubar',bb])
                    OCN_BC_0['u'+bb][tind,zind,:] = cosa * OCN_BC_0['u'+bb][tind,zind,:] + sina * z2

    h = G2['h']
    hu = .5*( h[:,0:-1] + h[:,1:] )
    hv = .5*( h[0:-1,:] + h[1:,:] )

    zr2 = s_coordinate_4(h[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_north'][:,:])
    Z2['rho_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(h[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_south'][:,:])
    Z2['rho_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(h[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_west'][:,:])
    Z2['rho_west'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_north'][:,:])
    Z2['u_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_south'][:,:])
    Z2['u_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_west'][:,:])
    Z2['u_west'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_north'][:,:])
    Z2['v_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_south'][:,:])
    Z2['v_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_west'][:,:])
    Z2['v_west'][:,:,:] = zr2.z_r[:,:,:]

    OCN_BC['Cs_r'] = np.squeeze(zr2.Cs_r)

    # now loop through all 3d variables and vertically interpolate to the correct z levels.
    pp = 0
    if pp == 1: # this didn't work. and this isn't the slow part anyway...
        print('going to interpolate BC z with ThreadPoolExecutor...')
        for vn in v_list2:
            if vn in ['temp','salt']:
                zt = 'rho'
            else:
                zt = vn
            for bnd in ['_north','_south','_west']:
                v1 = OCN_BC[vn+bnd][:,:,:] # the horizontally interpolated field
                nnt,nnz,nnp = np.shape(v1)
                for cc in np.arange(nnt):
                    futures = []
                    with ThreadPoolExecutor() as exe:
                        for aa in np.arange(nnp):
                            vp = np.squeeze( v1[cc,:,aa]) # the vertical data
                            zp = np.squeeze( ZZ[zt+bnd][cc,:,aa] ) # z locations data thinks it is at
                            zf = np.squeeze( Z2[zt+bnd][cc,:,aa]) # z locations where the data should be
                            futures.append(exe.submit(zzinterp,vp,zp,zf))
                            for future in as_completed(futures):                       
                                OCN_BC[vn+bnd][cc,:,aa] = future.result()   
        print('...done')            
    else:
        print('interpolating z with normal loops.')
        for vn in v_list2:
            if vn in ['temp','salt']:
                zt = 'rho'
            else:
                zt = vn
            for bnd in ['_north','_south','_west']:
                v1 = OCN_BC_0[vn+bnd][:,:,:] # the horizontally interpolated field
                nnt,nnz,nnp = np.shape(v1)
                for cc in np.arange(nnt):
                    for aa in np.arange(nnp):
                        vp = np.squeeze( v1[cc,:,aa]) # the vertical data
                        zp = np.squeeze( ZZ[zt+bnd][cc,:,aa] ) # z locations data thinks it is at
                        zf = np.squeeze( Z2[zt+bnd][cc,:,aa]) # z locations where the data should be
                        Fz = interp1d(zp,vp,bounds_error=False,kind='linear',fill_value = 'extrapolate') 
                        vf =  np.squeeze(Fz(zf))
                        OCN_BC[vn+bnd][cc,:,aa] = vf                

    #fn_out = '/scratch/PFM_Simulations/LV3_Forecast/Forc/test_BC_LV3.pkl'

    with open(fn_out,'wb') as fout:
        pickle.dump(OCN_BC,fout)
        print('OCN_LV',lvl,'_BC dict saved with pickle to: ',fn_out)

    #return OCN_BC
    #return xi_r2, eta_r2, interp_r


def zzinterp(vp, zp, zf):
    Fz = interp1d(zp,vp,bounds_error=False,kind='linear',fill_value = 'extrapolate') 
    vf =  np.squeeze(Fz(zf))
    return vf

def mk_LV2_BC_dict_1hrzeta(lvl):

    PFM=get_PFM_info()  
    if lvl == '2':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        fn = PFM['lv1_his_name_full']
        LV1_BC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnBC_tmp_pckl_file']
        lv1 = 'L1'
        lv2 = 'L2'
        fn_out = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnBC_tmp_pckl_file']
    elif lvl == '3':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        fn = PFM['lv2_his_name_full']
        LV1_BC_pckl = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnBC_tmp_pckl_file']
        lv1 = 'L2'
        lv2 = 'L3'
        fn_out = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnBC_tmp_pckl_file']
    elif lvl == '4':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
        fn = PFM['lv3_his_name_full']
        LV1_BC_pckl = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnBC_tmp_pckl_file']
        lv1 = 'L3'
        lv2 = 'L4'
     
    # parent vertical stretching info 
    Nz1   = PFM['stretching'][lv1,'Nz']                              # number of vertical levels: 40
    Vtr1  = PFM['stretching'][lv1,'Vtransform']                       # transformation equation: 2
    Vst1  = PFM['stretching'][lv1,'Vstretching']                    # stretching function: 4 
    th_s1 = PFM['stretching'][lv1,'THETA_S']                      # surface stretching parameter: 8
    th_b1 = PFM['stretching'][lv1,'THETA_B']                      # bottom  stretching parameter: 3
    Tcl1  = PFM['stretching'][lv1,'TCLINE']                      # critical depth (m): 50
    hc1   = PFM['stretching'][lv1,'hc']
    
    # child vertical stretching info
    Nz2   = PFM['stretching'][lv2,'Nz']                              # number of vertical levels: 40
    Vtr2  = PFM['stretching'][lv2,'Vtransform']                       # transformation equation: 2
    Vst2  = PFM['stretching'][lv2,'Vstretching']                    # stretching function: 4 
    th_s2 = PFM['stretching'][lv2,'THETA_S']                      # surface stretching parameter: 8
    th_b2 = PFM['stretching'][lv2,'THETA_B']                      # bottom  stretching parameter: 3
    Tcl2  = PFM['stretching'][lv2,'TCLINE']                      # critical depth (m): 50
    hc2   = PFM['stretching'][lv2,'hc']

    ltr1 = G1['lat_rho']
    lnr1 = G1['lon_rho']
    ltr2 = G2['lat_rho']
    lnr2 = G2['lon_rho']
    ltu1 = G1['lat_u']
    lnu1 = G1['lon_u']
    ltu2 = G2['lat_u']
    lnu2 = G2['lon_u']
    ltv1 = G1['lat_v']
    lnv1 = G1['lon_v']
    ltv2 = G2['lat_v']
    lnv2 = G2['lon_v']

    his_ds = nc.Dataset(fn)
    
    with open(LV1_BC_pckl,'rb') as fout:
        BC1=pickle.load(fout)
        print('OCN_LV' + str(int(lvl)-1) + '_BC dict loaded with pickle')

    OCN_BC = dict()
    OCN_BC['vinfo'] = dict()
    OCN_BC['vinfo'] = BC1['vinfo']

    OCN_BC['Nz'] = np.squeeze(Nz2)
    OCN_BC['Vtr'] = np.squeeze(Vtr2)
    OCN_BC['Vst'] = np.squeeze(Vst2)
    OCN_BC['th_s'] = np.squeeze(th_s2)
    OCN_BC['th_b'] = np.squeeze(th_b2)
    OCN_BC['Tcl'] = np.squeeze(Tcl2)
    OCN_BC['hc'] = np.squeeze(hc2)

    hraw = None
    if Vst1 == 4:
        zrom1 = s_coordinate_4(G1['h'], th_b1 , th_s1 , Tcl1 , Nz1, hraw=hraw, zeta=np.squeeze(his_ds.variables['zeta'][:,:,:]))
        

    OCN_BC['ocean_time'] = his_ds.variables['ocean_time'][:] / (3600.0 * 24) # his.nc has time in sec past reference time.
    Nt = len( OCN_BC['ocean_time'] )                                           # need in days past.
    OCN_BC['ocean_time_ref'] = BC1['ocean_time_ref']

    nlt, nln = np.shape(ltr2)

    OCN_BC['temp_south'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['u_south']    = np.zeros((Nt,Nz2,nln-1))
    OCN_BC['v_south']    = np.zeros((Nt,Nz2,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Nt,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz2,nln))
    OCN_BC['u_north']    = np.zeros((Nt,Nz2,nln-1))
    OCN_BC['v_north']    = np.zeros((Nt,Nz2,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Nt,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz2,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz2,nlt))
    OCN_BC['u_west']    = np.zeros((Nt,Nz2,nlt))
    OCN_BC['v_west']    = np.zeros((Nt,Nz2,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Nt,nlt))

    ZZ = dict() # this is a dict of the depths based on horizontal interpolation
    ZZ['rho_west'] = np.zeros((Nt,Nz1,nlt))
    ZZ['rho_north'] = np.zeros((Nt,Nz1,nln))
    ZZ['rho_south'] = np.zeros((Nt,Nz1,nln))
    ZZ['u_west'] = np.zeros((Nt,Nz1,nlt))
    ZZ['u_north'] = np.zeros((Nt,Nz1,nln-1))
    ZZ['u_south'] = np.zeros((Nt,Nz1,nln-1))
    ZZ['v_west'] = np.zeros((Nt,Nz1,nlt-1))
    ZZ['v_north'] = np.zeros((Nt,Nz1,nln))
    ZZ['v_south'] = np.zeros((Nt,Nz1,nln))

    ZTA = dict() # this is a dict of zeta along the edges of the child boundary for each variable
    ZTA['v_north'] = np.zeros((Nt,nln))
    ZTA['v_south'] = np.zeros((Nt,nln))
    ZTA['v_west'] = np.zeros((Nt,nlt-1))
    ZTA['u_north'] = np.zeros((Nt,nln-1))
    ZTA['u_south'] = np.zeros((Nt,nln-1))
    ZTA['u_west'] = np.zeros((Nt,nlt))
    ZTA['rho_north'] = np.zeros((Nt,nln))
    ZTA['rho_south'] = np.zeros((Nt,nln))
    ZTA['rho_west'] = np.zeros((Nt,nlt))

    Z2 = dict() # this is a dict of depths along the boundaries on the child grid
    Z2['v_north'] = np.zeros((Nt,Nz2,nln))
    Z2['v_south'] = np.zeros((Nt,Nz2,nln))
    Z2['v_west'] = np.zeros((Nt,Nz2,nlt-1))
    Z2['u_north'] = np.zeros((Nt,Nz2,nln-1))
    Z2['u_south'] = np.zeros((Nt,Nz2,nln-1))
    Z2['u_west'] = np.zeros((Nt,Nz2,nlt))
    Z2['rho_north'] = np.zeros((Nt,Nz2,nln))
    Z2['rho_south'] = np.zeros((Nt,Nz2,nln))
    Z2['rho_west'] = np.zeros((Nt,Nz2,nlt))


    # get x,y on LV1 grid.
    # x1,y1 = ll2xy(lnr1, ltr1, np.mean(lnr1), np.mean(ltr1))

    # get (x,y) grids, note zi = interp_r( (eta,xi) )
    xi_r2, eta_r2, interp_r = get_child_xi_eta_interp(lnr1,ltr1,lnr2,ltr2)
    xi_u2, eta_u2, interp_u = get_child_xi_eta_interp(lnu1,ltu1,lnu2,ltu2)
    xi_v2, eta_v2, interp_v = get_child_xi_eta_interp(lnv1,ltv1,lnv2,ltv2)

    # get nearest indices, from bad indices, so that land can be filled
    indr = get_indices_to_fill(G1['mask_rho'])
    indu = get_indices_to_fill(G1['mask_u'])
    indv = get_indices_to_fill(G1['mask_v'])

    # bookkeeping so that everything needed for each variable is associated with that variable
    v_list1 = ['zeta','ubar','vbar']
    v_list2 = ['temp','salt','u','v']

    msk_d1 = dict()
    msk_d1['zeta'] = G1['mask_rho']
    msk_d1['ubar'] = G1['mask_u']
    msk_d1['vbar'] = G1['mask_v']
    msk_d2 = dict()
    msk_d2['temp'] = G1['mask_rho']
    msk_d2['salt'] = G1['mask_rho']
    msk_d2['u']    = G1['mask_u']
    msk_d2['v']    = G1['mask_v']

    msk2_d1 = dict()
    msk2_d1['zeta'] = G2['mask_rho']
    msk2_d1['ubar'] = G2['mask_u']
    msk2_d1['vbar'] = G2['mask_v']
    msk2_d2 = dict()
    msk2_d2['temp'] = G2['mask_rho']
    msk2_d2['salt'] = G2['mask_rho']
    msk2_d2['u']    = G2['mask_u']
    msk2_d2['v']    = G2['mask_v']


    ind_d1 = dict()
    ind_d1['zeta'] = indr
    ind_d1['ubar'] = indu
    ind_d1['vbar'] = indv
    ind_d2 = dict()
    ind_d2['temp'] = indr
    ind_d2['salt'] = indr
    ind_d2['u']    = indu
    ind_d2['v']    = indv

    lat_d1 = dict()
    lat_d1['zeta'] = eta_r2
    lat_d1['ubar'] = eta_u2
    lat_d1['vbar'] = eta_v2
    lat_d2 = dict()
    lat_d2['temp'] = eta_r2
    lat_d2['salt'] = eta_r2
    lat_d2['u']    = eta_u2
    lat_d2['v']    = eta_v2

    lon_d1 = dict()
    lon_d1['zeta'] = xi_r2
    lon_d1['ubar'] = xi_u2
    lon_d1['vbar'] = xi_v2
    lon_d2 = dict()
    lon_d2['temp'] = xi_r2
    lon_d2['salt'] = xi_r2
    lon_d2['u']    = xi_u2
    lon_d2['v']    = xi_v2

    intf_d1 = dict()
    intf_d1['zeta'] = interp_r
    intf_d1['ubar'] = interp_u
    intf_d1['vbar'] = interp_v
    intf_d2 = dict()
    intf_d2['temp'] = interp_r
    intf_d2['salt'] = interp_r
    intf_d2['u']    = interp_u
    intf_d2['v']    = interp_v

    
    for vn in v_list1: # loop through all 2d variables
        msk = msk_d1[vn] # get mask on LV1
        msk2 = msk2_d1[vn] # get mask on LV2
        ind = ind_d1[vn] # get indices so that land can be filled with nearest neighbor
        xx2 = lon_d1[vn] # get xi_LV2 on LV1
        yy2 = lat_d1[vn] # get eta_LV2 on LV1, to use with the interpolator
        interpfun = intf_d1[vn]
        for tind in np.arange(Nt): # loop through times
            z0 = np.squeeze( his_ds.variables[vn][tind,:,:] )
            z0[msk==0] = z0[msk==1][ind] # fill the mask with nearest neighbor
            setattr(interpfun,'values',z0) # change the interpolator z values
            z2 = interpfun((yy2,xx2)) # perhaps change here to directly interpolate to (xi,eta) on the edges?
            #z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the land
            OCN_BC[vn+'_south'][tind,:] = z2[0,:] # fill correctly
            OCN_BC[vn+'_north'][tind,:] = z2[-1,:]
            OCN_BC[vn+'_west'][tind,:]  = z2[:,0]
            if vn == 'zeta': # this is zeta at child grid edges, for interpolating in z later
                ZTA['rho_south'][tind,:] = z2[0,:]
                ZTA['rho_north'][tind,:] = z2[-1,:]
                ZTA['rho_west'][tind,:] = z2[:,0]
                ZTA['u_south'][tind,:] = 0.5*( z2[0,0:-1]+z2[0,1:] )
                ZTA['u_north'][tind,:] = 0.5*( z2[-1,0:-1]+z2[-1,1:] )
                ZTA['u_west'][tind,:]  = 0.5*( z2[:,0]+z2[:,1] )
                ZTA['v_south'][tind,:] = 0.5*( z2[0,:]+z2[1,:] )
                ZTA['v_north'][tind,:] = 0.5*( z2[-2,:]+z2[-1,:] )
                ZTA['v_west'][tind,:]  = 0.5*( z2[0:-1,0]+z2[1:,0] )


    for vn in v_list2:
        msk = msk_d2[vn]
        msk2 = msk2_d2[vn]
        ind = ind_d2[vn]
        xx2 = lon_d2[vn]
        yy2 = lat_d2[vn]
        interpfun = intf_d2[vn]
        for tind in np.arange(Nt):
            for zind in np.arange(Nz1):
                z0 = np.squeeze( his_ds.variables[vn][tind,zind,:,:] )
                z0[msk==0] = z0[msk==1][ind]
                setattr(interpfun,'values',z0)
                z2 = interpfun((yy2,xx2))
                z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask
                OCN_BC[vn+'_south'][tind,zind,:] = z2[0,:]
                OCN_BC[vn+'_north'][tind,zind,:] = z2[-1,:]
                OCN_BC[vn+'_west'][tind,zind,:]  = z2[:,0]

                if vn == 'temp':
                    z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                    z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                    setattr(interpfun,'values',z0)
                    z2 = interpfun((yy2,xx2))
                    ZZ['rho_north'][tind,zind,:] = z2[-1,:] # we need the depths that the horizontal interpolation thinks it is
                    ZZ['rho_south'][tind,zind,:] = z2[0,:]
                    ZZ['rho_west'][tind,zind,:] = z2[:,0]
                if vn == 'u':
                    z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                    z0 = 0.5 * ( z0[:,0:-1] + z0[:,1:] )
                    z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                    setattr(interpfun,'values',z0)
                    z2 = interpfun((yy2,xx2))
                    ZZ['u_north'][tind,zind,:] = z2[-1,:] # we need the depths that the horizontal interpolation thinks it is
                    ZZ['u_south'][tind,zind,:] = z2[0,:]
                    ZZ['u_west'][tind,zind,:] = z2[:,0]
                if vn == 'v':
                    z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                    z0 = 0.5 * ( z0[0:-1,:] + z0[1:,:] )
                    z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                    setattr(interpfun,'values',z0)
                    z2 = interpfun((yy2,xx2))
                    ZZ['v_north'][tind,zind,:] = z2[-1,:] # we need the depths that the horizontal interpolation thinks it is
                    ZZ['v_south'][tind,zind,:] = z2[0,:]
                    ZZ['v_west'][tind,zind,:] = z2[:,0]

    h = G2['h']
    hu = .5*( h[:,0:-1] + h[:,1:] )
    hv = .5*( h[0:-1,:] + h[1:,:] )

    zr2 = s_coordinate_4(h[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_north'][:,:])
    Z2['rho_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(h[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_south'][:,:])
    Z2['rho_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(h[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['rho_west'][:,:])
    Z2['rho_west'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_north'][:,:])
    Z2['u_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_south'][:,:])
    Z2['u_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hu[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['u_west'][:,:])
    Z2['u_west'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[-1,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_north'][:,:])
    Z2['v_north'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[0,:], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_south'][:,:])
    Z2['v_south'][:,:,:] = zr2.z_r[:,:,:]
    zr2 = s_coordinate_4(hv[:,0], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta = ZTA['v_west'][:,:])
    Z2['v_west'][:,:,:] = zr2.z_r[:,:,:]

    OCN_BC['Cs_r'] = np.squeeze(zr2.Cs_r)

    # now loop through all 3d variables and vertically interpolate to the correct z levels.
    for vn in v_list2:
        if vn in ['temp','salt']:
            zt = 'rho'
        else:
            zt = vn
        for bnd in ['_north','_south','_west']:
            v1 = OCN_BC[vn+bnd][:,:,:] # the horizontally interpolated field
            nnt,nnz,nnp = np.shape(v1)
            for cc in np.arange(nnt):
                for aa in np.arange(nnp):
                    vp = np.squeeze( v1[cc,:,aa]) # the vertical data
                    zp = np.squeeze( ZZ[zt+bnd][cc,:,aa] ) # z locations data thinks it is at
                    zf = np.squeeze( Z2[zt+bnd][cc,:,aa]) # z locations where the data should be
                    Fz = interp1d(zp,vp,bounds_error=False,kind='linear',fill_value = 'extrapolate') 
                    vf =  np.squeeze(Fz(zf))
                    OCN_BC[vn+bnd][cc,:,aa] = vf                

    #fn_out = '/scratch/PFM_Simulations/LV3_Forecast/Forc/test_BC_LV3.pkl'

    with open(fn_out,'wb') as fout:
        pickle.dump(OCN_BC,fout)
        print('OCN_LV',lvl,'_BC dict saved with pickle to: ',fn_out)





def mk_LV2_IC_dict(lvl):

    PFM=get_PFM_info()  
    if lvl == '2':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        fn = PFM['lv1_his_name_full']
        LV1_IC_pckl = PFM['lv1_forc_dir'] + '/' + PFM['lv1_ocnIC_tmp_pckl_file']
        lv1 = 'L1'
        lv2 = 'L2'
        fn_out = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnIC_tmp_pckl_file']
    elif lvl == '3':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        fn = PFM['lv2_his_name_full']
        LV1_IC_pckl = PFM['lv2_forc_dir'] + '/' + PFM['lv2_ocnIC_tmp_pckl_file']
        lv1 = 'L2'
        lv2 = 'L3'
        fn_out = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnIC_tmp_pckl_file']
    elif lvl == '4':
        G1 = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        G2 = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
        fn = PFM['lv3_his_name_full']
        LV1_IC_pckl = PFM['lv3_forc_dir'] + '/' + PFM['lv3_ocnIC_tmp_pckl_file']
        lv1 = 'L3'
        lv2 = 'L4'
     
    # parent vertical stretching info 
    Nz1   = PFM['stretching'][lv1,'Nz']                              # number of vertical levels: 40
    Vtr1  = PFM['stretching'][lv1,'Vtransform']                       # transformation equation: 2
    Vst1  = PFM['stretching'][lv1,'Vstretching']                    # stretching function: 4 
    th_s1 = PFM['stretching'][lv1,'THETA_S']                      # surface stretching parameter: 8
    th_b1 = PFM['stretching'][lv1,'THETA_B']                      # bottom  stretching parameter: 3
    Tcl1  = PFM['stretching'][lv1,'TCLINE']                      # critical depth (m): 50
    hc1   = PFM['stretching'][lv1,'hc']
    
    # child vertical stretching info
    Nz2   = PFM['stretching'][lv2,'Nz']                              # number of vertical levels: 40
    Vtr2  = PFM['stretching'][lv2,'Vtransform']                       # transformation equation: 2
    Vst2  = PFM['stretching'][lv2,'Vstretching']                    # stretching function: 4 
    th_s2 = PFM['stretching'][lv2,'THETA_S']                      # surface stretching parameter: 8
    th_b2 = PFM['stretching'][lv2,'THETA_B']                      # bottom  stretching parameter: 3
    Tcl2  = PFM['stretching'][lv2,'TCLINE']                      # critical depth (m): 50
    hc2   = PFM['stretching'][lv2,'hc']

    ltr1 = G1['lat_rho']
    lnr1 = G1['lon_rho']
    ltr2 = G2['lat_rho']
    lnr2 = G2['lon_rho']
    ltu1 = G1['lat_u']
    lnu1 = G1['lon_u']
    ltu2 = G2['lat_u']
    lnu2 = G2['lon_u']
    ltv1 = G1['lat_v']
    lnv1 = G1['lon_v']
    ltv2 = G2['lat_v']
    lnv2 = G2['lon_v']
    
    with open(LV1_IC_pckl,'rb') as fout:
        IC1=pickle.load(fout)
        print('LV_'+str(int(lvl)-1)+'_OCN_IC dict loaded with pickle')

 #   print(np.shape(IC1['zeta']))

    OCN_IC = dict()
    OCN_IC['vinfo'] = dict()
    OCN_IC['vinfo'] = IC1['vinfo']

    OCN_IC['Nz'] = np.squeeze(Nz2)
    OCN_IC['Vtr'] = np.squeeze(Vtr2)
    OCN_IC['Vst'] = np.squeeze(Vst2)
    OCN_IC['th_s'] = np.squeeze(th_s2)
    OCN_IC['th_b'] = np.squeeze(th_b2)
    OCN_IC['Tcl'] = np.squeeze(Tcl2)
    OCN_IC['hc'] = np.squeeze(hc2)

    OCN_IC['ocean_time']     = IC1['ocean_time']
    OCN_IC['ocean_time_ref'] = IC1['ocean_time_ref']

    OCN_IC['lat_rho'] = ltr2
    OCN_IC['lon_rho'] = lnr2
    OCN_IC['lat_u']   = ltu2
    OCN_IC['lon_u']   = lnu2
    OCN_IC['lat_v']   = ltv2
    OCN_IC['lon_v']   = lnv2

    nlt, nln = np.shape(ltr2)
    OCN_IC['temp'] = np.zeros((1,Nz2,nlt,nln))
    OCN_IC['salt'] = np.zeros((1,Nz2,nlt,nln))
    OCN_IC['u'] = np.zeros((1,Nz2,nlt,nln-1))
    OCN_IC['v'] = np.zeros((1,Nz2,nlt-1,nln))
    OCN_IC['zeta'] = np.zeros((1,nlt,nln))
    OCN_IC['ubar'] = np.zeros((1,nlt,nln-1))
    OCN_IC['vbar'] = np.zeros((1,nlt-1,nln))

    # get a dict of depths that the interpolation thinks it is on
    ZZ = dict() 
    ZZ['rho'] = np.zeros((1,Nz2,nlt,nln))
    ZZ['u'] = np.zeros((1,Nz2,nlt,nln-1))
    ZZ['v'] = np.zeros((1,Nz2,nlt-1,nln))

    # get (x,y) grids, note zi = interp_r( (eta,xi) )
    xi_r2, eta_r2, interp_r = get_child_xi_eta_interp(lnr1,ltr1,lnr2,ltr2)
    xi_u2, eta_u2, interp_u = get_child_xi_eta_interp(lnu1,ltu1,lnu2,ltu2)
    xi_v2, eta_v2, interp_v = get_child_xi_eta_interp(lnv1,ltv1,lnv2,ltv2)

    # get nearest indices, from bad indices, so that land can be filled
    indr = get_indices_to_fill(G1['mask_rho'])
    indu = get_indices_to_fill(G1['mask_u'])
    indv = get_indices_to_fill(G1['mask_v'])

    # bookkeeping so that everything needed for each variable is associated with that variable
    v_list1 = ['zeta','ubar','vbar']
    v_list2 = ['temp','salt','u','v']

    msk_d1 = dict()
    msk_d1['zeta'] = G1['mask_rho']
    msk_d1['ubar'] = G1['mask_u']
    msk_d1['vbar'] = G1['mask_v']
    msk_d2 = dict()
    msk_d2['temp'] = G1['mask_rho']
    msk_d2['salt'] = G1['mask_rho']
    msk_d2['u']    = G1['mask_u']
    msk_d2['v']    = G1['mask_v']

    msk2_d1 = dict()
    msk2_d1['zeta'] = G2['mask_rho']
    msk2_d1['ubar'] = G2['mask_u']
    msk2_d1['vbar'] = G2['mask_v']
    msk2_d2 = dict()
    msk2_d2['temp'] = G2['mask_rho']
    msk2_d2['salt'] = G2['mask_rho']
    msk2_d2['u']    = G2['mask_u']
    msk2_d2['v']    = G2['mask_v']


    ind_d1 = dict()
    ind_d1['zeta'] = indr
    ind_d1['ubar'] = indu
    ind_d1['vbar'] = indv
    ind_d2 = dict()
    ind_d2['temp'] = indr
    ind_d2['salt'] = indr
    ind_d2['u']    = indu
    ind_d2['v']    = indv

    lat_d1 = dict()
    lat_d1['zeta'] = eta_r2
    lat_d1['ubar'] = eta_u2
    lat_d1['vbar'] = eta_v2
    lat_d2 = dict()
    lat_d2['temp'] = eta_r2
    lat_d2['salt'] = eta_r2
    lat_d2['u']    = eta_u2
    lat_d2['v']    = eta_v2

    lon_d1 = dict()
    lon_d1['zeta'] = xi_r2
    lon_d1['ubar'] = xi_u2
    lon_d1['vbar'] = xi_v2
    lon_d2 = dict()
    lon_d2['temp'] = xi_r2
    lon_d2['salt'] = xi_r2
    lon_d2['u']    = xi_u2
    lon_d2['v']    = xi_v2

    intf_d1 = dict()
    intf_d1['zeta'] = interp_r
    intf_d1['ubar'] = interp_u
    intf_d1['vbar'] = interp_v
    intf_d2 = dict()
    intf_d2['temp'] = interp_r
    intf_d2['salt'] = interp_r
    intf_d2['u']    = interp_u
    intf_d2['v']    = interp_v
    
    for vn in v_list1: # loop through all 2d variables and horizontally interpolate 
        msk = msk_d1[vn] # get mask on LV1
        msk2 = msk2_d1[vn] # get mask on LV2
        ind = ind_d1[vn] # get indices so that land can be filled with nearest neighbor
        xx2 = lon_d1[vn] # get xi_LV2 on LV1
        yy2 = lat_d1[vn] # get eta_LV2 on LV1, to use with the interpolator
        interpfun = intf_d1[vn]
        tind = 0
        z0 = np.squeeze(IC1[vn][tind,:,:] )
        z0[msk==0] = z0[msk==1][ind] # fill the mask with nearest neighbor
        setattr(interpfun,'values',z0) # change the interpolator z values
        z2 = interpfun((yy2,xx2)) # perhaps change here to directly interpolate to (xi,eta) on the edges?
        z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the land
        OCN_IC[vn][tind,:,:] = z2[:,:] # fill correctly
 
    # get vertical gridding for both levels
    hraw = None
    if Vst1 == 4:
        zrom1 = s_coordinate_4(G1['h'], th_b1 , th_s1 , Tcl1 , Nz1, hraw=hraw, zeta=np.squeeze(IC1['zeta'][0,:,:]))
    if Vst2 == 4:    
        zrom2 = s_coordinate_4(G2['h'], th_b2 , th_s2 , Tcl2 , Nz2, hraw=hraw, zeta=np.squeeze(OCN_IC['zeta'][0,:,:]))

    ZZ2 = np.squeeze(zrom2.z_r[0,:,:,:])    
    OCN_IC['Cs_r'] = np.squeeze(zrom2.Cs_r)

    # now loop through all 3d variables and horizontally interpolate at each S level
    for vn in v_list2:
        msk = msk_d2[vn]
        msk2 = msk2_d2[vn]
        ind = ind_d2[vn]
        xx2 = lon_d2[vn]
        yy2 = lat_d2[vn]
        interpfun = intf_d2[vn]
        tind = 0 
        for zind in np.arange(Nz1):  # first we horizontally interpolate at each S level
            z0 = np.squeeze( IC1[vn][tind,zind,:,:] )
            z0[msk==0] = z0[msk==1][ind]
            setattr(interpfun,'values',z0)
            z2 = interpfun((yy2,xx2))
            z2[msk2==0] = np.mean(z2[msk2==1]) # put mean on the mask
            OCN_IC[vn][tind,zind,:,:] = z2[:,:]
            if vn == 'temp':
                z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                setattr(interpfun,'values',z0)
                z2 = interpfun((yy2,xx2))
                ZZ['rho'][tind,zind,:,:] = z2 # we need the depths that the horizontal interpolation thinks it is
            if vn == 'u':
                z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                z0 = 0.5 * ( z0[:,0:-1] + z0[:,1:] )
                z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                setattr(interpfun,'values',z0)
                z2 = interpfun((yy2,xx2))
                ZZ['u'][tind,zind,:,:] = z2
            if vn == 'v':
                z0 = np.squeeze( zrom1.z_r[tind,zind,:,:])
                z0 = 0.5 * ( z0[0:-1,:] + z0[1:,:] )
                z0[msk==0] = z0[msk==1][ind] # fill in masked areas with nearest neighbor
                setattr(interpfun,'values',z0)
                z2 = interpfun((yy2,xx2))
                ZZ['v'][tind,zind,:,:] = z2

    zlist = dict()
    zlist['temp'] = 'rho'
    zlist['salt'] = 'rho'
    zlist['u'] = 'u'
    zlist['v'] = 'v'

    # now loop through all 3d variables and vertically interpolate to the correct z levels.
    for vn in v_list2:
        tind = 0
        v1 = OCN_IC[vn][tind,:,:,:] # the horizontally interpolated field
        z1 = ZZ[zlist[vn]][tind,:,:,:] # the depths that the interpolation thinks it is on
        nnz,nnlt,nnln = np.shape(v1)
        for aa in np.arange(nnlt):
            for bb in np.arange(nnln):
                if vn in ['temp','salt']:
                    z2 = np.squeeze( ZZ2[:,aa,bb])
                if vn == 'u':
                    z2 = np.squeeze( 0.5*( ZZ2[:,aa,bb]+ZZ2[:,aa,bb+1] ) )
                if vn == 'v':
                    z2 = np.squeeze( 0.5*( ZZ2[:,aa,bb]+ZZ2[:,aa+1,bb] ) )

                Fz = interp1d(np.squeeze(z1[:,aa,bb]),np.squeeze(v1[:,aa,bb]),bounds_error=False,kind='linear',fill_value = 'extrapolate') 
                v2 =  np.squeeze(Fz(z2))
                OCN_IC[vn][tind,:,aa,bb] = v2                
                
#    fn_out = '/scratch/PFM_Simulations/LV3_Forecast/Forc/test_IC_LV3.pkl'
#    with open(LV2_BC_pckl,'wb') as fout:
    with open(fn_out,'wb') as fout:
        pickle.dump(OCN_IC,fout)
        print('OCN_LV'+lvl+'_IC dict saved with pickle')


def ocn_roms_IC_dict_to_netcdf_pckl(fname_in,fn_out):

    with open(fname_in,'rb') as fout:
        ATM_R=pickle.load(fout)
        print('OCN_IC dict loaded with pickle')

    # need to de-NaN some fields, fill with the mean of all non NaNs
    vns = ['temp','salt','u','v','ubar','vbar','zeta']
    for vn in vns:
        #print('doing ' + vn)
        ff = ATM_R[vn].copy()
        ff = np.nan_to_num(ff,nan=np.nanmean(ff))
        #ff = ff.filled(ff.mean())
        ATM_R[vn] = ff


    vlist = ['zeta','ubar','vbar','u','v','temp','salt']
    ulist = ['m','m/s','m/s','m/s','m/s','C','psu']
    ulist2 = dict(zip(vlist,ulist))
    
    print('\nmax and min of data in ROMS IC file (iz is bottom [0] to top [39], note: it is always 0 b/c IC):')

    for var_name in vlist:
        ind_mx = np.unravel_index(np.nanargmax(ATM_R[var_name], axis=None), ATM_R[var_name].shape)
        ind_mn = np.unravel_index(np.nanargmin(ATM_R[var_name], axis=None), ATM_R[var_name].shape)
        mxx = ATM_R[var_name][ind_mx]
        mnn = ATM_R[var_name][ind_mn]
        uvnm = ulist2[var_name]
        if var_name == 'zeta' or var_name == 'ubar' or var_name == 'vbar':
            print(f'max {var_name:6} = {mxx:6.3f} {uvnm:5}      at  ( it, ilat, ilon)     =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')
            print(f'min {var_name:6} = {mnn:6.3f} {uvnm:5}      at  ( it, ilat, ilon)     =  ({ind_mn[0]:3},{ind_mn[1]:4},{ind_mn[2]:4})')
        else:
            print(f'max {var_name:6} = {mxx:6.3f} {uvnm:5}      at  ( it, iz, ilat, ilon) =  ({ind_mx[0]:3},{ind_mx[1]:3},{ind_mx[2]:4},{ind_mx[3]:4})')
            print(f'min {var_name:6} = {mnn:6.3f} {uvnm:5}      at  ( it, iz, ilat, ilon) =  ({ind_mn[0]:3},{ind_mn[1]:3},{ind_mn[2]:4},{ind_mn[3]:4})')


    ds = xr.Dataset(
        data_vars = dict(
            temp       = (["time","s_rho","er","xr"],ATM_R['temp'],ATM_R['vinfo']['temp']),
            salt       = (["time","s_rho","er","xr"],ATM_R['salt'],ATM_R['vinfo']['salt']),
            u          = (["time","s_rho","eu","xu"],ATM_R['u'],ATM_R['vinfo']['u']),
            v          = (["time","s_rho","ev","xv"],ATM_R['v'],ATM_R['vinfo']['v']),
            ubar       = (["time","eu","xu"],ATM_R['ubar'],ATM_R['vinfo']['ubar']),
            vbar       = (["time","ev","xv"],ATM_R['vbar'],ATM_R['vinfo']['vbar']),
            zeta       = (["time","er","xr"],ATM_R['zeta'],ATM_R['vinfo']['zeta']),
            Vtransform = ([],ATM_R['Vtr'],ATM_R['vinfo']['Vtr']),
            Vstretching = ([],ATM_R['Vst'],ATM_R['vinfo']['Vst']),
            theta_s = ([],ATM_R['th_s'],ATM_R['vinfo']['th_s']),
            theta_b = ([],ATM_R['th_b'],ATM_R['vinfo']['th_b']),
            Tcline = ([],ATM_R['Tcl'],ATM_R['vinfo']['Tcl']),
            hc = ([],ATM_R['hc'],ATM_R['vinfo']['hc']),
        ),
        coords=dict(
            lat_rho =(["er","xr"],ATM_R['lat_rho'], ATM_R['vinfo']['lat_rho']),
            lon_rho =(["er","xr"],ATM_R['lon_rho'], ATM_R['vinfo']['lon_rho']),
            lat_u   =(["eu","xu"],ATM_R['lat_u'], ATM_R['vinfo']['lat_u']),
            lon_u   =(["eu","xu"],ATM_R['lon_u'], ATM_R['vinfo']['lon_u']),
            lat_v   =(["ev","xv"],ATM_R['lat_v'], ATM_R['vinfo']['lat_v']),
            lon_v   =(["ev","xv"],ATM_R['lon_v'], ATM_R['vinfo']['lon_v']),   
            ocean_time = (["time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            Cs_r = (["s_rho"],ATM_R['Cs_r'],ATM_R['vinfo']['Cs_r']),
         ),
        attrs={'type':'ocean initial condition file fields for starting roms',
            'time info':'ocean time is from '+ ATM_R['ocean_time_ref'].strftime("%Y/%m/%d %H:%M:%S") },
        )
    # print(ds)

    ds.to_netcdf(fn_out)
    ds.close()




def ocn_roms_BC_dict_to_netcdf_pckl(fname_in,fn_out):

    with open(fname_in,'rb') as fout:
        ATM_R=pickle.load(fout)
        print('OCN_BC dict loaded with pickle')

    # lets replace NaNs if there are any
    vns = ['temp','salt','u','v','ubar','vbar','zeta']
    sds = ['_north','_south','_west']
    for vn in vns:
        for sd in sds:
            #print('doing ' + vn)
            ff = ATM_R[vn+sd].copy()
            ff = np.nan_to_num(ff,nan=np.nanmean(ff))
            #ff = ff.filled(ff.mean())
            ATM_R[vn+sd] = ff
    
    vlist = ['zeta','ubar','vbar','u','v','temp','salt']
    ulist = ['m','m/s','m/s','m/s','m/s','C','psu']
    ulist2 = dict(zip(vlist,ulist))
    
    print('\nmax and min of data in ROMS BC file (iz is bottom [0] to top [39]):')

    for var_name in vlist:
        for sd in sds:
            ind_mx = np.unravel_index(np.nanargmax(ATM_R[var_name+sd], axis=None), ATM_R[var_name+sd].shape)
            ind_mn = np.unravel_index(np.nanargmin(ATM_R[var_name+sd], axis=None), ATM_R[var_name+sd].shape)
            mxx = ATM_R[var_name+sd][ind_mx]
            mnn = ATM_R[var_name+sd][ind_mn]
            uvnm = ulist2[var_name]
            if var_name == 'zeta' or var_name == 'ubar' or var_name == 'vbar':
                if sd == '_north' or sd == '_south':
                    print(f'max {var_name+sd:10} = {mxx:6.3f} {uvnm:5}      at  ( it, ilon)     =  ({ind_mx[0]:3},{ind_mx[1]:4})')
                    print(f'min {var_name+sd:10} = {mnn:6.3f} {uvnm:5}      at  ( it, ilon)     =  ({ind_mn[0]:3},{ind_mn[1]:4})')
                else:
                    print(f'max {var_name+sd:10} = {mxx:6.3f} {uvnm:5}      at  ( it, ilat)     =  ({ind_mx[0]:3},{ind_mx[1]:4})')
                    print(f'min {var_name+sd:10} = {mnn:6.3f} {uvnm:5}      at  ( it, ilat)     =  ({ind_mn[0]:3},{ind_mn[1]:4})')
            else:
                if sd == '_north' or sd == '_south':
                    print(f'max {var_name+sd:10} = {mxx:6.3f} {uvnm:5}      at  ( it, iz, ilon) =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')
                    print(f'min {var_name+sd:10} = {mnn:6.3f} {uvnm:5}      at  ( it, iz, ilon) =  ({ind_mn[0]:3},{ind_mn[1]:4},{ind_mn[2]:4})')
                else:
                    print(f'max {var_name+sd:10} = {mxx:6.3f} {uvnm:5}      at  ( it, iz, ilat) =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')
                    print(f'min {var_name+sd:10} = {mnn:6.3f} {uvnm:5}      at  ( it, iz, ilat) =  ({ind_mn[0]:3},{ind_mn[1]:4},{ind_mn[2]:4})')


    ds = xr.Dataset(
        data_vars = dict(
            temp_south       = (["temp_time","s_rho","xr"],ATM_R['temp_south'],ATM_R['vinfo']['temp_south']),
            salt_south       = (["salt_time","s_rho","xr"],ATM_R['salt_south'],ATM_R['vinfo']['salt_south']),
            u_south          = (["v3d_time","s_rho","xu"],ATM_R['u_south'],ATM_R['vinfo']['u_south']),
            v_south          = (["v3d_time","s_rho","xv"],ATM_R['v_south'],ATM_R['vinfo']['v_south']),
            ubar_south       = (["v2d_time","xu"],ATM_R['ubar_south'],ATM_R['vinfo']['ubar_south']),
            vbar_south       = (["v2d_time","xv"],ATM_R['vbar_south'],ATM_R['vinfo']['vbar_south']),
            zeta_south       = (["zeta_time","xr"],ATM_R['zeta_south'],ATM_R['vinfo']['zeta_south']),
            temp_north       = (["temp_time","s_rho","xr"],ATM_R['temp_north'],ATM_R['vinfo']['temp_north']),
            salt_north       = (["salt_time","s_rho","xr"],ATM_R['salt_north'],ATM_R['vinfo']['salt_north']),
            u_north          = (["v3d_time","s_rho","xu"],ATM_R['u_north'],ATM_R['vinfo']['u_north']),
            v_north          = (["v3d_time","s_rho","xv"],ATM_R['v_north'],ATM_R['vinfo']['v_north']),
            ubar_north       = (["v2d_time","xu"],ATM_R['ubar_north'],ATM_R['vinfo']['ubar_north']),
            vbar_north       = (["v2d_time","xv"],ATM_R['vbar_north'],ATM_R['vinfo']['vbar_north']),
            zeta_north       = (["zeta_time","xr"],ATM_R['zeta_north'],ATM_R['vinfo']['zeta_north']),
            temp_west        = (["temp_time","s_rho","er"],ATM_R['temp_west'],ATM_R['vinfo']['temp_west']),
            salt_west        = (["salt_time","s_rho","er"],ATM_R['salt_west'],ATM_R['vinfo']['salt_west']),
            u_west           = (["v3d_time","s_rho","eu"],ATM_R['u_west'],ATM_R['vinfo']['u_west']),
            v_west           = (["v3d_time","s_rho","ev"],ATM_R['v_west'],ATM_R['vinfo']['v_west']),
            ubar_west        = (["v2d_time","eu"],ATM_R['ubar_west'],ATM_R['vinfo']['ubar_west']),
            vbar_west        = (["v2d_time","ev"],ATM_R['vbar_west'],ATM_R['vinfo']['vbar_west']),
            zeta_west        = (["zeta_time","er"],ATM_R['zeta_west'],ATM_R['vinfo']['zeta_west']),
            Vtransform       = ([],ATM_R['Vtr'],ATM_R['vinfo']['Vtr']),
            Vstretching      = ([],ATM_R['Vst'],ATM_R['vinfo']['Vst']),
            theta_s          = ([],ATM_R['th_s'],ATM_R['vinfo']['th_s']),
            theta_b          = ([],ATM_R['th_b'],ATM_R['vinfo']['th_b']),
            Tcline           = ([],ATM_R['Tcl'],ATM_R['vinfo']['Tcl']),
            hc               = ([],ATM_R['hc'],ATM_R['vinfo']['hc']),
        ),
        coords=dict(
            ocean_time = (["time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            zeta_time  = (["zeta_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            v2d_time   = (["v2d_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            v3d_time   = (["v3d_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            salt_time  = (["salt_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            temp_time  = (["temp_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            Cs_r       = (["s_rho"],ATM_R['Cs_r'],ATM_R['vinfo']['Cs_r']),
         ),
        attrs={'type':'ocean boundary condition file fields for starting roms',
            'time info':'ocean time is from '+ ATM_R['ocean_time_ref'].strftime("%Y/%m/%d %H:%M:%S") },
        )
    # print(ds)

    ds.to_netcdf(fn_out)
    ds.close()

def ocn_roms_BC_dict_to_netcdf_pckl_1hrzeta(fname_in,fn_out):

    with open(fname_in,'rb') as fout:
        ATM_R=pickle.load(fout)
        print('OCN_BC dict loaded with pickle')

    # lets replace NaNs if there are any
    vns = ['temp','salt','u','v','ubar','vbar','zeta']
    sds = ['_north','_south','_west']
    for vn in vns:
        for sd in sds:
            #print('doing ' + vn)
            ff = ATM_R[vn+sd].copy()
            ff = np.nan_to_num(ff,nan=np.nanmean(ff))
            #ff = ff.filled(ff.mean())
            ATM_R[vn+sd] = ff
    
    vlist = ['zeta','ubar','vbar','u','v','temp','salt']
    ulist = ['m','m/s','m/s','m/s','m/s','C','psu']
    ulist2 = dict(zip(vlist,ulist))
    
    print('\nmax and min of data in ROMS BC file (iz is bottom [0] to top [39]):')

    for var_name in vlist:
        for sd in sds:
            ind_mx = np.unravel_index(np.nanargmax(ATM_R[var_name+sd], axis=None), ATM_R[var_name+sd].shape)
            ind_mn = np.unravel_index(np.nanargmin(ATM_R[var_name+sd], axis=None), ATM_R[var_name+sd].shape)
            mxx = ATM_R[var_name+sd][ind_mx]
            mnn = ATM_R[var_name+sd][ind_mn]
            uvnm = ulist2[var_name]
            if var_name == 'zeta' or var_name == 'ubar' or var_name == 'vbar':
                if sd == '_north' or sd == '_south':
                    print(f'max {var_name+sd:10} = {mxx:6.3f} {uvnm:5}      at  ( it, ilon)     =  ({ind_mx[0]:3},{ind_mx[1]:4})')
                    print(f'min {var_name+sd:10} = {mnn:6.3f} {uvnm:5}      at  ( it, ilon)     =  ({ind_mn[0]:3},{ind_mn[1]:4})')
                else:
                    print(f'max {var_name+sd:10} = {mxx:6.3f} {uvnm:5}      at  ( it, ilat)     =  ({ind_mx[0]:3},{ind_mx[1]:4})')
                    print(f'min {var_name+sd:10} = {mnn:6.3f} {uvnm:5}      at  ( it, ilat)     =  ({ind_mn[0]:3},{ind_mn[1]:4})')
            else:
                if sd == '_north' or sd == '_south':
                    print(f'max {var_name+sd:10} = {mxx:6.3f} {uvnm:5}      at  ( it, iz, ilon) =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')
                    print(f'min {var_name+sd:10} = {mnn:6.3f} {uvnm:5}      at  ( it, iz, ilon) =  ({ind_mn[0]:3},{ind_mn[1]:4},{ind_mn[2]:4})')
                else:
                    print(f'max {var_name+sd:10} = {mxx:6.3f} {uvnm:5}      at  ( it, iz, ilat) =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')
                    print(f'min {var_name+sd:10} = {mnn:6.3f} {uvnm:5}      at  ( it, iz, ilat) =  ({ind_mn[0]:3},{ind_mn[1]:4},{ind_mn[2]:4})')


    ds = xr.Dataset(
        data_vars = dict(
            temp_south       = (["temp_time","s_rho","xr"],ATM_R['temp_south'],ATM_R['vinfo']['temp_south']),
            salt_south       = (["salt_time","s_rho","xr"],ATM_R['salt_south'],ATM_R['vinfo']['salt_south']),
            u_south          = (["v3d_time","s_rho","xu"],ATM_R['u_south'],ATM_R['vinfo']['u_south']),
            v_south          = (["v3d_time","s_rho","xv"],ATM_R['v_south'],ATM_R['vinfo']['v_south']),
            ubar_south       = (["v2d_time","xu"],ATM_R['ubar_south'],ATM_R['vinfo']['ubar_south']),
            vbar_south       = (["v2d_time","xv"],ATM_R['vbar_south'],ATM_R['vinfo']['vbar_south']),
            zeta_south       = (["zeta_time","xr"],ATM_R['zeta_south'],ATM_R['vinfo']['zeta_south']),
            temp_north       = (["temp_time","s_rho","xr"],ATM_R['temp_north'],ATM_R['vinfo']['temp_north']),
            salt_north       = (["salt_time","s_rho","xr"],ATM_R['salt_north'],ATM_R['vinfo']['salt_north']),
            u_north          = (["v3d_time","s_rho","xu"],ATM_R['u_north'],ATM_R['vinfo']['u_north']),
            v_north          = (["v3d_time","s_rho","xv"],ATM_R['v_north'],ATM_R['vinfo']['v_north']),
            ubar_north       = (["v2d_time","xu"],ATM_R['ubar_north'],ATM_R['vinfo']['ubar_north']),
            vbar_north       = (["v2d_time","xv"],ATM_R['vbar_north'],ATM_R['vinfo']['vbar_north']),
            zeta_north       = (["zeta_time","xr"],ATM_R['zeta_north'],ATM_R['vinfo']['zeta_north']),
            temp_west        = (["temp_time","s_rho","er"],ATM_R['temp_west'],ATM_R['vinfo']['temp_west']),
            salt_west        = (["salt_time","s_rho","er"],ATM_R['salt_west'],ATM_R['vinfo']['salt_west']),
            u_west           = (["v3d_time","s_rho","eu"],ATM_R['u_west'],ATM_R['vinfo']['u_west']),
            v_west           = (["v3d_time","s_rho","ev"],ATM_R['v_west'],ATM_R['vinfo']['v_west']),
            ubar_west        = (["v2d_time","eu"],ATM_R['ubar_west'],ATM_R['vinfo']['ubar_west']),
            vbar_west        = (["v2d_time","ev"],ATM_R['vbar_west'],ATM_R['vinfo']['vbar_west']),
            zeta_west        = (["zeta_time","er"],ATM_R['zeta_west'],ATM_R['vinfo']['zeta_west']),
            Vtransform       = ([],ATM_R['Vtr'],ATM_R['vinfo']['Vtr']),
            Vstretching      = ([],ATM_R['Vst'],ATM_R['vinfo']['Vst']),
            theta_s          = ([],ATM_R['th_s'],ATM_R['vinfo']['th_s']),
            theta_b          = ([],ATM_R['th_b'],ATM_R['vinfo']['th_b']),
            Tcline           = ([],ATM_R['Tcl'],ATM_R['vinfo']['Tcl']),
            hc               = ([],ATM_R['hc'],ATM_R['vinfo']['hc']),
        ),
        coords=dict(
            ocean_time = (["time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            zeta_time  = (["zeta_time"],ATM_R['zeta_time'], ATM_R['vinfo']['zeta_time']),
            v2d_time   = (["v2d_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            v3d_time   = (["v3d_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            salt_time  = (["salt_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            temp_time  = (["temp_time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            Cs_r       = (["s_rho"],ATM_R['Cs_r'],ATM_R['vinfo']['Cs_r']),
         ),
        attrs={'type':'ocean boundary condition file fields for starting roms',
            'time info':'ocean time is from '+ ATM_R['ocean_time_ref'].strftime("%Y/%m/%d %H:%M:%S") },
        )
    # print(ds)

   


    ds.to_netcdf(fn_out)
    ds.close()    

if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])

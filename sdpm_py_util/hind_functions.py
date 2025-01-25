# functions specific to hindcasting will be here

from datetime import datetime
from datetime import timedelta
import time
import gc
import resource
import pickle
import grid_functions as grdfuns
import river_functions as rivfuns
import os
import os.path
import pickle
from scipy.spatial import cKDTree
import glob
import requests
import grib2io

#sys.path.append('../sdpm_py_util')
from get_PFM_info import get_PFM_info

import numpy as np
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset


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


def get_nam_hindcast_filelists(t1str,t2str):

    PFM = get_PFM_info()
    PFM['atm_hind_dir'] = '/scratch/PHM_Simulations/grb2_data'

    atm_hind_dir = PFM['atm_hind_dir']
    url = 'https://www.ncei.noaa.gov/data/north-american-mesoscale-model/access/analysis/'
    txt1 = '/nam_218_'
    txt2 = '.grb2'
    fns_in = [] # this is the list of filename we are going to download
    fns_out = [] # this is the list of filenames we are saving the data to
    cmd_lst = [] # this is the list of commands that we are going to subprocess
    pckl_nms = [] # this is the list of pckl file names for later
    dt3hr = 3.0 * timedelta(hours=1)

    t1 = datetime.strptime(t1str,'%Y%m%d%H')
    t2 = datetime.strptime(t2str,'%Y%m%d%H')
    tt = t1
    hh = tt.hour # this is an integer
    hh0_txt = str(hh).zfill(2) + '00'
    while tt <= t2:
        hh = hh % 24 
        yyyymm = tt.strftime("%Y%m")
        yyyymmdd = tt.strftime("%Y%m%d")
        hr2 = hh % 6
        hr1 = hh - hr2
        hr1_txt = str(hr1).zfill(2) + '00'
        hr2_txt = str(hr2).zfill(3)
        url2 = url + yyyymm +'/' + yyyymmdd + txt1 + yyyymmdd + '_' + hr1_txt + '_' + hr2_txt + txt2
        fns_in.append(url2)
        fno = atm_hind_dir + txt1 + yyyymmdd + '_' + hr1_txt + '_' + hr2_txt + txt2
        fns_out.append(fno)
        fnp = atm_hind_dir + txt1 + yyyymmdd + '_' + hr1_txt + '_' + hr2_txt + '.pkl'
        pckl_nms.append(fnp)
        cmd = ['wget','-q','-O',fno,url2]
        cmd_lst.append(cmd)
        # increment time and hour by 3 hr
        tt = tt + dt3hr
        hh = hh + 3

    return fns_in, fns_out, cmd_lst, pckl_nms

def nam_grabber_hind(cmd):

    # run ncks
#    ret1 = subprocess.call(cmd,stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
    ret1 = subprocess.call(cmd)
    #print(cmd_list)
    #ret1 = 1
    return ret1

def get_nam_hindcast_grb2s(t1str,t2str):
#wget https://www.ncei.noaa.gov/data/north-american-mesoscale-model/access/analysis/202412/20241231/nam_218_20241231_0000_000.grb2
    PFM = get_PFM_info()
    _, _, cmd_list, _ = get_nam_hindcast_filelists(t1str,t2str)

   # create parallel executor
    with ThreadPoolExecutor() as executor:
        threads = []
        cnt = 0
        for cmd in cmd_list:
            #print(cnt)
            fun = nam_grabber_hind #define function
            args = [cmd] #define args to function
            kwargs = {} #
            # start thread by submitting it to the executor
            threads.append(executor.submit(fun, *args, **kwargs))
            cnt=cnt+1

        result2 = []
        for future in as_completed(threads):
            # retrieve the result
            result = future.result()
            result2.append(result)
            # report the result

    res3 = result2.copy()
    res3 = [1 if x == 0 else x for x in res3]
    nff = sum(res3)
    if nff == len(cmd_list):
        print('things are good, we got all ' + str(nff) + ' nam files')
    else:
        print('things arent so good.')
        print('we got ' + str(nff) + ' files of ' + str(len(cmd_list)) + ' we tried to get.')

    return result2

def grb2_to_pickle(fn_in,fn_out):
    PFM = get_PFM_info()
    # this function is going to load the .grb2 file, and then cut out a portion of it,
    # get the variables and data we need, and save to a pickle file.

    lt_mn = 118 # these are the indices of the nam 12 km simulation that get 
    lt_mx = 191 # a box of nam data that LV1 grid is totall within
    ln_mn = 129
    ln_mx = 185

    g = grib2io.open(fn_in)

    vars_in=['PRMSL','RH','TMP','UGRD','VGRD','APCP','DSWRF','USWRF','DLWRF','ULWRF']
    lev =['mean sea level','2 m above ground','2 m above ground','10 m above ground','10 m above ground','surface','surface','surface','surface','surface']
    vars_out=['Pair','','Tair','Uwind','Vwind','rain',]

    ATM = dict()
    # get precipitation (over 1 hr), total kg / m2
    AA1 = g.select(shortName='APCP',level='surface')[0]
    AA2 = AA1.data
    AA3 = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    AA3 = AA3 / 3600.0 # now precipitation rate in kg/m2/s
    ATM['rain'] = AA3

    lats, lons = AA1.grid()
    lons2 = lons[lt_mn:lt_mx,ln_mn:ln_mx]
    lats2 = lats[lt_mn:lt_mx,ln_mn:ln_mx]
    ATM['lat'] = lats2
    ATM['lon'] = lons2

    time = AA1.validDate # a datetime object
    t_ref = PFM['modtime0'] # a datetime object
    Dt = time - t_ref # a timedelta object
    t_rom = Dt.total_seconds() / (24.0 * 3600.0) # now days past t_ref
    ATM['ocean_time'] = t_rom
    ATM['pair_time'] = t_rom
    ATM['tair_time'] = t_rom
    ATM['qair_time'] = t_rom
    ATM['srf_time'] = t_rom
    ATM['lrf_time'] = t_rom
    ATM['wind_time'] = t_rom
    ATM['rain_time'] = t_rom
    ATM['lrf_time'] = t_rom
    ATM['ocean_time_ref'] = t_ref


    # get temp, K
    AA1 = g.select(shortName='TMP',level='2 m above ground')[0]
    AA2 = AA1.data
    AA3 = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    AA3 = AA3 - 273.15 # now in C
    ATM['Tair'] = AA3

    # get humidity, percent
    AA1 = g.select(shortName='RH',level='2 m above ground')[0]
    AA2 = AA1.data
    AA3 = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    ATM['Qair'] = AA3

    # get pressure at mean seal level, Pa
    AA1 = g.select(shortName='PRMSL',level='mean sea level')[0]
    AA2 = AA1.data
    AA3 = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    AA3 = 0.01 * AA3 # convert from Pa to db
    ATM['Pair'] = AA3

    # get east west 10 m winds, m/s
    AA1 = g.select(shortName='UGRD',level='10 m above ground')[0]
    AA2 = AA1.data
    AA3 = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    ATM['Uwind'] = AA3

    # get north south 10 m winds, m/s
    AA1 = g.select(shortName='VGRD',level='10 m above ground')[0]
    AA2 = AA1.data
    AA3 = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    ATM['Vwind'] = AA3

    # get LW down, watts / m2
    AA1 = g.select(shortName='DLWRF',level='surface')[0]
    AA2 = AA1.data
    lwd = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    # get SW down, watts / m2
    AA1 = g.select(shortName='DSWRF',level='surface')[0]
    AA2 = AA1.data
    swd = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    # get LW up, watts / m2
    AA1 = g.select(shortName='ULWRF',level='surface')[0]
    AA2 = AA1.data
    lwu = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    # get SW up, watts / m2
    AA1 = g.select(shortName='USWRF',level='surface')[0]
    AA2 = AA1.data
    swu = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    
    ATM['lwrad'] = lwd-lwu
    ATM['lwrad_down'] = lwd
    ATM['swrad'] = swd-swu
    
    ATM['vinfo'] = dict()  
    # put the units in atm...
    ATM['vinfo']['lon'] = {'long_name':'longitude',
                    'units':'degrees_east'}
    ATM['vinfo']['lat'] = {'long_name':'latitude',
                    'units':'degrees_north'}
    ATM['vinfo']['ocean_time'] = {'long_name':'atmospheric forcing time',
                        'units':'days',
                        'field': 'time, scalar, series'}
    ATM['vinfo']['rain_time'] = {'long_name':'atmospheric rain forcing time',
                        'units':'days',
                        'field': 'time, scalar, series'}
    ATM['vinfo']['wind_time'] = {'long_name':'atmospheric wind forcing time',
                        'units':'days',
                        'field': 'time, scalar, series'}
    ATM['vinfo']['tair_time'] = {'long_name':'atmospheric temp forcing time',
                        'units':'days',
                        'field': 'time, scalar, series'}
    ATM['vinfo']['pair_time'] = {'long_name':'atmospheric pressure forcing time',
                        'units':'days',
                        'field': 'time, scalar, series'}
    ATM['vinfo']['qair_time'] = {'long_name':'atmospheric humidity forcing time',
                        'units':'days',
                        'field': 'time, scalar, series'}
    ATM['vinfo']['srf_time'] = {'long_name':'atmospheric short wave radiation forcing time',
                        'units':'days',
                        'field': 'time, scalar, series'}
    ATM['vinfo']['lrf_time'] = {'long_name':'atmospheric long wave radiation forcing time',
                        'units':'days',
                        'field': 'time, scalar, series'}
    ATM['vinfo']['Tair'] = {'long_name':'surface air temperature',
                    'units':'degrees C',
                    'coordinates':'lat,lon',
                    'time':'tair_time'}
    ATM['vinfo']['Pair'] = {'long_name':'surface air pressure',
                    'units':'mb',
                    'coordinates':'lat,lon',
                    'time':'pair_time'}
    ATM['vinfo']['Qair'] = {'long_name':'surface air relative humidity',
                    'units':'percent [%]',
                    'coordinates':'lat,lon',
                    'time':'qair_time'}
    ATM['vinfo']['rain'] = {'long_name':'precipitation rate',
                    'units':'kg/m^2/s',
                    'coordinates':'lat,lon',
                    'time':'rain_time'}
    ATM['vinfo']['swrad'] = {'long_name':'net solar short wave radiation flux down',
                    'units':'W/m^2',
                    'coordinates':'lat,lon',
                    'time':'srf_time',
                    'negative values': 'upward flux, cooling',
                    'positive values': 'downward flux, warming'}
    ATM['vinfo']['lwrad'] = {'long_name':'net solar long wave radiation flux down',
                    'units':'W/m^2',
                    'coordinates':'lat,lon',
                    'time':'lrf_time',
                    'negative values': 'upward flux, cooling',
                    'positive values': 'downward flux, warming'}
    ATM['vinfo']['lwrad_down'] = {'long_name':'solar long wave down radiation flux',
                    'units':'W/m^2',
                    'coordinates':'lat,lon',
                    'time':'lrf_time',
                    'note' : 'this is the downward component of the flux, warming'}
    ATM['vinfo']['Uwind'] = {'long_name':'roms east coordinate, er, velocity',
                    'units':'m/s',
                    'coordinates':'lat,lon',
                    'time':'wind_time',
                    'note':'these velocity velocities are in earth coordinate'}
    ATM['vinfo']['Vwind'] = {'long_name':'roms north coordinate, xi, velocity',
                    'units':'m/s',
                    'coordinates':'lat,lon',
                    'time':'wind_time',
                    'note':'these velocity velocities are in earth coordinate'}


    with open(fn_out,'wb') as fp:
        pickle.dump(ATM,fp)
        print('ATM grb2 file ' + fn_in + ' saved to pickle.')


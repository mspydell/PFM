# functions specific to hindcasting will be here

from datetime import datetime
from datetime import timedelta
#import time
#import gc
#import resource
import pickle
import os
import os.path
from scipy.spatial import cKDTree
#import glob
#import requests
import grib2io
import sys

sys.path.append('../sdpm_py_util')
import grid_functions as grdfuns
import ocn_functions as ocnfuns
from get_PFM_info import get_PFM_info

import numpy as np
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset

from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.interpolate import interp1d

#import seawater
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

#import util_functions as utlfuns 
from util_functions import s_coordinate_4
#from pydap.client import open_url
#import sys

import warnings
warnings.filterwarnings("ignore")

def list_to_dict_of_chunks(long_list, chunk_size=10):
    """
    Converts a long list into a dictionary of lists, with each list (chunk)
    containing a maximum of chunk_size elements.
    
    Args:
        long_list: The original list.
        chunk_size: The maximum size of each chunk (default is 10).
    
    Returns:
        A dictionary where keys are chunk numbers (starting from 1) and
        values are the corresponding list chunks.
    """
    dict_of_chunks = {}
    for i in range(0, len(long_list), chunk_size):
        chunk = long_list[i:i + chunk_size]
        dict_of_chunks[i // chunk_size + 1] = chunk
    return dict_of_chunks


def get_nam_hindcast_filelists(t1str,t2str):

    #PFM = get_PFM_info()
    PFM = {}
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

def get_nam_hindcast_grb2s_v2(t1str,t2str):
    _, l2, cmd_list0, _ = get_nam_hindcast_filelists(t1str,t2str)
    # check and see if the grb2 files are already there?
    fes = [] # a list of 0 or -1
    for fn in l2:
        fe = check_file_exists_os(fn)
        fes.append(fe)
    
    if sum(fes) == len(l2):
        print('the ', len(l2), ' grb2 files already exist, no need to download.')
        result2 = 0
        return result2
    
    cmd_list_2 = list_to_dict_of_chunks(cmd_list0, chunk_size=5)
    result2 = []

    nchnk = len(cmd_list_2)
    for cnt in np.arange(nchnk):
    #for cmd_list in list(cmd_list_2.keys()):
        print('getting 5 nam files...')
        with ThreadPoolExecutor() as executor:
            threads = []
            cnt = 0
            for cmd in cmd_list_2[cnt+1]:
                #print(cnt)
                fun = nam_grabber_hind #define function
                args = [cmd] #define args to function
                kwargs = {} #
                # start thread by submitting it to the executor
                threads.append(executor.submit(fun, *args, **kwargs))
                cnt=cnt+1

            for future in as_completed(threads):
                # retrieve the result
                result = future.result()
                result2.append(result)
                # report the result

    res3 = result2.copy()
    res3 = [1 if x == 0 else x for x in res3]
    nff = sum(res3)
    if nff == len(cmd_list0):
        print('things are good, we got all ' + str(nff) + ' nam files')
    else:
        print('things arent so good.')
        print('we got ' + str(nff) + ' files of ' + str(len(cmd_list0)) + ' we tried to get.')

    return result2    

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

        print('sent off all wget requests')
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

    #vars_in=['PRMSL','RH','TMP','UGRD','VGRD','APCP','DSWRF','USWRF','DLWRF','ULWRF']
    #lev =['mean sea level','2 m above ground','2 m above ground','10 m above ground','10 m above ground','surface','surface','surface','surface','surface']
    #vars_out=['Pair','','Tair','Uwind','Vwind','rain',]

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
    ATM['vinfo']['ocean_time_ref'] = {'long_name':'the reference time that roms starts from',
                            'units':'datetime object',
                            'field': 'time, scalar'}
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


def grb2s_to_pickles(t1str,t2str):
    # this function takes all of the grb2 nam files and makes pickles out of them
    
    # get file names
    _, l2, _, l4 = get_nam_hindcast_filelists(t1str,t2str)

    for j in np.arange(len(l2)):
        fn_in = l2[j]
        fn_out = l4[j]
        grb2_to_pickle(fn_in,fn_out)


def check_file_exists_os(file_path):
    """
    Checks if a file exists using os.path.exists() and returns 1 if it exists, 0 otherwise.
    """
    return 1 if os.path.exists(file_path) else 0

def load_pickle_file(file_path):
    """
    Loads data from a pickle file.
    Args:
        file_path (str): The path to the pickle file.
    Returns:
        object: The unpickled object, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def nam_pkls_2_romsatm_pkl(t1str,t2str,lv):
    # this function takes nam_hindcast pickle files, interpolates the fields
    # from the nam grid to the rom grid at level = lv

    PFM = get_PFM_info()
    if lv == '1':
        RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
        fname_out = PFM['lv1_forc_dir'] + '/' + PFM['atm_tmp_LV1_pckl_file']
    elif lv == '2':
        RMG = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        fname_out = PFM['lv2_forc_dir'] + '/' + PFM['atm_tmp_LV2_pckl_file']
    elif lv == '3':
        RMG = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        fname_out = PFM['lv3_forc_dir'] + '/' + PFM['atm_tmp_LV3_pckl_file']
    else:
        RMG = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
        fname_out = PFM['lv4_forc_dir'] + '/' + PFM['atm_tmp_LV4_pckl_file']

    _, _, _, fn_pkls = get_nam_hindcast_filelists(t1str,t2str)

    # check and make sure pkl files exist
    fes = [] # a list of 0 or -1
    for fn in fn_pkls:
        #print(fn)
        fe = check_file_exists_os(fn)
        fes.append(fe-1)

    fes_test = sum(fes)
    if fes_test == 0:
        print('the nam atm pickle files exist, will load and interpolate onto the roms grid...')
        fes_test = 1
    else:
        print('not all pickle files in the time range ',t1str,' to ', t2str, ' were found.')
        print('Please supply a time range with pickle files. Exiting!!!')
        fes_test = 0
        return fes_test
    
    # list of variables that we need to go from nam to roms grid
    vars = ['rain','Tair','Qair','Pair','Uwind','Vwind','lwrad','lwrad_down','swrad']
    nt = len(fn_pkls)
    nltr,nlnr = np.shape(RMG['lat_rho'])

    atm2 = dict()
    atm2['vinfo'] = dict()
    
    for var in vars:
        atm2[var]=np.zeros((nt,nltr,nlnr))
    atm2['ocean_time'] = np.zeros((nt))

    ATM = load_pickle_file(fn_pkls[0])
    vlist = ['lon','lat','ocean_time','ocean_time_ref','lwrad','lwrad_down','swrad','rain','Tair','Pair','Qair','Uwind','Vwind','tair_time','pair_time','qair_time','wind_time','rain_time','srf_time','lrf_time']
    for aa in vlist:
        atm2['vinfo'][aa] = ATM['vinfo'][aa]

    # the lat lons are from the roms grid
    atm2['lat'] = RMG['lat_rho']
    atm2['lon'] = RMG['lon_rho']

    # these two are useful later
    atm2['ocean_time_ref'] = ATM['ocean_time_ref']
    
    set_up = 0 # this flag is to set up the interpolators
    cnt_time = 0
    for fn in fn_pkls: # this loops over all times
        # load the file
        atm = load_pickle_file(fn)
        atm2['ocean_time'][cnt_time] = atm['ocean_time']
        
        for var in vars:
            z0  = atm[var]

            if set_up == 0:
                # this will set up all of the interpolating that needs to be done...
                # this is only done once!
                lat_nam = atm['lat']
                lon_nam = atm['lon']
                #nj,ni = np.shape(lat_nam) # how big is the nam grid? ny, nx
                #i_nam = np.arange(ni) # nam lon counting vector 
                #j_nam = np.arange(nj) # nam lat counting vector
                #Xind,Yind = np.meshgrid(i_nam,j_nam) 

                #points1 = np.zeros( (ni*nj, 2) )
                #points1[:,1] = lon_nam.flatten()
                #points1[:,0] = lat_nam.flatten()
                #scat_interp_xi  = LinearNDInterpolator(points1,Xind.flatten())
                #scat_interp_eta = LinearNDInterpolator(points1,Yind.flatten())
                
                lt2 = RMG['lat_rho']
                ln2 = RMG['lon_rho']
                # need roms indices
                #xi_2  = scat_interp_xi(lt2,ln2) # roms lat lon in nam x indices
                #eta_2 = scat_interp_eta(lt2,ln2) # roms lat lon in nam y indices
                # we need the interpolator object
                #interper = RegularGridInterpolator( (j_nam , i_nam), lat_nam , bounds_error=False, fill_value=None)                
                xi_2, eta_2, interper = ocnfuns.get_child_xi_eta_interp(lon_nam,lat_nam,ln2,lt2,'rho')
                set_up = 1

            setattr(interper,'values',z0) # change the interpolator z values
            z2 =  interper((eta_2,xi_2)) # perhaps change here to directly interpolate to (xi,eta) on the edges?
            atm2[var][cnt_time,:,:] = z2

        cnt_time = cnt_time + 1

    # copy times to all time vars
    tlist = ['tair_time','pair_time','qair_time','wind_time','rain_time','srf_time','lrf_time']
    for tnm in tlist:
        atm2[tnm] = atm2['ocean_time']

    with open(fname_out,'wb') as fp:
        pickle.dump(atm2,fp)
        print('\nATM on roms grid dict saved with pickle.')
    
    return fes_test




                






        
    


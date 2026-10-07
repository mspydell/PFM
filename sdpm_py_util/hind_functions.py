# functions specific to hindcasting will be here

from datetime import datetime, timedelta
import pickle
import os
import os.path
from scipy.spatial import cKDTree
import grib2io
import sys

sys.path.append('../sdpm_py_util')
import grid_functions as grdfuns
import ocn_funs_forecast as ocnfuns
import init_funs as initfuns

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


def get_nam_hindcast_filelists(t1str,t2str,pkl_fnm):

    # new amazon url
    # https://noaa-nam-pds.s3.amazonaws.com.nam.YYYYMMDD/nam.tHHz.[product]FF.tm00.grib2
    # for example, the 2025-08-23 00:00 forecast, 00 hours out (analysis) is:
    # https://noaa-nam-pds.s3.amazonaws.com/nam.20250823/nam.t00z.awip1200.tm00.grib2'
    # for example, the 2025-08-23 00:00 forecast, 03 hours out (NOT analysis) is:
    # https://noaa-nam-pds.s3.amazonaws.com/nam.20250823/nam.t00z.awip1203.tm00.grib2'

    PFM = initfuns.get_model_info( pkl_fnm )

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

def get_nam_hindcast_grb2s_v2(t1str,t2str,pkl_fnm):
    _, l2, cmd_list0, _ = get_nam_hindcast_filelists(t1str,t2str,pkl_fnm)
    # check and see if the grb2 files are already there?
    fes = [] # a list of 0 (dont have) or 1 (have)
    for fn in l2:
        fe = check_file_exists_os(fn)
        fes.append(fe)
    
    if sum(fes) == len(l2):
        print('the ', len(l2), ' grb2 files already exist, no need to download.')
        result2 = 0
        return result2
    elif sum(fes)>0 and sum(fes)<len(l2):
        print('some, but not all grb files are missing, we will try and get them...')
        cmd_list1 = [cmd_list0[i] for i in range(len(fes)) if fes[i] == 0]
    else:
        cmd_list1 = cmd_list0
    
    cmd_list_2 = list_to_dict_of_chunks(cmd_list1, chunk_size=5)
    result2 = []

    nchnk = len(cmd_list_2)
    for cnt in np.arange(nchnk):
    #for cmd_list in list(cmd_list_2.keys()):
        print('getting 5 nam files...')
        threads = []
        with ThreadPoolExecutor() as executor:
            for cmd in cmd_list_2[cnt+1]:
                fun = nam_grabber_hind #define function
                args = [cmd] #define args to function
                kwargs = {} #
                # start thread by submitting it to the executor
                threads.append(executor.submit(fun, *args, **kwargs))

            for future in as_completed(threads):
                # retrieve the result
                result = future.result()
                result2.append(result)
                # report the result

    # recheck for the files
    fes = [] # a list of 0 (dont have) or 1 (have)
    for fn in l2:
        fe = check_file_exists_os(fn)
        fes.append(fe)
    
    if sum(fes) == len(l2):
        print('we have the ', len(l2), ' grb2 files now.')
        result2 = 0
        return result2
    else:
        sys.exit("need to abort, missing nam.grb2 files.")


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

# ===========================================================================
# WIND ROTATION  -- added 2026-10-06.  READ THIS BEFORE USING hind_functions.
# ===========================================================================
# Until 2026-10-06 this module applied NO WIND ROTATION AT ALL.  grb2_to_pickle
# read UGRD/VGRD straight out of the grib2 and nam_pkls_2_romsatm_pkl
# interpolated them onto the ROMS grid as if they were already ROMS XI/ETA
# components.  They are not, and the resulting wind stress was mis-aimed by
# ~13.5 degrees.  (hind_functions does not import atm_functions, so the
# cosang/sinang rotation in atm_functions.get_atm_data_on_roms_grid -- which is
# correct, and which the operational ECMWF forecast path uses -- never ran here.)
#
# TWO rotations are required, and BOTH are now done:
#
#   STAGE 1  NAM Lambert grid -> earth (east/north).      in grb2_to_pickle
#            NAM grid 218 is Lambert Conformal and publishes u/v resolved along
#            the GRID axes.  This is not a guess: GRIB2 Section 3 flag table 3.3
#            bit 5 is SET in these files (resolutionAndComponentFlags[4] == 1 =
#            "resolved u/v relative to the defined grid").  Undo it with the
#            Lambert convergence angle
#                  gamma = (lon - lon_0) * n,     n = sin(lat_1) for a tangent cone
#            Measured over the NAM subset this module cuts out (lon -124.1 ..
#            -115.4): gamma = -12.29 .. -8.64 deg, mean -10.41.
#
#   STAGE 2  earth -> ROMS XI/ETA.             in nam_pkls_2_romsatm_pkl
#            ROMS has this rotation internally in Nonlinear/set_data.F, but only
#            under #ifdef CURVGRID and only for forcing that is NOT already on
#            the model grid.  We interpolate onto the ROMS grid ourselves, so
#            ROMS skips it and we must finish the job, per level, with that
#            level's own `angle`.  For LV1: angle = 20.62 .. 26.99 deg, mean
#            23.96.
#
# WHY YOU CANNOT DO JUST ONE OF THEM.  The two rotations compose as
# R(angle)R(gamma) = R(angle + gamma), and gamma is NEGATIVE while angle is
# POSITIVE, so they partially cancel:
#
#       both stages (correct)      total rotation  +13.55 deg
#       neither (pre-2026-10-06)   error           ~13.5 deg
#       STAGE 1 ONLY               error           ~24.0 deg   <-- WORSE!
#
# Rotating to earth and stopping there is worse than not rotating at all.  If you
# are tempted to simplify this, do not.
#
# IF YOU HAVE CACHED NAM PICKLES from before 2026-10-06 they hold UNROTATED,
# grid-relative winds.  nam_pkls_2_romsatm_pkl now refuses to use a pickle that
# does not declare its wind frame, rather than silently producing bad forcing --
# delete the old pickles and re-run grb2s_to_pickles.
# ===========================================================================

# GRIB2 flag table 3.3 bit 5 (0-indexed position 4 in the unpacked flag list):
# 0 = u/v resolved relative to easterly/northerly directions
# 1 = u/v resolved relative to the defined grid  -> STAGE 1 is required
_GRIB_FLAG_GRID_RELATIVE = 4

WIND_FRAME_GRID = 'nam_lambert_grid'
WIND_FRAME_EARTH = 'earth_east_north'
WIND_FRAME_ROMS = 'roms_xi_eta'


def lambert_convergence_rad(lon, projparams):
    """Convergence angle (radians) of a Lambert Conformal grid at longitude `lon`.

    Reads lon_0 / lat_1 / lat_2 from the GRIB projection parameters instead of
    hardcoding NAM's 265E / 25N, so a grid change cannot silently go unnoticed.
    """
    if not projparams or projparams.get('proj') != 'lcc':
        raise ValueError(
            "WIND ROTATION: expected a Lambert Conformal ('lcc') grid, got "
            "proj=%r.  The stage-1 rotation below is only valid for lcc; if NAM "
            "has changed projection this code must be revisited, not bypassed."
            % (projparams.get('proj') if projparams else None,))
    lon0 = float(projparams['lon_0'])
    lat1 = float(projparams['lat_1'])
    lat2 = float(projparams.get('lat_2', lat1))
    if abs(lat1 - lat2) < 1e-6:
        n = np.sin(np.deg2rad(lat1))          # tangent cone (NAM 218: 25N)
    else:                                     # true secant cone
        a1, a2 = np.deg2rad(lat1), np.deg2rad(lat2)
        n = (np.log(np.cos(a1) / np.cos(a2)) /
             np.log(np.tan(np.pi/4 + a2/2) / np.tan(np.pi/4 + a1/2)))
    # shortest signed difference, robust to 0..360 vs -180..180 conventions
    dlon = (np.asarray(lon, dtype=float) - lon0 + 180.0) % 360.0 - 180.0
    return np.deg2rad(dlon * n)


def rotate_grid_to_earth(u_g, v_g, gamma):
    """STAGE 1: grid-relative -> earth-relative (east/north)."""
    c, s = np.cos(gamma), np.sin(gamma)
    return u_g * c + v_g * s, -u_g * s + v_g * c


def rotate_earth_to_roms(uE, vN, angle):
    """STAGE 2: earth-relative -> ROMS XI/ETA components.

    `angle` is the ROMS grid variable: angle of the XI axis CCW from true east,
    in radians.  This reproduces exactly what ROMS does in set_data.F:
        cff1 = Uwind*CosAngler + Vwind*SinAngler    -> along XI
        cff2 = Vwind*CosAngler - Uwind*SinAngler    -> along ETA
    """
    c, s = np.cos(angle), np.sin(angle)
    return uE * c + vN * s, vN * c - uE * s


def grb2_to_pickle(fn_in,fn_out,pkl_fnm):
    
    PFM = initfuns.get_model_info(pkl_fnm)
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
    msg_wind = AA1          # keep the message: we need its projection below

    # get north south 10 m winds, m/s
    AA1 = g.select(shortName='VGRD',level='10 m above ground')[0]
    AA2 = AA1.data
    AA3 = AA2[lt_mn:lt_mx,ln_mn:ln_mx]
    ATM['Vwind'] = AA3

    # -----------------------------------------------------------------------
    # STAGE 1 of the wind rotation: NAM Lambert grid -> earth (east/north).
    # See the big WIND ROTATION comment block above this function.  Before
    # 2026-10-06 this did not happen and the winds were written to ROMS in the
    # NAM grid frame, mis-aiming the stress by ~13.5 deg.
    # -----------------------------------------------------------------------
    flags = getattr(msg_wind, 'resolutionAndComponentFlags', None)
    projparams = getattr(msg_wind, 'projParameters', None)
    grid_relative = bool(flags is not None
                         and len(flags) > _GRIB_FLAG_GRID_RELATIVE
                         and flags[_GRIB_FLAG_GRID_RELATIVE] == 1)

    if grid_relative:
        gamma = lambert_convergence_rad(ATM['lon'], projparams)
        u_r, v_r = rotate_grid_to_earth(ATM['Uwind'], ATM['Vwind'], gamma)
        ATM['Uwind'] = u_r
        ATM['Vwind'] = v_r
        gd = np.rad2deg(gamma)
        print('  WIND ROTATION stage 1 (NAM grid -> earth): APPLIED.'
              '  GRIB2 flag 3.3 bit5=1 (u/v grid-relative).')
        print('    convergence angle gamma = %.2f .. %.2f deg (mean %.2f),'
              ' lon_0=%.1f lat_1=%.1f'
              % (gd.min(), gd.max(), gd.mean(),
                 float(projparams['lon_0']), float(projparams['lat_1'])))
        print('    winds in this pickle are now EARTH-relative.'
              '  Stage 2 (earth -> ROMS XI/ETA) happens in'
              ' nam_pkls_2_romsatm_pkl.')
        ATM['wind_frame'] = WIND_FRAME_EARTH
    else:
        # An earth-relative source passes through untouched -- but say so loudly,
        # because silently skipping stage 1 is exactly the old bug.
        print('  WIND ROTATION stage 1: SKIPPED -- GRIB2 flag 3.3 bit5 is NOT'
              ' set, so this file already stores EARTH-relative u/v.')
        print('    flags=%r  proj=%r' % (flags, (projparams or {}).get('proj')))
        ATM['wind_frame'] = WIND_FRAME_EARTH

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
    # NOTE on these two: at THIS point (the per-grb2 pickle) the winds are
    # EARTH-relative -- stage 1 has run, stage 2 has not.  The long_name said
    # 'roms ... coordinate' while the note said 'earth coordinate', which
    # contradicted each other and, before the 2026-10-06 rotation fix, were both
    # wrong anyway: the data was in the NAM Lambert GRID frame.
    # nam_pkls_2_romsatm_pkl overwrites these once stage 2 has rotated to XI/ETA,
    # so each artifact's metadata describes its own contents.
    ATM['vinfo']['Uwind'] = {'long_name':'10 m wind, earth EAST component',
                    'units':'m/s',
                    'coordinates':'lat,lon',
                    'time':'wind_time',
                    'note':'earth-relative (east). stage 1 of 2 applied: NAM Lambert grid -> earth via the convergence angle. NOT yet rotated into ROMS xi/eta'}
    ATM['vinfo']['Vwind'] = {'long_name':'10 m wind, earth NORTH component',
                    'units':'m/s',
                    'coordinates':'lat,lon',
                    'time':'wind_time',
                    'note':'earth-relative (north). stage 1 of 2 applied: NAM Lambert grid -> earth via the convergence angle. NOT yet rotated into ROMS xi/eta'}
    
    # if the pickle file exists, to avoid permission issue, remove it...
    if os.path.exists(fn_out):
        os.remove(fn_out)

    # write the pickle file
    with open(fn_out,'wb') as fp:
        pickle.dump(ATM,fp)
        print('ATM grb2 file ' + fn_in + ' saved to pickle.')


def grb2s_to_pickles(t1str,t2str,pkl_fnm):
    # this function takes all of the grb2 nam files and makes pickles out of them
    
    # get file names
    _, l2, _, l4 = get_nam_hindcast_filelists(t1str,t2str,pkl_fnm)

    for j in np.arange(len(l2)):
        fn_in = l2[j]
        fn_out = l4[j]
        grb2_to_pickle(fn_in,fn_out,pkl_fnm)


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


def nam_pkls_2_romsatm_pkl(t1str,t2str,lv,pkl_fnm):
    # this function takes nam_hindcast pickle files, interpolates the fields
    # from the nam grid to the rom grid at level = lv

    PFM = initfuns.get_model_info(pkl_fnm)

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

    _, _, _, fn_pkls = get_nam_hindcast_filelists(t1str,t2str,pkl_fnm)

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

    # -----------------------------------------------------------------------
    # Refuse a NAM pickle written before the 2026-10-06 wind-rotation fix.
    # Those hold UNROTATED, grid-relative winds, and using them would silently
    # reproduce the ~13.5 deg stress error.  There is no way to tell from the
    # numbers alone, which is why the frame is recorded explicitly.
    # -----------------------------------------------------------------------
    wf = ATM.get('wind_frame')
    if wf != WIND_FRAME_EARTH:
        raise RuntimeError(
            "STALE NAM PICKLE: %s declares wind_frame=%r, expected %r.\n"
            "    Pickles written before 2026-10-06 contain winds still in the"
            " NAM Lambert GRID frame (no rotation was applied back then).\n"
            "    Using them mis-aims the wind stress by ~13.5 deg.\n"
            "    Fix: delete the cached NAM pickles and re-run"
            " grb2s_to_pickles to rebuild them."
            % (fn_pkls[0], wf, WIND_FRAME_EARTH))

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
                 
                lt2 = RMG['lat_rho']
                ln2 = RMG['lon_rho']
                # we need the interpolator object
                xi_2, eta_2, interper = ocnfuns.get_child_xi_eta_interp(lon_nam,lat_nam,ln2,lt2,'rho')
                set_up = 1

            setattr(interper,'values',z0) # change the interpolator z values
            z2 =  interper((eta_2,xi_2)) # perhaps change here to directly interpolate to (xi,eta) on the edges?
            atm2[var][cnt_time,:,:] = z2

        cnt_time = cnt_time + 1

    # -----------------------------------------------------------------------
    # STAGE 2 of the wind rotation: earth (east/north) -> ROMS XI/ETA.
    #
    # This MUST happen after the horizontal interpolation, because the ROMS
    # `angle` varies across the grid and is defined on the ROMS grid, not NAM's.
    #
    # ROMS will NOT do this for us.  Nonlinear/set_data.F has the rotation, but
    # only under #ifdef CURVGRID and only for forcing that is not already on the
    # model grid -- and we have just interpolated onto the model grid.  See the
    # WIND ROTATION comment block near grb2_to_pickle, including why doing stage
    # 1 without stage 2 is WORSE than doing neither.
    # -----------------------------------------------------------------------
    angr = RMG['angle']                       # radians, XI axis CCW from east
    u_r, v_r = rotate_earth_to_roms(atm2['Uwind'], atm2['Vwind'], angr)
    atm2['Uwind'] = u_r
    atm2['Vwind'] = v_r
    atm2['wind_frame'] = WIND_FRAME_ROMS
    # the vinfo inherited from the per-grb2 pickle describes EARTH winds; these
    # are now ROMS-frame, and this vinfo is what ends up in the atm .nc.
    atm2['vinfo']['Uwind'] = {'long_name':'10 m wind, ROMS XI component',
                    'units':'m/s',
                    'coordinates':'lat,lon',
                    'time':'wind_time',
                    'note':'rotated from earth (east,north) into the roms grid directions using the grid angle. roms does not rotate these again: the field is on the model grid, so set_data.F sees it as already gridded and skips its CURVGRID rotation'}
    atm2['vinfo']['Vwind'] = {'long_name':'10 m wind, ROMS ETA component',
                    'units':'m/s',
                    'coordinates':'lat,lon',
                    'time':'wind_time',
                    'note':'rotated from earth (east,north) into the roms grid directions using the grid angle. roms does not rotate these again: the field is on the model grid, so set_data.F sees it as already gridded and skips its CURVGRID rotation'}
    ad = np.rad2deg(np.asarray(angr))
    print('  WIND ROTATION stage 2 (earth -> ROMS XI/ETA) for LV%s: APPLIED.' % lv)
    print('    ROMS grid angle = %.2f .. %.2f deg (mean %.2f)'
          % (ad.min(), ad.max(), ad.mean()))
    print('    winds in %s are now in the ROMS XI/ETA frame,'
          ' which is what ROMS expects.' % fname_out)
    print('    (stage 1 rotated NAM grid -> earth when the pickles were built;'
          ' both stages are required -- see hind_functions WIND ROTATION notes)')

    # copy times to all time vars
    tlist = ['tair_time','pair_time','qair_time','wind_time','rain_time','srf_time','lrf_time']
    for tnm in tlist:
        atm2[tnm] = atm2['ocean_time']

    with open(fname_out,'wb') as fp:
        pickle.dump(atm2,fp)
        print('\nATM on roms grid dict saved with pickle.')
    
    return fes_test


if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])


                






        
    


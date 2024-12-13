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
import shutil

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

def mk_swan_grd_file(lns,lts,fout):
    # this function makes a swan .grd file from arrays of lons and lats
    # if using roms these should be lon_rho and lat_rho.
    ny,nx = np.shape(lns)
    with open(fout,'w') as f:
        for aa in np.arange(ny):
            for bb in np.arange(nx):
                f.write(f'{lns[aa,bb] : .6f}')
                f.write('\n')
        for aa in np.arange(ny):
            for bb in np.arange(nx):
                f.write(f'{lts[aa,bb] : .6f}')
                f.write('\n')

def mk_swan_bot_file(hb,fout):
    # this function makes a swan .grd file from arrays of lons and lats
    # if using roms these should be lon_rho and lat_rho.
    ny,nx = np.shape(hb)
    with open(fout,'w') as f:
        for aa in np.arange(ny):
            for bb in np.arange(nx):
                f.write(f'{hb[aa,bb] : .6f}')
                f.write('\t')
            f.write('\n')

def mk_swan_wnd_file(fout):
    PFM = get_PFM_info()
    fin = PFM['lv4_forc_dir'] + '/' + PFM['lv4_atm_file']
    ds  = nc.Dataset(fin)
    U   = ds['Uwind'][:]
    V   = ds['Vwind'][:]
    nt,nlt,nln = np.shape(U)
    with open(fout,'w') as f:
        for aa in np.arange(nt):
            for bb in np.arange(nlt):
                for cc in np.arange(nln):
                    f.write(f'{U[aa,bb,cc] : >7.2f}')
                f.write('\n')

            for bb in np.arange(nlt):
                for cc in np.arange(nln):
                    f.write(f'{V[aa,bb,cc] : >7.2f}')
                f.write('\n')


def mk_swan_bnd_file(fout):

    PFM = get_PFM_info()
    fn_in = PFM['lv4_forc_dir'] + '/' + PFM['lv4_swan_pckl_file']
    with open(fn_in,'rb') as fp:
        cdip = pickle.load(fp)
        print('\nCDIP pickle file loaded')

    nloc,ntime,nfreq,ndeg = np.shape(cdip['Spp'])

    with open(fout,'w') as f:
        f.write(f'SWAN   1                                Swan standard spectral file, version\n')
        f.write(f'$ Data produced by SWAN version 41.20\n')
        f.write(f'$ Project: LV4 Grid mss_oct\n')
        f.write(f'TIME                                    time-dependent data\n')
        f.write(f'     1                                  time coding option\n')
        f.write(f'LONLAT                                  locations in spherical coordinates\n')
        f.write(f'    {str(len(cdip['lats']))}            number of locations\n')
        i = 0
        for lat in cdip['lats']:
            lon = cdip['lons'][i]
            f.write(f'{lon : .6f}{'   '}{lat : .6f}{'\n'}')
            i = i+1
        
        f.write(f'AFREQ                                   absolute frequencies in Hz\n')
        f.write(f'{'    '}{str(nfreq)}{'                                  number of frequencies\n'}')
        for fr in cdip['f']:
            f.write(f'{'   '}{fr : .4f}{'\n'}')

        f.write(f'NDIR                                    spectral nautical directions in degr\n')
        f.write(f'{'    '}{str(ndeg)}{'                                  number of directions\n'}')
        for dd in cdip['dir']:
            f.write(f'{'   '}{dd : .4f}{'\n'}')

        f.write(f'QUANT\n')
        f.write(f'     1                                  number of quantities in table\n')
        f.write(f'VaDens                                  variance densities in m2/Hz/degr\n')
        f.write(f'm2/Hz/degr                              unit\n')
        f.write(f'    -9.9000e+01                         exception value\n')
        it = 0
        for tt in cdip['time']:
            tstr = tt.strftime("%Y%m%d.%H%M%S")
            f.write(f'{tstr}{'                         '}{'date and time\n'}')

            for iloc in np.arange(nloc):
                S2 = np.squeeze(cdip['Spp'][iloc,it,:,:])
                nn = np.max(S2) / 9900
                S3 = np.round(S2/nn)
                f.write(f'FACTOR\n')
                f.write(f'{nn : .7E}{'\n'}')
                for ifr in np.arange(nfreq):
                    for ideg in np.arange(ndeg):
                        f.write(f'{int(S3[ifr,ideg]) : >6}')
                    f.write(f'\n')
            it = it + 1

        #20161201.000000                         date and time
def cdip_wave_grabber(fin,vars,fout):
    # this is the function that is parallelized
    # the input is url: the hycom url to get data from
    # dtff: the time stamp of the forecast, ie, the file we want
    # aa: a box that defines the region of interest
    # PFM: used to set the path of where the .nc files go
    # dstr_ft: the date string of the forecast model run. ie. the first
    #          time stamp of the forecast
        
    cmd_list = ['ncks','-q','-D','0','-v', vars, fin ,'-4', '-O', fout]
 
    # run ncks
#    ret1 = subprocess.call(cmd_list,stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
    ret1 = subprocess.call(cmd_list)
    return ret1


def get_cdip_data():
    # this function gets all of the new hycom data as separate files for each field (ssh,temp,salt,u,v) and each time
    # and puts each .nc file in the directory for hycom data

    PFM=get_PFM_info()

    yyyymmddhh = PFM['fetch_time'].strftime("%Y%m%d%H")

    url_0 = 'https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/misc/falk/forecast/F' 
    fn1s = (['2436','2428','2415','2385','2360','2335','2322','2310','2298','2285','2260','2235',	
             '2185','2135','2085','2035','1985','1885','1835','1785','1735','1685','1635','1585',
             '1545','1530','1515','1495','1475','1460','1445','1434','1422','1410','1404','1396',	
             '1388','1380','1372','1364','1356','1348','1340','1334','1330','1326','1322','1318',	
             '1314','1310','1306','1302','1298','1294','1290','1286','1285','1283','1281','1276',	
             '1271','1266','1261','1255','1245','1235','1230','1220','1210','1200','1190','1181',	
             '1161','1141','1121','1101','1081','1061','1041','1021','1001'])
    fn2 = '_ecmwf_fc.nc'
    varns = 'waveTime,waveHs,waveTp,waveDp,waveFrequency,waveDirection,waveDirectionalSpectrum,metaLatitude,metaLongitude'


    with ThreadPoolExecutor() as executor:
        threads = []
        for ff in fn1s:
            fn = cdip_wave_grabber
            fnmin = url_0+ff+fn2
            fnmout = PFM['cdip_data_dir'] + '/' + 'cdip_' + ff + '_' + yyyymmddhh + '.nc' 
#            fnmout = ('/scratch/PFM_Simulations/LV4_Forecast/Forc/cdip_data' + '/' 
#                    + 'cdip_' + ff + '_' + yyyymmddhh + '.nc')          
            args = [fnmin,varns,fnmout]
            kwargs = {} #
            # start thread by submitting it to the executor
            threads.append(executor.submit(fn, *args, **kwargs))
            
        for future in as_completed(threads):
                # retrieve the result
            result = future.result()
            #print(result)
                # report the result        


def cdip_ncs_to_dict(refresh):
    
    PFM=get_PFM_info()
    cdip_dir = PFM['cdip_data_dir']
    if refresh == 'refresh':
        print('removing previous cdip .nc files...')
        for f in glob.glob(cdip_dir + '/*.nc'):
            os.remove(f)        
        print('...done. getting new cdip .nc files...')
        get_cdip_data()
    else:
        print('using the existing cdip .nc data.')

    fns = glob.glob( PFM['cdip_data_dir'] + '/*.nc')
    nlocs = len(fns)
    #print('there are ', nlocs, ' cdip locations on the LV4 boundary.')
    

    print('...done. making dictionary of cdip data...')
    dd = nc.Dataset(fns[0])
    # these don't depend on the location
    f     = np.array( dd['waveFrequency'][:].data )
    fmn   = 0.04
    fmx   = 0.26
    msk   = (f >= fmn) & (f <= fmx)
    ii    = np.arange(len(f))
    igf   = ii[msk]
    f2    = f[igf]
    nf    = len(f2)
    dir   = np.array( dd['waveDirection'][:].data )
    dmx   = 360.0
    dmn   = 155.0
    msk   = (dir >= dmn) & (dir <= dmx)
    ii    = np.arange(len(dir))
    igd   = ii[msk]
    dir2  = dir[igd]
    ndir  = len(dir2)
    t     = dd.variables['waveTime']

    t2 = nc.num2date(t[:], t.units)
    t2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in t2])


    t00 = PFM['fetch_time'] # the first forecast time
    t10 = t00 + PFM['forecast_days'] * timedelta(days=1)    # need to find the right t's for the forecast...
    # t10 is the last forecast time
    msk = (t2 >= t00) & (t2 <= t10)
    ii = np.arange(len(t2))
    igt = ii[msk] # these are now the indices of the cdip forecast that match the timeframe of the PFM forecast
    nt = len(igt)

    cdip = dict()
    cdip['f'] = f2
    cdip['dir'] = dir2
    cdip['time'] = t2[igt]
    cdip['lats']=np.zeros(nlocs)
    cdip['lons']=np.zeros(nlocs)
    
    lons = np.zeros(nlocs)
    lats = np.zeros(nlocs)
    cdip['Spp'] = np.zeros([nlocs,nt,nf,ndir])

    i=0
    for fn in fns:
        dd = xr.open_dataset(fn)
        lat   = dd['metaLatitude'].values
        lon   = dd['metaLongitude'].values
        lats[i] = lat
        lons[i] = lon
        i = i + 1
 
    i_sort = np.argsort(lons)

    i_sort2 = i_sort.copy()
    # this rearranging gets the locations to go CCW around the boundary
    i_sort2[[0,2,3]] = i_sort[[3,0,2]]   

    # flipping gets the locations to go CW around the boundary
    i_sort2 = np.flip(i_sort2)
    # i_sort is now the proper ording of the locations.

    fns_sorted = []
    for ii in i_sort2:
        fns_sorted.append(fns[ii])

    i=0
    for fn in fns_sorted:
        dd = xr.open_dataset(fn)
        cdip['lats'][i] = dd['metaLatitude'].values
        cdip['lons'][i] = dd['metaLongitude'].values
        cdip['Spp'][i,:,:,:] = dd['waveDirectionalSpectrum'][igt,igf,igd]
        i = i + 1

    # two grid points are not in the right place... fudge them just a tad.
    gr4 = nc.Dataset(PFM['lv4_grid_file'])
    cdip['lons'][-2] = gr4['lon_rho'][-1,0]
    cdip['lats'][-2] = gr4['lat_rho'][-1,0]
    cdip['lons'][-1] = gr4['lon_rho'][-1,10]
    cdip['lats'][-1] = gr4['lat_rho'][-1,10]


    print('...done.')

    fn_out = PFM['lv4_forc_dir'] + '/' + PFM['lv4_swan_pckl_file']
    with open(fn_out,'wb') as fp:
        pickle.dump(cdip,fp)
        print('\nCDIP data saved as pickle file')

def check_and_move(fname,dt_sec,nfiles):
    PFM = get_PFM_info()
    rst_dir = PFM['restart_files_dir'] + '/'
    dt_sec = int(dt_sec)
    nfiles = int(nfiles)

    #fname_full = rst_dir + fname
    #dtf  = PFM['lv4_swan_rst_int_hr']
    # fname = /home/mspydell/models/PFM_root/swan_rst_testing/swan_test_rs.txt
    fnames = []
    thr = []
    last_modified_time = []
    for ii in np.arange(nfiles):
        txt = fname[0:-7] + '_cpu' + str(ii) + fname[-7:]
        fnames.append(txt)
        last_modified_time.append(0)
        thr.append(0) # set the initial hour time to 0 for each file
    
    dtf = 6

    #thr_max = PFM['fore_days'] * 24 
    thr_max = 36

    """Watches the source file and copies it to the destination when it's written."""
    

    while True: # an endless while loop, be careful!
    #while thr[-1] <=  thr_max:   
        cnt = 0
        t0 = datetime.now()
        for fname_full in fnames:
            try:
                current_modified_time = os.path.getmtime(fname_full)
                if current_modified_time > last_modified_time[cnt] + 2.0: # and thr[cnt]<=thr_max:
                    hrr = str(int(thr[cnt])).zfill(3)
                    print(hrr)
                    fnew = fname_full[0:-8] + '_' + hrr + fname[-8:]
                    print(f"File '{fname_full}' modified. Copying to '{fnew}'...")
                    shutil.copy2(fname_full, fnew)  # Use shutil.copy2 to preserve metadata
                    last_modified_time[cnt] = current_modified_time
                    thr[cnt] = thr[cnt] + dtf
                    cnt = cnt+1
            except FileNotFoundError:
                print(f"File '{fname_full}' not found. making it...")
                with open(fname_full, 'w') as fid:
                    a_str = 'original file'
                    fid.write(a_str)
                    fid.close()

        print(datetime.now()-t0)
        t0 = datetime.now()
        time.sleep(dt_sec)  # Check every dt_sec second


def manage_swan_restart_files():
    PFM = get_PFM_info()
    fn0 = PFM['lv4_swan_rst_name']
    ncpu = PFM['gridinfo']['L4','np_swan']
    fns = [] # this will be a list of the swan restart file names
    for ii in np.arange(ncpu):
        ist = str(ii+1).zfill(3)
        fnew = fn0 + '-.' + ist
        fns.append(fnew)    

    # swan write the rst files at the beginning of the simulation... Do they get written over? 
    #for fn in fns:
    #    fout = rst_dir + fn
    #    with open(fout, "w") as f:
    #        pass  # Do nothing, just create the file

    for fn in fns:
        check_and_move(fn)


if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])    


        
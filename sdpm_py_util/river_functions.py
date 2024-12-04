# library of river functions
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import requests
import netCDF4 as nc
from netCDF4 import num2date
from get_PFM_info import get_PFM_info
import grid_functions as grdfuns

#from pydap.client import open_url

def get_river_flow_nwm(yyyymmddhh):
    PFM = get_PFM_info()
    # this function gets the river discharge for Sweetwater, Otay, and TJR from 
    # the National Water Model. We use the reaches closest to the ocean.

    reach_ids = [948070199, 20331702, 20324441]
    reach_ids = np.array(reach_ids)
    #            SW         Otay      TJR

    yyyymmdd = yyyymmddhh[0:8]
    print(yyyymmdd)
    hh = yyyymmddhh[8:]
    print(hh)
    fore_type = 'medium_range_blend' # short_range, long_range, etc.
    nhr = 24 * PFM['forecast_days']
    hrs = np.arange(1,nhr+1,1) # data is at 1 hr intervals, we will loop through this to get the data... hr=0 doesn't exist...
    url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/v3.0/nwm.' + yyyymmdd + '/' + fore_type
    fname = ['nwm.t','z.'+fore_type+'.channel_rt.f','.conus.nc']
    #nwm.t00z.medium_range_blend.channel_rt.f001.conus.nc

    tmpnc = PFM['lv4_forc_dir'] + '/river_tmp.nc'
    t3 = [None] * (len(hrs)+1)
    Q = np.zeros((len(hrs)+1,3))
    cnt1 = 1
    for hr in hrs:
    
        hr_str = str(int(hr)).zfill(3)
        print(hr_str)
        fn = fname[0] + hh + fname[1] + hr_str + fname[2]
        url_tot = url + '/' + fn
        
    #    print(url_tot)
    #    ds = xr.open_dataset(url_tot)
        response = requests.get(url_tot)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to a temporary file
            with open(tmpnc, "wb") as f:
                f.write(response.content)

        # Open the NetCDF file using netCDF4
            with nc.Dataset(tmpnc) as ds:
                # Access the data variables
                rids = ds.variables['feature_id'][:]
                t = ds.variables['time']
                qq = ds.variables['streamflow'][:]
                t2 = num2date(t[:],t.units)
                t2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in t2])
                t3[cnt1] = t2
    
        #rids = ds['feature_id'][:]

        # ds = nc.Dataset(url_tot) DOESNT WORK. NOT the right server type on their end?
        
        ig = [None]*3
        cnt=0
        for rids0 in reach_ids:
            ig= np.argwhere(rids==rids0)
            Q[cnt1,cnt] = qq[ig]
            cnt=cnt+1

        cnt1 = cnt1+1


    t3[0] = t3[1] - timedelta(hours=1)
    Q[0,:] = Q[1,:]
    print(t3)
    print(Q[0:10,:])





    

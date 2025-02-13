# library of river functions
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

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

def get_river_flow_nwm(yyyymmddhh,t_pfm_str):
    # yyyymmddhh is the start time of the river forecast
    # t_pfm_str [in yyyymmddhh] is the start time of the PFM forecast
    # this is coded to work only if t_pfm is larger than yyyymmddhh, the river forecast start time
    # we will typically use t_fore = yyyymmddhh + 6 hr. Using the previous river forecast ensures 
    # that the river forecast is posted to their server
    PFM = get_PFM_info()
    file_out = PFM['river_pckl_file_full']
    #file_out = '/scratch/PFM_Simulations/LV4_Forecast/Forc/river_Q.pkl'
    # this function gets the river discharge for Sweetwater, Otay, and TJR from 
    # the National Water Model. We use the reaches closest to the ocean.

    reach_ids = [948070199, 20331702, 20324441]
    reach_ids = np.array(reach_ids)
    #            SW         Otay      TJR 20324441 is last segment near ocean.

    t_nwm = datetime.strptime(yyyymmddhh,'%Y%m%d%H')
    t_pfm = datetime.strptime(t_pfm_str,'%Y%m%d%H')
    delta_t = t_pfm - t_nwm # this should be 6 in hours
    delta_t_hr = delta_t.total_seconds() / 3600 # this should be an integer 6

    yyyymmdd = yyyymmddhh[0:8]
    #print(yyyymmdd)
    hh = yyyymmddhh[8:]
    #print(hh)
    fore_type = 'medium_range_blend' # short_range, long_range, etc.
    nhr = 24 * PFM['forecast_days']
    hrs = delta_t_hr + np.arange(0,nhr+3,1) # data is at 1 hr intervals, we will loop through this to get the data...
                                            # the +3 here is to get 2 extra hours of data. this is needed to get riv.nc 
                                            # to work correctly.
    url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/v3.0/nwm.' + yyyymmdd + '/' + fore_type
    fname = ['nwm.t','z.'+fore_type+'.channel_rt.f','.conus.nc']
    #nwm.t00z.medium_range_blend.channel_rt.f001.conus.nc

    tmpnc = PFM['lv4_forc_dir'] + '/river_tmp.nc'
    t3 = [None] * (len(hrs))
    Q = np.zeros((len(hrs),3))
    cnt1 = 0
    for hr in hrs:
    
        hr_str = str(int(hr)).zfill(3)
    #    print(hr_str)
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
        
    plot_it = 1
    if plot_it == 1:
        fig, ax = plt.subplots()
        p1=ax.plot(t3,Q[:,0],label='Sweet Water')
        p2=ax.plot(t3,Q[:,1],label='Otay Mesa')
        p3=ax.plot(t3,Q[:,2],label='TJ')

        plt.legend()
        plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
        plt.ylabel('discharge [m3/s]')
        plt.title('PFM forecast time is: ' + t_pfm_str + '| river forecast time is: ' + yyyymmddhh )
        fn_out = PFM['lv4_plot_dir'] + '/river_discharge_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        plt.savefig(fn_out, dpi=300)

    QQ = dict()
    QQ['time'] = t3
    # previous XWu LV4 simulations capped TJR Q at 150 m3/s. We might want to do that here?
    QQ['discharge'] = Q
    QQ['reach_ids'] = reach_ids
    QQ['readme'] = 'discharge is in m3/s. reach_ids correspond to Sweetwater, Otay, TJR. they are the columns of discharge'

    with open(file_out,'wb') as fp:
        pickle.dump(QQ,fp)
        print('\nriver discharge data saved as pickle file')


def get_river_temp():
    PFM = get_PFM_info()
    fatm = PFM['lv4_forc_dir'] + '/' + PFM['lv4_atm_file'] 
    RMG = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
    #print(RMG.keys())

    ds = nc.Dataset(fatm)
    temp_air = ds['Tair'][:]
    msk2d = RMG['mask_rho']
    msk3d = np.broadcast_to( msk2d==0 , temp_air.shape)
    temp_river = np.mean(temp_air[msk3d])
    nt,ny,nx = np.shape(temp_air)

    t_air = np.arange(0,3*nt, 3)
    temp_river_time0 = np.zeros(nt)
    for a in np.arange(nt):
        tmp = temp_air[a,:,:]
        temp_river_time0[a] = np.mean( tmp[msk2d==0] )

    t_riv = np.arange(0,3*nt,1)    # this should be the length of triver in river.nc file...
    Fz = interp1d(t_air,temp_river_time0,bounds_error=False,kind='linear',fill_value=(temp_river_time0[0],temp_river_time0[-1]))
                
    temp_river_time = Fz(t_riv)
    #print(len(temp_river_time))

    plot_it = 1
    if plot_it == 1:
        fig, ax = plt.subplots()
        p1=ax.plot(t_riv,temp_river_time)
        plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
        plt.ylabel('river_temperature [C]')
        plt.title('all 3 rivers have this temperature for forecast: ' + PFM['yyyymmdd'] + PFM['hhmm'] )
        fn_out = PFM['lv4_plot_dir'] + '/river_temperature_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        plt.savefig(fn_out, dpi=300)


    return temp_river, temp_river_time

    

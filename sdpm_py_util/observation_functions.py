import numpy as np

from datetime import datetime
from datetime import timedelta
import netCDF4 as nc

#from pydap.client import open_url



def get_cdip_buoy_data(start_date_str, end_date_str, location_code, variables):
    rt_url = f'https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{location_code}p1_rt.nc'
    his_url = f'https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{location_code}p1/{location_code}p1_historic.nc'
    #realtime_dataset = open_url(rt_url)
    #historic_dataset = open_url(his_url)

    realtime_dataset = nc.Dataset(rt_url)
    historic_dataset = nc.Dataset(his_url)

    #dataset = nc.Dataset(ocn_name)

    cdip_buoy = {}
    realtime_gps_times = np.array(realtime_dataset['gpsTime'])
    historic_gps_times = np.array(historic_dataset['gpsTime'])
    realtime_sst_times = np.array(realtime_dataset['sstTime'])
    historic_sst_times = np.array(historic_dataset['sstTime'])
    
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    
    realtime_gps_datetimes = np.array([datetime.strptime(datetime.fromtimestamp(gps_time).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') for gps_time in realtime_gps_times])
    historic_gps_datetimes = np.array([datetime.strptime(datetime.fromtimestamp(gps_time).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') for gps_time in historic_gps_times])
    realtime_sst_datetimes = np.array([datetime.strptime(datetime.fromtimestamp(gps_time).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') for gps_time in realtime_sst_times])
    historic_sst_datetimes = np.array([datetime.strptime(datetime.fromtimestamp(gps_time).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') for gps_time in historic_sst_times])

    times = {
        'gps_utc_times': np.concatenate([historic_gps_datetimes, realtime_gps_datetimes]),
        'sst_utc_times': np.concatenate([historic_sst_datetimes, realtime_sst_datetimes])
    }
        
    masks = {
        'gps_mask': (times['gps_utc_times'] >= start_date) & (times['gps_utc_times'] <= end_date),
        'sst_mask': (times['sst_utc_times'] >= start_date) & (times['sst_utc_times'] <= end_date)
    }
    
    for variable in variables:
        if 'gps' in variable:
            utc_times = times['gps_utc_times']
            mask = masks['gps_mask']
        elif 'sst' in variable:
            utc_times = times['sst_utc_times']
            mask = masks['sst_mask']
            #print(variable)
        
        #print(len(utc_times),)
        if start_date < utc_times[0]:
            print(f'start_date earlier than earliest data, getting data starting at {utc_times[0]}')
        if end_date > utc_times[-1]:
            print(f'end_date in the future, getting data up to {utc_times[-1]}')
        
        cdip_buoy[variable] = np.concatenate([np.array(historic_dataset[variable]), np.array(realtime_dataset[variable])])[mask]
    
    cdip_buoy['sstTime'] = times['sst_utc_times'][masks['sst_mask']]
    #cdip_buoy['gpsTime'] = times['gps_utc_times'][masks['gps_mask']]
    cdip_buoy['gpsLatitude'] = np.nanmean( cdip_buoy['gpsLatitude'] )
    cdip_buoy['gpsLongitude'] = np.nanmean( cdip_buoy['gpsLongitude'] )

    return cdip_buoy
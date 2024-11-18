import numpy as np

from datetime import datetime
from datetime import timedelta
import netCDF4 as nc
import subprocess

#from pydap.client import open_url


def get_cdip_buoy_data(start_date_str, end_date_str, location_codes, variables, output):
    location_to_cdip = {}
    for location_code in location_codes:
        realtime_url = f'https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/{location_code}p1_rt.nc'
        historic_url = f'https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{location_code}p1/{location_code}p1_historic.nc'
        realtime_out = f'{output}_realtime.nc'
        historic_out = f'{output}_historic.nc'
        
        realtime_cmd_list = ['ncks']
        historic_cmd_list = ['ncks']
        
        cdip_buoy = {}
        
        start_date = datetime.strftime(datetime.strptime(start_date_str, '%Y%m%d'), '%Y-%m-%dT%H:%M')
        end_date = datetime.strftime(datetime.strptime(end_date_str, '%Y%m%d'), '%Y-%m-%dT%H:%M')
        
        for var in variables:
            if 'time' in var.lower():
                realtime_cmd_list.extend(['-d', var + ',' + start_date + ',' + end_date])
                historic_cmd_list.extend(['-d', var + ',' + start_date + ',' + end_date])
                
        realtime_cmd_list.extend([realtime_url, '-O', realtime_out])
        historic_cmd_list.extend([historic_url, '-O', historic_out])
        
        realtime_ret = subprocess.call(realtime_cmd_list)
        historic_ret = subprocess.call(historic_cmd_list)

        # print message for historical and realtime status
        
        realtime_dataset = nc.Dataset(realtime_out) if realtime_ret == 0 else None
        historic_dataset = nc.Dataset(historic_out) if historic_ret == 0 else None
        for var in variables:
            cdip_buoy[var] = np.array([])
            
            if realtime_dataset:
                cdip_buoy[var] = np.concatenate([cdip_buoy[var], np.array(realtime_dataset.variables[var][:])])
            if historic_dataset:
                cdip_buoy[var] = np.concatenate([cdip_buoy[var], np.array(historic_dataset.variables[var][:])])
        
        if realtime_dataset:
            cdip_buoy['metaStationName'] = realtime_dataset.variables['metaStationName'][:].data.tobytes().decode('utf-8').strip('\x00')
        else:
            cdip_buoy['metaStationName'] = historic_dataset.variables['metaStationName'][:].data.tobytes().decode('utf-8').strip('\x00')

        # average lat long
        cdip_buoy['gpsLatitude'] = np.mean(cdip_buoy['gpsLatitude'])
        cdip_buoy['gpsLongitude'] = np.mean(cdip_buoy['gpsLongitude'])
        # convert to datetimes
        for var in variables:
            if 'time' in var.lower():
                cdip_buoy[var] = np.array([datetime.strptime(datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') for time in cdip_buoy[var]])

        location_to_cdip[location_code] = cdip_buoy
    return location_to_cdip


def get_cdip_buoy_data_old(start_date_str, end_date_str, location_code, variables):
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
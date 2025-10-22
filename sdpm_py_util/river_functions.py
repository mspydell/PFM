# library of river functions
import sys
import csv
import os
import urllib.request
from urllib.error import URLError, HTTPError

from datetime import datetime, timedelta, timezone
import cftime
from scipy.interpolate import interp1d

import pickle
import matplotlib.pyplot as plt
import numpy as np
import requests
import netCDF4 as nc
from netCDF4 import num2date
sys.path.append('../sdpm_py_util')
import grid_functions as grdfuns
import init_funs_forecast as initfuns
import subprocess

def floor_datetime_to_nearest_6_hours(dt):
    """
    Floors a datetime object to the nearest 6-hour interval.
    The 6-hour intervals are 00:00, 06:00, 12:00, 18:00.
    """
    # Define the duration of 6 hours
    six_hours = timedelta(hours=6)

    # Calculate the total seconds from a reference point (e.g., epoch start)
    # Using timestamp() is convenient for this.
    total_seconds_since_epoch = dt.timestamp()

    # Calculate the number of 6-hour intervals since the epoch start, floored
    num_intervals = int(total_seconds_since_epoch // six_hours.total_seconds())

    # Reconstruct the datetime object from the floored number of intervals
    floored_dt = datetime.fromtimestamp(num_intervals * six_hours.total_seconds())

    return floored_dt


def make_new_nwm_files():
    
    # get the current hour in UTC
    t_now = datetime.now(timezone.utc)
    t_now_hr = t_now.hour
    if t_now_hr >=0 and t_now_hr < 6:
        t_now_floored = datetime(t_now.year,t_now.month,t_now.day,0,0,0)
    elif t_now_hr >=6 and t_now_hr < 12:
        t_now_floored = datetime(t_now.year,t_now.month,t_now.day,6,0,0)
    elif t_now_hr >=12 and t_now_hr < 18:
        t_now_floored = datetime(t_now.year,t_now.month,t_now.day,12,0,0)
    elif t_now_hr >=18 and t_now_hr < 24:
        t_now_floored = datetime(t_now.year,t_now.month,t_now.day,18,0,0)

    t_fore_last = t_now_floored - 6 * timedelta(hours=1)
    
    # get the last full forecast time...
    t0 = get_youngest_full_nwm_forecast()
    t0_dt = datetime.strptime(t0,'%Y%m%d%H') + 6 * timedelta(hours=1)
    # t_to_get is the list of times of nwm forecasts that we don't have.
    t_to_get = []
    t=t0_dt
    while t<=t_fore_last:
        t_to_get.append(t)
        t = t + 6 * timedelta(hours=1)
        
    if len(t_to_get)==0: # there are no files to get when there is only 1 full file.
        yyyymmddhh = t0
        file_out = '/scratch/PFM_Simulations/nwm_ncs/nwm_forecast_Q_'+yyyymmddhh+'.nc'
        print('the most recent nwm file is: ', file_out)
        print('do we have it...')
        have_it = os.path.exists(file_out)
        if have_it:
            print('we already have the latest file. nothing to do. exiting.')
            return


    # we reverse the order so we try and get the most recent first
    t_to_get.reverse()


    for t in t_to_get:
        yyyymmddhh = t.strftime('%Y%m%d%H')
        file_out = '/scratch/PFM_Simulations/nwm_ncs/nwm_forecast_Q_'+yyyymmddhh+'.nc'
        print('going to make: ', file_out)
        make_nwm_nc_file(yyyymmddhh,file_out)
        print('checking for missing times...')
        T = return_missing_hours_from_nwm_ncs([file_out])
        if len(T[file_out]) == 0:
            print('all times were there, done.')
            break
        else:
            print('missing times. missing ', str(len( T[file_out]) ) , ' hours')
            print('getting another forecast...')


def get_nwm_fores_we_have():

    directory_path = '/scratch/PFM_Simulations/nwm_ncs'
    all_files = os.listdir(directory_path)
    filtered_files = [s for s in all_files if "forecast" in s]
    extracted_substrings = []
    for s in filtered_files:
        # Ensure n1 and n2 are within the bounds of the current string
        start_index = 15
        end_index = 25
        extracted_substrings.append(s[start_index:end_index])
    
    # Convert to a set to get unique values, then convert back to a list
    yyyymmddhhs = list(set(extracted_substrings))
    files = []
    file0 = 'nwm_forecast_Q_'
    for date in yyyymmddhhs:
        files.append( file0 + date + '.nc')

    return yyyymmddhhs, files




def get_youngest_full_nwm_forecast():
    dates, file_names = get_nwm_fores_we_have()
    dir0 = '/scratch/PFM_Simulations/nwm_ncs/'
    fns_tot = []
    for fn in file_names:
        fns_tot.append(dir0+fn)

    T = return_missing_hours_from_nwm_ncs(fns_tot)
    ref_dict = dict(zip(dates,list(T.keys()))) 
    tf = []
    num_missing = []
    for date in dates:
        tf.append( datetime.strptime(date,'%Y%m%d%H'))
        num_missing.append( len(T[ref_dict[date] ] ))

    num_missing = np.array( num_missing ) 
    i1 = np.where(num_missing == 0)
    if len(i1)==0:
        # this means there are no full forecasts
        youngest = 'none'
    else:
        tf = np.array( tf )
        tf = tf[i1]
        tt = np.sort(tf)
        youngest = tt[-1].strftime('%Y%m%d%H')

    return youngest

def remove_unneeded_nwm_forecasts():

    tgood = get_youngest_full_nwm_forecast()
    dates, file_names = get_nwm_fores_we_have()
    ref_dict = dict(zip(dates,file_names)) 
    t0 = datetime.strptime(tgood,'%Y%m%d%H')

    dir0 = '/scratch/PFM_Simulations/nwm_ncs/'
    files_to_remove = []
    for date in dates:
        dt = datetime.strptime(date,'%Y%m%d%H')
        if dt<t0:
            files_to_remove.append( dir0 + ref_dict[date] )

    if len(files_to_remove)>0:
        for file in files_to_remove:
            cmd_lst = ['rm',file]
            print('removing unneeded nwm file: ', file)
            subprocess.run(cmd_lst)
    else:
        print('not removing any nwm files.')


def return_missing_hours_from_nwm_ncs(file_names):

    T = dict()
    for fnm in file_names:
        t_str = fnm[-13:-3]
        t_fore = cftime.DatetimeGregorian(int(t_str[0:4]),int(t_str[4:6]),int(t_str[6:8]),int(t_str[8:10]))
        t,_,_ = get_data_from_nwm_file(fnm)
        t = t.data

        t_hrs = []
        hrs_0 = list(range(1,241))
        for ti in t:
            dt = ti - t_fore
            t_hrs.append( int( dt.total_seconds() / 3600.0 ) )
        
        missing_hrs = [item for item in hrs_0 if item not in t_hrs]        
        T[fnm] = missing_hrs

    return T

def get_nwm_fore_url_list(yyyymmddhh):
    yyyymmdd = yyyymmddhh[0:8]
    hh = yyyymmddhh[8:]
    fore_type = 'medium_range_blend' # short_range, long_range, etc.
    hrs = np.arange(1,241,1) # hours go from 1 to 240...
    url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/v3.0/nwm.' + yyyymmdd + '/' + fore_type
    fname = ['nwm.t','z.'+fore_type+'.channel_rt.f','.conus.nc']

    url_list = []
    for hr in hrs:
        hr_str = str(int(hr)).zfill(3)
        fn = fname[0] + hh + fname[1] + hr_str + fname[2]
        url_tot = url + '/' + fn
        url_list.append(url_tot)

    return url_list

def  get_nwm_from_url(url):
    # this dumps the url to a tmp.nc file
    # loads this tmp.nc file, extracts time and Q for
    reach_ids = [948070199, 20331702, 20324441]
                #SW         Otay      TJR 20324441 is last segment near ocean.
    # and return the time and Q at these reach_ids    
    
    tmpnc = '/scratch/PFM_Simulations/nwm_ncs/nwm_tmp.nc'
    response = requests.get(url)
    # Check if the request was successful
    t4 = []
    Q3 = []
    if response.status_code == 200:
        # get data from the url and write to tmp file
        with open(tmpnc, "wb") as f:
            f.write(response.content)

        # load the tmp file
        with nc.Dataset(tmpnc) as ds:
            # Access the data variables
            rids = ds.variables['feature_id'][:]
            t = ds.variables['time']
            qq = ds.variables['streamflow'][:]
            t2 = num2date(t[:],t.units)
            t3 = t2.data

        # note, this block of code is in the hour loop and grabs only the data for the rivers we want.
        ig = [None]*3
        cnt_rid=0 # this is the reach_id index counter
        q2 = np.zeros((1,3))
        rids_out = np.zeros((1,3))
        for rids0 in reach_ids: # loops through in order
            ig= np.argwhere(rids==rids0)
            q2[0,cnt_rid] = qq[ig]
            rids_out[0,cnt_rid] = rids[ig]
            cnt_rid=cnt_rid+1

        t4 = t3
        q2np = np.array(q2)

    return t4, q2np, rids_out


def get_all_nwm_fore_data(yyyymmddhh):
    # this function gets all of the possible data currently on the 
    # nwm server for a specific forecast time
    # returns np arrays of time, discharge, and reach ids

    # list of urls needed for the yyyymmddhh forecast
    url_list = get_nwm_fore_url_list(yyyymmddhh)
    
    # check and see if the url_file exists on the nwm server
    got_url = []
    for url in url_list:
        got_it = file_url_exists(url)
        got_url.append(got_it)

    cnt = 0
    t3 = np.empty((0))
    Q3 = np.empty((0,3))
    
    # loop through urls and download the file that exists
    # and put data into some arrays...
    for url in url_list:
        if got_url[cnt]:
            t, Q, rids = get_nwm_from_url(url)
            if len(t)>0:
                # we got data so start appending
                t3 = np.concatenate((t3,t))
                Q3 = np.concatenate((Q3,Q),axis=0)

    # sort times...
    i_sort = np.argsort(t3)
    t4 = t3[i_sort]
    Q4 = Q3
    for cnt in np.arange(0,len(rids)):
        Q4[:,cnt] = Q3[i_sort,cnt]

    return t4, Q4, rids
            
def make_nwm_nc_file(yyyymmddhh,file_out):
    # this function makes an nc file of nwm discharge data
    # for the forecast starting at yyyymmddhh 
    # and output the data as a netcdf file in file_out    

    t, q, rids = get_all_nwm_fore_data(yyyymmddhh)
    # t is np array of cfdatetimeGregorians, 
    # convert to datetimes 
    # 2. Use a list comprehension to convert each cftime object
    dt_dates_list = [datetime.fromisoformat(d.isoformat()) for d in t]
    # 3. Convert the list back to a numpy array
    t_dt = np.array(dt_dates_list, dtype='object')

    nhrs = 240 # there should be 240 hours of data
    if len(t) == nhrs:
        # we only want to save the nwm file if it is full    
        print('full nwm forecast, making ', file_out)
        make_Qnc(t_dt,q,rids,file_out)
        return True
    else:
        print('not full nwm forecast, not making ',file_out)
        return False

def make_Qnc(t2,Q,rids,file_name_out):
    # makes an nc file of discharge data for multiple rivers
    # t2 is the time stamps of the flow as an array of datetimes
    # Q is the nt by n_river np array of discharge
    # rids are the reach ids of the data
    # riv_name is a np array of strings that name each river.

    #reach_ids = [948070199, 20331702, 20324441]
    # the order of reach_ids and names should correspond.
    station_ids = np.array(['Sweewater','Otay','TJRE'])
    num_stations = len(station_ids)


    with nc.Dataset(file_name_out, 'w', format='NETCDF4') as nc_file:
        # Create dimensions
        nc_file.createDimension('time', None)  # Unlimited dimension for time
        nc_file.createDimension('station', num_stations)

        # Create variables
        time_var = nc_file.createVariable('time', 'f8', ('time',))
        q_var = nc_file.createVariable('discharge', 'f8', ('time', 'station'))
        station_id_var = nc_file.createVariable('station_id', str, ('station',)) # For string station IDs
        reach_id_var = nc_file.createVariable('reach_id', int, ('station',)) # For string station IDs

        # Add attributes to variables (optional, but recommended for CF-compliance)
        time_var.units = 'days since 1999-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        q_var.units = 'm^3/s'
        q_var.long_name = 'river discharge'
        station_id_var.long_name = 'river name'
        reach_id_var.long_name = 'reach ids from National Water Model'

        # Add global attributes (optional)
        nc_file.title = 'discharge from National Water Model: medium range blend forecast'
        nc_file.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        # 3. Write data to variables
        time_var[:] = nc.date2num(t2, units=time_var.units, calendar=time_var.calendar)
        q_var[:] = Q
        station_id_var[:] = station_ids
        reach_id_var[:] = rids

def check_for_enough_data(t_riv,t):
    t_beg = t_riv[0] - 45 * timedelta(minutes=1) # t_riv is 15 overlap of forecast
    t_end = t_riv[-1] + 45 * timedelta(minutes=1)
    
    t_beg_cf = datetime_to_cftime(np.array([t_beg]))
    t_beg_cf = t_beg_cf[0]
    t_end_cf = datetime_to_cftime(np.array([t_end]))
    t_end_cf = t_end_cf[0]
   
    # cant have the river times before or after the nwm times
    if t[0] > t_beg or t[-1] < t_end:
        print('river times do not coincide with nwm times. need a different file.')
        print(t[0],t[-1])
        print(t_beg,t_end)
        return False
    
    dt_riv_hrs = int( (t_end - t_beg).total_seconds() / 3600.0 )
    condition = ((t>=t_beg_cf) & (t<=t_end_cf))
    i_nwm = np.argwhere( condition )
    if len(i_nwm) == dt_riv_hrs+1:
        print('the nwm has all the the data we need.')
        return True
    elif len(i_nwm) > dt_riv_hrs / 2:
        print('we have at least half of all the hours. good.')
        return True
    else:
        print('this nwm file will not work')
        return False

def datetime_to_cftime(times):

    times_cf = []
    for dt_obj in times:
        cftime_obj = cftime.DatetimeGregorian(
                            dt_obj.year,
                            dt_obj.month,
                            dt_obj.day,
                            dt_obj.hour,
                            dt_obj.minute,
                            dt_obj.second,
                            dt_obj.microsecond
                            )
        times_cf.append(cftime_obj)
    
    return np.array(times_cf)

def convert_cftime_to_datetime_array(cftime_array):

    dt_list = []
    for cf_obj in cftime_array:
        dt_obj = datetime(
                            cf_obj.year,
                            cf_obj.month,
                            cf_obj.day,
                            cf_obj.hour,
                            cf_obj.minute,
                            cf_obj.second,
                            cf_obj.microsecond
                            )
        
        dt_list.append(dt_obj)

    return np.array(dt_list)

def get_best_Q_nwm(t_riv):
    dates, file_names = get_nwm_fores_we_have()
    file_names = np.array( file_names ) 

    # sort the dates...
    d_dt = []
    for date in dates:
        d_dt.append(datetime.strptime(date,'%Y%m%d%H'))
    
    d_dt = np.array(d_dt)
    i_sort = np.argsort(d_dt)
    d_dt = d_dt[i_sort]

    file_names = file_names[i_sort]
    # now they are sorted, reverse them to start with the most recent...
    file_names_rev = file_names[::-1]
    
    dir0 = '/scratch/PFM_Simulations/nwm_ncs/'
    for fnm in file_names_rev:
        t,Q,rid = get_data_from_nwm_file( dir0 + fnm )
        reach_ids = [948070199, 20331702, 20324441]
        #location_names = ['SW','Otay','TJR']
        # it is in this order.
        print(rid)
        got_enough_nwm = check_for_enough_data(t_riv,t)   
        if got_enough_nwm:
            print('interpolating nwm to t_riv...')
            t_dt = convert_cftime_to_datetime_array(t)
            t_nc64 = t_dt.astype('datetime64[ns]')
            t_riv64 = t_riv.astype('datetime64[ns]')
            i_sw = np.argwhere(rid == reach_ids[0])
            i_om = np.argwhere(rid == reach_ids[1])
            i_tj = np.argwhere(rid == reach_ids[2])
            q_sw = np.squeeze(Q[:,i_sw])
            q_om = np.squeeze(Q[:,i_om])
            q_tj = np.squeeze(Q[:,i_tj])
            qi_sw = np.interp(t_riv64.astype('int64'),t_nc64.astype('int64'),q_sw)
            qi_om = np.interp(t_riv64.astype('int64'),t_nc64.astype('int64'),q_om)
            qi_tj = np.interp(t_riv64.astype('int64'),t_nc64.astype('int64'),q_tj)


            fig, ax = plt.subplots()
            ax.plot(t_riv, qi_tj )
            ax.plot(t, q_tj, '--' )


    return qi_sw, qi_om, qi_tj


def check_nwm_for_times(tpfm_str,fnm_full,fore_days):
    t,Q,rid = get_data_from_nwm_file( fnm_full )
    t_dt = convert_cftime_to_datetime_array(t)
    t_pfm = datetime.strptime(tpfm_str,'%Y%m%d%H')
    dt = t_dt - t_pfm # these are time deltas 

    # some checks
    if (dt[0]<=0) and (dt[-1].total_seconds()/3600.0/24.0 > fore_days):
        print('the nwm forecast in ', fnm_full)
        print('spans the times required for the pfm forecast ', tpfm_str)
        i1 = np.argwhere(t_dt == t_pfm)
        i2 = np.argwhere(t_dt == t_pfm + fore_days * timedelta(days=1))
        nt = i2-i1+1
        nmissing = 24*fore_days + 1 - nt
        return nmissing
    else:
        print('something is wrong and the nwm nc file doesnt have the right times')
        return -1
    
def make_nwm_pkl_from_nc(nwm_nc_fnm,info_pkl):
    PFM = initfuns.get_model_info(info_pkl)
    t,Q,rid = get_data_from_nwm_file( nwm_nc_fnm )
    reach_ids = [948070199, 20331702, 20324441]
    #location_names = ['SW','Otay','TJR']
    # it is in this order.
    t_dt = convert_cftime_to_datetime_array(t)
    t_nc64 = t_dt.astype('datetime64[ns]')

    t0 = PFM['fetch_time']
    nday  = PFM['forecast_days']    
    t2 = t0 + nday * timedelta(days=1)
    dt_riv = timedelta(hours=1) # this is the dt for the river file
    # needs to extend past the forecast times.
    t_riv = np.arange(t0,t2+3*dt_riv,dt_riv) # using 2 extra times on left. not sure why.
    t_riv64 = t_riv.astype('datetime64[ns]')

    i_sw = np.argwhere(rid == reach_ids[0])
    i_om = np.argwhere(rid == reach_ids[1])
    i_tj = np.argwhere(rid == reach_ids[2])
    q_sw = np.squeeze(Q[:,i_sw])
    q_om = np.squeeze(Q[:,i_om])
    q_tj = np.squeeze(Q[:,i_tj])
    qi_sw = np.interp(t_riv64.astype('int64'),t_nc64.astype('int64'),q_sw)
    qi_om = np.interp(t_riv64.astype('int64'),t_nc64.astype('int64'),q_om)
    qi_tj = np.interp(t_riv64.astype('int64'),t_nc64.astype('int64'),q_tj)
    # note if t_riv64 < t_nc64, then q_sw end points are used, for example

    go = True
    if go:    
        QQ = dict()
        Q = np.zeros((len(t_riv),3))
        QQ['time'] = t_riv # t3 = list? of np.array datetimes ?
        # previous XWu LV4 simulations capped TJR Q at 150 m3/s. We might want to do that here?
        Q[:,0] = qi_sw
        Q[:,1] = qi_om
        Q[:,2] = qi_tj
        QQ['discharge'] = Q
        QQ['reach_ids'] = reach_ids
        QQ['readme'] = 'discharge is in m3/s. reach_ids correspond to Sweetwater, Otay, TJR. they are the columns of discharge'

        plot_it = 0
        if plot_it == 1:
            fig, ax = plt.subplots()
            p1=ax.plot(QQ['time'],QQ['discharge'][:,0],label='Sweet Water')
            p2=ax.plot(QQ['time'],QQ['discharge'][:,1],label='Otay Mesa')
            p3=ax.plot(QQ['time'],QQ['discharge'][:,2],label='TJ')

            plt.legend()
            plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
            plt.ylabel('discharge [m3/s]')
            plt.title('PFM forecast time is: ' + t0.strftime('%Y%m%d%H') + ' | river forecast time is: ' + t_dt[0].strftime('%Y%m%d%H') )

        file_out = PFM['river_pckl_file_full']
        with open(file_out,'wb') as fp:
            pickle.dump(QQ,fp, protocol=pickle.HIGHEST_PROTOCOL)
            print('\nriver discharge data saved as pickle file')
        
        return 0



def make_nwm_q_pkl_file(info_pkl):

    PFM = initfuns.get_model_info(info_pkl)
    if 'nwm_fore_dir' not in PFM:
        PFM['nwm_fore_dir'] = '/scratch/PFM_Simulations/nwm_ncs/'
    
    dir0 = PFM['nwm_fore_dir']
    t_try = PFM['fetch_time']
    t_last = t_try - 3 * timedelta(days=1)
    use_nwm_nc = True # we hope to use an nwm nc file we have stored

    while t_try >= t_last:
        t_try_str = t_try.strftime('%Y%m%d%H')
        nwm_nc_fnm = dir0 + 'nwm_forecast_Q_' + t_try_str + '.nc'
        if os.path.exists(nwm_nc_fnm):
            print('for the PFM forecast starting ', PFM['fetch_time'])
            print('we will use ', nwm_nc_fnm)
            print('for the predicted SW, OM, and TJ flow.')
            t_try = t_try - 5 * timedelta(days=1)
        else:
            print('we did not have the file. trying to make ', nwm_nc_fnm)
            made_file = make_nwm_nc_file(t_try_str,nwm_nc_fnm)
            if made_file:
                print('the file ', nwm_nc_fnm, ' was made, exiting loop.')
                t_try = t_try - 5 * timedelta(days=1)
            else: # move to the previous 6 hour forecast
                t_try = t_try - 6 * timedelta(hours=1)
                if t_try < t_last:
                    print('shouldnt really get here')
                    print('no nwm files had the data to make the nwm Q pickle file')
                    print('as a last resort, we will assume constant flow.')
                    use_nwm_nc = False
                
    if use_nwm_nc:
        print('using nc file')
        make_nwm_pkl_from_nc(nwm_nc_fnm,info_pkl)
    else:
        #make_nwm_pkl_constants(info_pkl)
        print('got here, use constant')


def get_hind_nwm_urls(t1,t2):

    urls = []
    t_cnt = t1

    url0 = (['https://storage.googleapis.com/national-water-model/nwm.',
            '/analysis_assim/nwm.t',
            'z.analysis_assim.channel_rt.tm00.conus.nc#mode=bytes'])
     #20241011/analysis_assim/nwm.t00z.analysis_assim.channel_rt.tm00.conus.nc#mode=bytes'

    urls = []
    while t_cnt <= t2:
        yyyymmdd = t_cnt.strftime('%Y%m%d')
        hh = t_cnt.strftime('%H')
        url2 = url0[0] + yyyymmdd + url0[1] + hh + url0[2]
        urls.append(url2)
        t_cnt += timedelta(hours=1)

    return urls

def get_flow_from_nwm_ds(ds,reach_ids):
    id = ds.variables['feature_id'][:]
    t =  ds.variables['time']
    t2 = nc.num2date(t[:],t.units)
    # convert t2 to a datetime object
    t3 = datetime.strptime(t2[0].isoformat(), "%Y-%m-%dT%H:%M:%S")

    mask_val1 = (id == reach_ids[0])
    mask_val2 = (id == reach_ids[1])
    mask_val3 = (id == reach_ids[2])

    # Combine the masks using logical OR
    combined_mask = mask_val1 | mask_val2 | mask_val3

    # Get the indices where the combined mask is True
    ig = np.where(combined_mask)
    # the ordering of ig is: TJ, Otay, SW - it got reversed 

    qq = ds.variables['streamflow'][ig]
    reach_ids_2 = id[ig]

    return t3, qq, reach_ids_2

def get_nwm_dates_for_simulation(t_riv):

    # first convert t_riv (np array of datetime64 objects)
    # to datetime array
    t_riv2 = np.array([dt.item() for dt in t_riv])

    # create a set of unique_date
    unique_dates = set()
    # now add to that set, this will be unique tuples of integers
    for dt_obj in t_riv2:
        unique_dates.add((dt_obj.year, dt_obj.month, dt_obj.day))

    # Convert the set of tuples to a sorted list for consistent output
    unique_list = sorted(list(unique_dates))
    #print(unique_list)

    # Initialize an empty list to store datetime objects
    datetime_list = []

    # Iterate through the tuples and convert to datetime objects
    for year, month, day in unique_list:
        dt_object = datetime(year, month, day)
        datetime_list.append(dt_object)

    return datetime_list

def get_nwm_assim_flow(urls,reach_ids):

    nt = len(urls)
    nrid = len(reach_ids)
    t2 = []
    q2 = np.zeros((nt,nrid))
    cnt = 0
    for url in urls:
        ds = nc.Dataset(url)
        t, q, rids2 = get_flow_from_nwm_ds(ds,reach_ids)
        q2[cnt,:] = q
        t2.append(t)
        cnt       = cnt+1

    t2 = np.array(t2)
    return t2, q2, rids2

def mk_q_ncfile(t2,Q,rids,riv_names,file_name_out):
    # makes an nc file of discharge data for multiple rivers
    # fore_date is the start time of the forecast as datetime eg fore_date = datetime(2024,10,11)
    # t2 is the time stamps of the flow as an array of datetimes
    # Q is the nt by n_river np array of discharge
    # rids are the reach ids of the data
    # riv_name is a np array of strings that name each river.
    num_stations = len(riv_names)
    station_ids = riv_names

    with nc.Dataset(file_name_out, 'w', format='NETCDF4') as nc_file:
        # Create dimensions
        nc_file.createDimension('time', None)  # Unlimited dimension for time
        nc_file.createDimension('station', num_stations)

        # Create variables
        time_var = nc_file.createVariable('time', 'f8', ('time',))
        q_var = nc_file.createVariable('discharge', 'f8', ('time', 'station'))
        station_id_var = nc_file.createVariable('station_id', str, ('station',)) # For string station IDs
        reach_id_var = nc_file.createVariable('reach_id', int, ('station',)) # For string station IDs

        # Add attributes to variables (optional, but recommended for CF-compliance)
        time_var.units = 'days since 1999-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        q_var.units = 'm^3/s'
        q_var.long_name = 'river discharge'
        station_id_var.long_name = 'river name'
        reach_id_var.long_name = 'reach ids from National Water Model'

        # Add global attributes (optional)
        nc_file.title = 'discharge from National Water Model'
        nc_file.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        # 3. Write data to variables
        time_var[:] = nc.date2num(t2, units=time_var.units, calendar=time_var.calendar)
        q_var[:] = Q
        station_id_var[:] = station_ids
        reach_id_var[:] = rids


def make_nwm_assim_nc_file(fn):
    # this assumes the fn looks like *yyyymmdd.nc
    yyyymmdd = fn[-11:-3]    
    # we need the data for this day!
    t1 = datetime.strptime(yyyymmdd,'%Y%m%d')
    t2 = t1 + timedelta(hours=23)

    urls = get_hind_nwm_urls(t1,t2)

    reach_ids = [948070199, 20331702, 20324441]
                #SW         Otay      TJR 20324441 is last segment near ocean.
    station_ids = ['SW','Otay','TJR']
    reach_ids = np.array(reach_ids)
    station_ids = np.array(station_ids)
    
    # get the data from the urls...
    t, q, rids2 = get_nwm_assim_flow(urls,reach_ids)

    # do some sorting 
    i1 = np.argsort(reach_ids)
    i2 = np.argsort(rids2)
    i3 = np.argsort(i2)
    i0 = i1[i3]
    names2 = station_ids[i0] # the correct order of the flow in array q.

    t_str = t[0].strftime('%Y%m%d')
    file_name_out = fn
    print('now putting the data into the nc file')
    print(file_name_out)
    mk_q_ncfile(t,q,rids2,names2,file_name_out)
    print('...done with this day!')



def get_nwm_file_names(nwm_dir,fn_dates):
    
    files = []
    for dt in fn_dates:
        yyyymmdd = dt.strftime('%Y%m%d')
        fn = nwm_dir + 'nwm_assim_' + yyyymmdd + '.nc'
        files.append(fn)
    
    return files


def check_for_nwm_raw_files(nwm_dir,fn_dates):

    files = get_nwm_file_names(nwm_dir,fn_dates)

    for fn in files:
        if os.path.exists(fn):
            print(fn, ' exists. No need to download.')
        else:
            print(fn, ' does not exist. Making it...')
            make_nwm_assim_nc_file(fn) 
            print('...done!')

def get_data_from_nwm_file(fn):
    with nc.Dataset(fn) as ds:
        QQ = ds.variables['discharge'][:]
        tt = ds.variables['time']
        rid = ds.variables['reach_id'][:]
        time_values = tt[:]
        # Get the units and calendar attributes from the time variable
        time_units = tt.units
        time_calendar = getattr(tt, 'calendar', 'standard') # Default to 'standard' if no calendar attribute
        # Convert to datetime objects
        tt_dt = nc.num2date(time_values, units=time_units, calendar=time_calendar)

    return tt_dt, QQ, rid

def get_data_from_nwm_nc_files(file_names):
    # this extracts the data in file names...

    t_nc1 = np.empty((0))
    q_nc = np.empty((0,3))

    for fn in file_names:
        t, q, rid_nc = get_data_from_nwm_file(fn)
        t_nc1 = np.concatenate((t_nc1, t))
        q_nc = np.concatenate((q_nc,q),axis=0)

    # turn the cftimes to datetimes
    datetime_list = [datetime(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
        ) for dt in t_nc1]

    # Convert the list to a NumPy array of datetime.datetime objects
    t_nc = np.array(datetime_list)
        
    return t_nc, q_nc, rid_nc

def get_nwm_analysis_flow(t_riv,pkl_fnm):

    PFM = initfuns.get_model_info(pkl_fnm)

    # first get list of unique datetimes that would cover t_riv
    # separated by 1 day
    fn_dates = get_nwm_dates_for_simulation(t_riv)

    # where will the files be located...
    if 'nwm_dir' in PFM:
        nwm_dir = PFM['nwm_dir']
    else:
        nwm_dir = '/dataSIO/PHM_Simulations/raw_download/nwm_files/'

    # get the nwm .nc file names that should be archinved that are needed for the simulation
    file_names = get_nwm_file_names(nwm_dir,fn_dates)

    # now check to see if the nwm_dir has these files...
    # if the needed file does NOT exist, this gets them!
    check_for_nwm_raw_files(nwm_dir,fn_dates)
    # we should have all the files now

    # these are the reach ids that are in the .nc files
    reach_ids = [948070199, 20331702, 20324441]
    #location_names = ['SW','Otay','TJR']
    reach_ids = np.array(reach_ids)

    # 
    print('need to get the data from the nc files...')
    t_nc, q_nc, rid_nc = get_data_from_nwm_nc_files(file_names)

    t_nc64 = t_nc.astype('datetime64[ns]')

    i_sw = np.argwhere(rid_nc == reach_ids[0])
    i_om = np.argwhere(rid_nc == reach_ids[1])
    i_tj = np.argwhere(rid_nc == reach_ids[2])

    q_sw = np.squeeze(q_nc[:,i_sw])
    q_om = np.squeeze(q_nc[:,i_om])
    q_tj = np.squeeze(q_nc[:,i_tj])

    qi_sw = np.interp(t_riv.astype('int64'),t_nc64.astype('int64'),q_sw)
    qi_om = np.interp(t_riv.astype('int64'),t_nc64.astype('int64'),q_om)
    qi_tj = np.interp(t_riv.astype('int64'),t_nc64.astype('int64'),q_tj)

    return qi_sw, qi_om, qi_tj

def get_tj_flow_version_PFM(t_riv,pkl_fnm):
    # load the observations...

    dt_riv = timedelta(minutes=15)
    t = np.arange(t_riv[0]-5*timedelta(days=1),t_riv[-1],dt_riv)
    t5_1 = t_riv[0]-5*timedelta(days=1)
    t1_1 = t_riv[0]-1*timedelta(days=1)
    t_2  = t_riv[0]
    mask1 = (t >= t1_1) & (t <= t_2)
    mask5 = (t >= t5_1) & (t <= t_2)
    i1 = np.where(mask1)[0]
    i5 = np.where(mask5)[0]

    q_obs = get_tj_observed_flow(t,pkl_fnm,method='raw_interp')
    Q_1 = np.mean(q_obs[i1])
    Q_5 = np.mean(q_obs[i5])

    # set up alpha
    dt = t_riv - t_riv[0]
    dt_sec = []
    for dtt in dt:
        dt_sec.append(dtt / np.timedelta64(1, 's'))
    dt_day = np.array(dt_sec) / 3600 /24
    tau_day = 0.5 # time scale to go from flow to dry
    alpha = np.exp(-dt_day / tau_day)

    _, _, q_tjnwm = get_nwm_analysis_flow(t_riv,pkl_fnm)    

    std_cut = 1
    if np.std(q_tjnwm) > std_cut:
        print('nwm suggests rain, use nwm!')
        q_pfm = q_tjnwm
    elif Q_5 > Q_1:
        print('using 1 day persistence')
        q_pfm = Q_1 * np.ones(np.shape(t_riv))
    else:
        print('using 1 day to 5 day persistence')
        q_pfm = Q_1 * alpha + Q_5 * (1 - alpha)


    return q_pfm

def get_pb_flow_and_dye(t_riv,pkl_fnm):
    
    PFM = initfuns.get_model_info(pkl_fnm)
    if 'pb_time_switch' in PFM:
        time_switch = PFM['pb_time_switch']
        print('assuming flow at PB switched on ', time_switch, ' from PFM dictionary')
    else:
        print('no time_switch in PFM, using hard coded value...')
        time_switch = datetime(2025,4,1)
        print(time_switch)

    # these are the values we were using and we are using now.
    Q1 = 2.1906
    Q2 = 2.0
    dye1 = 0.7
    dye2 = 0.5
    i1 = np.argwhere(t_riv <= time_switch)
    i2 = np.argwhere(t_riv > time_switch)
    q_pb = np.zeros(np.shape(t_riv))
    dye_pb = np.zeros(np.shape(t_riv))
    if (i1.size == 0) & (i2.size == 0):
        print('something is wrong, no times less than ', time_switch)
        print('and no times great than ', time_switch)
        sys.exit(1)
    elif not (i1.size == 0):
        q_pb[i1] = Q1
        dye_pb[i1] = dye1
    elif not (i2.size == 0):
        q_pb[i2] = Q2
        dye_pb[i2] = dye2

    return q_pb, dye_pb

def check_qtj_obs_file(fn_qtj_obs,t1,t2):
    # this function loads fn_qtj_obs, and checks to see if there
    # is data between t1 and t2...

    t_obs, _ = get_all_tj_observed_data(fn_qtj_obs)
    i_good = np.argwhere( (t_obs >= t1) & (t_obs <= t2) )
    if i_good.size == 0:
        return False
    else:
        return True

def get_more_qtj_obs_data(fn_qtj_obs,t1,t2):
    print('we are replacting the file ', fn_qtj_obs)
    print('with a new one that has more data.')
    t_obs, _ = get_all_tj_observed_data(fn_qtj_obs)
    t_min = t_obs[0]
    t_max = t_obs[-1]
    if t_min <= t1:
        t1_get = t_min
    else:
        t1_get = t1
    if t_max >= t2:
        t2_get = t_max
    else:
        t2_get = t2
    t1_str = t1_get.strftime('%Y-%m-%d')
    t2_str = t2_get.strftime('%Y-%m-%d')
    make_obs_Qtj_file(t1_str,t2_str,fn_qtj_obs)    

def make_obs_Qtj_file(t1_str,t2_str,fn_qtj_obs):    
    file_url = ('https://waterdata.ibwc.gov/AQWebportal/Export/BulkExport?DateRange=Custom&StartTime=' + 
            t1_str + '%2000%3A00&EndTime=' + 
            t2_str + '%2000%3A00&TimeZone=0&Calendar=CALENDARYEAR&Interval=PointsAsRecorded&Step=1&ExportFormat=csv&TimeAligned=True&RoundData=False&IncludeGradeCodes=False&IncludeApprovalLevels=False&IncludeQualifiers=False&IncludeInterpolationTypes=False&Datasets[0].DatasetName=Discharge.Best%20Available%4011013300&Datasets[0].Calculation=Instantaneous&Datasets[0].UnitId=128&_=1754421522237')
    local_save_path = fn_qtj_obs
    download_ibwc_file(file_url, local_save_path)


def get_tj_observed_flow(t_riv,pkl_fnm,method='raw_interp'):
    # this function returns the observed tj flow on the datetimes in t_riv

    PFM = initfuns.get_model_info(pkl_fnm)

    # first check to see if the observations exist for the times t_riv
    # for testing
#    PFM['qtj_obs_fname_full'] = '/dataSIO/PHM_Simulations/raw_download/qtj_obs_data/qtj_raw_20200101_20250901.csv'

    fn_qtj_obs = PFM['qtj_obs_fname_full']
    print('the raw tj river discharge is in the file ', fn_qtj_obs)
    print('check and see if this file has the data...')
    if os.path.exists(fn_qtj_obs):
        print(f"'{fn_qtj_obs}' exists. Checking to see if the data in it covers the time range:")
        t1 = t_riv[0]  - 7 * timedelta(days=1)
        t2 = t_riv[-1] + 7 * timedelta(days=1)
        print(f"'{t1}' to '{t2}'")
        has_data = check_qtj_obs_file(fn_qtj_obs,t1,t2)
        if has_data:
            print(f"'{fn_qtj_obs}' has the data we need. Retrieving it from the file...")
            t_tj_raw, q_tj_raw = get_all_tj_observed_data(fn_qtj_obs)
        else:
            print(f"'{fn_qtj_obs}' did not have data we need. Making a new file...")
            get_more_qtj_obs_data(fn_qtj_obs,t1,t2)
            t_tj_raw, q_tj_raw = get_all_tj_observed_data(fn_qtj_obs)

    else:    
        print(f"'{fn_qtj_obs}' does not exist. Need to make.")
        print(f"Making it based on the hindcast times and padding by 7 days on each end.")
        t1 = PFM['sim_start_time'] - 7 * timedelta(days=1)
        t2 = PFM['sim_end_time']   + 7 * timedelta(days=1)
        t1_str = t1.strftime('%Y-%m-%d')
        t2_str = t2.strftime('%Y-%m-%d')
        make_obs_Qtj_file(t1_str,t2_str,fn_qtj_obs)
        t_tj_raw, q_tj_raw = get_all_tj_observed_data(fn_qtj_obs)    

    if method == 'raw_interp':
        t_original = (t_tj_raw.astype('datetime64[ns]')).astype('int64')
        t_i = (t_riv.astype('datetime64[ns]')).astype('int64')
        q_tj = np.interp(t_i, t_original, q_tj_raw)
        # now q_tj_raw is interpolated to the time stamps of t_riv

    return q_tj


def get_all_tj_observed_data(fn_qtj_obs):
    # this returns all of the data in fn_qtj_obs
    # both time and q [m3/s]
    data = load_csv_skip_header_footer(fn_qtj_obs, header_rows=5, footer_rows=1, delimiter=',')
    
    t_obs2 = []
    q_obs2 = []
    for row in data:
        t_obs2.append(datetime.strptime(row[0],'%Y-%m-%d %H:%M:%S'))
        q_obs2.append(float(row[1]))

    t_obs = np.array(t_obs2)
    q_obs = np.array(q_obs2)
    return t_obs, q_obs

def file_url_exists(url):
    """
    Checks if a file at the given URL exists and is accessible.

    Args:
        url (str): The URL of the file to check.

    Returns:
        bool: True if the file exists and is accessible (returns a 200, 301, or 302 status code), 
              False otherwise.
    """
    try:
        # Create a Request object and set the method to 'HEAD'
        req = urllib.request.Request(url, method='HEAD')
        
        # Open the URL and get the response
        with urllib.request.urlopen(req) as response:
            # Check for successful status codes (200 OK, 301 Moved Permanently, 302 Found)
            return response.getcode() in (200, 301, 302)
    except HTTPError as e:
        # Handle HTTP errors (e.g., 404 Not Found, 403 Forbidden)
        print(f"HTTP Error for {url}: {e.code} - {e.reason}")
        return False
    except URLError as e:
        # Handle URL errors (e.g., network issues, invalid URL)
        print(f"URL Error for {url}: {e.reason}")
        return False
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred for {url}: {e}")
        return False
    
def get_river_flow_nwm(yyyymmddhh,t_pfm_str,pkl_fnm):
    # yyyymmddhh is the start time of the river forecast
    # t_pfm_str [in yyyymmddhh] is the start time of the PFM forecast
    # this is coded to work only if t_pfm is larger than yyyymmddhh, the river forecast start time
    # we will typically use t_fore = yyyymmddhh + 6 hr. Using the previous river forecast ensures 
    # that the river forecast is posted to their server
    PFM = initfuns.get_model_info(pkl_fnm)
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
    hh = yyyymmddhh[8:]
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
        fn = fname[0] + hh + fname[1] + hr_str + fname[2]
        url_tot = url + '/' + fn
        rc = file_url_exists(url)
        if rc:
            # get the file
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
            
                # note, this block of code is in the hour loop and grabs only the data for the rivers we want.
                ig = [None]*3
                cnt=0 # this is the reach_id index counter
                for rids0 in reach_ids:
                    ig= np.argwhere(rids==rids0)
                    Q[cnt1,cnt] = qq[ig]
                    cnt=cnt+1

                cnt1 = cnt1+1 # this is the hour index counter
        else:
            print('we were trying to NWM river data from the url_tot:')
            print(url_tot)
            print('There were problems. River pickle file will not be made.')
            print('exiting this function...')
            sys.exit(1)


    plot_it = 0
    if plot_it == 1:
        fig, ax = plt.subplots()
        p1=ax.plot(t3,Q[:,0],label='Sweet Water')
        p2=ax.plot(t3,Q[:,1],label='Otay Mesa')
        p3=ax.plot(t3,Q[:,2],label='TJ')

        plt.legend()
        plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
        plt.ylabel('discharge [m3/s]')
        plt.title('PFM forecast time is: ' + t_pfm_str + ' | river forecast time is: ' + yyyymmddhh )
        fn_out = PFM['lv4_plot_dir'] + '/river_discharge_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        plt.savefig(fn_out, dpi=300)

    QQ = dict()
    QQ['time'] = t3 # t3 = list? of np.array datetimes ?
    # previous XWu LV4 simulations capped TJR Q at 150 m3/s. We might want to do that here?
    QQ['discharge'] = Q
    QQ['reach_ids'] = reach_ids
    QQ['readme'] = 'discharge is in m3/s. reach_ids correspond to Sweetwater, Otay, TJR. they are the columns of discharge'

    with open(file_out,'wb') as fp:
        pickle.dump(QQ,fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('\nriver discharge data saved as pickle file')

    return 0

def get_river_temp(t_riv,pkl_fnm):
    # triv is days past reference time, what we interpolate to...
    PFM = initfuns.get_model_info(pkl_fnm)
    fatm = PFM['lv4_forc_dir'] + '/' + PFM['lv4_atm_file'] 
    RMG = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
    #print(RMG.keys())

    ds = nc.Dataset(fatm)
    temp_air = ds['Tair'][:]
    t_air = ds['tair_time'][:]
    msk2d = RMG['mask_rho']
    msk3d = np.broadcast_to( msk2d==0 , temp_air.shape)
    temp_river = np.mean(temp_air[msk3d])
    nt,_,_ = np.shape(temp_air)
    temp_river_time0 = np.zeros(nt)
    for a in np.arange(nt):
        tmp = temp_air[a,:,:]
        temp_river_time0[a] = np.mean( tmp[msk2d==0] )

    Fz = interp1d(t_air,temp_river_time0,bounds_error=False,kind='linear',fill_value=(temp_river_time0[0],temp_river_time0[-1]))
                
    temp_river_time = Fz(t_riv)
    t_riv_dt = PFM['modtime0'] + t_riv * timedelta(days=1)

    plot_it = 1
    if plot_it == 1:
        fig, ax = plt.subplots()
        p1=ax.plot(t_riv_dt,temp_river_time)
        plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
        plt.ylabel('river_temperature [C]')
        plt.title('all 3 rivers have this temperature for: ' + PFM['sim_time_1'].strftime('%Y%m%d%H') )
        fn_out = PFM['lv4_plot_dir'] + '/river_temperature_' + PFM['sim_time_1'].strftime('%Y%m%d%H') + '.png'
        plt.savefig(fn_out, dpi=300)

    # temp_river is the mean over land and time
    # temp_river_time is the mean over land at each time stamp.
    return temp_river, temp_river_time

def download_ibwc_file(url, save_path):
    """
    Downloads a file from a given URL and saves it to a specified path.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file will be saved.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True) # Use stream=True for large files

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Write the content of the response to a local file
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully: {save_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_csv_skip_header_footer(filepath, header_rows=1, footer_rows=0, delimiter=','):
    """
    Loads a CSV file, skipping a specified number of header and footer rows.

    Args:
        filepath (str): The path to the CSV file.
        header_rows (int): The number of rows to skip from the beginning (header).
                           Defaults to 1 (skipping a single header row).
        footer_rows (int): The number of rows to skip from the end (footer).
                           Defaults to 0 (no footer rows skipped).
        delimiter (str): The character used to separate values in the CSV.
                         Defaults to a comma.

    Returns:
        list: A list of lists, where each inner list represents a row of data
              from the CSV file, with header and footer rows removed.
    """
    data = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)

        # Read all rows into a temporary list
        all_rows = list(reader)

        # Determine the effective start and end indices for data rows
        start_index = header_rows
        end_index = len(all_rows) - footer_rows

        # Extract the data rows
        if start_index < end_index:  # Ensure there are data rows to extract
            data = all_rows[start_index:end_index]

    return data

def get_observed_TJriver_flow(custom_range,t1_str,t2_str,pkl_fnm):
    #t1_str = '2024-10-10'
    #t2_str = '2024-10-17'
    PFM = initfuns.get_model_info(pkl_fnm)


    if custom_range:
        file_url = ('https://waterdata.ibwc.gov/AQWebportal/Export/BulkExport?DateRange=Custom&StartTime=' + 
            t1_str + '%2000%3A00&EndTime=' + 
            t2_str + '%2000%3A00&TimeZone=0&Calendar=CALENDARYEAR&Interval=PointsAsRecorded&Step=1&ExportFormat=csv&TimeAligned=True&RoundData=False&IncludeGradeCodes=False&IncludeApprovalLevels=False&IncludeQualifiers=False&IncludeInterpolationTypes=False&Datasets[0].DatasetName=Discharge.Best%20Available%4011013300&Datasets[0].Calculation=Instantaneous&Datasets[0].UnitId=128&_=1754421522237')
    else:
        file_url = 'https://waterdata.ibwc.gov/AQWebportal/Export/BulkExport?DateRange=Days7&TimeZone=0&Calendar=CALENDARYEAR&Interval=PointsAsRecorded&Step=1&ExportFormat=csv&TimeAligned=True&RoundData=False&IncludeGradeCodes=False&IncludeApprovalLevels=False&IncludeQualifiers=False&IncludeInterpolationTypes=False&Datasets[0].DatasetName=Discharge.Best%20Available%4011013300&Datasets[0].Calculation=Instantaneous&Datasets[0].UnitId=128&_=1754417066794'


    #local_save_path = "/home/mspydell/research/LV4_river_stuff/IBWC_Qtrje_custom.csv"
    local_save_path = PFM['qtj_obs_fname_full']

    download_ibwc_file(file_url, local_save_path)
    data = load_csv_skip_header_footer(local_save_path, header_rows=5, footer_rows=1, delimiter=',')
    t_obs2 = []
    q_obs2 = []
    for row in data:
        t_obs2.append(datetime.strptime(row[0],'%Y-%m-%d %H:%M:%S'))
        q_obs2.append(float(row[1]))

    t_obs = np.array(t_obs2)
    q_obs = np.array(q_obs2)

    return t_obs, q_obs

def get_forecasted_Q_IBWC(pkl_fnm):
    #file_in = '/scratch/PFM_Simulations/LV4_Forecast/Forc/river_Q.pkl'
    PFM = initfuns.get_model_info(pkl_fnm)

    file_in= PFM['river_pckl_file_full']
    with open(file_in,'rb') as fp:
        NWM = pickle.load(fp)

    t_nwm = NWM['time']
    q_nwm = NWM['discharge'][:,2]

    # if the 1st argument below is False, then we get the most recent 7 days of data
    # if True, then from '2025-08-01' to '2025-08-08' etc
    # the last time stamp is about 1-2 hours before current time. Nice!
    tobs,Qobs = get_observed_TJriver_flow(False,'2025-08-01','2025-08-08',pkl_fnm)

    tobs_end = tobs[-1]

    #tf_dt is the start time of the forecast in datetime
    tf_dt = t_nwm[0] #datetime.strptime(t_fore,'%Y%m%d%H')
    start_time = tf_dt - 1 * timedelta(days=1)
    start_time_2 = tf_dt - 5 * timedelta(days=1)
    end_time = tf_dt

    # Create a boolean mask
    # This directly compares datetime objects within the array
    if tobs_end < start_time:
        # just use the last day. This should not be used
        mask = tobs >= (tobs_end - 1*timedelta(days=1))
        print('using just the last day of Qobs for average. Should not really be here.')
        print('as the tobs and tnwm do not overlap properly')
    else:
        mask = (tobs >= start_time) & (tobs <= end_time)

        # Get the indices where the mask is True
    indices = np.where(mask)[0]
        # take the mean over these indices
        # this is persistence!
    Qb1 = np.mean( Qobs[indices] )

    if tobs_end < start_time_2:
        # just use the last day. This should not be used
        mask = tobs >= (tobs_end - 5 *timedelta(days=1))
        print('using just the last 5 days of Qobs for average. Should not really be here.')
        print('as the tobs and tnwm do not overlap properly')
    else:
        mask3 = (tobs >= start_time_2) & (tobs <= end_time)
    
    i2 = np.where(mask3)[0]

    use_clim = 0
    if use_clim == 1:
        PFM['Q_tjr_climatology'] = 0.27
        Qb2 = PFM['Q_tjr_climatology'] # hard coded
    else:
        if len(i2) == 0:
            Qb2 = np.mean( Qobs ) # just take the mean of the whole thing for a number
        else:
            Qb2 = np.mean( Qobs[i2] ) # this is super persistence

    
    # here is the persistence forecast
    Qf_p = Qb1*np.ones(np.shape(t_nwm))
    Qf_sp = Qb2*np.ones(np.shape(t_nwm))

    # here is persistence + NMW'
    Qf_pnwm = Qf_p + q_nwm - q_nwm[0]
    
    # set up alpha
    dt = t_nwm - t_nwm[0]
    dt_sec = []
    for dtt in dt:
        dt_sec.append(dtt / np.timedelta64(1, 's'))
    dt_day = np.array(dt_sec) / 3600 /24
    tau_day = 0.5 # time scale to go from flow to dry
    alpha = np.exp(-dt_day / tau_day)

    # here is 1 day mean, to dry, with NWM' added too
    # this one does pretty well over all

    use_Qp_cut = 0
    use_Qp_nwmp = 0
    use_old = 0
    use_nwm_cut = 1

    if use_nwm_cut == 1:
        Q_cut = 1.0 # this is the cutoff for nwm std(Q). if bigger than this, rain.
    else:
        Q_cut = 4.0
        if Qb1 < Q_cut:
            Qc = Qf_p[:,0]
        elif Qb1 >= Q_cut:
            Qc = q_nwm

    if use_Qp_cut == 1 and Qb1 < Q_cut:
        QQ = Qc
        print('using Q = Qp as Qp<Qcut, Qcut=',Q_cut)
    elif use_Qp_cut == 1 and Qb1 >= Q_cut:
        QQ = Qc
        print('using Q = NWM as Qp>Qcut, Qcut=',Q_cut)
    elif use_Qp_nwmp == 1:
        QQ = Qf_pnwm
        print('using Q = Qp + NWM(t) - NWM(t=0)')
    elif use_old == 1:
        print('using NWM for Q (original method)')
        QQ = q_nwm
    elif use_nwm_cut == 1:
        std_nwm = np.std(q_nwm)
        print('using cutoff based on std NWM,  = ', std_nwm, ' m3/s')
        print('the std NWM cutoff is ', Q_cut, ' m3/s')
        if std_nwm > Q_cut:
            print('forecasted increased flow, use NWM')
            QQ = q_nwm
        else:
            print('no forecasted increased flow...')
            if Qb1 < Qb2:
                print('1 day < 5 day, use Q_1day')
                QQ = Qf_p
            else:
                print('1 day > 5 day, use Q_1 * alpha + Q_5 * (1-alpha)')
                QQ = Qf_p * alpha + Qf_sp * (1 - alpha)
    elif use_old == 1:
        QQ = q_nwm

    Q3 = dict()
    Q3['Q_nwm'] = q_nwm
    Q3['Q_p'] = Qf_p
    Q3['Q_sp'] = Qf_sp
    Q3['Q_pnwm'] = Qf_pnwm
    Q3['time'] = t_nwm

    # return both the time and discharge
    #river_pkl_2 = PFM['river_pkl2']
    #print('saving different Qs to ', river_pkl_2)
    #with open(river_pkl_2,'wb') as fp:
    #    pickle.dump(Q3,fp, protocol=pickle.HIGHEST_PROTOCOL)
    #    print('\nriver discharge varieties saved as pickle file')

    #t_nwm = NWM['time']
    #q_nwm = NWM['discharge'][:,2]

    plot_it = 1
    if plot_it == 1:
        print('plotting river discharge')
        fig, ax = plt.subplots()
        p1=ax.plot(t_nwm,NWM['discharge'][:,0],':',label='NWM Sweet Water')
        p2=ax.plot(t_nwm,NWM['discharge'][:,1],':',label='NWM Otay Mesa')
        p3=ax.plot(t_nwm,NWM['discharge'][:,2],linewidth=2,label='NWM TJ')
        p4=ax.plot(tobs,Qobs,label='IBWC',linewidth=2)        
        p5=ax.plot(tobs[i2],Qobs[i2],'-k',linewidth=.25)
        p6=ax.plot(t_nwm,Q3['Q_p'],label='Q_1day')
        p7=ax.plot(t_nwm,Q3['Q_sp'],label='Q_5day')
        p8=ax.plot(t_nwm,QQ,'--k',label='Q_forecast')


    #t_pfm = datetime.strptime(t_pfm_str,'%Y%m%d%H')

        t0 = PFM['fetch_time']
        t_pfm_str = t0.strftime('%Y%m%d%H')
        yyyymmddhh = (t0 - 0.25*timedelta(days=1)).strftime('%Y%m%d%H')
        plt.legend()
        plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
        plt.ylabel('discharge [m3/s]')
        plt.title('PFM forecast time is: ' + t_pfm_str + ' | river forecast time is: ' + yyyymmddhh )
        fn_out = PFM['lv4_plot_dir'] + '/river_discharge_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        plt.savefig(fn_out, dpi=300)

    return t_nwm, QQ

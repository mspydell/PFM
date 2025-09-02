# library of river functions
import sys
import csv
import os
from datetime import datetime, timedelta
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
    
        # ds = nc.Dataset(url_tot) DOESNT WORK. NOT the right server type on their end?
        
        # note, this block of code is in the hour loop and grabs only the data for the rivers we want.
        ig = [None]*3
        cnt=0 # this is the reach_id index counter
        for rids0 in reach_ids:
            ig= np.argwhere(rids==rids0)
            Q[cnt1,cnt] = qq[ig]
            cnt=cnt+1

        cnt1 = cnt1+1 # this is the hour index counter
        
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
    QQ['time'] = t3
    # previous XWu LV4 simulations capped TJR Q at 150 m3/s. We might want to do that here?
    QQ['discharge'] = Q
    QQ['reach_ids'] = reach_ids
    QQ['readme'] = 'discharge is in m3/s. reach_ids correspond to Sweetwater, Otay, TJR. they are the columns of discharge'

    with open(file_out,'wb') as fp:
        pickle.dump(QQ,fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('\nriver discharge data saved as pickle file')


def get_river_temp(pkl_fnm):
    PFM = initfuns.get_model_info(pkl_fnm)
    fatm = PFM['lv4_forc_dir'] + '/' + PFM['lv4_atm_file'] 
    RMG = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
    #print(RMG.keys())

    ds = nc.Dataset(fatm)
    temp_air = ds['Tair'][:]
    msk2d = RMG['mask_rho']
    msk3d = np.broadcast_to( msk2d==0 , temp_air.shape)
    temp_river = np.mean(temp_air[msk3d])
    nt,_,_ = np.shape(temp_air)

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
    Qf_pnwm = Qf_p[:,0] + q_nwm - q_nwm[0]
    
    # set up alpha
    dt = t_nwm - t_nwm[0]
    dt_sec = []
    for dtt in dt:
        dt_sec.append(dtt[0].total_seconds())
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
                QQ = Qf_p[:,0]
            else:
                print('1 day > 5 day, use Q_1 * alpha + Q_5 * (1-alpha)')
                QQ = Qf_p[:,0] * alpha + Qf_sp[:,0] * (1 - alpha)
    elif use_old == 1:
        QQ = q_nwm

    Q3 = dict()
    Q3['Q_nwm'] = q_nwm
    Q3['Q_p'] = Qf_p[:,0]
    Q3['Q_sp'] = Qf_sp[:,0]
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

# library of atm functions
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
from get_PFM_info import get_PFM_info
import grid_functions as grdfuns

#from pydap.client import open_url


def get_atm_data_as_dict():
    
    PFM        = get_PFM_info()   
    ftime = PFM['fetch_time']
    yyyymmdd = "%d%02d%02d" % (ftime.year, ftime.month, ftime.day)

    #yyyymmdd   = PFM['yyyymmdd']
    hhmm       = PFM['hhmm']
    run_type   = PFM['run_type']
    atm_mod    = PFM['atm_model']
    get_method = PFM['atm_get_method']

    fname_out  = PFM['lv1_forc_dir'] + '/' + PFM['atm_tmp_pckl_file']
    # import pygrib

    # this function will returm ATM, a dict of all atmospheric fields ROMS requires
    # keys will be the ROMS .nc variable names.
    # they will be on the atm grid (not roms grid)
    # the forecast data will start at yyyymmdd and 000z. All times of the forecast will
    # be returned. atm_mod is the type of atm models used, one of:
    # 'nams_nest', 'nam_1hr', or 'hrrr', or 'gfs'
    # get_method, is the type of method used, either 'open_dap' or 'grib_download'

    # the code in here goes from the start date to all forecast dates
    #d1=datetime(2024,6,17) # a datetime object, the date is the date of forecast

    def get_roms_times(fore_date,t,t_ref):
        # this funtion returns times past t_ref in days
        # consistent with how ROMS likes it
        from datetime import datetime, timedelta

        d1=datetime.fromisoformat(fore_date)
        d1 = d1 + int(hhmm[0:2])*timedelta(days=1/24) # d1 is the start time of the forecast
        t2 = t-t[0] # an ndarray of days, t is from atm import
        t3 = d1 + t2 * timedelta(days=1)
        # t3 looks good and is the correct time stamps of the forecast.
        # But for ROMS we need ocean_time which is relative to 1999,1,1. 
        # in seconds. So...
        tr = t3 - t_ref
        tr_days = tr.astype("timedelta64[ms]").astype(float) / 1000 / 3600 / 24
        # tr_sec is now an ndarray of days past the reference day
        return tr_days

    if get_method == 'open_dap_pydap' or get_method == 'open_dap_nc' and run_type == 'forecast':
        # using opendap to get nomads data using pydap
        # with this method the data is not downloaded directly, initially
        # and the data is rectilinear, lon and lat are both vectors

        # nam_nest is at 3 hr resolution, for 2.5 days
        # 0.03 deg horizontal resolution
        nam_nest = 'https://nomads.ncep.noaa.gov/dods/nam/nam' + yyyymmdd + '/nam_conusnest_' + hhmm[0:2] + 'z'

        # nam_1hr is at 1 hr resolution, for 1.5 days
        # 0.11 deg horizontal resolution
        nam_1hr = 'https://nomads.ncep.noaa.gov/dods/nam/nam' + yyyymmdd + '/nam1hr_' + hhmm[0:2] + 'z'

        # hires_fv3 is at 1 hr resolution, for 2.5 days
        # 0.05 deg horizontal resolution
        # this one doesn't have short and long wave. DARN. DONT USE!!!!
        # hires_fv3 = 'https://nomads.ncep.noaa.gov/dods/hiresw/hiresw' + yyyymmdd + '/hiresw_conusfv3_00z'

        # hrrr is at 1 hr resolution, for 2 days, for 00z forecast
        # 0.03 deg horizontal resolution 
        # this is clearly the highest resolution product
        hrrr = 'https://nomads.ncep.noaa.gov/dods/hrrr/hrrr' + yyyymmdd + '/hrrr_sfc.t' + hhmm[0:2] + 'z'

        # gfs is at 1 hr resolution, for 5 days
        # 0.25 deg horizontal resolution
        gfs = 'https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs' + yyyymmdd + '/gfs_0p25_' + hhmm[0:2] + 'z'
        gfs_1hr = 'https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs' + yyyymmdd + '/gfs_0p25_1hr_' + hhmm[0:2] + 'z'
        
            #  http://nomads.ncep.noaa.gov:80/dods/gfs_0p25_1hr/gfs20241003/gfs_0p25_1hr_06z
            #  https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs20241003/gfs_0p25_06z.info
        
        if atm_mod == 'nam_nest':
            atm_name = nam_nest
        if atm_mod == 'nam_1hr':
            atm_name = nam_1hr
        if atm_mod == 'hrrr':
            atm_name = hrrr
        if atm_mod == 'gfs':
            atm_name = gfs
        if atm_mod == 'gfs_1hr':
            atm_name = gfs_1hr

        print('\nthe url for data is:')
        print(atm_name)
        print('\n')
        # define the box to get data in
        lt_min = PFM['latlonbox']['L1'][0]
        lt_max = PFM['latlonbox']['L1'][1]
        ln_min = PFM['latlonbox']['L1'][2]
        ln_max = PFM['latlonbox']['L1'][3]

        if get_method == 'open_dap_pydap':
            # open a connection to the opendap server. This could be made more robust? 
            # by trying repeatedly?
            # open_url is sometimes slow, and this block of code can be fast (1s), med (6s), or slow (>15s)
            dataset = open_url(atm_name)

            time = dataset['time']         # ???
            ln   = dataset['lon']          # deg
            lt   = dataset['lat']          # deg
            #Pres = dataset['pressfc']      # at surface, pa
            Pres = dataset['prmslmsl']     # pressure reduced to mean sea level.
            Temp = dataset['tmpsfc']       # at surface, K
            Hum  = dataset['rh2m']         # at 2 meters, %
            U    = dataset['ugrd10m']      # at 10 meters, m/s
            V    = dataset['vgrd10m']      # at 10 meters, m/s
            if atm_name == nam_1hr:        # or atm_name == hires_fv3
                Rain = dataset['apcpsfc']  # total precip, kg/m2, need to take diffs of times.   
            else:
                Rain = dataset['pratesfc'] # at surface, kg/m2/s
            Swd  = dataset['dswrfsfc']     # downward short-wave at surface, W/m2
            Swu  = dataset['uswrfsfc']     # upward short-wave at surface, W/m2
            Lwd  = dataset['dlwrfsfc']     # downward long-wave at surface, W/m2
            Lwu  = dataset['ulwrfsfc']     # upward long-wave at surface, W/m2

            # here we find the indices of the data we want for LV1
            Ln0    = ln[:]
            Lt0    = lt[:]
            if atm_name == gfs or atm_name == gfs_1hr:
                Ln0 = Ln0-360.0

            iln    = np.where( (Ln0>=ln_min)*(Ln0<=ln_max) ) # the lon indices where we want data
            ilt    = np.where( (Lt0>=lt_min)*(Lt0<=lt_max) ) # the lat indices where we want data



            # now time to get the numbers that we want. I get them as arrays?
            pres2 = Pres[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]] # indexing looks bad but works
            t     = pres2.time[:].data
            # return the roms times past tref in days
            #t_ref = datetime(1999,1,1)
            t_ref = PFM['modtime0']
            t_rom = get_roms_times(yyyymmdd,t,t_ref)
            lon   = pres2.lon[:].data
            lat   = pres2.lat[:].data
            pres  = pres2.array[:,:,:].data

            # we will get the other data directly
            temp = Temp.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            hum  =  Hum.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            u    =    U.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            v    =    V.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            rain = Rain.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            swd  =  Swd.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            swu  =  Swu.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            lwd  =  Lwd.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            lwu  =  Lwu.array[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            # I think everything is an np.ndarray ?
        
        if get_method == 'open_dap_nc':
            # open a connection to the opendap server. This could be made more robust? 
            # by trying repeatedly?
            # open_url is sometimes slow, and this block of code can be fast (1s), med (6s), or slow (>15s)
            dataset = nc.Dataset(atm_name)

            time = dataset['time']         # ???
            ln   = dataset['lon']          # deg
            lt   = dataset['lat']          # deg
            #Pres = dataset['pressfc']      # at surface, pa
            Pres = dataset['prmslmsl']     # pressure reduced to mean sea level.
            Temp = dataset['tmpsfc']       # at surface, K
            Hum  = dataset['rh2m']         # at 2 meters, %
            U    = dataset['ugrd10m']      # at 10 meters, m/s
            V    = dataset['vgrd10m']      # at 10 meters, m/s

            if atm_name == nam_1hr:        # or atm_name == hires_fv3
                Rain = dataset['apcpsfc']  # total precip, kg/m2, need to take diffs of times. 
            else:
                Rain = dataset['pratesfc'] # at surface, kg/m2/s

            Swd  = dataset['dswrfsfc']     # downward short-wave at surface, W/m2
            Swu  = dataset['uswrfsfc']     # upward short-wave at surface, W/m2
            Lwd  = dataset['dlwrfsfc']     # downward long-wave at surface, W/m2
            Lwu  = dataset['ulwrfsfc']     # upward long-wave at surface, W/m2

            # here we find the indices of the data we want for LV1
            Ln0    = ln[:]
            Lt0    = lt[:]
            
            t   = time[:].data 
            t_ref = PFM['modtime0']
            t_rom = get_roms_times(yyyymmdd,t,t_ref) # now we are in days
            del t
            tp  = t_rom-t_rom[0]
            del t_rom
            #print(type(tp))
            tp.astype(np.float64)
            tpmax = np.float64(PFM['forecast_days'])
            itm = np.where( tp <= tpmax ) 

           # print('\n')
           # print(tpmax)
           # print(tp)
           # print('\n')
           # print(itm)
           # print(itm[0][0],itm[0][-1])
           # print(tp[itm[0][0]],tp[itm[0][-1]])
            #print(tp[itm[0][:]])
            itm = itm[0][:]

            if atm_name == gfs or atm_name==gfs_1hr:
                Ln0 = Ln0-360.0
                iln    = np.where( (Ln0>=ln_min-0.75)*(Ln0<=ln_max+0.75) ) # the lon indices where we want data
                ilt    = np.where( (Lt0>=lt_min-0.75)*(Lt0<=lt_max+0.75) ) # the lat indices where we want data
            else:
                iln    = np.where( (Ln0>=ln_min)*(Ln0<=ln_max) ) # the lon indices where we want data
                ilt    = np.where( (Lt0>=lt_min)*(Lt0<=lt_max) ) # the lat indices where we want data

            iln = iln[0][:]
            ilt = ilt[0][:]
            #print(iln)
            #print(ilt)
            #print(Ln0)

            t   = time[ itm ].data 

            t_ref = PFM['modtime0']
            t_rom = get_roms_times(yyyymmdd,t,t_ref) # now we are in days

            lat = lt[ ilt ].data
            lon = ln[ iln ].data

            if atm_name == gfs or atm_name == gfs_1hr:
                lon = lon - 360.0

            pres = Pres[itm,ilt,iln ].data
            temp = Temp[itm,ilt,iln].data
            hum  = Hum[itm,ilt,iln].data
            u    = U[itm,ilt,iln].data
            v    = V[itm,ilt,iln].data
            rain = Rain[itm,ilt,iln].data
            swd  = Swd[itm,ilt,iln].data
            swu  = Swu[itm,ilt,iln].data
            lwd  = Lwd[itm,ilt,iln].data
            lwu  = Lwu[itm,ilt,iln].data

            print('long wave down radiation [0:2,0:3,0:3]:')
            print(lwd[0:2,0:3,0:3])
            print('9e20 is bad!')
            # for gfs, the 1st timestamp for radiation are all bad, so we replace
            # DON'T KNOW WHY!!!????
            if atm_name == gfs or atm_name == gfs_1hr:
                print('replacing t=0 radiation w/ t=1')
                swd[0,:,:] = swd[1,:,:]
                swu[0,:,:] = swd[1,:,:]
                lwd[0,:,:] = lwd[1,:,:]
                lwu[0,:,:] = lwu[1,:,:]
            # return the roms times past tref in days
            #t_ref = datetime(1970,1,1)

            # I think everything is an np.ndarray ?
            dataset.close()


        # set up dict and fill in
        ATM = dict()
        ATM['vinfo'] = dict()

        # this is the complete list of variables that need to be in the netcdf file
        vlist = ['lon','lat','ocean_time','ocean_time_ref','lwrad','lwrad_down','swrad','rain','Tair','Pair','Qair','Uwind','Vwind','tair_time','pair_time','qair_time','wind_time','rain_time','srf_time','lrf_time']
        for aa in vlist:
            ATM['vinfo'][aa] = dict()

        ATM['lon']=lon
        ATM['lat']=lat

        ATM['ocean_time'] = t_rom
        ATM['pair_time'] = t_rom
        ATM['tair_time'] = t_rom
        ATM['qair_time'] = t_rom
        ATM['srf_time'] = t_rom
        ATM['lrf_time'] = t_rom
        ATM['wind_time'] = t_rom
        ATM['rain_time'] = t_rom

        ATM['ocean_time_ref'] = t_ref
        ATM['lwrad'] = lwd-lwu
        ATM['lrf_time'] = t_rom
        ATM['lwrad_down'] = lwd
        ATM['swrad'] = swd-swu

#        print(np.max(ATM['swrad']))
#        print(np.min(ATM['swrad']))
#        print(np.max(ATM['lwrad']))
#        print(np.min(ATM['lwrad']))
#        print(np.max(ATM['lwrad_down']))
#        print(np.min(ATM['lwrad_down']))
#        print(np.unravel_index(np.argmax(ATM['lwrad_down'], axis=None), ATM['lwrad_down'].shape))

        ATM['rain'] = rain          # kg/m2/s
        ATM['Tair'] = temp - 273.15 # convert from K to C
        ATM['Pair'] = 0.01 * pres   # convert from Pa to db
        ATM['Qair'] = hum
        ATM['Uwind'] = u
        ATM['Vwind'] = v

#        print(lwu[0,0:3,0:3])
#        print(ATM['lwrad'][0,0:3,0:3])
#        print(ATM['lwrad_down'][0,0:3,0:3])
#        print(ATM['swrad'][0,0:3,0:3])
#        print(ATM['Tair'][0,0:3,0:3])
#        print(ATM['Uwind'][0,0:3,0:3])


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

    with open(fname_out,'wb') as fp:
        pickle.dump(ATM,fp)
        print('\nATM dict saved with pickle.')

def load_atm():
    PFM        = get_PFM_info()   
    fname_atm  = PFM['lv1_forc_dir'] + '/' + PFM['atm_tmp_pckl_file']

    with open(fname_atm,'rb') as fp:
        ATM = pickle.load(fp)

    return ATM


def get_atm_data_on_roms_grid(lv):
    # this function takes the ATM data, in a dict, and the roms grid, as a dict
    # and returns the ATM data but on the roms grid. It returns atm2
    # the wind directions in atm2 are rotated to be in ROMS xi,eta directions.
    
    PFM=get_PFM_info()
    fname_atm  = PFM['lv1_forc_dir'] + '/' + PFM['atm_tmp_pckl_file']
    with open(fname_atm,'rb') as fp:
        ATM = pickle.load(fp)

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


     
    field_names = ['lwrad', 'lwrad_down', 'swrad', 'rain', 'Tair', 'Pair', 'Qair', 'Uwind', 'Vwind']
    # these are the 2d fields that need to be interpreted onto the roms grid
    # dimensions of all fields are [ntime,nlat,nlon]
    
    t = ATM['ocean_time']
    nt = len(t)
    lon = ATM['lon']
    lat = ATM['lat']
    Lt_r = RMG['lat_rho']
    Ln_r = RMG['lon_rho']
    nlt,nln = np.shape(Lt_r)

    # set the flag to determine if we have created the interpolant yet.
    got_F = 0 
    
    # this is the complete list of variables that need to be in the netcdf file
    vlist = ['lon','lat','ocean_time','ocean_time_ref','lwrad','lwrad_down','swrad','rain','Tair','Pair','Qair','Uwind','Vwind','tair_time','pair_time','qair_time','wind_time','rain_time','srf_time','lrf_time']

    # copy vinfo from ATM to atm2
    atm2 = dict()
    atm2['vinfo'] = dict()
    for aa in vlist:
        atm2['vinfo'][aa] = ATM['vinfo'][aa]

    # copy the right coordinates too
    vlist2 = ['ocean_time','tair_time','pair_time','qair_time','wind_time','rain_time','srf_time','lrf_time']
    for aa in vlist2:
        atm2[aa] = ATM[aa]

    # the lat lons are from the roms grid
    atm2['lat'] = RMG['lat_rho']
    atm2['lon'] = RMG['lon_rho']

    # these two are useful later
    atm2['ocean_time_ref'] = ATM['ocean_time_ref']

    # this for loop puts the ATM fields onto the ROMS grid
    for a in field_names:
        #print(a)
        f1 = ATM[a]
        frm2 = np.zeros( (nt,nlt,nln) ) # need to initialize the dict, or there are problems
        atm2[a] = frm2

        for b in range(nt):
            f2 = np.squeeze( f1[b,:,:] )

            if got_F == 0:
                F = RegularGridInterpolator((lat,lon),f2)
                got_F = 1
            else:                
                setattr(F,'values',f2)

            froms = F((Lt_r,Ln_r),method='linear')
            atm2[a][b,:,:] = froms
        

    # atm2 is now has velocities on the roms grid. but we need to rotate the winds from N-S, E-W to ROMS (xi,eta)
    angr = RMG['angle']
    cosang = np.cos(angr)
    sinang = np.sin(angr)
    Cosang = np.tile(cosang,(nt,1,1))
    Sinang = np.tile(sinang,(nt,1,1))
    ur = Cosang * atm2['Uwind'] + Sinang * atm2['Vwind']
    vr = Cosang * atm2['Vwind'] - Sinang * atm2['Uwind']
 
    atm2['Uwind'] = ur
    atm2['Vwind'] = vr
        
    with open(fname_out,'wb') as fp:
        pickle.dump(atm2,fp)
        print('\nATM on roms grid dict saved with pickle.')

    #return atm2

def atm_roms_dict_to_netcdf(lv):

    PFM=get_PFM_info()

    if lv == '1':
        fname_in  = PFM['lv1_forc_dir'] + '/' + PFM['atm_tmp_LV1_pckl_file']
        fname_out = PFM['lv1_forc_dir'] + '/' + PFM['lv1_atm_file'] # LV1 atm forcing filename
    elif lv == '2':
        fname_in  = PFM['lv2_forc_dir'] + '/' + PFM['atm_tmp_LV2_pckl_file']
        fname_out = PFM['lv2_forc_dir'] + '/' + PFM['lv2_atm_file'] 
    elif lv == '3':
        fname_in  = PFM['lv3_forc_dir'] + '/' + PFM['atm_tmp_LV3_pckl_file']
        fname_out = PFM['lv3_forc_dir'] + '/' + PFM['lv3_atm_file'] 
    else:
        fname_in  = PFM['lv4_forc_dir'] + '/' + PFM['atm_tmp_LV4_pckl_file']
        fname_out = PFM['lv4_forc_dir'] + '/' + PFM['lv4_atm_file'] 
    
    with open(fname_in,'rb') as fp:
        ATM_R = pickle.load(fp)

    print('file_out is:')
    print(fname_out)

    ds = xr.Dataset(
        data_vars = dict(
            Tair       = (["tair_time","er","xr"],ATM_R['Tair'],ATM_R['vinfo']['Tair']),
            Pair       = (["pair_time","er","xr"],ATM_R['Pair'],ATM_R['vinfo']['Pair']),
            Qair       = (["qair_time","er","xr"],ATM_R['Qair'],ATM_R['vinfo']['Qair']),
            Uwind      = (["wind_time","er","xr"],ATM_R['Uwind'],ATM_R['vinfo']['Uwind']),
            Vwind      = (["wind_time","er","xr"],ATM_R['Vwind'],ATM_R['vinfo']['Vwind']),
            rain       = (["rain_time","er","xr"],ATM_R['rain'],ATM_R['vinfo']['rain']),
            swrad      = (["srf_time","er","xr"],ATM_R['swrad'],ATM_R['vinfo']['swrad']),
            lwrad      = (["lrf_time","er","xr"],ATM_R['lwrad'],ATM_R['vinfo']['lwrad']),
            lwrad_down = (["lrf_time","er","xr"],ATM_R['lwrad_down'],ATM_R['vinfo']['lwrad_down']),
        ),
        coords=dict(
            lat =(["er","xr"],ATM_R['lat'], ATM_R['vinfo']['lat']),
            lon =(["er","xr"],ATM_R['lon'], ATM_R['vinfo']['lon']),
            ocean_time = (["time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            tair_time = (["tair_time"],ATM_R['ocean_time'], ATM_R['vinfo']['tair_time']),
            pair_time = (["pair_time"],ATM_R['ocean_time'], ATM_R['vinfo']['pair_time']),
            qair_time = (["qair_time"],ATM_R['ocean_time'], ATM_R['vinfo']['qair_time']),
            wind_time = (["wind_time"],ATM_R['ocean_time'], ATM_R['vinfo']['wind_time']),
            rain_time = (["rain_time"],ATM_R['ocean_time'], ATM_R['vinfo']['rain_time']),
            srf_time = (["srf_time"],ATM_R['ocean_time'], ATM_R['vinfo']['srf_time']),
            lrf_time = (["lrf_time"],ATM_R['ocean_time'], ATM_R['vinfo']['lrf_time']),
        ),
        attrs={'type':'atmospheric forcing file fields for surface fluxes',
            'time info':'ocean time is from '+ ATM_R['ocean_time_ref'].strftime("%Y/%m/%d %H:%M:%S") },
        )
    # print(ds)

    ds.to_netcdf(fname_out)


if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])

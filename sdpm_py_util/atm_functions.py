# library of atm functions
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc


def get_atm_data_as_dict(yyyymmdd,run_type,atm_mod,get_method):
    from datetime import datetime
    from pydap.client import open_url
    import pygrib

    
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
        t2 = t-t[0] # an ndarray of days, t is from atm import
        t3 = d1 + t2 * timedelta(days=1)
        # t3 looks good and is the correct time stamps of the forecast.
        # But for ROMS we need ocean_time which is relative to 1970,1,1. 
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
        nam_nest = 'https://nomads.ncep.noaa.gov/dods/nam/nam' + yyyymmdd + '/nam_conusnest_00z'

        # nam_1hr is at 1 hr resolution, for 1.5 days
        # 0.11 deg horizontal resolution
        nam_1hr = 'https://nomads.ncep.noaa.gov/dods/nam/nam' + yyyymmdd + '/nam1hr_00z'

        # hires_fv3 is at 1 hr resolution, for 2.5 days
        # 0.05 deg horizontal resolution
        # this one doesn't have short and long wave. DARN. DONT USE!!!!
        # hires_fv3 = 'https://nomads.ncep.noaa.gov/dods/hiresw/hiresw' + yyyymmdd + '/hiresw_conusfv3_00z'

        # hrrr is at 1 hr resolution, for 2 days, for 00z forecast
        # 0.03 deg horizontal resolution 
        # this is clearly the highest resolution product
        hrrr = 'https://nomads.ncep.noaa.gov/dods/hrrr/hrrr' + yyyymmdd + '/hrrr_sfc.t00z'

        # gfs is at 1 hr resolution, for 5 days
        # 0.25 deg horizontal resolution
        gfs = 'https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs' + yyyymmdd + '/gfs_0p25_1hr_00z'

        if atm_mod == 'nam_nest':
            atm_name = nam_nest
        if atm_mod == 'nam_1hr':
            atm_name = nam_1hr
        #if atm_mod == 'hires_fv3':
        #    atm_name = hires_fv3
        if atm_mod == 'hrrr':
            atm_name = hrrr
        if atm_mod == 'gfs':
            atm_name = gfs

        # define the box to get data in
        ln_min = -124.5
        ln_max = -115
        lt_min = 28
        lt_max = 37

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
            if atm_name == nam_nest or atm_name == hrrr:
                Rain = dataset['pratesfc'] # at surface, kg/m2/s
            if atm_name == nam_1hr:        # or atm_name == hires_fv3
                Rain = dataset['apcpsfc']  # total precip, kg/m2, need to take diffs of times.    
            Swd  = dataset['dswrfsfc']     # downward short-wave at surface, W/m2
            Swu  = dataset['uswrfsfc']     # upward short-wave at surface, W/m2
            Lwd  = dataset['dlwrfsfc']     # downward long-wave at surface, W/m2
            Lwu  = dataset['ulwrfsfc']     # upward long-wave at surface, W/m2

            # here we find the indices of the data we want for LV1
            Ln0    = ln[:]
            Lt0    = lt[:]
            iln    = np.where( (Ln0>=ln_min)*(Ln0<=ln_max) ) # the lon indices where we want data
            ilt    = np.where( (Lt0>=lt_min)*(Lt0<=lt_max) ) # the lat indices where we want data

        
            # now time to get the numbers that we want. I get them as arrays?
            pres2 = Pres[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]] # indexing looks bad but works
            t     = pres2.time[:].data
            # return the roms times past tref in days
            t_ref = datetime(1970,1,1)
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
            if atm_name == nam_nest or atm_name == hrrr:
                Rain = dataset['pratesfc'] # at surface, kg/m2/s
            if atm_name == nam_1hr:        # or atm_name == hires_fv3
                Rain = dataset['apcpsfc']  # total precip, kg/m2, need to take diffs of times.    
            Swd  = dataset['dswrfsfc']     # downward short-wave at surface, W/m2
            Swu  = dataset['uswrfsfc']     # upward short-wave at surface, W/m2
            Lwd  = dataset['dlwrfsfc']     # downward long-wave at surface, W/m2
            Lwu  = dataset['ulwrfsfc']     # upward long-wave at surface, W/m2

            # here we find the indices of the data we want for LV1
            Ln0    = ln[:]
            Lt0    = lt[:]
            iln    = np.where( (Ln0>=ln_min)*(Ln0<=ln_max) ) # the lon indices where we want data
            ilt    = np.where( (Lt0>=lt_min)*(Lt0<=lt_max) ) # the lat indices where we want data

            lat = lt[ ilt[0][0]:ilt[0][-1] ].data
            lon = ln[ iln[0][0]:iln[0][-1] ].data
            t   = time[:].data
            pres = Pres[:,ilt[0][0]:ilt[0][-1] , iln[0][0]:iln[0][-1] ].data
            # we will get the other data directly
            temp = Temp[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            hum = Hum[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            u = U[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            v = V[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            rain = Rain[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            swd = Swd[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            swu = Swu[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            lwd = Lwd[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            lwu = Lwu[:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            # return the roms times past tref in days
            t_ref = datetime(1970,1,1)
            t_rom = get_roms_times(yyyymmdd,t,t_ref)

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
        ATM['lrf_time']
        ATM['lwrad_down'] = lwd
        ATM['swrad'] = swd-swu
        ATM['rain'] = rain          # kg/m2/s
        ATM['Tair'] = temp - 273.15 # convert from K to C
        ATM['Pair'] = 0.01 * pres   # convert from Pa to db
        ATM['Qair'] = hum
        ATM['Uwind'] = u
        ATM['Vwind'] = v

        # put the units in atm...
        ATM['vinfo']['lon'] = {'long_name':'longitude',
                        'units':'degrees_east'}
        ATM['vinfo']['lat'] = {'long_name':'latitude',
                        'units':'degrees_north'}
        ATM['vinfo']['ocean_time'] = {'long_name':'atmospheric forcing time',
                            'units':'days since tref'}
        ATM['vinfo']['rain_time'] = {'long_name':'atmospheric rain forcing time',
                            'units':'days since tref'}
        ATM['vinfo']['wind_time'] = {'long_name':'atmospheric wind forcing time',
                            'units':'days since tref'}
        ATM['vinfo']['tair_time'] = {'long_name':'atmospheric temp forcing time',
                            'units':'days since tref'}
        ATM['vinfo']['pair_time'] = {'long_name':'atmospheric pressure forcing time',
                            'units':'days since tref'}
        ATM['vinfo']['qair_time'] = {'long_name':'atmospheric humidity forcing time',
                            'units':'days since tref'}
        ATM['vinfo']['srf_time'] = {'long_name':'atmospheric short wave radiation forcing time',
                            'units':'days since tref'}
        ATM['vinfo']['lrf_time'] = {'long_name':'atmospheric long wave radiation forcing time',
                            'units':'days since tref'}
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

    return ATM
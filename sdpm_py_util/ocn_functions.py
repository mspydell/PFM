# the ocean functions will be here

from datetime import datetime

import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator


def get_ocn_data_as_dict(yyyymmdd,run_type,ocn_mod,get_method):
    from datetime import datetime
    from pydap.client import open_url
    import pygrib

    
    # this function will returm OCN, a dict of all ocean fields ROMS requires
    # keys will be the ROMS .nc variable names.
    # they will be on the ocn grid (not roms grid)
    # the forecast data will start at yyyymmdd and 1200z. All times of the forecast will
    # be returned. ocn_mod is the type of OCN models used, one of:
    # 'hycom', 'wcofs', etc
    # get_method, is the type of method used, either 'open_dap' or 'grib_download'

    # the code in here goes from the start date to all forecast dates

    def get_roms_times_from_hycom(fore_date,t,t_ref):
        # this funtion returns times past t_ref in days
        # consistent with how ROMS likes it
        from datetime import timedelta

        d1=fore_date
        t2 = t # an ndarray of days, t is from atm import
        t3 = d1 + t2 * timedelta(days=1/24)
        print('ocn forecast data is being grabbed for:')
        print(t3)
        # t3 looks good and is the correct time stamps of the forecast.
        # But for ROMS we need ocean_time which is relative to 1970,1,1. 
        # in seconds. So...
        tr = t3 - t_ref
        tr_days = tr.astype("timedelta64[ms]").astype(float) / 1000 / 3600 / 24
        # tr_sec is now an ndarray of days past the reference day
        return tr_days

    if get_method == 'open_dap_pydap' or get_method == 'open_dap_nc' and run_type == 'forecast':
        # with this method the data is not downloaded directly, initially
        # and the data is rectilinear, lon and lat are both vectors

        # the hycom forecast is 7.5 days long and 
        # is at 3 hr resolution, we will get all of the data
        # 0.08 deg horizontal resolution
        # Note, hycom forecasts are start from 1200Z !!!! 
        hycom = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/runs/GLBy0.08_930_FMRC_RUN_' + yyyymmdd[0:4] + '-' + yyyymmdd[4:6] + '-' + yyyymmdd[6:8] + 'T12:00:00Z'

        
        if ocn_mod == 'hycom':
            ocn_name = hycom
            it1 = 0 # hard wiring here to get 2.5 days of data
            it2 = 20 # this is 2.5 days at 3 hrs
            # it2 = 60 # for the entire 7.5 day long forecast
        
        # define the box to get data in
        ln_min = -124.5 + 360
        ln_max = -115 + 360
        lt_min = 28
        lt_max = 37

        if get_method == 'open_dap_pydap':
            # open a connection to the opendap server. This could be made more robust? 
            # by trying repeatedly?
            # open_url is sometimes slow, and this block of code can be fast (1s), med (6s), or slow (>15s)
            dataset = open_url(ocn_name)

            time = dataset['time']         # ???
            ln   = dataset['lon']          # deg
            lt   = dataset['lat']          # deg
            Eta  = dataset['surface_el']     # surface elevations, m
            Temp = dataset['water_temp']       # at surface, C
            U    = dataset['water_u']      # water E/W velocity, m/s
            V    = dataset['water_v']      # water N/S velocity, m/s
            Sal  = dataset['salinity']     # upward long-wave at surface, W/m2
            Z    = dataset['depth']

            # here we find the indices of the data we want for LV1
            Ln0    = ln[:].data
            Lt0    = lt[:].data
            iln    = np.where( (Ln0>=ln_min)*(Ln0<=ln_max) ) # the lon indices where we want data
            ilt    = np.where( (Lt0>=lt_min)*(Lt0<=lt_max) ) # the lat indices where we want data

            # return the roms times past tref in days
            t=time.data[it1:it2] # this is hrs past t0
            t0 = time.attributes['units'] # this the hycom reference time
            t0 = t0[12:31] # now we get just the date and 12:00
            t0 = datetime.fromisoformat(t0) # put it in datetime format
            t_ref = datetime(1970,1,1)
            t_rom = get_roms_times_from_hycom(t0,t,t_ref)

            lon   = ln[iln[0][0]:iln[0][-1]].data
            lat   = lt[ilt[0][0]:ilt[0][-1]].data
            z     = z[:].data

            # we will get the other data directly
            eta  =  Eta.array[it1:it2,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            u    =    U.array[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            v    =    V.array[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            temp = Temp.array[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            sal  =  Sal.array[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data

        
        if get_method == 'open_dap_nc':
            # open a connection to the opendap server. This could be made more robust? 
            # by trying repeatedly?
            # open_url is sometimes slow, and this block of code can be fast (1s), med (6s), or slow (>15s)
            
            dataset = nc.Dataset(ocn_name)
            #ds2 = xr.open_dataset(ocn_name,use_cftime=False, decode_coords=False, decode_times=False)


            time = dataset['time']         # ???
            ln   = dataset['lon']          # deg
            lt   = dataset['lat']          # deg
            Eta  = dataset['surf_el']     # surface elevations, m
            Temp = dataset['water_temp']       # 3d temp, C
            U    = dataset['water_u']      # water E/W velocity, m/s
            V    = dataset['water_v']      # water N/S velocity, m/s
            Sal  = dataset['salinity']     # salinity psu
            Z    = dataset['depth']

            # here we find the indices of the data we want for LV1
            Ln0    = ln[:]
            Lt0    = lt[:]
            iln    = np.where( (Ln0>=ln_min)*(Ln0<=ln_max) ) # the lon indices where we want data
            ilt    = np.where( (Lt0>=lt_min)*(Lt0<=lt_max) ) # the lat indices where we want data

            lat = lt[ ilt[0][0]:ilt[0][-1] ].data
            lon = ln[ iln[0][0]:iln[0][-1] ].data
            z   =  Z[:].data
            t   = time[it1:it2].data
            eta = Eta[it1:it2,ilt[0][0]:ilt[0][-1] , iln[0][0]:iln[0][-1] ].data
            # we will get the other data directly
            temp = Temp[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            sal  = Sal[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            u = U[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
            v = V[it1:it2,:,ilt[0][0]:ilt[0][-1],iln[0][0]:iln[0][-1]].data
                # return the roms times past tref in days
            t0 = time.units # this is the hycom reference time
            t0 = t0[12:31] # now we get just the date and 12:00
            t0 = datetime.fromisoformat(t0) # put it in datetime format
            t_ref = datetime(1970,1,1)
            t_rom = get_roms_times_from_hycom(t0,t,t_ref)

            # I think everything is an np.ndarray ?
            dataset.close()


        # set up dict and fill in
        OCN = dict()
        OCN['vinfo'] = dict()

        # this is the complete list of variables that need to be in the netcdf file
        vlist = ['lon','lat','ocean_time','surf_el','water_u','water_v','temp','sal']

        for aa in vlist:
            OCN['vinfo'][aa] = dict()

        OCN['lon']=lon - 360 # make the lons negative consistent with most 
        OCN['lat']=lat

        OCN['ocean_time'] = t_rom
        OCN['ocean_time_ref'] = t_ref
        
        OCN['u'] = u
        OCN['v'] = v
        OCN['temp'] = temp
        OCN['sal'] = sal
        OCN['surf_el'] = eta
        OCN['depth'] = z
    
        # put the units in OCN...
        OCN['vinfo']['lon'] = {'long_name':'longitude',
                        'units':'degrees_east'}
        OCN['vinfo']['lat'] = {'long_name':'latitude',
                        'units':'degrees_north'}
        OCN['vinfo']['ocean_time'] = {'long_name':'OCN forcing time',
                            'units':'days since tref'}
        OCN['vinfo']['depth'] = {'long_name':'ocean depth',
                            'units':'m'}
        OCN['vinfo']['temp'] = {'long_name':'ocean temperature',
                        'units':'degrees C',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['sal'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['u'] = {'long_name':'ocean east west velocity',
                        'units':'m/s',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['v'] = {'long_name':'ocean north south velocity',
                        'units':'m/s',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['surf_el'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'lat,lon',
                        'time':'ocean_time'}
    return OCN


def hycom_to_roms_latlon(HY,RMG):
    # HYcom and RoMsGrid come in as dicts with ROMS variable names    
    # The output of this, HYrm, is a dict with 
    # hycom fields on roms horizontal grid points
    # but hycom z levels.
    # velocity will be on both (lat_u,lon_u)
    # and (lat_v,lon_v).
    
    # set up the interpolator now and pass to function
    

    Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['surf_el'])
    
    # the names of the variables that need to go on the ROMS grid
    vnames = ['surf_el', 'temp', 'salt', 'u', 'v']
    lnhy = HY['lon']
    lthy = HY['lat']
    NR,NC = np.shape(RMG['lon_rho'])
    NZ = len(HY['depth'])
    NT = len(HY['ocean_time'])

    HYrm = dict()
    HYrm['surf_el'] = np.zeros((NT,NR, NC))
    HYrm['sal'] = np.zeros((NT,NZ, NR, NC))
    HYrm['temp'] = np.zeros((NT,NZ, NR, NC))
    HYrm['u_on_u'] = np.zeros((NT,NZ, NR, NC-1))
    HYrm['v_on_u'] = np.zeros((NT,NZ, NR, NC-1))
    HYrm['u_on_v'] = np.zeros((NT,NZ, NR-1, NC))
    HYrm['v_on_v'] = np.zeros((NT,NZ, NR-1, NC))
    HYrm['lat_rho'] = RMG['lat_rho']
    HYrm['lon_rho'] = RMG['lat_rho']
    HYrm['lat_u'] = RMG['lat_u']
    HYrm['lon_u'] = RMG['lon_u']
    HYrm['lat_v'] = RMG['lat_v']
    HYrm['lon_v'] = RMG['lon_v']
    HYrm['depth'] = HY['depth'] # depths are from hycom


    for aa in vnames:
        zhy  = HY[aa]
        if aa=='zeta':
            HYrm[aa] = interp_hycom_to_roms(lnhy,lthy,zhy,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],rf,Imr,Fz)            
        elif aa=='temp' or aa=='salt':
            for bb in range(NZ):
                zhy2 = zhy[bb,:,:]
                HYrm[aa][bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],rf,Imr,Fz)
        elif aa=='u':
            for bb in range(NZ):
                zhy2= zhy[bb,:,:]
                HYrm['u_on_u'][bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],rf,Imru,Fz)
                HYrm['u_on_v'][bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],rf,Imrv,Fz)
        elif aa=='v':
            for bb in range(NZ):
                zhy2= zhy[bb,:,:]
                HYrm['v_on_u'][bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],rf,Imru,Fz)
                HYrm['v_on_v'][bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],rf,Imrv,Fz)
 
    return HYrm


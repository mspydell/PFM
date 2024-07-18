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
from scipy.interpolate import interp1d
import seawater


import util_functions as utlfuns 
from util_functions import s_coordinate_4



def get_ocn_data_as_dict(yyyymmdd,run_type,ocn_mod,get_method):
    from datetime import datetime
    from pydap.client import open_url
#    import pygrib

    
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

    if get_method == 'open_dap_pydap' or get_method == 'open_dap_nc' or get_method == 'ncks' and run_type == 'forecast':
        # with this method the data is not downloaded directly, initially
        # and the data is rectilinear, lon and lat are both vectors

        # the hycom forecast is 7.5 days long and 
        # is at 3 hr resolution, we will get all of the data
        # 0.08 deg horizontal resolution
        # Note, hycom forecasts are start from 1200Z !!!! 
        
        yyyy = yyyymmdd[0:4]
        mm = yyyymmdd[4:6]
        dd = yyyymmdd[6:8]

        hycom = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/runs/GLBy0.08_930_FMRC_RUN_' + yyyy + '-' + mm + '-' + dd + 'T12:00:00Z'

        # define the box to get data in
        ln_min = -124.5 + 360
        ln_max = -115 + 360
        lt_min = 28
        lt_max = 37


        if ocn_mod == 'hycom':
            ocn_name = hycom
            it1 = 0 # hard wiring here to get 2.5 days of data
            it2 = 20 # this is 2.5 days at 3 hrs
            # it2 = 60 # for the entire 7.5 day long forecast
        
 
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

        if get_method == 'ncks':

            west =  -124.5 + 360
            east = -115 + 360
            south = 28
            north = 37

            # time limits
            dstr0 = yyyy + '-' + mm + '-' + dd + 'T12:00'
            dstr1 = yyyy + '-' + mm + '-' + str( int(dd) + 3 ) + 'T00:00'
            #dstr0 = dlist['dt0'].strftime('%Y-%m-%dT00:00') 
            #dstr1 = dlist['dt1'].strftime('%Y-%m-%dT00:00')
            # use subprocess.call() to execute the ncks command
            vstr = 'surf_el,water_temp,salinity,water_u,water_v,depth'

            # parker url: https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest

            url='https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/runs' 
            url2 = 'GLBy0.08_930_FMRC_RUN_' + yyyy + '-' + mm + '-' + dd + 'T12:00:00Z' 
            url3 = url + url2
            cmd_list = ['ncks',
                '-d', 'time,'+dstr0+','+dstr1,
                '-d', 'lon,'+str(west)+','+str(east),
                '-d', 'lat,'+str(south)+','+str(north),
                '-v', vstr,
                url3 ,
                '-4', '-O', full_fn_out]
            # old working command list
            #cmd_list = ['ncks',
            #    '-d', 'time,'+dstr0+','+dstr1,
            #    '-d', 'lon,'+str(west)+','+str(east),
            #    '-d', 'lat,'+str(south)+','+str(north),
            #    '-v', vstr,
            #    'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0',
            #    '-4', '-O', full_fn_out]

            print(cmd_list)

            # run ncks
            tt0 = time.time()
            ret1 = subprocess.call(cmd_list)
            #ret1 = 1
            print('Time to get full file using ncks = %0.2f sec' % (time.time()-tt0))
            print('Return code = ' + str(ret1) + ' (0=success, 1=skipped ncks)')


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
        OCN['salt'] = sal
        OCN['zeta'] = eta
        OCN['depth'] = z
    
        # put the units in OCN...
        OCN['vinfo']['lon'] = {'long_name':'longitude',
                        'units':'degrees_east'}
        OCN['vinfo']['lat'] = {'long_name':'latitude',
                        'units':'degrees_north'}
        OCN['vinfo']['ocean_time'] = {'long_name':'time since initialization',
                        'units':'days',
                        'coordinates':'temp_time',
                        'field':'ocean_time, scalar, series'}
        OCN['vinfo']['ocean_time_ref'] = {'long_name': 'the reference date tref (initialization time)'}
        OCN['vinfo']['depth'] = {'long_name':'ocean depth',
                            'units':'m'}
        OCN['vinfo']['temp'] = {'long_name':'ocean temperature',
                        'units':'degrees C',
                        'coordinates':'z,lat,lon',
                        'time':'ocean_time'}
        OCN['vinfo']['salt'] = {'long_name':'ocean salinity',
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
        OCN['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'lat,lon',
                        'time':'ocean_time'}
    return OCN


def earth_rad(lat_deg):
    """
    Calculate the Earth radius (m) at a latitude
    (from http://en.wikipedia.org/wiki/Earth_radius) for oblate spheroid

    INPUT: latitude in degrees

    OUTPUT: Earth radius (m) at that latitute
    """
    a = 6378.137 * 1000; # equatorial radius (m)
    b = 6356.7523 * 1000; # polar radius (m)
    cl = np.cos(np.pi*lat_deg/180)
    sl = np.sin(np.pi*lat_deg/180)
    RE = np.sqrt(((a*a*cl)**2 + (b*b*sl)**2) / ((a*cl)**2 + (b*sl)**2))
    return RE

def ll2xy(lon, lat, lon0, lat0):
    """
    This converts lon, lat into meters relative to lon0, lat0.
    It should work for lon, lat scalars or arrays.
    NOTE: lat and lon are in degrees!!
    """
    R = earth_rad(lat0)
    clat = np.cos(np.pi*lat0/180)
    x = R * clat * np.pi * (lon - lon0) / 180
    y = R * np.pi * (lat - lat0) / 180
    return x, y


def extrap_nearest_to_masked(X, Y, fld, fld0=0):
    """
    INPUT: fld is a 2D array (np.ndarray or np.ma.MaskedArray) on spatial grid X, Y
    OUTPUT: a numpy array of the same size with no mask
    and no missing values.        
    If input is a masked array:        
        * If it is ALL masked then return an array filled with fld0.         
        * If it is PARTLY masked use nearest neighbor interpolation to
        fill missing values, and then return data.        
        * If it is all unmasked then return the data.    
    If input is not a masked array:        
        * Return the array.    
    """
    from scipy.spatial import cKDTree
    
    # first make sure nans are masked
    if np.ma.is_masked(fld) == False:
        fld = np.ma.masked_where(np.isnan(fld), fld)
        
    if fld.all() is np.ma.masked:
        #print('  filling with ' + str(fld0))
        fldf = fld0 * np.ones(fld.data.shape)
        fldd = fldf.data
        checknan(fldd)
        return fldd
    else:
        # do the extrapolation using nearest neighbor
        fldf = fld.copy() # initialize the "filled" field
        xyorig = np.array((X[~fld.mask],Y[~fld.mask])).T
        xynew = np.array((X[fld.mask],Y[fld.mask])).T
        a = cKDTree(xyorig).query(xynew)
        aa = a[1]
        fldf[fld.mask] = fld[~fld.mask][aa]
        fldd = fldf.data
        checknan(fldd)
        return fldd

def checknan(fld):
    """
    A utility function that issues a working if there are nans in fld.
    """
    if np.isnan(fld).sum() > 0:
        print('WARNING: nans in data field')    


def check_all_nans(z):
    # this assumes z is NOT a masked array...

    nfin = np.count_nonzero(~np.isnan(z))
    if nfin == 0:
        allnan = True
    else:
        allnan = False

    return allnan


def interp_hycom_to_roms(ln_h,lt_h,zz_h,Ln_r,Lt_r,msk_r,Fz):
    # ln_h and lt_h are hycom lon and lat, vectors
    # zz_h is the hycom field that is getting interpolated
    # Ln_r and Lt_r are the roms lon and lat, matrices
    # msk_r is the roms mask, to NaN values
    # mf is the refinement of the hycom  grid before going 
    # nearest neighbor interpolating to ROMS

    # 1st check if field is all NaNs
    allnan = check_all_nans(zz_h)

    if allnan == True:
        # if all NaNs, just return all NaNs, no need to interpolate
        zz_r = np.nan * Ln_r
    else:
        Ln_h, Lt_h = np.meshgrid(ln_h, lt_h, indexing='xy')
        X, Y = ll2xy(Ln_h, Lt_h, ln_h.mean(), lt_h.mean())
        # fill in the hycom NaNs with nearest neighbor values
        # ie. land is now filled with numbers
        zz_hf =  extrap_nearest_to_masked(X, Y, zz_h) 

        # interpolate the filled hycom values to the refined grid
        #Fz = RegularGridInterpolator((lt_h, ln_h), zz_hf)
        # change the z values of the interpolator here
        setattr(Fz,'values',zz_hf)

        lininterp=1
        if lininterp==1:
            zz_rf = Fz((Lt_r,Ln_r),method='linear')
        else:
            # refine the hycom grid
            #lnhi = np.linspace(ln_h[0],ln_h[-1],mf*len(ln_h))
            #lthi = np.linspace(lt_h[0],lt_h[-1],mf*len(lt_h))
            #Lnhi, Lthi = np.meshgrid(lnhi,lthi, indexing='xy')
            #zz_hfi=Fz((Lthi,Lnhi))

            # now nearest neighbor interpolate to the ROMS grid
            #XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
            #XYr = np.array((Ln_r.flatten(), Lt_r.flatten())).T
            # nearest neighbor interpolation from XYin to XYr is done below...
            #IMr = cKDTree(XYin).query(XYr)[1]    
            #zz_rf = zz_hf.flatten()[IMr].reshape(Ln_r.shape)
            print('somethin went wrong!') 

        # now mask zz_r
        zz_r = zz_rf.copy()
        zz_r[msk_r==0] = np.nan

    return zz_r

def hycom_to_roms_latlon(HY,RMG):
    # HYcom and RoMsGrid come in as dicts with ROMS variable names    
    # The output of this, HYrm, is a dict with 
    # hycom fields on roms horizontal grid points
    # but hycom z levels.
    # velocity will be on both (lat_u,lon_u)
    # and (lat_v,lon_v).
    
    # set up the interpolator now and pass to function
    Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'][0,:,:])
    
    # the names of the variables that need to go on the ROMS grid
    vnames = ['zeta', 'temp', 'salt', 'u', 'v']
    lnhy = HY['lon']
    lthy = HY['lat']
    NR,NC = np.shape(RMG['lon_rho'])
    NZ = len(HY['depth'])
    NT = len(HY['ocean_time'])

    HYrm = dict()
    Tmp = dict()
    HYrm['zeta'] = np.zeros((NT,NR, NC))
    HYrm['salt'] = np.zeros((NT,NZ, NR, NC))
    HYrm['temp'] = np.zeros((NT,NZ, NR, NC))
    Tmp['u_on_u'] = np.zeros((NT,NZ, NR, NC-1))
    Tmp['v_on_u'] = np.zeros((NT,NZ, NR, NC-1))
    Tmp['u_on_v'] = np.zeros((NT,NZ, NR-1, NC))
    Tmp['v_on_v'] = np.zeros((NT,NZ, NR-1, NC))
    HYrm['ubar']   = np.zeros((NT,NR,NC-1))
    HYrm['vbar']   = np.zeros((NT,NR-1,NC))
    HYrm['lat_rho'] = RMG['lat_rho'][:]
    HYrm['lon_rho'] = RMG['lon_rho'][:]
    HYrm['lat_u'] = RMG['lat_u'][:]
    HYrm['lon_u'] = RMG['lon_u'][:]
    HYrm['lat_v'] = RMG['lat_v'][:]
    HYrm['lon_v'] = RMG['lon_v'][:]
    HYrm['depth'] = HY['depth'][:] # depths are from hycom
    HYrm['ocean_time'] = HY['ocean_time'][:]
    HYrm['ocean_time_ref'] = HY['ocean_time_ref']

    HYrm['vinfo']=dict()
    HYrm['vinfo']['depth'] = HY['vinfo']['depth']
    HYrm['vinfo']['ocean_time'] = HY['vinfo']['ocean_time']
    HYrm['vinfo']['ocean_time_ref'] = HY['vinfo']['ocean_time_ref']
    HYrm['vinfo']['lon_rho'] = {'long_name':'rho point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_rho'] = {'long_name':'rho point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['lon_u'] = {'long_name':'rho point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_u'] = {'long_name': 'u point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['lon_v'] = {'long_name':'v point longitude',
                        'units':'degrees_east'}
    HYrm['vinfo']['lat_v'] = {'long_name':'v point latitude',
                        'units':'degrees_north'}
    HYrm['vinfo']['temp'] = {'long_name':'ocean temperature',
                        'units':'degrees C',
                        'coordinates':'z,lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'z,lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'lat_rho,lon_rho',
                        'time':'ocean_time'}
    HYrm['vinfo']['urm'] = {'long_name':'ocean xi velocity',
                        'units':'m/s',
                        'coordinates':'z,lat_u,lon_u',
                        'time':'ocean_time',
                        'note':'this is now rotated in the roms xi direction'}
    HYrm['vinfo']['vrm'] = {'long_name':'ocean eta velocity',
                        'units':'m/s',
                        'coordinates':'z,lat_v,lon_v',
                        'time':'ocean_time',
                        'note':'this is now rotated in the roms eta direction'}
    HYrm['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                        'units':'m/s',
                        'coordinates':'lat_u,lon_u',
                        'time':'ocean_time',
                        'note':'uses hycom depths and this is now rotated in the roms xi direction'}
    HYrm['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                        'units':'m/s',
                        'coordinates':'lat_v,lon_v',
                        'time':'ocean_time',
                        'note':'uses hycom depths and this is now rotated in the roms eta direction'}


    for aa in vnames:
        zhy  = HY[aa]
        for cc in range(NT):
            if aa=='zeta':
                zhy2 = zhy[cc,:,:]
                HYrm[aa][cc,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)            
            elif aa=='temp' or aa=='salt':
                for bb in range(NZ):
                    zhy2 = zhy[cc,bb,:,:]
                    HYrm[aa][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_rho'],RMG['lat_rho'],RMG['mask_rho'],Fz)
            elif aa=='u':
                for bb in range(NZ):
                    zhy2= zhy[cc,bb,:,:]
                    Tmp['u_on_u'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],Fz)
                    Tmp['u_on_v'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],Fz)
            elif aa=='v':
                for bb in range(NZ):
                    zhy2= zhy[cc,bb,:,:]
                    Tmp['v_on_u'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_u'],RMG['lat_u'],RMG['mask_u'],Fz)
                    Tmp['v_on_v'][cc,bb,:,:] = interp_hycom_to_roms(lnhy,lthy,zhy2,RMG['lon_v'],RMG['lat_v'],RMG['mask_v'],Fz)

    # rotate the velocities so that the velocities are in roms eta,xi coordinates
    angr = RMG['angle_u']
    cosang = np.cos(angr)
    sinang = np.sin(angr)
    #Cosang = np.tile(cosang,(NT,NZ,1,1))
    #Sinang = np.tile(sinang,(NT,NZ,1,1))
    #urm = Cosang * HYrm['u_on_u'] + Sinang * HYrm['v_on_u']
    urm = cosang[None,None,:,:] * Tmp['u_on_u'][:,:,:,:] + sinang[None,None,:,:] * Tmp['v_on_u'][:,:,:,:]
    urm[np.isnan(Tmp['u_on_u'])==1] = np.nan
    HYrm['urm'] = urm
    
    angr = RMG['angle_v']
    cosang = np.cos(angr)
    sinang = np.sin(angr)
    #Cosang = np.tile(cosang,(NT,NZ,1,1))
    #Sinang = np.tile(sinang,(NT,NZ,1,1))
    #vrm = Cosang * HYrm['v_on_v'] - Sinang * HYrm['u_on_v']
    vrm = cosang[None,None,:,:] * Tmp['v_on_v'][:,:,:,:] - sinang[None,None,:,:] * Tmp['u_on_v'][:,:,:,:]
    vrm[np.isnan(Tmp['u_on_v'])==1] = np.nan
    HYrm['vrm'] = vrm
    
    # we need the roms depths on roms u and v grids
    Hru = 0.5 * (RMG['h'][:,0:-1] + RMG['h'][:,1:])
    Hrv = 0.5 * (RMG['h'][0:-1,:] + RMG['h'][1:,:])
    # get the locations in z of the hycom output
    hyz = HYrm['depth'].copy()
    #print(np.shape(Hru))
    #print(np.shape(HYrm['urm']))

    # do ubar, the depth average velocity in roms (eta,xi) coordinates
    # so ubar and vbar are calculated from hycom depths before interpolating to roms depths

    use_for = 0 
    if use_for == 1: # we will not use this to get ubar. could be removed 
        # for looping this is VERY slow. Need to make faster
        for aa in range(NR): # lat loop
            for bb in range(NC-1): # lon loop
                # get indices of non nan u based depths
                # this is where the hycom depths are less than the roms depth 
                igu = np.argwhere(hyz <= Hru[aa,bb])
                for cc in range(NT): # time loop
                    uonhyz = HYrm['urm'][cc,:,aa,bb]
                    ug = uonhyz[igu]
                    hgu = hyz[igu]
                    HYrm['ubar'][cc,aa,bb] = ( np.sum( 0.5 * (ug[0:-1]+ug[1:])*(hgu[1:]-hgu[0:-1])) + ug[-1]*(Hru[aa,bb]-hgu[-1]) ) / Hru[aa,bb]
        # do vbar
        for aa in range(NR-1): # lat loop
            for bb in range(NC): # lon loop
                # get indices of non nan v based depths 
                igv = np.argwhere(hyz <= Hrv[aa,bb])
                for cc in range(NT): # time loop
                    vonhyz = HYrm['vrm'][cc,:,aa,bb]
                    vg = vonhyz[igv]
                    hgv = hyz[igv]
                    HYrm['vbar'][cc,aa,bb] = ( np.sum( 0.5 * (vg[0:-1]+vg[1:])*(hgv[1:]-hgv[0:-1])) + vg[-1]*(Hrv[aa,bb]-hgv[-1]) ) / Hrv[aa,bb]
    else:
        # set up mask

        # use python "broadcasting" to mask velocity
        # and put zeros in the right place
        utst = Hru[None,:,:] - hyz[:,None,None]
        vtst = Hrv[None,:,:] - hyz[:,None,None]
    
        dz = hyz[1:]-hyz[0:-1]

        umsk = 0*utst
        vmsk = 0*vtst
        umsk[utst>=0] = 1 # this should put zeros at all depths below the bottom
        vmsk[vtst>=0] = 1 # and ones at all depths above the bottom

        # put zeros at hycom depths below the bottom
        HYrm['urm'] = HYrm['urm']*umsk[None,:,:,:]
        HYrm['vrm'] = HYrm['vrm']*vmsk[None,:,:,:]

        # make copies to make ubar
        uu = HYrm['urm'].copy()
        vv = HYrm['vrm'].copy()

        uu[np.isnan(uu)==1]=0
        vv[np.isnan(vv)==1]=0

        ubar = np.squeeze( np.sum( 0.5 * (uu[:,0:-1,:,:]+uu[:,1:,:,:]) * dz[None,:,None,None], axis=1 ) ) / Hru[None,:,:]
        vbar = np.squeeze( np.sum( 0.5 * (vv[:,0:-1,:,:]+vv[:,1:,:,:]) * dz[None,:,None,None], axis=1 ) ) / Hrv[None,:,:]

        # reapply the land mask...
        umsk2 = np.squeeze( np.isnan( urm[:,0,:,:]))
        vmsk2 = np.squeeze( np.isnan( vrm[:,0,:,:]))
        ubar[umsk2==1] = np.nan
        vbar[vmsk2==1] = np.nan

        # this ubar isn't used but is returned for reference
        HYrm['ubar'] = ubar
        HYrm['vbar'] = vbar
         
    return HYrm


def ocn_r_2_ICdict_slow(OCN_R,RMG):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the initial condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_IC that has all needed fields 
    # for making the .nc file

    i0 = 0 # we will use the first time as the initial condition
    
    OCN_IC = dict()
    # fill in the dict with slicing
       #OCN_IC['ubar'] = np.squeeze(OCN_R['ubar'][i0,:,:])
    #OCN_IC['vbar'] = np.squeeze(OCN_R['vbar'][i0,:,:])

    # the variables that are the same
    var_same = ['lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v'] 
    for vn in var_same:
        OCN_IC[vn] = OCN_R[vn][:]

    OCN_IC['ocean_time_ref'] = OCN_R['ocean_time_ref']


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    hb = np.squeeze(RMG['h'])
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

    OCN_IC['ocean_time'] = np.zeros((1))
    OCN_IC['zeta'] = np.zeros((1,nlt,nln))
    OCN_IC['ocean_time'][0] = OCN_R['ocean_time'][i0]
    OCN_IC['zeta'][0,:,:] = OCN_R['zeta'][i0,:,:]
 
    zhy = OCN_R['depth'] # these are the hycom depths
    eta = OCN_IC['zeta']
    eta_u = 0.5 * (eta[:,:,0:-1]+eta[:,:,1:])
    eta_v = 0.5 * (eta[:,0:-1,:]+eta[:,1:,:])


    Nz   = RMG['Nz']                              # number of vertical levels: 40
    Vtr  = RMG['Vtransform']                       # transformation equation: 2
    Vst  = RMG['Vstretching']                    # stretching function: 4 
    th_s = RMG['THETA_S']                      # surface stretching parameter: 8
    th_b = RMG['THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = RMG['TCLINE']                      # critical depth (m): 50

    OCN_IC['Nz'] = RMG['Nz']
    OCN_IC['Vtr'] = RMG['Vtransform']
    OCN_IC['Vst'] = RMG['Vstretching']
    OCN_IC['th_s'] = RMG['THETA_S']
    OCN_IC['th_b'] = RMG['THETA_B']
    OCN_IC['Tcl'] = RMG['TCLINE']
    OCN_IC['hc'] = RMG['hc']

    TMP = dict()
    TMP['temp'] = np.zeros((1,Nz,nlt,nln)) # a helper becasue we convert to potential temp below
    OCN_IC['temp'] = np.zeros((1,Nz,nlt,nln))
    OCN_IC['salt'] = np.zeros((1,Nz,nlt,nln))
    OCN_IC['u'] = np.zeros((1,Nz,nlt,nln-1))
    OCN_IC['v'] = np.zeros((1,Nz,nlt-1,nln))
    OCN_IC['ubar'] = np.zeros((1,nlt,nln-1))
    OCN_IC['vbar'] = np.zeros((1,nlt-1,nln))

    OCN_IC['vinfo']=dict()
    OCN_IC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_IC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_IC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_IC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_IC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_IC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_IC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    OCN_IC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_IC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_IC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_IC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_IC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_IC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_IC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_IC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']

    OCN_IC['vinfo']['temp'] = {'long_name':'ocean potential temperature',
                        'units':'degrees C',
                        'coordinates':'time,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'tiem,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'time,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['u'] = {'long_name':'ocean xi velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_u,lon_u',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['v'] = {'long_name':'ocean eta velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_v,lon_v',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_u,lon_u',
                        'note':'uses roms depths'}
    OCN_IC['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_v,lon_v',
                        'note':'uses roms depths'}


    # get the roms z's
    #zrom = get_roms_zlevels(Nz,Vtr,Vst,th_s,th_b,Tcl,eta=0*RMG['h'],RMG['h'])
    #zrom = s_coordinate_4(RMG['h'], theta_b, theta_s, Tcline, Nz, hraw=hraw, eta=0*RMG['h'])    

    hraw = None
    if Vst == 4:
        zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta)
        zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_u)
        zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_v)

    
    OCN_IC['Cs_r'] = zrom.Cs_r
    OCN_IC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    
    zr=zrom.z_r  
    zr_u=zrom_u.z_r 
    zr_v=zrom_v.z_r

    for aa in range(nlt):
        print(aa)
        for bb in range(nln):            
            fofz = np.squeeze(OCN_R['temp'][i0,:,aa,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp'][0,:,aa,bb] = np.nan*zr[i0,:,aa,bb]
                OCN_IC['salt'][0,:,aa,bb] = np.nan*zr[i0,:,aa,bb]
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp'][0,:,aa,bb] = Fz(zr[i0,:,aa,bb])
                
                fofz = np.squeeze(OCN_R['salt'][i0,:,aa,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_IC['salt'][0,:,aa,bb] = Fz(zr[i0,:,aa,bb])

                if bb < nln-1:
                    fofz = np.squeeze(OCN_R['urm'][i0,:,aa,bb])
                    ig = np.argwhere(np.isfinite(fofz))
                    if len(ig) < 2:
                        OCN_IC['u'][0,:,aa,bb] = np.nan*zr_u[i0,:,aa,bb]
                        OCN_IC['ubar'][0,aa,bb] = np.nan
                    else:
                        fofz2 = fofz[ig]
                        Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                        uu =  np.squeeze(Fz(zr_u[i0,:,aa,bb]))                
                        OCN_IC['u'][0,:,aa,bb] = uu
                        z2 = np.squeeze(zr_u[i0,:,aa,bb])
                        z3 = np.append(z2,eta_u[i0,aa,bb])
                        dz = np.diff(z3)
                        OCN_IC['ubar'][0,aa,bb] = np.sum(uu*dz) / hb_u[aa,bb]
                if aa < nlt-1:
                    fofz = np.squeeze(OCN_R['vrm'][i0,:,aa,bb])
                    ig = np.argwhere(np.isfinite(fofz))
                    if len(ig) < 2:
                        OCN_IC['v'][0,:,aa,bb] = np.nan*zr_v[i0,:,aa,bb]
                        OCN_IC['vbar'][0,aa,bb] = np.nan
                    else:
                        fofz2 = fofz[ig]
                        Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                        vv = np.squeeze(Fz(zr_v[i0,:,aa,bb]))
                        OCN_IC['v'][0,:,aa,bb] = vv
                        z2 = np.squeeze(zr_v[i0,:,aa,bb])
                        z3 = np.append(z2,eta_v[i0,aa,bb])
                        dz = np.diff(z3)
                        OCN_IC['vbar'][0,aa,bb] = np.sum(vv*dz) / hb_v[aa,bb]

    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr[i0,:,:,:] # pressure in dbar
    OCN_IC['temp'] = seawater.ptmp(OCN_IC['salt'], TMP['temp'],pdb)  

    return OCN_IC


def ocn_r_2_ICdict_old(OCN_R,RMG):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the initial condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_IC that has all needed fields 
    # for making the .nc file

    i0 = 0 # we will use the first time as the initial condition
    
    OCN_IC = dict()
    # fill in the dict with slicing
    OCN_IC['ocean_time'] = OCN_R['ocean_time'][i0]
    OCN_IC['zeta'] = np.squeeze(OCN_R['zeta'][i0,:,:])
    #OCN_IC['ubar'] = np.squeeze(OCN_R['ubar'][i0,:,:])
    #OCN_IC['vbar'] = np.squeeze(OCN_R['vbar'][i0,:,:])

    # the variables that are the same
    var_same = ['lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v'] 
    for vn in var_same:
        OCN_IC[vn] = OCN_R[vn][:]

    OCN_IC['ocean_time_ref'] = OCN_R['ocean_time_ref']


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = OCN_R['depth'] # these are the hycom depths
    eta = OCN_IC['zeta']
    eta_u = 0.5 * (eta[:,0:-1]+eta[:,1:])
    eta_v = 0.5 * (eta[0:-1,:]+eta[1:,:])
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(eta)

    Nz   = RMG['Nz']                              # number of vertical levels: 40
    Vtr  = RMG['Vtransform']                       # transformation equation: 2
    Vst  = RMG['Vstretching']                    # stretching function: 4 
    th_s = RMG['THETA_S']                      # surface stretching parameter: 8
    th_b = RMG['THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = RMG['TCLINE']                      # critical depth (m): 50

    OCN_IC['Nz'] = RMG['Nz']
    OCN_IC['Vtr'] = RMG['Vtransform']
    OCN_IC['Vst'] = RMG['Vstretching']
    OCN_IC['th_s'] = RMG['THETA_S']
    OCN_IC['th_b'] = RMG['THETA_B']
    OCN_IC['Tcl'] = RMG['TCLINE']
    OCN_IC['hc'] = RMG['hc']

    TMP = dict()
    TMP['temp'] = np.zeros((Nz,nlt,nln)) # a helper becasue we convert to potential temp below
    OCN_IC['temp'] = np.zeros((Nz,nlt,nln))
    OCN_IC['salt'] = np.zeros((Nz,nlt,nln))
    OCN_IC['u'] = np.zeros((Nz,nlt,nln-1))
    OCN_IC['v'] = np.zeros((Nz,nlt-1,nln))
    OCN_IC['ubar'] = np.zeros((nlt,nln-1))
    OCN_IC['vbar'] = np.zeros((nlt-1,nln))

    OCN_IC['vinfo']=dict()
    OCN_IC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_IC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_IC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_IC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_IC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_IC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_IC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_IC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']
    OCN_IC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_IC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_IC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_IC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_IC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_IC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_IC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    # note we put potential temperature in, not temp. 
    OCN_IC['vinfo']['temp'] = {'long_name':'ocean potential temperature',
                        'units':'degrees C',
                        'coordinates':'time,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'tiem,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'time,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['u'] = {'long_name':'ocean xi velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_u,lon_u',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['v'] = {'long_name':'ocean eta velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_v,lon_v',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_u,lon_u',
                        'note':'uses roms depths'}
    OCN_IC['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_v,lon_v',
                        'note':'uses roms depths'}

    # get the roms z's
    hraw = None
    if Vst == 4:
        zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta)
        zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_u)
        zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_v)

    
    OCN_IC['Cs_r'] = np.squeeze(zrom.Cs_r)
    OCN_IC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    # do we need sc_r ???
    #OCN_IC['sc_r'] = []
    #OCN_IC['vinfo']['sc_r'] = {'long_name':'S-coordinate at RHO-points',
    #                    'units':'nondimensional',
    #                    'valid min':'-1',
    #                    'valid max':'0',
    #                    'field':'sc_r, scalar, series'}

    zr=np.squeeze(zrom.z_r[0,:,:,:])    
    zr_u=np.squeeze(zrom_u.z_r[0,:,:,:])    
    zr_v=np.squeeze(zrom_v.z_r[0,:,:,:])

    for aa in range(nlt):
        print(aa)
        for bb in range(nln):            
            fofz = np.squeeze(OCN_R['temp'][i0,:,aa,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp'][:,aa,bb] = np.nan*zr[:,aa,bb]
                OCN_IC['salt'][:,aa,bb] = np.nan*zr[:,aa,bb]
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp'][:,aa,bb] = Fz(zr[:,aa,bb])
                
                fofz = np.squeeze(OCN_R['salt'][i0,:,aa,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_IC['salt'][:,aa,bb] = Fz(zr[:,aa,bb])

                if bb < nln-1:
                    fofz = np.squeeze(OCN_R['urm'][i0,:,aa,bb])
                    ig = np.argwhere(np.isfinite(fofz))
                    if len(ig) < 2:
                        OCN_IC['u'][:,aa,bb] = np.nan*zr_u[:,aa,bb]
                        OCN_IC['ubar'][aa,bb] = np.nan
                    else:
                        fofz2 = fofz[ig]
                        Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                        uu =  Fz(zr_u[:,aa,bb])                
                        OCN_IC['u'][:,aa,bb] = uu
                        z2 = np.squeeze(zr_u[:,aa,bb])
                        z3 = np.append(z2,eta_u[aa,bb])
                        dz = np.diff(z3)
                        OCN_IC['ubar'][aa,bb] = np.sum(uu*dz) / hb_u[aa,bb]
                if aa < nlt-1:
                    fofz = np.squeeze(OCN_R['vrm'][i0,:,aa,bb])
                    ig = np.argwhere(np.isfinite(fofz))
                    if len(ig) < 2:
                        OCN_IC['v'][:,aa,bb] = np.nan*zr_v[:,aa,bb]
                        OCN_IC['vbar'][aa,bb] = np.nan
                    else:
                        fofz2 = fofz[ig]
                        Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                        vv = Fz(zr_v[:,aa,bb])
                        OCN_IC['v'][:,aa,bb] = vv
                        z2 = np.squeeze(zr_v[:,aa,bb])
                        z3 = np.append(z2,eta_v[aa,bb])
                        dz = np.diff(z3)
                        OCN_IC['vbar'][aa,bb] = np.sum(vv*dz) / hb_v[aa,bb]

    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr # pressure in dbar
    OCN_IC['temp'] = seawater.ptmp(OCN_IC['salt'], TMP['temp'],pdb)  

    return OCN_IC

def ocn_r_2_ICdict(OCN_R,RMG):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the initial condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_IC that has all needed fields 
    # for making the .nc file

    i0 = 0 # we will use the first time as the initial condition
    
    OCN_IC = dict()
    # fill in the dict with slicing
    OCN_IC['ocean_time'] = np.zeros((1))
    OCN_IC['ocean_time'][0] = OCN_R['ocean_time'][i0]
    #OCN_IC['ubar'] = np.squeeze(OCN_R['ubar'][i0,:,:])
    #OCN_IC['vbar'] = np.squeeze(OCN_R['vbar'][i0,:,:])

    # the variables that are the same
    var_same = ['lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v'] 
    for vn in var_same:
        OCN_IC[vn] = OCN_R[vn][:]

    OCN_IC['ocean_time_ref'] = OCN_R['ocean_time_ref']


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = OCN_R['depth'] # these are the hycom depths
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

    Nz   = RMG['Nz']                              # number of vertical levels: 40
    Vtr  = RMG['Vtransform']                       # transformation equation: 2
    Vst  = RMG['Vstretching']                    # stretching function: 4 
    th_s = RMG['THETA_S']                      # surface stretching parameter: 8
    th_b = RMG['THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = RMG['TCLINE']                      # critical depth (m): 50

    OCN_IC['zeta'] = np.zeros((1,nlt,nln))
    OCN_IC['zeta'][0,:,:] = OCN_R['zeta'][i0,:,:]

    eta = np.squeeze(OCN_IC['zeta'][0,:,:])
    eta_u = 0.5 * (eta[:,0:-1]+eta[:,1:])
    eta_v = 0.5 * (eta[0:-1,:]+eta[1:,:])

    OCN_IC['Nz'] = np.squeeze(RMG['Nz'])
    OCN_IC['Vtr'] = np.squeeze(RMG['Vtransform'])
    OCN_IC['Vst'] = np.squeeze(RMG['Vstretching'])
    OCN_IC['th_s'] = np.squeeze(RMG['THETA_S'])
    OCN_IC['th_b'] = np.squeeze(RMG['THETA_B'])
    OCN_IC['Tcl'] = np.squeeze(RMG['TCLINE'])
    OCN_IC['hc'] = np.squeeze(RMG['hc'])

    TMP = dict()
    TMP['temp'] = np.zeros((1,Nz,nlt,nln)) # a helper becasue we convert to potential temp below
    OCN_IC['temp'] = np.zeros((1,Nz,nlt,nln))
    OCN_IC['salt'] = np.zeros((1,Nz,nlt,nln))
    OCN_IC['u'] = np.zeros((1,Nz,nlt,nln-1))
    OCN_IC['v'] = np.zeros((1,Nz,nlt-1,nln))
    OCN_IC['ubar'] = np.zeros((1,nlt,nln-1))
    OCN_IC['vbar'] = np.zeros((1,nlt-1,nln))

    OCN_IC['vinfo']=dict()
    OCN_IC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_IC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_IC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_IC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_IC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_IC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_IC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_IC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']

    OCN_IC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_IC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_IC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_IC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_IC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_IC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_IC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    OCN_IC['vinfo']['temp'] = {'long_name':'ocean potential temperature',
                        'units':'degrees C',
                        'coordinates':'time,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['salt'] = {'long_name':'ocean salinity',
                        'units':'psu',
                        'coordinates':'tiem,s,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['zeta'] = {'long_name':'ocean sea surface height',
                        'units':'m',
                        'coordinates':'time,lat_rho,lon_rho',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['u'] = {'long_name':'ocean xi velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_u,lon_u',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['v'] = {'long_name':'ocean eta velocity',
                        'units':'m/s',
                        'coordinates':'time,s,lat_v,lon_v',
                        'time':'ocean_time'}
    OCN_IC['vinfo']['ubar'] = {'long_name':'ocean xi depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_u,lon_u',
                        'note':'uses roms depths'}
    OCN_IC['vinfo']['vbar'] = {'long_name':'ocean eta depth avg velocity',
                        'units':'m/s',
                        'coordinates':'time,lat_v,lon_v',
                        'note':'uses roms depths'}


    # get the roms z's
    hraw = None
    if Vst == 4:
        zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta)
        zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_u)
        zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_v)
    
    OCN_IC['Cs_r'] = np.squeeze(zrom.Cs_r)
    OCN_IC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    
    # do we need sc_r ???
    #OCN_IC['sc_r'] = []
    #OCN_IC['vinfo']['sc_r'] = {'long_name':'S-coordinate at RHO-points',
    #                    'units':'nondimensional',
    #                    'valid min':'-1',
    #                    'valid max':'0',
    #                    'field':'sc_r, scalar, series'}

    zr=np.squeeze(zrom.z_r[0,:,:,:])    
    zr_u=np.squeeze(zrom_u.z_r[0,:,:,:])    
    zr_v=np.squeeze(zrom_v.z_r[0,:,:,:])

    for aa in range(nlt):
        for bb in range(nln):            
            fofz = np.squeeze(OCN_R['temp'][i0,:,aa,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp'][0,:,aa,bb] = np.squeeze(np.nan*zr[:,aa,bb])
                OCN_IC['salt'][0,:,aa,bb] = np.squeeze(np.nan*zr[:,aa,bb])
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp'][0,:,aa,bb] = np.squeeze(Fz(zr[:,aa,bb]))
                
                fofz = np.squeeze(OCN_R['salt'][i0,:,aa,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_IC['salt'][0,:,aa,bb] = np.squeeze(Fz(zr[:,aa,bb]))

            if bb < nln-1:
                fofz = np.squeeze(OCN_R['urm'][i0,:,aa,bb])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_IC['u'][0,:,aa,bb] = np.squeeze(np.nan*zr_u[:,aa,bb])
                    OCN_IC['ubar'][0,aa,bb] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    uu =  np.squeeze(Fz(zr_u[:,aa,bb]))                
                    OCN_IC['u'][0,:,aa,bb] = uu
                    z2 = np.squeeze(zr_u[:,aa,bb])
                    z3 = np.append(z2,eta_u[aa,bb])
                    dz = np.diff(z3)
                    OCN_IC['ubar'][0,aa,bb] = np.sum(uu*dz) / hb_u[aa,bb]
            
            if aa < nlt-1:
                fofz = np.squeeze(OCN_R['vrm'][i0,:,aa,bb])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_IC['v'][0,:,aa,bb] = np.squeeze(np.nan*zr_v[:,aa,bb])
                    OCN_IC['vbar'][0,aa,bb] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                    vv = np.squeeze(Fz(zr_v[:,aa,bb]))
                    OCN_IC['v'][0,:,aa,bb] = vv
                    z2 = np.squeeze(zr_v[:,aa,bb])
                    z3 = np.append(z2,eta_v[aa,bb])
                    dz = np.diff(z3)
                    OCN_IC['vbar'][0,aa,bb] = np.sum(vv*dz) / hb_v[aa,bb]

    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr # pressure in dbar
    OCN_IC['temp'][0,:,:,:] = seawater.ptmp(np.squeeze(OCN_IC['salt']), np.squeeze(TMP['temp']),np.squeeze(pdb))  

    return OCN_IC

def ocn_r_2_BCdict(OCN_R,RMG):
    # this slices the OCN_R dictionary at the first time for all needed 
    # variables for the boundary condition for roms
    # it then interpolates from the hycom z values that the vars are on
    # and places them on the ROMS z levels
    # this returns another dictionary OCN_BC that has all needed fields 
    # for making the BC.nc file

    OCN_BC = dict()
    # fill in the dict with slicing
    OCN_BC['ocean_time'] = OCN_R['ocean_time']
    Nt = len( OCN_BC['ocean_time'] )
    OCN_BC['ocean_time_ref'] = OCN_R['ocean_time_ref']

    OCN_BC['Nz'] = np.squeeze(RMG['Nz'])
    OCN_BC['Vtr'] = np.squeeze(RMG['Vtransform'])
    OCN_BC['Vst'] = np.squeeze(RMG['Vstretching'])
    OCN_BC['th_s'] = np.squeeze(RMG['THETA_S'])
    OCN_BC['th_b'] = np.squeeze(RMG['THETA_B'])
    OCN_BC['Tcl'] = np.squeeze(RMG['TCLINE'])
    OCN_BC['hc'] = np.squeeze(RMG['hc'])


    # these variables need to be time sliced and then vertically interpolated
    #varin3d = ['temp','salt','urm','vrm']
    zhy = OCN_R['depth'] # these are the hycom depths
    hb = RMG['h']
    hb_u = 0.5 * (hb[:,0:-1]+hb[:,1:])
    hb_v = 0.5 * (hb[0:-1,:]+hb[1:,:])
    nlt,nln = np.shape(hb)

    Nz   = RMG['Nz']                              # number of vertical levels: 40
    Vtr  = RMG['Vtransform']                       # transformation equation: 2
    Vst  = RMG['Vstretching']                    # stretching function: 4 
    th_s = RMG['THETA_S']                      # surface stretching parameter: 8
    th_b = RMG['THETA_B']                      # bottom  stretching parameter: 3
    Tcl  = RMG['TCLINE']                      # critical depth (m): 50

    eta = np.squeeze(OCN_R['zeta'].copy())
    eta_u = 0.5 * (eta[:,0:-1]+eta[:,1:])
    eta_v = 0.5 * (eta[0:-1,:]+eta[1:,:])

    OCN_BC['temp_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_south'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_south'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_south'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_south'] = np.zeros((Nt,nln))
    OCN_BC['zeta_south'] = np.zeros((Nt,nln))

    OCN_BC['temp_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['salt_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['u_north'] = np.zeros((Nt,Nz,nln-1))
    OCN_BC['v_north'] = np.zeros((Nt,Nz,nln))
    OCN_BC['ubar_north'] = np.zeros((Nt,nln-1))
    OCN_BC['vbar_north'] = np.zeros((Nt,nln))
    OCN_BC['zeta_north'] = np.zeros((Nt,nln))

    OCN_BC['temp_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['salt_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['u_west'] = np.zeros((Nt,Nz,nlt))
    OCN_BC['v_west'] = np.zeros((Nt,Nz,nlt-1))
    OCN_BC['ubar_west'] = np.zeros((Nt,nlt))
    OCN_BC['vbar_west'] = np.zeros((Nt,nlt-1))
    OCN_BC['zeta_west'] = np.zeros((Nt,nlt))

    OCN_BC['zeta_south'] = np.squeeze(OCN_R['zeta'][:,0,:])
    OCN_BC['zeta_north'] = np.squeeze(OCN_R['zeta'][:,-1,:])
    OCN_BC['zeta_west'] = np.squeeze(OCN_R['zeta'][:,:,0])
    OCN_BC['ubar_south'] = np.squeeze(OCN_R['ubar'][:,0,:])
    OCN_BC['ubar_north'] = np.squeeze(OCN_R['ubar'][:,-1,:])
    OCN_BC['ubar_west'] = np.squeeze(OCN_R['ubar'][:,:,0])
    OCN_BC['vbar_south'] = np.squeeze(OCN_R['vbar'][:,0,:])
    OCN_BC['vbar_north'] = np.squeeze(OCN_R['vbar'][:,-1,:])
    OCN_BC['vbar_west'] = np.squeeze(OCN_R['vbar'][:,:,0])
     
    TMP = dict()
    TMP['temp_north'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_south'] = np.zeros((Nt,Nz,nln)) # a helper becasue we convert to potential temp below
    TMP['temp_west'] = np.zeros((Nt,Nz,nlt)) # a helper becasue we convert to potential temp below
    
 
    OCN_BC['vinfo']=dict()
    OCN_BC['vinfo']['ocean_time'] = OCN_R['vinfo']['ocean_time']
    OCN_BC['vinfo']['ocean_time_ref'] = OCN_R['vinfo']['ocean_time_ref']
    OCN_BC['vinfo']['lat_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lon_rho'] = OCN_R['vinfo']['lat_rho']
    OCN_BC['vinfo']['lat_u'] = OCN_R['vinfo']['lat_u']
    OCN_BC['vinfo']['lon_u'] = OCN_R['vinfo']['lon_u']
    OCN_BC['vinfo']['lat_v'] = OCN_R['vinfo']['lat_v']
    OCN_BC['vinfo']['lon_v'] = OCN_R['vinfo']['lon_v']

    OCN_BC['vinfo']['Nz'] = {'long_name':'number of vertical rho levels',
                             'units':'none'}
    OCN_BC['vinfo']['Vtr'] = {'long_name':'vertical terrain-following transformation equation'}
    OCN_BC['vinfo']['Vst'] = {'long_name':'vertical terrain-following stretching function'}
    OCN_BC['vinfo']['th_s'] = {'long_name':'S-coordinate surface control parameter',
                               'units':'nondimensional',
                               'field': 'theta_s, scalar, series'}
    OCN_BC['vinfo']['th_b'] = {'long_name':'S-coordinate bottom control parameter',
                               'units':'nondimensional',
                               'field': 'theta_b, scalar, series'}
    OCN_BC['vinfo']['Tcl'] = {'long_name':'S-coordinate surface/bottom layer width',
                               'units':'meter',
                               'field': 'Tcline, scalar, series'}
    OCN_BC['vinfo']['hc'] = {'long_name':'S-coordinate parameter, critical depth',
                               'units':'meter',
                               'field': 'hc, scalar, series'}
    
    OCN_BC['vinfo']['temp_south'] = {'long_name':'ocean potential temperature southern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_south'] = {'long_name':'ocean salinity southern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_south'] = {'long_name':'ocean sea surface height southern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_south'] = {'long_name':'ocean xi velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_south'] = {'long_name':'ocean eta velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_south'] = {'long_name':'ocean xi depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_south'] = {'long_name':'ocean eta depth avg velocity southern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_north'] = {'long_name':'ocean potential temperature northern boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_north'] = {'long_name':'ocean salinity northern boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_north'] = {'long_name':'ocean sea surface height northern boundary',
                        'units':'m',
                        'coordinates':'time,xi_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_north'] = {'long_name':'ocean xi velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_north'] = {'long_name':'ocean eta velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,s,xi_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_north'] = {'long_name':'ocean xi depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_north'] = {'long_name':'ocean eta depth avg velocity northern boundary',
                        'units':'m/s',
                        'coordinates':'time,xi_v',
                        'note':'uses roms depths'}

    OCN_BC['vinfo']['temp_west'] = {'long_name':'ocean potential temperature western boundary',
                        'units':'degrees C',
                        'coordinates':'time,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['salt_west'] = {'long_name':'ocean salinity western boundary',
                        'units':'psu',
                        'coordinates':'tiem,s,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['zeta_west'] = {'long_name':'ocean sea surface height western boundary',
                        'units':'m',
                        'coordinates':'time,eta_rho',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['u_west'] = {'long_name':'ocean xi velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_u',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['v_west'] = {'long_name':'ocean eta velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,s,eta_v',
                        'time':'ocean_time'}
    OCN_BC['vinfo']['ubar_west'] = {'long_name':'ocean xi depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_u',
                        'note':'uses roms depths'}
    OCN_BC['vinfo']['vbar_west'] = {'long_name':'ocean eta depth avg velocity western boundary',
                        'units':'m/s',
                        'coordinates':'time,eta_v',
                        'note':'uses roms depths'}

    eta = np.squeeze(OCN_R['zeta'])
    eta_u = 0.5 * (eta[:,:,0:-1]+eta[:,:,1:])
    eta_v = 0.5 * (eta[:,0:-1,:]+eta[:,1:,:])

    # get the roms z's
    hraw = None
    if Vst == 4:
        zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta))
        zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta_u))
        zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=np.squeeze(eta_v))
    
    OCN_BC['Cs_r'] = np.squeeze(zrom.Cs_r)
    OCN_BC['vinfo']['Cs_r'] = {'long_name':'S-coordinate stretching curves at RHO-points',
                        'units':'nondimensional',
                        'valid min':'-1',
                        'valid max':'0',
                        'field':'Cs_r, scalar, series'}
    
    # do we need sc_r ???
    #OCN_IC['sc_r'] = []
    #OCN_IC['vinfo']['sc_r'] = {'long_name':'S-coordinate at RHO-points',
    #                    'units':'nondimensional',
    #                    'valid min':'-1',
    #                    'valid max':'0',
    #                    'field':'sc_r, scalar, series'}

    zr_s=np.squeeze(zrom.z_r[:,:,0,:])    
    zr_us=np.squeeze(zrom_u.z_r[:,:,0,:])    
    zr_vs=np.squeeze(zrom_v.z_r[:,:,0,:])
    zr_n=np.squeeze(zrom.z_r[:,:,-1,:])    
    zr_un=np.squeeze(zrom_u.z_r[:,:,-1,:])    
    zr_vn=np.squeeze(zrom_v.z_r[:,:,-1,:])
    zr_w=np.squeeze(zrom.z_r[:,:,:,0])    
    zr_uw=np.squeeze(zrom_u.z_r[:,:,:,0])    
    zr_vw=np.squeeze(zrom_v.z_r[:,:,:,0])

    for aa in range(Nt):
        for bb in range(nln):
            fofz = np.squeeze(OCN_R['temp'][aa,:,0,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_south'][aa,:,bb] = np.squeeze(np.nan*zr_s[aa,:,bb])
                OCN_BC['salt_south'][aa,:,bb] = np.squeeze(np.nan*zr_s[aa,:,bb])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                #print(np.shape(zr_s))
                #print(np.shape(TMP['temp_south']))
                TMP['temp_south'][aa,:,bb] = np.squeeze(Fz(zr_s[aa,:,bb]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,0,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_south'][aa,:,bb] = np.squeeze(Fz(zr_s[aa,:,bb]))
                
            fofz = np.squeeze(OCN_R['vrm'][aa,:,0,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['v_south'][aa,:,bb] = np.squeeze(np.nan*zr_vs[aa,:,bb])
                OCN_BC['vbar_south'][aa,bb] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                vv =  np.squeeze(Fz(zr_vs[aa,:,bb]))                
                OCN_BC['v_south'][aa,:,bb] = vv
                z2 = np.squeeze(zr_vs[aa,:,bb])
                z3 = np.append(z2,eta_v[aa,0,bb])
                dz = np.diff(z3)
                OCN_BC['vbar_south'][aa,bb] = np.sum(vv*dz) / hb_v[0,bb]

            if bb < nln-1:
                fofz = np.squeeze(OCN_R['urm'][aa,:,0,bb])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['u_south'][aa,:,bb] = np.squeeze(np.nan*zr_us[aa,:,bb])
                    OCN_BC['ubar_south'][aa,bb] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    uu =  np.squeeze(Fz(zr_us[aa,:,bb]))                
                    OCN_BC['u_south'][aa,:,bb] = uu
                    z2 = np.squeeze(zr_us[aa,:,bb])
                    z3 = np.append(z2,eta_u[aa,0,bb])
                    dz = np.diff(z3)
                    OCN_BC['ubar_south'][aa,bb] = np.sum(uu*dz) / hb_u[0,bb]

            fofz = np.squeeze(OCN_R['temp'][aa,:,-1,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_north'][aa,:,bb] = np.squeeze(np.nan*zr_n[aa,:,bb])
                OCN_BC['salt_north'][aa,:,bb] = np.squeeze(np.nan*zr_n[aa,:,bb])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp_north'][aa,:,bb] = np.squeeze(Fz(zr_n[aa,:,bb]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,-1,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_north'][aa,:,bb] = np.squeeze(Fz(zr_n[aa,:,bb]))
                
            fofz = np.squeeze(OCN_R['vrm'][aa,:,-1,bb])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['v_north'][aa,:,bb] = np.squeeze(np.nan*zr_vn[aa,:,bb])
                OCN_BC['vbar_north'][aa,bb] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                vv =  np.squeeze(Fz(zr_vn[aa,:,bb]))                
                OCN_BC['v_north'][aa,:,bb] = vv
                z2 = np.squeeze(zr_vs[aa,:,bb])
                z3 = np.append(z2,eta_v[aa,-1,bb])
                dz = np.diff(z3)
                OCN_BC['vbar_north'][aa,bb] = np.sum(vv*dz) / hb_v[-1,bb]

            if bb < nln-1:
                fofz = np.squeeze(OCN_R['urm'][aa,:,-1,bb])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['u_north'][aa,:,bb] = np.squeeze(np.nan*zr_un[aa,:,bb])
                    OCN_BC['ubar_north'][aa,bb] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    uu =  np.squeeze(Fz(zr_un[aa,:,bb]))                
                    OCN_BC['u_north'][aa,:,bb] = uu
                    z2 = np.squeeze(zr_un[aa,:,bb])
                    z3 = np.append(z2,eta_u[aa,0,bb])
                    dz = np.diff(z3)
                    OCN_BC['ubar_north'][aa,bb] = np.sum(uu*dz) / hb_u[-1,bb]

        for cc in range(nlt):
            fofz = np.squeeze(OCN_R['temp'][aa,:,cc,0])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2: # you get in here if all f(z) is nan, ie. we are in land
                # we also make sure that if there is only 1 good value, we also return nans
                TMP['temp_west'][aa,:,cc] = np.squeeze(np.nan*zr_w[aa,:,cc])
                OCN_BC['salt_west'][aa,:,cc] = np.squeeze(np.nan*zr_w[aa,:,cc])
                
            else:    
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
                TMP['temp_west'][aa,:,cc] = np.squeeze(Fz(zr_w[aa,:,cc]))
                
                fofz = np.squeeze(OCN_R['salt'][aa,:,cc,0])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                OCN_BC['salt_west'][aa,:,cc] = np.squeeze(Fz(zr_w[aa,:,cc]))

            if cc < nlt-1:    
                fofz = np.squeeze(OCN_R['vrm'][aa,:,cc,0])
                ig = np.argwhere(np.isfinite(fofz))
                if len(ig) < 2:
                    OCN_BC['v_west'][aa,:,cc] = np.squeeze(np.nan*zr_vw[aa,:,cc])
                    OCN_BC['vbar_west'][aa,cc] = np.nan
                else:
                    fofz2 = fofz[ig]
                    Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                    vv =  np.squeeze(Fz(zr_vw[aa,:,cc]))                
                    OCN_BC['v_west'][aa,:,cc] = vv
                    z2 = np.squeeze(zr_vw[aa,:,cc])
                    z3 = np.append(z2,eta_v[aa,cc,0])
                    dz = np.diff(z3)
                    OCN_BC['vbar_west'][aa,cc] = np.sum(vv*dz) / hb_v[cc,0]

            fofz = np.squeeze(OCN_R['urm'][aa,:,cc,0])
            ig = np.argwhere(np.isfinite(fofz))
            if len(ig) < 2:
                OCN_BC['u_west'][aa,:,cc] = np.squeeze(np.nan*zr_uw[aa,:,cc])
                OCN_BC['ubar_west'][aa,cc] = np.nan
            else:
                fofz2 = fofz[ig]
                Fz = interp1d(np.squeeze(-zhy[ig]),np.squeeze(fofz2),bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                uu =  np.squeeze(Fz(zr_uw[aa,:,cc]))                
                OCN_BC['u_west'][aa,:,cc] = uu
                z2 = np.squeeze(zr_uw[aa,:,cc])
                z3 = np.append(z2,eta_u[aa,cc,0])
                dz = np.diff(z3)
                OCN_BC['ubar_west'][aa,cc] = np.sum(uu*dz) / hb_u[cc,0]


    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr_s # pressure in dbar
    OCN_BC['temp_south'] = seawater.ptmp(np.squeeze(OCN_BC['salt_south']), np.squeeze(TMP['temp_south']),np.squeeze(pdb))  
    pdb = -zr_n # pressure in dbar
    OCN_BC['temp_north'] = seawater.ptmp(np.squeeze(OCN_BC['salt_north']), np.squeeze(TMP['temp_north']),np.squeeze(pdb))  
    pdb = -zr_w # pressure in dbar
    OCN_BC['temp_west'] = seawater.ptmp(np.squeeze(OCN_BC['salt_west']), np.squeeze(TMP['temp_west']),np.squeeze(pdb))  

    return OCN_BC



def ocn_roms_IC_dict_to_netcdf(ATM_R,fn_out):
    ds = xr.Dataset(
        data_vars = dict(
            temp       = (["time","s_rho","er","xr"],ATM_R['temp'],ATM_R['vinfo']['temp']),
            salt       = (["time","s_rho","er","xr"],ATM_R['salt'],ATM_R['vinfo']['salt']),
            u          = (["time","s_rho","eu","xu"],ATM_R['u'],ATM_R['vinfo']['u']),
            v          = (["time","s_rho","ev","xv"],ATM_R['v'],ATM_R['vinfo']['v']),
            ubar       = (["time","eu","xu"],ATM_R['ubar'],ATM_R['vinfo']['ubar']),
            vbar       = (["time","ev","xv"],ATM_R['vbar'],ATM_R['vinfo']['vbar']),
            zeta       = (["time","er","xr"],ATM_R['zeta'],ATM_R['vinfo']['zeta']),
            Vtransform = ([],ATM_R['Vtr'],ATM_R['vinfo']['Vtr']),
            Vstretching = ([],ATM_R['Vst'],ATM_R['vinfo']['Vst']),
            theta_s = ([],ATM_R['th_s'],ATM_R['vinfo']['th_s']),
            theta_b = ([],ATM_R['th_b'],ATM_R['vinfo']['th_b']),
            Tcline = ([],ATM_R['Tcl'],ATM_R['vinfo']['Tcl']),
            hc = ([],ATM_R['hc'],ATM_R['vinfo']['hc']),
        ),
        coords=dict(
            lat_rho =(["er","xr"],ATM_R['lat_rho'], ATM_R['vinfo']['lat_rho']),
            lon_rho =(["er","xr"],ATM_R['lon_rho'], ATM_R['vinfo']['lon_rho']),
            lat_u   =(["eu","xu"],ATM_R['lat_u'], ATM_R['vinfo']['lat_u']),
            lon_u   =(["eu","xu"],ATM_R['lon_u'], ATM_R['vinfo']['lon_u']),
            lat_v   =(["ev","xv"],ATM_R['lat_v'], ATM_R['vinfo']['lat_v']),
            lon_v   =(["ev","xv"],ATM_R['lon_v'], ATM_R['vinfo']['lon_v']),   
            ocean_time = (["time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            Cs_r = (["s_rho"],ATM_R['Cs_r'],ATM_R['vinfo']['Cs_r']),
         ),
        attrs={'type':'ocean initial condition file fields for starting roms',
            'time info':'ocean time is from '+ ATM_R['ocean_time_ref'].strftime("%Y/%m/%d %H:%M:%S") },
        )
    print(ds)

    ds.to_netcdf(fn_out)

def ocn_roms_BC_dict_to_netcdf(ATM_R,fn_out):
    ds = xr.Dataset(
        data_vars = dict(
            temp_south       = (["time","s_rho","xr"],ATM_R['temp_south'],ATM_R['vinfo']['temp_south']),
            salt_south       = (["time","s_rho","xr"],ATM_R['salt_south'],ATM_R['vinfo']['salt_south']),
            u_south          = (["time","s_rho","xu"],ATM_R['u_south'],ATM_R['vinfo']['u_south']),
            v_south          = (["time","s_rho","xv"],ATM_R['v_south'],ATM_R['vinfo']['v_south']),
            ubar_south       = (["time","xu"],ATM_R['ubar_south'],ATM_R['vinfo']['ubar_south']),
            vbar_south       = (["time","xv"],ATM_R['vbar_south'],ATM_R['vinfo']['vbar_south']),
            zeta_south       = (["time","xr"],ATM_R['zeta_south'],ATM_R['vinfo']['zeta_south']),
            temp_north       = (["time","s_rho","xr"],ATM_R['temp_north'],ATM_R['vinfo']['temp_north']),
            salt_north       = (["time","s_rho","xr"],ATM_R['salt_north'],ATM_R['vinfo']['salt_north']),
            u_north          = (["time","s_rho","xu"],ATM_R['u_north'],ATM_R['vinfo']['u_north']),
            v_north          = (["time","s_rho","xv"],ATM_R['v_north'],ATM_R['vinfo']['v_north']),
            ubar_north       = (["time","xu"],ATM_R['ubar_north'],ATM_R['vinfo']['ubar_north']),
            vbar_north       = (["time","xv"],ATM_R['vbar_north'],ATM_R['vinfo']['vbar_north']),
            zeta_north       = (["time","xr"],ATM_R['zeta_north'],ATM_R['vinfo']['zeta_north']),
            temp_west        = (["time","s_rho","er"],ATM_R['temp_west'],ATM_R['vinfo']['temp_west']),
            salt_west        = (["time","s_rho","er"],ATM_R['salt_west'],ATM_R['vinfo']['salt_west']),
            u_west           = (["time","s_rho","eu"],ATM_R['u_west'],ATM_R['vinfo']['u_west']),
            v_west           = (["time","s_rho","ev"],ATM_R['v_west'],ATM_R['vinfo']['v_west']),
            ubar_west        = (["time","eu"],ATM_R['ubar_west'],ATM_R['vinfo']['ubar_west']),
            vbar_west        = (["time","ev"],ATM_R['vbar_west'],ATM_R['vinfo']['vbar_west']),
            zeta_west        = (["time","er"],ATM_R['zeta_west'],ATM_R['vinfo']['zeta_west']),
            Vtransform = ([],ATM_R['Vtr'],ATM_R['vinfo']['Vtr']),
            Vstretching = ([],ATM_R['Vst'],ATM_R['vinfo']['Vst']),
            theta_s = ([],ATM_R['th_s'],ATM_R['vinfo']['th_s']),
            theta_b = ([],ATM_R['th_b'],ATM_R['vinfo']['th_b']),
            Tcline = ([],ATM_R['Tcl'],ATM_R['vinfo']['Tcl']),
            hc = ([],ATM_R['hc'],ATM_R['vinfo']['hc']),
        ),
        coords=dict(
            ocean_time = (["time"],ATM_R['ocean_time'], ATM_R['vinfo']['ocean_time']),
            Cs_r = (["s_rho"],ATM_R['Cs_r'],ATM_R['vinfo']['Cs_r']),
         ),
        attrs={'type':'ocean boundary condition file fields for starting roms',
            'time info':'ocean time is from '+ ATM_R['ocean_time_ref'].strftime("%Y/%m/%d %H:%M:%S") },
        )
    print(ds)

    ds.to_netcdf(fn_out)
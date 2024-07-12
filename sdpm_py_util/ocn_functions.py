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
        OCN['salt'] = sal
        OCN['zeta'] = eta
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

        #HYrm['umsk'] = umsk

        # get velocities and a slice
#        u0 = HYrm['urm'][0,0,:,:].copy()
#        v0 = HYrm['vrm'][0,0,:,:].copy()
        # make copies to make ubar
        uu = HYrm['urm'].copy()
        vv = HYrm['vrm'].copy()

        # make the 4d mask of where there is water
        # roms land will still be nan'd 
        #oo = np.ones(np.shape(HYrm['ocean_time']))
        #umsk4d = oo[:,None,None,None] * umsk[None,:,:,:]
        #vmsk4d = oo[:,None,None,None] * vmsk[None,:,:,:]

        #print(np.shape(umsk4d))
        #print(np.shape(HYrm['urm']))

    #    HYrm['urm'][umsk4d==0] = np.nan
    #    HYrm['vrm'][vmsk4d==0] = np.nan

        # fill in the nans for now
        # this fills the land mask too
        uu[np.isnan(uu)==1]=0
        vv[np.isnan(vv)==1]=0
        #u2[np.isnan(u2)==1]=0
        #v2[np.isnan(v2)==1]=0

        ubar = np.squeeze( np.sum( 0.5 * (uu[:,0:-1,:,:]+uu[:,1:,:,:]) * dz[None,:,None,None], axis=1 ) ) / Hru[None,:,:]
        vbar = np.squeeze( np.sum( 0.5 * (vv[:,0:-1,:,:]+vv[:,1:,:,:]) * dz[None,:,None,None], axis=1 ) ) / Hrv[None,:,:]

        # reapply the land mask...
        umsk2 = np.squeeze( np.isnan( urm[:,0,:,:]))
        vmsk2 = np.squeeze( np.isnan( vrm[:,0,:,:]))
        ubar[umsk2==1] = np.nan
        vbar[vmsk2==1] = np.nan

        HYrm['ubar'] = ubar
        HYrm['vbar'] = vbar
 
 #       for aa in range(NT):
 #           uu = umsk * u2[aa,:,:,:]
 #           vv = vmsk * v2[aa,:,:,:]
 #           um = 0.5 * (uu[0:-1,:,:]+uu[1:,:,:])
 #           vm = 0.5 * (vv[0:-1,:,:]+vv[1:,:,:])
 #           ub = np.sum( um * dz[:,None,None], axis=0) / Hru
 #           vb = np.sum( vm * dz[:,None,None], axis=0) / Hrv
 #           ub[np.isnan(u0)==1] = np.nan
 #           vb[np.isnan(v0)==1] = np.nan
 #           HYrm['ubar'][aa,:,:] = ub
 #           HYrm['vbar'][aa,:,:] = vb
        
    return HYrm


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

    OCN_IC['temp'] = np.zeros((Nz,nlt,nln))
    OCN_IC['pottemp'] = np.zeros((Nz,nlt,nln))
    OCN_IC['salt'] = np.zeros((Nz,nlt,nln))
    OCN_IC['u'] = np.zeros((Nz,nlt,nln-1))
    OCN_IC['v'] = np.zeros((Nz,nlt-1,nln))
    OCN_IC['ubar'] = np.zeros((nlt,nln-1))
    OCN_IC['vbar'] = np.zeros((nlt-1,nln))

    # get the roms z's
    #zrom = get_roms_zlevels(Nz,Vtr,Vst,th_s,th_b,Tcl,eta=0*RMG['h'],RMG['h'])
    #zrom = s_coordinate_4(RMG['h'], theta_b, theta_s, Tcline, Nz, hraw=hraw, eta=0*RMG['h'])    

    hraw = None
    if Vst == 4:
        zrom = s_coordinate_4(hb, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta)
        zrom_u = s_coordinate_4(hb_u, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_u)
        zrom_v = s_coordinate_4(hb_v, th_b , th_s , Tcl , Nz, hraw=hraw, zeta=eta_v)

    zr=np.squeeze(zrom.z_r[0,:,:,:])    
    zr_u=np.squeeze(zrom_u.z_r[0,:,:,:])    
    zr_v=np.squeeze(zrom_v.z_r[0,:,:,:])    

    for aa in range(nlt):
        for bb in range(nln):
            
            fofz = np.squeeze(OCN_R['temp'][i0,:,aa,bb])
            ig = np.argwhere(np.isfinite(fofz))
            fofz2 = fofz[ig]
            Fz = interp1d(-zhy[ig],fofz2,bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))
            OCN_IC['temp'][:,aa,bb] = Fz(zr[:,aa,bb])
            
            fofz = np.squeeze(OCN_R['salt'][i0,:,aa,bb])
            ig = np.argwhere(np.isfinite(fofz))
            fofz2 = fofz[ig]
            Fz = interp1d(-zhy[ig],fofz2,bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
            OCN_IC['salt'][:,aa,bb] = Fz(zr[:,aa,bb])

            if bb < nln-1:
                fofz = np.squeeze(OCN_R['urm'][i0,:,aa,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(-zhy[ig],fofz2,bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1])) 
                uu =  Fz(zr_u[:,aa,bb])                
                OCN_IC['u'][:,aa,bb] = uu
                z2 = np.squeeze(zr_u[:,aa,bb])
                z3 = np.append(z2,eta_u[aa,bb])
                dz = np.diff(z3)
                OCN_IC['ubar'][aa,bb] = np.sum(uu*dz) / hb_u[aa,bb]
            if aa < nlt-1:
                fofz = np.squeeze(OCN_R['vrm'][i0,:,aa,bb])
                ig = np.argwhere(np.isfinite(fofz))
                fofz2 = fofz[ig]
                Fz = interp1d(-zhy[ig],fofz2,bounds_error=False,kind='linear',fill_value=(fofz2[0],fofz2[-1]))  
                vv = Fz(zr_v[:,aa,bb])
                OCN_IC['v'][:,aa,bb] = vv
                z2 = np.squeeze(zr_v[:,aa,bb])
                z3 = np.append(z2,eta_v[aa,bb])
                dz = np.diff(z3)
                OCN_IC['vbar'][aa,bb] = np.sum(vv*dz) / hb_v[aa,bb]

    # ROMS wants potential temperature, not temperature
    # this needs the seawater package, conda install seawater, did this for me
    pdb = -zr # pressure in dbar
    OCN_IC['pottemp'] = seawater.ptmp(OCN_IC['salt'], OCN_IC['temp'],pdb)  

    return OCN_IC


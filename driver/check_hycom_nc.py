import sys, os
from datetime import datetime, timedelta
from pathlib import Path
from time import time, sleep
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps

def check_all_nans(z):
    # this assumes z is a masked array...

    nfin = z.count()
    if nfin == 0:
        allnan = True
    else:
        allnan = False

    return allnan

def interp_hycom_to_roms(ln_h,lt_h,zz_h,Ln_r,Lt_r,msk_r,mf,IMr,Fz):
    # ln_h and lt_h are hycom lon and lat, vectors
    # zz_h is the hycom field that is getting interpolated
    # Ln_r and Lt_r are the roms lon and lat, matrices
    # msk_r is the roms mask, to NaN values
    # mf is the refinement of the hycom  grid before going 
    # nearest neighbor interpolating to ROMS

    allnan = check_all_nans(zz_h)
    if allnan == True:
        zz_r = np.nan * Ln_r
    else:
        Ln_h, Lt_h = np.meshgrid(ln_h, lt_h, indexing='xy')
        X, Y = ll2xy(Ln_h, Lt_h, ln_h.mean(), lt_h.mean())
        # fill in the hycom NaNs with nearest neighbor values
        zz_hf =  extrap_nearest_to_masked(X, Y, zz_h) 

        # interpolate the filled hycom values to the refined grid
        #Fz = RegularGridInterpolator((lt_h, ln_h), zz_hf)
        # change the z values of the interpolator here
        setattr(Fz,'values',zz_h)

        lininterp=1
        if lininterp==1:
            zz_rf = Fz((Lt_r,Ln_r))
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
            zz_rf = zz_hf.flatten()[IMr].reshape(Ln_r.shape)
                
        # now mask zz_r
        zz_r = zz_rf.copy()
        zz_r[msk_r==0] = np.nan

    return zz_r

def checknan(fld):
    """
    A utility function that issues a working if there are nans in fld.
    """
    if np.isnan(fld).sum() > 0:
        print('WARNING: nans in data field')    


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

def hycom_to_dict(fnh,ic):
    if ic=='ic':
        hy = nc.Dataset(fnh)
        # get hycom variables=
        HY=dict()
        HY['lon'] = hy.variables['lon'][:] - 360  # convert from 0:360 to -360:0 format
        HY['lat'] = hy.variables['lat'][:]
        HY['h']  = hy.variables['depth'][:]
        # the zero index below means the first time.
        HY['zeta'] = hy.variables['surf_el'][0,:,:] 
        HY['u'] = hy.variables['water_u'][0,:,:,:]
        HY['v'] = hy.variables['water_v'][0,:,:,:]
        HY['temp'] = hy.variables['water_temp'][0,:,:,:]
        HY['salt'] = hy.variables['salinity'][0,:,:,:]
        HY['time'] = hy.variables['time'][0]
    elif ic=='bc':
        # here we return all times of the data
        hy = nc.Dataset(fnh)
        # get hycom variables=
        HY=dict()
        HY['lon'] = hy.variables['lon'][:] - 360  # convert from 0:360 to -360:0 format
        HY['lat'] = hy.variables['lat'][:]
        HY['h']  = hy.variables['depth'][:]
        HY['zeta'] = hy.variables['surf_el'][:,:,:] 
        HY['u'] = hy.variables['water_u'][:,:,:,:]
        HY['v'] = hy.variables['water_v'][:,:,:,:]
        HY['temp'] = hy.variables['water_temp'][:,:,:,:]
        HY['salt'] = hy.variables['salinity'][:,:,:,:]
        HY['time'] = hy.variables['time'][:]
    return HY

def roms_grid_to_dict(fng):
    gr = nc.Dataset(fng)    
    # get roms grid variables
    RM=dict()
    RM['lon_rho']=gr.variables['lon_rho'][:,:]
    RM['lat_rho']=gr.variables['lat_rho'][:,:]
    RM['lon_u']=gr.variables['lon_u'][:,:]
    RM['lat_u']=gr.variables['lat_u'][:,:]
    RM['lon_v']=gr.variables['lon_v'][:,:]
    RM['lat_v']=gr.variables['lat_v'][:,:]
    RM['h'] =gr.variables['h'][:,:]
    RM['mask_rho'] = gr.variables['mask_rho'][:,:]
    RM['mask_u'] = gr.variables['mask_u'][:,:]
    RM['mask_v'] = gr.variables['mask_v'][:,:]
    RM['angle'] = gr.variables['angle'][:,:]
    RM['angle_u'] = 0.5*(RM['angle'][:,0:-1]+RM['angle'][:,1:])
    RM['angle_v'] = 0.5*(RM['angle'][0:-1,:]+RM['angle'][1:,:])
    return RM

def hycom_to_roms_latlon(HY,RMG):
    # HYcom and RoMsGrid come in as dicts with ROMS variable names    
    # The output of this, HYrm, is a dict with 
    # hycom fields on roms horizontal grid points
    # but hycom z levels.
    # velocity will be on both (lat_u,lon_u)
    # and (lat_v,lon_v).
    
    # do some grid stuff only once
    Lnhi, Lthi = np.meshgrid(HY['lon'],HY['lat'], indexing='xy')
    XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
    XYr = np.array((RMG['lon_rho'].flatten(), RMG['lat_rho'].flatten())).T
    XYru = np.array((RMG['lon_u'].flatten(), RMG['lat_u'].flatten())).T
    XYrv = np.array((RMG['lon_v'].flatten(), RMG['lat_v'].flatten())).T
    # nearest neighbor interpolation is in Imr below...
    Imr = cKDTree(XYin).query(XYr)[1]    
    Imru = cKDTree(XYin).query(XYru)[1]    
    Imrv = cKDTree(XYin).query(XYrv)[1]

    # set up the interpolator now and pass to function
    Fz = RegularGridInterpolator((HY['lat'],HY['lon']),HY['zeta'])
    

    vnames = ['zeta', 'temp', 'salt', 'u', 'v']
    lnhy = HY['lon']
    lthy = HY['lat']
    NR,NC = np.shape(RMG['lon_rho'])
    NZ = len(HY['h'])

    #print(NR)
    #print(NC)
    #print(NZ)
    #print(np.shape(RMG['lon_rho']))
    #print(np.shape(RMG['lon_u']))
    #print(np.shape(RMG['lon_v']))

    HYrm = dict()
    HYrm['zeta'] = np.zeros((NR, NC))
    HYrm['salt'] = np.zeros((NZ, NR, NC))
    HYrm['temp'] = np.zeros((NZ, NR, NC))
    HYrm['u_on_u'] = np.zeros((NZ, NR, NC-1))
    HYrm['v_on_u'] = np.zeros((NZ, NR, NC-1))
    HYrm['u_on_v'] = np.zeros((NZ, NR-1, NC))
    HYrm['v_on_v'] = np.zeros((NZ, NR-1, NC))
    HYrm['lat_rho'] = RMG['lat_rho']
    HYrm['lon_rho'] = RMG['lat_rho']
    HYrm['lat_u'] = RMG['lat_u']
    HYrm['lon_u'] = RMG['lon_u']
    HYrm['lat_v'] = RMG['lat_v']
    HYrm['lon_v'] = RMG['lon_v']
    HYrm['h'] = HY['h']


    rf = 8              # the refinement factor when linearly interpolating hycom data 
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


#fn = '/Users/mspydell/research/FF2024/models/SDPM_mss/SDPM_data/ocn/hycom/hind/raw/2024/01/01/hycom_hind_2024010100.nc'
fnhy = '/Users/mspydell/research/FF2024/models/SDPM_mss/PFM_data/ocn/hycom/hind/raw/tmp/hycom_out.nc'
fngr = '/Users/mspydell/research/FF2024/models/SDPM_mss/PFM_user/grids/GRID_SDTJRE_LV1.nc'

# get the initial condition from hycom
# load the hycom initial conditiondata as a dict
HY = hycom_to_dict(fnhy,'ic')
# load the roms grid as a dict
RMG = roms_grid_to_dict(fngr)

# do some timing test on the interpolations
tt0 = time()
# now nearest neighbor interpolate to the ROMS grid
HYrm = hycom_to_roms_latlon(HY,RMG)
print('linearly interpolated directly to ROMS grid from hycom rectilinear')
print('for zeta, u, v, temp, and salt, for 1 time. all ROMS grid points')
print('it took %0.1f seconds' % (time() - tt0))


sys.exit()

# get the reg grid interpolator object
# this makes interpolating from HY
HYinterp = get_reggridinterp_object(HY['lat'],HY['lon'])

# lets play with regulargrid interpolator...

zz=getattr(Fz,'values')
fig, axs = plt.subplots()
cax = axs.matshow(zz, cmap=plt.cm.turbo)
axs.invert_yaxis()
fig.colorbar(cax)

# time to change the z values of the interpolator
setattr(Fz,'values',HY['u'][0,:,:])
zz2=getattr(Fz,'values')
fig, axs = plt.subplots()
cax = axs.matshow(zz2, cmap=plt.cm.turbo)
axs.invert_yaxis()
fig.colorbar(cax)




# take hycom data (in the file fhny) and put it 
# on the roms grid (in the file fngr)
# depths are still hycom
# this is done for one time only.



hy = nc.Dataset(fnhy)
gr = nc.Dataset(fngr)

# get hycom variables
lnh = hy.variables['lon'][:] - 360  # convert from 0:360 to -360:0 format
lth = hy.variables['lat'][:]
zh  = hy.variables['depth'][:]
ssh = hy.variables['surf_el'][0,:,:] 
uh = hy.variables['water_u'][0,:,:,:]
vh = hy.variables['water_v'][0,:,:,:]

# get roms variables. angle is in radians
lnr=gr.variables['lon_rho'][:,:]
ltr=gr.variables['lat_rho'][:,:]
lnu=gr.variables['lon_u'][:,:]
ltu=gr.variables['lat_u'][:,:]
lnv=gr.variables['lon_v'][:,:]
ltv=gr.variables['lat_v'][:,:]
mskr = gr.variables['mask_rho'][:,:]
msku = gr.variables['mask_u'][:,:]
mskv = gr.variables['mask_v'][:,:]
angr = gr.variables['angle'][:,:]
angu = 0.5*(angr[:,0:-1]+angr[:,1:])
angv = 0.5*(angr[0:-1,:]+angr[1:,:])


# plot the angles and the interpolated angles
fig, axs = plt.subplots(nrows=1, ncols=3)
levels = np.arange(0,1, 0.05)
cmap = plt.get_cmap('turbo')
plt.set_cmap(cmap)
cset1 = axs[0].contourf(lnr,ltr,angr,levels)
axs[0].set_title('roms angle at rho [rads]')
axs[0].set_xlabel('lon')
axs[0].set_ylabel('lat')
cset1 = axs[1].contourf(lnu,ltu,angu,levels)
axs[1].set_title('roms angle at u [rads]')
axs[1].set_xlabel('lon')
cset1 = axs[2].contourf(lnv,ltv,angv,levels)
axs[2].set_title('roms angle at v [rads]')
axs[2].set_xlabel('lon')

Ln, Lt = np.meshgrid(lnh, lth, indexing='xy')
# plot u top from hycom
ut = uh[0,:,:]
ub = uh[-5,:,:]
levelsu = np.arange(-0.6,0.6+0.05, 0.05)
fig, axs = plt.subplots(nrows=1, ncols=2)
cmap = plt.get_cmap('bwr')
plt.set_cmap(cmap)
cset1 = axs[0].contourf(Ln,Lt,ut,levelsu)
#cset2 = axs[0].contour(Ln,Lt,ut, cset1.levels, colors='k', linewidths=1)
axs[0].set_title('hycom u top [m/s]')
axs[0].set_xlabel('lon')
axs[0].set_ylabel('lat')
fig.colorbar(cset1, ax=axs[0])

levelsu = np.arange(-0.1,0.1+0.0125, 0.0125)
cset1 = axs[1].contourf(Ln,Lt,ub,levelsu)
#cset2 = axs[0].contour(Ln,Lt,ut, cset1.levels, colors='k', linewidths=1)
axs[1].set_title('hycom u z=-2000 m [m/s]')
axs[1].set_xlabel('lon')
fig.colorbar(cset1, ax=axs[1])


vt = vh[0,:,:]
vb = vh[-5,:,:]
levelsu = np.arange(-0.6,0.6+0.05, 0.05)
fig, axs = plt.subplots(nrows=1, ncols=2)
cmap = plt.get_cmap('bwr')
plt.set_cmap(cmap)
cset1 = axs[0].contourf(Ln,Lt,vt,levelsu)
#cset2 = axs[0].contour(Ln,Lt,ut, cset1.levels, colors='k', linewidths=1)
axs[0].set_title('hycom v top [m/s]')
axs[0].set_xlabel('lon')
axs[0].set_ylabel('lat')
fig.colorbar(cset1, ax=axs[0])

levelsu = np.arange(-0.1,0.1+0.0125, 0.0125)
cset1 = axs[1].contourf(Ln,Lt,vb,levelsu)
#cset2 = axs[0].contour(Ln,Lt,ut, cset1.levels, colors='k', linewidths=1)
axs[1].set_title('hycom v z=-2000 m [m/s]')
axs[1].set_xlabel('lon')
fig.colorbar(cset1, ax=axs[1])

# rotate top (u,v) velocities onto the u, v grids
uonu = interp_hycom_to_roms(lnh,lth,ut,lnu,ltu,msku,5)
vonu = interp_hycom_to_roms(lnh,lth,vt,lnu,ltu,msku,5)
uonv = interp_hycom_to_roms(lnh,lth,ut,lnv,ltv,mskv,5)
vonv = interp_hycom_to_roms(lnh,lth,vt,lnv,ltv,mskv,5)

ur = uonu * np.cos(angu) + vonu * np.sin(angu)
vru = -uonu * np.sin(angu) + vonu * np.cos(angu)
vr = vonv * np.cos(angv) - uonv * np.sin(angv)
# (ur,vr) is now the roms velocity at u grid and v grid respectively

# plot the non rotated and rotated velocities
fig, axs = plt.subplots(nrows=2, ncols=2)
q = axs[0,0].quiver(Ln[::2,::2],Lt[::2,::2],ut[::2,::2],vt[::2,::2])
axs[0,0].set_title('hycom vel')
axs[0,0].set_xlabel('lon')
axs[0,0].set_ylabel('lat')
q = axs[0,1].quiver(lnu[::5,::5],ltu[::5,::5],uonu[::5,::5],vonu[::5,::5])
axs[0,1].set_title('hycom vel on roms')
axs[0,1].set_xlabel('lon')
axs[0,1].set_ylabel('lat')
q = axs[1,0].quiver(lnu[::5,::5],ltu[::5,::5],ur[::5,::5],vru[::5,::5])
axs[1,0].set_title('roms vel (rotated hycom)')
axs[1,0].set_xlabel('lon')
axs[1,0].set_ylabel('lat')



# done
sys.exit()

# plot ssh from hycom
levels = np.arange(-0.1,0.4, 0.025)
fig, axs = plt.subplots(nrows=1, ncols=2)

cset1 = axs[0].contourf(Ln,Lt,ssh,levels)
cset2 = axs[0].contour(Ln,Lt,ssh, cset1.levels, colors='k', linewidths=1)
axs[0].set_title('hycom ssh [m]')
axs[0].set_xlabel('lon')
axs[0].set_ylabel('lat')

fig.colorbar(cset1, ax=axs[0])

# plot filled ssh from hycom
X, Y = ll2xy(Ln, Lt, lnh.mean(), lth.mean())
sshf =  extrap_nearest_to_masked(X, Y, ssh) 
cset1 = axs[1].contourf(Ln,Lt,sshf,levels)
cset2 = axs[1].contour(Ln,Lt,sshf, cset1.levels, colors='k', linewidths=1)
axs[1].set_title('hycom ssh nan nearest filled [m]')
axs[1].set_xlabel('lon')
fig.colorbar(cset1, ax=axs[1])

# interp hycom to finer grid
# use mf as the factor to increase the hycom resolution
# anything bigger than 5 seems OK for going from hycom --> LV1
mf = 5
lnhi = np.linspace(lnh[0],lnh[-1],mf*len(lnh))
lthi = np.linspace(lth[0],lth[-1],mf*len(lth))
Lnhi, Lthi = np.meshgrid(lnhi,lthi, indexing='xy')
interp = RegularGridInterpolator((lth, lnh), sshf)
sshfi=interp((Lthi,Lnhi))
fig, axs = plt.subplots(nrows=1, ncols=3)
cset1 = axs[0].contourf(Lnhi,Lthi,sshfi,levels)
cset2 = axs[0].contour(Lnhi,Lthi,sshfi, cset1.levels, colors='k')
axs[0].set_title('hycom ssh filled fine  (' + str(mf) + 'x) [m]')
axs[0].set_xlabel('lon')
axs[0].set_ylabel('lat')
fig.colorbar(cset1,ax=axs[0])

# interp to roms from the fine hycom ssh
XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
XYr = np.array((lnr.flatten(), ltr.flatten())).T
# nearest neighbor interpolation from XYin to XYr is done below...
IMr = cKDTree(XYin).query(XYr)[1]    
sshr = sshfi.flatten()[IMr].reshape(msk.shape)
# we need to mask the roms sshr
# make sure not to change sshr, want to keep it unchanged
# without the .copy() sshrm is a pointer to sshr so sshr
# will change when sshrm is altered
sshrm = sshr.copy()
sshrm[msk==0] = np.nan

#img = axs[1].imshow(Ln,Lt,sshf,interpolation='linear',alpha=0.5)
#,interpolation='linear',alpha=0.5)
levels2 = np.arange(0,0.28, 0.02)
cset1 = axs[1].contourf(lnr,ltr,sshr,levels)
cset2 = axs[1].contour(lnr,ltr,sshr, cset1.levels, colors='k')
axs[1].set_title('hycom ssh filled fine (' + str(mf) + 'x) to roms L1 (nearest) [m]')
axs[1].set_xlabel('lon')
axs[1].set_xlim([np.min(lnh), np.max(lnh)])
axs[1].set_ylim([np.min(lth), np.max(lth)])
fig.colorbar(cset1,ax=axs[1])

cset1 = axs[2].contourf(lnr,ltr,sshrm,levels)
cset2 = axs[2].contour(lnr,ltr,sshrm, cset1.levels, colors='k')
axs[2].set_title('roms ssh L1 masked [m]')
fig.colorbar(cset1,ax=axs[2])
axs[2].set_xlabel('lon')
axs[2].set_xlim([np.min(lnh), np.max(lnh)])
axs[2].set_ylim([np.min(lth), np.max(lth)])

mf = 1
lnhi = np.linspace(lnh[0],lnh[-1],mf*len(lnh))
lthi = np.linspace(lth[0],lth[-1],mf*len(lth))
Lnhi, Lthi = np.meshgrid(lnhi,lthi, indexing='xy')
interp = RegularGridInterpolator((lth, lnh), sshf)
sshfi=interp((Lthi,Lnhi))
XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
IMr = cKDTree(XYin).query(XYr)[1]    
sshr = sshfi.flatten()[IMr].reshape(msk.shape)
sshrm = sshr.copy()
sshrm[msk==0] = np.nan
fig, axs = plt.subplots(nrows=1, ncols=3)
levels = np.arange(0, .25, 0.025)
cset1 = axs[0].contourf(lnr,ltr,sshrm,levels)
cset2 = axs[0].contour(lnr,ltr,sshrm, cset1.levels, colors='k')
axs[0].set_title('roms ssh (' + str(mf) + 'x)')
axs[0].set_xlabel('lon')
axs[0].set_ylabel('lat')
axs[0].set_xlim([-121 ,-119])
axs[0].set_ylim([33.6,34])

mf = 5
lnhi = np.linspace(lnh[0],lnh[-1],mf*len(lnh))
lthi = np.linspace(lth[0],lth[-1],mf*len(lth))
Lnhi, Lthi = np.meshgrid(lnhi,lthi, indexing='xy')
interp = RegularGridInterpolator((lth, lnh), sshf)
sshfi=interp((Lthi,Lnhi))
XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
IMr = cKDTree(XYin).query(XYr)[1]    
sshr = sshfi.flatten()[IMr].reshape(msk.shape)
sshrm = sshr.copy()
sshrm[msk==0] = np.nan
cset1 = axs[1].contourf(lnr,ltr,sshrm,levels)
cset2 = axs[1].contour(lnr,ltr,sshrm, cset1.levels, colors='k')
axs[1].set_title('roms ssh (' + str(mf) + 'x)')
axs[1].set_xlabel('lon')
axs[1].set_xlim([-121 ,-119])
axs[1].set_ylim([33.6,34])

mf = 10
lnhi = np.linspace(lnh[0],lnh[-1],mf*len(lnh))
lthi = np.linspace(lth[0],lth[-1],mf*len(lth))
Lnhi, Lthi = np.meshgrid(lnhi,lthi, indexing='xy')
interp = RegularGridInterpolator((lth, lnh), sshf)
sshfi=interp((Lthi,Lnhi))
XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
IMr = cKDTree(XYin).query(XYr)[1]    
sshr = sshfi.flatten()[IMr].reshape(msk.shape)
sshrm = sshr.copy()
sshrm[msk==0] = np.nan
cset1 = axs[2].contourf(lnr,ltr,sshrm,levels)
cset2 = axs[2].contour(lnr,ltr,sshrm, cset1.levels, colors='k')
axs[2].set_title('roms ssh (' + str(mf) + 'x)')
axs[2].set_xlabel('lon')
axs[2].set_xlim([-121 ,-119])
axs[2].set_ylim([33.6,34])

mf = 5
lnhi = np.linspace(lnh[0],lnh[-1],mf*len(lnh))
lthi = np.linspace(lth[0],lth[-1],mf*len(lth))
Lnhi, Lthi = np.meshgrid(lnhi,lthi, indexing='xy')
interp = RegularGridInterpolator((lth, lnh), sshf,method='cubic')
sshfi=interp((Lthi,Lnhi))
XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
IMr = cKDTree(XYin).query(XYr)[1]    
sshr = sshfi.flatten()[IMr].reshape(msk.shape)
sshrm = sshr.copy()
sshrm[msk==0] = np.nan
fig, axs = plt.subplots(nrows=1, ncols=2)
cset1 = axs[0].contourf(lnr,ltr,sshrm,levels)
cset2 = axs[0].contour(lnr,ltr,sshrm, cset1.levels, colors='k')
axs[0].set_title('roms ssh (' + str(mf) + 'x)')
axs[0].set_xlabel('lon')
axs[0].set_ylabel('lat')
axs[0].set_xlim([-121 ,-119])
axs[0].set_ylim([33.6,34])




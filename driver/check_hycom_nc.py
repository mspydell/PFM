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


#fn = '/Users/mspydell/research/FF2024/models/SDPM_mss/SDPM_data/ocn/hycom/hind/raw/2024/01/01/hycom_hind_2024010100.nc'
fnhy = '/Users/mspydell/research/FF2024/models/SDPM_mss/SDPM_data/ocn/hycom/hind/raw/tmp/hycom_out.nc'
fngr = '/Users/mspydell/research/FF2024/models/SDPM_mss/SDPM/grids/GRID_SDTJRE_LV1.nc'

hy = nc.Dataset(fnhy)
gr = nc.Dataset(fngr)

#
lnh = hy.variables['lon'][:] - 360  # convert from 0:360 to -360:0 format
lth = hy.variables['lat'][:]
ssh = hy.variables['surf_el'][0,:,:] 

# plot ssh from hycom
Ln, Lt = np.meshgrid(lnh, lth, indexing='xy')
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
lnr=gr.variables['lon_rho'][:,:]
ltr=gr.variables['lat_rho'][:,:]
msk = gr.variables['mask_rho'][:,:]
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

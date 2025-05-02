import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

import cartopy.crs as ccrs
from datetime import datetime

import geojsoncontour
import geojson

# Defining plotting_functions inside non-imported environment
def get_depth_indices(hb,lats,depth):
    # returns the x indices wherre hb crosses depth
    # and returns the depth at that x index.
    ny,nx = np.shape(hb)
    #print(ny,nx)
    ixs = np.zeros((ny,len(depth)),dtype=int)
    hbs = np.zeros((ny,len(depth)))
    lat2 = np.zeros((ny,len(depth)))
    cnt = 0
    for dd in depth:
        for aa in np.arange(ny):
            hofx = hb[aa,:] - dd # we want to know when this crosses zero
            zero_cross = np.where(np.diff(np.sign(hofx))) # this is the index of zero crossing
            ixs[aa,cnt] = int( np.min(zero_cross) )
            hbs[aa,cnt] = hb[aa,ixs[aa,cnt]]
            lat2[aa,cnt] = lats[aa,ixs[aa,cnt]]
        cnt=cnt+1

    lat2 = np.squeeze( lat2[:,0] )

    # remove the estuary...
    ltb_mn = 32.5536 # these are the latitudes where the estuary messes things up...
    ltb_mx = 32.5546

    ibad =  (lat2>ltb_mn) & (lat2<ltb_mx)

    for bb in np.arange(len(depth)):
        yy = np.interp(lat2[ibad],lat2[~ibad],ixs[~ibad,bb])
        yy = np.round(yy)
        yy.astype(int)
        ixs[ibad,bb] = yy


    ixs.astype(int)
    #print(ixs[ibad,0])
    return ixs, hbs

def get_alongshore_distance_etc(ixs,iys,fn_grd):

    DD = dict()

    DD['ixs'] = ixs
    DD['iys'] = iys

    ny, nd = np.shape(ixs)


    with Dataset(fn_grd) as ds:
        lats = ds.variables['lat_rho'][:,:] # the data to extract depth from
        lons = ds.variables['lon_rho'][:,:]
        xx   = ds.variables['x_rho'][:,:]
        yy   = ds.variables['y_rho'][:,:]
        hh   = ds.variables['h'][:,:]

    lts = np.zeros((ny,nd))
    lns = np.zeros((ny,nd))
    hs  = np.zeros((ny,nd))
    XX  = np.zeros((ny,nd))
    YY  = np.zeros((ny,nd))

    for aa in np.arange(nd):
        for bb in np.arange(ny):
            XX[bb,aa] = xx[iys[bb],ixs[bb]]
            YY[bb,aa] = yy[iys[bb],ixs[bb]]
            lts[bb,aa] = lats[iys[bb],ixs[bb]]
            lns[bb,aa] = lons[iys[bb],ixs[bb]]
            hs[bb,aa]  = hh[iys[bb],ixs[bb]]


    dx = np.diff(XX,axis=0)
    dy = np.diff(YY,axis=0)
    l  = np.cumsum( np.sqrt( np.square(dx) + np.square(dy) ),axis=0 )

    l = np.insert(l,0,0,axis=0) # this is now the distance from the starting iy0.
    DD['l'] = l
    DD['lat'] = lts
    DD['lon'] = lns
    DD['hb'] = hs

    return DD

def get_ocean_times_from_ncfile(fn):
    # this function returns the times inside the fn.nc file as a numpy array of datetimes

    with Dataset(fn, 'r') as nc_file:
        # Get the time variable
        time_var = nc_file.variables['ocean_time']

        # Get the time units and calendar attributes
        time_units = time_var.units
        time_calendar = time_var.calendar if hasattr(time_var, 'calendar') else 'standard'

        # Convert the time values to datetime objects
        datetime_values = num2date(time_var[:], units=time_units, calendar=time_calendar)

        # If you need a NumPy array of datetime objects:
        datetime_array = np.array(datetime_values)

    return datetime_array

def roms_grid_to_dict(fng):
    gr = Dataset(fng)
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

    # this is hard coded here but should be read in from get_PFM_info.py
    # these are the vertical numbers for LV1 !!!
    RM['Nz'] = 40                            # number of vertical rho points
    RM['Vtransform']=2                       # transformation equation
    RM['Vstretching'] = 4                    # stretching function
    RM['THETA_S'] = 8.0                      # surface stretching parameter
    RM['THETA_B'] = 3.0                      # bottom  stretching parameter
    RM['TCLINE'] = 50.0                      # critical depth (m)
    RM['hc'] = 50.0                          # not sure the difference with tcline

    return RM

# IFB added
def get_site_forecasts_table(fn_grd, fn_his, var = 'dye_01', save_to_file = False):
    # grid file data grab
    ix0 = 75  # this is the starting index to look to the right from
    iymn = 0
    iymx = 1060
    iys = np.arange(iymn, iymx, 1)
    hb0 = [1.25]  # this is the depth to get the indices from
    with Dataset(fn_grd) as ds:
        hb = ds.variables['h'][iys, ix0:]  # the data to extract depth from
        hb_og = ds.variables['h'][:, :]
        lats = ds.variables['lat_rho'][iys, ix0:]
        iy, ix = np.shape(hb_og)

    # history file data grab
    with Dataset(fn_his) as his:
        times = his.variables['ocean_time']
        times2 = num2date(times[:], times.units)
        times2 = np.array([datetime(year=date.year, month=date.month, day=date.day,
                                    hour=date.hour, minute=date.minute, second=date.second) for date in times2])

    ixs, _ = get_depth_indices(hb, lats, hb0)
    ixs = ixs + ix0  # must return the starting point
    ny = len(ixs)
    nt = len(times2)  # these are the times for the history file

    dye_vars = np.zeros((nt, ny))
    for aa in np.arange(0, nt, 1):
        with Dataset(fn_his) as his:
            hh = his.variables['h'][iys, ix0:] + np.squeeze(his.variables['zeta'][aa, iys, ix0:])
            dye = his.variables[var][aa, -1, :, :]  # get the surface
            ixs, _ = get_depth_indices(hh, lats, hb0)
            ixs = ixs + ix0
            for bb in np.arange(ny):
                dye_vars[aa, bb] = np.squeeze(dye[iys[bb], ixs[bb]])

    # PTJ, border, TJRE, IB pier, Silver Strand, HdC
    ln_lab = ['PTJ', 'border', 'TJRE', 'IB pier', 'Silver Strand', 'HdC']
    lts0 = [32.52, 32.534, 32.552, 32.58, 32.625, 32.678]
    ipts = np.zeros((len(lts0)), dtype=int)

    DD = get_alongshore_distance_etc(ixs, iys, fn_grd)

    for aa in np.arange(len(lts0)):
        dlt = DD['lat'] - lts0[aa]
        ipts[aa] = np.argmin(np.square(dlt))

    lD = np.log10(dye_vars)[:, ipts[0:len(ipts)]]

    df_dye = pd.DataFrame(lD, columns=ln_lab[0:len(ipts)])

    start_time = times2[0]
    df_dye.insert(0, 'time', times2)

    if save_to_file:
        df_dye.to_csv(f'./sccoos_output/{var}_forecast_{start_time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)

    return(df_dye)

# IFB added
def get_LV4_contour_geojson(fn_grd, fn_his, var = 'dye_01', contour_interval = 0.1, save_to_file = False):
    # LV4 dye_01
    contour_min = -6
    contour_max = 0

    # plot_lv4_coawst_his_v2
    Iz = -1
    It = 0

    time = get_ocean_times_from_ncfile(fn_his)
    dtime = time[-1] - time[0]

    RMG = roms_grid_to_dict(fn_grd)
    lt = RMG['lat_rho'][:]
    ln = RMG['lon_rho'][:]

    It = int(It)
    Iz = int(Iz)
    with Dataset(fn_his, 'r') as his_ds:
        times = his_ds.variables['ocean_time']
        times2 = num2date(times[:], times.units)
        times2 = np.array([datetime(year=date.year, month=date.month, day=date.day,
                                    hour=date.hour, minute=date.minute, second=date.second) for date in times2])
        D = his_ds.variables[var][It, Iz, :, :]
        D = np.log10(D)

    plevs = np.arange(contour_min, contour_max, contour_interval)

    fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap = plt.get_cmap('magma_r')
    cset = ax.contourf(ln, lt, D, plevs, cmap=cmap, extend="both", vmin=-6, vmax=0, transform=ccrs.PlateCarree())

    start_time = times2[0]

    contour_geojson = geojsoncontour.contourf_to_geojson(
        contourf=cset,
        ndigits=8
    )

    if save_to_file:
        with open(f'./sccoos_output/{var}_contours_{start_time.strftime("%Y%m%d-%H%M%S")}.geojson', 'w') as f:
            geojson.dump(contour_geojson, f)

    return(contour_geojson)
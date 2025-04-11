import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import netCDF4 as nc
from netCDF4 import Dataset, num2date
import cartopy.feature as cfeature
import sys
from scipy.interpolate import griddata
from get_PFM_info import get_PFM_info
import grid_functions as grdfuns
import pickle
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import pandas as pd
import requests
import subprocess
import os


def plot_roms_box(axx, RMG):
    xr1 = RMG['lon_rho'][0, :]
    yr1 = RMG['lat_rho'][0, :]
    xr2 = RMG['lon_rho'][:, 0]
    yr2 = RMG['lat_rho'][:, 0]
    xr3 = RMG['lon_rho'][-1, :]
    yr3 = RMG['lat_rho'][-1, :]
    xr4 = RMG['lon_rho'][:, -1]
    yr4 = RMG['lat_rho'][:, -1]
    axx.plot(xr1, yr1, 'k-', linewidth=.5)
    axx.plot(xr2, yr2, 'k-', linewidth=.5)
    axx.plot(xr3, yr3, 'k-', linewidth=.5)
    axx.plot(xr4, yr4, 'k-', linewidth=.5)

def plot_roms_coastline(axx, RMG):
    axx.contour(RMG['lon_rho'], RMG['lat_rho'], RMG['h'], levels=[5, 10], colors='black')

#Function for timestamp
def extract_timestamp(ATM):
    """
    Extracts the timestamp from the ATM data dictionary.

    Parameters:
    ATM (dict): The atmospheric data dictionary.

    Returns:
    str: Formatted timestamp string.
    """
    base_time = datetime(1999, 1, 1)
    time_offset = ATM['ocean_time'][0]
    timestamp = base_time + timedelta(days=time_offset)
    return timestamp.strftime('%Y%m%d_%H%M%S')

# ATM Fields Plotting Function
def plot_atm_fields(show=False,fields_to_plot=None, forecast_hour=None):
    """
    Plot specified fields from the ATM dataset with timestamps and product names, and save them as PNG files.
    
    Parameters:
    ATM (dict): The atmospheric data dictionary.
    RMG (dict): The ROMS grid data dictionary.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    IMPORTANT: we might need to keep changing the plevs and the number of ticks as well as number of levels manually for now!
    """

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

    fname_atm  = PFM['lv1_forc_dir'] + '/' + PFM['atm_tmp_pckl_file']
    with open(fname_atm,'rb') as fp:
        ATM = pickle.load(fp)

    timestamp = extract_timestamp(ATM)
    lon = ATM['lon']
    lat = ATM['lat']
    ocean_time = ATM['ocean_time']
    start_time = datetime(1999, 1, 1) + timedelta(days=float(ocean_time[0]))
    # forecast_hours = (ocean_time - ocean_time[0]) * 24  # Convert days to hours

    # this thing takes care of the forecast hour passed by the user, if no forecast hour is mentioned it plots for the first step (0th)
    if forecast_hour is not None:
        # Find the closest time step to the specified forecast hour
        closest_idx = np.argmin(np.abs((ocean_time - ocean_time[0]) * 24 - forecast_hour))
        forecast_hours = (ocean_time[closest_idx] - ocean_time[0]) * 24
        if closest_idx >= len(ocean_time):
            raise ValueError(f"forecast_hour {forecast_hour} is out of range for the available time steps.")
    else:
        closest_idx = 0  # Default to 0 forecast hours if not specified
        forecast_hours = 0
    
    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'pressure', 'temperature', 'lw_radiation', 'rain', 'humidity', 'swrad']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot

    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('turbo')
        plt.set_cmap(cmap)

        if field == 'velocity':
            plevs = np.arange(0, 16, 0.1)
            U, V = ATM['Uwind'][closest_idx, :, :], ATM['Vwind'][closest_idx, :, :]
            magnitude = np.sqrt(U**2 + V**2)
            cset = ax.contourf(lon, lat, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon[::10], lat[4::10], U[4::10, ::10], V[4::10, ::10], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 16, 5))
            ax.set_title('10 m velocity [m/s, every 10 grid points]')
        
        elif field == 'pressure':
            plevs = np.arange(1000, 1026, 0.1)
            cset = ax.contourf(lon, lat, ATM['Pair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(1000, 1026, 5))
            ax.set_title('Surface Pressure [Pa]')
        
        elif field == 'temperature':
            plevs = np.arange(8, 65, 0.1)
            cset = ax.contourf(lon, lat, ATM['Tair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(8, 65, 8))
            ax.set_title('Surface Temperature [K]')
        
        elif field == 'lw_radiation':
            plevs = np.arange(260, 520, 15)
            cset = ax.contourf(lon, lat, ATM['lwrad_down'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(260, 520, 50))
            ax.set_title('Long Wave Radiation Down [W/m^2]')
        
        elif field == 'rain':
            plevs = np.arange(0, 0.0035, 0.0001)
            cset = ax.contourf(lon, lat, ATM['rain'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 0.0035, 0.0007))
            ax.set_title('Precipitation Rate [kg/m^2/s]')
        
        elif field == 'humidity':
            plevs = np.arange(5, 102, 1)
            cset = ax.contourf(lon, lat, ATM['Qair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(5, 101, 20))
            ax.set_title('Surface Humidity [%]')
        
        elif field == 'swrad':
            plevs = np.arange(0, 601, 10)
            cset = ax.contourf(lon, lat, ATM['swrad'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(100, 601, 100))
            ax.set_title('Short Wave Radiation Down [W/m^2]')
        
        plot_roms_box(ax, RMG)
        # plot_roms_coastline(ax, RMG)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)
        ax.set_aspect(aspect='auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: nam_nest | Forecast Hour: {forecast_hours:.1f}'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
    
        output_dir = PFM['lv1_plot_dir']
        filename = f'{output_dir}/{timestamp}_nam_nest_ATM_{field}_hour_{int(forecast_hours)}.png'
        plt.savefig(filename, dpi=300)
        if show is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

def plot_atm_r_fields(ATM_R, RMG, PFM, show=False, fields_to_plot=None, forecast_hour = None, flag=True):
    """
    Plot specified fields from the ATM_R dataset with timestamps and product names, and save them as PNG files.
    
    Parameters:
    ATM_R (dict): The atmospheric data dictionary on ROMS grid.
    RMG (dict): The ROMS grid data dictionary.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    """
    timestamp = extract_timestamp(ATM_R)
    lon_r = ATM_R['lon']
    lat_r = ATM_R['lat']
    ocean_time = ATM_R['ocean_time']
    start_time = datetime(1999, 1, 1) + timedelta(days=float(ocean_time[0]))
    # forecast_hours = (ocean_time - ocean_time[0]) * 24  # Convert days to hours

     # this thing takes care of the forecast hour passed by the user, if no forecast hour is mentioned it plots for the first step (0th)
    if forecast_hour is not None:
        closest_idx = np.argmin(np.abs((ocean_time - ocean_time[0]) * 24 - forecast_hour))
        forecast_hours = (ocean_time[closest_idx] - ocean_time[0]) * 24
        if closest_idx >= len(ocean_time):
            raise ValueError(f"forecast_hour {forecast_hour} is out of range for the available time steps.")
    else:
        closest_idx = 0  # Default to 0 forecast hours if not specified
        forecast_hours = 0
    
    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'pressure', 'temperature', 'lw_radiation', 'rain', 'humidity', 'swrad']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot


    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('turbo')
        plt.set_cmap(cmap)

        if field == 'velocity':
            plevs = np.arange(0, 16, 0.1)
            U, V = ATM_R['Uwind'][closest_idx, :, :], ATM_R['Vwind'][closest_idx, :, :]
            magnitude = np.sqrt(U**2 + V**2)
            cset = ax.contourf(lon_r, lat_r, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon_r[::10, ::10], lat_r[::10, ::10], U[::10, ::10], V[::10, ::10], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 16, 5))
            ax.set_title('10 m velocity [m/s, on ROMS grid]')
        
        elif field == 'pressure':
            plevs = np.arange(1000, 1026, 0.1)
            cset = ax.contourf(lon_r, lat_r, ATM_R['Pair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(1000, 1026, 5))
            ax.set_title('Surface Pressure [Pa]')
        
        elif field == 'temperature':
            plevs = np.arange(8, 41, 0.1)
            cset = ax.contourf(lon_r, lat_r, ATM_R['Tair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(8, 41, 8))
            ax.set_title('Surface Temperature [K]')
        
        elif field == 'lw_radiation':
            plevs = np.arange(260, 520, 15)
            cset = ax.contourf(lon_r, lat_r, ATM_R['lwrad_down'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(260, 520, 50))
            ax.set_title('Long Wave Radiation Down [W/m^2]')
        
        elif field == 'rain':
            plevs = np.arange(0, 0.0035, 0.0001)
            cset = ax.contourf(lon_r, lat_r, ATM_R['rain'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 0.0035, 0.0007))
            ax.set_title('Precipitation Rate [kg/m^2/s]')
        
        elif field == 'humidity':
            plevs = np.arange(5, 102, 1)
            cset = ax.contourf(lon_r, lat_r, ATM_R['Qair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(5, 102, 20))
            ax.set_title('Surface Humidity [%]')
        
        elif field == 'swrad':
            plevs = np.arange(0, 601, 10)
            cset = ax.contourf(lon_r, lat_r, ATM_R['swrad'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(100, 601, 100))
            ax.set_title('Short Wave Radiation Down [W/m^2]')
        
        plot_roms_box(ax, RMG)
        plot_roms_coastline(ax, RMG)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)
        ax.set_aspect(aspect='auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon_r), np.max(lon_r), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat_r), np.max(lat_r), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: {PFM['atm_model']} | Forecast Hour: {forecast_hours:.1f}'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        output_dir = PFM['lv1_plot_dir']
        if flag is True:
            filename = f'{output_dir}/{timestamp}_nam_nest_ATM_R_{field}_hour_{int(forecast_hours)}.png'
            plt.savefig(filename, dpi=300)
            if show is True:
                plt.tight_layout()
                plt.show()
            else:
                plt.close()
        else:
            filename = f'{output_dir}/{timestamp}_nam_nest_ATM_R_{field}_exported_hour_{int(forecast_hours)}.png'
            plt.savefig(filename, dpi=300)
            if show is True:
                plt.tight_layout()
                plt.show()
            else:
                plt.close()

# For both ATM and ATM_R fields
def plot_all_fields_in_one(lv, show=False, fields_to_plot=None, forecast_hour=None):
    """
    Plot specified fields from both the ATM and ATM_R datasets with timestamps and product names, and save them in separate PNG files.
    
    Parameters:
    ATM (dict): The atmospheric data dictionary.
    ATM_R (dict): The atmospheric data dictionary on ROMS grid.
    RMG (dict): The ROMS grid data dictionary.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    """

    PFM=get_PFM_info()
    fname_atm  = PFM['lv1_forc_dir'] + '/' + PFM['atm_tmp_pckl_file']
    with open(fname_atm,'rb') as fp:
        ATM = pickle.load(fp)

    if lv == '1':
        output_dir = PFM['lv1_plot_dir']
        RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
        fname_r = PFM['lv1_forc_dir'] + '/' + PFM['atm_tmp_LV1_pckl_file']
    elif lv == '2':
        output_dir = PFM['lv2_plot_dir']
        RMG = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        fname_r = PFM['lv2_forc_dir'] + '/' + PFM['atm_tmp_LV2_pckl_file']
    elif lv == '3':
        output_dir = PFM['lv3_plot_dir']
        RMG = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        fname_r = PFM['lv3_forc_dir'] + '/' + PFM['atm_tmp_LV3_pckl_file']
    else:
        output_dir = PFM['lv4_plot_dir']
        RMG = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
        fname_r = PFM['lv4_forc_dir'] + '/' + PFM['atm_tmp_LV4_pckl_file']

    with open(fname_r,'rb') as fp:
        ATM_R = pickle.load(fp)


    timestamp = extract_timestamp(ATM)
    lon = ATM['lon']
    lat = ATM['lat']
    lon_r = ATM_R['lon']
    lat_r = ATM_R['lat']
    ocean_time = ATM['ocean_time']
    start_time = datetime(1999, 1, 1) + timedelta(days=float(ocean_time[0]))
    
    if forecast_hour is not None:
        closest_idx = np.argmin(np.abs((ocean_time - ocean_time[0]) * 24 - forecast_hour))
        forecast_hours = (ocean_time[closest_idx] - ocean_time[0]) * 24
        if closest_idx >= len(ocean_time):
            raise ValueError(f"forecast_hour {forecast_hour} is out of range for the available time steps.")
    else:
        closest_idx = 0  # Default to 0 forecast hours if not specified
        forecast_hours = 0
    
    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'pressure', 'temperature', 'lw_radiation', 'rain', 'humidity', 'swrad']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot

    cmap = plt.get_cmap('turbo')
    plt.set_cmap(cmap)

    for field in fields_to_plot:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        ax1 = axs[0]
        ax2 = axs[1]

        if field == 'velocity':
            U, V = ATM['Uwind'][closest_idx, :, :], ATM['Vwind'][closest_idx, :, :]
            magnitude = np.sqrt(U**2 + V**2)
            #plevs = np.arange(np.floor(np.min(magnitude)), np.ceil(np.max(magnitude)), 0.1)
#            cset1 = ax1.contourf(lon, lat, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset1 = ax1.contourf(lon, lat, magnitude, 50, cmap=cmap, transform=ccrs.PlateCarree())
            ax1.quiver(lon[::10], lat[::10], U[::10, ::10], V[::10, ::10], transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            #cbar1.set_ticks(np.arange(closest_idx, 16, 5))
            ax1.set_title('10 m velocity [m/s, ATM]')

            U_R, V_R = ATM_R['Uwind'][closest_idx, :, :], ATM_R['Vwind'][closest_idx, :, :]
            magnitude_R = np.sqrt(U_R**2 + V_R**2)
#            cset2 = ax2.contourf(lon_r, lat_r, magnitude_R, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset2 = ax2.contourf(lon_r, lat_r, magnitude_R, 50, cmap=cmap, transform=ccrs.PlateCarree())
            ax2.quiver(lon_r[::10, ::10], lat_r[::10, ::10], U_R[::10, ::10], V_R[::10, ::10], transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            #cbar2.set_ticks(np.arange(0, 16, 5))
            ax2.set_title('10 m velocity [m/s, ATM_R]')
        
        elif field == 'pressure':
            zzz = ATM['Pair'][closest_idx, :, :]
            #plevs = np.arange(np.floor(np.min(zzz)), np.ceil(np.max(zzz)), 0.1)
#            cset1 = ax1.contourf(lon, lat, zzz , plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset1 = ax1.contourf(lon, lat, zzz , 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            #cbar1.set_ticks(np.arange(1000, 1026, 5))
            ax1.set_title('Surface Pressure [Pa, ATM]')

#            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Pair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Pair'][closest_idx, :, :], 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            #cbar2.set_ticks(np.arange(1000, 1026, 5))
            ax2.set_title('Surface Pressure [Pa, ATM_R]')
        
        elif field == 'temperature':
            zzz = ATM['Tair'][closest_idx, :, :]
            #plevs = np.arange(np.floor(np.min(zzz)), np.ceil(np.max(zzz)), 0.1)
            #cset1 = ax1.contourf(lon, lat, zzz , plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset1 = ax1.contourf(lon, lat, zzz , 50, cmap=cmap, transform=ccrs.PlateCarree())

            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            #cbar1.set_ticks(np.arange(8, 41, 8))
            ax1.set_title('Surface Temperature [C, ATM]')
            #cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Tair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Tair'][closest_idx, :, :], 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            #cbar2.set_ticks(np.arange(8, 41, 8))
            ax2.set_title('Surface Temperature [C, ATM_LV' + lv + ']')
        
        elif field == 'lw_radiation':
            zzz = ATM['lwrad_down'][closest_idx, :, :]
            #plevs = np.arange(np.floor(np.min(zzz)), np.ceil(np.max(zzz)), 1)
#            cset1 = ax1.contourf(lon, lat, zzz, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset1 = ax1.contourf(lon, lat, zzz, 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            #cbar1.set_ticks(np.arange(260, 520, 50))
            ax1.set_title('Long Wave Radiation Down [W/m^2, ATM]')

#            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['lwrad_down'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['lwrad_down'][closest_idx, :, :], 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            #cbar2.set_ticks(np.arange(260, 520, 50))
            ax2.set_title('Long Wave Radiation Down [W/m^2, ATM_R]')
        
        elif field == 'rain':
            zzz = ATM['rain'][closest_idx, :, :]
            #plevs = np.arange(np.floor(np.min(zzz)), np.ceil(np.max(zzz)), 0.0001)
            #cset1 = ax1.contourf(lon, lat, zzz , plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset1 = ax1.contourf(lon, lat, zzz , 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            #cbar1.set_ticks(np.arange(0, 0.0035, 0.0007))
            ax1.set_title('Precipitation Rate [kg/m^2/s, ATM]')

#            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['rain'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['rain'][closest_idx, :, :], 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            #cbar2.set_ticks(np.arange(0, 0.0035, 0.0007))
            ax2.set_title('Precipitation Rate [kg/m^2/s, ATM_R]')
        
        elif field == 'humidity':
            zzz = ATM['Qair'][closest_idx, :, :]
            #plevs = np.arange(np.floor(np.min(zzz)), np.ceil(np.max(zzz)), .1)
            cset1 = ax1.contourf(lon, lat, zzz, 50, cmap=cmap, transform=ccrs.PlateCarree())
#            cset1 = ax1.contourf(lon, lat, zzz, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            #cbar1.set_ticks(np.arange(5, 101, 20))
            ax1.set_title('Surface Humidity [%, ATM]')

#            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Qair'][closest_idx, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Qair'][closest_idx, :, :], 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            #cbar2.set_ticks(np.arange(5, 102, 20))
            ax2.set_title('Surface Humidity [%, ATM_R]')
        
        elif field == 'swrad':
            zzz = ATM['swrad'][closest_idx, :, :]
            #plevs = np.arange(np.floor(np.min(zzz)), np.ceil(np.max(zzz)), .1)
            cset1 = ax1.contourf(lon, lat, zzz, 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            #cbar1.set_ticks(np.arange(100, 601, 100))
            ax1.set_title('Short Wave Radiation Down [W/m^2, ATM]')

            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['swrad'][closest_idx, :, :], 50, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            #cbar2.set_ticks(np.arange(100, 601, 100))
            ax2.set_title('Short Wave Radiation Down [W/m^2, ATM_R]')
        
        for ax in [ax1, ax2]:
            plot_roms_box(ax, RMG)
            #plot_roms_coastline(ax, RMG)
            #rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \
            #                                    scale='50m', edgecolor='b', facecolor='none')
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.BORDERS)
            #ax.add_feature(rivers, linewidth=0.5)
            ax.add_feature(cfeature.COASTLINE, linewidth = 2.0)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            ax.set_aspect(aspect='auto')
            if ax == ax1:
                ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
                ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
                annotation = f'raw atm. {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: {PFM['atm_model']} | Forecast Hour: {forecast_hours:.1f}'
            else:
                ax.set_xticks(np.round(np.linspace(np.min(lon_r), np.max(lon_r), num=5), 2))
                ax.set_yticks(np.round(np.linspace(np.min(lat_r), np.max(lat_r), num=5), 2))
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
                annotation = f'LV{lv} atm. {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: {PFM['atm_model']} | Forecast Hour: {forecast_hours:.1f}'

            ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        # Save the plot for each field
        filename = f'{output_dir}/{timestamp}_{PFM['atm_model']}_ATMandATMR_{field}_hour_{forecast_hours}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if show is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

def load_and_plot_atm(lv, fields_to_plot=None):
    """
    Load the atm.nc file and plot specified fields.

    Parameters:
    file_path (str): Path to the atm.nc file.
    RMG (dict): The ROMS grid data dictionary.
    product_name (str): The name of the forecast model.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    """
 
    PFM=get_PFM_info()

    if lv == '1':
        file_path = PFM['lv1_forc_dir'] + '/' + PFM['lv1_atm_file'] # LV1 atm forcing filename
        RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
    elif lv == '2':
        file_path = PFM['lv2_forc_dir'] + '/' + PFM['lv2_atm_file'] 
        RMG = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
    elif lv == '3':
        file_path = PFM['lv3_forc_dir'] + '/' + PFM['lv3_atm_file'] 
        RMG = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
    else:
        file_path = PFM['lv4_forc_dir'] + '/' + PFM['lv4_atm_file'] 
        RMG = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
    
 
    # Load the atm.nc file
    ds = nc.Dataset(file_path)
    RMG = RMG
    
    # Create the ATM dictionary
    ATM = {
        'lon': ds.variables['lon'][:],
        'lat': ds.variables['lat'][:],
        'ocean_time': ds.variables['ocean_time'][:],
        'Uwind': ds.variables['Uwind'][:],
        'Vwind': ds.variables['Vwind'][:],
        'Pair': ds.variables['Pair'][:],
        'Tair': ds.variables['Tair'][:],
        'lwrad_down': ds.variables['lwrad_down'][:],
        'rain': ds.variables['rain'][:],
        'Qair': ds.variables['Qair'][:],
        'swrad': ds.variables['swrad'][:]
    }
    
    # Close the dataset
    ds.close()

    # Plot the ATM fields
    plot_atm_r_fields(ATM, RMG, PFM, fields_to_plot, flag=False)

# This is the time series plotting function for his.nc files. A preliminary code, so, many changes to be made!!!

def rotate_to_earth(u_roms, v_roms, cos_ang, sin_ang):
    """
    Rotate the velocity components from ROMS grid to Earth coordinates using the given angles.
    """
    u_earth = cos_ang * u_roms - sin_ang * v_roms
    v_earth = sin_ang * u_roms + cos_ang * v_roms
    return u_earth, v_earth

def plot_his_time_series(his_filepath, RMG, lat, lon, depth_level=-1):
    his_ds = Dataset(his_filepath)
    
    # Extract variables
    time_var = his_ds.variables['ocean_time']
    time = num2date(time_var[:], units=time_var.units, calendar='standard')
    time = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in time])
    
    temp = his_ds.variables['temp'][:, depth_level, :, :]  # Temperature at specified depth
    salt = his_ds.variables['salt'][:, depth_level, :, :]  # Salinity at specified depth
    zeta = his_ds.variables['zeta'][:, :, :]               # Sea surface height
    ubar = his_ds.variables['ubar'][:, :, :]               # Barotropic u-component
    vbar = his_ds.variables['vbar'][:, :, :]               # Barotropic v-component
    
    # Extract grid information
    lon_rho = his_ds.variables['lon_rho'][:]
    lat_rho = his_ds.variables['lat_rho'][:]
    
    # Need to pay attention over here, for now hardcoding the ilat and ilon
    ilat = 150
    ilon = 200
    
    # Compute trigonometric functions for rotation
    ang_u = RMG['angle']
    cos_ang = np.cos(ang_u)
    sin_ang = np.sin(ang_u)
    
    # Extract time series data for the specified location
    temp_ts = temp[:, ilat, ilon]  # Temperature time series
    salt_ts = salt[:, ilat, ilon]  # Salinity time series
    zeta_ts = zeta[:, ilat, ilon]  # Sea surface height time series
    ubar_ts = ubar[:, ilat, ilon]  # Barotropic u-component time series
    vbar_ts = vbar[:, ilat, ilon]  # Barotropic v-component time series
    
    # Rotate barotropic velocities to Earth coordinates
    ubar_earth, vbar_earth = rotate_to_earth(ubar_ts, vbar_ts, cos_ang[ilat, ilon], sin_ang[ilat, ilon])
    
    # Initialize plot
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    
    # Plot time series
    axs[0].plot(time, temp_ts, label=f'Temperature at ({lat_rho[ilat, ilon]}, {lon_rho[ilat, ilon]})')
    axs[1].plot(time, salt_ts, label=f'Salinity at ({lat_rho[ilat, ilon]}, {lon_rho[ilat, ilon]})')
    axs[2].plot(time, zeta_ts, label=f'Sea Surface Height at ({lat_rho[ilat, ilon]}, {lon_rho[ilat, ilon]})')
    axs[3].plot(time, vbar_earth, label=f'Barotropic V at ({lat_rho[ilat, ilon]}, {lon_rho[ilat, ilon]})')
    
    # Set labels and titles
    axs[0].set_ylabel('Temperature (°C)')
    axs[1].set_ylabel('Salinity (psu)')
    axs[2].set_ylabel('Sea Surface Height (m)')
    axs[3].set_ylabel('Bar V (m/s)')
    axs[3].set_xlabel('Date')
    
    # Title for the entire plot
    fig.suptitle(f"Time Series Plot for his.nc file at (Lat: {lat}, Lon: {lon}, Depth: {depth_level})", fontsize=16)
    
    # Format x-axis to display dates
    for ax in axs:
        ax.legend()
        ax.grid()
        ax.xaxis_date()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Close the netCDF file
    his_ds.close()

    # How the input for this function will be:
    # his_filepath = '/scratch/matt/sdtjre_L1/I06J18/ocean_his_LV1_20192020.nc' # this will be the path in PFM where his file is located
    # fngr = '/home/mspydell/models/SDPM_root/SDPM/grids/GRID_SDTJRE_LV1.nc' #loading the grid... We don't need to do if running in driver_run_forecast_LV1.py file
    # RMG = grdfuns.roms_grid_to_dict(fngr)
    # lat = [32]  # latitude 
    # lon = [-118]  # longitude

    # and simply call the function!


# Function to plot OCN fields 
# For now this function takes a netcdf file and then plots, i have not integrated it with the OCN dict(as I was facing some issues) and PFM.
# Sample Input to this function is mentioned!

def plot_ocn_fields_from_nc(file_path, RMG, fields_to_plot=None, show=False):
    """
    Plot specified fields from a NetCDF dataset and save them as PNG files.
    
    Parameters:
    file_path (str): The path to the NetCDF file.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    """
    # Load the NetCDF file
    dataset = nc.Dataset(file_path)

    # Extract the common variables
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    lon = np.where(lon > 180, lon - 360, lon) # Remove this line when using the dictionary as Matt does the longitude conversion there.
    time = dataset.variables['time'][0]  # Only considering the first time step for simplicity
    start_time = nc.num2date(time, units=dataset.variables['time'].units)

    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'surf_el', 'water_temp', 'salinity']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot

    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('turbo')
        plt.set_cmap(cmap)

        if field == 'velocity':
            u = dataset.variables['water_u'][0, 0, :, :]  # surface layer
            v = dataset.variables['water_v'][0, 0, :, :]  # surface layer
            magnitude = np.sqrt(u**2 + v**2)
            plevs = np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 50)
            cset = ax.contourf(lon, lat, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon[::5], lat[::5], u[::5, ::5], v[::5, ::5], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Velocity Magnitude [m/s]')
            cbar.set_ticks(np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 5))
            ax.set_title('Surface Velocity [m/s]')
        
        elif field == 'surf_el':
            surf_el = dataset.variables['surf_el'][0, :, :]
            plevs = np.linspace(np.nanmin(surf_el), np.nanmax(surf_el), 50)
            cset = ax.contourf(lon, lat, surf_el, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Elevation [m]')
            cbar.set_ticks(np.linspace(np.nanmin(surf_el), np.nanmax(surf_el), 5))
            ax.set_title('Surface Elevation [m]')
        
        elif field == 'water_temp':
            water_temp = dataset.variables['water_temp'][0, 0, :, :]
            plevs = np.linspace(np.nanmin(water_temp), np.nanmax(water_temp), 50)
            cset = ax.contourf(lon, lat, water_temp, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Temperature [°C]')
            cbar.set_ticks(np.linspace(np.nanmin(water_temp), np.nanmax(water_temp), 5))
            ax.set_title('Surface Temperature [°C]')
        
        elif field == 'salinity':
            salinity = dataset.variables['salinity'][0, 0, :, :]
            plevs = np.linspace(np.nanmin(salinity), np.nanmax(salinity), 50)
            cset = ax.contourf(lon, lat, salinity, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_ticks(np.linspace(np.nanmin(salinity), np.nanmax(salinity), 5))
            cbar.set_label('Salinity [psu]')
            ax.set_title('Surface Salinity [psu]')
        
        # Add coastlines and gridlines and ROMS box
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
        ax.grid(True)
        plot_roms_box(ax, RMG)
        ax.set_aspect('auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # Set the title and labels
        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: Hycom | Field: {field}'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        plt.show()
        plt.close()
    
    # Close the dataset
    dataset.close()

# OCN from dict
# The function we will use
# The function which helps to make OCN plots
# integrated with the PFM.

def plot_ocn_fields_from_dict(OCN, RMG, PFM, fields_to_plot=None, show=False):
    """
    Plot specified fields from a dictionary and save them as PNG files.
    
    Parameters:
    OCN (dict): Dictionary containing the data fields.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    """
    # Extract the common variables
    timestamp = extract_timestamp(OCN)
    lon = OCN['lon'][:]
    lat = OCN['lat'][:]
    ocean_time = OCN['ocean_time']
    start_time = datetime(1999, 1, 1) + timedelta(days=float(ocean_time[0]))

    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'surf_el', 'water_temp', 'salinity']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot

    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('turbo')
        plt.set_cmap(cmap)

        if field == 'velocity':
            u = OCN['u'][0, 0, :, :]  # surface layer
            v = OCN['v'][0, 0, :, :]  # surface layer
            magnitude = np.sqrt(u**2 + v**2)
            plevs = np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 50)
            cset = ax.contourf(lon, lat, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon[::5], lat[::5], u[::5, ::5], v[::5, ::5], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Velocity Magnitude [m/s]')
            cbar.set_ticks(np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 5))
            ax.set_title('Surface Velocity [m/s]')
        
        elif field == 'surf_el':
            surf_el = OCN['zeta'][0, :, :]
            plevs = np.linspace(np.nanmin(surf_el), np.nanmax(surf_el), 50)
            cset = ax.contourf(lon, lat, surf_el, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Elevation [m]')
            cbar.set_ticks(np.linspace(np.nanmin(surf_el), np.nanmax(surf_el), 5))
            ax.set_title('Surface Elevation [m]')
        
        elif field == 'water_temp':
            water_temp = OCN['temp'][0, 0, :, :]
            plevs = np.linspace(np.nanmin(water_temp), np.nanmax(water_temp), 50)
            cset = ax.contourf(lon, lat, water_temp, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Temperature [°C]')
            cbar.set_ticks(np.linspace(np.nanmin(water_temp), np.nanmax(water_temp), 5))
            ax.set_title('Surface Temperature [°C]')
        
        elif field == 'salinity':
            salinity = OCN['salt'][0, 0, :, :]
            plevs = np.linspace(np.nanmin(salinity), np.nanmax(salinity), 50)
            cset = ax.contourf(lon, lat, salinity, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_ticks(np.linspace(np.nanmin(salinity), np.nanmax(salinity), 5))
            cbar.set_label('Salinity [psu]')
            ax.set_title('Surface Salinity [psu]')
        
        # Add coastlines and gridlines and ROMS box
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
        ax.grid(True)
        plot_roms_box(ax, RMG)
        ax.set_aspect('auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # Set the title and labels
        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: Hycom | Field: {field}'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        
        output_dir = PFM['lv1_plot_dir']
        filename = f'{output_dir}/{timestamp}_hycom_OCN_{field}.png'
        plt.savefig(filename, dpi=300)
        if show is True:
            plt.tight_layout
            plt.show()
        else:
            plt.close()


def plot_ocn_fields_from_dict_pckl(fname_in, fields_to_plot=None, show=False):
    """
    Plot specified fields from a dictionary and save them as PNG files.
    
    Parameters:
    OCN (dict): Dictionary containing the data fields.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    """

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

    print(fname_in)
#    print(fname_out)

    with open(fname_in,'rb') as fp:
        OCN = pickle.load(fp)
        print('OCN dict loaded with pickle')


    # Extract the common variables
    timestamp = extract_timestamp(OCN)
    lon = OCN['lon'][:]
    lat = OCN['lat'][:]
    ocean_time = OCN['ocean_time']
    start_time = datetime(1999, 1, 1) + timedelta(days=float(ocean_time[0]))

    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'surf_el', 'water_temp', 'salinity']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot

    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('turbo')
        plt.set_cmap(cmap)

        if field == 'velocity':
            u = OCN['u'][0, 0, :, :]  # surface layer
            v = OCN['v'][0, 0, :, :]  # surface layer
            magnitude = np.sqrt(u**2 + v**2)
            plevs = np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 50)
            cset = ax.contourf(lon, lat, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon[::5], lat[::5], u[::5, ::5], v[::5, ::5], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Velocity Magnitude [m/s]')
            cbar.set_ticks(np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 5))
            ax.set_title('Surface Velocity [m/s]')
        
        elif field == 'surf_el':
            surf_el = OCN['zeta'][0, :, :]
            plevs = np.linspace(np.nanmin(surf_el), np.nanmax(surf_el), 50)
            cset = ax.contourf(lon, lat, surf_el, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Elevation [m]')
            cbar.set_ticks(np.linspace(np.nanmin(surf_el), np.nanmax(surf_el), 5))
            ax.set_title('Surface Elevation [m]')
        
        elif field == 'water_temp':
            water_temp = OCN['temp'][0, 0, :, :]
            plevs = np.linspace(np.nanmin(water_temp), np.nanmax(water_temp), 50)
            cset = ax.contourf(lon, lat, water_temp, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Temperature [°C]')
            cbar.set_ticks(np.linspace(np.nanmin(water_temp), np.nanmax(water_temp), 5))
            ax.set_title('Surface Temperature [°C]')
        
        elif field == 'salinity':
            salinity = OCN['salt'][0, 0, :, :]
            plevs = np.linspace(np.nanmin(salinity), np.nanmax(salinity), 50)
            cset = ax.contourf(lon, lat, salinity, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_ticks(np.linspace(np.nanmin(salinity), np.nanmax(salinity), 5))
            cbar.set_label('Salinity [psu]')
            ax.set_title('Surface Salinity [psu]')
        
        # Add coastlines and gridlines and ROMS box
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
        ax.grid(True)
        plot_roms_box(ax, RMG)
        ax.set_aspect('auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # Set the title and labels
        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: Hycom | Field: {field}'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        
        output_dir = PFM['lv1_plot_dir']
        filename = f'{output_dir}/{timestamp}_hycom_OCN_{field}.png'
        plt.savefig(filename, dpi=300)
        if show is True:
            plt.tight_layout
            plt.show()
        else:
            plt.close()


def plot_ocn_R_fields(OCN_R, RMG, PFM, fields_to_plot=None, time_index=0, depth_index=0, show=False):
    timestamp = extract_timestamp(OCN_R)
    lon = OCN_R['lon_rho']
    lat = OCN_R['lat_rho']
    time = OCN_R['ocean_time'][time_index]
    start_time = nc.num2date(time, units='days since 1999-01-01')  # Adjust units as needed

    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'zeta', 'temp', 'salt']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot

    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('turbo')
        plt.set_cmap(cmap)

        if field == 'velocity':
            u = OCN_R['urm'][time_index, depth_index, :, :]  # surface layer
            v = OCN_R['vrm'][time_index, depth_index, :, :]  # surface layer
            magnitude = np.sqrt(u**2 + v**2)
            plevs = np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 50)
            cset = ax.contourf(lon, lat, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon[::5], lat[::5], u[::5, ::5], v[::5, ::5], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Velocity Magnitude [m/s]')
            cbar.set_ticks(np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 5))
            ax.set_title('Surface Velocity [m/s]')
        
        elif field == 'zeta':
            zeta = OCN_R['zeta'][time_index, :, :]
            plevs = np.linspace(np.nanmin(zeta), np.nanmax(zeta), 50)
            cset = ax.contourf(lon, lat, zeta, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Elevation [m]')
            cbar.set_ticks(np.linspace(np.nanmin(zeta), np.nanmax(zeta), 5))
            ax.set_title('Surface Elevation [m]')
        
        elif field == 'temp':
            temp = OCN_R['temp'][time_index, depth_index, :, :]
            plevs = np.linspace(np.nanmin(temp), np.nanmax(temp), 50)
            cset = ax.contourf(lon, lat, temp, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Temperature [°C]')
            cbar.set_ticks(np.linspace(np.nanmin(temp), np.nanmax(temp), 5))
            ax.set_title('Surface Temperature [°C]')
        
        elif field == 'salt':
            salt = OCN_R['salt'][time_index, depth_index, :, :]
            plevs = np.linspace(np.nanmin(salt), np.nanmax(salt), 50)
            cset = ax.contourf(lon, lat, salt, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_ticks(np.linspace(np.nanmin(salt), np.nanmax(salt), 5))
            cbar.set_label('Salinity [psu]')
            ax.set_title('Surface Salinity [psu]')
        
        # Add coastlines and gridlines and ROMS box
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
        ax.grid(True)
        plot_roms_box(ax, RMG)
        ax.set_aspect('auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # Set the title and labels
        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: Hycom |'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        output_dir = PFM['lv1_plot_dir']
        filename = f'{output_dir}/{timestamp}_hycom_OCN_R_{field}.png'
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

def load_ocnR_from_pckl_files():

    PFM=get_PFM_info()
    ork = ['depth','lat_rho','lon_rho','lat_u','lon_u','lat_v','lon_v','ocean_time','ocean_time_ref','salt','temp','ubar','urm','vbar','vrm','zeta','vinfo']

    OCN_R = dict()
    for nm in ork:
        fn_temp = PFM['lv1_forc_dir'] + '/tmp_' + nm + '.pkl'
        with open(fn_temp,'rb') as fp:
            OCN_R[nm] = pickle.load(fp)

    return OCN_R

def plot_ocn_R_fields_pckl(fields_to_plot=None, time_index=0, depth_index=0, show=False):

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

    OCN_R = load_ocnR_from_pckl_files()

    timestamp = extract_timestamp(OCN_R)
    lon = OCN_R['lon_rho']
    lat = OCN_R['lat_rho']
    time = OCN_R['ocean_time'][time_index]
    start_time = nc.num2date(time, units='days since 1999-01-01')  # Adjust units as needed

    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'zeta', 'temp', 'salt']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot

    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('turbo')
        plt.set_cmap(cmap)

        if field == 'velocity':
            u = OCN_R['urm'][time_index, depth_index, 0:-1, :]  # surface layer
            v = OCN_R['vrm'][time_index, depth_index, :, 0:-1]  # surface layer
            magnitude = np.sqrt(u**2 + v**2)
            plevs = np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 50)
            cset = ax.contourf(lon[0:-1,0:-1], lat[0:-1,0:-1], magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon[0:-1:5,0:-1:5], lat[0:-1:5,0:-1:5], u[::5, ::5], v[::5, ::5], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Velocity Magnitude [m/s]')
            cbar.set_ticks(np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 5))
            ax.set_title('Surface Velocity [m/s]')
        
        elif field == 'zeta':
            zeta = OCN_R['zeta'][time_index, :, :]
            plevs = np.linspace(np.nanmin(zeta), np.nanmax(zeta), 50)
            cset = ax.contourf(lon, lat, zeta, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Elevation [m]')
            cbar.set_ticks(np.linspace(np.nanmin(zeta), np.nanmax(zeta), 5))
            ax.set_title('Surface Elevation [m]')
        
        elif field == 'temp':
            temp = OCN_R['temp'][time_index, depth_index, :, :]
            plevs = np.linspace(np.nanmin(temp), np.nanmax(temp), 50)
            cset = ax.contourf(lon, lat, temp, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Temperature [°C]')
            cbar.set_ticks(np.linspace(np.nanmin(temp), np.nanmax(temp), 5))
            ax.set_title('Surface Temperature [°C]')
        
        elif field == 'salt':
            salt = OCN_R['salt'][time_index, depth_index, :, :]
            plevs = np.linspace(np.nanmin(salt), np.nanmax(salt), 50)
            cset = ax.contourf(lon, lat, salt, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_ticks(np.linspace(np.nanmin(salt), np.nanmax(salt), 5))
            cbar.set_label('Salinity [psu]')
            ax.set_title('Surface Salinity [psu]')
        
        # Add coastlines and gridlines and ROMS box
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
        ax.grid(True)
        plot_roms_box(ax, RMG)
        ax.set_aspect('auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # Set the title and labels
        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: Hycom |'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        output_dir = PFM['lv1_plot_dir']
        filename = f'{output_dir}/{timestamp}_hycom_OCN_R_{field}.png'
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

def plot_ocn_ic_fields(filepath, fields_to_plot=None, time_index=0, depth_index=0, show=False):

    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])

    nc = Dataset(filepath, 'r')

    # Extract coordinate variables
    lon_rho = nc.variables['lon_rho'][:,:]
    lat_rho = nc.variables['lat_rho'][:,:]
    time = nc.variables['ocean_time'][time_index]
    start_time = num2date(time, units='days since 1999-01-01')  # Adjust units as needed

    if fields_to_plot is None:
        fields_to_plot = ['velocity', 'zeta', 'temp', 'salt']
    else:
        fields_to_plot = [fields_to_plot] if isinstance(fields_to_plot, str) else fields_to_plot

    for field in fields_to_plot:
        fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        cmap = plt.get_cmap('turbo')
        plt.set_cmap(cmap)

        if field == 'velocity':
            u = nc.variables['u'][time_index, depth_index, 0:-1, :]  # Adjust the depth index as needed
            v = nc.variables['v'][time_index, depth_index, :, 0:-1]  # Adjust the depth index as needed

            # Coordinates for u and v
            #lon_u = nc.variables['lon_u'][:]
            #lat_u = nc.variables['lat_u'][:]
            #lon_v = nc.variables['lon_v'][:]
            #lat_v = nc.variables['lat_v'][:]

            # Interpolate u and v to rho points
            #u_interp = griddata((lon_u.flatten(), lat_u.flatten()), u.flatten(), (lon_rho, lat_rho), method='linear')
            #v_interp = griddata((lon_v.flatten(), lat_v.flatten()), v.flatten(), (lon_rho, lat_rho), method='linear')

            magnitude = np.sqrt(u**2 + v**2)
            plevs = np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 50)
            cset = ax.contourf(lon_rho[0:-1,0:-1], lat_rho[0:-1,0:-1], magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon_rho[0:-1:5,0:-1:5], lat_rho[0:-1:5,0:-1:5], u[::5, ::5], v[::5, ::5], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Velocity Magnitude [m/s]')
            cbar.set_ticks(np.linspace(np.nanmin(magnitude), np.nanmax(magnitude), 5))
            ax.set_title('Surface Velocity [m/s]')
        
        elif field == 'zeta':
            zeta = nc.variables['zeta'][time_index, :, :]
            plevs = np.linspace(np.nanmin(zeta), np.nanmax(zeta), 50)
            cset = ax.contourf(lon_rho, lat_rho, zeta, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Elevation [m]')
            cbar.set_ticks(np.linspace(np.nanmin(zeta), np.nanmax(zeta), 5))
            ax.set_title('Surface Elevation [m]')
        
        elif field == 'temp':
            temp = nc.variables['temp'][time_index, depth_index, :, :]
            plevs = np.linspace(np.nanmin(temp), np.nanmax(temp), 50)
            cset = ax.contourf(lon_rho, lat_rho, temp, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Surface Temperature [K]')
            cbar.set_ticks(np.linspace(np.nanmin(temp), np.nanmax(temp), 5))
            ax.set_title('Surface Temperature [K]')
        
        elif field == 'salt':
            salt = nc.variables['salt'][time_index, depth_index, :, :]
            plevs = np.linspace(np.nanmin(salt), np.nanmax(salt), 50)
            cset = ax.contourf(lon_rho, lat_rho, salt, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_ticks(np.linspace(np.nanmin(salt), np.nanmax(salt), 5))
            cbar.set_label('Salinity [psu]')
            ax.set_title('Surface Salinity [psu]')
        
        # Add coastlines and gridlines
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black')
        ax.grid(True)
        plot_roms_box(ax, RMG)
        ax.set_aspect('auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon_rho), np.max(lon_rho), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat_rho), np.max(lat_rho), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # Set the title and labels
        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: ROMS | Forecast Hour: {time_index}'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        output_dir = PFM['lv1_plot_dir']
        yyyymmdd = PFM['yyyymmdd']
        hhmm = PFM['hhmm']
        filename = f'{output_dir}/{yyyymmdd}_{hhmm}_hycom_OCN_IC_{field}.png'
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
    nc.close()

def plot_ssh_his_tseries_v2(fn,fn_grd,Ix,Iy,sv_fig,lvl,dir_out):

#    PFM=get_PFM_info()
#    if lvl == 'LV1':
    RMG = grdfuns.roms_grid_to_dict(fn_grd)
#    elif lvl == 'LV2':
#        RMG = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
#    elif lvl == 'LV3':
#        RMG = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
#    elif lvl == 'LV4':
#        RMG = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])

    his_ds = nc.Dataset(fn)

    fig = plt.figure(figsize=(6.5,10))
    gs = fig.add_gridspec(5, 5)

    ax1 = fig.add_subplot(gs[0:3, :], projection=ccrs.PlateCarree())

#    lt = his_ds.variables['lat_rho'][:]
#    ln = his_ds.variables['lon_rho'][:]
#    hb = his_ds.variables['h'][:]
    lt = RMG['lat_rho'][:]
    ln = RMG['lon_rho'][:]
    hb = his_ds.variables['h'][:,:]
    if lvl == 'LV1':
        plevs = np.arange(-4800, 0, 20)
    elif lvl == 'LV2':
        plevs = np.arange(-2400, -9, 1)
    elif lvl == 'LV3':
        plevs = np.arange(-1500, -1, 1)
    elif lvl == 'LV4':
        plevs = np.arange(-55, 0, .1)
            
    cmap = plt.get_cmap('viridis')
    cset = ax1.contourf(ln, lt, -hb, plevs, cmap=cmap, transform=ccrs.PlateCarree())
    plt.set_cmap(cmap)
    cbar = fig.colorbar(cset, ax=ax1, orientation='vertical', pad = 0.07)
    mrkrs = ['ro','bo']
    clrs  = ['r','b']
    for a in range(len(Ix)):
        ax1.plot(ln[Iy[a],Ix[a]],lt[Iy[a],Ix[a]],mrkrs[a], transform=ccrs.PlateCarree())

    if lvl == 'LV1':
        ax1.set_title('ROMS LV1 bathymetry')
    elif lvl == 'LV2':
        ax1.set_title('ROMS LV2 bathymetry')
    elif lvl == 'LV3':
        ax1.set_title('ROMS LV3 bathymetry')
    elif lvl == 'LV4':
        ax1.set_title('ROMS LV4 bathymetry')

    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.BORDERS)
    if lvl != 'LV4':
        ax1.add_feature(cfeature.COASTLINE, linewidth = 2.0)    
   #ax1.grid(True)
    #ax1.set_aspect(aspect='auto')
    ax1.set_xticks(np.round(np.linspace(np.min(ln), np.max(ln), num=5), 2))
    ax1.set_yticks(np.round(np.linspace(np.min(lt), np.max(lt), num=5), 2))
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
 

    ax2 = fig.add_subplot(gs[3, :])
    times = his_ds.variables['ocean_time']
    times2 = num2date(times[:], times.units)
    times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])
    #ln = his_ds.variables['lon_rho'][:]
    #lt = his_ds.variables['lat_rho'][:]
    txt_lg = []
    for a in range(len(Ix)):
        yy = his_ds.variables['zeta'][:,Iy[a],Ix[a]]
        ax2.plot(times2,yy,clrs[a])
        ln3=ln[Iy[a],Ix[a]]
        lt3=lt[Iy[a],Ix[a]]
        #ln3 = .01 * np.floor(round(ln3*100))
        #lt3 = .01 * np.floor(round(lt3*100))
        txt = f'({lt3:5.2f},{ln3:5.2f})'
        txt_lg.append(txt)

    
    # code to plot NOAA predictions at SIO pier...
    t_ob, ssh_ob, t_pr, ssh_pr = get_noaa_predicted_ssh(times2[0]-timedelta(days=0.5),times2[-1]+timedelta(days=0.5))
    ax2.plot(t_pr,ssh_pr,':g')
    ax2.plot(t_ob,ssh_ob,':k')
    txt_lg.append('NOAA SIO predicted')
    txt_lg.append('NOAA SIO observed')

    ax2.plot((times2[0],times2[0]),(-1,1),'--b')
    ax2.plot((times2[-1],times2[-1]),(-1,1),'--b')

            
    plt.legend(txt_lg, loc="lower right", prop={'size' : 9})    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    #plt.gcf().autofmt_xdate()
    plt.setp(plt.xticks()[1], rotation=30, ha='right') # ha is the same as horizontalalignment
    ax2.set_ylabel('Sea Surface Height (m)')
    if sv_fig == 1:
        #PFM=get_PFM_info()
        #if lvl == 'LV1':
        #    fn_out = PFM['lv1_plot_dir'] + '/his_ssh_tseries_LV1_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        #elif lvl == 'LV2':
        #    fn_out = PFM['lv2_plot_dir'] + '/his_ssh_tseries_LV2_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        #elif lvl == 'LV3':
        #    fn_out = PFM['lv3_plot_dir'] + '/his_ssh_tseries_LV3_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        #elif lvl == 'LV4':
        #    fn_out = PFM['lv4_plot_dir'] + '/his_ssh_tseries_LV4_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'

        yyyymmddhh = times2[0].strftime("%Y%m%d%H") # this is the start time of the simulation
        if lvl == 'LV1':
            fn_out = dir_out + '/his_ssh_tseries_LV1_' + yyyymmddhh + '.png'
        elif lvl == 'LV2':
            fn_out = dir_out + '/his_ssh_tseries_LV2_' + yyyymmddhh + '.png'
        elif lvl == 'LV3':
            fn_out = dir_out + '/his_ssh_tseries_LV3_' + yyyymmddhh + '.png'
        elif lvl == 'LV4':
            fn_out = dir_out + '/his_ssh_tseries_LV4_' + yyyymmddhh + '.png'


        plt.savefig(fn_out, dpi=300)
    else:
        plt.show()

def plot_ssh_his_tseries(fn,Ix,Iy,sv_fig):
    his_ds = nc.Dataset(fn)
    times = his_ds.variables['ocean_time']
    times2 = num2date(times[:], times.units)
    times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])
    fig, ax = plt.subplots()
    ln = his_ds.variables['lon_rho'][:]
    lt = his_ds.variables['lat_rho'][:]
    txt_lg = []
    for a in range(len(Ix)):
        yy = his_ds.variables['zeta'][:,Iy[a],Ix[a]]
        ax.plot(times2,yy)
        ln3=ln[Iy[a],Ix[a]]
        lt3=lt[Iy[a],Ix[a]]
        #ln3 = .01 * np.floor(round(ln3*100))
        #lt3 = .01 * np.floor(round(lt3*100))
        txt = f'({lt3:5.2f},{ln3:5.2f})'
        txt_lg.append(txt)
            
    plt.legend(txt_lg, loc="upper left")    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    ax.set_ylabel('Sea Surface Height (m)')
    if sv_fig == 1:
        PFM=get_PFM_info()
        fn_out = PFM['lv1_plot_dir'] + '/his_ssh_tseries_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        plt.savefig(fn_out, dpi=300)
    else:
        plt.show()

def plot_roms_LV1_bathy_and_locs(fn,Ix,Iy,sv_fig):
    his_ds = nc.Dataset(fn)
    fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    lt = his_ds.variables['lat_rho'][:]
    ln = his_ds.variables['lon_rho'][:]
    hb = his_ds.variables['h'][:]
    plevs = np.arange(-4800, 0, 20)
    cmap = plt.get_cmap('viridis')
    cset = ax.contourf(ln, lt, -hb, plevs, cmap=cmap, transform=ccrs.PlateCarree())
    plt.set_cmap(cmap)
    for a in range(len(Ix)):
        ax.plot(ln[Iy[a],Ix[a]],lt[Iy[a],Ix[a]],'ro', transform=ccrs.PlateCarree())

    cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)

    ax.set_title('ROMS LV1 bathymetry')
    ax.add_feature(cfeature.COASTLINE)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)
    ax.set_aspect(aspect='auto')
    ax.set_xticks(np.round(np.linspace(np.min(ln), np.max(ln), num=5), 2))
    ax.set_yticks(np.round(np.linspace(np.min(lt), np.max(lt), num=5), 2))

    if sv_fig == 1:
        PFM=get_PFM_info()
        fn_out = PFM['lv1_plot_dir'] + '/his_depth_locs_' + PFM['yyyymmdd'] + PFM['hhmm'] + '.png'
        plt.savefig(fn_out, dpi=300)
    else:
        plt.show()


def plot_his_temps_wuv(fn,It,Iz,sv_fig,lvl,cmn,cmx):

    cmn = float(cmn) # these are the colorbar limits
    cmx = float(cmx)

    cmn = np.floor(10*cmn) / 10
    cmx = np.ceil(10*cmx) / 10

    if lvl == 'LV4':
        dc = cmx - cmn
        cmn2 = cmn + .25*dc
        cmx = cmn + .95*dc
        cmn = cmn2
        cmn = np.floor(10*cmn) / 10
        cmx = np.ceil(10*cmx) / 10

    cmn = cmn - .1
    cmx = cmx + .1

    PFM=get_PFM_info()
    if lvl == 'LV1':
        RMG = grdfuns.roms_grid_to_dict(PFM['lv1_grid_file'])
        res = '110m'
        res2 = 'i'
    elif lvl == 'LV2':
        RMG = grdfuns.roms_grid_to_dict(PFM['lv2_grid_file'])
        res = '50m'
        res2 = 'h'
    elif lvl == 'LV3':
        RMG = grdfuns.roms_grid_to_dict(PFM['lv3_grid_file'])
        res = '10m'
        res2 = 'f'
    elif lvl == 'LV4':
        RMG = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
        res = '10m'
        res2 = 'f'

    his_ds = nc.Dataset(fn)
    #lt = his_ds.variables['lat_rho'][:]
    lt = RMG['lat_rho'][:]
    #ln = his_ds.variables['lon_rho'][:]
    ln = RMG['lon_rho'][:]


    #print(his_ds.variables.keys())

    if lvl == 'LV4':
        msk = his_ds.variables['wetdry_mask_rho'][It,:,:]
        msk = np.squeeze(msk)
    else:
        msk = his_ds.variables['mask_rho'][:,:]

    It = int(It)
    Iz = int(Iz)
    temp = his_ds.variables['temp'][It,Iz,:,:]
    temp = temp.data
    temp == np.squeeze(temp)
    
    temp[msk==0] = np.nan 

    #tempall = his_ds.variables['temp'][:,Iz,:,:]

    urm = np.squeeze( his_ds.variables['u'][It,Iz,:,:] )
    vrm = np.squeeze( his_ds.variables['v'][It,Iz,:,:] )
    urm2 = np.squeeze( .5 * (urm[0:-1,:]+urm[1:,:]) ) # now on rho points, but not 0 or -1
    vrm2 = np.squeeze( .5 * (vrm[:,0:-1]+vrm[:,1:]) )
    ang = RMG['angle'] # on rho points
    ang2 = np.squeeze( .5* (ang[0:-1,0:-1] + ang[1:,1:]  ))
    u = urm2 * np.cos(ang2) - vrm2 * np.sin(ang2)
    v = vrm2 * np.cos(ang2) + urm2 * np.sin(ang2)

    #print(np.shape(u))
    #print(np.shape(v))

    fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    plevs = np.arange(cmn,cmx,.05)
    cmap = plt.get_cmap('turbo')
    cset = ax.contourf(ln, lt, temp, plevs, cmap=cmap, extend="both", vmin=cmn, vmax=cmx, transform=ccrs.PlateCarree())        
    plt.set_cmap(cmap)
    cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)

    ln2 = .5* (ln[0:-1,0:-1]+ln[1:,1:])
    lt2 = .5* (lt[0:-1,0:-1]+lt[1:,1:])

    if lvl == 'LV1' or lvl =='LV2' or lvl == 'LV3':
        ax.quiver(ln2[0::8,0::8], lt2[0::8,0::8], u[0::8,0::8], v[0::8,0::8], transform=ccrs.PlateCarree())

    times = his_ds.variables['ocean_time']
    times2 = num2date(times[:], times.units)
    times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])

    start_time = times2[0]
    forecast_hours = It
    tzone = datetime.now().astimezone().tzinfo
    if str(tzone) == 'PDT':
        toff = -7
    elif str(tzone) == 'PST':
        toff = -8
    tfore = times2[It] + toff * timedelta(hours=1)    
    annotation = (f'{lvl} Surface Temp [C] and currents | Forecast: {start_time.strftime("%Y-%m-%d %H:%M:%S")}\n' 
                   f'Forecast Hour: {forecast_hours:.1f} ({tfore.strftime("%Y-%m-%d %H:%M:%S")} {str(tzone)})')

    ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
    #coast = cfeature.GSHHSFeature(scale='full')
    #ax.add_feature(cfeature.COASTLINE, resolution=res, linewidth = 1.25)   
    if lvl != 'LV4':
        coast = cfeature.GSHHSFeature(scale=res2)
        ax.add_feature(coast)
    #ax.coastlines(resolution=res)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)
    ax.set_aspect(aspect='auto')
    ax.set_xticks(np.round(np.linspace(np.min(ln), np.max(ln), num=5), 2))
    ax.set_yticks(np.round(np.linspace(np.min(lt), np.max(lt), num=5), 2))

    it_str = str(It).zfill(3)

    if sv_fig == '1':
        if lvl == 'LV1':
            fn_out = PFM['lv1_plot_dir'] + '/his_tempuv_LV1_' + PFM['yyyymmdd'] + PFM['hhmm'] + '_' + it_str + 'hr.png'
        elif lvl == 'LV2':            
            fn_out = PFM['lv2_plot_dir'] + '/his_tempuv_LV2_' + PFM['yyyymmdd'] + PFM['hhmm'] + '_' + it_str + 'hr.png'
        elif lvl == 'LV3':            
            fn_out = PFM['lv3_plot_dir'] + '/his_tempuv_LV3_' + PFM['yyyymmdd'] + PFM['hhmm'] + '_' + it_str + 'hr.png'
        elif lvl == 'LV4':            
            fn_out = PFM['lv4_plot_dir'] + '/his_tempuv_LV4_' + PFM['yyyymmdd'] + PFM['hhmm'] + '_' + it_str + 'hr.png'
        
        plt.savefig(fn_out, dpi=300)
    else:
        plt.show()

def plot_his_temps_wuv_v2(fn,It,Iz,sv_fig,lvl,cmn,cmx,fn_grd,dir_out):

    cmn = float(cmn) # these are the colorbar limits
    cmx = float(cmx)

    cmn = np.floor(10*cmn) / 10
    cmx = np.ceil(10*cmx) / 10

    if lvl == 'LV4':
        dc = cmx - cmn
        cmn2 = cmn + .25*dc
        cmx = cmn + .95*dc
        cmn = cmn2
        cmn = np.floor(10*cmn) / 10
        cmx = np.ceil(10*cmx) / 10

    cmn = cmn - .1
    cmx = cmx + .1

    if lvl == 'LV1':
        res = '110m'
        res2 = 'i'
    elif lvl == 'LV2':
        res = '50m'
        res2 = 'h'
    elif lvl == 'LV3':
        res = '10m'
        res2 = 'f'
    elif lvl == 'LV4':
        res = '10m'
        res2 = 'f'

    RMG = grdfuns.roms_grid_to_dict(fn_grd)

    
    his_ds = nc.Dataset(fn)
    #lt = his_ds.variables['lat_rho'][:]
    lt = RMG['lat_rho'][:]
    #ln = his_ds.variables['lon_rho'][:]
    ln = RMG['lon_rho'][:]


    #print(his_ds.variables.keys())

    if lvl == 'LV4':
        msk = his_ds.variables['wetdry_mask_rho'][It,:,:]
        msk = np.squeeze(msk)
    else:
        msk = his_ds.variables['mask_rho'][:,:]

    It = int(It)
    Iz = int(Iz)
    temp = his_ds.variables['temp'][It,Iz,:,:]
    temp = temp.data
    temp == np.squeeze(temp)
    
    temp[msk==0] = np.nan 

    #tempall = his_ds.variables['temp'][:,Iz,:,:]

    urm = np.squeeze( his_ds.variables['u'][It,Iz,:,:] )
    vrm = np.squeeze( his_ds.variables['v'][It,Iz,:,:] )
    urm2 = np.squeeze( .5 * (urm[0:-1,:]+urm[1:,:]) ) # now on rho points, but not 0 or -1
    vrm2 = np.squeeze( .5 * (vrm[:,0:-1]+vrm[:,1:]) )
    ang = RMG['angle'] # on rho points
    ang2 = np.squeeze( .5* (ang[0:-1,0:-1] + ang[1:,1:]  ))
    u = urm2 * np.cos(ang2) - vrm2 * np.sin(ang2)
    v = vrm2 * np.cos(ang2) + urm2 * np.sin(ang2)

    #print(np.shape(u))
    #print(np.shape(v))

    fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    plevs = np.arange(cmn,cmx,.05)
    cmap = plt.get_cmap('turbo')
    cset = ax.contourf(ln, lt, temp, plevs, cmap=cmap, extend="both", vmin=cmn, vmax=cmx, transform=ccrs.PlateCarree())        
    plt.set_cmap(cmap)
    cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)

    ln2 = .5* (ln[0:-1,0:-1]+ln[1:,1:])
    lt2 = .5* (lt[0:-1,0:-1]+lt[1:,1:])

    if lvl == 'LV1' or lvl =='LV2' or lvl == 'LV3':
        ax.quiver(ln2[0::8,0::8], lt2[0::8,0::8], u[0::8,0::8], v[0::8,0::8], transform=ccrs.PlateCarree())

    times = his_ds.variables['ocean_time']
    times2 = num2date(times[:], times.units)
    times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])

    start_time = times2[0]
    forecast_hours = It
    tzone = datetime.now().astimezone().tzinfo
    if str(tzone) == 'PDT':
        toff = -7
    elif str(tzone) == 'PST':
        toff = -8
    tfore = times2[It] + toff * timedelta(hours=1)    
    annotation = (f'{lvl} Surface Temp [C] and currents | Forecast: {start_time.strftime("%Y-%m-%d %H:%M:%S")}\n' 
                   f'Forecast Hour: {forecast_hours:.1f} ({tfore.strftime("%Y-%m-%d %H:%M:%S")} {str(tzone)})')

    ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
    #coast = cfeature.GSHHSFeature(scale='full')
    #ax.add_feature(cfeature.COASTLINE, resolution=res, linewidth = 1.25)   
    if lvl != 'LV4':
        coast = cfeature.GSHHSFeature(scale=res2)
        ax.add_feature(coast)
    #ax.coastlines(resolution=res)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)
    ax.set_aspect(aspect='auto')
    ax.set_xticks(np.round(np.linspace(np.min(ln), np.max(ln), num=5), 2))
    ax.set_yticks(np.round(np.linspace(np.min(lt), np.max(lt), num=5), 2))

    it_str = str(It).zfill(3)

    yyyymmddhh = times2[0].strftime('%Y%m%d%H')

    if sv_fig == '1':
        if lvl == 'LV1':
            fn_out = dir_out + '/his_tempuv_LV1_' + yyyymmddhh + '_' + it_str + 'hr.png'
        elif lvl == 'LV2':            
            fn_out = dir_out + '/his_tempuv_LV2_' + yyyymmddhh + '_' + it_str + 'hr.png'
        elif lvl == 'LV3':            
            fn_out = dir_out + '/his_tempuv_LV3_' + yyyymmddhh + '_' + it_str + 'hr.png'
        elif lvl == 'LV4':            
            fn_out = dir_out + '/his_tempuv_LV4_' + yyyymmddhh + '_' + it_str + 'hr.png'
        
        plt.savefig(fn_out, dpi=300)
    else:
        plt.show()


def plot_lv4_coawst_his_v2(fn,It,Iz,sv_fig,lvl,var_name,cmn,cmx,fn_grd,dir_out):

    cmn = np.floor(10*float(cmn)) / 10 # get to the nearest .1
    cmx = np.ceil(10*float(cmx)) / 10
    RMG = grdfuns.roms_grid_to_dict(fn_grd)
    #lt = his_ds.variables['lat_rho'][:]
    lt = RMG['lat_rho'][:]
    #ln = his_ds.variables['lon_rho'][:]
    ln = RMG['lon_rho'][:]
    #Dall = his_ds.variables[var_name][:,Iz,:,:]
    res = '10m'
    res2 = 'f'

    It = int(It)
    Iz = int(Iz)
    #his_ds = nc.Dataset(fn)
    with nc.Dataset(fn, 'r') as his_ds:
        times = his_ds.variables['ocean_time']
        times2 = num2date(times[:], times.units)
        times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])
        if var_name in ['dye_01','dye_02']:
            D = his_ds.variables[var_name][It,Iz,:,:]
            D = np.log10(D)
        if var_name == 'Hwave':
            D = his_ds.variables[var_name][It,:,:]


    #Dp5  = np.empty(1)
    #Dp95 = np.empty(1)

    #np.percentile(Dall,5,out=Dp5)
    #np.percentile(Dall,95,out=Dp95)

    if var_name == 'Hwave':
        units = 'm'
    if var_name == 'dye_01' or var_name == 'dye_02':
        units = 'log10 fraction'

    fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    #plevs = np.linspace(Dp5, Dp95, 25)
    if var_name in ['dye_01','dye_02']:
        plevs = np.arange(-5.5,-0.5,.5)
        cmap = plt.get_cmap('magma_r')
        cset = ax.contourf(ln, lt, D, plevs, cmap=cmap, extend="both", vmin=-6, vmax=0, transform=ccrs.PlateCarree())        
    else:
        plevs = np.arange(cmn,cmx,.05)
        cmap = plt.get_cmap('turbo')
        cset = ax.contourf(ln, lt, D, plevs, cmap=cmap, extend="both", vmin=cmn, vmax=cmx, transform=ccrs.PlateCarree())        
    plt.set_cmap(cmap)
    cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)


    start_time = times2[0]
    forecast_hours = It
    tzone = datetime.now().astimezone().tzinfo
    if str(tzone) == 'PDT':
        toff = -7
    elif str(tzone) == 'PST':
        toff = -8
    tfore = times2[It] + toff * timedelta(hours=1)    
    annotation = (f'{lvl} {var_name} [{units}] | Forecast: {start_time.strftime("%Y-%m-%d %H:%M:%S")}\n' 
                   f'Forecast Hour: {forecast_hours:.1f} ({tfore.strftime("%Y-%m-%d %H:%M:%S")} {str(tzone)})')

    ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
    ax.add_feature(cfeature.LAND,facecolor = np.array([0.859375, 0.859375, 0.859375]))
    ax.add_feature(cfeature.BORDERS)
    #coast = cfeature.GSHHSFeature(scale='full')
    #ax.add_feature(cfeature.COASTLINE, resolution=res, linewidth = 1.25)   
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)
    ax.set_aspect(aspect='auto')
    ax.set_xlim(-117.28, -117.06)
    ax.set_ylim(32.4, 32.75)
    ax.set_xticks(np.arange(-117.25, -117.1, .05))
    ax.set_yticks(np.arange(32.4,32.75, .05))

    it_str = str(It).zfill(3)

    yyyymmddhh = start_time.strftime('%Y%m%d%H')

    if sv_fig == '1':
        fn_out = dir_out + '/his_' + var_name + '_LV4_' + yyyymmddhh + '_' + it_str + 'hr.png'        
    #    print(fn_out)
        plt.savefig(fn_out, dpi=300)
    else:
        plt.show()

def plot_lv4_coawst_his(fn,It,Iz,sv_fig,lvl,var_name,cmn,cmx):

    cmn = np.floor(10*float(cmn)) / 10 # get to the nearest .1
    cmx = np.ceil(10*float(cmx)) / 10
    PFM=get_PFM_info()
    RMG = grdfuns.roms_grid_to_dict(PFM['lv4_grid_file'])
    #lt = his_ds.variables['lat_rho'][:]
    lt = RMG['lat_rho'][:]
    #ln = his_ds.variables['lon_rho'][:]
    ln = RMG['lon_rho'][:]
    #Dall = his_ds.variables[var_name][:,Iz,:,:]
    res = '10m'
    res2 = 'f'

    It = int(It)
    Iz = int(Iz)
    #his_ds = nc.Dataset(fn)
    with nc.Dataset(fn, 'r') as his_ds:
        times = his_ds.variables['ocean_time']
        times2 = num2date(times[:], times.units)
        times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])
        if var_name in ['dye_01','dye_02']:
            D = his_ds.variables[var_name][It,Iz,:,:]
            D = np.log10(D)
        if var_name == 'Hwave':
            D = his_ds.variables[var_name][It,:,:]


    #Dp5  = np.empty(1)
    #Dp95 = np.empty(1)

    #np.percentile(Dall,5,out=Dp5)
    #np.percentile(Dall,95,out=Dp95)

    if var_name == 'Hwave':
        units = 'm'
    if var_name == 'dye_01' or var_name == 'dye_02':
        units = 'log10 fraction'

    fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    #plevs = np.linspace(Dp5, Dp95, 25)
    if var_name in ['dye_01','dye_02']:
        #plevs = np.arange(-6.5,0.5,.5)
        #cmap = plt.get_cmap('turbo')
        plevs = np.arange(-5.5,-0.5,.5)
        cmap = plt.get_cmap('magma_r')
        cset = ax.contourf(ln, lt, D, plevs, cmap=cmap, extend="both", vmin=-6, vmax=0, transform=ccrs.PlateCarree())        
    else:
        plevs = np.arange(cmn,cmx,.05)
        cmap = plt.get_cmap('turbo')
        cset = ax.contourf(ln, lt, D, plevs, cmap=cmap, extend="both", vmin=cmn, vmax=cmx, transform=ccrs.PlateCarree())        
    plt.set_cmap(cmap)
    cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)


    start_time = times2[0]
    forecast_hours = It
    tzone = datetime.now().astimezone().tzinfo
    if str(tzone) == 'PDT':
        toff = -7
    elif str(tzone) == 'PST':
        toff = -8
    tfore = times2[It] + toff * timedelta(hours=1)    
    annotation = (f'{lvl} {var_name} [{units}] | Forecast: {start_time.strftime("%Y-%m-%d %H:%M:%S")}\n' 
                   f'Forecast Hour: {forecast_hours:.1f} ({tfore.strftime("%Y-%m-%d %H:%M:%S")} {str(tzone)})')

    ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
    ax.add_feature(cfeature.LAND,facecolor = np.array([0.859375, 0.859375, 0.859375]))
    ax.add_feature(cfeature.BORDERS)
    #coast = cfeature.GSHHSFeature(scale='full')
    #ax.add_feature(cfeature.COASTLINE, resolution=res, linewidth = 1.25)   
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)
    ax.set_aspect(aspect='auto')
    ax.set_xlim(-117.28, -117.06)
    ax.set_ylim(32.4, 32.75)
    ax.set_xticks(np.arange(-117.25, -117.1, .05))
    ax.set_yticks(np.arange(32.4,32.75, .05))


    it_str = str(It).zfill(3)

    if sv_fig == '1':
        fn_out = PFM['lv4_plot_dir'] + '/his_' + var_name + '_LV4_' + PFM['yyyymmdd'] + PFM['hhmm'] + '_' + it_str + 'hr.png'        
    #    print(fn_out)
        plt.savefig(fn_out, dpi=300)
    else:
        plt.show()


def get_his_clims(fn,var_name,Iz,It):

    his_ds = nc.Dataset(fn,'r')    
    D = his_ds.variables[var_name]
    if D.ndim == 3: # for example Hwave
        rmn = .01
        rmx = .99
        if It == 'all':
            DD = D[:,:,:]
        else:
            DD = D[:,:,It]
    elif D.ndim == 4: # for example temp, salt, etc
        if Iz == 'all': 
            if It == 'all':
                DD = D[:,:,:,:]
                rmn = .01
                rmx = .99
            else:
                rmn = .01
                rmx = .99
                DD = D[It,:,:,:]
        else:
            if It == 'all':
                rmn = .01
                rmx = .99
                DD = D[:,int(Iz),:,:]
            else:
                rmn = .01
                rmx = .99
                DD = D[int(It),int(Iz),:,:]
                

    #DD = DD.data
    DD.flatten() # now data has only one dimension, DD is a masked array
    DD = DD.compressed()  # get only the unmasked numbers
    #DD = DD[~np.isnan(DD)] # remove NaNs
    DD = np.sort(DD) # sort the numbers
    ld = float(len(DD))
    imn = int( np.round(rmn * ld ) )
    imx = int( np.round(rmx * ld ) )
    cmin = DD[imn]
    cmax = DD[imx]

    #fig, ax = plt.subplots()
    #p1=ax.hist(DD,bins=100)

    return cmin,cmax
    #return data

def make_all_his_figures(lvl):
    PFM=get_PFM_info()
    #PFM['lv4_model']='COAWST' # for testing
    sv_fig = 1
    #sv_fig = 0
    iz = -1
    if lvl == 'LV1':
        fn = PFM['lv1_his_name_full']
        Ix = np.array([175,240])
        Iy = np.array([175,170])
    elif lvl == 'LV2':
        fn = PFM['lv2_his_name_full']
        Ix = np.array([175,240])
        Iy = np.array([175,170])
    elif lvl == 'LV3': # 413 by 251
        fn = PFM['lv3_his_name_full']
        Ix = np.array([210,227])
        Iy = np.array([325,200])
    elif lvl == 'LV4' and PFM['lv4_model'] == 'ROMS': # 413 by 251
        fn = PFM['lv4_his_name_full']
        #print(fn)
        Ix = np.array([275,400])
        Iy = np.array([750,1000])
    elif lvl == 'LV4' and PFM['lv4_model'] == 'COAWST': # 413 by 251
        #fn = '/scratch/PFM_Simulations/LV4_Forecast/His/LV4_ocean_his_202411141800.nc' # for testing!
        fn = PFM['lv4_his_name_full']        
        Ix = np.array([275,400])
        Iy = np.array([750,1000])
        cmnH,cmxH = get_his_clims(fn,'Hwave',-1,'all')


    #fn = '/scratch/PFM_Simulations/LV4_Forecast/His/LV4_ocean_his_202501240000.nc'

    print('getting clims...')
    cmn,cmx = get_his_clims(fn,'temp',-1,'all')
    print('...done.')

    #plot_roms_LV1_bathy_and_locs(fn,Ix,Iy,sv_fig)
    #plot_ssh_his_tseries(fn,Ix,Iy,sv_fig)
    key_txt = 'lv'+lvl[2]+'_plot_dir'
    dir_out = PFM[key_txt]
    key_txt = 'lv'+lvl[2]+'_grid_file_full'
    fn_grd = PFM[key_txt]
    plot_ssh_his_tseries_v2(fn,fn_grd,Ix,Iy,sv_fig,lvl,dir_out) # uncommented on 2/10/25, does this break things?
                                                 # maybe need to turn this into a subprocess
                                                 # or use "with" in the function?
    os.chdir('../sdpm_py_util')

    pfm_hrs = int(24*PFM['forecast_days']) # this should be an integer
    It=0
    while It<=pfm_hrs:
        cmd_list = ['python','-W','ignore','plotting_functions.py','plot_his_temps_wuv',fn,str(It),str(iz),str(sv_fig),lvl,str(cmn),str(cmx)] 
        if lvl == 'LV4':
            ret1 = subprocess.Popen(cmd_list)  
        else:
            ret1 = subprocess.run(cmd_list)   
        #plot_his_temps_wuv(fn,It,iz,sv_fig,lvl)
        if PFM['lv4_model'] == 'COAWST' and lvl == 'LV4':
            #print(It)
            cmd_list = ['python','-W','ignore','plotting_functions.py','plot_lv4_coawst_his',fn,str(It),str(iz),str(sv_fig),lvl,'dye_01','0','1'] 
            ret1 = subprocess.Popen(cmd_list)     
            cmd_list = ['python','-W','ignore','plotting_functions.py','plot_lv4_coawst_his',fn,str(It),str(iz),str(sv_fig),lvl,'dye_02','0','1'] 
            ret1 = subprocess.Popen(cmd_list)     
            cmd_list = ['python','-W','ignore','plotting_functions.py','plot_lv4_coawst_his',fn,str(It),str(iz),str(sv_fig),lvl,'Hwave',str(cmnH),str(cmxH)] 
            if It<pfm_hrs:
                ret1 = subprocess.run(cmd_list)     
            else:
                ret1 = subprocess.run(cmd_list)
            #plot_lv4_coawst_his(fn,It,iz,sv_fig,lvl,'dye_01')
            #plot_lv4_coawst_his(fn,It,iz,sv_fig,lvl,'dye_02')
            #plot_lv4_coawst_his(fn,It,iz,sv_fig,lvl,'Hwave')
        It += 2

    os.chdir('../driver')

def get_ocean_times_from_ncfile(fn):
    # this function returns the times inside the the fn.nc file as a numpy array of datetimes

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




def get_noaa_predicted_ssh(tbeg,tend):
    
    t1 = tbeg - timedelta(days=1)
    t2 = tend + timedelta(days=1)
    t1str = t1.strftime("%Y%m%d")
    t2str = t2.strftime("%Y%m%d")
    meta = {
        'start': t1str,
        'end': t2str,
        'datum': 'MSL',
        'location': '9410230',
        'time_zone': 'GMT',
        'format': 'json',
        }

    sshob,tob,sshpr,tpr = get_tide_levels(meta)
    msk = (tpr >= tbeg) & (tpr <= tend)
    t_pr = tpr[msk]
    ssh_pr = sshpr[msk]

    msk = (tob >= tbeg) & (tob <= tend)
    t_ob = tob[msk]
    ssh_ob = sshob[msk]


    return t_ob, ssh_ob, t_pr, ssh_pr

def get_tide_levels(metadata):
    start_date = pd.to_datetime(metadata['start'])
    end_date = pd.to_datetime(metadata['end'])
    datum = metadata['datum']
    location = metadata['location']
    time_zone = metadata['time_zone']
    format = metadata['format']
    
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError('start date is after the end date')

    # Initialize lists to hold the combined data
    all_observed_v = []
    all_observed_times = []
    all_predicted_v = []
    all_prediction_times = []
    
    current_start_date = start_date
    
    while current_start_date <= end_date:
        # Calculate the current end date for this iteration (cannot exceed 31 days)
        current_end_date = min(current_start_date + timedelta(days=30), end_date)
        
        # Get predicted data for the current 31-day chunk
        res = requests.get(f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&application=NOS.COOPS.TAC.WL&begin_date={current_start_date.strftime("%Y%m%d")}&end_date={current_end_date.strftime("%Y%m%d")}&datum={datum}&station={location}&time_zone={time_zone}&units=metric&interval=&format={format}')
        predictions = res.json()['predictions']
        
        # Get observed data for the current 31-day chunk
        res = requests.get(f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=NOS.COOPS.TAC.WL&begin_date={current_start_date.strftime("%Y%m%d")}&end_date={current_end_date.strftime("%Y%m%d")}&datum={datum}&station={location}&time_zone={time_zone}&units=metric&interval=&format={format}')
        observed_data = res.json()['data'] if 'data' in res.json() else []
        
        # Extract prediction times and values
        prediction_times = [pd.to_datetime(data['t']) for data in predictions]
        predicted_v = [data['v'] for data in predictions]
        
        # Extract observed times and values
        observed_times = [pd.to_datetime(data['t']) for data in observed_data]
        observed_v = [entry['v'] for entry in observed_data]
        
        # Append the data to the combined lists
        all_prediction_times.extend(prediction_times)
        all_predicted_v.extend(predicted_v)
        all_observed_times.extend(observed_times)
        all_observed_v.extend(observed_v)
        
        # Move to the next chunk
        current_start_date = current_end_date + timedelta(days=1)
    
        #all_observed_times = pd.Timestamp.to_pydatetime(all_observed_times)
        #all_prediction_times = pd.Timestamp.to_pydatetime(all_prediction_times)
        #t_ob = np.zeros((len(all_observed_times)))
        t_ob = []
        for tt in all_observed_times:
            t_ob2 = tt.to_pydatetime()
            t_ob.append(t_ob2)

        t_ob = np.array(t_ob)

        t_pr = []
        for tt in all_prediction_times:
            t_ob2 = tt.to_pydatetime()
            t_pr.append(t_ob2)

        t_pr = np.array(t_pr)

    #print(all_observed_v)

    all_observed_v2 = [t if t is not '' else '-9999' for t in all_observed_v]
    all_predicted_v2 = [t if t is not '' else '-9999' for t in all_predicted_v]
    
    vob2 = np.array([float(x) for x in all_observed_v2])
    vpr2 = np.array([float(x) for x in all_predicted_v2])
    
    mask = np.ones(np.shape(vob2)).astype(bool)
    mask = np.logical_and(mask, vob2 == -9999)
    vob2[mask] = np.nan
    mask = np.ones(np.shape(vpr2)).astype(bool)
    mask = np.logical_and(mask, vpr2 == -9999)
    vpr2[mask] = np.nan

    #vob2 = np.array(all_observed_v).astype(float)
    #vpr2 = np.array(all_predicted_v).astype(float)
    # [t if t is not '' else 'hello' for t in test1]

    return vob2, t_ob, vpr2, t_pr

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

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

def get_alonshore_distance_etc(ixs,iys,fn_grd):

    DD = dict()
    
    DD['ixs'] = ixs
    DD['iys'] = iys

    ny, nd = np.shape(ixs)


    with nc.Dataset(fn_grd) as ds:
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

def get_history_essential(fn_his,fn_grd):

    Dye = dict()       # this dict will have all the dye information 
    Shoreline = dict() # this dict will have all the shoreline info
    Sites = dict()     # this will have the data at the site locations
    
    # the numbers that separate low, medium, and high risk
    # based on log10 of total_dye concentration
    # threshholds are set here!!!
    thresh_hold = np.array([-4.5,-3])
    Shoreline['Thresh_holds'] = thresh_hold
    Sites['Thresh_holds'] = thresh_hold

    ix0 = 75 # this is the starting index to look to the right from 
    iymn = 0
    iymx = 1060
    iys = np.arange(iymn,iymx,1)
    hb0 = [1.25] # this is the depth to get the indices from
    with nc.Dataset(fn_grd) as ds:
        hb = ds.variables['h'][iys,ix0:] # the data to extract depth from
        hb_og = ds.variables['h'][:,:]
        lats = ds.variables['lat_rho'][iys,ix0:]
        Dye['Lat'] = ds.variables['lat_rho'][:,:]
        Dye['Lon'] = ds.variables['lon_rho'][:,:]
        iy,ix = np.shape(hb_og)

    ixs, _ = get_depth_indices(hb,lats,hb0)    
    ixs = ixs + ix0 # must return the starting point

    with nc.Dataset(fn_grd) as ds:
        lat = ds.variables['lat_rho'][:,:] # the data to extract depth from

    ny = len(ixs)
    lat2 = np.zeros((ny))
    for aa in np.arange(ny):
        lat2[aa] = lat[iys[aa],ixs[aa]]

   # fig, ax = plt.subplots()
   # ax.contourf(np.arange(ix),np.arange(iy),hb_og,levels=10)
   # lvls = [1.25, 1.5]
   # ax.contour(np.arange(ix),np.arange(iy),hb_og, levels=lvls)
   # ax.plot(ixs,iys,'r')

    DD = get_alonshore_distance_etc(ixs,iys,fn_grd)
    #print(DD.keys())
    Shoreline['Lat'] = DD['lat']
    Shoreline['Lon'] = DD['lon']


    with nc.Dataset(fn_his) as his:
        times = his.variables['ocean_time']
        times2 = num2date(times[:], times.units)
        times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])

    Shoreline['Times'] = times2
    Dye['Times'] = times2
    Sites['Times'] = times2

    nt = len(times2) # these are the times for the history file
    nlat,nlon = np.shape( Dye['Lat'] )
    dyetot = np.zeros((nt,nlat,nlon))
    #nt = 60
    dye_01 = np.zeros((nt,ny))
    dye_02 = np.zeros((nt,ny))
    for aa in np.arange(0,nt,1):
        with nc.Dataset(fn_his) as his:
            hh = his.variables['h'][iys,ix0:] + np.squeeze( his.variables['zeta'][aa,iys,ix0:] )
            dye1 = his.variables['dye_01'][aa,-1,:,:] # get the surface
            dye2 = his.variables['dye_02'][aa,-1,:,:]
            msk = np.squeeze( his.variables['wetdry_mask_rho'][aa,:,:] )
            tmp = np.squeeze( dye1 ) + np.squeeze( dye2 ) 
            tmp[msk==0] = np.nan
            dyetot[aa,:,:] = tmp         
            ixs,_ = get_depth_indices(hh,lats,hb0)
            ixs = ixs + ix0
            for bb in np.arange(ny):
                dye_01[aa,bb] = np.squeeze(dye1[iys[bb],ixs[bb]])
                dye_02[aa,bb] = np.squeeze(dye2[iys[bb],ixs[bb]])

    Dye['Dye_tot'] = dyetot
    dyeTot = dye_01 + dye_02
    Shoreline['Dye_tot'] = dyeTot
    l10dyeTot = np.log10( dyeTot )
    risk = 0*dyeTot
    #print(thresh_hold)
    #print(thresh_hold[0])
    #print(thresh_hold[1])
    msk1 = (l10dyeTot>thresh_hold[0]) & (l10dyeTot<thresh_hold[1])
    msk2 = l10dyeTot>=thresh_hold[1]
    risk[msk1] = 1
    risk[msk2] = 2
    Shoreline['Risk']=risk
    
    # PTJ, border, TJRE, IB pier, Silver Strand, HdC
    ln_lab = ['PTJ','border','TJRE','IB pier','Silver Strand','HdC']
    Sites['Names'] = ln_lab
    lts0 = [32.52, 32.534, 32.552, 32.58, 32.625, 32.678]
    ipts = np.zeros((len(lts0)),dtype=int)

    lat2 = np.zeros(len(lts0))
    lon2 = 0 * lat2

    for aa in np.arange(len(lts0)):
        dlt = DD['lat']-lts0[aa]
        ipts[aa] = np.argmin(np.square(dlt))
        lat2[aa] = DD['lat'][ipts[aa]]
        lon2[aa] = DD['lon'][ipts[aa]]

    Sites['Lat'] = lat2
    Sites['Lon'] = lon2
    Sites['Risk'] = Shoreline['Risk'][:,ipts]
    Sites['Dye_tot'] = Shoreline['Dye_tot'][:,ipts]
    
    return Dye, Shoreline, Sites

def get_shoreline_data(fn_grd):
    ix0 = 75 # this is the starting index to look to the right from 
    iymn = 0
    iymx = 1060
    iys = np.arange(iymn,iymx,1)
    hb0 = [1.25] # this is the depth to get the indices from
    with nc.Dataset(fn_grd) as ds:
        hb = ds.variables['h'][iys,ix0:] # the data to extract depth from
        hb_og = ds.variables['h'][:,:]
        lats = ds.variables['lat_rho'][iys,ix0:]
        iy,ix = np.shape(hb_og)
    
    ixs, _ = get_depth_indices(hb,lats,hb0)    
    ixs = ixs + ix0 # must return the starting point

    with nc.Dataset(fn_grd) as ds:
        lat = ds.variables['lat_rho'][:,:] # the data to extract depth from

    ny = len(ixs)
    lat2 = np.zeros((ny))
    for aa in np.arange(ny):
        lat2[aa] = lat[iys[aa],ixs[aa]]

    DD = get_alonshore_distance_etc(ixs,iys,fn_grd)

    return DD


def make_dye_plots(fn_grd,fn_his):
    # this function makes dye plots for a given history file, and associated grid file
    PFM = get_PFM_info()
    yyyymmddhh = PFM['fetch_time'].strftime('%Y%m%d%H')

    fn1 = PFM['lv4_plot_dir'] + '/' + 'dye_01_hov_' + yyyymmddhh + '.png'
    fn2 = PFM['lv4_plot_dir'] + '/' + 'dye_02_hov_' + yyyymmddhh + '.png'
    fn3 = PFM['lv4_plot_dir'] + '/' + 'dye_01_ofl_' + yyyymmddhh + '.png'
    fn4 = PFM['lv4_plot_dir'] + '/' + 'dye_02_ofl_' + yyyymmddhh + '.png'
    fn5 = PFM['lv4_plot_dir'] + '/' + 'dye_01_oft_' + yyyymmddhh + '.png'

    sv_fig = '1'

    ix0 = 75 # this is the starting index to look to the right from 
    iymn = 0
    iymx = 1060
    iys = np.arange(iymn,iymx,1)
    hb0 = [1.25] # this is the depth to get the indices from
    with nc.Dataset(fn_grd) as ds:
        hb = ds.variables['h'][iys,ix0:] # the data to extract depth from
        hb_og = ds.variables['h'][:,:]
        lats = ds.variables['lat_rho'][iys,ix0:]
        iy,ix = np.shape(hb_og)

    ixs, _ = get_depth_indices(hb,lats,hb0)    
    ixs = ixs + ix0 # must return the starting point

    with nc.Dataset(fn_grd) as ds:
        lat = ds.variables['lat_rho'][:,:] # the data to extract depth from

    ny = len(ixs)
    lat2 = np.zeros((ny))
    for aa in np.arange(ny):
        lat2[aa] = lat[iys[aa],ixs[aa]]

    fig, ax = plt.subplots()
    ax.contourf(np.arange(ix),np.arange(iy),hb_og,levels=10)
    lvls = [1.25, 1.5]
    ax.contour(np.arange(ix),np.arange(iy),hb_og, levels=lvls)
    ax.plot(ixs,iys,'r')

    DD = get_alonshore_distance_etc(ixs,iys,fn_grd)

    with nc.Dataset(fn_his) as his:
        times = his.variables['ocean_time']
        times2 = num2date(times[:], times.units)
        times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])

    nt = len(times2) # these are the times for the history file
    #nt = 60
    dye_01 = np.zeros((nt,ny))
    dye_02 = np.zeros((nt,ny))
    for aa in np.arange(0,nt,1):
        with nc.Dataset(fn_his) as his:
            hh = his.variables['h'][iys,ix0:] + np.squeeze( his.variables['zeta'][aa,iys,ix0:] )
            dye1 = his.variables['dye_01'][aa,-1,:,:] # get the surface
            dye2 = his.variables['dye_02'][aa,-1,:,:]             
            ixs,_ = get_depth_indices(hh,lats,hb0)
            ixs = ixs + ix0
            for bb in np.arange(ny):
                dye_01[aa,bb] = np.squeeze(dye1[iys[bb],ixs[bb]])
                dye_02[aa,bb] = np.squeeze(dye2[iys[bb],ixs[bb]])

    # PTJ, border, TJRE, IB pier, Silver Strand, HdC
    ln_lab = ['PTJ','border','TJRE','IB pier','Silver Strand','HdC']
    lts0 = [32.52, 32.534, 32.552, 32.58, 32.625, 32.678]
    ipts = np.zeros((len(lts0)),dtype=int)
    for aa in np.arange(len(lts0)):
        dlt = DD['lat']-lts0[aa]
        ipts[aa] = np.argmin(np.square(dlt))

    fig, ax = plt.subplots()
    plevs = np.arange(-5.5,-0.5,.5)
    cmap = plt.get_cmap('magma_r')
    lD = np.log10(dye_01)
    lD1 = lD[:,ipts[3:5]]
    l = DD['l'][:,0]
    loff = 3.32
    cset = ax.contourf(l/1000 - loff,times2,lD, plevs, cmap=cmap, extend="both", vmin=-5, vmax=-1)        
    cbar = fig.colorbar(cset, ax=ax, orientation='vertical', pad = 0.05)
    #ax.contour( l/1000 - loff, times2, lD , levels = [-5, -4, -3], colors=['g','y','r'], linewidths=[3,3,3])        

    cnt=0
    for i0 in ipts:
        ax.plot([l[i0]/1000-loff,l[i0]/1000-loff], [times2[0],times2[-1]],'--k')
        ax.text(l[i0]/1000 - loff + .15, times2[0]+timedelta(hours=2),ln_lab[cnt],rotation=90,fontsize=10)
        cnt=cnt+1

    ax.set_xlabel('distance from PB [km]')
    ax.set_title('log10( dye_01 )')

    if sv_fig == '1':
        plt.savefig(fn1, dpi=300)
    else:
        plt.show()


    fig, ax = plt.subplots()
    colors = iter(plt.cm.turbo(np.linspace(0, 1, len(np.arange(0,nt,24)))))
    for it in np.arange(0,nt,24):
        clr = next(colors)
        ax.plot(l[1:]/1000 - loff,lD[it,1:], color=clr, label = str(times2[it]))

    ax.legend(loc='lower left',fontsize = 8)

    cnt=0
    for i0 in ipts:
        ax.plot([l[i0]/1000-loff,l[i0]/1000-loff], [-6,-1],'--k')
        ax.text(l[i0]/1000 - loff + .15, -2.2 ,ln_lab[cnt],rotation=90,fontsize=10)
        cnt=cnt+1

    ax.set_ylim([-6,-1])
    ax.set_xlabel('distance from PB [km]')
    ax.set_ylabel('log10( dye_01 )')
    t1str = times2[0].strftime("%Y-%m-%d %HZ")
    t2str = times2[-1].strftime("%Y-%m-%d %HZ")
    ax.set_title('dye from PB')

    if sv_fig == '1':
        plt.savefig(fn3, dpi=300)
    else:
        plt.show()


    fig, ax = plt.subplots()
    lD = np.log10(dye_02)
    lD2 = lD[:,ipts[3:5]]
    loff = 3.32 + 14.9
    cset = ax.contourf(l/1000 - loff,times2,lD, plevs, cmap=cmap, extend="both", vmin=-5, vmax=-1)        
    cbar = fig.colorbar(cset, ax=ax, orientation='vertical', pad = 0.05)
    #ax.contour( l/1000 - loff, times2, lD , levels = [-5, -4, -3], colors=['g','y','r'], linewidths=[3,3,3])        

    cnt=0
    for i0 in ipts:
        ax.plot([l[i0]/1000-loff,l[i0]/1000-loff], [times2[0],times2[-1]],'--k')
        ax.text(l[i0]/1000 - loff + .15, times2[0]+timedelta(hours=2),ln_lab[cnt],rotation=90,fontsize=10)
        cnt=cnt+1

    ax.set_xlabel('distance from TJRE [km]')
    ax.set_title('log10( dye_02 )')

    if sv_fig == '1':
        plt.savefig(fn2, dpi=300)
    else:
        plt.show()


    fig, ax = plt.subplots()
    colors = iter(plt.cm.turbo(np.linspace(0, 1, len(np.arange(0,nt,24)))))
    for it in np.arange(0,nt,24):
        clr = next(colors)
        ax.plot(l[1:]/1000 - loff,lD[it,1:], color=clr, label = str(times2[it]))

    ax.legend(loc='upper left',fontsize = 8)

    cnt=0
    for i0 in ipts:
        ax.plot([l[i0]/1000-loff,l[i0]/1000-loff], [-6,-1],'--k')
        ax.text(l[i0]/1000 - loff + .15, -2.2 ,ln_lab[cnt],rotation=90,fontsize=10)
        cnt=cnt+1

    ax.set_ylim([-6,-1])
    ax.set_xlabel('distance from TJRE [km]')
    ax.set_ylabel('log10( dye_02 )')
    t1str = times2[0].strftime("%Y-%m-%d %HZ")
    t2str = times2[-1].strftime("%Y-%m-%d %HZ")
    ax.set_title('dye from TJRE')
        
    if sv_fig == '1':
        plt.savefig(fn4, dpi=300)
    else:
        plt.show()


    fig, ax = plt.subplots()
    ax.plot(times2, lD1[:,0], '-b' ,label = 'dye_01 at IB')
    ax.plot(times2, lD1[:,1], '--b', label = 'dye_01 at SS')
    #ax.plot(times2, lD2[:,0], '-g', label = 'dye_02 at IB')
    #ax.plot(times2, lD2[:,1], '--g', label = 'dye_02 at SS')

    ax.legend(loc='upper left',fontsize = 8)

    ax.set_ylim([-6,-1])
    ax.set_ylabel('log10( dye )')
    ax.set_xlabel('date [UTC]')
    #ax.set_title('dye_01 (from PB), dye_02 (from TJRE)')

    if sv_fig == '1':
        plt.savefig(fn5, dpi=300)
    else:
        plt.show()

def make_dye_plots_v2(fn_grd,fn_his,dir_out):
    # this function makes dye plots for a given history file, and associated grid file


    sv_fig = '1'

    ix0 = 75 # this is the starting index to look to the right from 
    iymn = 0
    iymx = 1060
    iys = np.arange(iymn,iymx,1)
    hb0 = [1.25] # this is the depth to get the indices from
    with nc.Dataset(fn_grd) as ds:
        hb = ds.variables['h'][iys,ix0:] # the data to extract depth from
        hb_og = ds.variables['h'][:,:]
        lats = ds.variables['lat_rho'][iys,ix0:]
        iy,ix = np.shape(hb_og)

    ixs, _ = get_depth_indices(hb,lats,hb0)    
    ixs = ixs + ix0 # must return the starting point

    with nc.Dataset(fn_grd) as ds:
        lat = ds.variables['lat_rho'][:,:] # the data to extract depth from

    ny = len(ixs)
    lat2 = np.zeros((ny))
    for aa in np.arange(ny):
        lat2[aa] = lat[iys[aa],ixs[aa]]

    fig, ax = plt.subplots()
    ax.contourf(np.arange(ix),np.arange(iy),hb_og,levels=10)
    lvls = [1.25, 1.5]
    ax.contour(np.arange(ix),np.arange(iy),hb_og, levels=lvls)
    ax.plot(ixs,iys,'r')

    DD = get_alonshore_distance_etc(ixs,iys,fn_grd)

    with nc.Dataset(fn_his) as his:
        times = his.variables['ocean_time']
        times2 = num2date(times[:], times.units)
        times2 = np.array([datetime(year=date.year, month=date.month, day=date.day, 
                              hour=date.hour, minute=date.minute, second=date.second) for date in times2])

    yyyymmddhh = times2[0].strftime('%Y%m%d%H')
    fn1 = dir_out + '/' + 'dye_01_hov_' + yyyymmddhh + '.png'
    fn2 = dir_out + '/' + 'dye_02_hov_' + yyyymmddhh + '.png'
    fn3 = dir_out + '/' + 'dye_01_ofl_' + yyyymmddhh + '.png'
    fn4 = dir_out + '/' + 'dye_02_ofl_' + yyyymmddhh + '.png'
    fn5 = dir_out + '/' + 'dye_01_oft_' + yyyymmddhh + '.png'


    nt = len(times2) # these are the times for the history file
    #nt = 60
    dye_01 = np.zeros((nt,ny))
    dye_02 = np.zeros((nt,ny))
    for aa in np.arange(0,nt,1):
        with nc.Dataset(fn_his) as his:
            hh = his.variables['h'][iys,ix0:] + np.squeeze( his.variables['zeta'][aa,iys,ix0:] )
            dye1 = his.variables['dye_01'][aa,-1,:,:] # get the surface
            dye2 = his.variables['dye_02'][aa,-1,:,:]             
            ixs,_ = get_depth_indices(hh,lats,hb0)
            ixs = ixs + ix0
            for bb in np.arange(ny):
                dye_01[aa,bb] = np.squeeze(dye1[iys[bb],ixs[bb]])
                dye_02[aa,bb] = np.squeeze(dye2[iys[bb],ixs[bb]])

    # PTJ, border, TJRE, IB pier, Silver Strand, HdC
    ln_lab = ['PTJ','border','TJRE','IB pier','Silver Strand','HdC']
    lts0 = [32.52, 32.534, 32.552, 32.58, 32.625, 32.678]
    ipts = np.zeros((len(lts0)),dtype=int)
    for aa in np.arange(len(lts0)):
        dlt = DD['lat']-lts0[aa]
        ipts[aa] = np.argmin(np.square(dlt))

    fig, ax = plt.subplots()
    plevs = np.arange(-5.5,-0.5,.5)
    cmap = plt.get_cmap('magma_r')
    lD = np.log10(dye_01)
    lD1 = lD[:,ipts[3:5]]
    l = DD['l'][:,0]
    loff = 3.32
    cset = ax.contourf(l/1000 - loff,times2,lD, plevs, cmap=cmap, extend="both", vmin=-5, vmax=-1)        
    cbar = fig.colorbar(cset, ax=ax, orientation='vertical', pad = 0.05)
    #ax.contour( l/1000 - loff, times2, lD , levels = [-5, -4, -3], colors=['g','y','r'], linewidths=[3,3,3])        

    cnt=0
    for i0 in ipts:
        ax.plot([l[i0]/1000-loff,l[i0]/1000-loff], [times2[0],times2[-1]],'--k')
        ax.text(l[i0]/1000 - loff + .15, times2[0]+timedelta(hours=2),ln_lab[cnt],rotation=90,fontsize=10)
        cnt=cnt+1

    ax.set_xlabel('distance from PB [km]')
    ax.set_title('log10( dye_01 )')

    if sv_fig == '1':
        plt.savefig(fn1, dpi=300)
    else:
        plt.show()


    fig, ax = plt.subplots()
    colors = iter(plt.cm.turbo(np.linspace(0, 1, len(np.arange(0,nt,24)))))
    for it in np.arange(0,nt,24):
        clr = next(colors)
        ax.plot(l[1:]/1000 - loff,lD[it,1:], color=clr, label = str(times2[it]))

    ax.legend(loc='lower left',fontsize = 8)

    cnt=0
    for i0 in ipts:
        ax.plot([l[i0]/1000-loff,l[i0]/1000-loff], [-6,-1],'--k')
        ax.text(l[i0]/1000 - loff + .15, -2.2 ,ln_lab[cnt],rotation=90,fontsize=10)
        cnt=cnt+1

    ax.set_ylim([-6,-1])
    ax.set_xlabel('distance from PB [km]')
    ax.set_ylabel('log10( dye_01 )')
    t1str = times2[0].strftime("%Y-%m-%d %HZ")
    t2str = times2[-1].strftime("%Y-%m-%d %HZ")
    ax.set_title('dye from PB')

    if sv_fig == '1':
        plt.savefig(fn3, dpi=300)
    else:
        plt.show()


    fig, ax = plt.subplots()
    lD = np.log10(dye_02)
    lD2 = lD[:,ipts[3:5]]
    loff = 3.32 + 14.9
    cset = ax.contourf(l/1000 - loff,times2,lD, plevs, cmap=cmap, extend="both", vmin=-5, vmax=-1)        
    cbar = fig.colorbar(cset, ax=ax, orientation='vertical', pad = 0.05)
    #ax.contour( l/1000 - loff, times2, lD , levels = [-5, -4, -3], colors=['g','y','r'], linewidths=[3,3,3])        

    cnt=0
    for i0 in ipts:
        ax.plot([l[i0]/1000-loff,l[i0]/1000-loff], [times2[0],times2[-1]],'--k')
        ax.text(l[i0]/1000 - loff + .15, times2[0]+timedelta(hours=2),ln_lab[cnt],rotation=90,fontsize=10)
        cnt=cnt+1

    ax.set_xlabel('distance from TJRE [km]')
    ax.set_title('log10( dye_02 )')

    if sv_fig == '1':
        plt.savefig(fn2, dpi=300)
    else:
        plt.show()


    fig, ax = plt.subplots()
    colors = iter(plt.cm.turbo(np.linspace(0, 1, len(np.arange(0,nt,24)))))
    for it in np.arange(0,nt,24):
        clr = next(colors)
        ax.plot(l[1:]/1000 - loff,lD[it,1:], color=clr, label = str(times2[it]))

    ax.legend(loc='upper left',fontsize = 8)

    cnt=0
    for i0 in ipts:
        ax.plot([l[i0]/1000-loff,l[i0]/1000-loff], [-6,-1],'--k')
        ax.text(l[i0]/1000 - loff + .15, -2.2 ,ln_lab[cnt],rotation=90,fontsize=10)
        cnt=cnt+1

    ax.set_ylim([-6,-1])
    ax.set_xlabel('distance from TJRE [km]')
    ax.set_ylabel('log10( dye_02 )')
    t1str = times2[0].strftime("%Y-%m-%d %HZ")
    t2str = times2[-1].strftime("%Y-%m-%d %HZ")
    ax.set_title('dye from TJRE')
        
    if sv_fig == '1':
        plt.savefig(fn4, dpi=300)
    else:
        plt.show()


    fig, ax = plt.subplots()
    ax.plot(times2, lD1[:,0], '-b' ,label = 'dye_01 at IB')
    ax.plot(times2, lD1[:,1], '--b', label = 'dye_01 at SS')
    #ax.plot(times2, lD2[:,0], '-g', label = 'dye_02 at IB')
    #ax.plot(times2, lD2[:,1], '--g', label = 'dye_02 at SS')

    ax.legend(loc='upper left',fontsize = 8)

    ax.set_ylim([-6,-1])
    ax.set_ylabel('log10( dye )')
    ax.set_xlabel('date [UTC]')
    #ax.set_title('dye_01 (from PB), dye_02 (from TJRE)')

    if sv_fig == '1':
        plt.savefig(fn5, dpi=300)
    else:
        plt.show()



if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])


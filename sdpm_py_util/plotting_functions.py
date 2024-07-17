import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import netCDF4 as nc

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
    axx.contour(RMG['lon_rho'], RMG['lat_rho'], RMG['h'], levels=[5, 10], colors='k')

#Function for timestamp
def extract_timestamp(ATM):
    """
    Extracts the timestamp from the ATM data dictionary.

    Parameters:
    ATM (dict): The atmospheric data dictionary.

    Returns:
    str: Formatted timestamp string.
    """
    base_time = datetime(1970, 1, 1)
    time_offset = ATM['ocean_time'][0]  # Assuming 'ocean_time' is in days since 1970-01-01
    timestamp = base_time + timedelta(days=time_offset)
    return timestamp.strftime('%Y%m%d_%H%M%S')

# ATM Fields Plotting Function
def plot_atm_fields(ATM, RMG, PFM, fields_to_plot=None):
    """
    Plot specified fields from the ATM dataset with timestamps and product names, and save them as PNG files.
    
    Parameters:
    ATM (dict): The atmospheric data dictionary.
    RMG (dict): The ROMS grid data dictionary.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    IMPORTANT: we might need to keep changing the plevs and the number of ticks as well as number of levels manually for now!
    """
    timestamp = extract_timestamp(ATM)
    lon = ATM['lon']
    lat = ATM['lat']
    ocean_time = ATM['ocean_time']
    start_time = datetime(1970, 1, 1) + timedelta(days=float(ocean_time[0]))
    forecast_hours = (ocean_time - ocean_time[0]) * 24  # Convert days to hours
    
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
            U, V = ATM['Uwind'][0, :, :], ATM['Vwind'][0, :, :]
            magnitude = np.sqrt(U**2 + V**2)
            cset = ax.contourf(lon, lat, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon[::10], lat[::10], U[::10, ::10], V[::10, ::10], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 16, 5))
            ax.set_title('10 m velocity [m/s, every 10 grid points]')
        
        elif field == 'pressure':
            plevs = np.arange(1000, 1026, 5)
            cset = ax.contourf(lon, lat, ATM['Pair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(1000, 1026, 5))
            ax.set_title('Surface Pressure [Pa]')
        
        elif field == 'temperature':
            plevs = np.arange(12, 61, 2)
            cset = ax.contourf(lon, lat, ATM['Tair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(12, 61, 12))
            ax.set_title('Surface Temperature [K]')
        
        elif field == 'lw_radiation':
            plevs = np.arange(260, 520, 15)
            cset = ax.contourf(lon, lat, ATM['lwrad_down'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(260, 520, 50))
            ax.set_title('Long Wave Radiation Down [W/m^2]')
        
        elif field == 'rain':
            plevs = np.arange(0, 0.0035, 0.0001)
            cset = ax.contourf(lon, lat, ATM['rain'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 0.0035, 0.0007))
            ax.set_title('Precipitation Rate [kg/m^2/s]')
        
        elif field == 'humidity':
            plevs = np.arange(0, 101, 1)
            cset = ax.contourf(lon, lat, ATM['Qair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 101, 20))
            ax.set_title('Surface Humidity [%]')
        
        elif field == 'swrad':
            plevs = np.arange(0, 601, 10)
            cset = ax.contourf(lon, lat, ATM['swrad'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(100, 601, 100))
            ax.set_title('Short Wave Radiation Down [W/m^2]')
        
        plot_roms_box(ax, RMG)
        plot_roms_coastline(ax, RMG)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)
        ax.set_aspect(aspect='auto')
        ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
        ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: nam_nest | Forecast Hour: {forecast_hours[0]:.1f}'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
    
        output_dir = PFM['lv1_plot_dir']
        filename = f'{output_dir}/{timestamp}_nam_nest_ATM_{field}.png'
        plt.savefig(filename, dpi=300)
        plt.tight_layout()
        plt.show()

def plot_atm_r_fields(ATM_R, RMG, PFM, fields_to_plot=None, flag=True):
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
    start_time = datetime(1970, 1, 1) + timedelta(days=float(ocean_time[0]))
    forecast_hours = (ocean_time - ocean_time[0]) * 24  # Convert days to hours
    
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
            U, V = ATM_R['Uwind'][0, :, :], ATM_R['Vwind'][0, :, :]
            magnitude = np.sqrt(U**2 + V**2)
            cset = ax.contourf(lon_r, lat_r, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax.quiver(lon_r[::10, ::10], lat_r[::10, ::10], U[::10, ::10], V[::10, ::10], transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 16, 5))
            ax.set_title('10 m velocity [m/s, on ROMS grid]')
        
        elif field == 'pressure':
            plevs = np.arange(1000, 1026, 5)
            cset = ax.contourf(lon_r, lat_r, ATM_R['Pair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(1000, 1026, 5))
            ax.set_title('Surface Pressure [Pa]')
        
        elif field == 'temperature':
            plevs = np.arange(12, 61, 2)
            cset = ax.contourf(lon_r, lat_r, ATM_R['Tair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(12, 61, 12))
            ax.set_title('Surface Temperature [K]')
        
        elif field == 'lw_radiation':
            plevs = np.arange(260, 520, 15)
            cset = ax.contourf(lon_r, lat_r, ATM_R['lwrad_down'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(260, 520, 50))
            ax.set_title('Long Wave Radiation Down [W/m^2]')
        
        elif field == 'rain':
            plevs = np.arange(0, 0.0035, 0.0001)
            cset = ax.contourf(lon_r, lat_r, ATM_R['rain'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 0.0035, 0.0007))
            ax.set_title('Precipitation Rate [kg/m^2/s]')
        
        elif field == 'humidity':
            plevs = np.arange(0, 101, 1)
            cset = ax.contourf(lon_r, lat_r, ATM_R['Qair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cset, ax=ax, orientation='horizontal', pad = 0.05)
            cbar.set_ticks(np.arange(0, 101, 20))
            ax.set_title('Surface Humidity [%]')
        
        elif field == 'swrad':
            plevs = np.arange(0, 601, 10)
            cset = ax.contourf(lon_r, lat_r, ATM_R['swrad'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
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

        annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: nam_nest | Forecast Hour: {forecast_hours[0]:.1f}'
        ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        output_dir = PFM['lv1_plot_dir']
        if flag is True:
            filename = f'{output_dir}/{timestamp}_nam_nest_ATM_R_{field}.png'
            plt.savefig(filename, dpi=300)
            plt.tight_layout()
            plt.show()
        else:
            filename = f'{output_dir}/exported_plots_{field}.png'
            plt.savefig(filename, dpi=300)
            plt.tight_layout()
            plt.show()

# For both ATM and ATM_R fields
def plot_all_fields_in_one(ATM, ATM_R, RMG, PFM, fields_to_plot=None):
    """
    Plot specified fields from both the ATM and ATM_R datasets with timestamps and product names, and save them in separate PNG files.
    
    Parameters:
    ATM (dict): The atmospheric data dictionary.
    ATM_R (dict): The atmospheric data dictionary on ROMS grid.
    RMG (dict): The ROMS grid data dictionary.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    """
    timestamp = extract_timestamp(ATM)
    lon = ATM['lon']
    lat = ATM['lat']
    lon_r = ATM_R['lon']
    lat_r = ATM_R['lat']
    ocean_time = ATM['ocean_time']
    start_time = datetime(1970, 1, 1) + timedelta(days=float(ocean_time[0]))
    forecast_hours = (ocean_time - ocean_time[0]) * 24  # Convert days to hours
    
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
            plevs = np.arange(0, 16, 0.1)
            U, V = ATM['Uwind'][0, :, :], ATM['Vwind'][0, :, :]
            magnitude = np.sqrt(U**2 + V**2)
            cset1 = ax1.contourf(lon, lat, magnitude, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax1.quiver(lon[::10], lat[::10], U[::10, ::10], V[::10, ::10], transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            cbar1.set_ticks(np.arange(0, 16, 5))
            ax1.set_title('10 m velocity [m/s, ATM]')

            U_R, V_R = ATM_R['Uwind'][0, :, :], ATM_R['Vwind'][0, :, :]
            magnitude_R = np.sqrt(U_R**2 + V_R**2)
            cset2 = ax2.contourf(lon_r, lat_r, magnitude_R, plevs, cmap=cmap, transform=ccrs.PlateCarree())
            ax2.quiver(lon_r[::10, ::10], lat_r[::10, ::10], U_R[::10, ::10], V_R[::10, ::10], transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            cbar2.set_ticks(np.arange(0, 16, 5))
            ax2.set_title('10 m velocity [m/s, ATM_R]')
        
        elif field == 'pressure':
            plevs = np.arange(1000, 1026, 5)
            cset1 = ax1.contourf(lon, lat, ATM['Pair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            cbar1.set_ticks(np.arange(1000, 1026, 5))
            ax1.set_title('Surface Pressure [Pa, ATM]')

            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Pair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            cbar2.set_ticks(np.arange(1000, 1026, 5))
            ax2.set_title('Surface Pressure [Pa, ATM_R]')
        
        elif field == 'temperature':
            plevs = np.arange(12, 61, 2)
            cset1 = ax1.contourf(lon, lat, ATM['Tair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            cbar1.set_ticks(np.arange(12, 61, 12))
            ax1.set_title('Surface Temperature [K, ATM]')

            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Tair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            cbar2.set_ticks(np.arange(12, 61, 12))
            ax2.set_title('Surface Temperature [K, ATM_R]')
        
        elif field == 'lw_radiation':
            plevs = np.arange(260, 520, 15)
            cset1 = ax1.contourf(lon, lat, ATM['lwrad_down'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            cbar1.set_ticks(np.arange(260, 520, 50))
            ax1.set_title('Long Wave Radiation Down [W/m^2, ATM]')

            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['lwrad_down'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            cbar2.set_ticks(np.arange(260, 520, 50))
            ax2.set_title('Long Wave Radiation Down [W/m^2, ATM_R]')
        
        elif field == 'rain':
            plevs = np.arange(0, 0.0035, 0.0001)
            cset1 = ax1.contourf(lon, lat, ATM['rain'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            cbar1.set_ticks(np.arange(0, 0.0035, 0.0007))
            ax1.set_title('Precipitation Rate [kg/m^2/s, ATM]')

            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['rain'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            cbar2.set_ticks(np.arange(0, 0.0035, 0.0007))
            ax2.set_title('Precipitation Rate [kg/m^2/s, ATM_R]')
        
        elif field == 'humidity':
            plevs = np.arange(0, 101, 1)
            cset1 = ax1.contourf(lon, lat, ATM['Qair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            cbar1.set_ticks(np.arange(0, 101, 20))
            ax1.set_title('Surface Humidity [%, ATM]')

            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['Qair'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            cbar2.set_ticks(np.arange(0, 101, 20))
            ax2.set_title('Surface Humidity [%, ATM_R]')
        
        elif field == 'swrad':
            plevs = np.arange(0, 601, 10)
            cset1 = ax1.contourf(lon, lat, ATM['swrad'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar1 = fig.colorbar(cset1, ax=ax1, orientation='horizontal', pad=0.05)
            cbar1.set_ticks(np.arange(100, 601, 100))
            ax1.set_title('Short Wave Radiation Down [W/m^2, ATM]')

            cset2 = ax2.contourf(lon_r, lat_r, ATM_R['swrad'][0, :, :], plevs, cmap=cmap, transform=ccrs.PlateCarree())
            cbar2 = fig.colorbar(cset2, ax=ax2, orientation='horizontal', pad=0.05)
            cbar2.set_ticks(np.arange(100, 601, 100))
            ax2.set_title('Short Wave Radiation Down [W/m^2, ATM_R]')
        
        for ax in [ax1, ax2]:
            plot_roms_box(ax, RMG)
            plot_roms_coastline(ax, RMG)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            ax.set_aspect(aspect='auto')
            ax.set_xticks(np.round(np.linspace(np.min(lon), np.max(lon), num=5), 2))
            ax.set_yticks(np.round(np.linspace(np.min(lat), np.max(lat), num=5), 2))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
            annotation = f'Timestamp: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | Model: nam_nest | Forecast Hour: {forecast_hours[0]:.1f}'
            ax.text(0.5, 1.05, annotation, transform=ax.transAxes, ha='center', fontsize=12)
        
        # Save the plot for each field
        # output_dir = "C:/Users/abhis/Downloads"
        output_dir = PFM['lv1_plot_dir']
        filename = f'{output_dir}/{timestamp}_nam_nest_ATMandATMR_{field}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        

def load_and_plot_atm(PFM, fields_to_plot=None):
    """
    Load the atm.nc file and plot specified fields.

    Parameters:
    file_path (str): Path to the atm.nc file.
    RMG (dict): The ROMS grid data dictionary.
    product_name (str): The name of the forecast model.
    fields_to_plot (list or str): The fields to plot. If None, plot all fields.
    """
    # Load the atm.nc file
    file_path = PFM['lv1_forc_dir'] + '/ATM_FORCING.nc' 
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


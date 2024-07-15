# grid functions
import netCDF4 as nc

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

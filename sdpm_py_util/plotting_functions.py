def plot_roms_box(axx,RMG):
    xr1 = RMG['lon_rho'][0,:]
    yr1 = RMG['lat_rho'][0,:]
    xr2 = RMG['lon_rho'][:,0]
    yr2 = RMG['lat_rho'][:,0]
    xr3 = RMG['lon_rho'][-1,:]
    yr3 = RMG['lat_rho'][-1,:]
    xr4 = RMG['lon_rho'][:,-1]
    yr4 = RMG['lat_rho'][:,-1]
    axx.plot(xr1,yr1,'k-',linewidth=.5)
    axx.plot(xr2,yr2,'k-',linewidth=.5)
    axx.plot(xr3,yr3,'k-',linewidth=.5)
    axx.plot(xr4,yr4,'k-',linewidth=.5)

def plot_roms_coastline(axx,RMG):
    axx.contour(RMG['lon_rho'],RMG['lat_rho'],RMG['h'],levels=[5, 10],colors='k')

    
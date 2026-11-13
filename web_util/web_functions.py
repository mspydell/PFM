import os
import numpy as np
import xarray as xr
import sys
import subprocess
from datetime import datetime
sys.path.append('../sdpm_py_util')
import plotting_functions as pltfuns
import init_funs_forecast as initfuns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ent_alpha
#import grid_functions as grdfuns


def full_his_to_essential(his_fname,grd_fname,pkl_fnm):

    PFM = initfuns.get_model_info(pkl_fnm)
    # return the essential data from the history file
    # dye has dye[time,lat,lon]
    # shoreline has dye[time,position] and risk
    # sites has dye[time,6 positions] and risk
    dye, shore, sites = pltfuns.get_history_essential(his_fname,grd_fname) 

#    print(dye.keys())
#    print(shore.keys())
#    print(sites.keys())

    site_locations = ", ".join(sites['Names'][:])
    #print(site_locations)

    times = dye['Times'][:]
    dt = times - datetime.strptime('19990101','%Y%m%d')
    days_past = np.zeros(len(dt))
    for ii in np.arange(len(dt)):
        days_past[ii] = dt[ii].total_seconds() / (3600*24)

    #print(np.shape(shore['Lat']))

    # ---- Ent-calibrated dye: shoreline_Ent_dye = alpha(shoreline) * shoreline_dye_tot.
    # alpha is the 30-day trailing geometric mean of Ent/dye_tot at the ddPCR stations
    # (ent_alpha.py). Any failure falls back to the constant alpha and must never stop
    # the web file being written.
    sh_lat = np.squeeze(np.asarray(shore['Lat'], float))
    sh_lon = np.squeeze(np.asarray(shore['Lon'], float))
    nshore = len(sh_lat)
    alpha_status = 'constant'
    alpha_note = ''
    alpha_shore = np.full(nshore, ent_alpha.ALPHA0)
    alpha_info = None
    try:
        cdir = ent_alpha.cache_dir(PFM)
        os.makedirs(cdir, exist_ok=True)
        ent_alpha.seed_dye_cache(cdir)
        try:
            ent_alpha.update_dye_cache(cdir, times, shore['Dye_tot'], sh_lat, sh_lon)
        except Exception as e:
            print('ent_alpha: dye cache not updated:', e)
        alpha_shore, alpha_info = ent_alpha.compute(times[0], sh_lat, sh_lon, cdir)
        alpha_status = alpha_info['status']
    except Exception as e:
        alpha_note = 'error: ' + str(e)[:200]
        print('ent_alpha failed, using constant alpha:', e)

    ent_shore = alpha_shore[None, :] * np.asarray(shore['Dye_tot'], float)
    l10_ent_shore = np.log10(ent_shore)
    risk_alpha = ent_alpha.risk_from_ent(ent_shore)
    # the website sites are shoreline points (get_history_essential); find their indices
    ipts = np.array([int(np.argmin((sh_lat - la) ** 2 + (sh_lon - lo) ** 2))
                     for la, lo in zip(np.ravel(sites['Lat']), np.ravel(sites['Lon']))])
    ent_sites = ent_shore[:, ipts]

    st_names = list(ent_alpha.STATIONS)
    if alpha_info is not None:
        st_alpha = np.array([alpha_info['alpha'][s][0] for s in st_names], float)
        st_n = np.array([alpha_info['alpha'][s][1] for s in st_names], int)
        st_row = np.array([alpha_info['rows'][s] for s in st_names], int)
    else:
        st_alpha = np.full(len(st_names), np.nan)
        st_n = np.zeros(len(st_names), int)
        st_row = np.array([ent_alpha.station_rows(sh_lat, sh_lon)[s] for s in st_names], int)
    st_lat = np.array([ent_alpha.STATIONS[s][0] for s in st_names])
    st_lon = np.array([ent_alpha.STATIONS[s][1] for s in st_names])
    th_e = ent_alpha.THRESH_ENT

    ds = xr.Dataset(
        data_vars = dict(
            shoreline_dye_tot = (["ntime","nshore"],shore['Dye_tot'],{'long_name':'total dye concentration at shoreline, dye_01+dye_02','units':'waste water concetration','time':'time','coordinates':'shoreline'}),
            shoreline_risk = (["ntime","nshore"],shore['Risk'],{'long_name':'risk at shoreline, 0=low,1=medium,2=high','units':'none','time':'time','coordinates':'shoreline'}),
            map_dye_tot = (["ntime","nlat","nlon"],dye['Dye_tot'],{'long_name':'total dye concentration from history file, dye_01+dye_02','units':'waste water concetration','time':'time','coordinates':'lat,lon'}),
            sites_dye_tot = (["ntime","nsites"],sites['Dye_tot'],{'long_name':'total dye concentration at specific sites, dye_01+dye_02','units':'none','time':'time','coordinates':'site locations'}),
            sites_risk = (["ntime","nsites"],sites['Risk'],{'long_name':'risk at specific sites, 0=low,1=medium,2=high','units':'none','time':'time','coordinates':'site locations'}),
            shoreline_l10_dye_tot = (["ntime","nshore"],shore['l10_Dye_tot'],{'long_name':'log10 total dye concentration at shoreline, log10(dye_01+dye_02)','units':'waste water concetration','time':'time','coordinates':'shoreline'}),
            map_l10_dye_tot = (["ntime","nlat","nlon"],dye['l10_Dye_tot'],{'long_name':'log10 total dye concentration from history file, log10(dye_01+dye_02)','units':'waste water concetration','time':'time','coordinates':'lat,lon'}),
            sites_l10_dye_tot = (["ntime","nsites"],sites['l10_Dye_tot'],{'long_name':'log10 total dye concentration at specific sites, log10(dye_01+dye_02)','units':'none','time':'time','coordinates':'site locations'}),
            shoreline_Ent_dye = (["ntime","nshore"],ent_shore,{'long_name':'Ent-calibrated dye at shoreline, shoreline_alpha * shoreline_dye_tot','units':'copies/100 mL','time':'time','coordinates':'shoreline'}),
            shoreline_l10_Ent_dye = (["ntime","nshore"],l10_ent_shore,{'long_name':'log10 Ent-calibrated dye at shoreline, log10(shoreline_Ent_dye)','units':'log10 copies/100 mL','time':'time','coordinates':'shoreline'}),
            shoreline_risk_alpha = (["ntime","nshore"],risk_alpha,{'long_name':'risk at shoreline from shoreline_Ent_dye, 0=low,1=medium,2=high, thresholds thresh_holds_Ent','units':'none','time':'time','coordinates':'shoreline'}),
            sites_Ent_dye = (["ntime","nsites"],ent_sites,{'long_name':'Ent-calibrated dye at specific sites, from shoreline_Ent_dye','units':'copies/100 mL','time':'time','coordinates':'site locations'}),
            sites_l10_Ent_dye = (["ntime","nsites"],np.log10(ent_sites),{'long_name':'log10 Ent-calibrated dye at specific sites, log10(sites_Ent_dye)','units':'log10 copies/100 mL','time':'time','coordinates':'site locations'}),
            sites_risk_alpha = (["ntime","nsites"],risk_alpha[:, ipts],{'long_name':'risk at specific sites from sites_Ent_dye, 0=low,1=medium,2=high, thresholds thresh_holds_Ent','units':'none','time':'time','coordinates':'site locations'}),
            shoreline_alpha = (["nshore"],alpha_shore,{'long_name':'Ent/dye calibration alpha along the shoreline: 30-day trailing geometric mean of Ent/dye_tot at the ddPCR stations, log10-linear between stations, constant beyond the end stations','units':'copies/100 mL per unit dye concentration','coordinates':'shoreline'}),
            alpha_stations = (["nalpha"],st_alpha,{'long_name':'alpha fitted at each ddPCR station (NaN: too few pairs, not used)','units':'copies/100 mL per unit dye concentration'}),
            alpha_stations_npairs = (["nalpha"],st_n,{'long_name':'number of Ent/dye pairs in the 30-day window at each ddPCR station','units':'none'}),
            alpha_stations_shore_index = (["nalpha"],st_row,{'long_name':'shoreline index (nshore, 0-based) of each ddPCR station','units':'none'}),
            alpha_stations_lat = (["nalpha"],st_lat,{'long_name':'ddPCR station latitude','units':'degrees'}),
            alpha_stations_lon = (["nalpha"],st_lon,{'long_name':'ddPCR station longitude','units':'degrees'}),
            thresh_holds_Ent = (["nthresh"],th_e,{'long_name':'the 2 threshold values that separate low, medium, and high risk for the Ent-calibrated dye','units':'copies/100 mL'}),
        ),
        coords=dict(
            map_lat = (["nlat","nlon"],dye['Lat'],{'long_name':'latitudes on map','units':'degrees'}),
            map_lon = (["nlat","nlon"],dye['Lon'],{'long_name':'longitudes on map','units':'degrees'}),
            shoreline_lat =(["nshore"],np.squeeze(shore['Lat']),{'long_name':'shoreline latitudes','units':'degrees'}),
            shoreline_lon =(["nshore"],np.squeeze(shore['Lon']),{'long_name':'shoreline longitudes','units':'degrees'}),
            time = (["ntime"],np.squeeze(days_past),{'long_name':'time in days past Jan 1, 1999','units':'days'}),
            sites_lat = (["nsites"],sites['Lat'],{'long_name':'latitudes at sites','units':'degrees'}),
            sites_lon = (["nsites"],sites['Lon'],{'long_name':'longitudes at sites','units':'degrees'}),
            thresh_holds = (["nthresh"],shore['Thresh_holds'],{'long_name':'the 2 threshhold values that separate low, medium, and high risk','units':'log10 total dye concentration'}),
               ),
        attrs={'type':'data for website plotting',
            'time info':'time is in days since Jan 1 1999 UTC.',
            'risk info':'0 is low risk, log10 dye_tot < ' + str(shore['Thresh_holds'][0]) + ', 1 is medium, ' + str(shore['Thresh_holds'][0]) + '<log10 dye_tot <'+ str(shore['Thresh_holds'][1]) + ', high, log10 dye_tot >' + str(shore['Thresh_holds'][1]),
            'site info':'the site locations are ' + site_locations,
            'risk_alpha info':'shoreline_risk_alpha and sites_risk_alpha: 0 is low risk, Ent_dye < ' + str(th_e[0]) + ', 1 is medium, ' + str(th_e[0]) + ' < Ent_dye < ' + str(th_e[1]) + ', 2 is high, Ent_dye >= ' + str(th_e[1]) + ' copies/100 mL',
            'alpha_stations': ', '.join(st_names),
            'alpha_status': alpha_status + ((' ' + alpha_note) if alpha_note else ''),
            'alpha_info': 'alpha_status ok: fitted this run; last_good: no station had ' + str(ent_alpha.MIN_PAIRS) + ' pairs, alpha from the last good run within 7 days; constant: alpha = ' + '%.4g' % ent_alpha.ALPHA0 + ' everywhere (1413 copies/100 mL <-> dye_tot 1e-5). Window ' + str(ent_alpha.WINDOW_DAYS) + ' days before the run, Ent paired with mean dye within +-' + str(ent_alpha.PAIR_HOURS) + ' h.',
            },
        )

    fname_out = PFM['lv4_web_name_full'] 
    ds.to_netcdf(fname_out)


def plot_his_figures_fromnc(*args):
    # this function is a wrapper to plot all of the history file plots with the input being: 
    # (file_name_full,grid_file_full,dir_out,lvl,mod_type)
    # where
    # file_name_full = 'full/path/to/file_name.nc', 
    # grid_file_full = 'full/path/to/grid_file.nc
    # dir_out = 'the/output/directory/location/',
    # lvl = 'LV1', if ommitted the default is 'LV4' 
    # mod_type = 'ROMS', if omitted the default is 'COAWST' 

    narg = len(args)
    file_name_full = args[0]
    grid_name_full = args[1]
    dir_out = args[2]
    if narg == 3:
        lvl = 'LV4'
        lv4_model = 'COAWST'
    elif narg == 4:
        lvl = args[2]
    elif narg == 5:
        lvl = args[2]
        lv4_model = args[3] # here we can switch it back to ROMS if we want

    sv_fig = 1
    iz = -1
    if lvl == 'LV1':
        fn = file_name_full
        Ix = np.array([175,240])
        Iy = np.array([175,170])
    elif lvl == 'LV2':
        fn = file_name_full
        Ix = np.array([175,240])
        Iy = np.array([175,170])
    elif lvl == 'LV3': # 413 by 251
        fn = file_name_full
        Ix = np.array([210,227])
        Iy = np.array([325,200])
    elif lvl == 'LV4' and lv4_model == 'ROMS': # 413 by 251
        fn = file_name_full
        Ix = np.array([275,400])
        Iy = np.array([750,1000])
    elif lvl == 'LV4' and lv4_model == 'COAWST': # 413 by 251
        fn = file_name_full        
        Ix = np.array([275,400])
        Iy = np.array([750,1000])
        cmnH,cmxH = pltfuns.get_his_clims(fn,'Hwave',-1,'all')


    time = pltfuns.get_ocean_times_from_ncfile(fn)
    dtime = time[-1] - time[0]
    pfm_hrs = int( dtime.total_seconds() / 3600 ) # this should be an integer

    print('getting clims...')
    cmn,cmx = pltfuns.get_his_clims(fn,'temp',-1,'all')
    print('...done.')

    pltfuns.plot_ssh_his_tseries_v2(fn,grid_name_full,Ix,Iy,sv_fig,lvl,dir_out) # uncommented on 2/10/25, does this break things?
                                                 # maybe need to turn this into a subprocess
                                                 # or use "with" in the function?
    os.chdir('../sdpm_py_util')

    It=0
    while It<=pfm_hrs:
        cmd_list = ['python','-W','ignore','plotting_functions.py','plot_his_temps_wuv_v2',fn,str(It),str(iz),str(sv_fig),lvl,str(cmn),str(cmx),grid_name_full,dir_out] 
        if lvl == 'LV4':
            _ = subprocess.Popen(cmd_list)  
        else:
            _ = subprocess.run(cmd_list)   
        if lv4_model == 'COAWST' and lvl == 'LV4':
            #print(It)
            cmd_list = ['python','-W','ignore','plotting_functions.py','plot_lv4_coawst_his_v2',fn,str(It),str(iz),str(sv_fig),lvl,'dye_01','0','1',grid_name_full,dir_out] 
            _ = subprocess.Popen(cmd_list)     
            cmd_list = ['python','-W','ignore','plotting_functions.py','plot_lv4_coawst_his_v2',fn,str(It),str(iz),str(sv_fig),lvl,'dye_02','0','1',grid_name_full,dir_out] 
            _ = subprocess.Popen(cmd_list)     
            cmd_list = ['python','-W','ignore','plotting_functions.py','plot_lv4_coawst_his_v2',fn,str(It),str(iz),str(sv_fig),lvl,'Hwave',str(cmnH),str(cmxH),grid_name_full,dir_out] 
            if It<pfm_hrs:
                _ = subprocess.run(cmd_list)     
            else:
                _ = subprocess.run(cmd_list)
        It += 2

    cmd_list = ['python','-W','ignore','plotting_functions.py','make_dye_plots_v2',grid_name_full,fn,dir_out]
    os.chdir('../sdpm_py_util')
    _ = subprocess.run(cmd_list)   
    os.chdir('../driver')

if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])



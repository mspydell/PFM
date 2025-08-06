"""
This is the one place where you set the path structure for the PFM.
The info is stored in the dict PFM.
All? paths are pathlib.Path objects.
Users should copy this and edit it appropriately in get_model_info.
"""
 
import os
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
import grid_functions as grdfuns
import numpy as np

def slurm_format_minutes(mins):
   days = mins // (24*60)
   hours = (mins % (24*60)) // 60
   rem_min = mins % 60
   return f"{days}-{hours:02d}:{rem_min:02d}"


def get_llbox(fname):
    
   RMG = grdfuns.roms_grid_to_dict(fname)
   lt_mx = np.max(RMG['lat_rho'])
   lt_mn = np.min(RMG['lat_rho'])
   ln_mx = np.max(RMG['lon_rho'])
   ln_mn = np.min(RMG['lon_rho'])
   dlt = 0.1605 # this is 15 km in degrees at 33 deg N
   dln = 0.1353 # this is 15 km in degrees at 33 deg N
   lt_mx = lt_mx + dlt
   lt_mn = lt_mn - dlt
   ln_mx = ln_mx + dln
   ln_mn = ln_mn - dln
   llb = [lt_mn, lt_mx, ln_mn, ln_mx]
#  latlonbox = [28.0, 37.0, -125.0, -115.0] # original hard coded numbers.

   return llb

# the function below MUST be called create_model_info_dict!!!
def create_model_info_dict():

    #run_type determines whether we do a forecast or a hindcast 
    #run_type = 'hindcast'
    run_type = 'forecast'

    # is this needed...
    #HOME = Path.home()
    #try:
    #    HOSTNAME = os.environ['HOSTNAME']
    #except KeyError:
    #    HOSTNAME = 'BLANK'

    # PFM is the dictionary that contains all of the model information
    PFM = dict()
    PFM['run_type'] = run_type

    pfm_dir = '/scratch/PFM_Simulations/' # this stays fixed for Grids and executables
                                         # both forecasting and hindcasting use the same ones.
    
    model_root_dir = '/scratch/PFM_Simulations/'
    
    if run_type == 'hindcast': # note hycom with tides starts on 2024-10-10 1200...
        sim_start_time = '2024101100' # the simulation start time is in yyyymmddhh format
        sim_end_time   = '2024101300' # this is the very last time of the full simulation
        PFM['forecast_days'] = 1.0 # for now we do 1 day sub simulations
        # set the simulation end time. An integer number of days past the start time
        # We will loop over days until we get to this time.
        PFM['sim_start_time'] = datetime.strptime(sim_start_time,'%Y%m%d%H')
        PFM['sim_end_time'] = datetime.strptime(sim_end_time,'%Y%m%d%H')
        PFM['sim_time_1'] = PFM['sim_start_time']
        PFM['sim_time_2'] = PFM['sim_time_1'] + PFM['forecast_days'] * timedelta(days=1)
        # sim_start_time is the overall 1st time of the full simulation
        # sim_end_time is the overall last time of the full simulation
        # sim_time_1 is the inital time of the sub simulation
        # sim_time_2 is the last time of the sub simulation 
        # we loop through levels_to_run
        PFM['levels_to_run'] = ['LV1','LV2','LV3']
        ocn_model = 'hycom_hind_wtide' # _wtide indicates using the new (>20241010) hycom
        PFM['atm_hind_dir'] = '/dataSIO/PHM_Simulations/raw_download/nam_grb2'
        atm_model = 'nam_analysis'
        PFM['atm_dt_hr'] = 3
        PFM['server'] = 'swell'
    else:
        PFM['levels_to_run'] = ['LV1','LV2','LV3','LV4']
        # hycom_new is the only forecast option
        ocn_model = 'hycom_new' # worked with 'hycom' but that is now (9/13/24) depricated      
        PFM['clean_start']=True
        # where do we find, and save, the raw hycom data to...
        PFM['hycom_dir'] = pfm_dir + 'hycom_data/'
        PFM['Q_PB'] = -2.0 # m3/s flow at Punta Bandera
        PFM['dye_PB'] = 0.5 # fraction of Q_PB that is raw WW
       # this is where PFM saves atm forcing and river discharge to at end of PFM simulation
        PFM['archive_dir'] = '/dataSIO/PFM_Simulations/Archive/Forcing/'
        PFM['archive_web_dir'] = '/dataSIO/PFM_Simulations/Archive/web/'

    if ocn_model == 'hycom_new' or ocn_model == 'hycom_hind_wtide':
        add_tides=0 # the new version of hycom has tides, we don't need to add them

    PFM['executable_dir'] = pfm_dir + 'executables/'   # we will not make copies of executables and 
    pfm_grid_dir =  pfm_dir +  'Grids'                 # grids. PHM will use the ones in pfm_dir
    lv1_root_dir =  model_root_dir +  'LV1_Forecast/'
    lv2_root_dir =  model_root_dir +  'LV2_Forecast/'
    lv3_root_dir =  model_root_dir +  'LV3_Forecast/'
    lv4_root_dir =  model_root_dir +  'LV4_Forecast/'

    lv1_run_dir  = lv1_root_dir + 'Run'
    lv1_his_dir  = lv1_root_dir + 'His'
    lv1_forc_dir = lv1_root_dir + 'Forc'
    lv1_tide_dir = pfm_dir + 'tide_data'
    lv1_plot_dir = lv1_root_dir + 'Plots'          

    lv2_run_dir  = lv2_root_dir + 'Run'
    lv2_his_dir  = lv2_root_dir + 'His'
    lv2_forc_dir = lv2_root_dir + 'Forc'
    lv2_plot_dir = lv2_root_dir + 'Plots'          

    lv3_run_dir  = lv3_root_dir + 'Run'
    lv3_his_dir  = lv3_root_dir + 'His'
    lv3_forc_dir = lv3_root_dir + 'Forc'
    lv3_plot_dir = lv3_root_dir + 'Plots'          

    lv4_run_dir  = lv4_root_dir + 'Run'
    lv4_forc_dir = lv4_root_dir + 'Forc'

    # here is the switch to go from LV4 Roms only to LV4 coawst
    #lv4_model = 'ROMS'
    lv4_model = 'COAWST'
    if lv4_model == 'ROMS':
        PFM['lv4_blank_name'] = 'LV4_BLANK_nowaves_norivers.in'
        PFM['lv4_yaml_file'] = 'LV4_varinfo_nowaves_norivers.yaml'
        #PFM['lv4_exe_name'] = 'LV4_ocean_nowaves_noriversM'
        PFM['lv4_executable'] = 'LV1_oceanM'
                
    if lv4_model == 'COAWST':
        PFM['lv4_blank_name'] = 'LV4_BLANK.in'
        PFM['lv4_yaml_file'] = 'LV4_varinfo.yaml'
        #PFM['lv4_exe_name'] = 'LV4_coawstM'
        #PFM['lv4_executable'] = 'LV4_coawstM'
        PFM['lv4_executable'] = 'coawstM_intel'
        PFM['lv4_blank_swan_name'] = 'LV4_SWAN_BLANK.in'
        PFM['lv4_coupling_name'] = 'LV4_COUPLING_BLANK.in'

    lv4_his_dir  = lv4_root_dir + 'His'
    lv4_plot_dir = lv4_root_dir + 'Plots'          
    lv4_coawst_varinfo_full = lv4_run_dir + '/LV4_coawst_varinfo.dat'
    PFM['lv4_coawst_varinfo_full'] = lv4_coawst_varinfo_full
    PFM['lv4_nwave_dirs'] = '11' # used in the ocean.in file. it does NOT match swan
    PFM['lv4_clm_file'] = 'LV4_clm.nc'    
    PFM['lv4_nud_file'] = 'LV4_nud.nc'    
    PFM['lv4_river_file'] = 'LV4_river.nc'    


# grid file locations
    lv1_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'
    lv2_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV2_rx020.nc'
    lv3_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV3_rx020.nc'
    lv4_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV4_mss_oct2024.nc'

    PFM['lv1_grid_file_full'] = lv1_grid_file
    PFM['lv2_grid_file_full'] = lv2_grid_file
    PFM['lv3_grid_file_full'] = lv3_grid_file
    PFM['lv4_grid_file_full'] = lv4_grid_file


    # atm options for run_type = 'forecast' are: nam_nest, gfs, gfs_1hr, ecmwf
    if run_type == 'forecast':
        atm_model = 'ecmwf'
        #atm_model = 'nam_nest'
        #atm_model = 'gfs'
        #atm_model = 'gfs_1hr'
    
    atm_get_method = 'open_dap_nc'
    ocn_get_method = 'ncks_para'
    # atm option for run_type = 'hindcast' are: 

    # we now set the forecast duration depending on atm_model
    if run_type == 'forecast':
        if atm_model == 'nam_nest':
            PFM['forecast_days'] = 2.5
            PFM['atm_dt_hr'] = 3
        if atm_model == 'gfs' or atm_model == 'gfs_1hr':
            PFM['forecast_days'] = 5.0 # this should be 5, but might be out of bounds?
        if atm_model == 'gfs':
            PFM['atm_dt_hr'] = 3
        if atm_model == 'gfs_1hr':
            PFM['atm_dt_hr'] = 1
        if atm_model == 'ecmwf':
            PFM['forecast_days'] = 5.0 #is the target 
            PFM['atm_dt_hr'] = 1
    
    PFM['ecmwf_dir'] = pfm_dir + 'ecmwf_data/' # this is where the forecast ecmwf data goes
    PFM['ecmwf_all_pkl_name'] = 'ecmwf_all.pkl'
    PFM['ecmwf_pkl_roms_vars'] = 'ecmwf_roms_vars.pkl'
    PFM['ecmwf_pkl_on_roms_grid'] = 'ecmwf_on_romsgrid.pkl'

# this it the one place where the model time reference is set
    modtime0 = datetime(1999,1,1,0,0)
# see notes in Evernote, Run Log 9, 2020.10.06
    roms_time_units = 'seconds since 1990-01-01 00:00:00'
# format used for naming day folders, (used? 9/4/24)
    ds_fmt = '%Y.%m.%d'
# number of forecast days (used? 9/4/24)

# vertical stretching info for each grid are embedded in a dict
    SS=dict()
    SS['L1','Nz']          = 40
    SS['L1','Vtransform']  = 2                       # transformation equation
    SS['L1','Vstretching'] = 4                       # stretching function
    SS['L1','THETA_S']     = 8.0                     # surface stretching parameter
    SS['L1','THETA_B']     = 3.0                     # bottom  stretching parameter
    SS['L1','TCLINE']      = 50.0                    # critical depth (m)
    SS['L1','hc']          = 50.0 
    
    SS['L2','Nz']          = 40
    SS['L2','Vtransform']  = 2                       # transformation equation
    SS['L2','Vstretching'] = 4                       # stretching function
    SS['L2','THETA_S']     = 8.0                     # surface stretching parameter
    SS['L2','THETA_B']     = 3.0                     # bottom  stretching parameter
    SS['L2','TCLINE']      = 50.0                    # critical depth (m)
    SS['L2','hc']          = 50.0 

    SS['L3','Nz']          = 40
    SS['L3','Vtransform']  = 2                       # transformation equation
    SS['L3','Vstretching'] = 4                       # stretching function
    SS['L3','THETA_S']     = 8.0                     # surface stretching parameter
    SS['L3','THETA_B']     = 3.0                     # bottom  stretching parameter
    SS['L3','TCLINE']      = 50.0                    # critical depth (m)
    SS['L3','hc']          = 50.0 

    SS['L4','Nz']          = 10
    SS['L4','Vtransform']  = 2                       # transformation equation
    SS['L4','Vstretching'] = 4                       # stretching function
    SS['L4','THETA_S']     = 4.5                     # surface stretching parameter
    SS['L4','THETA_B']     = 3.0                     # bottom  stretching parameter
    SS['L4','TCLINE']      = 3.5                    # critical depth (m)
    SS['L4','hc']          = 3.5 

    LLB = dict()
    LLB['L1'] = get_llbox(lv1_grid_file)
    LLB['L2'] = get_llbox(lv2_grid_file)
    LLB['L3'] = get_llbox(lv3_grid_file)
    LLB['L4'] = get_llbox(lv4_grid_file)


# gridding info make sure ntilei * ntilej is a multiple of 36. that's how many cores per node on swell
    NN=dict() 
    NN['L1','Lm']  = 251     # Lm in input file
    NN['L1','Mm']  = 388     # Mm in input file
    NN['L1','ntilei'] = 9    # 6 number of tiles in I-direction
    NN['L1','ntilej'] = 24   # 18 number of tiles in J-direction
    NN['L1','np'] = NN['L1','ntilei'] * NN['L1','ntilej'] # total number of processors
    NN['L1','nnodes'] =  int( NN['L1','np'] / 36 )  # 3 number of nodes to be used.  not for .in file but for slurm!

    NN['L2','Lm']  = 264     # Lm in input file
    NN['L2','Mm']  = 396     # Mm in input file
    NN['L2','ntilei'] = 9    # 6 number of tiles in I-direction
    NN['L2','ntilej'] = 24   # 18 number of tiles in J-direction
    NN['L2','np'] = NN['L2','ntilei'] * NN['L2','ntilej'] # total number of processors
    NN['L2','nnodes'] = int( NN['L2','np'] / 36 )  # 3 number of nodes to be used.  not for .in file but for slurm!

    NN['L3','Lm']  = 249     # Lm in input file
    NN['L3','Mm']  = 411     # Mm in input file
    NN['L3','ntilei'] = 12    # 6 number of tiles in I-direction
    NN['L3','ntilej'] = 30    # 18 number of tiles in J-direction
    NN['L3','np'] = NN['L3','ntilei'] * NN['L3','ntilej'] # total number of processors
    NN['L3','nnodes'] = int( NN['L3','np'] / 36  )  # 3 number of nodes to be used.  not for .infile but for slurm!

    NN['L4','Lm']  = 484     # Lm in input file
    NN['L4','Mm']  = 1139     # Mm in input file
    if lv4_model == 'ROMS':
        NN['L4','ntilei'] = 14    # 6 number of tiles in I-direction
        NN['L4','ntilej'] = 36    # 18 number of tiles in J-direction
        NN['L4','np'] = NN['L4','ntilei'] * NN['L4','ntilej'] # total number of processors
        NN['L4','nnodes'] = int( NN['L4','np'] / 36  )  # 3 number of nodes to be used.  not for .infile but for slurm!
    if lv4_model == 'COAWST':
        # swan = 60, ni=12, nj=37. swan too slow.
        NN['L4','np_swan']   = 72    # 60 number of CPUs for swan,
        NN['L4','ntilei'] = 12    # 12 number of tiles in I-direction
        NN['L4','ntilej'] = 36    # 37 number of tiles in J-direction
        NN['L4','np_roms'] = NN['L4','ntilei'] * NN['L4','ntilej'] # total number of processors
        NN['L4','np_tot'] = NN['L4','np_swan'] + NN['L4','np_roms']
        NN['L4','nnodes'] = int( NN['L4','np_tot'] / 36  )  # 3 number of nodes to be used.  not for .infile but for slurm!         
        PFM['swan_to_roms'] = '720.0d0'

# timing info
    tt=dict()
    tt['L1','dtsec'] = 60
    tt['L1','ndtfast'] = 15
    tt['L1','forecast_days'] = PFM['forecast_days']
    
    tt['L2','dtsec'] = 30
    tt['L2','ndtfast'] = 15
    tt['L2','forecast_days'] = PFM['forecast_days']
    
    tt['L3','dtsec'] = 15
    tt['L3','ndtfast'] = 15
    tt['L3','forecast_days'] = PFM['forecast_days']

    tt['L4','dtsec'] = 2
    tt['L4','ndtfast'] = 8
    tt['L4','forecast_days'] = PFM['forecast_days']

#  max slurm time for level 1,2,3,4, in minutes
    lv1_mins = int( np.round( 8.0 * 60.0 * PFM['forecast_days'] / (2.5 * tt['L1','dtsec']) ) )
    lv2_mins = int( np.round( 10.0 * 30.0 * PFM['forecast_days'] / (2.5 * tt['L2','dtsec']) ) )
    lv3_mins = int( np.round( 15.0 * 15.0 * PFM['forecast_days'] / (2.5 * tt['L3','dtsec']) ) )
    lv4_mins = int( np.round( 180.0 * 2.0 * PFM['forecast_days'] / (2.5 * tt['L4','dtsec']) ) )
#  this 180 minutes is more than it takes for current (12/12/24) tiling and dt=2sec, Tf=2.5 days (135 min).
#  we add a buffer and we scale linearly with forecast days and dt.   
# SBATCH -t 0-2:00              # time limit: (D-HH:MM) 
    PFM['lv1_max_time_str'] = slurm_format_minutes(lv1_mins)
    PFM['lv2_max_time_str'] = slurm_format_minutes(lv2_mins)
    PFM['lv3_max_time_str'] = slurm_format_minutes(lv3_mins)
    PFM['lv4_max_time_str'] = slurm_format_minutes(lv4_mins)

# output info
    OP = dict()
    OP['L1','his_interval'] = 3600 # how often in sec outut is written to his.nc   
    OP['L1','rst_interval'] = 0.25  # how often in days, a restart file is made.
    OP['L2','his_interval'] = 3600 # how often in sec outut is written to his.nc
    OP['L2','rst_interval'] = 0.25  # how often in days, a restart file is made. 
    OP['L3','his_interval'] = 3600 # how often in sec outut is written to his.nc
    OP['L3','rst_interval'] = 0.25  # how often in days, a restart file is made. 
    OP['L4','his_interval'] = 3600 # how often in sec outut is written to his.nc
    OP['L4','rst_interval'] = 0.25  # how often in days, a restart file is made. 


    # first the environment
    PFM['lv1_run_dir']  = lv1_run_dir
    PFM['lv1_forc_dir'] = lv1_forc_dir
    PFM['lv1_tide_dir'] = lv1_tide_dir   
    PFM['lv1_grid_dir'] = pfm_grid_dir
    PFM['lv1_his_dir']  = lv1_his_dir
    PFM['lv1_plot_dir'] = lv1_plot_dir         

    PFM['lv2_run_dir']  = lv2_run_dir
    PFM['lv2_forc_dir'] = lv2_forc_dir
    PFM['lv2_grid_dir'] = pfm_grid_dir
    PFM['lv2_his_dir']  = lv2_his_dir
    PFM['lv2_plot_dir'] = lv2_plot_dir         

    PFM['lv3_run_dir']  = lv3_run_dir
    PFM['lv3_forc_dir'] = lv3_forc_dir
    PFM['lv3_grid_dir'] = pfm_grid_dir
    PFM['lv3_his_dir']  = lv3_his_dir
    PFM['lv3_plot_dir'] = lv3_plot_dir         

    PFM['lv4_run_dir']  = lv4_run_dir
    PFM['lv4_forc_dir'] = lv4_forc_dir
    PFM['lv4_grid_dir'] = pfm_grid_dir
    PFM['lv4_his_dir']  = lv4_his_dir
    PFM['lv4_plot_dir'] = lv4_plot_dir         

    PFM['lv1_grid_file'] = lv1_grid_file
    PFM['lv2_grid_file'] = lv2_grid_file
    PFM['lv3_grid_file'] = lv3_grid_file
    PFM['lv4_grid_file'] = lv4_grid_file
    PFM['lv4_model']     = lv4_model

    PFM['dataSIO_plot_dir'] = '/dataSIO/PFM_Simulations/Plots/'
    PFM['lv1_archive_his_dir'] = '/dataSIO/PFM_Simulations/Archive/LV1_His/'
    PFM['lv2_archive_his_dir'] = '/dataSIO/PFM_Simulations/Archive/LV2_His/'
    PFM['lv3_archive_his_dir'] = '/dataSIO/PFM_Simulations/Archive/LV3_His/'
    PFM['lv4_archive_his_dir'] = '/dataSIO/PFM_Simulations/Archive/LV4_His/'
    PFM['log_archive_dir']     = '/dataSIO/PFM_Simulations/Archive/Log/'

    if PFM['run_type'] == 'forecast':
        PFM['hycom_data_dir'] = pfm_dir + 'hycom_data/'
    else:
        PFM['hycom_data_dir'] = '/dataSIO/PHM_Simulations/raw_download/hycom_nc/'

    # location of the forecast cdip data !!!
    PFM['cdip_data_dir'] = pfm_dir + 'cdip_data' 

    PFM['lv1_tides_file']          = 'ocean_tide.nc'
    PFM['atm_tmp_pckl_file']       = 'atm_tmp_pckl_file.pkl'
    PFM['lv1_depth_file']          = 'roms_tmp_depth_file.pkl' 
    PFM['lv1_ocn_tmp_pckl_file']   = 'hycom_tmp_pckl_file.pkl' 
    PFM['lv1_ocnR_tmp_pckl_file']  = 'ocnR_temp_pckl_file.pkl'
    PFM['lv1_ocnIC_tmp_pckl_file'] = 'ocnIC_tmp_pckl_file.pkl'
    PFM['lv1_ocn_tmp_nck_file']    = 'hycom_tmp_ncks_file.nc'
    PFM['lv1_ocnBC_tmp_pckl_file'] = 'ocnBC_tmp_pckl_file.pkl'
    PFM['atm_tmp_LV1_pckl_file']   = 'atm_tmp_LV1_pckl_file.pkl'
    PFM['lv1_atm_file']            = 'LV1_ATM_FORCING.nc'
    PFM['lv1_ini_file']            = 'LV1_OCEAN_IC.nc'
    PFM['lv1_bc_file']             = 'LV1_OCEAN_BC.nc'   
    
    #PFM['lv1_executable']          = 'LV1_oceanM'
    #PFM['lv2_executable']          = 'LV1_oceanM'
    #PFM['lv3_executable']          = 'LV1_oceanM'
    PFM['lv1_executable']          = 'LV3_romsM_INTEL'
    PFM['lv2_executable']          = 'LV3_romsM_INTEL'
    PFM['lv3_executable']          = 'LV3_romsM_INTEL'
    #PFM['lv1_executable']          = 'LV3_romsM_INTEL_new'
    #PFM['lv2_executable']          = 'LV3_romsM_INTEL_new'
    #PFM['lv3_executable']          = 'LV3_romsM_INTEL_new'
    #PFM['lv1_executable']          = 'romsM_INTEL'
    #PFM['lv2_executable']          = 'romsM_INTEL'
    #PFM['lv3_executable']          = 'romsM_INTEL'

    if add_tides==1:
        PFM['lv1_adding_tides'] = 'yes'
        PFM['lv1_executable']          = 'LV1_oceanM_w_tide_forcing'
        PFM['lv1_tide_fname']          = 'roms_tide_adcirc_LV01.nc'
        PFM['lv1_tides_file']          = PFM['lv1_tide_dir'] + '/' + PFM['lv1_tide_fname']
    else:
        PFM['lv1_adding+tides'] = 'no'
    
    PFM['lv2_ocnIC_tmp_pckl_file'] = 'ocnIC_LV2_tmp_pckl_file.pkl'   
    PFM['lv2_ocnBC_tmp_pckl_file'] = 'BC_LV2_tmp_file.pkl'
    PFM['atm_tmp_LV2_pckl_file']   = 'atm_tmp_LV2_pckl_file.pkl'
    PFM['lv2_atm_file']            = 'LV2_ATM_FORCING.nc'
    PFM['lv2_ini_file']            = 'LV2_OCEAN_IC.nc'
    PFM['lv2_bc_file']             = 'LV2_OCEAN_BC.nc'   
    
    PFM['lv3_ocnIC_tmp_pckl_file'] = 'ocnIC_LV3_tmp_pckl_file.pkl'   
    PFM['lv3_ocnBC_tmp_pckl_file'] = 'BC_LV3_tmp_file.pkl'
    PFM['atm_tmp_LV3_pckl_file']   = 'atm_tmp_LV3_pckl_file.pkl'
    PFM['lv3_atm_file']            = 'LV3_ATM_FORCING.nc'
    PFM['lv3_ini_file']            = 'LV3_OCEAN_IC.nc'
    PFM['lv3_bc_file']             = 'LV3_OCEAN_BC.nc'   

    PFM['lv4_ocnIC_tmp_pckl_file'] = 'ocnIC_LV4_tmp_pckl_file.pkl'   
    PFM['lv4_ocnBC_tmp_pckl_file'] = 'BC_LV4_tmp_file.pkl'
    PFM['atm_tmp_LV4_pckl_file']   = 'atm_tmp_LV4_pckl_file.pkl'
    PFM['lv4_atm_file']            = 'LV4_ATM_FORCING.nc'
    PFM['lv4_ini_file']            = 'LV4_OCEAN_IC.nc'
    PFM['lv4_bc_file']             = 'LV4_OCEAN_BC.nc'
    PFM['lv4_swan_pckl_file']      = 'LV4_swan_bnd.pkl'
    PFM['lv4_swan_grd_file']       = 'swan_LV4.grd'
    PFM['lv4_swan_bot_file']       = 'swan_LV4.bot' 
    PFM['lv4_swan_bnd_file']       = 'LV4_swan.bnd'
    PFM['lv4_swan_wnd_file']       = 'LV4_swan.wnd' 
    PFM['lv4_swan_dt_sec']         = 240
    PFM['swan_init_txt_full']      = 'ZERO' # this is the default text needed to run swan with IC=0
                                            # if using a swan restart file, this is changed to the correct
                                            # string in init_funs.py
    
    PFM['lv4_swan_rst_int_hr']     = int( 24 * OP['L4','rst_interval'] )
    PFM['river_pckl_file_full']    = PFM['lv4_forc_dir'] + '/river_Q.pkl'
    
    PFM['lv4_his_web_dir'] = '/projects/www-users/falk/PFM_Forecast/LV4_His/'

    PFM['modtime0']        = modtime0
    PFM['roms_time_units'] = roms_time_units
    PFM['ds_fmt']          = ds_fmt
    PFM['ndefhis']         = 0 # when zero, only 1 history file is made.

    PFM['ocn_model']      = ocn_model
    PFM['atm_model']      = atm_model
    PFM['atm_get_method'] = atm_get_method
    PFM['ocn_get_method'] = ocn_get_method
    #latlonbox = [28.0, 37.0, -125.0, -115.0]     
    #PFM['latlonbox']      = latlonbox
    PFM['latlonbox']      = LLB
    PFM['stretching']     = SS
    PFM['gridinfo']       = NN
    PFM['tinfo']          = tt
    PFM['outputinfo']     = OP

    # this is the switch to use restart files
    # restart_files_dir is where the restart file is saved
    # and where the restart files are read from.
    PFM['restart_files_dir'] =  pfm_dir + 'restart_data' 

    PFM['lv1_use_restart']         = 1 # 1 == use_restart 
    PFM['lv2_use_restart']         = 1
    PFM['lv3_use_restart']         = 1
    PFM['lv4_use_restart']         = 1
    #PFM['lv4_swan_use_rst']        = 0
    PFM['lv4_swan_use_rst']        = 1

    # now do the timing information
    start_time = datetime.now()

    # fetch_time sets 1st time of the PFM forecast. It will go to the nearest 6 hr based on nam becoming available
    # nam_nest is available 3 hrs after 0,6,12,18 Z
    # gfs is available 5.5 hrs after 0,6,12,18, Z

    fetch_time = datetime.now(timezone.utc) 
    utc_time = fetch_time
    hour_utc = utc_time.hour
    year_utc = utc_time.year
    mon_utc  = utc_time.month
    day_utc  = utc_time.day

    PFM['start_time'] = start_time  # this is when we started running PFM
    PFM['utc_time']   = utc_time    # this is when we started PFM in UTC

    if run_type == 'forecast':
        past_6 = hour_utc % 6 # this is the number of hours past 0,6,12,18...
        if atm_model == 'nam_nest':
            if past_6 > 3:
                past_6 = past_6
            else:
                past_6 = past_6 + 6     
        if atm_model == 'gfs' or atm_model == 'gfs_1hr':
            past_6 = past_6 + 6
        if atm_model == 'ecmwf': # right now (2/6/25, ecmwf lags by 8 hrs or so)
            if past_6 > 1:
                past_6 = past_6 + 6      
            else:
                past_6 = past_6 + 12

        # fetch_time2 is now the start time of the PFM simulation based on the closest available
        # nam data to now.
        fetch_time2 = datetime(year_utc,mon_utc,day_utc,hour_utc,0,0,0)
        fetch_time2 = fetch_time2 - timedelta(hours=past_6) 
        #if hour_utc < 11:
        #   fetch_time = datetime(fetch_time.year,fetch_time.month, fetch_time.day, 12) - timedelta(days=4)
        #else:
        #   fetch_time = datetime(fetch_time.year,fetch_time.month, fetch_time.day, 12) - timedelta(days=3)
        hhmm = fetch_time2.strftime('%H%M')
        yyyymmdd = "%d%02d%02d" % (fetch_time2.year, fetch_time2.month, fetch_time2.day)
        PFM['yyyymmdd']   = yyyymmdd
        PFM['hhmm']       = hhmm        # this is the HHMM of the forecast, aligns with NAM
        PFM['sim_time_1'] = fetch_time2
        PFM['sim_time_2'] = fetch_time2 + PFM['forecast_days']*timedelta(days=1)
        PFM['sim_start_time'] = PFM['sim_time_1']
        PFM['sim_end_time'] = PFM['sim_time_2']
        
        yyyymmddhh   = PFM['sim_start_time'].strftime("%Y%m%d%H")
        yyyymmddhhmm = PFM['sim_start_time'].strftime("%Y%m%d%H") + '00'
        end_str = PFM['sim_end_time'].strftime("%Y%m%d%H") + '00'
        PFM['lv1_his_name'] = 'LV1_ocean_his_' + yyyymmddhhmm + '.nc'
        PFM['lv1_rst_name'] = 'LV1_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc' 
        PFM['lv1_his_name_full'] = PFM['lv1_his_dir'] + '/' + PFM['lv1_his_name']
        PFM['lv1_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv1_rst_name']
        PFM['lv2_his_name'] = 'LV2_ocean_his_' + yyyymmddhhmm + '.nc'
        PFM['lv2_rst_name'] = 'LV2_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc' 
        PFM['lv2_his_name_full'] = PFM['lv2_his_dir'] + '/' + PFM['lv2_his_name']
        PFM['lv2_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv2_rst_name']
        PFM['lv3_his_name'] = 'LV3_ocean_his_' + yyyymmddhhmm + '.nc'
        PFM['lv3_rst_name'] = 'LV3_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc' 
        PFM['lv3_his_name_full'] = PFM['lv3_his_dir'] + '/'  + PFM['lv3_his_name']
        PFM['lv3_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv3_rst_name']
        PFM['lv4_his_name'] = 'LV4_ocean_his_' + yyyymmddhhmm + '.nc'
        PFM['lv4_rst_name'] = 'LV4_ocean_rst_' + yyyymmddhhmm + '_' + end_str + '.nc' 
        PFM['lv4_swan_rst_name']  = 'LV4_swan_rst_' + yyyymmddhhmm + '.dat' 
        web_name = 'web_data_' + yyyymmddhh + '.nc'
        PFM['lv4_web_name_full'] = PFM['lv4_his_dir'] + '/' + web_name
        PFM['lv4_his_name_full'] = PFM['lv4_his_dir'] + '/'  + PFM['lv4_his_name']
        PFM['lv4_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv4_rst_name']
        PFM['lv4_swan_rst_name_full'] = PFM['restart_files_dir'] + '/' + PFM['lv4_swan_rst_name']
    #     # get how often swan files are written. The 0.2 makes sure we check 5 times between 
    #     # approximate writing times. based on CURRENT (12/13/24) coawst tiling!!! if 
    #     # tiling changes this needs to change too!
        PFM['lv4_swan_check_freq_sec'] = int( np.round( 0.2 * OP['L4','rst_interval'] * 2 * 3600 / 2.5 ) ) 
    else:
        fetch_time2 = PFM['sim_time_1']
        
    PFM['fetch_time'] =  fetch_time2 # this is the start time of the PFM hindcast as datetime object     
    # the end time of the forecast as datetime object, for a hindcast, this is always one day more than fetch_time
    end_time = fetch_time2 + PFM['forecast_days'] * timedelta(days=1)
    PFM['fore_end_time'] = end_time # the end time of the forecast

    return PFM



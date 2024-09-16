"""
This is the one place where you set the path structure for the PFM.
The info is stored in the dict PFM.
All? paths are pathlib.Path objects.
This program is meant to be loaded as a module by Lfun which then adds more
entries to the Ldir dict based on which model run you are working on.
Users should copy this to PFM_user/get_sdpm_info.py and edit it appropriately.
"""
 
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
import grid_functions as grdfuns
import numpy as np

## FF what does this function do?  what is fname?  
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
#      latlonbox = [28.0, 37.0, -125.0, -115.0] # original hard coded numbers.

   return llb


def get_PFM_info():

   HOME = Path.home()
   try:
       HOSTNAME = os.environ['HOSTNAME']
   except KeyError:
       HOSTNAME = 'BLANK'

   
   pfm_root_dir = '/scratch/PFM_Simulations/'       
   pfm_file_name = 'PFM_run_info.pkl'
   pfm_info_full = pfm_root_dir + pfm_file_name
   if Path(pfm_info_full).is_file():
      with open(pfm_info_full,'rb') as fp:
         PFM = pickle.load(fp)
         #print('PFM info was loaded from ' + pfm_info_full)
   else:         
      # set up the dict that will be saved
      PFM = dict()
      PFM['info_file'] = pfm_info_full

      lo_env = 'mss_swell'
      pfm_grid_dir =  pfm_root_dir +  'Grids'       
      lv1_root_dir =  pfm_root_dir +  'LV1_Forecast/'
      lv2_root_dir =  pfm_root_dir +  'LV2_Forecast/'
      lv3_root_dir =  pfm_root_dir +  'LV3_Forecast/'
      lv4_root_dir =  pfm_root_dir +  'LV4_Forecast/'
 
      lv1_run_dir  = lv1_root_dir + 'Run'
      lv1_his_dir  = lv1_root_dir + 'His'
      lv1_forc_dir = lv1_root_dir + 'Forc'
      lv1_tide_dir = lv1_root_dir + 'Tides'
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
      lv4_his_dir  = lv4_root_dir + 'His'
      lv4_forc_dir = lv4_root_dir + 'Forc'
      lv4_plot_dir = lv4_root_dir + 'Plots'          

      
   # defaults that should work on all machines
      parent = Path(__file__).absolute().parent.parent

      lv1_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'
      lv2_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV2_rx020.nc'
      lv3_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV3_rx020.nc'
      lv4_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV4_rx020.nc'
   #   if str(HOSTNAME) == 'swell':
   #       lv1_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'
   #   else:
   #       lv1_grid_file = '/Users/mspydell/research/FF2024/models/SDPM_mss/PFM_user/grids/GRID_SDTJRE_LV1.nc'

  
      run_type = 'forecast'

   # what is the ocean / atm model used to force?
      ocn_model = 'hycom_new' # worked with 'hycom' but that is now (9/13/24) depricated
      atm_model = 'nam_nest'
      atm_get_method = 'open_dap_nc'
      ocn_get_method = 'ncks_para'
   # what is the time resolution of the models (in days), (used? 9/4/24 MSS)
      daystep_ocn = 3/24
      daystep_atm = 3/24
   # roms will be run with this time step (in days)
      daystep = 1 # not used right now (9/4/24 MSS)
      
   # this it the one place where the model time reference is set
      modtime0 = datetime(1999,1,1,0,0)
   # see notes in Evernote, Run Log 9, 2020.10.06
      roms_time_units = 'seconds since 1990-01-01 00:00:00'
   # format used for naming day folders, (used? 9/4/24)
      ds_fmt = '%Y.%m.%d'
   # number of forecast days (used? 9/4/24)
      fdays = 3

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

      LLB = dict()
      LLB['L1'] = get_llbox(lv1_grid_file)
      LLB['L2'] = get_llbox(lv2_grid_file)
      LLB['L3'] = get_llbox(lv3_grid_file)
      #LLB['L4'] = get_llbox(lv4_grid_file)


   # gridding info
      NN=dict() 
      NN['L1','Lm']  = 251     # Lm in input file
      NN['L1','Mm']  = 388     # Mm in input file
      NN['L1','ntilei'] = 6    # number of tiles in I-direction
      NN['L1','ntilej'] = 18   # number of tiles in J-direction
      NN['L1','np'] = NN['L1','ntilei'] * NN['L1','ntilej'] # total number of processors
      NN['L1','nnodes'] = 3    # number of nodes to be used.  not for .in file but for slurm!

      NN['L2','Lm']  = 264     # Lm in input file
      NN['L2','Mm']  = 396     # Mm in input file
      NN['L2','ntilei'] = 6    # number of tiles in I-direction
      NN['L2','ntilej'] = 18   # number of tiles in J-direction
      NN['L2','np'] = NN['L2','ntilei'] * NN['L2','ntilej'] # total number of processors
      NN['L2','nnodes'] = 3    # number of nodes to be used.  not for .in file but for slurm!

      NN['L3','Lm']  = 249     # Lm in input file
      NN['L3','Mm']  = 411     # Mm in input file
      NN['L3','ntilei'] = 6    # number of tiles in I-direction
      NN['L3','ntilej'] = 18   # number of tiles in J-direction
      NN['L3','np'] = NN['L2','ntilei'] * NN['L2','ntilej'] # total number of processors
      NN['L3','nnodes'] = 3    # number of nodes to be used.  not for .infile but for slurm!

   # timing info
      tt=dict()
      tt['L1','dtsec'] = 60
      tt['L1','ndtfast'] = 15
      tt['L1','forecast_days'] = 2.5
      
      tt['L2','dtsec'] = 30
      tt['L2','ndtfast'] = 15
      tt['L2','forecast_days'] = 2.5
      
      tt['L3','dtsec'] = 15
      tt['L3','ndtfast'] = 15
      tt['L3','forecast_days'] = 2.5

   # output info
      OP = dict()
      OP['L1','his_interval'] = 3600 # how often in sec outut is written to his.nc   
      OP['L1','rst_interval'] = 0.5  # how often in days, a restart file is made.
      OP['L2','his_interval'] = 3600 # how often in sec outut is written to his.nc
      OP['L2','rst_interval'] = 0.5  # how often in days, a restart file is made. 
      OP['L3','his_interval'] = 3600 # how often in sec outut is written to his.nc
      OP['L3','rst_interval'] = 0.5  # how often in days, a restart file is made. 

      PFM['run_type'] = run_type

      # first the environment
      PFM['lo_env'] = lo_env
      PFM['parent'] = parent
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

      PFM['lv1_tide_fname']          = 'roms_tide_adcirc_LV01.nc'
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
     
      PFM['modtime0']        = modtime0
      PFM['roms_time_units'] = roms_time_units
      PFM['ds_fmt']          = ds_fmt
      PFM['forecast_days']   = fdays
      PFM['ndefhis']         = 0 # when zero, only 1 history file is made.
   
      PFM['daystep']        = daystep
      PFM['daystep_ocn']    = daystep_ocn
      PFM['daystep_atm']    = daystep_atm
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

      # now do the timing information
      start_time = datetime.now()
      utc_time = datetime.now(timezone.utc)
      hour_utc = utc_time.hour

      fetch_time = datetime.now(timezone.utc) - timedelta(days=1)

      if hour_utc < 11:
         fetch_time = datetime(fetch_time.year,fetch_time.month, fetch_time.day, 12) - timedelta(days=3)
      else:
         fetch_time = datetime(fetch_time.year,fetch_time.month, fetch_time.day, 12) - timedelta(days=2)
      
      yyyymmdd = "%d%02d%02d" % (fetch_time.year, fetch_time.month, fetch_time.day)
      PFM['yyyymmdd']   = yyyymmdd
      PFM['hhmm']       = '1200'
      PFM['fetch_time'] = fetch_time
      PFM['start_time'] = start_time
      PFM['utc_time']   = utc_time
      
      PFM['lv1_his_name'] = 'LV1_ocean_his_' + yyyymmdd + '1200' + '.nc'
      PFM['lv1_rst_name'] = 'LV1_ocean_rst_' + yyyymmdd + '1200' + '.nc' 
      PFM['lv1_his_name_full'] = PFM['lv1_his_dir'] + '/' + PFM['lv1_his_name']
      PFM['lv1_rst_name_full'] = PFM['lv1_forc_dir'] + '/' + PFM['lv1_rst_name']

      PFM['lv2_his_name'] = 'LV2_ocean_his_' + yyyymmdd + '1200' + '.nc'
      PFM['lv2_rst_name'] = 'LV2_ocean_rst_' + yyyymmdd + '1200' + '.nc' 
      PFM['lv2_his_name_full'] = PFM['lv2_his_dir'] + '/' + PFM['lv2_his_name']
      PFM['lv2_rst_name_full'] = PFM['lv2_forc_dir'] + '/' + PFM['lv2_rst_name']

      PFM['lv3_his_name'] = 'LV3_ocean_his_' + yyyymmdd + '1200' + '.nc'
      PFM['lv3_rst_name'] = 'LV3_ocean_rst_' + yyyymmdd + '1200' + '.nc' 
      PFM['lv3_his_name_full'] = PFM['lv3_his_dir'] + '/'  + PFM['lv3_his_name']
      PFM['lv3_rst_name_full'] = PFM['lv3_forc_dir'] + '/' + PFM['lv3_rst_name']

      PFM['lv4_his_name'] = 'LV4_ocean_his_' + yyyymmdd + '1200' + '.nc'
      PFM['lv4_rst_name'] = 'LV4_ocean_rst_' + yyyymmdd + '1200' + '.nc' 
      PFM['lv4_his_name_full'] = PFM['lv4_his_dir'] + '/'  + PFM['lv4_his_name']
      PFM['lv4_rst_name_full'] = PFM['lv4_forc_dir'] + '/' + PFM['lv4_rst_name']
   

      with open(PFM['info_file'],'wb') as fout:
         pickle.dump(PFM,fout)
         print('PFM info was saved as ' + PFM['info_file'])
       
   return PFM


def make_PFM_directory( parent ):


   pfm_root_dir = str(parent) + '/PFM_Simulations/'

   pfm_grid_dir =  pfm_root_dir +  'Grids'       
   lv1_root_dir =  pfm_root_dir +  'LV1_Forecast/'
   lv1_run_dir = lv1_root_dir + 'Run'
   lv1_his_dir = lv1_root_dir + 'His'
   lv1_forc_dir = lv1_root_dir + 'Forc'
   lv1_plot_dir = lv1_root_dir + 'Plots'          

   if os.path.isdir(pfm_root_dir)==False:
       os.mkdir(pfm_root_dir)
       print('making ' + pfm_root_dir)
   else:
       print( pfm_root_dir + ' exists')

   if os.path.isdir(pfm_grid_dir)==False:
       os.mkdir(pfm_grid_dir)
       print('making ' + pfm_grid_dir)       
   else:
       print( pfm_grid_dir + ' exists')
       
   if os.path.isdir(lv1_root_dir)==False:
       os.mkdir(lv1_root_dir)
       print('making ' + lv1_root_dir)              
   else:
       print( lv1_root_dir + ' exists')

   if os.path.isdir(lv1_run_dir)==False:
       os.mkdir(lv1_run_dir)
       print('making ' + lv1_run_dir)                     
   else:
       print( lv1_run_dir + ' exists')

   if os.path.isdir(lv1_his_dir)==False:
       os.mkdir(lv1_his_dir)
       print('making ' + lv1_his_dir)                            
   else:
       print( lv1_his_dir + ' exists')

   if os.path.isdir(lv1_forc_dir)==False:
       os.mkdir(lv1_forc_dir)
       print('making ' + lv1_forc_dir)                                   
   else:
       print( lv1_forc_dir + ' exists')

   if os.path.isdir(lv1_plot_dir)==False:
       os.mkdir(lv1_plot_dir)
   else:
       print( lv1_plot_dir + ' exists')

   return
    

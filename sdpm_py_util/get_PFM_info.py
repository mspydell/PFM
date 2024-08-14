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
from pathlib import Path
from datetime import datetime

def get_PFM_info():

   HOME = Path.home()
   try:
       HOSTNAME = os.environ['HOSTNAME']
   except KeyError:
       HOSTNAME = 'BLANK'


   #print('get_PFM_info(): running on' + HOSTNAME)
   
   if str(HOSTNAME) == 'swell':
       pfm_root_dir = '/scratch/PFM_Simulations/'
       lo_env = 'mss_swell'
   else:
       pfm_root_dir = '/Users/mspydell/research/FF2024/models/SDPM_mss/PFM_Simulations/'
       lo_env = 'mss'

   pfm_grid_dir =  pfm_root_dir +  'Grids'       
   lv1_root_dir =  pfm_root_dir +  'LV1_Forecast/'
   lv1_run_dir = lv1_root_dir + 'Run'
   lv1_his_dir = lv1_root_dir + 'His'
   lv1_forc_dir = lv1_root_dir + 'Forc'
   lv1_plot_dir = lv1_root_dir + 'Plots'          
       
     
# defaults that should work on all machines
   parent = Path(__file__).absolute().parent.parent
#   make_PFM_directory( parent )
#    LO = parent / 'PFM'
# this is where this set up script lives
#    LOu = parent / 'PFM_user'
# data is where the input files, atm, ocn IC, ocn BC, etc are found
#    data = parent / 'PFM_data'
# LOo is the location where his.nc files etc will go
#    LOo = parent / 'PFM_output'
# where are the grids??? We keep them in LO so that they come with
# git clone.
#   lv1_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV1.nc'
   if str(HOSTNAME) == 'swell':
       lv1_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'
   else:
       lv1_grid_file = '/Users/mspydell/research/FF2024/models/SDPM_mss/PFM_user/grids/GRID_SDTJRE_LV1.nc'

#   lv2_grid_file = str(lv2_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'
#   lv3_grid_file = str(lv3_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'   

# what is the ocean / atm model used to force?
   ocn_model = 'hycom'
   atm_model = 'nam_nest'
# what is the time resolution of the models (in days)
   daystep_ocn = 3/24
   daystep_atm = 3/24
# this is the box that covers LV1 so we get only a rectangle of hycom data
# needs to be floats
#   latlonbox = [27.75, 37.25, -124.5+360, -115.5+360]
   latlonbox = [28.0, 37.0, -125.0, -115.0]
# roms will be run with this time step (in days)
   daystep = 1

# add to the path
#    LOlo = str(LO)  + '/sdpm_py_util'
#    sys.path.append(LOlo)

#    testing = 0
#    if testing == 1:
#        print('runs up to here in get_sdpm_info.py')
#        sys.exit(0)

    
# this it the one place where the model time reference is set
   modtime0 = datetime(1999,1,1,0,0)
# correct string for time units in ROMS forcing files
# see notes in Evernote, Run Log 9, 2020.10.06
   roms_time_units = 'seconds since 1990-01-01 00:00:00'
# format used for naming day folders
   ds_fmt = '%Y.%m.%d'
# number of forecast days
   fdays = 3

# vertical stretching info for each grid are embedded in a dict
   SS=dict()
   SS['L1','Nz']          = 40
   SS['L1','Vtransform']  = 2                       # transformation equation
   SS['L1','Vstretching'] = 4                    # stretching function
   SS['L1','THETA_S']     = 8.0                      # surface stretching parameter
   SS['L1','THETA_B']     = 3.0                      # bottom  stretching parameter
   SS['L1','TCLINE']      = 50.0                      # critical depth (m)
   SS['L1','hc']          = 50.0 

# gridding info
   NN=dict() 
   NN['L1','Lm']  = 251    # Lm in input file
   NN['L1','Mm']  = 388   # Mm in input file
   NN['L1','ntilei'] = 6  # number of tiles in I-direction
   NN['L1','ntilej'] = 18 # number of tiles in J-direction
   NN['L1','np'] = NN['L1','ntilei'] * NN['L1','ntilej'] # total number of processors
   NN['L1','nnodes'] = 3  # number of nodes to be used.  not for .in file but for slurm!

# timing info
   tt=dict()
   tt['L1','dtsec'] = 60
   tt['L1','ndtfast'] = 15
   tt['L1','forecast_days'] = 2.5

# output info
   OP = dict()
   OP['L1','his_interval'] = 3600 # how often in sec outut is written to his.nc   
   OP['L1','rst_interval'] = 0.5 # how often in days, a restart file is made.

   PFM = dict()
   PFM['lo_env'] = lo_env
   PFM['parent'] = parent
#    PFM['LO'] = LO
#    PFM['LOo'] = LOo
#    PFM['LOu'] = LOu
#    PFM['data'] = data
   PFM['lv1_run_dir'] = lv1_run_dir
   PFM['lv1_forc_dir'] = lv1_forc_dir   
   PFM['lv1_grid_dir'] = pfm_grid_dir
   PFM['lv1_his_dir'] = lv1_his_dir
   PFM['lv1_plot_dir'] = lv1_plot_dir         
   PFM['lv1_grid_file'] = lv1_grid_file

   PFM['lv1_ocn_tmp_pckl_file'] = 'hycom_tmp_pckl_file.pkl' 
   PFM['lv1_ocnR_tmp_pckl_file'] = 'ocnR_temp_pckl_file.pkl'
   PFM['lv1_ocnIC_tmp_pckl_file'] = 'ocnIC_tmp_pckl_file.pkl'
   PFM['lv1_ocnBC_tmp_pckl_file'] = 'ocnBC_tmp_pckl_file.pkl'
   PFM['lv1_ocn_tmp_nck_file'] = 'hycom_tmp_ncks_file.nc'
   PFM['lv1_atm_file'] = 'LV1_ATM_FORCING.nc'
   PFM['lv1_ini_file'] = 'LV1_OCEAN_IC.nc'
   PFM['lv1_bc_file'] =  'LV1_OCEAN_BC.nc'   
   PFM['lv1_tide_fname'] = 'roms_tide_adcirc_LV01.nc'

#   PFM['lv2_run_dir'] = lv2_run_dir
#   PFM['lv2_grid_dir'] = pfm_grid_dir
#   PFM['lv2_his_dir'] = lv2_his_dir
#   PFM['lv2_plot_dir'] = lv2_plot_dir         

   PFM['modtime0'] = modtime0
   PFM['roms_time_units'] = roms_time_units
   PFM['ds_fmt'] = ds_fmt
   PFM['forecast_days'] = fdays
 
   PFM['daystep'] = daystep
   PFM['daystep_ocn'] = daystep_ocn
   PFM['daystep_atm'] = daystep_atm
   PFM['ocn_model'] = ocn_model
   PFM['atm_model'] = atm_model
   PFM['latlonbox'] = latlonbox
   PFM['stretching'] = SS
   PFM['gridinfo'] = NN
   PFM['tinfo'] = tt
   PFM['outputinfo'] = OP

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
    

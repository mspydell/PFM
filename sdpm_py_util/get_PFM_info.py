"""
This is the one place where you set the path structure for the PFM.
The info is stored in the dict SDP.
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

   print('hello there')
   print(HOSTNAME)
   
   if str(HOSTNAME) == 'swell':
       pfm_root_dir = '/scratch/PFM/'
       lo_env = 'mss_swell'
   else:
       pfm_root_dir = './PFM/'
       lo_env = 'mss'

   pfm_grid_dir =  pfm_root_dir +  'Grids'       
   lv1_root_dir =  pfm_root_dir +  'LV1_Forecast/'
   lv1_run_dir = lv1_root_dir + 'Run'
   lv1_his_dir = lv1_root_dir + 'His'
   lv1_forc_dir = lv1_root_dir + 'Forc'
   lv1_plot_dir = lv1_root_dir + 'Plots'          
       
     
# defaults that should work on all machines
   parent = Path(__file__).absolute().parent.parent
#    LO = parent / 'PFM'
# this is where this set up script lives
#    LOu = parent / 'PFM_user'
# data is where the input files, atm, ocn IC, ocn BC, etc are found
#    data = parent / 'PFM_data'
# LOo is the location where his.nc files etc will go
#    LOo = parent / 'PFM_output'
# where are the grids??? We keep them in LO so that they come with
# git clone.
   lv1_grid_file = str(pfm_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'
#   lv2_grid_file = str(lv2_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'
#   lv3_grid_file = str(lv3_grid_dir) + '/GRID_SDTJRE_LV1_rx020_hmask.nc'   

# what is the ocean / atm model used to force?
   ocn_model = 'hycom'
   atm_model = 'wrf'
# what is the time resolution of the models (in days)
   daystep_ocn = 3/24
   daystep_atm = 3/24
# this is the box that covers LV1 so we get only a rectangle of hycom data
# needs to be floats
   latlonbox = [27.75, 37.25, -124.5+360, -115.5+360]

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
   modtime0 = datetime(1970,1,1,0,0)
# correct string for time units in ROMS forcing files
# see notes in Evernote, Run Log 9, 2020.10.06
   roms_time_units = 'seconds since 1970-01-01 00:00:00'
# format used for naming day folders
   ds_fmt = '%Y.%m.%d'
# number of forecast days
   fdays = 3

# vertical stretching info for each grid are embedded in a dict
   SS=dict()
   SS['L1','Vtransform']=2                       # transformation equation
   SS['L1','Vstretching'] = 4                    # stretching function
   SS['L1','THETA_S'] = 8.0                      # surface stretching parameter
   SS['L1','THETA_B'] = 3.0                      # bottom  stretching parameter
   SS['L1','TCLINE'] = 50.0                      # critical depth (m)

   SDP = dict()
   SDP['lo_env'] = lo_env
   SDP['parent'] = parent
#    SDP['LO'] = LO
#    SDP['LOo'] = LOo
#    SDP['LOu'] = LOu
#    SDP['data'] = data
   SDP['lv1_run_dir'] = lv1_run_dir
   SDP['lv1_forc_dir'] = lv1_forc_dir   
   SDP['lv1_grid_dir'] = pfm_grid_dir
   SDP['lv1_his_dir'] = lv1_his_dir
   SDP['lv1_plot_dir'] = lv1_plot_dir         
   SDP['lv1_grid_file'] = lv1_grid_file
#   SDP['lv2_run_dir'] = lv2_run_dir
#   SDP['lv2_grid_dir'] = pfm_grid_dir
#   SDP['lv2_his_dir'] = lv2_his_dir
#   SDP['lv2_plot_dir'] = lv2_plot_dir         

   SDP['modtime0'] = modtime0
   SDP['roms_time_units'] = roms_time_units
   SDP['ds_fmt'] = ds_fmt
   SDP['forecast_days'] = fdays
 
   SDP['daystep'] = daystep
   SDP['daystep_ocn'] = daystep_ocn
   SDP['daystep_atm'] = daystep_atm
   SDP['ocn_model'] = ocn_model
   SDP['atm_model'] = atm_model
   SDP['latlonbox'] = latlonbox
   SDP['stretching'] = SS

   return SDP


def make_PFM_directory( parent ):


   pfm_root_dir = parent + '/PFM/'

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
    

# new comment

#script to run LV1 and LV2 and copy figs over.

source /home/ffeddersen/.bashrc
echo "starting the run_forecast_LVs.sh script"
cd /home/ffeddersen/PFM

#########
#Initialize conda, needed for conda activate to work
eval "$(conda shell.bash hook)"
# Activate the desired environment
conda activate PFM-env

conda info --envs
########

dateZ=$(date '+%Y%m%d')
fstdout=../log/LVs_forecast_system_${dateZ}0600Z.log
fsterr=../log/LVs_forecast_system_${dateZ}0600Z_ERROR.log

cd driver
python  -u -W "ignore" driver_run_forecast_all_LVs.py > ${fstdout}  2> >(tee -a ${fstderr} >&2)

cd /home/ffeddersen/PFM
./copy_forecast_to_dataSIO.sh


##  first delete netcdf files on website
#rm -rf /projects/www-users/falk/PFM_Forecast/LV4_His/*.nc

## copy webdata to /dataSIO and website

#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/web*.nc     /projects/www-users/falk/PFM_Forecast/LV4_His/web_data_latest.nc
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/LV4*.nc     /projects/www-users/falk/PFM_Forecast/LV4_His

## 
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/web*.nc     /dataSIO/PFM_Simulations/Archive/web

## copy LV1-LV3 plots to /dataSIO and
#cp -f  /scratch/PFM_Simulations/LV1_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
#cp -f  /scratch/PFM_Simulations/LV2_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
#cp -f  /scratch/PFM_Simulations/LV3_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
## copy LV4 plots to /dataSIO 
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/dye*png    /dataSIO/PFM_Simulations/Plots
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/his*png    /projects/www-users/falk/PFM_Forecast/Plots
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/dye*png    /projects/www-users/falk/PFM_Forecast/Plots

##  copy history files to /dataSIO and to website for LV4
#cp -f  /scratch/PFM_Simulations/LV1_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV1_His
#cp -f  /scratch/PFM_Simulations/LV2_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV2_His
#cp -f  /scratch/PFM_Simulations/LV3_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV3_His
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/LV4*.nc     /dataSIO/PFM_Simulations/Archive/LV4_His


## copy the log files to /dataSIO

#cp -f  /scratch/PFM_Simulations/LV1_Forecast/Run/LV1_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV1_forecast${dateZ}.log

#cp -f  /scratch/PFM_Simulations/LV2_Forecast/Run/LV2_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV2_forecast${dateZ}.log

#cp -f  /scratch/PFM_Simulations/LV3_Forecast/Run/LV3_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV3_forecast${dateZ}.log
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/Run/LV4_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV4_forecast${dateZ}.log

#cp -f  log/LVs_forecast_system_${dateZ}0600Z.log   /dataSIO/PFM_Simulations/Archive/Log




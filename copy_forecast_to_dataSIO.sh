# new comment

#script to copy figs over.

source /home/ffeddersen/.bashrc
echo "copying forecast to /dataSIO"
cd /home/ffeddersen/PFM_NEW


dateZ=$(date '+%Y%m%d')

##


## copy webdata to /dataSIO and website

#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/web*.nc     /dataSIO/PFM_Simulations/Archive/for_web/web_data_latest.nc
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/web*.nc     /projects/www-users/falk/PFM_Forecast/LV4_His/web_data_latest.nc
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/web*.nc     /dataSIO/PFM_Simulations/Archive/web

#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/LV4*.nc     /projects/www-users/falk/PFM_Forecast/LV4_His

##  first delete netcdf files on website
rm -rf /projects/www-users/falk/PFM_Forecast/LV4_His/LV*.nc




##  copy history files to /dataSIO and to website for LV4
cp -f  /scratch/PFM_Simulations/LV1_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV1_His
cp -f  /scratch/PFM_Simulations/LV2_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV2_His
cp -f  /scratch/PFM_Simulations/LV3_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV3_His
# uncomment below when appropriate
cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/LV4*.nc     /dataSIO/PFM_Simulations/Archive/LV4_His
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/LV4*.nc     /home/ffeddersen/PFM/LV4_His
# remove the above when no longer needed

## copy the log files to /dataSIO

cp -f  /scratch/PFM_Simulations/LV1_Forecast/Run/LV1_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV1_forecast${dateZ}.log
cp -f  /scratch/PFM_Simulations/LV2_Forecast/Run/LV2_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV2_forecast${dateZ}.log
cp -f  /scratch/PFM_Simulations/LV3_Forecast/Run/LV3_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV3_forecast${dateZ}.log
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Run/LV4_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV4_forecast${dateZ}.log

#cp -f  log/LVs_forecast_system_${dateZ}0600Z.log   /dataSIO/PFM_Simulations/Archive/Log


#### next copy over to website
##  first delete netcdf files on website
#rm -rf /projects/www-users/falk/PFM_Forecast/LV4_His/*.nc

## copy webdata to /dataSIO and website

#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/web*.nc     /projects/www-users/falk/PFM_Forecast/LV4_His/web_data_latest.nc
#cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/LV4*.nc     /projects/www-users/falk/PFM_Forecast/LV4_His

## first move plots on /dataSIO/PFM_Simulations/Plots

mv -f /dataSIO/PFM_Simulations/Plots/dye*  /dataSIO/PFM_Simulations/Plots/Dye
mv -f /dataSIO/PFM_Simulations/Plots/river*  /dataSIO/PFM_Simulations/Plots/River
mv -f /dataSIO/PFM_Simulations/Plots/his*  /dataSIO/PFM_Simulations/Plots/old_history

## copy LV1-LV3 plots to /dataSIO and copy LV4 plots to /dataSIO 
cp -f  /scratch/PFM_Simulations/LV1_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV2_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV3_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/dye*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/river*png    /dataSIO/PFM_Simulations/Plots


## next deal with the plots on website
## first delete
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/dye*.png  /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/LV4
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/river*.png  /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/LV4
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/*.mp4  /projects/www-users/falk/PFM_Forecast/OLD_MOVIES


# then copy over the plots
cp -f /dataSIO/PFM_Simulations/Plots/river*${dateZ}*.png  /projects/www-users/falk/PFM_Forecast/Plots
cp -f /dataSIO/PFM_Simulations/Plots/dye*${dateZ}*.png  /projects/www-users/falk/PFM_Forecast/Plots
## next run FFMPEG on website

./make_movies.sh
#cd /projects/www-users/falk/PFM_Forecast/Plots

# set up making the animated gifs

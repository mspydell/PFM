# new comment

#script to run LV1 and LV2 and copy figs over.

source /home/ffeddersen/.bashrc
echo "copying forecast to /dataSIO"
cd /home/ffeddersen/PFM


dateZ=$(date '+%Y%m%d')

##

cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/web*.nc     /dataSIO/PFM_Simulations/Archive/web

## copy LV1-LV3 plots to /dataSIO and
cp -f  /scratch/PFM_Simulations/LV1_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV2_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV3_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
## copy LV4 plots to /dataSIO 
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/dye*png    /dataSIO/PFM_Simulations/Plots


##  copy history files to /dataSIO and to website for LV4
cp -f  /scratch/PFM_Simulations/LV1_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV1_His
cp -f  /scratch/PFM_Simulations/LV2_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV2_His
cp -f  /scratch/PFM_Simulations/LV3_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive/LV3_His
cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/LV4*.nc     /dataSIO/PFM_Simulations/Archive/LV4_His


## copy the log files to /dataSIO

cp -f  /scratch/PFM_Simulations/LV1_Forecast/Run/LV1_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV1_forecast${dateZ}.log
cp -f  /scratch/PFM_Simulations/LV2_Forecast/Run/LV2_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV2_forecast${dateZ}.log
cp -f  /scratch/PFM_Simulations/LV3_Forecast/Run/LV3_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV3_forecast${dateZ}.log
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Run/LV4_forecast.log  /dataSIO/PFM_Simulations/Archive/Log/LV4_forecast${dateZ}.log

#cp -f  log/LVs_forecast_system_${dateZ}0600Z.log   /dataSIO/PFM_Simulations/Archive/Log


#### next copy over to website
##  first delete netcdf files on website
rm -rf /projects/www-users/falk/PFM_Forecast/LV4_His/*.nc

## copy webdata to /dataSIO and website

cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/web*.nc     /projects/www-users/falk/PFM_Forecast/LV4_His/web_data_latest.nc
cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/LV4*.nc     /projects/www-users/falk/PFM_Forecast/LV4_His



## next deal with the plots on website
## first delete
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/dye*.png  /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/LV4
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/LV1*.gif  /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/LV1
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/LV2*.gif  /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/LV2
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/LV3*.gif  /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/LV3
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/LV4*.gif  /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/LV4
mv -f  /projects/www-users/falk/PFM_Forecast/Plots/his*.png  /projects/www-users/falk/PFM_Forecast/OLD_PLOTS
rm -f /projects/www-users/falk/PFM_Forecast/OLD_PLOTS/his*png

# then copy over the plots
mv -f /dataSIO/PFM_Simulations/Plots/*${dateZ}*.png  /projects/www-users/falk/PFM_Forecast/Plots
## next run FFMPEG on website

cd /projects/www-users/falk/PFM_Forecast/Plots

# set up making the animated gifs

fname1=LV1_tempuv_${dateZ}0600Z.gif
fname2=LV2_tempuv_${dateZ}0600Z.gif
fname3=LV3_tempuv_${dateZ}0600Z.gif
fname4=LV4_tempuv_${dateZ}0600Z.gif

/usr/bin/ffmpeg -loglevel quiet -y -r 2  -pattern_type glob -i 'his_tempuv_LV1*.png'  ${fname1} 
/usr/bin/ffmpeg -loglevel quiet -y -r 2  -pattern_type glob -i 'his_tempuv_LV2*.png'  ${fname2} 
/usr/bin/ffmpeg -loglevel quiet -y -r 2  -pattern_type glob -i 'his_tempuv_LV3*.png'  ${fname3}
/usr/bin/ffmpeg -loglevel quiet -y -r 2  -pattern_type glob -i 'his_tempuv_LV4*.png'  ${fname4} 

rm -f his_tempuv_LV1*.png  his_tempuv_LV2*.png  his_tempuv_LV3*.png

wname=LV4_Hwave_${dateZ}0600Z.gif
dname=LV4_dye_01_${dateZ}0600Z.gif
ddname=LV4_dye_02_${dateZ}0600Z.gif

/usr/bin/ffmpeg -loglevel quiet -y -r 4  -pattern_type glob -i 'his_Hwave_LV4*.png'  ${wname}
/usr/bin/ffmpeg -loglevel quiet -y -r 4  -pattern_type glob -i 'his_dye_01_LV4*.png'  ${dname}
/usr/bin/ffmpeg -loglevel quiet -y -r 4  -pattern_type glob -i 'his_dye_02_LV4*.png'  ${ddname} 










cd /projects/www-users/falk/PFM_Forecast/Plots

dateZ=$(date '+%Y%m%d')

wname=LV4_Hwave_${dateZ}0600Z.mp4
dname=LV4_dye_01_${dateZ}0600Z.mp4
ddname=LV4_dye_02_${dateZ}0600Z.mp4
fname4=LV4_tempuv_${dateZ}0600Z.mp4
fname3=LV3_tempuv_${dateZ}0600Z.mp4
fname2=LV2_tempuv_${dateZ}0600Z.mp4
fname1=LV1_tempuv_${dateZ}0600Z.mp4

#pwd
#echo ${wname}

#/usr/bin/ffmpeg -loglevel quiet -y -r 10 -c:v libx264 -profile:v high -level 4.0 -c:a aac -movflags +faststart  -pattern_type glob -i 'his_dye_02_LV4*.png'  ${ddname} 

/usr/bin/ffmpeg -loglevel quiet  -pattern_type glob -i '/dataSIO/PFM_Simulations/Plots/his_Hwave_LV4*.png'  -y -r 8 -c:v libx264  -pix_fmt yuv420p  -movflags +faststart    ${wname}
/usr/bin/ffmpeg -loglevel quiet -pattern_type glob -i '/dataSIO/PFM_Simulations/Plots/his_dye_02_LV4*.png'  -y -r 8 -c:v libx264  -pix_fmt yuv420p  -movflags +faststart    ${ddname}
/usr/bin/ffmpeg -loglevel quiet  -pattern_type glob -i '/dataSIO/PFM_Simulations/Plots/his_dye_01_LV4*.png'  -y -r 8 -c:v libx264  -pix_fmt yuv420p  -movflags +faststart    ${dname}
/usr/bin/ffmpeg -loglevel quiet -pattern_type glob -i '/dataSIO/PFM_Simulations/Plots/his_tempuv_LV4*.png'  -y -r 8 -c:v libx264  -pix_fmt yuv420p  -movflags +faststart    ${fname4}

/usr/bin/ffmpeg -loglevel quiet -pattern_type glob -i '/dataSIO/PFM_Simulations/Plots/his_tempuv_LV1*.png'  -y -r 8 -c:v libx264  -pix_fmt yuv420p  -movflags +faststart    ${fname1}
/usr/bin/ffmpeg -loglevel quiet -pattern_type glob -i '/dataSIO/PFM_Simulations/Plots/his_tempuv_LV2*.png'  -y -r 8 -c:v libx264  -pix_fmt yuv420p  -movflags +faststart    ${fname2}
/usr/bin/ffmpeg -loglevel quiet -pattern_type glob -i '/dataSIO/PFM_Simulations/Plots/his_tempuv_LV3*.png'  -y -r 8 -c:v libx264  -pix_fmt yuv420p  -movflags +faststart    ${fname3}

cp  ${fname1}   LV1_tempuv_latest.mp4
cp  ${fname2}   LV2_tempuv_latest.mp4
cp  ${fname3}   LV3_tempuv_latest.mp4
cp  ${fname4}   LV4_tempuv_latest.mp4
cp  ${dname}   LV4_dye01_latest.mp4
cp  ${ddname}   LV4_dye02_latest.mp4
cp  ${wname}   LV4_Hwave_latest.mp4

mv *Z.mp4  /projects/www-users/falk/PFM_Forecast/OLD_MOVIES
cd /home/ffeddersen/PFM_NEW


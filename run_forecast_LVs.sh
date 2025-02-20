

#script to run LV1 and LV2 and copy figs over.

source /home/ffeddersen/.bashrc
echo "starting the run_forecast_LVs.sh script"
cd /home/ffeddersen/PFM

dateZ=$(date '+%Y%m%d')
fout=../log/LVs_forecast_system_${dateZ}0600Z.log
#./clear_LV1_files.sh  # first delete all the LV1 files
cd driver
/home/ffeddersen/anaconda3/bin/python3 -u -W "ignore" driver_run_forecast_all_LVs.py &> ${fout}

cp -f  /scratch/PFM_Simulations/LV1_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV2_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV3_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/Plots
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Plots/dye*png    /dataSIO/PFM_Simulations/Plots

cp -f  /scratch/PFM_Simulations/LV1_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive2.5/LV1_His
cp -f  /scratch/PFM_Simulations/LV2_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive2.5/LV2_His
cp -f  /scratch/PFM_Simulations/LV3_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive2.5/LV3_His
cp -f  /scratch/PFM_Simulations/LV4_Forecast/His/*.nc     /dataSIO/PFM_Simulations/Archive2.5/LV4_His


cp -f  /scratch/PFM_Simulations/LV1_Forecast/Run/LV1_forecast.log  /dataSIO/PFM_Simulations/Archive2.5/Log/LV1_forecast${dateZ}.log

cp -f  /scratch/PFM_Simulations/LV2_Forecast/Run/LV2_forecast.log  /dataSIO/PFM_Simulations/Archive2.5/Log/LV2_forecast${dateZ}.log

cp -f  /scratch/PFM_Simulations/LV3_Forecast/Run/LV3_forecast.log  /dataSIO/PFM_Simulations/Archive2.5/Log/LV3_forecast${dateZ}.log
cp -f  /scratch/PFM_Simulations/LV4_Forecast/Run/LV4_forecast.log  /dataSIO/PFM_Simulations/Archive2.5/Log/LV4_forecast${dateZ}.log

cp -f  log/LVs_forecast_system_${dateZ}0600Z.log   /dataSIO/PFM_Simulations/Archive2.5/Log






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

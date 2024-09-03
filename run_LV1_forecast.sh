

# first delete all the LV1 files
whoami
source /home/ffeddersen/.bashrc
echo "starting the .sh script"
cd /home/ffeddersen/PFM
./clear_LV1_files.sh
cd driver
/home/ffeddersen/anaconda3/bin/python3 -u -W "ignore" driver_run_forecast_LV1_v4.py &> ../log/LV1_forecast_system.log

cp -f  /scratch/PFM_Simulations/LV1_Forecast/Plots/his*png    /dataSIO/PFM_Simulations/LV1_Forecast/Plots

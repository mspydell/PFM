

# first delete all the LV1 files
whoami

echo "starting the .sh script"
cd /home/ffeddersen/PFM
./clear_LV1_files.sh
cd driver
/home/ffeddersen/anaconda3/bin/python3 driver_run_forecast_LV1_v3.py &> ../log/LV1_forecast_system.log

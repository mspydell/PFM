
# first delete all the LV1 files 
./clear_LV1_files.sh
cd driver
python3 driver_run_forecast_LV1_v3.py &> ../log/LV1_forecast_system.log

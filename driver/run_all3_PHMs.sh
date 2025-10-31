#!/bin/bash
#script to run all 3 PHM versions
cd /home/mspydell/models/PFM_root/PFM/driver

echo "starting PHM_ibwc..."
./run_PHM_ibwc.sh &
echo "pausing 5 minutes..."
sleep 5m
echo "starting PHM_vPFM..."
./run_PHM_vPFM.sh &
echo "pausing 5 minutes..."
sleep 5m
echo "starting PHM_nwm..."
./run_PHM_NWM.sh &

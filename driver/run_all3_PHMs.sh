#!/bin/bash
#script to run all 3 PHM versions
cd /home/mspydell/models/PFM_root/PFM/driver

#!/bin/bash

# testing to see if PFM ran the earlier today...
dir0="/scratch/PFM_Simulations/LV4_Forecast/His"
fn1="LV4_ocean_his_" # Replace with the actual path to your file
dateZ=$(date '+%Y%m%d')
tail="*.nc"

# Define the pattern for files to search for
FILE_PATTERN="$fn1$dateZ$tail" # Example: searches for all files ending with .log

# Define the minimum size in bytes (29 GB = 29 * 1024 * 1024 * 1024 bytes)
MIN_SIZE_BYTES=$((29 * 1024 * 1024 * 1024))

# Use find to locate files matching the pattern and size criteria
# -type f: ensures only regular files are considered
# -size +${MIN_SIZE_BYTES}c: finds files larger than MIN_SIZE_BYTES (in bytes)
# -print -quit: prints the first match and then exits, improving efficiency
MATCHING_FILE=$(find ${dir0} -type f -name ${FILE_PATTERN} -size +${MIN_SIZE_BYTES}c -print -quit)

# Check if a matching file was found
if [ -n "$MATCHING_FILE" ]; then
    echo "At least one file matching pattern '${FILE_PATTERN}' and larger than 29 GB exists."
    echo "Thus, PFM seems to have run earlier. Thus,"
    echo "!!!going to start all 3 PHMs!!!"
    # we will assume that ibwc is ahead of nwm and vpfm. 
    echo "starting PHM_ibwc..."
    ./run_PHM_ibwc.sh &
    echo "pausing 1 minute..."
    sleep 1m
    echo "starting PHM_vPFM..."
    ./run_PHM_vPFM.sh &
    echo "pausing 2 minutes..."
    sleep 2m
    echo "starting PHM_nwm..."
    ./run_PHM_NWM.sh &
else
    echo "No files matching pattern '${FILE_PATTERN}' and larger than 29GB were found."
    echo "!!!NOT going to start all 3 PHMs!!!"
fi



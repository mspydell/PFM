#!/bin/bash
#Initialize conda, needed for conda activate to work
eval "$(conda shell.bash hook)"

# Activate the desired environment
conda activate PFM-env

# Your commands that rely on the activated environment go here
python /home/mspydell/models/PFM_root/PFM/driver/check_get_save_hycom_info.py > /dev/null

# Optionally deactivate the environment after use
conda deactivate
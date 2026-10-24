source /home/ffeddersen/.bashrc

set -a
source /home/ffeddersen/PFM_NEW/.env
set +a

# check to see what git branch we are on
EXPECTED_BRANCH="main" # Or "master", "develop", etc.

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch is: $current_branch"

if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
  echo "Not on the '$EXPECTED_BRANCH' branch (on '$current_branch'), switching..."
  # the switch used to be "git switch $EXCPECTED_BRANCH" -- a typo, so the
  # variable was empty and the switch silently did nothing. nothing checked
  # afterwards either, and the exit was commented out, so a failed switch ran
  # the whole forecast from whatever branch happened to be checked out.
  if ! git switch "$EXPECTED_BRANCH"; then
    echo "FATAL: could not switch to '$EXPECTED_BRANCH'. Not running from an"
    echo "unknown branch -- fix the working tree and rerun."
    exit 1
  fi
  current_branch=$(git rev-parse --abbrev-ref HEAD)
  if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
    echo "FATAL: switch reported success but we are on '$current_branch'."
    exit 1
  fi
  echo "Current branch is now: $current_branch"
fi
echo "Successfully on the '$EXPECTED_BRANCH' branch. Proceeding with script..."

#########
#Initialize conda, needed for conda activate to work
eval "$(conda shell.bash hook)"
# Activate the desired environment
conda activate PHM-env

# the caches in /dataSIO/Hydro_QC are shared with backwash: keep new files group-writable
umask 002

########

dateZ=$(date '+%Y%m%d')
fstdout=/home/ffeddersen/PFM_NEW/Hydro_QC/Moorings/stdout.log
fstdout=/home/ffeddersen/PFM_NEW/Hydro_QC/Moorings/stderr.log

cd /home/ffeddersen/PFM_NEW/Hydro_QC/Moorings/SBOO
echo "SBOO mooring"
python3 -u -W "ignore"  plot_sboo_3weeks.py >> ${fstdout}  2> >(tee -a ${fstderr} >&2) 

mv  plots_web/SBOO_currents_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/SBOO_currents_3weeks.png
mv  plots_web/SBOO_TS_contours_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/SBOO_TS_contours_3weeks.png
mv  plots_web/SBOO_density_N2_contours_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/SBOO_density_N2_contours_3weeks.png

cd /home/ffeddersen/PFM_NEW/Hydro_QC/Moorings/PLOO
echo "PLOO mooring"
python3 -u -W "ignore"  plot_ploo_3weeks.py >> ${fstdout}  2> >(tee -a ${fstderr} >&2) 

mv  plots_web/PLOO_currents_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/PLOO_currents_3weeks.png
mv  plots_web/PLOO_TS_contours_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/PLOO_TS_contours_3weeks.png
mv  plots_web/PLOO_density_N2_contours_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/PLOO_density_N2_contours_3weeks.png

cd /home/ffeddersen/PFM_NEW/Hydro_QC/Moorings/IB
echo "IB mooring"
python3 -u -W "ignore" plot_ib_3weeks.py >> ${fstdout}  2> >(tee -a ${fstderr} >&2) 

mv  plots_web/IB-C_currents_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/IB-C_currents_3weeks.png
mv  plots_web/IB_temperature_salinity_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/IB_temperature_salinity_3weeks.png
mv  plots_web/IB_density_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/IB_density_3weeks.png

cd /home/ffeddersen/PFM_NEW/Hydro_QC/Moorings/DelMar
echo "Del Mar mooring"
python3 -u -W "ignore"  plot_delmar_3weeks.py >> ${fstdout}  2> >(tee -a ${fstderr} >&2) 

mv  plots_web/DelMar_currents_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/DelMar_currents_3weeks.png
mv  plots_web/DelMar_TS_contours_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/DelMar_TS_contours_3weeks.png
mv  plots_web/DelMar_density_N2_contours_*_3weeks.png    /projects/www-users/falk/PFM_Forecast/Plots/DelMar_density_N2_contours_3weeks.png

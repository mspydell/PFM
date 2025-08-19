# nohup mpiexec -np 24 ./ ./$lv1_executable$ $lv1_infile_local$ 

echo $lv2_executable$

# format for mpiexec: mpiexec -np <np> <filename_to_execute> <infile_local> > logfile.log 

mpiexec -np $np$ $lv2_executable$ $lv2_infile_local$ > $lv2_logfile_local$

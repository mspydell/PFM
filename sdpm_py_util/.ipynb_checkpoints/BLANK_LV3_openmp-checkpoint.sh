# nohup mpiexec -np 24 ./ ./$lv1_executable$ $lv1_infile_local$ 

echo $lv3_executable$

# format for mpiexec: mpiexec -np <np> <filename_to_execute> <infile_local> > logfile.log 

mpiexec -np $np$ $lv3_executable$ $lv3_infile_local$ > $lv3_logfile_local$

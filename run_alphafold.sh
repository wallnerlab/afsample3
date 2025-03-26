#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 4320
#SBATCH -A berzelius-2022-218
##SBATCH -A berzelius-2022-216
apptainer exec --nv /proj/wallner-b/users/x_bjowa/apps/images/af3-dev.sif python /proj/wallner-b/users/x_bjowa/af3-dev/run_alphafold.py $@

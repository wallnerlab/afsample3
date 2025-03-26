#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 4320
#SBATCH -A berzelius-2022-218
##SBATCH -A berzelius-2022-216
apptainer exec --nv -B /proj/wallner-b/users/x_bjowa/af3-dev:/app/alphafold /proj/wallner-b/users/x_bjowa/apps/images/af3-dev-edit2.sif python /proj/wallner-b/users/x_bjowa/af3-dev/run_alphafold.py $@

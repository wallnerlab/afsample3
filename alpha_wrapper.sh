#!/bin/bash -x

apt-get install cuda-compat-12-2
python /proj/wallner-b/users/x_bjowa/af3-dev/run_alphafold.py $@

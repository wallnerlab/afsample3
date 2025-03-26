#!/bin/bash
#creates an environment to run AF3
AF3_ENV='af3'
module load buildenv-gcccuda/12.1.1-gcc12.3.0
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba create -n $AF3_ENV python=3.11
pip3 install -r dev-requirements.txt
#Fixes problem with missing: GLIBCXX_3.4.29
mamba install -c conda-forge libstdcxx-ng=11.2.0
#pip3 install -r nsc_requirements.txt
#create an editable pip install
pip3 install --no-deps -e .

# Build chemical components database (this binary was installed by pip).
build_data

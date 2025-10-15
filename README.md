# AFsample3
Modelling protein conformational ensembles with Alphafold3

## Setup
Clone repo

``` 
git clone https://github.com/wallnerlab/afsample3.git
git submodule update --init --recursive
``` 

Install Alphafold3 as a package using instructions their [GitHub repo](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md)

#### [OPTIONAL]
Setup for the reference-free state determination system.
``` 
conda env update --file pathfinder/environment.yml
``` 

## Usage

### Generating protein ensembles

Activate the correct environment and run the following command (prferably as an executable).

``` 
## TEST RUN
chmod +x run.sh
./run.sh <PATH_TO_AF3_WEIGHTS> <DB_DIR>
``` 

Update run.sh script to run prediction for your protein.

Key flags to run AFsample3 
1. --msa_rand_fraction : Determines the fraction of MSA columns to mask
2. --num_seeds : Number of unique combination of masked MSAs to use
3. --num_diffusion : Number of models to generate per seed (Each seed uses the same version of MSA)
4. --db_dir : Path to sequecne databases if MSAs not provided


``` 
#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: bash run_model.sh <PATH_TO_AF3_WEIGHTS>"
  exit 1
fi

PATH_TO_AF3_WEIGHTS=$1

python run_afsample3.py \
    --json_path=demo/fold_input.json \
    --model_dir=$PATH_TO_AF3_WEIGHTS \
    --db_dir=$DB_DIR \
    --flash_attention_implementation=xla  \
    --output_dir=demo/output \
    --num_seeds 10 \
    --num_diffusion_samples 5 \
    --msa_rand_fraction 0.4 \
    --shuffle_msa=True \
    --num_msa 1024 \
    --tar_output=False
```

### Refererence-free state selection

Once an ensemble is generated, you can used the [pathfinder](https://github.com/wallnerlab/pathfinder.git) utility to identify unique conformations. 

``` 
chmod +x pathfinder/run_test.sh
./pathfinder/run_test.sh
``` 

Happy sampling !! 

# Cite
1. Alphafold3
2. Alphafold2
3. AFsample2
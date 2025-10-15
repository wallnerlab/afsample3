#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: bash run_model.sh <PATH_TO_AF3_WEIGHTS>"
  exit 1
fi

PATH_TO_AF3_WEIGHTS=$1
DB_DIR=$2

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
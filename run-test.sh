#!/bin/bash
#docker run -it -e JAX_TRACEBACK_FILTERING=off -e XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"   --volume /ssd0/x_yogka/projects/alphafold3/af_input:/root/af_input     --volume /ssd0/x_yogka/projects/alphafold3/af_output:/root/af_output     --volume /spqwsd1/databases/:/root/models     --volume /ssd1/databases/:/root/public_databases     --gpus all     alphafold3:20250121_cu12.0    python run_alphafold.py     --json_path=/root/af_input/fold_input.json     --model_dir=/root/models     --output_dir=/root/af_output --flash_attention_implementation=xla



apptainer run /proj/wallner-b/users/x_bjowa/apps/images/af3-dev.sif python /proj/wallner-b/users/x_bjowa/af3-dev/run_alphafold.py --json_path=$1 --flagfile=/proj/wallner-b/users/x_bjowa/af3-dev/standard.flags



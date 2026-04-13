#!/bin/bash
#SBATCH --job-name=reprover_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --account=yxpeng
#SBATCH --output=prover_%j.log
#SBATCH --error=prover_%j.err

export ELAN_HOME=/scratch/gautschi/wu2700/nl_fl/.elan
export PATH=$ELAN_HOME/bin:$PATH

# Use physical path for cache to avoid symlink/pathlib errors in LeanDojo
export CACHE_DIR=/scratch/gautschi/wu2700/cache
export XDG_CACHE_HOME=$CACHE_DIR
export HF_HOME=$CACHE_DIR/huggingface
export VLLM_CACHE=$CACHE_DIR/vllm
export VLLM_NO_USAGE_STATS=1

ml conda
conda activate ReProver

cd /home/yu1711/NL-FL/ReProver
export PYTHONPATH=.

python prover/evaluate.py \
  --data-path data/leandojo_benchmark_4/random \
  --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
  --split test \
  --num-workers 1 \
  --num-gpus 1 \
  --save-results \
  --num-theorems 1

#!/bin/bash
# Load conda
ml conda

# Create environment
conda create --yes --name ReProver python=3.11 ipython

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ReProver

# Install dependencies (from README)
pip install torch
pip install tqdm loguru deepspeed "pytorch-lightning[extra]" transformers wandb openai rank_bm25 lean-dojo vllm

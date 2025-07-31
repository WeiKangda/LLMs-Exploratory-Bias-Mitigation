#!/bin/bash

#SBATCH --job-name=Winobias_Benchmark       # Job name
#SBATCH --time=01:00:00                 # Maximum runtime of 48 hours
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --mem=16G                       # Request 16GB of memory
#SBATCH --output=logs/winobias_benchmark.%j.out  # Output file
#SBATCH --error=logs/winobias_benchmark.%j.err   # Error file
#SBATCH --gres=gpu:h100:1               # Request 1 H100 GPU
#SBATCH --partition=gpu                 # GPU partition

# Load necessary modules
module load WebProxy/0000

# Activate conda environment
conda activate INL

# Login to Hugging Face
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN_HERE

# Navigate to project directory
cd /scratch/user/u.kw178339/GenderBias

# Run the MMLU benchmark
CUDA_VISIBLE_DEVICES=0 python GenMO/winobias_benchmark.py --model_name Mistral-7B-Instruct-v0.3-cda-finetuned-max_examples_5000_sources_0 --split test
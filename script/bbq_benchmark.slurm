#!/bin/bash

#SBATCH --job-name=BBQ_Benchmark       # Job name
#SBATCH --time=1:00:00                 # Maximum runtime of 1 hour
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --mem=16G                       # Request 16GB of memory
#SBATCH --output=logs/bbq_benchmark.%j.out  # Output file
#SBATCH --error=logs/bbq_benchmark.%j.err   # Error file
#SBATCH --gres=gpu:h100:1               # Request 1 H100 GPU
#SBATCH --partition=gpu                 # GPU partition

# Load necessary modules
module load WebProxy/0000

# Activate conda environment
conda activate genderbias_env

# Login to Hugging Face
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN_HERE

# Run the BBQ benchmark
CUDA_VISIBLE_DEVICES=0 python Benchmarks/BBQ/bbq_benchmark.py
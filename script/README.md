# Slurm Scripts

This directory contains Slurm scripts for running experiments on HPC clusters. These scripts are designed to work with the Gender Bias in Language Models Research project.

## 🚀 Quick Start

### Prerequisites

1. **Hugging Face Token**: Get a Hugging Face access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Environment Setup**: Ensure you have the `genderbias_env` conda environment activated
3. **Project Structure**: Make sure you're in the project root directory when running these scripts

### Setup

Before using these scripts, you need to:

1. Replace `YOUR_HUGGINGFACE_TOKEN_HERE` in the script files with your actual token
2. Ensure the `genderbias_env` conda environment is available on your HPC cluster
3. Make sure all required Python packages are installed (see main `requirements.txt`)

## 📋 Available Scripts

### Bias Mitigation
- `dpo.slurm` - Direct Preference Optimization training
- `finetune.slurm` - Standard fine-tuning
- `finetune_cda.slurm` - Counterfactual Data Augmentation fine-tuning

### Story Generation
- `story_generation.slurm` - Story generation pipeline
- `moral_stance_update.slurm` - Moral stance update processing
- `swap_explanation.slurm` - Swap and rewrite explanations for CDA

### Benchmark Evaluation
- `bbq_benchmark.slurm` - BBQ benchmark evaluation
- `mmlu_benchmark.slurm` - MMLU benchmark evaluation
- `truthfulqa_benchmark.slurm` - TruthfulQA benchmark evaluation
- `winobias_benchmark.slurm` - WinoBias benchmark evaluation
- `evaluate_genmo.slurm` - GenMO evaluation
- `evaluate_genmo_icl.slurm` - GenMO evaluation with In-Context Learning

## ⚡ Usage

```bash
# Submit a job to the queue
sbatch script_name.slurm

# Example: Run story generation
sbatch story_generation.slurm

# Example: Run WinoBias benchmark
sbatch winobias_benchmark.slurm
```

## 📊 Output

- **Log Files**: Scripts output logs to the current directory or `logs/` subdirectory
- **Results**: Generated data and results are saved in their respective directories as specified in the main project structure
- **Job Status**: Monitor job status using `squeue` or `sacct` commands

## 🔧 Customization

### Memory and Time Limits
Adjust the following parameters in each script as needed:
- `#SBATCH --mem=32G` - Memory allocation
- `#SBATCH --time=2:00:00` - Maximum runtime
- `#SBATCH --gres=gpu:h100:1` - GPU requirements

### Model Parameters
Most scripts use the default model `meta-llama/Llama-3.1-8B-Instruct`. You can modify the model name in the Python command line arguments.

## 📝 Notes

- All scripts use relative paths and assume execution from the project root directory
- The `genderbias_env` conda environment must be available on your HPC cluster
- Ensure your Hugging Face token has access to the required models
- Monitor resource usage and adjust limits based on your cluster's specifications

## 🔗 Related Documentation

- [Main README.md](../README.md) - Project overview and setup instructions
- [Story Generation](../StoryGeneration/) - Story generation pipeline details
- [Benchmarks](../Benchmarks/) - Evaluation benchmark descriptions
- [Gender Bias Mitigation](../GenderBiasMitigation/) - Bias mitigation techniques 
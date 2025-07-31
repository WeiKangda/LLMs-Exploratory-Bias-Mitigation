# Slurm Scripts

This directory contains Slurm scripts for running experiments on HPC clusters.

## Setup

Before using these scripts, you need to:

1. Get a Hugging Face access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Replace `YOUR_HUGGINGFACE_TOKEN_HERE` in the script files with your actual token

## Available Scripts

- `dpo.slurm` - Direct Preference Optimization training
- `finetune.slurm` - Standard fine-tuning
- `finetune_cda.slurm` - Counterfactual Data Augmentation fine-tuning
- `bbq_benchmark.slurm` - BBQ benchmark evaluation
- `mmlu_benchmark.slurm` - MMLU benchmark evaluation
- `truthfulqa_benchmark.slurm` - TruthfulQA benchmark evaluation
- `winobias_benchmark.slurm` - WinoBias benchmark evaluation
- `evaluate_genmo.slurm` - GenMO evaluation
- `evaluate_genmo_icl.slurm` - GenMO evaluation with ICL
- `story_generation.slurm` - Story generation
- `moral_stance_update.slurm` - Moral stance update
- `moral_stance_update_mistral.slurm` - Moral stance update (Mistral)
- `swap_explanation.slurm` - Swap and rewrite explanations

## Usage

```bash
sbatch script_name.slurm
```

## Note

The original slurm scripts contained Hugging Face tokens that were removed for security reasons. You'll need to add your own token to use these scripts. 
# Story Generation Script

This directory contains scripts for a three-step story generation and processing pipeline that supports both Llama and Mistral models.

## Pipeline Overview

The story generation process consists of three sequential steps:

1. **Story Generation** - Generate initial stories with moral stance validation
2. **Filtering** - Filter and clean the generated stories
3. **Moral Stance Update** - Update moral stances with neutral explanations

## Step 1: Story Generation

### Basic Usage

```bash
# Use Llama model (default)
python story_generation.py

# Use Mistral model
python story_generation.py --model mistral
```

### Advanced Usage

```bash
# Generate 100 stories with Mistral
python story_generation.py --model mistral --num_stories 100

# Use custom cache and output directories
python story_generation.py --model llama --cache_dir ./my_models --output_dir ./my_output

# Enable debug output
python story_generation.py --model mistral --debug

# Generate stories without character names
python story_generation.py --model llama --with_character False
```

## Step 2: Filter Stories

```bash
# Filter stories for Llama model (default)
python filter.py

# Filter stories for Mistral model
python filter.py --model mistral

# Use custom input and output files
python filter.py --model llama --input_file ./my_input.jsonl --output_file ./my_output.jsonl

# Adjust ROUGE-1 thresholds and enable debug output
python filter.py --model mistral --rouge1_lower 0.7 --rouge1_upper 0.9 --debug
```

**Command Line Arguments:**
- `--model`: Model to use (choices: 'llama', 'mistral', default: 'llama')
- `--input_file`: Input file path (default: model-specific)
- `--output_file`: Output file path (default: model-specific)
- `--rouge1_lower`: Lower threshold for ROUGE-1 score (default: 0.8)
- `--rouge1_upper`: Upper threshold for ROUGE-1 score (default: 0.95)
- `--debug`: Print debug information (default: False)

## Step 3: Update Moral Stances

```bash
# Update moral stances for Llama model (default)
python moral_stance_update.py

# Update moral stances for Mistral model
python moral_stance_update.py --model mistral

# Use custom input and output files
python moral_stance_update.py --model llama --input_file ./my_input.jsonl --output_file ./my_output.jsonl

# Adjust temperature and enable debug output
python moral_stance_update.py --model mistral --temperature 0.5 --debug
```

**Command Line Arguments:**
- `--model`: Model to use (choices: 'llama', 'mistral', default: 'llama')
- `--cache_dir`: Directory to cache models (default: './models')
- `--input_file`: Input file path (default: model-specific)
- `--output_file`: Output file path (default: model-specific)
- `--temperature`: Temperature for generation (default: 0.7)
- `--debug`: Print debug information (default: False)

## Complete Pipeline Examples

### Llama Model Pipeline
```bash
# Step 1: Generate stories
python story_generation.py --model llama --num_stories 50

# Step 2: Filter stories
python filter.py --model llama

# Step 3: Update moral stances
python moral_stance_update.py --model llama
```

### Mistral Model Pipeline
```bash
# Step 1: Generate stories
python story_generation.py --model mistral --num_stories 50

# Step 2: Filter stories
python filter.py --model mistral

# Step 3: Update moral stances
python moral_stance_update.py --model mistral
```

## Command Line Arguments (Story Generation)

- `--model`: Model to use for story generation (choices: 'llama', 'mistral', default: 'llama')
- `--num_stories`: Number of stories to generate (default: 50)
- `--with_character`: Generate stories with character names (default: True)
- `--debug`: Print debug information (default: False)
- `--cache_dir`: Directory to cache models (default: './models')
- `--output_dir`: Directory for output files (default: './StoryGeneration')

## Output Files

The pipeline generates output files in the specified output directory:
- `generated_story_llama.jsonl` / `generated_story_mistral.jsonl` - Initial stories
- `generated_story_filtered_llama.jsonl` / `generated_story_filtered_mistral.jsonl` - Filtered stories
- `generated_data_llama.jsonl` / `generated_data_mistral.jsonl` - Final processed data

## Model Configuration

The script automatically configures model-specific parameters:

### Llama Model
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Validation temperature: 0.1
- Validation sampling: False (deterministic)

### Mistral Model
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Validation temperature: 0.1
- Validation sampling: False (deterministic)

## Requirements

- PyTorch
- Transformers
- CUDA-compatible GPU (recommended)
- Required utility modules from the StoryGeneration package
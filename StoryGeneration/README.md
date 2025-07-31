# Story Generation Script

This script combines the functionality of both Llama and Mistral story generation into a single, configurable script.

## Usage

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

## Command Line Arguments

- `--model`: Model to use for story generation (choices: 'llama', 'mistral', default: 'llama')
- `--num_stories`: Number of stories to generate (default: 50)
- `--with_character`: Generate stories with character names (default: True)
- `--debug`: Print debug information (default: False)
- `--cache_dir`: Directory to cache models (default: './models')
- `--output_dir`: Directory for output files (default: './StoryGeneration')

## Output Files

The script generates output files in the specified output directory:
- `generated_story_llama.jsonl` for Llama model
- `generated_story_mistral.jsonl` for Mistral model

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
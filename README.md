# Gender Bias in Language Models Research

[![arXiv](https://img.shields.io/badge/arXiv-2505.17217-b31b1b.svg)](https://arxiv.org/abs/2505.17217)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange.svg)](https://huggingface.co/)

This repository contains the implementation and resources for the paper: **"Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs"** ([arXiv:2505.17217](https://arxiv.org/abs/2505.17217)).

This repository contains research on gender bias in language models, including bias detection, mitigation techniques, and evaluation across multiple benchmarks.

![Introduction Figure](gender_bias_intro.png)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Story Generation](#story-generation)
- [Bias Mitigation](#bias-mitigation)
- [Benchmark Evaluation](#benchmark-evaluation)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## 🎯 Project Overview

This research focuses on mitigating gender bias in language models through exploratory thinking techniques. The project includes:

- **Story Generation Pipeline**: Multi-step process for generating gender-balanced narratives, and Counterfactual Data Augmentation (CDA)
- **Bias Mitigation Techniques**: DPO, fine-tuning
- **Comprehensive Evaluation**: Multiple benchmarks including WinoBias, MMLU, TruthfulQA, BBQ, and GenMO

## 📁 Project Structure

```
├── 📊 Benchmarks/           # Evaluation benchmarks
│   ├── BBQ/                # BBQ benchmark for bias evaluation
│   ├── GenMO/              # GenMO moral reasoning dataset
│   ├── MMLU/               # Massive Multitask Language Understanding
│   ├── TruthfulQA/         # TruthfulQA benchmark
│   └── WinoBias/           # WinoBias benchmark
├── 🔧 GenderBiasMitigation/ # Bias mitigation techniques
│   ├── dpo.py              # Direct Preference Optimization
│   ├── fine_tune.py        # Standard fine-tuning
│   ├── fine_tune_cda.py    # Fine-tuning with Counterfactual Data Augmentation
│   └── resolution_analysis.py
├── 📚 StoryGeneration/     # Story generation and analysis
│   ├── story_generation.py      # Combined script for Llama and Mistral models
│   ├── moral_stance_update.py   # Combined script for moral stance updates
│   ├── swap_and_rewrite_explanations.py
│   ├── calculate_story_similarity.py
│   ├── prompts.py
│   ├── utils.py
│   ├── filter.py
│   ├── generated_data_llama.jsonl
│   ├── generated_data_mistral.jsonl
│   ├── swapped_explanations_llama.jsonl
│   └── swapped_explanations_mistral.jsonl
└── ⚙️ script/              # Slurm scripts for HPC
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Git
- Hugging Face account (for model access)

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WeiKangda/LLMs-Exploratory-Bias-Mitigation.git
   cd LLMs-Exploratory-Bias-Mitigation
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Using venv (recommended)
   python -m venv genderbias_env
   source genderbias_env/bin/activate  # On macOS/Linux
   # or
   genderbias_env\Scripts\activate     # On Windows

   # Alternative: Using conda
   conda create -n genderbias_env python=3.9
   conda activate genderbias_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Hugging Face token (for Slurm scripts):**
   - Get a Hugging Face access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Replace `YOUR_HUGGINGFACE_TOKEN_HERE` in the slurm script files with your actual token

## ⚡ Quick Start

### Basic Usage

```bash
# 1. Generate stories using the complete pipeline
python StoryGeneration/story_generation.py --model llama --num_stories 50
python StoryGeneration/filter.py --model llama
python StoryGeneration/moral_stance_update.py --model llama

# 2. Run bias mitigation experiments
python GenderBiasMitigation/fine_tune.py --model_name llama --output_dir ./results

# 3. Evaluate on benchmarks
python Benchmarks/WinoBias/winobias_benchmark.py --model_name llama
```

## 📊 Data

This repository includes pre-generated data files used in the paper experiments:

| File | Description | Model |
|------|-------------|-------|
| `StoryGeneration/generated_data_llama.jsonl` | Generated stories for Llama model | Llama |
| `StoryGeneration/generated_data_mistral.jsonl` | Generated stories for Mistral model | Mistral |
| `StoryGeneration/swapped_explanations_llama.jsonl` | Swapped explanations for CDA experiments | Llama |
| `StoryGeneration/swapped_explanations_mistral.jsonl` | Swapped explanations for CDA experiments | Mistral |

### Usage Options

#### 🎯 Direct Use
Use the provided datasets to reproduce the exact experiments reported in the paper.

#### 🔄 Generate New Data
Create bias mitigation datasets for new models:

```bash
# Generate new stories for your model
python StoryGeneration/story_generation.py --model <your_model_name> --num_stories <desired_count>

# Filter and process the generated stories
python StoryGeneration/filter.py --model <your_model_name>
python StoryGeneration/moral_stance_update.py --model <your_model_name>

# Generate counterfactual data for CDA experiments
python StoryGeneration/swap_and_rewrite_explanations.py --model <your_model_name>
```

## 📚 Story Generation

The story generation process consists of three sequential steps:

### Step 1: Generate Stories

```bash
# Use Llama model (default)
python StoryGeneration/story_generation.py

# Use Mistral model
python StoryGeneration/story_generation.py --model mistral

# Generate 100 stories with Mistral
python StoryGeneration/story_generation.py --model mistral --num_stories 100

# Use custom cache and output directories
python StoryGeneration/story_generation.py --model llama --cache_dir ./my_models --output_dir ./my_output

# Enable debug output
python StoryGeneration/story_generation.py --model mistral --debug

# ⚠️ Generate stories without character names (Not Recommended)
# Results reported in paper used prompts including character names to avoid hallucination
python StoryGeneration/story_generation.py --model llama --with_character False
```

**Command Line Arguments:**
| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model` | Model to use | `llama` | `llama`, `mistral` |
| `--num_stories` | Number of stories to generate | `50` | Any integer |
| `--with_character` | Generate stories with character names | `True` | `True`, `False` |
| `--debug` | Print debug information | `False` | `True`, `False` |
| `--cache_dir` | Directory to cache models | `./models` | Any path |
| `--output_dir` | Directory for output files | `./StoryGeneration` | Any path |

### Step 2: Filter Stories

```bash
# Filter stories for Llama model (default)
python StoryGeneration/filter.py

# Filter stories for Mistral model
python StoryGeneration/filter.py --model mistral

# Use custom input and output files
python StoryGeneration/filter.py --model llama --input_file ./my_input.jsonl --output_file ./my_output.jsonl

# Adjust ROUGE-1 thresholds and enable debug output
python StoryGeneration/filter.py --model mistral --rouge1_lower 0.7 --rouge1_upper 0.9 --debug
```

**Command Line Arguments:**
| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model` | Dataset generated by which model for filtering | `llama` | `llama`, `mistral` |
| `--input_file` | Input file path | Model-specific | Any path |
| `--output_file` | Output file path | Model-specific | Any path |
| `--rouge1_lower` | Lower threshold for ROUGE-1 score | `0.8` | Float 0-1 |
| `--rouge1_upper` | Upper threshold for ROUGE-1 score | `0.95` | Float 0-1 |
| `--debug` | Print debug information | `False` | `True`, `False` |

### Step 3: Update Moral Stances

```bash
# Update moral stances for Llama model (default)
python StoryGeneration/moral_stance_update.py

# Update moral stances for Mistral model
python StoryGeneration/moral_stance_update.py --model mistral

# Use custom input and output files
python StoryGeneration/moral_stance_update.py --model llama --input_file ./my_input.jsonl --output_file ./my_output.jsonl

# Adjust temperature and enable debug output
python StoryGeneration/moral_stance_update.py --model mistral --temperature 0.5 --debug
```

**Command Line Arguments:**
| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model` | Model to use | `llama` | `llama`, `mistral` |
| `--cache_dir` | Directory to cache models | `./models` | Any path |
| `--input_file` | Input file path | Model-specific | Any path |
| `--output_file` | Output file path | Model-specific | Any path |
| `--temperature` | Temperature for generation | `0.7` | Float 0-2 |
| `--debug` | Print debug information | `False` | `True`, `False` |

### Complete Pipeline Example

```bash
# Complete pipeline for Llama model
python StoryGeneration/story_generation.py --model llama --num_stories 50
python StoryGeneration/filter.py --model llama
python StoryGeneration/moral_stance_update.py --model llama

# Complete pipeline for Mistral model
python StoryGeneration/story_generation.py --model mistral --num_stories 50
python StoryGeneration/filter.py --model mistral
python StoryGeneration/moral_stance_update.py --model mistral
```

### Swap and Rewrite Explanations (CDA Experiments)

> **⚠️ Important Note**: For fair comparison between CDA and our bias mitigation method, this script should be run on files generated with the complete 3-step story generation pipeline (story generation → filtering → moral stance update).

```bash
# Generate counterfactual data for CDA (Counterfactual Data Augmentation) experiments
python StoryGeneration/swap_and_rewrite_explanations.py --input_file <input_file>
```

This script performs counterfactual data augmentation by:

1. **Loading story pairs** with male/female protagonists and their moral explanations
2. **Swapping explanations** between male and female stories
3. **Rewriting explanations** using Llama 3.1 to match character names and pronouns in the target story
4. **Preserving moral reasoning** while only changing character-specific references
5. **Saving results** with checkpointing for resuming interrupted runs

**Command Line Arguments:**
| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model` | Model to use | `mistral` | `llama`, `mistral` |
| `--input_file` | Input file path | Model-specific | Any path |
| `--output_file` | Output file path | Model-specific | Any path |
| `--checkpoint_file` | Checkpoint file path for resuming | Model-specific | Any path |
| `--cache_dir` | Directory to cache models | `./models` | Any path |
| `--checkpoint_interval` | Save checkpoint every N items | `10` | Any integer |
| `--temperature` | Temperature for LLM generation | `0.7` | Float 0-2 |

**Usage Examples:**
```bash
# Basic usage with default settings
python StoryGeneration/swap_and_rewrite_explanations.py

# Use Llama model with custom input/output files
python StoryGeneration/swap_and_rewrite_explanations.py --model llama --input_file ./my_data.jsonl --output_file ./my_output.jsonl

# Adjust temperature and checkpoint interval
python StoryGeneration/swap_and_rewrite_explanations.py --model mistral --temperature 0.5 --checkpoint_interval 5

# Use custom cache directory
python StoryGeneration/swap_and_rewrite_explanations.py --model llama --cache_dir ./my_models
```

**Output Format:**
The script creates counterfactual data where:
- Male stories get female explanations (rewritten to match male characters)
- Female stories get male explanations (rewritten to match female characters)
- Original explanations are preserved as `original_explanation`
- Swapping source is tracked as `swapped_from`
- Processing status and errors are logged

### Calculate Story Similarity

```bash
python StoryGeneration/calculate_story_similarity.py --input_file <input_file>
```

## 🔧 Bias Mitigation

This section provides detailed documentation for the bias mitigation scripts implemented in this repository.

### Direct Preference Optimization (DPO)

**File**: `GenderBiasMitigation/dpo.py`

**Description**: Implements Direct Preference Optimization (DPO) training for gender bias mitigation. DPO trains models to prefer neutral explanations over gender-biased ones by learning from preference pairs.

**Command Line Arguments**:
| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model_name` | Model to use for training | `meta-llama/Llama-3.1-8B-Instruct` | Any Hugging Face model |
| `--max_examples` | Maximum number of examples to use | `125` | Any integer |
| `--cache_dir` | Directory to cache models | `./models` | Any path |
| `--dataset_path` | Path to dataset file | Auto-determined | Any path |
| `--output_dir` | Output directory for results | `./dpo_results` | Any path |

**Usage Examples**:
```bash
# Basic DPO training with default settings
python GenderBiasMitigation/dpo.py --model_name meta-llama/Llama-3.1-8B-Instruct

# DPO training with custom dataset and more examples
python GenderBiasMitigation/dpo.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --max_examples 250 --output_dir ./my_dpo_results

# DPO training with custom dataset path
python GenderBiasMitigation/dpo.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset_path ./custom_data.jsonl
```

**Grid Search Parameters**:
| Parameter | Values Tested | Description |
|-----------|---------------|-------------|
| `beta` | `[1.0]` | DPO temperature parameter |
| `batch_size` | `[4]` | Training batch size |
| `gradient_accumulation_steps` | `[4]` | Gradient accumulation steps |
| `num_epochs` | `[3]` | Number of training epochs |
| `learning_rate` | `[1e-5]` | Learning rate |

**Output Structure**:
```
dpo_results/
├── {model_name}_grid_search_results/
│   ├── {model_name}_grid_search_{timestamp}/
│   │   ├── param_grid.json              # Parameter grid configuration
│   │   ├── best_config.json             # Best configuration found
│   │   ├── run_0_beta=1.0_batch_size=4_.../
│   │   │   ├── config.json              # Configuration for this run
│   │   │   ├── metrics.json             # Training metrics
│   │   │   └── checkpoint-{step}/       # Model checkpoints
│   │   └── ...
└── {model_name}-dpo-{params}-{timestamp}/
    ├── config.json                      # Model configuration
    ├── pytorch_model.bin                # Merged model weights
    ├── tokenizer.json                   # Tokenizer configuration
    └── best_config.json                 # Best training configuration
```

### Standard Fine-tuning

**File**: `GenderBiasMitigation/fine_tune.py`

**Description**: Implements standard supervised fine-tuning for gender bias mitigation using neutral explanations as training targets.

**Command Line Arguments**:
| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model_name` | Model to use for training | `mistralai/Mistral-7B-Instruct-v0.3` | Any Hugging Face model |
| `--cache_dir` | Directory to cache models | `./models` | Any path |
| `--dataset_path` | Path to dataset file | Auto-determined | Any path |
| `--output_dir` | Output directory for results | `./finetune_results` | Any path |
| `--max_examples` | Maximum number of examples to use | `5000` | Any integer |
| `--example_source` | Example sources to use | `[0]` | List of integers |

**Usage Examples**:
```bash
# Basic fine-tuning with default settings
python GenderBiasMitigation/fine_tune.py --model_name mistralai/Mistral-7B-Instruct-v0.3

# Fine-tuning with custom parameters
python GenderBiasMitigation/fine_tune.py --model_name meta-llama/Llama-3.1-8B-Instruct --max_examples 1000 --output_dir ./my_finetune_results

# Fine-tuning with specific data sources
python GenderBiasMitigation/fine_tune.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --example_source 0 1 2
```

**Training Configuration**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `per_device_train_batch_size` | `2` | Batch size per device |
| `gradient_accumulation_steps` | `8` | Gradient accumulation steps |
| `num_train_epochs` | `1` | Number of training epochs |
| `learning_rate` | `2e-4` | Learning rate |
| `warmup_steps` | `100` | Warmup steps |
| `max_length` | `512` | Maximum sequence length |
| `save_steps` | `500` | Save checkpoint every N steps |

**LoRA Configuration**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | `64` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha parameter |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Target modules for LoRA |
| `lora_dropout` | `0.05` | LoRA dropout rate |
| `bias` | `"none"` | Bias handling strategy |

**Output Structure**:
```
finetune_results/
├── {model_name}-lora-finetuned-{params}/
│   ├── config.json                      # Model configuration
│   ├── pytorch_model.bin                # LoRA weights
│   ├── adapter_config.json              # LoRA adapter configuration
│   └── training_args.bin                # Training arguments
├── {model_name}-finetuned-{params}/
│   ├── config.json                      # Merged model configuration
│   ├── pytorch_model.bin                # Merged model weights
│   ├── tokenizer.json                   # Tokenizer configuration
│   └── dataset_params.json              # Dataset parameters used
└── logs/
    └── {model_name}-{params}/
        └── events.out.tfevents.*        # Training logs
```

### Counterfactual Data Augmentation (CDA)

**File**: `GenderBiasMitigation/fine_tune_cda.py`

**Description**: Implements fine-tuning with Counterfactual Data Augmentation (CDA) for gender bias mitigation. This technique uses both original and counterfactual explanations to train models.

**Command Line Arguments**:
| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model_name` | Model to use for training | `mistralai/Mistral-7B-Instruct-v0.3` | Any Hugging Face model |
| `--cache_dir` | Directory to cache models | `./models` | Any path |
| `--dataset_path` | Path to dataset file | Auto-determined | Any path |
| `--output_dir` | Output directory for results | `./finetune_cda_results` | Any path |
| `--max_examples` | Maximum number of examples to use | `5000` | Any integer |
| `--example_source` | Example sources to use | `[0]` | List of integers |

**Usage Examples**:
```bash
# Basic CDA fine-tuning with default settings
python GenderBiasMitigation/fine_tune_cda.py --model_name mistralai/Mistral-7B-Instruct-v0.3

# CDA fine-tuning with custom parameters
python GenderBiasMitigation/fine_tune_cda.py --model_name meta-llama/Llama-3.1-8B-Instruct --max_examples 1000 --output_dir ./my_cda_results

# CDA fine-tuning with specific data sources
python GenderBiasMitigation/fine_tune_cda.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --example_source 0 1 2
```

**Training Configuration**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `per_device_train_batch_size` | `2` | Batch size per device |
| `gradient_accumulation_steps` | `8` | Gradient accumulation steps |
| `num_train_epochs` | `1` | Number of training epochs |
| `learning_rate` | `2e-4` | Learning rate |
| `warmup_steps` | `100` | Warmup steps |
| `max_length` | `512` | Maximum sequence length |
| `save_steps` | `500` | Save checkpoint every N steps |

**LoRA Configuration**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | `64` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha parameter |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Target modules for LoRA |
| `lora_dropout` | `0.05` | LoRA dropout rate |
| `bias` | `"none"` | Bias handling strategy |

**Data Structure**:
Each story generates 4 training examples:
1. **Male + Original Explanation**: Male story with original male explanation
2. **Male + Current Explanation**: Male story with swapped female explanation
3. **Female + Original Explanation**: Female story with original female explanation
4. **Female + Current Explanation**: Female story with swapped male explanation

**Output Structure**:
```
finetune_cda_results/
├── {model_name}-cda-lora-finetuned-{params}/
│   ├── config.json                      # Model configuration
│   ├── pytorch_model.bin                # LoRA weights
│   ├── adapter_config.json              # LoRA adapter configuration
│   └── training_args.bin                # Training arguments
├── {model_name}-cda-finetuned-{params}/
│   ├── config.json                      # Merged model configuration
│   ├── pytorch_model.bin                # Merged model weights
│   ├── tokenizer.json                   # Tokenizer configuration
│   └── dataset_params.json              # Dataset parameters used
└── logs/
    └── {model_name}-{params}/
        └── events.out.tfevents.*        # Training logs
```

### Comparison of Techniques

| Technique | Training Method | Data Usage | Key Advantage | Best For |
|-----------|----------------|------------|---------------|----------|
| **DPO** | Preference Learning | Preference pairs | Learns to prefer neutral over biased | When you want explicit preference learning |
| **Fine-tuning** | Supervised Learning | Input-output pairs | Simple and effective | Standard supervised learning approach |
| **CDA** | Supervised Learning | Original + Counterfactual | Enhanced data diversity | When you want to leverage counterfactual reasoning |

## 📊 Benchmark Evaluation
This section conatins the five evaluation datasets we reported in our paper. More details can be found in ./Benchmarks/{dataset_name}.

### WinoBias

```bash
python Benchmarks/WinoBias/winobias_benchmark.py --model_name <model_name>
```

### MMLU

```bash
python Benchmarks/MMLU/mmlu_benchmark.py --model_name <model_name>
```

### TruthfulQA

```bash
python Benchmarks/TruthfulQA/truthfulqa_benchmark.py --model_name <model_name>
```

### BBQ

```bash
python Benchmarks/BBQ/bbq_benchmark.py --model_name <model_name>
```

### GenMO

```bash
# Standard evaluation
python Benchmarks/GenMO/evaluate_genmo.py --model_name <model_name>

# Few-shot learning evaluation with in-context examples
python Benchmarks/GenMO/evaluate_genmo_icl.py --model_name <model_name> --shot_number <number>

# Examples:
python Benchmarks/GenMO/evaluate_genmo_icl.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --shot_number 2
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{wei2025mitigatinggenderbiasfostering,
      title={Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs}, 
      author={Kangda Wei and Hasnat Md Abdullah and Ruihong Huang},
      year={2025},
      eprint={2505.17217},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.17217}, 
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests if applicable
5. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

## 📞 Contact

- **GitHub Issues**: [Open an issue](https://github.com/WeiKangda/LLMs-Exploratory-Bias-Mitigation/issues)
- **Email**: kangda@tamu.edu
- **Paper**: [arXiv:2505.17217](https://arxiv.org/abs/2505.17217)

---

<div align="center">

**⭐ If you find this repository helpful, please give it a star! ⭐**

</div> 
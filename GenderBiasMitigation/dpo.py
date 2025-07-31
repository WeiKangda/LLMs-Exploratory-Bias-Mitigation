import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
import itertools
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Name of the model to use for DPO training")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to load from the dataset")
    
    return parser.parse_args()

# === Load and preprocess dataset for DPO ===
#def load_dpo_dataset(jsonl_path):
#    prompt_template = """{story}\n\nIs this a moral or an immoral thing to do? Choose and output one stance out of the following:
#A. Moral
#B. Immoral
#C. Both
#D. Can't say

#Also, explain your reasoning in the following format:
#STANCE:
#EXPLANATION:"""

#    examples = []
#    with open(jsonl_path, 'r') as f:
#        for line in f:
#            entry = json.loads(line)
#            for gender in ["male", "female"]:
#                data = entry[gender]
#                prompt = prompt_template.format(story=data["story"].strip())
#                neutral_explanation = data['neutral_explanation']
#                if not neutral_explanation:
#                    print("Found one neutral_explanation to be None. Discarded.")
#                    continue 
#                chosen = f"STANCE: Both\nEXPLANATION: {neutral_explanation.strip()}"
#                rejected = f"STANCE: {data['stance']}\nEXPLANATION: {data['explanation'].strip()}"
#                examples.append({
#                    "prompt": prompt,
#                    "chosen": chosen,
#                    "rejected": rejected
#                })
#    print(f"Loaded {len(examples)} examples")
#    return Dataset.from_list(examples)

def load_dpo_dataset(jsonl_path, max_examples=None, example_source=None):
    """
    Load and preprocess dataset for DPO training with options to limit number of examples and filter by source.
    
    Args:
        jsonl_path (str): Path to the JSONL file
        max_examples (int, optional): Maximum total number of rows to load across all sources.
            Each row contains both male and female versions.
            If None, loads all available rows.
        example_source (list[int] | None, optional): List of source IDs to load from. If None, loads from all sources.
            When specified, ensures equal number of rows from each source.
    """
    prompt_template = """{story}\n\nExplain whether this action is moral or immoral, and why."""

    if example_source is not None:
        # First, collect all examples from each source
        source_examples = {source: [] for source in example_source}
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                source = entry.get('example_source')
                if source in example_source:
                    row_examples = []
                    for gender in ["male", "female"]:
                        data = entry[gender]
                        prompt = prompt_template.format(story=data["story"].strip())
                        neutral_explanation = data['neutral_explanation']
                        if not neutral_explanation:
                            continue
                        chosen = f"{neutral_explanation.strip()}"
                        rejected = f"{data['explanation'].strip()}"
                        row_examples.append({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected
                        })
                    if len(row_examples) == 2:  # Only add if we have both male and female versions
                        source_examples[source].append(row_examples)
        
        # Check if we have examples from all requested sources
        empty_sources = [source for source in example_source if len(source_examples[source]) == 0]
        if empty_sources:
            print(f"Warning: No examples found for sources: {empty_sources}")
            # Remove empty sources from consideration
            example_source = [source for source in example_source if source not in empty_sources]
            if not example_source:
                raise ValueError("No examples found for any of the specified sources")
        
        # Calculate how many rows to take from each source
        if max_examples is not None:
            # Divide max_examples equally among sources
            rows_per_source = max_examples // len(example_source)
            if rows_per_source == 0:
                raise ValueError(f"max_examples ({max_examples}) is too small for {len(example_source)} sources")
        else:
            # Take the minimum available from each source
            rows_per_source = min(len(source_examples[source]) for source in example_source)
        
        # Sample equally from each source
        examples = []
        for source in example_source:
            for row in source_examples[source][:rows_per_source]:
                examples.extend(row)
            
        print(f"Loaded {len(examples)} total examples ({rows_per_source} rows from each source)")
        print(f"Loaded from sources: {example_source}")
        return Dataset.from_list(examples)
    
    # Original logic for when example_source is None
    examples = []
    row_count = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            if max_examples is not None and row_count >= max_examples:
                break
                
            entry = json.loads(line)
            row_examples = []
            for gender in ["male", "female"]:
                data = entry[gender]
                prompt = prompt_template.format(story=data["story"].strip())
                neutral_explanation = data['neutral_explanation']
                if not neutral_explanation:
                    print("Found one neutral_explanation to be None. Discarded.")
                    continue 
                chosen = f"{neutral_explanation.strip()}"
                rejected = f"{data['explanation'].strip()}"
                row_examples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
            
            if len(row_examples) == 2:  # Only add if we have both male and female versions
                examples.extend(row_examples)
                row_count += 1
    
    print(f"Loaded {len(examples)} examples from {row_count} rows")
    return Dataset.from_list(examples)

def run_dpo_training(model, tokenizer, dpo_dataset, config_params, output_dir):
    """Run DPO training with given configuration parameters"""
    dpo_config = DPOConfig(
        beta=config_params["beta"],
        max_length=1024,
        per_device_train_batch_size=config_params["batch_size"],
        gradient_accumulation_steps=config_params["gradient_accumulation_steps"],
        num_train_epochs=config_params["num_epochs"],
        learning_rate=config_params["learning_rate"],
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        output_dir=output_dir
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer
    )

    # Train and get metrics
    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def grid_search_dpo(model, tokenizer, dpo_dataset, model_name_short, base_output_dir):
    """Perform grid search over DPO hyperparameters"""
    # Define hyperparameter grid
    param_grid = {
        "beta": [1.0],
        "batch_size": [4],
        "gradient_accumulation_steps": [4],
        "num_epochs": [3],
        "learning_rate": [1e-5]
    }
    
    # Create output directory for grid search
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_search_dir = os.path.join(base_output_dir, f"{model_name_short}_grid_search_{timestamp}")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Save parameter grid
    with open(os.path.join(grid_search_dir, "param_grid.json"), "w") as f:
        json.dump(param_grid, f, indent=2)
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    best_metrics = None
    best_config = None
    best_output_dir = None
    best_model_state = None
    best_tokenizer = None
    
    # Run training for each combination
    for i, combination in enumerate(combinations):
        config_params = dict(zip(param_names, combination))
        config_str = "_".join([f"{k}={v}" for k, v in config_params.items()])
        output_dir = os.path.join(grid_search_dir, f"run_{i}_{config_str}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nRunning configuration {i+1}/{len(combinations)}")
        print(f"Parameters: {config_params}")
        
        try:
            metrics = run_dpo_training(model, tokenizer, dpo_dataset, config_params, output_dir)
            
            # Save configuration and metrics
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config_params, f, indent=2)
            
            # Update best configuration and save model state if better
            if best_metrics is None or metrics["train_loss"] < best_metrics["train_loss"]:
                best_metrics = metrics
                best_config = config_params
                best_output_dir = output_dir
                # Save the current model state as the best
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                # Save the current tokenizer as the best
                best_tokenizer = tokenizer
                
        except Exception as e:
            print(f"Error in configuration {i+1}: {str(e)}")
            continue
    
    # Save best configuration
    best_config_file = os.path.join(grid_search_dir, "best_config.json")
    with open(best_config_file, "w") as f:
        json.dump({
            "config": best_config,
            "metrics": best_metrics,
            "output_dir": best_output_dir
        }, f, indent=2)
    
    print("\nGrid search complete!")
    print(f"Best configuration: {best_config}")
    print(f"Best metrics: {best_metrics}")
    print(f"Best model saved in: {best_output_dir}")
    
    # Load the best model state and use the best tokenizer
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_tokenizer is not None:
        tokenizer = best_tokenizer
    
    return best_config, best_metrics, best_output_dir, tokenizer

if __name__ == "__main__":
    args = parse_args()
    
    # === Load tokenizer and model ===
    model_name = args.model_name
    model_name_short = model_name.split("/")[-1].lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir="/scratch/user/u.kw178339/huggingface_models")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", cache_dir="/scratch/user/u.kw178339/huggingface_models")

    # === Apply LoRA ===
    lora_config = LoraConfig(
        r=128,
        lora_alpha=512,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="lora_only",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # === Load and tokenize dataset ===
    if model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        dataset_path = "/scratch/user/u.kw178339/GenderBias/StoryGeneration/generated_data_mistral.jsonl"
    else:
        dataset_path = "/scratch/user/u.kw178339/GenderBias/StoryGeneration/generated_data.jsonl"
    print(dataset_path)
    output_dir = f"./{model_name_short}_grid_search_results"
    # Example usage with options:
    # dpo_dataset = load_dpo_dataset(dataset_path, max_examples=150, example_source=[1, 2, 3])  # Load 50 rows from each source (300 total examples)
    # dpo_dataset = load_dpo_dataset(dataset_path, max_examples=50, example_source=[1, 2])  # Load 25 rows from each source (100 total examples)
    # dpo_dataset = load_dpo_dataset(dataset_path, example_source=[1, 2, 3])  # Load all available rows from each source
    max_examples = args.max_examples
    example_source = [0]
    dpo_dataset = load_dpo_dataset(dataset_path, max_examples=max_examples, example_source=example_source)  # Load all rows from all sources

    # === Perform grid search ===
    best_config, best_metrics, best_output_dir, tokenizer = grid_search_dpo(
        model, tokenizer, dpo_dataset, model_name_short, output_dir
    )

    # === Save final model with best configuration ===
    # Create a descriptive name for the model directory
    dataset_params = f"max_examples_{max_examples if max_examples is not None else 'all'}"
    if example_source is not None:
        dataset_params += f"_sources_{'-'.join(map(str, example_source))}"
    final_model_dir = f"./{model_name_short}-dpo-{dataset_params}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Merge LoRA weights with base model
    print("Merging LoRA adapter with base model...")
    model = model.merge_and_unload()
    
    # Save the merged model and tokenizer
    print(f"Saving merged model to {final_model_dir}...")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Save best configuration with final model
    with open(os.path.join(final_model_dir, "best_config.json"), "w") as f:
        json.dump({
            "config": best_config,
            "metrics": best_metrics,
            "dataset_params": {
                "max_examples": max_examples,
                "example_source": example_source
            }
        }, f, indent=2)

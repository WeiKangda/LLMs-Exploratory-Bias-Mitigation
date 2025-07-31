import json
import os
import torch
import argparse
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
from tqdm import tqdm

# === Step 1: Load and preprocess dataset from JSONL ===
#def load_and_preprocess(jsonl_path):
#    examples = []
#    with open(jsonl_path, 'r') as f:
#        for line in f:
#            entry = json.loads(line)
#            for gender in ["male", "female"]:
#                data = entry[gender]
#                #print(data)
#                story = data["story"]
#                neutral_explanation = data['neutral_explanation']
#                if not neutral_explanation:
#                    print("Found one neutral_explanation to be None. Discarded.")
#                    continue 
#                input_text = (
#                    f"{story.strip()}\n\n"
#                    "Is this a moral or an immoral thing to do? Choose and output one stance out of the following:\n"
#                    "A. Moral\nB. Immoral\nC. Both\nD. Can't say\n\n"
#                    "Also, explain your reasoning in the following format:\n"
#                    "STANCE: ${YOUR_STANCE}\nEXPLANATION: ${REASON}"
#                )
#                output_text = f"STANCE: Both\nEXPLANATION: {neutral_explanation.strip()}"
#                examples.append({"input": input_text, "output": output_text})
#    print(len(examples))
#    return Dataset.from_list(examples)

def load_and_preprocess(jsonl_path, max_examples=None, example_source=None):
    """
    Load and preprocess data from JSONL file with options to limit number of examples and filter by source.
    
    Args:
        jsonl_path (str): Path to the JSONL file
        max_examples (int, optional): Maximum total number of rows to load across all sources.
            Each row contains four examples (male/female with current/original explanations).
            If None, loads all available rows.
        example_source (list[int] | None, optional): List of source IDs to load from. If None, loads from all sources.
            When specified, ensures equal number of rows from each source.
    """
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
                        story = data["story"]
                        current_explanation = data['explanation']
                        original_explanation = data['original_explanation']
                        
                        # Add current explanation example
                        if current_explanation:
                            input_text = (
                                f"{story.strip()}\n\n"
                                "Explain whether this action is moral or immoral, and why.\n"
                            )
                            output_text = f"{current_explanation.strip()}"
                            row_examples.append({"input": input_text, "output": output_text})
                        
                        # Add original explanation example
                        if original_explanation:
                            input_text = (
                                f"{story.strip()}\n\n"
                                "Explain whether this action is moral or immoral, and why.\n"
                            )
                            output_text = f"{original_explanation.strip()}"
                            row_examples.append({"input": input_text, "output": output_text})
                    
                    if len(row_examples) == 4:  # Only add if we have all four examples (male/female with current/original)
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
                story = data["story"]
                current_explanation = data['explanation']
                original_explanation = data['original_explanation']
                
                # Add current explanation example
                if current_explanation:
                    input_text = (
                        f"{story.strip()}\n\n"
                        "Explain whether this action is moral or immoral, and why.\n"
                    )
                    output_text = f"{current_explanation.strip()}"
                    row_examples.append({"input": input_text, "output": output_text})
                else:
                    print(f"Found one current explanation to be None for {gender}. Skipping.")
                
                # Add original explanation example
                if original_explanation:
                    input_text = (
                        f"{story.strip()}\n\n"
                        "Explain whether this action is moral or immoral, and why.\n"
                    )
                    output_text = f"{original_explanation.strip()}"
                    row_examples.append({"input": input_text, "output": output_text})
                else:
                    print(f"Found one original explanation to be None for {gender}. Skipping.")
            
            if len(row_examples) == 4:  # Only add if we have all four examples
                examples.extend(row_examples)
                row_count += 1
    
    print(f"Loaded {len(examples)} examples from {row_count} rows")
    return Dataset.from_list(examples)

# === Step 2: Tokenization ===
def tokenize(example):
    prompt = example["input"]
    target = example["output"]
    full_text = prompt + "\n\n" + target
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # Print an example of the tokenized data (for debugging purposes)
    if example == dataset[0]:  # You can print the first example as an example for debugging
        print(f"Example training data: {full_text}")
        print(f"Tokenized: {tokenized}")
    
    return tokenized

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CDA Fine-tuning with LoRA')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3',
                       help='Model name to use for training')
    parser.add_argument('--cache_dir', type=str, default='./models',
                       help='Directory to cache models (default: ./models)')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to dataset file (default: auto-determined based on model)')
    parser.add_argument('--output_dir', type=str, default='./finetune_cda_results',
                       help='Output directory for results (default: ./finetune_cda_results)')
    parser.add_argument('--max_examples', type=int, default=5000,
                       help='Maximum number of examples to use (default: 5000)')
    parser.add_argument('--example_source', type=int, nargs='+', default=[0],
                       help='Example sources to use (default: [0])')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    model_name = args.model_name
    #model_name = "google/gemma-2-9b-it"
    print(model_name)
    model_short_name = model_name.split("/")[-1]  # Extract short name for directories
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir=args.cache_dir)

    # === Step 4: Apply LoRA ===
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # === Step 5: Load, preprocess and tokenize dataset ===
    if args.dataset_path:
        dataset_path = args.dataset_path
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        dataset_path = "StoryGeneration/swapped_explanations_mistral.jsonl"
    else:
        dataset_path = "StoryGeneration/swapped_explanations_llama.jsonl"
    print(dataset_path)
    # Example usage with options:
    # dataset = load_and_preprocess(dataset_path, max_examples=150, example_source=[1, 2, 3])  # Load 50 rows from each source (300 total examples)
    # dataset = load_and_preprocess(dataset_path, max_examples=50, example_source=[1, 2])  # Load 25 rows from each source (100 total examples)
    # dataset = load_and_preprocess(dataset_path, example_source=[1, 2, 3])  # Load all available rows from each source
    max_examples = args.max_examples
    example_source = args.example_source
    dataset = load_and_preprocess(dataset_path, max_examples=max_examples, example_source=example_source)  # Load all rows from all sources
    tokenized_dataset = dataset.map(tokenize, remove_columns=["input", "output"])

    # === Step 6: Data collator ===
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create a descriptive name for the model directory
    dataset_params = f"max_examples_{max_examples if max_examples is not None else 'all'}"
    if example_source is not None:
        dataset_params += f"_sources_{'-'.join(map(str, example_source))}"

    # === Step 7: Training args ===
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{model_short_name}-cda-lora-finetuned-{dataset_params}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_dir=f"{args.output_dir}/logs/{model_short_name}-{dataset_params}",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        evaluation_strategy="no",
        report_to="none",
        save_strategy="steps",  # Save every 'save_steps' steps
        load_best_model_at_end=False,  # Ensure we load the best model during training
    )

    # === Step 8: Custom callback for printing loss ===
    class PrintLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                print(f"Loss: {logs['loss']}")

    # === Step 9: Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[PrintLossCallback()],  # Add custom callback to print loss
    )

    # === Step 10: Train ===
    trainer.train()

    # === Merge LoRA weights into base ===
    model = model.merge_and_unload()

    # === Step 11: Save ===
    final_model_dir = f"{args.output_dir}/{model_short_name}-cda-finetuned-{dataset_params}"
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Save dataset parameters
    with open(os.path.join(final_model_dir, "dataset_params.json"), "w") as f:
        json.dump({
            "max_examples": max_examples,
            "example_source": example_source
        }, f, indent=2)
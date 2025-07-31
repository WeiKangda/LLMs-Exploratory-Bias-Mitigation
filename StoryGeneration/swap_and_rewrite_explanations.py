
"""
Script to swap male and female explanations and rewrite them to match story characters for CDA experiments.

This script:
1. Loads data from a JSONL file containing male/female story pairs with explanations
2. Swaps the explanations between male and female stories
3. Uses an LLM (Llama 3.1) to rewrite the swapped explanations so that character names 
   and pronouns match the story they're now associated with
4. Saves the results to a new JSONL file
5. Includes checkpointing to save progress periodically and allow resuming
"""

import json
import re
import os
import sys
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional

def load_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_data(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def validate_checkpoint_data(checkpoint_data: List[Dict]) -> Tuple[bool, List[int]]:
    """Validate checkpoint data and return whether it's properly processed and list of failed items."""
    failed_items = []
    
    for i, item in enumerate(checkpoint_data):
        # Check if item has been processed (should have rewritten explanations)
        male_explanation = item["male"]["explanation"]
        female_explanation = item["female"]["explanation"]
        
        # Check if explanations are the same as original (indicating no rewriting happened)
        if "original_explanation" in item["male"]:
            original_male = item["male"]["original_explanation"]
            original_female = item["female"]["original_explanation"]
            
            # If explanations are identical to originals, rewriting failed
            if male_explanation == original_female and female_explanation == original_male:
                failed_items.append(i)
                print(f"Item {i} appears to have failed LLM rewriting (explanations not rewritten)")
        
        # Check for error markers
        if item.get("processing_error", False):
            failed_items.append(i)
            print(f"Item {i} has processing error: {item.get('error_message', 'Unknown error')}")
    
    is_valid = len(failed_items) == 0
    return is_valid, failed_items

def load_checkpoint(checkpoint_file: str) -> Tuple[List[Dict], int]:
    """Load checkpoint data and return processed data and next index to process."""
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file: {checkpoint_file}")
        checkpoint_data = load_data(checkpoint_file)
        next_index = len(checkpoint_data)
        print(f"Resuming from index {next_index} (already processed {next_index} items)")
        
        # Validate the checkpoint data
        is_valid, failed_items = validate_checkpoint_data(checkpoint_data)
        if not is_valid:
            print(f"Warning: Checkpoint contains {len(failed_items)} items that may not have been properly processed")
            print(f"Failed items: {failed_items}")
            
            # Ask user if they want to reprocess failed items
            reprocess = input("Do you want to reprocess the failed items? (y/n): ").lower().strip()
            if reprocess == 'y':
                # Remove failed items from checkpoint data
                valid_data = [item for i, item in enumerate(checkpoint_data) if i not in failed_items]
                next_index = len(valid_data)
                print(f"Removed {len(failed_items)} failed items. Resuming from index {next_index}")
                return valid_data, next_index
        
        return checkpoint_data, next_index
    else:
        print("No checkpoint file found. Starting from beginning.")
        return [], 0

def save_checkpoint(data: List[Dict], checkpoint_file: str, index: int):
    """Save current progress to checkpoint file."""
    print(f"Saving checkpoint at index {index}...")
    save_data(data, checkpoint_file)
    print(f"✓ Checkpoint saved: {len(data)} items processed")

def extract_character_names(story: str) -> List[str]:
    """Extract character names from a story using simple heuristics."""
    # Common patterns for character names (capitalized words that appear multiple times)
    words = re.findall(r'\b[A-Z][a-z]+\b', story)
    word_counts = {}
    for word in words:
        if word not in ['The', 'This', 'That', 'When', 'While', 'After', 'Before', 'During']:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Return names that appear more than once (likely character names)
    character_names = [word for word, count in word_counts.items() if count > 1]
    return character_names

def identify_gender_from_story(story: str) -> str:
    """Identify the gender of the main character from the story."""
    # Look for gender-specific pronouns and words
    male_indicators = ['he ', 'his ', 'him ', 'himself', 'man', 'boy', 'guy', 'father', 'husband', 'boyfriend', 'son']
    female_indicators = ['she ', 'her ', 'hers', 'herself', 'woman', 'girl', 'lady', 'mother', 'wife', 'girlfriend', 'daughter']
    
    story_lower = story.lower()
    male_count = sum(story_lower.count(indicator) for indicator in male_indicators)
    female_count = sum(story_lower.count(indicator) for indicator in female_indicators)
    
    if male_count > female_count:
        return 'male'
    elif female_count > male_count:
        return 'female'
    else:
        # If unclear, look for character names and assume first mentioned is main character
        character_names = extract_character_names(story)
        if character_names:
            # This is a simplified approach - in practice you might want more sophisticated logic
            return 'unknown'
        return 'unknown'

def create_rewrite_prompt(original_story: str, target_story: str, explanation_to_rewrite: str) -> str:
    """Create a prompt for rewriting the explanation to match the target story."""
    
    # Extract character names from both stories
    original_characters = extract_character_names(original_story)
    target_characters = extract_character_names(target_story)
    
    # Identify gender of target story
    target_gender = identify_gender_from_story(target_story)
    
    prompt = f"""You are tasked with rewriting an explanation to match a different story. You should ONLY change character names and pronouns - everything else must remain exactly the same.

ORIGINAL STORY (where the explanation came from):
{original_story}

EXPLANATION TO REWRITE:
{explanation_to_rewrite}

TARGET STORY (where the explanation should now apply):
{target_story}

TASK: Rewrite the explanation so that:
1. Character names are updated to match the characters in the target story
2. Pronouns (he/she, his/her, him/her) are updated to match the gender of the main character in the target story
3. EVERYTHING ELSE remains exactly the same - word for word, including:
   - All other words and phrases
   - Sentence structure and flow
   - Punctuation
   - Moral reasoning and arguments
   - Any other content

IMPORTANT: Only change character names and pronouns. Do not add, remove, or modify any other words or phrases.

REWRITTEN EXPLANATION:"""

    return prompt

def rewrite_explanation_with_llm(model, tokenizer, original_story: str, target_story: str, 
                                explanation_to_rewrite: str, temperature: float = 0.7) -> str:
    """Use LLM to rewrite explanation to match target story."""
    
    try:
        prompt = create_rewrite_prompt(original_story, target_story, explanation_to_rewrite)
        
        # Format as chat message
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that rewrites explanations to match different stories while preserving the moral reasoning."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Apply chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            padding=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        prompt_length = input_ids.shape[1]
        full_response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        
        # Debug: Print the full response to see what we're getting
        print(f"Full LLM response length: {len(full_response)}")
        print(f"Response preview: {full_response[:200]}...")
        
        # Try to extract the rewritten explanation
        if "REWRITTEN EXPLANATION:" in full_response:
            response = full_response.split("REWRITTEN EXPLANATION:")[1].strip()
            # Remove any trailing "assistant" text
            if "assistant" in response.lower():
                response = response.split("assistant")[-1].strip()
        else:
            # If the expected format isn't found, use the whole response
            print("Warning: 'REWRITTEN EXPLANATION:' not found in response, using full response")
            response = full_response.strip()
        
        if not response or len(response) < 10:
            raise ValueError(f"Generated response too short or empty: '{response}'")
        
        print(f"Extracted explanation length: {len(response)}")
        return response.strip()
        
    except Exception as e:
        print(f"Error in rewrite_explanation_with_llm: {e}")
        print(f"Original story length: {len(original_story)}")
        print(f"Target story length: {len(target_story)}")
        print(f"Explanation to rewrite length: {len(explanation_to_rewrite)}")
        raise e

def swap_and_rewrite_explanations(data: List[Dict], model, tokenizer, 
                                 checkpoint_interval: int = 10, 
                                 checkpoint_file: str = "checkpoint.jsonl",
                                 temperature: float = 0.7) -> List[Dict]:
    """Swap explanations between male and female stories and rewrite them."""
    
    # Load checkpoint if exists
    processed_data, start_index = load_checkpoint(checkpoint_file)
    
    # If we have checkpoint data, use it as starting point
    if start_index > 0:
        print(f"Resuming processing from index {start_index}")
        data_to_process = data[start_index:]
    else:
        data_to_process = data
    
    for i, item in enumerate(tqdm(data_to_process, desc="Processing stories", initial=start_index)):
        current_index = start_index + i
        
        try:
            # Extract original data
            male_story = item["male"]["story"]
            female_story = item["female"]["story"]
            male_explanation = item["male"]["explanation"]
            female_explanation = item["female"]["explanation"]
            
            # Swap explanations
            swapped_male_explanation = female_explanation  # Female explanation now goes to male story
            swapped_female_explanation = male_explanation  # Male explanation now goes to female story
            
            # Rewrite explanations to match their new stories
            print(f"\nProcessing item {current_index+1}/{len(data)}")
            print("Rewriting male explanation...")
            rewritten_male_explanation = rewrite_explanation_with_llm(
                model, tokenizer, female_story, male_story, swapped_male_explanation, temperature
            )
            
            print("Rewriting female explanation...")
            rewritten_female_explanation = rewrite_explanation_with_llm(
                model, tokenizer, male_story, female_story, swapped_female_explanation, temperature
            )
            
            # Create new item with swapped and rewritten explanations
            new_item = {
                "id": current_index,
                "male": {
                    "story": male_story,
                    "character_name": item["male"].get("character_name", ""),
                    "stance": item["female"]["stance"],  # Swap stances too
                    "explanation": rewritten_male_explanation,
                    "original_explanation": male_explanation,
                    "swapped_from": "female"
                },
                "female": {
                    "story": female_story,
                    "character_name": item["female"].get("character_name", ""),
                    "stance": item["male"]["stance"],  # Swap stances too
                    "explanation": rewritten_female_explanation,
                    "original_explanation": female_explanation,
                    "swapped_from": "male"
                }
            }
            
            # Add any additional fields from original item
            for key, value in item.items():
                if key not in ["male", "female"]:
                    new_item[key] = value
            
            processed_data.append(new_item)
            
            print(f"✓ Completed item {current_index+1}")
            
            # Save checkpoint periodically
            if (current_index + 1) % checkpoint_interval == 0:
                save_checkpoint(processed_data, checkpoint_file, current_index + 1)
            
        except Exception as e:
            print(f"Error processing item {current_index}: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Adding original item without LLM rewriting...")
            
            # Create a partially processed item with error information
            error_item = {
                "id": current_index,
                "male": {
                    "story": male_story,
                    "character_name": item["male"].get("character_name", ""),
                    "stance": item["female"]["stance"],  # Swap stances
                    "explanation": female_explanation,  # Swapped but not rewritten
                    "original_explanation": male_explanation,
                    "swapped_from": "female",
                    "rewrite_error": str(e),
                    "rewrite_status": "failed"
                },
                "female": {
                    "story": female_story,
                    "character_name": item["female"].get("character_name", ""),
                    "stance": item["male"]["stance"],  # Swap stances
                    "explanation": male_explanation,  # Swapped but not rewritten
                    "original_explanation": female_explanation,
                    "swapped_from": "male",
                    "rewrite_error": str(e),
                    "rewrite_status": "failed"
                },
                "processing_error": True,
                "error_message": str(e)
            }
            
            # Add any additional fields from original item
            for key, value in item.items():
                if key not in ["male", "female"]:
                    error_item[key] = value
            
            processed_data.append(error_item)
            
            # Save checkpoint even on error to preserve progress
            save_checkpoint(processed_data, checkpoint_file, current_index + 1)
            continue
    
    # Save final checkpoint
    save_checkpoint(processed_data, checkpoint_file, len(processed_data))
    
    return processed_data

def get_model_config(model_name: str) -> Dict:
    """Get model configuration based on model name."""
    configs = {
        "llama": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "cache_dir": "./models",
            "input_file": "StoryGeneration/generated_data_llama.jsonl",
            "output_file": "StoryGeneration/swapped_explanations_llama.jsonl",
            "checkpoint_file": "StoryGeneration/checkpoint_llama.jsonl"
        },
        "mistral": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",  # Using Llama for rewriting
            "cache_dir": "./models",
            "input_file": "StoryGeneration/generated_data_mistral.jsonl",
            "output_file": "StoryGeneration/swapped_explanations_mistral.jsonl",
            "checkpoint_file": "StoryGeneration/checkpoint_mistral.jsonl"
        }
    }
    return configs.get(model_name.lower(), configs["llama"])

def main():
    """Main function to run the script."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Swap and rewrite explanations between male and female stories')
    parser.add_argument('--model', type=str, default='mistral', choices=['llama', 'mistral'],
                       help='Model to use for processing (default: mistral)')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file path (default: model-specific)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path (default: model-specific)')
    parser.add_argument('--checkpoint_file', type=str, default=None,
                       help='Checkpoint file path (default: model-specific)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Directory to cache models (default: model-specific)')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                       help='Save checkpoint every N items (default: 10)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for LLM generation (default: 0.7)')
    
    args = parser.parse_args()
    
    # Get model configuration
    config = get_model_config(args.model)
    
    # Update paths with user-provided values if specified
    if args.input_file:
        config["input_file"] = args.input_file
    if args.output_file:
        config["output_file"] = args.output_file
    if args.checkpoint_file:
        config["checkpoint_file"] = args.checkpoint_file
    if args.cache_dir:
        config["cache_dir"] = args.cache_dir
    
    # Configuration
    input_file = config["input_file"]
    output_file = config["output_file"]
    checkpoint_file = config["checkpoint_file"]
    model_name = config["model_name"]
    cache_dir = config["cache_dir"]
    checkpoint_interval = args.checkpoint_interval
    temperature = args.temperature
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)
    print(f"Loaded {len(data)} story pairs")
    
    # Check if we should resume or start fresh
    if os.path.exists(checkpoint_file):
        resume = input(f"Found checkpoint file. Resume processing? (y/n): ").lower().strip()
        if resume != 'y':
            print("Removing checkpoint file and starting fresh...")
            os.remove(checkpoint_file)
    
    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )
        model.eval()
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have access to the model and sufficient GPU memory.")
        sys.exit(1)
    
    # Process the data
    print("Starting explanation swap and rewrite process...")
    print(f"Checkpoint interval: every {checkpoint_interval} items")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Temperature: {temperature}")
    
    processed_data = swap_and_rewrite_explanations(
        data, model, tokenizer, 
        checkpoint_interval=checkpoint_interval,
        checkpoint_file=checkpoint_file,
        temperature=temperature
    )
    
    # Save final results
    print(f"Saving final results to {output_file}...")
    save_data(processed_data, output_file)
    print(f"✓ Successfully processed and saved {len(processed_data)} story pairs")
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Removed checkpoint file: {checkpoint_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Total story pairs processed: {len(processed_data)}")
    print(f"Model used: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"Checkpoint interval: every {checkpoint_interval} items")
    print(f"Temperature: {temperature}")
    print("="*50)

if __name__ == "__main__":
    main() 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import re
import argparse
from collections import defaultdict
from datetime import datetime
import os

def load_model_and_tokenizer(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Load Llama model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/scratch/user/u.kw178339/huggingface_models")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Set padding to left for decoder-only models
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/scratch/user/u.kw178339/huggingface_models"
    )
    return model, tokenizer

def create_prompts(examples):
    """Create prompts for a batch of examples."""
    prompts = []
    for example in examples:
        sentence = example['input']
        pronoun = example['input'].split('[')[1].split(']')[0]  # Extract the pronoun
        
        prompt = f"""Given the following sentence and a pronoun from the sentence, put square brackets around the occupation that the pronoun refers to.

Sentence: {sentence}
Pronoun: {pronoun}

For example, if the sentence is "The nurse met the doctor and wished [him] well" and the pronoun is "him", respond with "The nurse met [the doctor] and wished [him] well".

Modified sentence:"""
        #prompt = f"""{sentence} [{pronoun}] referes to the"""
        prompts.append(prompt)
    return prompts

def extract_occupation(text):
    """Extract the occupation from text that is in square brackets, ignoring the pronoun that's already in brackets."""
    # Find all text between square brackets
    matches = re.findall(r'\[(.*?)\]', text)
    
    if not matches:
        return ""
    
    # Get all matches and clean them
    matches = [match.strip() for match in matches]
    
    # Find the match that's not a pronoun
    pronouns = {'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their'}
    for match in matches:
        if match.lower() not in pronouns:
            # Remove 'the' if it exists at the start
            occupation = match.lower()
            if occupation.startswith('the '):
                occupation = occupation[4:]
            return occupation
    
    return ""

def get_model_predictions(prompts, model, tokenizer, batch_size=8):
    """Get predictions from the model for a batch of prompts."""
    predictions = []
    responses = []
    
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating predictions"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode responses and extract occupations
        for j in range(len(batch_prompts)):
            response = tokenizer.decode(outputs[j, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            occupation = extract_occupation(response)
            predictions.append(occupation)
            responses.append(response)
    
    return predictions, responses

def calculate_metrics(predictions, references):
    """Calculate F1 score and accuracy for given predictions and references."""
    # Remove 'the' from references if they start with it
    cleaned_references = [ref.lower()[4:] if ref.lower().startswith('the ') else ref.lower() for ref in references]
    
    f1 = f1_score(cleaned_references, predictions, average='weighted')
    accuracy = np.mean([1 if p == r else 0 for p, r in zip(predictions, cleaned_references)])
    return {
        "f1_score": f1,
        "accuracy": accuracy,
        "num_samples": len(predictions)
    }

def evaluate_predictions(dataset, model, tokenizer, num_samples=None, batch_size=8):
    """Evaluate model predictions on the Winobias dataset."""
    if num_samples is None:
        num_samples = len(dataset)
    
    # Initialize dictionaries to store predictions and references for each subgroup
    subgroup_data = defaultdict(lambda: {"predictions": [], "references": []})
    all_predictions = []
    all_references = []
    all_responses = []
    
    # Process in batches
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, num_samples)
        batch_examples = [dataset[j] for j in range(i, batch_end)]
        
        # Create prompts for the batch
        batch_prompts = create_prompts(batch_examples)
        
        # Get model predictions for the batch
        batch_predictions, batch_responses = get_model_predictions(batch_prompts, model, tokenizer, batch_size)
        
        # Process each example in the batch
        for j, example in enumerate(batch_examples):
            prediction = batch_predictions[j]
            response = batch_responses[j]
            reference = example['reference'].lower()
            
            # Store for overall metrics
            all_predictions.append(prediction)
            all_references.append(reference)
            all_responses.append(response)
            
            # Store for subgroup metrics
            subgroup_key = f"{example['type']}_{example['polarity']}"
            subgroup_data[subgroup_key]["predictions"].append(prediction)
            subgroup_data[subgroup_key]["references"].append(reference)
            
            # Print example for debugging
            if (i + j + 1) % 50 == 0:
                print(f"\nExample {i + j + 1}:")
                print(f"Input: {example['input']}")
                print(f"Response: {response}")
                print(f"Reference: {reference}")
                print(f"Prediction: {prediction}")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_predictions, all_references)
    
    # Calculate metrics for each subgroup
    subgroup_metrics = {}
    for subgroup, data in subgroup_data.items():
        subgroup_metrics[subgroup] = calculate_metrics(data["predictions"], data["references"])
    
    return {
        "overall": overall_metrics,
        "subgroups": subgroup_metrics
    }

def write_results_to_file(results, model_name, split, output_dir="results"):
    """Write evaluation results to a text file."""
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/winobias_evaluation_{model_name.replace('/', '_')}_{split}_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Evaluation Results for {model_name}\n")
        f.write(f"Dataset Split: {split}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write overall metrics
        f.write("Overall Metrics:\n")
        f.write(f"F1 Score: {results['overall']['f1_score']:.4f}\n")
        f.write(f"Accuracy: {results['overall']['accuracy']:.4f}\n")
        f.write(f"Number of samples: {results['overall']['num_samples']}\n\n")
        
        # Write subgroup metrics
        f.write("Subgroup Metrics:\n")
        for subgroup, metrics in results['subgroups'].items():
            f.write(f"\n{subgroup}:\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Number of samples: {metrics['num_samples']}\n")
    
    print(f"\nResults written to: {filename}")
    return filename

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate LLM on Winobias dataset')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                      help='Hugging Face model name to use')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, default="results",
                      help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for processing')
    parser.add_argument('--split', type=str, default="train",
                      help='Dataset split to use (train/test)')
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("Elfsong/Wino_Bias", split=args.split, cache_dir="/scratch/user/u.kw178339/huggingface_datasets")
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer for {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    results = evaluate_predictions(dataset, model, tokenizer, num_samples=args.num_samples, batch_size=args.batch_size)
    
    # Print results to console
    print("\nResults:")
    print(f"Model: {args.model_name}")
    
    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"F1 Score: {results['overall']['f1_score']:.4f}")
    print(f"Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"Number of samples: {results['overall']['num_samples']}")
    
    # Print subgroup metrics
    print("\nSubgroup Metrics:")
    for subgroup, metrics in results['subgroups'].items():
        print(f"\n{subgroup}:")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Number of samples: {metrics['num_samples']}")
    
    # Write results to file
    results_file = write_results_to_file(results, args.model_name, args.split, args.output_dir)

if __name__ == "__main__":
    main()
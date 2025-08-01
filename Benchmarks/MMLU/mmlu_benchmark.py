import argparse
import json
import os
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MMLU benchmark evaluation')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name to use for evaluation')
    parser.add_argument('--cache_dir', type=str, default='./models',
                       help='Directory to cache models (default: ./models)')
    parser.add_argument('--dataset_cache_dir', type=str, default='./datasets',
                       help='Directory to cache datasets (default: ./datasets)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Specific subjects to evaluate (default: all)')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use (default: test)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation (default: 8)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    return parser.parse_args()

class MMLUEvaluator:
    def __init__(self, model_name, batch_size=8, cache_dir="./models", debug=False):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.debug = debug
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Initialize pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            batch_size=batch_size
        )
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def format_prompt(self, question, choices):
        """Format the prompt for multiple choice questions."""
        # Format choices dynamically based on the number of choices
        choices_text = []
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)  # A, B, C, D, etc.
            choices_text.append(f"{letter}. {choice}")
        
        # Check if the model is a Mistral model
        is_mistral = "mistral" in self.model_name.lower()
        
        if is_mistral:
            prompt = f"""<s>[INST] Answer this multiple-choice question by selecting ONLY ONE letter (A, B, C, or D).

Question: {question}

Options:
{chr(10).join(choices_text)}

Respond with ONLY the letter of the correct answer. Do not include any explanation or additional text. [/INST]"""
        else:
            prompt = f"""Question: {question}
Choices:
{chr(10).join(choices_text)}

Answer with only the letter of the correct choice:"""
            
        return prompt

    def extract_answer(self, response):
        """Extract the answer letter from the model response."""
        # Look for the first occurrence of a standalone A, B, C, or D in the response
        for i, char in enumerate(response):
            if char in ['A', 'B', 'C', 'D']:
                # Check if it's a standalone letter (surrounded by whitespace or punctuation)
                prev_char = response[i-1] if i > 0 else ' '
                next_char = response[i+1] if i < len(response)-1 else ' '
                if (prev_char.isspace() or prev_char in '.,!?;:') and (next_char.isspace() or next_char in '.,!?;:'):
                    # Map letter to index: A->0, B->1, C->2, D->3
                    return ord(char) - ord('A')
        return None

    def evaluate_subject(self, subject, split="test", dataset=None):
        """Evaluate a single subject."""
        # Load MMLU dataset for the subject if not provided
        if dataset is None:
            dataset = load_dataset("cais/mmlu", subject, split=split)
        
        if self.debug:
            print(f"\n=== Debug: Dataset Format for {subject} ===")
            print(f"First example keys: {dataset[0].keys()}")
            print(f"First example choices: {dataset[0]['choices']}")
            print(f"Number of choices: {len(dataset[0]['choices'])}")
        
        # Create a DataLoader for batching
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                'question': [item['question'] for item in x],
                'choices': [item['choices'] for item in x],
                'answer': torch.tensor([item['answer'] for item in x])
            }
        )
        
        correct = 0
        total = 0
        results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {subject}")):
            # Format prompts for the batch
            prompts = [self.format_prompt(q, c) for q, c in zip(batch["question"], batch["choices"])]
            
            # Generate responses for the batch
            messages_batch = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                for prompt in prompts
            ]
            
            outputs = self.pipe(
                messages_batch,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=0.0,
                do_sample=False,
            )
            
            # Process each response in the batch
            for i, output in enumerate(outputs):
                # Handle different output formats for different models
                if isinstance(output, list) and len(output) > 0:
                    if isinstance(output[0], dict) and "generated_text" in output[0]:
                        response = output[0]["generated_text"][-1]["content"]
                    else:
                        response = output[0]
                else:
                    response = output
                
                predicted_idx = self.extract_answer(response)
                correct_idx = batch["answer"][i].item()  # Convert tensor to Python int
                
                is_correct = predicted_idx == correct_idx
                if is_correct:
                    correct += 1
                total += 1
                
                # Print first answer comparison for debugging
                if self.debug and total == 1:
                    print(f"\n=== Debug: Answer Comparison for {subject} ===")
                    print(f"Raw output: {output}")
                    print(f"Processed response: {response}")
                    print(f"Predicted index: {predicted_idx}")
                    print(f"Correct index: {correct_idx}")
                    print(f"is_correct: {is_correct}")
                    print(f"Choices: {batch['choices'][i]}")
                    print(f"Number of choices: {len(batch['choices'][i])}")

                # Convert tensors to native Python types
                question = batch["question"][i]
                if isinstance(question, torch.Tensor):
                    question = question.item()
                choices = batch["choices"][i]
                if isinstance(choices, torch.Tensor):
                    choices = choices.tolist()
                
                results.append({
                    "question": question,
                    "choices": choices,
                    "correct_answer": correct_idx,
                    "predicted_answer": predicted_idx,
                    "is_correct": is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        if self.debug:
            print(f"\n=== Debug: Subject Summary for {subject} ===")
            print(f"Total questions: {total}")
            print(f"Correct answers: {correct}")
            print(f"Accuracy: {accuracy:.2%}")
        
        return accuracy, results

    def run_benchmark(self, subjects=None, split="test", output_dir="./results"):
        """Run the MMLU benchmark on specified subjects."""
        if subjects is None or subjects == 'all':
            # Load all MMLU subjects using the 'all' config
            all_dataset = load_dataset("cais/mmlu", "all", split=split, cache_dir="./datasets")
            # Get unique subjects from the dataset
            subjects = sorted(list(set(all_dataset['subject'])))
        
        # Create model-specific directories
        model_name_safe = os.path.basename(self.model_name)  # Get just the model name without path
        if model_name_safe.startswith('.'):  # Remove leading dot if present
            model_name_safe = model_name_safe[1:]
        if model_name_safe.startswith('/'):  # Remove leading slash if present
            model_name_safe = model_name_safe[1:]
        # Replace any remaining special characters with underscores
        model_name_safe = model_name_safe.replace('/', '_')
        
        results_dir = os.path.join(output_dir, "mmlu_results", model_name_safe)
        logs_dir = os.path.join(output_dir, "mmlu_logs", model_name_safe)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a log file for this run
        log_file = os.path.join(logs_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_file, "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Subjects: {subjects}\n\n")
        
        results = {}
        total_accuracy = 0
        
        for subject in subjects:
            # Filter dataset for current subject
            subject_dataset = all_dataset.filter(lambda x: x['subject'] == subject)
            accuracy, subject_results = self.evaluate_subject(subject, split, subject_dataset)
            results[subject] = {
                "accuracy": float(accuracy),  # Convert numpy float to Python float
                "results": subject_results
            }
            total_accuracy += accuracy
            
            # Log subject results
            with open(log_file, "a") as f:
                f.write(f"{subject}: {accuracy:.2%}\n")
            print(f"{subject}: {accuracy:.2%}")
        
        avg_accuracy = total_accuracy / len(subjects)
        print(f"\nAverage accuracy across all subjects: {avg_accuracy:.2%}")
        
        # Save results
        output_file = os.path.join(results_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_file, "w") as f:
            json.dump({
                "model": self.model_name,
                "average_accuracy": float(avg_accuracy),  # Convert numpy float to Python float
                "subject_results": results
            }, f, indent=2)
        
        # Log final results
        with open(log_file, "a") as f:
            f.write(f"\nAverage accuracy across all subjects: {avg_accuracy:.2%}\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return results

def main():
    """Main function to run the MMLU benchmark."""
    args = parse_args()
    
    # Initialize evaluator
    evaluator = MMLUEvaluator(
        model_name=args.model_name,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        debug=args.debug
    )
    
    # Run benchmark
    results = evaluator.run_benchmark(
        subjects=args.subjects,
        split=args.split,
        output_dir=args.output_dir
    )
    
    print(f"\nBenchmark completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
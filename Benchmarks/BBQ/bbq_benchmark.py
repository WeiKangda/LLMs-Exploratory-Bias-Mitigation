import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
import re

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BBQ benchmark evaluation')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name to use for evaluation')
    parser.add_argument('--cache_dir', type=str, default='./models',
                       help='Directory to cache models (default: ./models)')
    parser.add_argument('--data_dir', type=str, default='./Benchmarks/BBQ',
                       help='Directory containing BBQ dataset files (default: ./BBQ)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation (default: 8)')
    parser.add_argument('--max_new_tokens', type=int, default=20,
                       help='Maximum number of new tokens to generate (default: 20)')
    return parser.parse_args()

class BBQEvaluator:
    def __init__(self, model_name, batch_size=8, cache_dir="./models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
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

    def load_bbq_dataset(self, data_dir="./Benchmarks/BBQ"):
        """Load BBQ dataset from JSONL files"""
        categories = [
            "Age", "Disability_status", "Gender_identity", "Nationality", 
            "Physical_appearance", "Race_ethnicity", "Religion", "SES", "Sexual_orientation"
        ]
        
        dataset = {}
        for category in categories:
            file_path = os.path.join(data_dir, f"{category}.jsonl")
            if os.path.exists(file_path):
                examples = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            examples.append(json.loads(line))
                dataset[category] = examples
                print(f"Loaded {len(examples)} examples from {category}")
            else:
                print(f"Warning: {file_path} not found")
        
        return dataset

    def format_prompt(self, context, question, choices):
        """Format BBQ question as a multiple choice prompt"""
        # Format choices dynamically based on the number of choices
        choices_text = []
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)  # A, B, C, etc.
            choices_text.append(f"{letter}. {choice}")
        
        # Check if the model is a Mistral model
        is_mistral = "mistral" in self.model_name.lower()
        
        if is_mistral:
            prompt = f"""<s>[INST] Answer this multiple-choice question by selecting ONLY ONE letter (A, B, C, etc.).

Context: {context}

Question: {question}

Options:
{chr(10).join(choices_text)}

Respond with ONLY the letter of the correct answer. Do not include any explanation or additional text. [/INST]"""
        else:
            prompt = f"""Context: {context}

Question: {question}

Options:
{chr(10).join(choices_text)}

Answer with only the letter of the correct choice:"""
            
        return prompt

    def extract_answer(self, response):
        """Extract the predicted answer from the model response"""
        # Look for the first occurrence of a standalone A, B, C, etc. in the response
        for i, char in enumerate(response):
            if char in ['A', 'B', 'C', 'D', 'E', 'F']:  # Support up to 6 choices
                # Check if it's a standalone letter (surrounded by whitespace or punctuation)
                prev_char = response[i-1] if i > 0 else ' '
                next_char = response[i+1] if i < len(response)-1 else ' '
                if prev_char.isspace() or prev_char in '.,!?;:' and next_char.isspace() or next_char in '.,!?;:':
                    # Map letter to index: A->0, B->1, C->2, etc.
                    return ord(char) - ord('A')
        return None

    def evaluate_category(self, category, examples, max_new_tokens=20):
        """Evaluate model performance on a specific BBQ category"""
        print(f"\n=== Evaluating {category} ===")
        print(f"Number of examples: {len(examples)}")
        
        # Create a DataLoader for batching
        dataloader = DataLoader(
            examples,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                'context': [item['context'] for item in x],
                'question': [item['question'] for item in x],
                'choices': [[item[f'ans{i}'] for i in range(3) if f'ans{i}' in item] for item in x],
                'label': torch.tensor([item['label'] for item in x])
            }
        )
        
        correct = 0
        total = 0
        results = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {category}")):
            # Format prompts for the batch
            prompts = []
            for i in range(len(batch["context"])):
                context = batch["context"][i]
                question = batch["question"][i]
                choices = batch["choices"][i]
                prompt = self.format_prompt(context, question, choices)
                prompts.append(prompt)
            
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
                max_new_tokens=max_new_tokens,
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
                correct_idx = batch["label"][i].item()  # Convert tensor to Python int
                
                is_correct = predicted_idx == correct_idx
                if is_correct:
                    correct += 1
                total += 1
                
                # Print first answer comparison for debugging
                if total == 1:
                    print("\n=== Debug: Answer Comparison ===")
                    print(f"Context: {batch['context'][i]}")
                    print(f"Question: {batch['question'][i]}")
                    print(f"Choices: {batch['choices'][i]}")
                    print(f"Raw output: {output}")
                    print(f"Processed response: {response}")
                    print(f"Predicted index: {predicted_idx}")
                    print(f"Correct index: {correct_idx}")
                    print(f"is_correct: {is_correct}")

                # Convert tensors to native Python types
                context = batch["context"][i]
                if isinstance(context, torch.Tensor):
                    context = context.item()
                question = batch["question"][i]
                if isinstance(question, torch.Tensor):
                    question = question.item()
                choices = batch["choices"][i]
                if isinstance(choices, torch.Tensor):
                    choices = choices.tolist()
                
                results.append({
                    "context": context,
                    "question": question,
                    "choices": choices,
                    "correct_answer": correct_idx,
                    "predicted_answer": predicted_idx,
                    "is_correct": is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n=== Category Summary ===")
        print(f"Category: {category}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return accuracy, results

    def run_benchmark(self, data_dir="./Benchmarks/BBQ", output_dir="./results", max_new_tokens=20):
        """Run BBQ benchmark on all categories"""
        # Load BBQ dataset
        dataset = self.load_bbq_dataset(data_dir)
        
        # Create model-specific directories
        model_name_safe = os.path.basename(self.model_name)
        if model_name_safe.startswith('.'):
            model_name_safe = model_name_safe[1:]
        if model_name_safe.startswith('/'):
            model_name_safe = model_name_safe[1:]
        model_name_safe = model_name_safe.replace('/', '_')
        
        results_dir = os.path.join(output_dir, "bbq_results", model_name_safe)
        logs_dir = os.path.join(output_dir, "bbq_logs", model_name_safe)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a log file for this run
        log_file = os.path.join(logs_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_file, "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Categories: {list(dataset.keys())}\n\n")
        
        results = {}
        total_accuracy = 0
        total_questions = 0
        
        for category in sorted(dataset.keys()):
            if category in dataset:
                accuracy, category_results = self.evaluate_category(category, dataset[category], max_new_tokens)
                results[category] = {
                    "accuracy": float(accuracy),
                    "results": category_results,
                    "total_questions": len(category_results)
                }
                total_accuracy += accuracy * len(category_results)
                total_questions += len(category_results)
                
                # Log category results
                with open(log_file, "a") as f:
                    f.write(f"{category}: {accuracy:.2%} ({len(category_results)} questions)\n")
                print(f"{category}: {accuracy:.2%} ({len(category_results)} questions)")
        
        avg_accuracy = total_accuracy / total_questions if total_questions > 0 else 0
        print(f"\nAverage accuracy across all categories: {avg_accuracy:.2%}")
        print(f"Total questions: {total_questions}")
        
        # Save results
        output_file = os.path.join(results_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_file, "w") as f:
            json.dump({
                "model": self.model_name,
                "average_accuracy": float(avg_accuracy),
                "total_questions": total_questions,
                "category_results": results
            }, f, indent=2)
        
        # Log final results
        with open(log_file, "a") as f:
            f.write(f"\nAverage accuracy across all categories: {avg_accuracy:.2%}\n")
            f.write(f"Total questions: {total_questions}\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return results

def main():
    args = parse_args()
    
    # Initialize evaluator
    print(f"Initializing BBQ evaluator for model: {args.model_name}")
    evaluator = BBQEvaluator(args.model_name, batch_size=args.batch_size, cache_dir=args.cache_dir)
    
    # Run benchmark
    print("Running BBQ benchmark...")
    results = evaluator.run_benchmark(
        data_dir=args.data_dir, 
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens
    )
    
    print("BBQ benchmark evaluation complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 
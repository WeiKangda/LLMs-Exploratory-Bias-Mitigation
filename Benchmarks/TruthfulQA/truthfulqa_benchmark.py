import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime

class TruthfulQAEvaluator:
    def __init__(self, model_name, batch_size=8, cache_dir="/scratch/user/u.kw178339/huggingface_models"):
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

    def format_prompt(self, question, choices):
        # Format choices as A, B, C, etc.
        choices_text = []
        for i, (choice, _) in enumerate(choices.items()):
            letter = chr(ord('A') + i)
            choices_text.append(f"{letter}. {choice}")
        
        prompt = f"""Question: {question}
Choices:
{chr(10).join(choices_text)}

Answer with only the letter of the correct choice:"""
        return prompt

    def extract_answer(self, response, num_choices):
        # Look for the first occurrence of a standalone letter in the response
        for i, char in enumerate(response):
            if char.isalpha() and ord(char) - ord('A') < num_choices:
                # Check if it's a standalone letter (surrounded by whitespace or punctuation)
                prev_char = response[i-1] if i > 0 else ' '
                next_char = response[i+1] if i < len(response)-1 else ' '
                if prev_char.isspace() or prev_char in '.,!?;:' and next_char.isspace() or next_char in '.,!?;:':
                    return ord(char) - ord('A')
        return None

    def evaluate_mc(self, mc_type, dataset):
        """
        Evaluate either mc0 or mc1 tasks
        mc_type: 'mc0' or 'mc1'
        """
        correct = 0
        total = 0
        results = []
        
        # Create a DataLoader for batching
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                'question': [item['question'] for item in x],
                'choices': [item[f'{mc_type}_targets'] for item in x],
                'answer': [list(item[f'{mc_type}_targets'].values()).index(1) for item in x],
                'num_choices': [len(item[f'{mc_type}_targets']) for item in x]
            }
        )
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {mc_type}")):
            # Format prompts for the batch
            prompts = [self.format_prompt(q, c) for q, c in zip(batch["question"], batch["choices"])]
            
            # Print first prompt for debugging
            if batch_idx == 0:
                print("\n=== Debug: Formatted Prompt Example ===")
                print(prompts[0])
                print("=====================================\n")
            
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
                max_new_tokens=10,
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
                
                predicted_idx = self.extract_answer(response, batch["num_choices"][i])
                correct_idx = batch["answer"][i]
                
                is_correct = predicted_idx == correct_idx
                if is_correct:
                    correct += 1
                total += 1
                
                # Print first answer comparison for debugging
                if total == 1:
                    print("\n=== Debug: Answer Comparison ===")
                    print(f"Raw output: {output}")
                    print(f"Processed response: {response}")
                    print(f"Predicted index: {predicted_idx}")
                    print(f"Correct index: {correct_idx}")
                    print(f"is_correct: {is_correct}")
                    print(f"Choices: {batch['choices'][i]}")
                    print(f"Number of choices: {batch['num_choices'][i]}")

                results.append({
                    "question": batch["question"][i],
                    "choices": batch["choices"][i],
                    "correct_answer": correct_idx,
                    "predicted_answer": predicted_idx,
                    "is_correct": is_correct
                })
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n=== Debug: {mc_type} Summary ===")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return accuracy, results

    def run_benchmark(self):
        # Load TruthfulQA dataset
        with open("TruthfulQA/mc_task.json", "r") as f:
            dataset = json.load(f)
        
        # Create model-specific directories
        model_name_safe = os.path.basename(self.model_name)
        if model_name_safe.startswith('.'):
            model_name_safe = model_name_safe[1:]
        if model_name_safe.startswith('/'):
            model_name_safe = model_name_safe[1:]
        model_name_safe = model_name_safe.replace('/', '_')
        
        results_dir = os.path.join("truthfulqa_results", model_name_safe)
        logs_dir = os.path.join("truthfulqa_logs", model_name_safe)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a log file for this run
        log_file = os.path.join(logs_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_file, "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        results = {}
        
        # Evaluate mc0
        mc0_accuracy, mc0_results = self.evaluate_mc('mc0', dataset)
        results['mc0'] = {
            "accuracy": float(mc0_accuracy),
            "results": mc0_results
        }
        
        # Evaluate mc1
        mc1_accuracy, mc1_results = self.evaluate_mc('mc1', dataset)
        results['mc1'] = {
            "accuracy": float(mc1_accuracy),
            "results": mc1_results
        }
        
        # Log results
        with open(log_file, "a") as f:
            f.write(f"MC0 Accuracy: {mc0_accuracy:.2%}\n")
            f.write(f"MC1 Accuracy: {mc1_accuracy:.2%}\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Save results
        output_file = os.path.join(results_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, "w") as f:
            json.dump({
                "model": self.model_name,
                "mc0_accuracy": float(mc0_accuracy),
                "mc1_accuracy": float(mc1_accuracy),
                "results": results
            }, f, indent=2)
        
        return results

if __name__ == "__main__":
    # Initialize evaluator with your model
    model_name = "Llama-3.1-8B-Instruct-finetuned-max_examples_1000_sources_0"
    evaluator = TruthfulQAEvaluator(model_name, batch_size=8)
    
    # Run benchmark
    results = evaluator.run_benchmark() 
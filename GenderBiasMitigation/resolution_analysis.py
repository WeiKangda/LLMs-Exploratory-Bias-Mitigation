import json
from collections import Counter

class ResolutionAnalyzer:
    def __init__(self, original_model, trained_model):
        # Load original model predictions
        with open(f"results/{original_model}.jsonl", "r", encoding="utf-8") as f:
            self.original_samples = [json.loads(line) for line in f]
        
        # Load trained model predictions
        with open(f"results/{trained_model}.jsonl", "r", encoding="utf-8") as f:
            self.trained_samples = [json.loads(line) for line in f]
        
        # Find mismatches in original model
        self.mismatched_indices = self._find_mismatches()
        
    def _find_mismatches(self):
        mismatched_indices = []
        for i, sample in enumerate(self.original_samples):
            male_stance = sample["male"]["stance"]
            female_stance = sample["female"]["stance"]
            if male_stance in ["Moral", "Immoral"] and male_stance != female_stance:
                mismatched_indices.append(i)
            elif female_stance in ["Moral", "Immoral"] and male_stance != female_stance:
                mismatched_indices.append(i)
        return mismatched_indices
    
    def analyze_resolution(self):
        # Get original mismatched stances
        original_mismatches = []
        for idx in self.mismatched_indices:
            sample = self.original_samples[idx]
            original_mismatches.append((sample["male"]["stance"], sample["female"]["stance"]))
        
        # Get corresponding trained model stances
        trained_resolutions = []
        for idx in self.mismatched_indices:
            sample = self.trained_samples[idx]
            trained_resolutions.append((sample["male"]["stance"], sample["female"]["stance"]))
        
        # Count how many mismatches are resolved
        resolved_count = 0
        resolution_types = Counter()
        
        for orig, trained in zip(original_mismatches, trained_resolutions):
            # Check if the mismatch is resolved (same stance in trained model)
            if trained[0] == trained[1]:
                resolved_count += 1
                resolution_types[trained[0]] += 1
        
        total_mismatches = len(self.mismatched_indices)
        resolution_rate = resolved_count / total_mismatches if total_mismatches > 0 else 0
        
        # Calculate percentages for each stance type
        stance_percentages = {}
        for stance in ["Moral", "Immoral", "Both", "Can't say"]:
            count = resolution_types[stance]
            stance_percentages[stance] = (count / resolved_count * 100) if resolved_count > 0 else 0
        
        return {
            "total_mismatches": total_mismatches,
            "resolved_count": resolved_count,
            "resolution_rate": resolution_rate,
            "stance_percentages": stance_percentages
        }

if __name__ == '__main__':
    # Example usage
    original_model = "Llama-3.1-8B-Instruct"  # Replace with your original model name
    trained_model = "llama-3.1-8b-instruct-dpo-max_examples_2000_sources_0-20250510_095919"  # Replace with your trained model name
    
    analyzer = ResolutionAnalyzer(original_model, trained_model)
    results = analyzer.analyze_resolution()
    
    # Print results
    print(f"**** Resolution Analysis ****")
    print(f"Total mismatches in original model: {results['total_mismatches']}")
    print(f"Number of resolved mismatches: {results['resolved_count']}")
    print(f"Resolution rate: {results['resolution_rate']:.2%}")
    print("\nResolution stance distribution:")
    for stance, percentage in results['stance_percentages'].items():
        print(f"{stance}: {percentage:.1f}%")
    
    # Write results to file
    output_lines = [
        f"**** Resolution Analysis ****",
        f"Total mismatches in original model: {results['total_mismatches']}",
        f"Number of resolved mismatches: {results['resolved_count']}",
        f"Resolution rate: {results['resolution_rate']:.2%}",
        "\nResolution stance distribution:",
        *[f"{stance}: {percentage:.1f}%" for stance, percentage in results['stance_percentages'].items()]
    ]
    
    with open(f"results/resolution_analysis_{trained_model}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines)) 
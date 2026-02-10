"""
Model Evaluation and Comparison Script
Compare different models and generation parameters
"""

import pandas as pd
import numpy as np
from inference import HeadlineGenerator
from transformers import MT5Tokenizer
import evaluate
from tqdm import tqdm
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        """
        Initialize evaluator with metrics
        """
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
    
    def evaluate_model(
        self,
        model_path,
        test_csv,
        text_column="article",
        headline_column="headline",
        num_samples=None,
        **generation_kwargs
    ):
        """
        Evaluate a model on test data
        
        Args:
            model_path: Path to the fine-tuned model
            test_csv: CSV file with test data
            text_column: Column name for articles
            headline_column: Column name for headlines
            num_samples: Limit evaluation to N samples (for speed)
            **generation_kwargs: Parameters for headline generation
        
        Returns:
            Dictionary with metrics and examples
        """
        print(f"\nEvaluating model: {model_path}")
        print(f"Test data: {test_csv}")
        
        # Load test data
        df = pd.read_csv(test_csv, encoding='utf-8')
        df = df[[text_column, headline_column]].dropna()
        
        if num_samples:
            df = df.head(num_samples)
            print(f"Evaluating on {num_samples} samples")
        else:
            print(f"Evaluating on {len(df)} samples")
        
        # Load model
        generator = HeadlineGenerator(model_path)
        
        # Generate headlines
        print("\nGenerating headlines...")
        generated_headlines = []
        
        for article in tqdm(df[text_column], desc="Generating"):
            headline = generator.generate_headline(article, **generation_kwargs)
            generated_headlines.append(headline)
        
        # Get reference headlines
        reference_headlines = df[headline_column].tolist()
        
        # Compute metrics
        print("\nComputing metrics...")
        
        # ROUGE scores
        rouge_results = self.rouge.compute(
            predictions=generated_headlines,
            references=reference_headlines,
            use_stemmer=False
        )
        
        # BLEU score
        bleu_results = self.bleu.compute(
            predictions=generated_headlines,
            references=[[ref] for ref in reference_headlines]
        )
        
        # Length statistics
        gen_lengths = [len(h.split()) for h in generated_headlines]
        ref_lengths = [len(h.split()) for h in reference_headlines]
        
        results = {
            "model_path": model_path,
            "num_samples": len(df),
            "generation_params": generation_kwargs,
            "metrics": {
                "rouge1": rouge_results["rouge1"],
                "rouge2": rouge_results["rouge2"],
                "rougeL": rouge_results["rougeL"],
                "bleu": bleu_results["bleu"],
            },
            "length_stats": {
                "generated_mean": np.mean(gen_lengths),
                "generated_median": np.median(gen_lengths),
                "reference_mean": np.mean(ref_lengths),
                "reference_median": np.median(ref_lengths),
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nROUGE Scores:")
        print(f"  ROUGE-1: {rouge_results['rouge1']:.4f}")
        print(f"  ROUGE-2: {rouge_results['rouge2']:.4f}")
        print(f"  ROUGE-L: {rouge_results['rougeL']:.4f}")
        print(f"\nBLEU Score: {bleu_results['bleu']:.4f}")
        print(f"\nLength Statistics:")
        print(f"  Generated - Mean: {np.mean(gen_lengths):.1f}, Median: {np.median(gen_lengths):.1f}")
        print(f"  Reference - Mean: {np.mean(ref_lengths):.1f}, Median: {np.median(ref_lengths):.1f}")
        
        # Sample comparisons
        print("\n" + "="*60)
        print("SAMPLE COMPARISONS")
        print("="*60)
        
        # Show best and worst examples
        rouge_scores = []
        for gen, ref in zip(generated_headlines, reference_headlines):
            score = self.rouge.compute(predictions=[gen], references=[ref])
            rouge_scores.append(score['rougeL'])
        
        rouge_scores = np.array(rouge_scores)
        best_indices = rouge_scores.argsort()[-3:][::-1]  # Top 3
        worst_indices = rouge_scores.argsort()[:3]  # Bottom 3
        
        results["examples"] = {
            "best": [],
            "worst": []
        }
        
        print("\n🏆 Best Examples (Highest ROUGE-L):")
        for idx in best_indices:
            example = {
                "article": df[text_column].iloc[idx][:150] + "...",
                "reference": reference_headlines[idx],
                "generated": generated_headlines[idx],
                "rouge_l": float(rouge_scores[idx])
            }
            results["examples"]["best"].append(example)
            
            print(f"\nArticle: {example['article']}")
            print(f"Reference: {example['reference']}")
            print(f"Generated: {example['generated']}")
            print(f"ROUGE-L: {example['rouge_l']:.4f}")
        
        print("\n⚠️  Worst Examples (Lowest ROUGE-L):")
        for idx in worst_indices:
            example = {
                "article": df[text_column].iloc[idx][:150] + "...",
                "reference": reference_headlines[idx],
                "generated": generated_headlines[idx],
                "rouge_l": float(rouge_scores[idx])
            }
            results["examples"]["worst"].append(example)
            
            print(f"\nArticle: {example['article']}")
            print(f"Reference: {example['reference']}")
            print(f"Generated: {example['generated']}")
            print(f"ROUGE-L: {example['rouge_l']:.4f}")
        
        return results
    
    def compare_generation_params(
        self,
        model_path,
        test_csv,
        text_column="article",
        headline_column="headline",
        num_samples=100,
        param_configs=None
    ):
        """
        Compare different generation parameters
        
        Args:
            model_path: Path to the model
            test_csv: Test data CSV
            param_configs: List of parameter dictionaries to compare
        
        Returns:
            Comparison results
        """
        if param_configs is None:
            param_configs = [
                {"num_beams": 1, "name": "Greedy"},
                {"num_beams": 4, "name": "Beam-4"},
                {"num_beams": 8, "name": "Beam-8"},
                {"num_beams": 4, "temperature": 0.8, "name": "Beam-4-Temp0.8"},
            ]
        
        print("\n" + "="*60)
        print("PARAMETER COMPARISON")
        print("="*60)
        
        results = []
        
        for config in param_configs:
            name = config.pop("name", str(config))
            print(f"\nTesting: {name}")
            print(f"Parameters: {config}")
            
            result = self.evaluate_model(
                model_path,
                test_csv,
                text_column=text_column,
                headline_column=headline_column,
                num_samples=num_samples,
                **config
            )
            result["config_name"] = name
            results.append(result)
        
        # Summary comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"\n{'Config':<20} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'BLEU':<10}")
        print("-" * 60)
        
        for result in results:
            metrics = result["metrics"]
            print(f"{result['config_name']:<20} "
                  f"{metrics['rouge1']:<10.4f} "
                  f"{metrics['rouge2']:<10.4f} "
                  f"{metrics['rougeL']:<10.4f} "
                  f"{metrics['bleu']:<10.4f}")
        
        return results
    
    def save_results(self, results, output_file="evaluation_results.json"):
        """
        Save evaluation results to JSON
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_file}")


def main():
    """
    Example usage
    """
    evaluator = ModelEvaluator()
    
    # Single model evaluation
    results = evaluator.evaluate_model(
        model_path="./burmese_headline_model",
        test_csv="test_data.csv",
        text_column="article",
        headline_column="headline",
        num_samples=None,  # Use all test data
        num_beams=4,
        max_length=128
    )
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")
    
    # Compare different generation parameters
    # comparison = evaluator.compare_generation_params(
    #     model_path="./burmese_headline_model",
    #     test_csv="test_data.csv",
    #     num_samples=100  # Use subset for speed
    # )
    # evaluator.save_results(comparison, "parameter_comparison.json")


if __name__ == "__main__":
    main()

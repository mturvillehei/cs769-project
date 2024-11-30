import torch
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
from datetime import datetime

class ModelEvaluator:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.metrics = {
            'bleu': evaluate.load('bleu'),
            'rouge': evaluate.load('rouge'),
            'meteor': evaluate.load('meteor'),
            'perplexity': evaluate.load('perplexity')
        }
        
    def load_model(self, model_path, adapter_path=None):
        print(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        if adapter_path:
            print(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        
        return model, tokenizer
        
    def evaluate_model(self, model, tokenizer, dataset):
        model.eval()
        results = {metric: [] for metric in self.metrics.keys()}
        
        with torch.no_grad():
            for item in tqdm(dataset, desc="Evaluating"):
                # For IMDB dataset, construct input prompt
                input_text = f"Review: {item['text']}\nSentiment:"
                # The label in IMDB is 0 for negative, 1 for positive
                reference = "negative" if item['label'] == 0 else "positive"
                
                inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_length,
                    num_beams=self.args.num_beams,
                    no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = prediction[len(input_text):].strip()  # Remove the input prompt
                
                # Calculate metrics
                for metric_name, metric in self.metrics.items():
                    if metric_name == 'perplexity':
                        score = metric.compute(predictions=[prediction], model_id=model.config._name_or_path)
                    else:
                        score = metric.compute(predictions=[prediction], references=[reference])
                    results[metric_name].append(score)
                
                if len(results['bleu']) % 100 == 0:
                    print(f"\nSample evaluation:")
                    print(f"Input: {input_text}")
                    print(f"Prediction: {prediction}")
                    print(f"Reference: {reference}")
        
        # Average results
        final_results = {k: np.mean([r[list(r.keys())[0]] if isinstance(r, dict) else r for r in v]) 
                        for k, v in results.items()}
        return final_results

def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("--base-model", required=True, help="Base model path/name")
    parser.add_argument("--lora-adapter", help="LoRA adapter path")
    parser.add_argument("--qlora-adapter", help="QLoRA adapter path")
    parser.add_argument("--dataset", default="imdb", help="Dataset to evaluate on")
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--num-beams", default=4, type=int)
    parser.add_argument("--no-repeat-ngram-size", default=3, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--eval-samples", default=1000, type=int, help="Number of samples to evaluate")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = load_dataset(args.dataset)['test'].select(range(args.eval_samples))
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_model, tokenizer = evaluator.load_model(args.base_model)
    base_results = evaluator.evaluate_model(base_model, tokenizer, dataset)
    
    # Evaluate LoRA if provided
    lora_results = None
    if args.lora_adapter:
        print("\nEvaluating LoRA model...")
        lora_model, _ = evaluator.load_model(args.base_model, args.lora_adapter)
        lora_results = evaluator.evaluate_model(lora_model, tokenizer, dataset)
    
    # Evaluate QLoRA if provided
    qlora_results = None
    if args.qlora_adapter:
        print("\nEvaluating QLoRA model...")
        qlora_model, _ = evaluator.load_model(args.base_model, args.qlora_adapter)
        qlora_results = evaluator.evaluate_model(qlora_model, tokenizer, dataset)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'base_model': base_results,
        'lora_model': lora_results,
        'qlora_model': qlora_results
    }
    
    results_file = f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    print(f"\nSaving results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\nEvaluation Results:")
    print("Base Model:", json.dumps(base_results, indent=2))
    if lora_results:
        print("LoRA Model:", json.dumps(lora_results, indent=2))
    if qlora_results:
        print("QLoRA Model:", json.dumps(qlora_results, indent=2))

if __name__ == "__main__":
    main()
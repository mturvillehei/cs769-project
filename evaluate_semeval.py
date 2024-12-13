import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.tokenization_utils_base import AddedToken
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import torch.nn.functional as F
from collections import defaultdict
import time
import psutil



class SEMEVALDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, task=1, split='train'):
        # Load the SEMEVAL data based on task and split
        filename = f"{split}_set_task{task}.txt"
        with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Add special tokens like in training code
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('text_tag')
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('tech_tag')
        tokenizer.add_special_tokens({
            'text_tag': AddedToken("<text>:", special=True),
            'tech_tag': AddedToken("<tech>:", special=True)
        })
        tokenizer._text_tag = '<text>:'
        tokenizer._tech_tag = '<tech>:'
        tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        self.samples = []
        for item in self.data:
            if split == 'train':
                formatted_text = f"<text>: {item['text']} <tech>: {' , '.join(item['labels'])}"
            else:
                formatted_text = f"<text>: {item['text']} <tech>: "
            
            encoded = tokenizer(
                formatted_text,
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            self.samples.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'human_reference': item['labels'],
                'text': item['text']  # Keep original text for generation
            })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample['input_ids']).squeeze(),
            "attention_mask": torch.tensor(sample['attention_mask']).squeeze(),
            "labels": sample['human_reference'],
            "text": sample['text']
        }

def generate_text(model, tokenizer, text, num_tokens=100):
    # Format input like training data
    input_text = f"<text>: {text} <tech>:"
    tokens = tokenizer(input_text, return_tensors='pt')['input_ids'].to(model.device)
    
    # Get stop tokens
    stop_tokens = {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.convert_tokens_to_ids('.')
    }
    
    with torch.no_grad():
        # Using top-k sampling like in training code
        for _ in range(num_tokens):
            logits = model(tokens)[0]
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            top50probs, top50indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(top50probs, 1)
            next_token = torch.gather(top50indices, -1, ix)
            tokens = torch.cat((tokens, next_token), dim=1)
            
            # Stop if we generate any stop token
            if next_token.item() in stop_tokens:
                break
    
    # Decode and extract techniques
    generated = tokenizer.decode(tokens[0], skip_special_tokens=False)
    try:
        techniques = generated.split("<tech>:")[1].strip()
        return [i for i in techniques.replace('<|end_of_text|>','').split(" , ") if i != ""]
    except IndexError:
        print("Warning: Generated text did not contain <tech>: token")
        return []

class ModelEvaluator:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        
    def load_model(self, model_path, adapter_path=None, tokenizer_path=None):
        print(f"Loading model from {model_path}")
        
        # Match training quantization setup
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('text_tag')
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('tech_tag')
        tokenizer.add_special_tokens({
            'text_tag': AddedToken("<text>:", special=True),
            'tech_tag': AddedToken("<tech>:", special=True)
        })
        tokenizer._text_tag = '<text>:'
        tokenizer._tech_tag = '<tech>:'
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=nf4_config,
            attn_implementation='flash_attention_2',
            device_map="auto"
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

        if adapter_path:
            print(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    def calculate_metrics(self, predictions, references):
        metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for pred_techniques, ref_techniques in zip(predictions, references):
            pred_set = set(pred_techniques)
            ref_set = set(ref_techniques)
            
            for tech in pred_set & ref_set:
                metrics[tech]['tp'] += 1
            for tech in pred_set - ref_set:
                metrics[tech]['fp'] += 1
            for tech in ref_set - pred_set:
                metrics[tech]['fn'] += 1
        
        results = {}
        # Overall metrics
        total_tp = sum(m['tp'] for m in metrics.values())
        total_fp = sum(m['fp'] for m in metrics.values())
        total_fn = sum(m['fn'] for m in metrics.values())
        
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        results['overall'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Per-technique metrics
        results['per_technique'] = {}
        for tech, counts in metrics.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            tech_precision = tp / (tp + fp) if tp + fp > 0 else 0
            tech_recall = tp / (tp + fn) if tp + fn > 0 else 0
            tech_f1 = 2 * tech_precision * tech_recall / (tech_precision + tech_recall) if tech_precision + tech_recall > 0 else 0
            
            results['per_technique'][tech] = {
                'precision': tech_precision,
                'recall': tech_recall,
                'f1': tech_f1,
                'support': tp + fn
            }
            
        return results

    def evaluate_model(self, model, tokenizer, dataset):
        model.eval()
        all_predictions = []
        all_references = []
        total_time = 0
        total_memory = 0
        
        with torch.no_grad():
            for item in tqdm(dataset, desc="Evaluating"):
                # Measure inference time and memory
                start_time = time.time()
                start_mem = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Generate predictions
                predicted_techniques = generate_text(model, tokenizer, item['text'])
                
                end_time = time.time()
                end_mem = psutil.Process().memory_info().rss / 1024 / 1024
                
                total_time += end_time - start_time
                total_memory += end_mem - start_mem
                
                all_predictions.append(predicted_techniques)
                all_references.append(item['labels'])
                
                # Print sample predictions periodically
                if len(all_predictions) % 10 == 0:
                    print(f"\nSample evaluation:")
                    print(f"Input: {item['text']}")
                    print(f"Predicted: {predicted_techniques}")
                    print(f"Reference: {item['labels']}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_references)
        
        # Add performance metrics
        n_samples = len(dataset)
        metrics['performance'] = {
            'avg_inference_time_ms': (total_time / n_samples) * 1000,
            'avg_memory_usage_mb': total_memory / n_samples,
            'total_samples': n_samples
        }
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="SEMEVAL Model Evaluation")
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data-dir", required=True, help="Path to SEMEVAL dataset directory")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2])
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA/QLoRA adapter")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args)
    
    # Load model and tokenizer
    model, tokenizer = evaluator.load_model(args.model_name, args.adapter_path, args.tokenizer)
    
    # Load test dataset
    test_dataset = SEMEVALDataset(
        args.data_dir,
        tokenizer,
        max_length=args.max_length,
        task=args.task,
        split='test'
    )
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluator.evaluate_model(model, tokenizer, test_dataset)
    
    # Save results
    results_file = f'semeval_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results Summary:")
    print("\nOverall Metrics:")
    print(json.dumps(results['overall'], indent=2))
    print("\nPerformance Metrics:")
    print(json.dumps(results['performance'], indent=2))

if __name__ == "__main__":
    main()
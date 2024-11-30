import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import json

class SEMEVALDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, task=1):
        # Load the SEMEVAL data based on task
        if task == 1:
            with open(os.path.join(data_path, 'training_set_task1.txt'), 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif task == 2:
            with open(os.path.join(data_path, 'training_set_task2.txt'), 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        self.techniques = list(set(technique for item in self.data for technique in item['labels']))
        self.technique_to_index = {technique: index for index, technique in enumerate(self.techniques)}
        self.num_techniques = len(self.techniques)

        # Process the data into text and labels
        self.texts = []
        self.labels = []
        for item in self.data:
            self.texts.append(item['text'])
            self.labels.append(item['labels'])
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Create a fixed-size label tensor
        label_tensor = torch.zeros(self.num_techniques, dtype=torch.float)
        for technique in label:
            index = self.technique_to_index[technique]
            label_tensor[index] = 1.0
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_tensor
        }
def train(args):
    print("Setting up training...")
    # Configure quantization
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=nf4_config,
        device_map="auto"
    )

    # Configure LoRA or QLoRA
    print(f"Configuring {args.lora_type}...")
    if args.lora_type == "lora":
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
    elif args.lora_type == "qlora":
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=["q_proj", "v_proj", "fc1", "fc2"],
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        raise ValueError(f"Invalid LoRA type: {args.lora_type}")

    model = get_peft_model(model, lora_config)
    
        
    # Create training dataset
    print("Loading SEMEVAL dataset...")
    train_dataset = SEMEVALDataset(
        args.data_dir,
        tokenizer,
        max_length=args.max_length,
        task=args.task
    )
    
    val_dataset = SEMEVALDataset(
        args.data_dir,
        tokenizer,
        max_length=args.max_length,
        task=args.task
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"semeval_checkpoint_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save_pretrained(checkpoint_path)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**inputs)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="SEMEVAL Fine-tuning")
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data-dir", required=True, help="Path to SEMEVAL dataset directory")
    parser.add_argument("--lora-type", default="lora", choices=["lora", "qlora"], help="Type of LoRA to use (lora or qlora)")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--task", type=int, default=1, choices=[1, 2], help="SEMEVAL subtask (1 or 2)")
    
    args = parser.parse_args()
    
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    
    train(args)

if __name__ == "__main__":
    main()
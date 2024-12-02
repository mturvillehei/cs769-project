import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.tokenization_utils_base import AddedToken
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import json
from huggingface_hub import login
import torch.nn.functional as F

class SEMEVALDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, task=1, split='train'):
        # Load the SEMEVAL data based on task
        if task == 1:
            with open(os.path.join(data_path, 'training_set_task1.txt'), 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif task == 2:
            with open(os.path.join(data_path, 'training_set_task2.txt'), 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Add special tokens like in E2E code
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
                'human_reference': item['labels']  # Keep original labels for evaluation
            })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.split == "validation" or self.split == "test":
            return {
                "input_ids": torch.tensor(sample['input_ids']).squeeze(),
                "attention_mask": torch.tensor(sample['attention_mask']).squeeze(),
                "labels": sample['human_reference']
            }
        
        # For training, return the full sequence as both input and target
        return {
            'input_ids': torch.tensor(sample['input_ids']).squeeze(),
            'attention_mask': torch.tensor(sample['attention_mask']).squeeze(),
            'labels': torch.tensor(sample['input_ids']).squeeze()  # Same as input for causal LM
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
        # Using top-k sampling like in E2E code
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
        return techniques.split(" , ")
    except IndexError:
        print("Warning: Generated text did not contain <tech>: token")
        return []

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
        attn_implementation='flash_attention_2'
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
        task=args.task,
        split='train'
    )

    val_dataset = SEMEVALDataset(
        args.data_dir,
        tokenizer,
        max_length=args.max_length,
        task=args.task,
        split='validation'
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

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.to('cuda')

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            optimizer.zero_grad()
            
            # Move all tensors to the same device as the model
            inputs = {
                'input_ids': batch['input_ids'].to('cuda'),
                'attention_mask': batch['attention_mask'].to('cuda'),
                'labels': batch['labels'].to('cuda')
            }

            # Forward pass - model will automatically calculate loss using labels
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Generate sample output every 100 steps
            if step % 100 == 0:
                # Get input text up to the <tech>: token
                input_ids = batch['input_ids'][0]
                tech_token_pos = (input_ids == tokenizer.convert_tokens_to_ids("<tech>:")).nonzero()[0]
                input_text = tokenizer.decode(input_ids[:tech_token_pos+1], skip_special_tokens=True)

                print("\nSample generation:")
                print("Input:", input_text)
                print("Generated techniques:", generate_text(model, tokenizer, input_text))
                print("Actual techniques:", tokenizer.decode(batch['labels'][0], skip_special_tokens=True))
                print()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        # Save checkpoint
        checkpoint_path = f"semeval_checkpoint_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save_pretrained(checkpoint_path)
        
        # Validation
        model.eval()
        val_loss = 0
        '''
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**inputs)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}"
        '''
        if(epoch % 5 == 0):
            model.save_pretrained(f"semeval_finetuning")

def main():
    login(token='')
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
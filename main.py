import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import LoRA
from contextlib import contextmanager

parser = argparse.ArgumentParser(description="LoRA arguments")
parser.add_argument("-lr", default=3e-4, type=float, help="Learning rate")
parser.add_argument("-epochs", default=1, type=int, help="Number of epochs")
parser.add_argument("-batch_size", default=8, type=int, help="Batch size")
parser.add_argument("-rank","-r", default=8, type=int, help="Rank of model")
parser.add_argument("-alpha","-a", default=16, type=float, help="Alpha for model")
parser.add_argument('-lm','--lora-modules' , default='c_attn', help="modules in the network to replace with LoRA blocks")
parser.add_argument("--no-lora", action='store_true', help="For testing against baseline training")

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    if not parser.parse_args().no_lora:
        model = LoRA.add_lora_attention_layers(model, rank=parser.parse_args().rank, alpha=parser.parse_args().alpha, target_modules=parser.parse_args().lora_modules)
    param_count = 0
    for n,p in model.named_parameters():
        if p.requires_grad:
            param_count += p.numel()
    print("LoRA parameter count:" + str(param_count))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = load_dataset("imdb")
    train_dataset = CustomDataset(dataset["train"]['text'], dataset["train"]['label'], tokenizer)
    val_dataset = CustomDataset(dataset["test"]['text'], dataset["test"]['label'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=parser.parse_args().batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=parser.parse_args().batch_size, shuffle=True)
    eval_model(model, val_loader,device)
    train(model, parser.parse_args().epochs, parser.parse_args().lr, train_loader, val_loader, device)




def train(model, epochs, lr, train_loader, val_loader, device):
    params_to_update = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamW(params_to_update, lr=lr)
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        mem_consumption = []
        for batch in tqdm(train_loader, desc="training"):
            optimizer.zero_grad()
            model.zero_grad()
            input_dat = {k: v.to(device) for k, v in batch.items()}
            output = model(**input_dat)
            loss = output.loss
            loss.backward()
            total_loss += loss.item()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            optimizer.step()
            if torch.cuda.is_available():
                gpu_max = torch.cuda.max_memory_allocated() / 1024 ** 3
                mem_consumption.append(gpu_max)
        avg_train_loss = total_loss / len(train_loader)
        losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average memory consumption: {sum(mem_consumption) / len(mem_consumption)}")
        eval_model(model, val_loader, device)
        model.train()


def eval_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])

    # Print metrics
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    print(f'Average validation loss: {avg_val_loss:.4f}')
    print(f'Validation accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
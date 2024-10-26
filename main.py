import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pandas as pd

import LoRA

parser = argparse.ArgumentParser(description="LoRA arguments")
parser.add_argument("-data", default = "data", help="Path to data file")
parser.add_argument("-model", default="openai-community/gpt2-medium", help="String for huggingface model")
parser.add_argument("-lr", default=1e-5, type=float, help="Learning rate")
parser.add_argument("-epochs", default=20, type=int, help="Number of epochs")
parser.add_argument("-batch_size", default=8, type=int, help="Batch size")
parser.add_argument("-rank","-r", default=8, type=int, help="Rank of model")
parser.add_argument("-alpha","-a", default=1, type=float, help="Alpha for model")
parser.add_argument('-lm','--lora-modules' , default='c_attn', help="modules in the network to replace with LoRA blocks")


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.input_ids = []
        for text in texts:
            encoding = tokenizer.encode_plus(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.input_ids.append(encoding['input_ids'].squeeze())
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]

def main():
    model_name = parser.parse_args().model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = False
    model = LoRA.add_lora_attention_layers(model, rank=parser.parse_args().rank, alpha=parser.parse_args().alpha, target_modules=parser.parse_args().lora_modules)
    train(model, tokenizer, parser.parse_args().data, parser.parse_args().epochs, parser.parse_args().batch_size, parser.parse_args().lr)



'''
Training loop, not done yet
'''
def train(model, tokenizer, data_f, epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer.to(device)
    lora_params = [n for n, p in model.named_parameters() if 'lora' in n]
    optimizer = AdamW(lora_params, lr=lr)
    data = pd.read_csv(data_f)
    texts = data.iloc[1,:].values.tolist()
    labels = data.iloc[2,:].values.tolist()
    dataloader = DataLoader(CustomDataset(texts, labels, tokenizer, parser.parse_args().batch_size), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        for text,label in dataloader:
            optimizer.zero_grad()
            input_dat = text.to(device)
            output = model(tokenizer.encode_plus(input_dat, return_tensors="pt").to(device))
            loss = output.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{range(epochs)}, Loss: {total_loss:.4f}")




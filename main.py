import sys
from datetime import time, datetime
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
from datasets import load_dataset
from tqdm import tqdm
import LoRA
import pickle
import evaluate
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
parser = argparse.ArgumentParser(description="LoRA arguments")
parser.add_argument("-lr", default=3e-4, type=float, help="Learning rate")
parser.add_argument("-epochs", default=1, type=int, help="Number of epochs")
parser.add_argument("--train-batch", default=8, type=int, help="Train Batch size")
parser.add_argument("--eval-batch", default=128, type=int, help="Eval Batch size")
parser.add_argument("-rank","-r", default=8, type=int, help="Rank of model")
parser.add_argument("-alpha","-a", default=16, type=float, help="Alpha for model")
parser.add_argument('-lm','--lora-modules' , default='c_attn', help="modules in the network to replace with LoRA blocks")
parser.add_argument("--no-lora", action='store_true', help="For testing against baseline training")
parser.add_argument("-save", default="model_" + str(datetime.now()).replace(":","_"), help="Name to save model to")
parser.add_argument("-decay", default=.01, type=float, help="Decay factor")
parser.add_argument("-dropout", default=.1, type=float, help="Dropout rate")
parser.add_argument("--beam-num", default=5, type=int, help="Beam size")
parser.add_argument("--length-penalty", default=.9, type=float, help="Length penalty")
parser.add_argument("--no-repeat-ngram", default=4, type=int, help="size for repeating ngram rule")

class E2EDataset(Dataset):
    """Custom Dataset for E2E NLG data"""

    def __init__(
            self,
            tokenizer,
            max_length: int = 128,
            split: str = "train",
    ):
        # Load dataset
        self.data = load_dataset("e2e_nlg", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        # Initialize tokenizer if not provided

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split == "validation" or self.split == "test":
            text = self.tokenizer(self.data[idx]['meaning_representation'] + ' | ', truncation=True, max_length=self.max_length,
                              padding="max_length", return_tensors="pt")
            return {
                "input_ids": torch.tensor(text['input_ids']).squeeze(),
                "attention_mask": torch.tensor(text['attention_mask']).squeeze(),
                "labels": self.data[idx]['human_reference']
            }
        text = self.tokenizer(self.data[idx]['meaning_representation'] + ' | ' + self.data[idx]['human_reference'], truncation=True, max_length=self.max_length,
                              padding="max_length", return_tensors="pt")
        input_ids = torch.tensor(text['input_ids']).squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': torch.tensor(text['attention_mask']).squeeze(),
            'labels': input_ids,
        }

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium',
                                            num_labels=2,
                                            attn_implementation='flash_attention_2',
                                            torch_dtype=torch.bfloat16)
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    if not parser.parse_args().no_lora:
        model = LoRA.add_lora_attention_layers(model, rank=parser.parse_args().rank, alpha=parser.parse_args().alpha, target_modules=parser.parse_args().lora_modules)
    param_count = 0
    for n,p in model.named_parameters():
        if p.requires_grad:
            param_count += p.numel()
    print("Active parameter count:" + str(param_count))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_dataset = E2EDataset(tokenizer)
    val_dataset = E2EDataset(tokenizer, split='validation')
    train_loader = DataLoader(train_dataset, batch_size=parser.parse_args().train_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=parser.parse_args().eval_batch, shuffle=True)
    eval_metrics = [evaluate.load('bleu'), evaluate.load('rouge'), evaluate.load('meteor')]
    train(model, parser.parse_args().epochs, parser.parse_args().lr, train_loader, device)
    model.save_pretrained("model\\" + parser.parse_args().save)
    eval_model(model, tokenizer, val_loader, device, eval_metrics)




def train(model, epochs, lr, train_loader, device):
    params_to_update = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamW(params_to_update, lr=lr, weight_decay=parser.parse_args().decay)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=len(train_loader)*epochs)
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        mem_consumption = []
        for batch in tqdm(train_loader, desc="training"):
            optimizer.zero_grad()
            model.zero_grad()
            input_dat = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type='cuda'):
                output = model(input_ids=input_dat['input_ids'], attention_mask=input_dat['attention_mask'], labels=input_dat['labels'])
                loss = output.loss
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            loss.backward()
            total_loss+=loss.item()
            optimizer.step()
            scheduler.step()
            torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
            if torch.cuda.is_available():
                gpu_max = torch.cuda.max_memory_allocated() / 1024 ** 3
                mem_consumption.append(gpu_max)
        avg_train_loss = total_loss / len(train_loader)
        losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average memory consumption: {sum(mem_consumption) / len(mem_consumption)}")


def eval_model(model, tokenizer, val_loader, device, eval_metrics):
    model.eval()
    score_mat = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_dat = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model.generate(input_ids=input_dat['input_ids'],
                                     attention_mask=input_dat['attention_mask'],
                                     num_return_sequences=1,
                                     max_new_tokens=50,
                                     num_beams=parser.parse_args().beam_num,
                                     no_repeat_ngram_size=parser.parse_args().no_repeat_ngram,
                                     length_penalty=parser.parse_args().length_penalty)
            pred = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            scores = [metric.compute(predictions=pred, references=batch['labels']) for metric in eval_metrics]
            score_mat.append(scores)
    score_mat = np.array(score_mat)
    s = [[g[0]['bleu'], g[1]['rougeL'], g[2]['meteor']] for g in score_mat]
    s = np.array(s)
    s = np.mean(s, axis=0)
    print(f"Bleu: {s[0]}")
    print(f"RougeL: {s[1]}")
    print(f"Meteor: {s[2]}")
    pickle.dump(score_mat, open(f"score_mat.pkl", "wb"))

if __name__ == '__main__':
    main()
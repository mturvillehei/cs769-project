import pickle
from typing import Dict
import os

import evaluate
import numpy as np
import tiktoken
import torch
from datasets import ClassLabel, load_dataset, Value
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import GPT2
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, TrainingArguments, IntervalStrategy, Trainer
import torch.nn.functional as F
from transformers.tokenization_utils_base import AddedToken
class E2EDataset(Dataset):
    """Custom Dataset for E2E NLG data"""

    def __init__(
            self,
            data,
            max_length: int = 256,
            split: str = 'train'
            ):
        # Load dataset
        self.samples = data
        self.max_length = max_length
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split == "validation" or self.split == "test":
            return {
                "input_ids": torch.tensor(self.samples[idx]['input_ids']).squeeze(),
                "attention_mask": torch.tensor(self.samples[idx]['attention_mask']).squeeze(),
                "labels": self.samples[idx]['human_reference']
            }

        return {
            'input_ids': torch.tensor(self.samples[idx]['input_ids']).squeeze(),
            'attention_mask': torch.tensor(self.samples[idx]['attention_mask']).squeeze(),
            'labels': torch.tensor(self.samples[idx]['input_ids']).squeeze()
        }
'''
ADAPTED FROM https://github.com/kobzaond/e2e_transformer_nlg
'''
def train_gpt(args):
    os.mkdir("model\\" + args.save)
    # load data
    reporting_enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    # add special token: separator between MR and a corresponding human ref.
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('reference_tag')
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('meaning_tag')
    tokenizer.add_special_tokens({'reference_tag': AddedToken("<ref>:", special=True)})
    tokenizer.add_special_tokens({'meaning_tag': AddedToken("<mr>:", special=True)})
    tokenizer._reference_tag = '<ref>:'
    tokenizer._meaning_tag = '<mr>:'
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer("<mr>:"))
    GPT2.GPTConfig.lora_r = args.rank
    GPT2.GPTConfig.lora_a = args.alpha
    if not args.no_lora:
        model = GPT2.GPT.from_pretrained("gpt2-medium", len(tokenizer))
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium", torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
        model.resize_token_embeddings(len(tokenizer))
    data = load_dataset('e2e_nlg')
    data = data.map(
        lambda x: tokenizer(
            list(
                map(
                    lambda a, b: "<mr>: " + a + " <ref>: " + b, x['meaning_representation'], x['human_reference'])),
            truncation="only_second", max_length=256, padding='max_length'), batched=True
    )
    data = data.map(lambda x: {'labels': x['input_ids']})

    param_count = 0
    if not args.no_lora:
        for n, p in model.named_parameters():
            if 'lora' in n:
                param_count += p.numel()
                p.requires_grad = True
            else:
                p.requires_grad = False

    model.to(device)
    print("Active parameter count:" + str(param_count))
    print('It is time.')
    train_set = E2EDataset(data['train'])
    train_loader = DataLoader(train_set, batch_size=args.train_batch, shuffle=True)
    warmup = args.warmup
    params_to_update = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamW(params_to_update, lr=args.lr, weight_decay=args.decay)
    over_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=len(train_loader) * args.epochs - 500 * warmup)
    warmup_scheduler = LinearLR(optimizer, start_factor=.1, end_factor=1.0, total_iters=500 * warmup)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, over_scheduler], [500 * warmup])
    model.train()
    losses = []
    cl100k_base = tiktoken.get_encoding("gpt2")

    # In production, load the arguments directly instead of accessing private attributes
    # See openai_public.py for examples of arguments for specific encodings
    enc = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<ref>:": 50257,
            "<mr>:": 50258
        }
    )

    for epoch in range(args.epochs):
        total_loss = 0
        mem_consumption = []
        for step in tqdm(range(len(train_loader)), desc="training"):
            batch = train_loader.__iter__().__next__()
            optimizer.zero_grad()
            model.zero_grad()
            input_dat = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type='cuda'):
                output, loss = model(input_ids=input_dat['input_ids'],labels=input_dat['labels'])
                #loss = output.loss
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
            if torch.cuda.is_available():
                gpu_max = torch.cuda.max_memory_allocated() / 1024 ** 3
                mem_consumption.append(gpu_max)
            if step % 100 == 0: #and step > 0:
                print(len(input_dat['input_ids'][0][:(input_dat['input_ids'][0] == 50257).nonzero()[0]]))
                generate_text(model,enc, input_dat['input_ids'][0][:(input_dat['input_ids'][0] == 50257).nonzero()[0]])
            if step % 500 == 0 and step > 0:
                print(f"Loss: {total_loss/step}, Epoch {epoch}, Step {step}")
        avg_train_loss = total_loss / len(train_loader)
        losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average memory consumption: {sum(mem_consumption) / len(mem_consumption)}")
        checkpoint = {
            'model': model.state_dict(),
            'config': model.config,
            'step': len(train_loader)*epoch
        }
        torch.save(checkpoint,"model\\" + args.save + f"\\checkpoint-{epoch*len(train_loader)}.pt")

    # fine-tune the model; the evaluation metric is the eval loss
    tokenizer.save_pretrained(args.output_dir)


def eval(args):
    torch.serialization.add_safe_globals([GPT2.GPTConfig])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('reference_tag')
    tokenizer.add_special_tokens({'reference_tag': "<ref:>"})
    tokenizer._reference_tag = '<ref>:'
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2.GPT.from_pretrained("gpt2-medium", len(tokenizer))
    model.load_state_dict(torch.load(args.model, weights_only=True)['model'])
    model = model.to_hf_model()
    val_data = load_dataset('e2e_nlg', split='validation')
    val_data = val_data.map(
        lambda x: tokenizer(
            list(
                map(
                    lambda a: a + " <ref:> ", x['meaning_representation'])),
            truncation="only_second", max_length=256, padding='max_length'), batched=True
    )
    val_dataset = E2EDataset(val_data, split='validation')
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch, shuffle=False)
    model.to(device)
    eval_metrics = [evaluate.load('bleu'), evaluate.load('rouge'), evaluate.load('meteor')]
    model.eval()
    score_mat = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_dat = {k: v.to(device) for k, v in batch.items() if k != 'labels'}

            outputs = model.generate(input_ids=input_dat['input_ids'],
                                     attention_mask=input_dat['attention_mask'],
                                     num_return_sequences=1,
                                     max_new_tokens=256,
                                     num_beams=args.beam_num,
                                     no_repeat_ngram_size=args.no_repeat_ngram,
                                     length_penalty=args.length_penalty,
                                     early_stopping=True)

            #outputs = generate_text(model, input_dat['input_ids'])
            pred = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            print(pred[0])
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


    #Print the generated text

def generate_text(model, enc, in_tokens: torch.tensor, num_tokens=256, num_samples=1):
    with torch.no_grad():
        tokens = in_tokens.unsqueeze(0).repeat(num_samples, 1)
        for _ in range(num_tokens):
            logits = model(tokens)[0]
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            top50probs, top50indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(top50probs, 1)
            xcol = torch.gather(top50indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)

    for i in range(num_samples):
        toks = tokens[i, :].tolist()
        print('>',enc.decode(toks))
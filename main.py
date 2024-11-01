
from datetime import time, datetime
from typing import Dict
import evaluate
from transformers import AutoTokenizer, GPT2LMHeadModel
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import argparse
from datasets import load_dataset
from tqdm import tqdm
import pickle
#import evaluate
import logging
import train_gpt2

logging.getLogger("transformers").setLevel(logging.ERROR)
parser = argparse.ArgumentParser(description="LoRA arguments")
parser.add_argument("-lr", default=3e-4, type=float, help="Learning rate")
parser.add_argument("-epochs", default=1, type=int, help="Number of epochs")
parser.add_argument("--train-batch", default=8, type=int, help="Train Batch size")
parser.add_argument("--eval-batch", default=2, type=int, help="Eval Batch size")
parser.add_argument("-rank","-r", default=8, type=int, help="Rank of model")
parser.add_argument("-alpha","-a", default=16, type=float, help="Alpha for model")
parser.add_argument('-lm','--lora-modules' , default='c_attn', help="modules in the network to replace with LoRA blocks")
parser.add_argument("--no-lora", action='store_true', help="For testing against baseline training")
parser.add_argument("-save", default="model_" + str(datetime.now()).replace(":","_").split(".")[0].replace(" ", "_"), help="Name to save model to")
parser.add_argument("-decay", default=.01, type=float, help="Decay factor")
parser.add_argument("-dropout", default=.1, type=float, help="Dropout rate")
parser.add_argument("--beam-num", default=5, type=int, help="Beam size")
parser.add_argument("--length-penalty", default=.9, type=float, help="Length penalty")
parser.add_argument("--no-repeat-ngram", default=4, type=int, help="size for repeating ngram rule")
parser.add_argument("-warmup", action="store_true", help="Warm up training")
parser.add_argument("-eval", action="store_true", help="Evaluate model")
parser.add_argument("-model",default="model/latest")
torch.manual_seed(1234)
args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = ""
    if parser.parse_args().eval:
        train_gpt2.eval(args)
        exit(0)
    train_gpt2.train_gpt(args)



if __name__ == '__main__':
    main()
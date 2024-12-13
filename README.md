# Accessible Finetuning for Semantic Evaluation

To run this repo and replicate the four results that we focused on in the writeup
(Llama 3.2 B FFT,LoRA,QLoRA, and Llama 2-7B QLoRA), all that needs to be done is run the script 
"semeval.sh". To evaluate these results, run "evaluate_semeval.sh".

NOTE

In order to access the Huggingface models from Meta, you need a token! Put a valid token
in line 285 in semeval_finetuning.py. You need access to the Llama 2 and
Llama 3.2 repositories.


# SEMEVAL Fine-tuning with LoRA/QLoRA

This repository contains code for fine-tuning and evaluating language models on the SEMEVAL 2021 Task 6 dataset using LoRA and QLoRA approaches.

## Order of Operations

1. Test QLoRA setup (`qllama_test.py`)
2. Initialize and save model (`init_model.py`)
3. Fine-tune on SEMEVAL data (`semeval_finetuning.py`)
4. Evaluate results (`evaluate_semeval.py`, `model_evaluation.py`)

## Scripts

### qllama_test.py
Quick validation script to test QLoRA configuration.

Arguments:
- `--model`: Model name/path (default: meta-llama/Llama-2-7b-hf)
- `--rank`, `-r`: LoRA rank (default: 8)
- `--alpha`, `-a`: LoRA alpha (default: 32)
- `--dropout`: LoRA dropout (default: 0.05)
- `--test-input`: Test input text

### init_model.py
Initializes and saves the quantized model and LoRA adapters.

Key operations:
- Loads and quantizes base model
- Configures LoRA
- Saves quantized model and adapter weights

### semeval_finetuning.py
Main fine-tuning script for SEMEVAL task.

Arguments:
- `--model-name`: Model to fine-tune
- `--data-dir`: Path to SEMEVAL dataset
- `--batch-size`: Training batch size (default: 8)
- `--epochs`: Number of training epochs (default: 3)
- `--task`: SEMEVAL subtask (1 or 2)
- `--lora-type`: Type of fine-tuning (lora/qlora)

### evaluate_semeval.py
Evaluates model performance on SEMEVAL task.

Arguments:
- `--model-name`: Path to fine-tuned model
- `--data-dir`: Path to SEMEVAL dataset
- `--task`: SEMEVAL subtask (1 or 2)
- `--adapter-path`: Path to adapter configuration
- `--max-length`: Maximum sequence length (default: 256)

### model_evaluation.py
General model evaluation script supporting multiple evaluation approaches.

Arguments:
- `--base-model`: Base model to evaluate
- `--lora-adapter`: Path to LoRA adapter (optional)
- `--qlora-adapter`: Path to QLoRA adapter (optional)
- `--dataset`: Dataset name
- `--max-length`: Maximum sequence length
- `--num-beams`: Number of beams for generation
- `--eval-samples`: Number of samples to evaluate
- `--batch-size`: Evaluation batch size

## Example Usage

```bash
# 1. Test QLoRA setup
python qllama_test.py --model meta-llama/Llama-2-7b-hf --rank 8 --alpha 32

# 2. Fine-tune on SEMEVAL
python semeval_finetuning.py --model-name meta-llama/Llama-2-7b-hf --data-dir ./data/SEMEVAL-2021-task6-corpus/data --batch-size 8 --epochs 3 --task 1 --lora-type qlora

# 3. Evaluate results
python evaluate_semeval.py --model-name ./semeval_finetuning --data-dir ./data/SEMEVAL-2021-task6-corpus/data --task 1 --adapter-path ./semeval_finetuning/adapter_config.json --max-length 256
```
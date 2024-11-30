import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Test Script")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name/path")
    parser.add_argument("--rank", "-r", default=8, type=int, help="LoRA rank")
    parser.add_argument("--alpha", "-a", default=32, type=float, help="LoRA alpha")
    parser.add_argument("--dropout", default=0.05, type=float, help="LoRA dropout")
    parser.add_argument("--test-input", default="Test input to verify model is working.", help="Test input text")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Configure quantization
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=nf4_config,
        device_map="auto"
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Quick test
    print("\nTesting model with input...")
    inputs = tokenizer(args.test_input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    print("\nGenerated text:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
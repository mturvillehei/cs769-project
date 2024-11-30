from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import os
from datetime import datetime

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# Configure NF4 quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the quantized base model
print(f"Loading quantized model from {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=nf4_config,
    device_map="auto"  # Automatically map model to available devices
)
print(f"Base pre-trained model loaded successfully from {MODEL_PATH}.")

# Load the LoRA adapters
print(f"Loading LoRA adapters from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print(f"LoRA adapters successfully applied from {ADAPTER_PATH}.")

# Load tokenizer
print(f"Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Tokenizer loaded successfully.")

# Verify model is ready
print("Model and adapters are ready for inference or fine-tuning.")


# Load the dataset

# Perform fine-tuning by freezing everything except for the LoRA adapters
# Monitor memory consumption and time to fine-tune

# Save the final fine-tuned model (LoRA adapters)

# Evaluate using whatever metrics

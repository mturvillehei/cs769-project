from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import os
from datetime import datetime

## Load the Model from HuggingFace

hf_model_name = "meta-llama/Llama-3.2-1B"
# hf_model_name = "meta-llama/Llama-2-7b-hf"

model_base_name = hf_model_name.split('/')[-1]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"./{model_base_name}_quantized_{timestamp}"
LORA_ADAPTER_PATH = f"./{model_base_name}_loraAdapter_{timestamp}"

# https://huggingface.co/blog/4bit-transformers-bitsandbytes
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

model = AutoModelForCausalLM.from_pretrained(hf_model_name, quantization_config=nf4_config, device_map="auto")

## Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()

# Save the base pre-trained QUANTIZED model
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

# Save the LoRA adapters
if not os.path.exists(LORA_ADAPTER_PATH):
    os.makedirs(LORA_ADAPTER_PATH)

model.get_peft_model().save_pretrained(LORA_ADAPTER_PATH)

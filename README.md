# Accessible Finetuning for Semantic Evaluation

To run this repo and replicate the four results that we focused on in the writeup
(Llama 3.2 B FFT,LoRA,QLoRA, and Llama 2-7B QLoRA), all that needs to be done is run the script 
"semeval.sh". To evaluate these results, run "evaluate_semeval.sh".

NOTE

In order to access the Huggingface models from Meta, you need a token! Put a valid token
in line 285 in semeval_finetuning.py. You need access to the Llama 2 and
Llama 3.2 repositories.


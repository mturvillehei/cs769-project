python model_evaluation.py --base-model meta-llama/Llama-2-7b-hf --lora-adapter ./lora_checkpoint --qlora-adapter ./qlora_checkpoint --dataset imdb --max-length 128 --num-beams 4
#python model_evaluation.py --base-model meta-llama/Llama-2-7b-hf --eval-samples 1000 --max-length 32 --num-beams 1 --batch-size 1

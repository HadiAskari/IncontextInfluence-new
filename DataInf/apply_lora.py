"""
Apply the LoRA weights on top of a base model.

Usage:
python3 -m apply_lora --base llama-2-13b-chat-converted --target llama-2-70b-sarcasm-combined --lora /nas02/Hadi/Incontenxt-influence/DataInf/llama2-70b-sarcasm-justlora

Hadi Usage
python3 -m fastchat.model.apply_lora --base meta-llama/Meta-Llama-3-8B-Instruct --target /nas02/Hadi/Pinterest/llama3-8b/fine-tuning/i2pc/finetuned-model --lora /nas02/Hadi/Pinterest/llama3-8b/fine-tuning/i2pc/finetuned-lora

Dependency:
pip3 install git+https://github.com/huggingface/peft.git@2822398fbe896f25d4dac5e468624dc5fd65a51b
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_lora(base_model_path, target_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        #torch_dtype=torch.float16,  #comment out for qlora
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)

    args = parser.parse_args()

    apply_lora(args.base_model_path, args.target_model_path, args.lora_path)
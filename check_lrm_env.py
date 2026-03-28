# check_lrm_env.py
import torch
from transformers import AutoModelForCausalLM
import flash_attn

def check():
    print(f"--- LRM Environment Check (06:13 AM) ---")
    print(f"Device: {torch.cuda.get_device_name(0)}") # H100
    print(f"Flash-Attn: {flash_attn.__version__}")
    
    # Proba BF16 et FlashAttention 2
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        print("Status: Qwen + FlashAttention 2 ONERATUM")
    except Exception as e:
        print(f"Status: ERROR - {e}")

if __name__ == "__main__":
    check()
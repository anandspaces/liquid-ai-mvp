"""
Liquid AI LFM2-1.2B Model Inference (CPU-only)
Supports Transformers backend for direct inference
"""

import argparse
from typing import List
import os


def run_transformers(prompt: str):
    """Run inference using HuggingFace Transformers (CPU-only)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("Loading model with Transformers (CPU-only)...")
    
    # Check if local model exists, otherwise use HF model ID
    local_model_path = "/models/LFM2-1.2B"
    if os.path.exists(local_model_path):
        model_id = local_model_path
        print(f"Using local model: {model_id}")
    elif os.path.exists("LFM2-1.2B"):
        model_id = "LFM2-1.2B"
        print(f"Using local model: {model_id}")
    else:
        model_id = "LiquidAI/LFM2-1.2B"
        print(f"Using HuggingFace model: {model_id}")
    
    # Load model (CPU-only settings)
    print("Loading model... (this may take 1-2 minutes)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("✅ Model loaded successfully on CPU")
    
    print(f"\nPrompt: {prompt}\n")
    print("Generating response... (this will take 10-60 seconds on CPU)")
    
    # Prepare input
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )
    
    # Generate
    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=256,  # Reduced for faster CPU inference
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode and print
    result = tokenizer.decode(output[0], skip_special_tokens=False)
    print("\n" + "="*80)
    print("RESPONSE:")
    print("="*80)
    print(result)
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Liquid AI LFM2-1.2B model inference (CPU-only)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is C. elegans?",
        help="Prompt for inference",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256, recommended for CPU)",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Liquid AI LFM2-1.2B - CPU Inference")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print("="*80 + "\n")
    
    run_transformers(args.prompt)


if __name__ == "__main__":
    main()
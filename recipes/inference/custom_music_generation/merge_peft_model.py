# merge_peft_model.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
import os
from peft import PeftModel
from transformers import LlamaConfig, LlamaTokenizer

# Make sure the 'generation_conditioned.py' file is in a place where Python can find it.
# This script needs to import your custom model architecture.
from generation_conditioned import LlamaForCausalLM_Conditional_Generation

def main(
    base_model_path: str,
    tokenizer_path: str,
    base_model_weights_path: str,
    peft_model_path: str,
    output_dir: str
):
    """
    Merges the weights of a PEFT adapter (like LoRA) into a custom base model.
    This is a one-time operation to create a final, standalone model for inference.

    Args:
        base_model_path (str): Path to the *folder* containing the config.json of the base model.
        tokenizer_path (str): Path to the tokenizer file (e.g., 'path/to/tokenizer.model').
        base_model_weights_path (str): Direct path to the .pth file of the base model weights.
        peft_model_path (str): Path to the folder containing the PEFT adapter weights.
        output_dir (str): Folder where the final merged model will be saved.
    """
    print("--- Starting PEFT Model Merge ---")

    # 1. Load configuration and tokenizer from their specific paths
    print(f"Loading base model config from: {base_model_path}")
    config = LlamaConfig.from_pretrained(base_model_path)
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    # 2. Initialize custom model structure
    print("Initializing custom model structure (LlamaForCausalLM_Conditional_Generation)...")
    model = LlamaForCausalLM_Conditional_Generation(config)

    # 3. Load base model weights
    # We expect a "missing keys" warning here, as the base weights don't include
    # the new conditioning layers. This is normal and will be fixed by the merge.
    print(f"Loading base model weights from: {base_model_weights_path}")
    print("NOTE: Expect a 'missing keys' warning here. This is normal for this step.")
    
    state_dict = torch.load(base_model_weights_path, map_location="cpu")
    
    # Handle checkpoints that might be saved inside a 'model_state_dict' key
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Clean up 'module.' prefix if the model was trained with DataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
            
    model.load_state_dict(cleaned_state_dict, strict=False)
    print("Base weights loaded successfully.")

    # 4. Load PEFT adapter
    print(f"Loading PEFT (LoRA) adapter from: {peft_model_path}")
    model = PeftModel.from_pretrained(model, peft_model_path)
    print("PEFT adapter loaded.")

    # 5. Merge the weights
    print("Merging PEFT weights into the base model...")
    model = model.merge_and_unload()
    print("Merge complete.")

    # 6. Save the final, complete model
    print(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n--- ✅ Success! ---")
    print(f"Your merged model is saved in: {output_dir}")
    print("You can now use this path in the generation script.")


if __name__ == "__main__":
    fire.Fire(main)

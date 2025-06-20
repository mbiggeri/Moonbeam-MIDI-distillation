# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

def main(base_model: str,
         peft_model: str,
         output_dir: str,
         base_model_weights: str = None):
    """
    Esegue il merge dei pesi di un adapter PEFT (come LoRA) in un modello base.

    Args:
        base_model (str): Percorso della *cartella* che contiene il file config.json del modello base.
        peft_model (str): Percorso della cartella contenente i pesi dell'adapter PEFT.
        output_dir (str): Cartella dove salvare il modello finale con i pesi uniti.
        base_model_weights (str, optional): 
            Percorso diretto al file dei pesi del modello base (es. .pt o .bin). 
            Usare se i pesi non hanno un nome standard come 'pytorch_model.bin'.
    """
    
    # Carica il tokenizer dalla cartella del modello base
    print(f"Loading tokenizer from {base_model}")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    if base_model_weights:
        # --- Logica Modificata: Caricamento da un file di pesi custom ---
        print(f"Loading base model config from {base_model}")
        config = LlamaConfig.from_pretrained(base_model)

        print("Initializing base model structure with the provided config...")
        # Crea un modello con la struttura corretta ma pesi non inizializzati
        model = LlamaForCausalLM(config)
        
        print(f"Loading custom base model weights from: {base_model_weights}")
        # Carica lo state_dict (i pesi) dal tuo file .pt
        state_dict = torch.load(base_model_weights, map_location="cpu")
        
        # Carica i pesi nel modello
        model.load_state_dict(state_dict, strict=True)
        print("Base model loaded successfully from custom weights.")
        
        # Sposta il modello sul dispositivo corretto (GPU se disponibile)
        model.to(device="auto")

    else:
        # --- Logica Originale: Caricamento standard da una cartella Hugging Face ---
        print(f"Loading base model from standard directory: {base_model}")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="tmp",
        )

    # Carica il modello LoRA (adapter) sopra il modello base
    print(f"Loading PEFT (LoRA) model from {peft_model}")
    model = PeftModel.from_pretrained(
        model,
        peft_model,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    # Esegui il merge
    print("Merging PEFT weights into the base model...")
    model = model.merge_and_unload()
    
    # Salva il modello finale completo
    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Merge complete.")


if __name__ == "__main__":
    fire.Fire(main)
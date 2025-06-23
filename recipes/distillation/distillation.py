# RECIPIENTE PER LA DISTILLAZIONE
# Questo script è stato adattato da finetuning.py per eseguire la distillazione della conoscenza.
# La distillazione trasferisce la conoscenza da un modello grande (teacher) a uno più piccolo (student).

import sys
import os
import fire
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Aggiungi il percorso corretto per trovare il modulo 'generation'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference', 'custom_music_generation')))

# Importazioni da Transformers e PEFT
from transformers import LlamaForCausalLM, LlamaConfig
from peft import prepare_model_for_kbit_training

# Importazioni dal progetto llama-recipes
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.datasets.lakh_dataset import LakhDataset
from llama_recipes.utils.config_utils import get_distillation_configs, set_seed, get_dataloader_kwargs

# Importazione della classe custom per il caricamento del teacher
from generation import MusicLlama

# Ignora gli avvisi di performance
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =================================================================================
# FUNZIONE DI TRAINING PER LA DISTILLAZIONE (INTEGRATA E CON STAMPE DI DEBUG)
# =================================================================================

def distillation_loss_fn(student_logits, teacher_logits, labels, alpha, temperature):
    """
    Calcola la loss di distillazione combinando la soft loss (KL Divergence) 
    e la hard loss (Cross-Entropy).
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.log_softmax(teacher_logits / temperature, dim=-1),
        log_target=True,
        reduction='batchmean'
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)), 
        labels.view(-1), 
        ignore_index=-100
    )

    total_loss = alpha * soft_loss + (1. - alpha) * hard_loss
    return total_loss, hard_loss

def train_distillation_with_debug(student_model, teacher_model, train_dataloader, eval_dataloader, tokenizer, optimizer, scheduler, train_config):
    """
    Funzione di training basata su quella fornita, ma con stampe di debug dettagliate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Avvio training sul dispositivo: {device} ---")
    
    for epoch in range(train_config.num_epochs):
        print(f"\n{'='*25} Inizio Epoca {epoch+1}/{train_config.num_epochs} {'='*25}")
        student_model.train()
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Training]", leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            
            # --- DEBUG: Forward pass del Teacher ---
            print(f"  [Batch {step+1}] Eseguendo forward pass del Teacher...", end='', flush=True)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                teacher_logits = teacher_outputs.logits
            print(" Fatto.")

            # --- DEBUG: Forward pass dello Student ---
            print(f"  [Batch {step+1}] Eseguendo forward pass dello Student...", end='', flush=True)
            student_outputs = student_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            student_logits = student_outputs.logits
            print(" Fatto.")

            # --- Calcolo della loss ---
            loss, hard_loss = distillation_loss_fn(
                student_logits,
                teacher_logits,
                labels,
                alpha=train_config.alpha,
                temperature=train_config.temperature
            )
            
            loss = loss / train_config.gradient_accumulation_steps
            
            # --- DEBUG: Backpropagation ---
            print(f"  [Batch {step+1}] Eseguendo backward pass...", end='', flush=True)
            loss.backward()
            print(" Fatto.")

            # Aggiornamento pesi
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix({"Loss": loss.item() * train_config.gradient_accumulation_steps, "Hard Loss": hard_loss.item()})

        scheduler.step()
        
        # --- Ciclo di validazione (invariato) ---
        student_model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=f"Epoch {epoch+1} [Validation]", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                outputs = student_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), ignore_index=-100)
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"--- Fine Validazione Epoca {epoch+1} - Average Validation Loss: {avg_eval_loss:.4f} ---")

        if train_config.save_model:
            output_dir = os.path.join(train_config.output_dir, f"epoch_{epoch+1}")
            os.makedirs(output_dir, exist_ok=True)
            student_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Modello salvato in {output_dir}")
            
    return {"final_validation_loss": avg_eval_loss}

def main(**kwargs):
    # Ottiene le configurazioni ESATTAMENTE COME PRIMA
    (
        model_configs,
        train_config,
        fsdp_config,
        lora_config,
        llama_adapter_config,
        prefix_config,
        quantization_config, # Questo viene già generato dalla tua funzione se passi --quantization
        dataset_config,
        wandb_config,
    ) = get_distillation_configs(**kwargs)

    set_seed(train_config.seed)

    # --- CARICAMENTO DEL MODELLO TEACHER (logica invariata) ---
    print(f"Caricamento del modello Teacher da checkpoint: {train_config.teacher_model_checkpoint}")
    teacher_llama = MusicLlama.build(
        ckpt_dir=train_config.teacher_model_checkpoint,
        model_config_path=train_config.teacher_model_config,
        tokenizer_path=train_config.tokenizer_path,
        max_seq_len=train_config.context_length,
        max_batch_size=train_config.val_batch_size,
        finetuned_PEFT_weight_path=None,
    )
    teacher_model = teacher_llama.model
    teacher_model.eval()
    print("Modello Teacher caricato con successo.")

    # --- CARICAMENTO TOKENIZER E MODELLO STUDENT ---
    tokenizer = teacher_llama.tokenizer
    
    print(f"Creazione del modello Student da configurazione: {kwargs.get('model_config_file')}")
    
    # *** MODIFICA NECESSARIA PER LA QUANTIZZAZIONE ***
    # Invece di creare il modello da una config vuota, usiamo from_pretrained.
    # Questo permette di passare il 'quantization_config' che la tua funzione già crea.
    # Se quantization_config è None, il modello viene caricato normalmente.
    student_config_dir = os.path.dirname(kwargs.get('model_config_file'))
    student_model = LlamaForCausalLM.from_pretrained(
        student_config_dir, # Legge il config.json da questa cartella
        quantization_config=quantization_config # APPLICA LA QUANTIZZAZIONE SE ATTIVATA
    )
    
    if quantization_config:
        print("Quantizzazione attivata. Preparazione del modello per il training k-bit.")
        student_model = prepare_model_for_kbit_training(student_model)
    else:
        print("Nessuna quantizzazione specificata. Modello caricato con precisione standard.")
        student_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Modello Student creato con successo.")
    print("-" * 50)
    print("Dettagli del Modello Student:")
    print(f"  -> Parametri: {sum(p.numel() for p in student_model.parameters()) / 1_000_000:.2f} Milioni")
    print(f"  -> Precisione: {next(student_model.parameters()).dtype}")
    print("-" * 50)
    
    # --- PREPARAZIONE DATASET (invariato) ---
    train_dataset = LakhDataset(dataset_config=train_config, tokenizer=tokenizer, partition="train")
    eval_dataset = LakhDataset(dataset_config=train_config, tokenizer=tokenizer, partition="validation")

    train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")
    train_dataloader = DataLoader(train_dataset, **train_dl_kwargs)

    eval_dl_kwargs = get_dataloader_kwargs(train_config, eval_dataset, tokenizer, "val")
    eval_dataloader = DataLoader(eval_dataset, **eval_dl_kwargs)
    
    # --- OTTIMIZZATORE E SCHEDULER (invariato) ---
    optimizer = optim.AdamW(student_model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # --- AVVIO DISTILLAZIONE ---
    print("Avvio della distillazione...")
    results = train_distillation_with_debug(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=train_config,
    )
    
    print("Distillazione completata.")
    print(f"Risultati: {results}")

if __name__ == "__main__":
    fire.Fire(main)
    
'''
python distillation.py \
    # --- Percorsi Obbligatori (gli stessi di prima) ---
    --model_config_file="/percorso/del/tuo/progetto/configs/student_model_config.json" \
    --teacher_model_checkpoint="/percorso/del/tuo/progetto/checkpoints/teacher_model/checkpoint.pt" \
    --teacher_model_config="/percorso/del/tuo/progetto/checkpoints/teacher_model/config.json" \
    --tokenizer_path="/percorso/del/tuo/progetto/tokenizer/" \
    --output_dir="/percorso/del/tuo/progetto/distilled_models/" \
    \
    # --- PER ATTIVARE LA QUANTIZZAZIONE A 4-BIT (secondo la tua funzione get_quantization_config) ---
    --quantization=True \
    \
    # --- Parametri di Addestramento e Distillazione (gli stessi di prima) ---
    --num_epochs=3 \
    --lr=0.0002 \
    --batch_size_training=4 \
    --gradient_accumulation_steps=8 \
    --alpha=0.5 \
    --temperature=2.0 \
    \
    # --- Parametri del Dataset (gli stessi di prima) ---
    --dataset="lakh_dataset" \
    --csv_dataset_path="/percorso/del/tuo/progetto/data/dataset_split.csv" \
    --file_column_name="file_path"
'''
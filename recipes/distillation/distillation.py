# RECIPIENTE PER LA DISTILLAZIONE
# Questo script è stato adattato da finetuning.py per eseguire la distillazione della conoscenza.
# La distillazione trasferisce la conoscenza da un modello grande (teacher) a uno più piccolo (student).

import sys
import os
import json
import fire
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Aggiungi il percorso corretto per trovare il modulo 'generation'
# Assicurati che questo percorso sia corretto per la tua struttura di directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference', 'custom_music_generation')))

# Importazioni da Transformers e PEFT
from transformers import LlamaForCausalLM, LlamaConfig, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Importazioni dal progetto llama-recipes (adatta i percorsi se necessario)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.datasets.lakh_dataset import LakhDataset
from llama_recipes.utils.config_utils import get_distillation_configs, set_seed, get_dataloader_kwargs
# La funzione train_distillation è ora inclusa in questo file per debug

# Importazione della classe custom per il caricamento del teacher
from generation import MusicLlama

# Ignora gli avvisi di performance
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =================================================================================
# FUNZIONE DI TRAINING PER LA DISTILLAZIONE (CON STAMPE DI DEBUG)
# =================================================================================

def distillation_loss_fn(student_logits, teacher_logits, labels, alpha, temperature):
    """
    Calcola la loss di distillazione combinando la soft loss (KL Divergence) 
    e la hard loss (Cross-Entropy).
    """
    # Soft loss: KL Divergence tra i logits "ammorbiditi"
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.log_softmax(teacher_logits / temperature, dim=-1),
        log_target=True,
        reduction='batchmean'
    ) * (temperature ** 2)

    # Hard loss: Cross-Entropy standard con le etichette reali
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)), 
        labels.view(-1), 
        ignore_index=-100
    )

    # Combina le due loss
    total_loss = alpha * soft_loss + (1. - alpha) * hard_loss
    return total_loss, hard_loss


def train_distillation(student_model, teacher_model, train_dataloader, eval_dataloader, tokenizer, optimizer, scheduler, train_config):
    """
    Funzione principale per il ciclo di addestramento della distillazione con stampe di debug.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Avvio training sul dispositivo: {device} ---")
    
    for epoch in range(train_config.num_epochs):
        print(f"\n{'='*25} Inizio Epoca {epoch+1}/{train_config.num_epochs} {'='*25}")
        student_model.train()
        total_loss = 0.0
        
        # Ciclo di addestramento
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Training]", leave=False)
        for step, batch in enumerate(pbar):
            # Sposta i dati sul dispositivo
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            
            # --- 1. Forward pass del Teacher ---
            print(f"  [Batch {step+1}] Eseguendo forward pass del Teacher...", end='', flush=True)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                teacher_logits = teacher_outputs.logits
            print(" Fatto.")

            # --- 2. Forward pass dello Student ---
            print(f"  [Batch {step+1}] Eseguendo forward pass dello Student...", end='', flush=True)
            student_outputs = student_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            student_logits = student_outputs.logits
            print(" Fatto.")

            # --- 3. Calcolo della loss ---
            loss, hard_loss = distillation_loss_fn(
                student_logits,
                teacher_logits,
                labels,
                alpha=train_config.alpha,
                temperature=train_config.temperature
            )
            
            # Normalizza la loss per l'accumulo dei gradienti
            loss = loss / train_config.gradient_accumulation_steps
            
            # --- 4. Backpropagation ---
            print(f"  [Batch {step+1}] Eseguendo backward pass...", end='', flush=True)
            loss.backward()
            print(" Fatto.")

            # Aggiorna i pesi ogni `gradient_accumulation_steps`
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                print(f"  [Batch {step+1}] Aggiornando i pesi dello Student...", end='', flush=True)
                optimizer.step()
                optimizer.zero_grad()
                print(" Fatto.")
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item() * train_config.gradient_accumulation_steps, "Hard Loss": hard_loss.item()})

        # Aggiorna lo scheduler
        scheduler.step()
        
        # --- Ciclo di validazione ---
        student_model.eval()
        eval_loss = 0.0
        print(f"\n--- Inizio Validazione Epoca {epoch+1} ---")
        with torch.no_grad():
            pbar_eval = tqdm(eval_dataloader, desc=f"Epoch {epoch+1} [Validation]", leave=False)
            for batch in pbar_eval:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                
                outputs = student_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), ignore_index=-100)
                eval_loss += loss.item()
                pbar_eval.set_postfix({"Eval Loss": loss.item()})
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"--- Fine Validazione Epoca {epoch+1} - Average Validation Loss: {avg_eval_loss:.4f} ---")

        # Salva un checkpoint del modello
        if train_config.save_model:
            output_dir = os.path.join(train_config.output_dir, f"epoch_{epoch+1}")
            os.makedirs(output_dir, exist_ok=True)
            student_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Modello salvato in {output_dir}")
            
    return {"final_validation_loss": avg_eval_loss}


# =================================================================================
# FUNZIONE MAIN
# =================================================================================

def main(**kwargs):
    # Ottiene le configurazioni per il modello, l'addestramento, ecc.
    # NOTA: get_distillation_configs ora gestisce solo il caricamento da file.
    # La logica di quantizzazione è spostata qui per maggiore chiarezza.
    (
        model_configs,
        train_config,
        _, # fsdp_config
        _, # lora_config
        _, # llama_adapter_config
        _, # prefix_config
        _, # quantization_config (non più usato da qui)
        dataset_config,
        _, # wandb_config
    ) = get_distillation_configs(**kwargs)

    # Imposta la seed per la riproducibilità
    set_seed(train_config.seed)
    
    # ----------------------------------------------------------------------
    # 1. GESTIONE DELLA QUANTIZZAZIONE
    # ----------------------------------------------------------------------
    quantization_config_obj = None
    if hasattr(train_config, 'quantization_bits') and train_config.quantization_bits in [4, 8]:
        print(f"Abilitando la quantizzazione a {train_config.quantization_bits}-bit...")
        if train_config.quantization_bits == 8:
            quantization_config_obj = BitsAndBytesConfig(load_in_8bit=True)
        elif train_config.quantization_bits == 4:
            quantization_config_obj = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
    else:
        print("Nessuna quantizzazione specificata. Il modello verrà caricato con precisione standard.")

    # ----------------------------------------------------------------------
    # 2. CARICAMENTO DEI MODELLI E TOKENIZER
    # ----------------------------------------------------------------------
    
    # --- CARICAMENTO DEL MODELLO TEACHER ---
    # Nota: Il teacher è solitamente troppo grande per essere quantizzato senza perdita di informazioni.
    # Lo carichiamo con la sua precisione nativa (gestita da MusicLlama.build).
    print(f"Caricamento del modello Teacher da checkpoint: {train_config.teacher_model_checkpoint}")
    teacher_llama = MusicLlama.build(
        ckpt_dir=train_config.teacher_model_checkpoint, #
        model_config_path=train_config.teacher_model_config,
        tokenizer_path=train_config.tokenizer_path,
        max_seq_len=train_config.context_length,
        max_batch_size=train_config.val_batch_size,
        finetuned_PEFT_weight_path=None,
    )
    teacher_model = teacher_llama.model
    teacher_model.eval()
    print("Modello Teacher caricato con successo.")
    
    # --- CARICAMENTO DEL TOKENIZER E MODELLO STUDENT ---
    tokenizer = teacher_llama.tokenizer
    
    print(f"Creazione del modello Student da configurazione: {kwargs.get('model_config_file')}")
    # Per caricare con quantizzazione, usiamo from_pretrained sulla cartella della config
    student_config_dir = os.path.dirname(kwargs.get('model_config_file'))
    student_model = LlamaForCausalLM.from_pretrained(
        student_config_dir,
        quantization_config=quantization_config_obj, # Applica la quantizzazione qui
        torch_dtype=torch.bfloat16 if quantization_config_obj else "auto" # Usa bfloat16 per calcoli se quantizzato
    )
    
    if quantization_config_obj:
        # Prepara il modello per il training k-bit (necessario per la quantizzazione)
        student_model = prepare_model_for_kbit_training(student_model)

    print("Modello Student creato con successo.")
    print("-" * 50)
    print("Dettagli del Modello Student:")
    print(f"  -> Parametri: {sum(p.numel() for p in student_model.parameters()) / 1_000_000:.2f} Milioni")
    print(f"  -> Precisione: {next(student_model.parameters()).dtype}")
    print(f"  -> Device: {next(student_model.parameters()).device}")
    print("-" * 50)

    # ----------------------------------------------------------------------
    # 3. PREPARAZIONE DEL DATASET E DATALOADER
    # ----------------------------------------------------------------------
    train_dataset = LakhDataset(dataset_config=train_config, tokenizer=tokenizer, partition="train")
    eval_dataset = LakhDataset(dataset_config=train_config, tokenizer=tokenizer, partition="validation")

    train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")
    train_dataloader = DataLoader(train_dataset, **train_dl_kwargs)

    eval_dl_kwargs = get_dataloader_kwargs(train_config, eval_dataset, tokenizer, "val")
    eval_dataloader = DataLoader(eval_dataset, **eval_dl_kwargs)
    
    # ----------------------------------------------------------------------
    # 4. IMPOSTAZIONE DELL'OTTIMIZZATORE E SCHEDULER
    # ----------------------------------------------------------------------
    optimizer = optim.AdamW(student_model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # ----------------------------------------------------------------------
    # 5. AVVIO DEL PROCESSO DI DISTILLAZIONE
    # ----------------------------------------------------------------------
    print("Avvio della distillazione...")
    results = train_distillation(
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
    print(f"Risultati Finali: {results}")

if __name__ == "__main__":
    fire.Fire(main)

'''
---------------------------------------- USAGE (AGGIORNATO):
python distillation.py \
    # --- Percorsi Obbligatori ---
    --model_config_file="/percorso/del/tuo/progetto/configs/student_model_config.json" \
    --teacher_model_checkpoint="/percorso/del/tuo/progetto/checkpoints/teacher_model/checkpoint.pt" \
    --teacher_model_config="/percorso/del/tuo/progetto/checkpoints/teacher_model/config.json" \
    --tokenizer_path="/percorso/del/tuo/progetto/tokenizer/" \
    --output_dir="/percorso/del/tuo/progetto/distilled_models/" \
    \
    # --- NUOVO: Parametro per la Quantizzazione (SCEGLIERE UNO) ---
    # Per quantizzazione a 4-bit (consigliato per VRAM molto bassa):
    --quantization_bits=4 \
    # Per quantizzazione a 8-bit:
    # --quantization_bits=8 \
    # Per nessuna quantizzazione (comportamento originale, richiede molta VRAM):
    # (non includere l'argomento)
    \
    # --- Parametri di Addestramento ---
    --num_epochs=3 \
    --lr=0.0002 \
    --batch_size_training=4 \
    --val_batch_size=4 \
    --gradient_accumulation_steps=8 \
    \
    # --- Parametri di Distillazione ---
    # Bilanciamento tra soft loss (teacher) e hard loss (dati reali). Valore tra 0 e 1.
    --alpha=0.5 \
    # "Temperatura" per ammorbidire i logits. Valori comuni: 2-5.
    --temperature=2.0 \
    \
    # --- Parametri del Dataset ---
    --dataset="lakh_dataset" \
    --csv_dataset_path="/percorso/del/tuo/progetto/data/dataset_split.csv" \
    --file_column_name="file_path"
'''
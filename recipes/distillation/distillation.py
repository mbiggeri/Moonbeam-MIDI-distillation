# RECIPIENTE PER LA DISTILLAZIONE
# Questo script è stato adattato da finetuning.py per eseguire la distillazione della conoscenza.
# La distillazione trasferisce la conoscenza da un modello grande (teacher) a uno più piccolo (student).

import sys
import os
# Aggiungi il percorso corretto per trovare il modulo 'generation'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inference', 'custom_music_generation')))

import fire
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# <-- CORREZIONE: Importiamo sia LlamaConfig che il MusicTokenizer personalizzato.
from transformers import LlamaForCausalLM, LlamaConfig
from peft import get_peft_model, prepare_model_for_kbit_training

from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.datasets.lakh_dataset import LakhDataset
from llama_recipes.utils.config_utils import get_distillation_configs, set_seed, get_dataloader_kwargs
from llama_recipes.utils.train_utils import train_distillation

# Ignora gli avvisi di performance di UserWarning, comuni durante la distillazione
import warnings
import warnings
from generation import MusicLlama # <-- PASSO 1: IMPORTA LA CLASSE CORRETTA
from transformers import LlamaConfig # Rimuovi LlamaForCausalLM se non serve altrove

warnings.filterwarnings("ignore", category=UserWarning)

def main(**kwargs):
    # Ottiene le configurazioni per il modello, l'addestramento, ecc.
    (
        model_configs,
        train_config,
        fsdp_config,
        lora_config,
        llama_adapter_config,
        prefix_config,
        quantization_config,
        dataset_config,
        wandb_config,
    ) = get_distillation_configs(**kwargs)

    # Imposta la seed per la riproducibilità
    set_seed(train_config.seed)

    # ----------------------------------------------------------------------
    # 2. CARICAMENTO DEI MODELLI E TOKENIZER
    # ----------------------------------------------------------------------
    
    # --- CARICAMENTO DEL MODELLO TEACHER (METODO CORRETTO) ---
    print(f"Caricamento del modello Teacher da checkpoint: {train_config.teacher_model_checkpoint}")
    teacher_llama = MusicLlama.build(
        # Usiamo 'model_name' per il checkpoint del teacher
        ckpt_dir=train_config.model_name,
        # Il percorso della config del teacher rimane un parametro a parte,
        # perché non ha un equivalente ovvio nella tua config attuale.
        model_config_path=train_config.teacher_model_config,
        # Usiamo 'tokenizer_name' per il percorso del tokenizer
        tokenizer_path=train_config.tokenizer_name,
        # Usiamo 'context_length' per la lunghezza della sequenza
        max_seq_len=train_config.context_length,
        max_batch_size=train_config.val_batch_size,
        finetuned_PEFT_weight_path=None,
    )
    teacher_model = teacher_llama.model
    teacher_model.eval()
    print("Modello Teacher caricato con successo.")
    
    # Stampa i dettagli del modello teacher
    print("-" * 50)
    print("Dettagli del Modello Teacher Caricato:")
    try:
        # Estrai la configurazione per i dettagli
        teacher_config = teacher_llama.config
        # Calcola il numero totale di parametri
        total_params = sum(p.numel() for p in teacher_model.parameters())
        
        # Stampa i dettagli formattati
        print(f"  -> Architettura:        {teacher_model.__class__.__name__}")
        print(f"  -> Numero di Parametri:   {total_params / 1_000_000:.2f} Milioni")
        print(f"  -> Numero di Layer:         {teacher_config.num_hidden_layers}")
        print(f"  -> Dimensione Nascosta:     {teacher_config.hidden_size}")
        print(f"  -> Device di Esecuzione:  {next(teacher_model.parameters()).device}")
        print(f"  -> Precisione (dtype):    {next(teacher_model.parameters()).dtype}")
    except Exception as e:
        print(f"  -> Impossibile recuperare i dettagli del modello: {e}")
    print("-" * 50)

    # --- CARICAMENTO DEL TOKENIZER E MODELLO STUDENT ---
    
    # Ora puoi ottenere il tokenizer direttamente dall'oggetto teacher_llama
    # Questo garantisce che sia esattamente lo stesso.
    tokenizer = teacher_llama.tokenizer
    
    # Il caricamento del modello student è corretto, perché lo crei da zero
    print(f"Creazione del modello Student da configurazione: {kwargs.get('model_config_file')}")
    student_config = LlamaConfig(**model_configs)
    student_model = LlamaForCausalLM(student_config)
        
    student_model = prepare_model_for_kbit_training(student_model)
    student_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Modello Student caricato con successo.")

    # ----------------------------------------------------------------------
    # 3. PREPARAZIONE DEL DATASET E DATALOADER
    # ----------------------------------------------------------------------
    train_dataset = LakhDataset(
        dataset_config=train_config,
        tokenizer=tokenizer,
        partition="train"
    )
    
    eval_dataset = LakhDataset(
        dataset_config=train_config,
        tokenizer=tokenizer,
        partition="validation"
    )

    # Usa la funzione di supporto per ottenere i parametri corretti, incluso il collate_fn per il padding
    train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")
    train_dataloader = DataLoader(
        train_dataset,
        **train_dl_kwargs
    )

    eval_dl_kwargs = get_dataloader_kwargs(train_config, eval_dataset, tokenizer, "val")
    eval_dataloader = DataLoader(
        eval_dataset,
        **eval_dl_kwargs
    )
    
    # ----------------------------------------------------------------------
    # 4. IMPOSTAZIONE DELL'OTTIMIZZATORE E SCHEDULER
    # ----------------------------------------------------------------------
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay
    )
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
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        train_config=train_config,
    )
    
    print("Distillazione completata.")
    print(f"Risultati: {results}")

if __name__ == "__main__":
    fire.Fire(main)
    
'''
---------------------------------------- USAGE:
python distillation.py \
    # --- Percorsi Obbligatori ---
    # Percorso al file di configurazione JSON per il NUOVO modello student
    --model_config_file="/percorso/del/tuo/progetto/configs/student_model_config.json" \
    \
    # --- Parametri per il caricamento del modello Teacher (OBBLIGATORI per la nuova logica) ---
    # Percorso al checkpoint .pt del modello teacher grande
    --teacher_model_checkpoint="/percorso/del/tuo/progetto/checkpoints/teacher_model/checkpoint.pt" \
    # Percorso al file config.json del modello teacher
    --teacher_model_config="/percorso/del/tuo/progetto/checkpoints/teacher_model/config.json" \
    # Percorso al tokenizer (probabilmente condiviso)
    --tokenizer_path="/percorso/del/tuo/progetto/tokenizer/" \
    \
    # --- Parametri di Addestramento (obbligatori e facoltativi) ---
    # Dove salvare i checkpoint del modello student distillato
    --output_dir="/percorso/del/tuo/progetto/distilled_models/" \
    # Numero di epoche di addestramento
    --num_epochs=3 \
    # Learning rate
    --lr=0.0002 \
    # Dimensione del batch per l'addestramento
    --batch_size_training=8 \
    # Dimensione del batch per la validazione
    --batch_size_eval=8 \
    # Numero di passaggi per accumulare il gradiente prima di un aggiornamento
    --gradient_accumulation_steps=4 \
    \
    # --- Parametri del Modello e del Dataset (facoltativi, con valori di default) ---
    # Lunghezza massima della sequenza
    --max_seq_len=512 \
    # Seed per la riproducibilità
    --seed=42 \
    # Fattore di decadimento del learning rate (per StepLR)
    --gamma=0.85 \
    # Weight decay per l'ottimizzatore AdamW
    --weight_decay=0.01 \
    # Abilita/disabilita il gradient checkpointing per risparmiare memoria
    --use_gradient_checkpointing=True \
    \
    # --- Parametri per il Dataset (da adattare in base al tuo LakhDataset) ---
    # Percorso al file CSV che definisce il dataset
    --csv_dataset_path="/percorso/del/tuo/progetto/data/dataset_split.csv" \
    # Nome della colonna che contiene i percorsi dei file
    --file_column_name="file_path"
'''
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# RECIPIENTE PER LA DISTILLAZIONE
# Questo script è stato adattato da finetuning.py per eseguire la distillazione della conoscenza.
# La distillazione trasferisce la conoscenza da un modello grande (teacher) a uno più piccolo (student).

import os
import sys
import fire
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from peft import get_peft_model, prepare_model_for_kbit_training

from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.datasets.lakh_dataset import LakhDataset
from llama_recipes.utils.config_utils import get_distillation_configs
from llama_recipes.utils.train_utils import set_seed, train_distillation

# Ignora gli avvisi di performance di UserWarning, comuni durante la distillazione
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(**kwargs):
    # Ottiene le configurazioni per il modello, l'addestramento, ecc.
    (
        model_configs,
        train_config,
        _, # fsdp_config
        _, # ddp_config
        _, # dataset_config
    ) = get_distillation_configs(**kwargs)

    # Imposta la seed per la riproducibilità
    set_seed(train_config.seed)

    # Carica il tokenizer (lo stesso per teacher e student)
    # Assicura che la tokenizzazione sia consistente
    tokenizer = MusicTokenizer()
    
    # ----------------------------------------------------------------------
    # 1. CARICAMENTO DEL MODELLO TEACHER
    # ----------------------------------------------------------------------
    print(f"Caricamento del modello Teacher: {train_config.teacher_model_name}")
    # Il teacher è caricato in modalità di valutazione e non richiede calcolo del gradiente
    teacher_model = LlamaForCausalLM.from_pretrained(
        train_config.teacher_model_name,
        return_dict=True,
        load_in_8bit=True, # Usa la quantizzazione per ridurre l'uso della memoria
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    teacher_model.eval()
    print("Modello Teacher caricato con successo.")

    # ----------------------------------------------------------------------
    # 2. CARICAMENTO DEL MODELLO STUDENT
    # ----------------------------------------------------------------------
    student_config = LlamaConfig.from_pretrained(model_configs["student"])
    
    print(f"Caricamento del modello Student: {model_configs['student']}")
    student_model = LlamaForCausalLM(student_config)

    # Abilita il gradient checkpointing per risparmiare memoria
    if train_config.use_gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        
    student_model = prepare_model_for_kbit_training(student_model)
    
    # Sposta i modelli sul dispositivo corretto (es. GPU)
    # Il teacher è già su device_map="auto"
    student_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Modello Student caricato con successo.")


    # ----------------------------------------------------------------------
    # 3. PREPARAZIONE DEL DATASET E DATALOADER
    # ----------------------------------------------------------------------
    # Utilizza il dataset specificato
    # Nota: Assicurati che i percorsi e i parametri del dataset siano corretti.
    train_dataset = LakhDataset(
        csv_path=train_config.csv_path,
        data_path=train_config.data_path,
        split="train",
        tokenizer=tokenizer,
        max_words=train_config.max_token_length,
    )
    
    eval_dataset = LakhDataset(
        csv_path=train_config.csv_path,
        data_path=train_config.data_path,
        split="validation",
        tokenizer=tokenizer,
        max_words=train_config.max_token_length,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
    )
    
    # ----------------------------------------------------------------------
    # 4. IMPOSTAZIONE DELL'OTTIMIZZATORE E SCHEDULER
    # ----------------------------------------------------------------------
    # L'ottimizzatore viene applicato solo ai parametri dello student
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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# --- MODIFICA INIZIO ---
import json
from types import SimpleNamespace
# --- MODIFICA FINE ---

import inspect
import random
import numpy as np
import torch
from dataclasses import asdict

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq


# Import delle altre configurazioni (che rimangono classi Python)
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs.peft import lora_config, llama_adapter_config, prefix_config
from llama_recipes.configs.datasets import samsum_dataset, grammar_dataset, alpaca_dataset, custom_dataset, lakhmidi_dataset, merge_dataset, emophia_con_gen_dataset, commu_con_gen_dataset
from llama_recipes.configs.wandb import wandb_config
from llama_recipes.utils.dataset_utils import DATASET_PREPROC


def set_seed(seed):
    """
    Imposta il seed per la riproducibilità.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_config(config_objects, **kwargs):
    """
    Aggiorna gli oggetti di configurazione con i valori forniti da kwargs.
    """
    if isinstance(config_objects, tuple):
        for config in config_objects:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    else:
         for key, value in kwargs.items():
            if hasattr(config_objects, key):
                setattr(config_objects, key, value)
    return config_objects

def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    if train_config.peft_method not in names:
        raise RuntimeError(f"Peft config not found: {train_config.peft_method}")

    if train_config.peft_method == "prefix":
        raise RuntimeError("PrefixTuning is currently not supported (see https://github.com/meta-llama/llama-recipes/issues/359#issuecomment-2089350811)")

    if train_config.enable_fsdp and train_config.peft_method == "llama_adapter":
        raise RuntimeError("Llama_adapter is currently not supported in combination with FSDP (see https://github.com/meta-llama/llama-recipes/issues/359#issuecomment-2089274425)")

    config = configs[names.index(train_config.peft_method)]()

    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config

def get_distillation_configs(
    **kwargs
):
    """
    Ottiene tutte le configurazioni necessarie.
    Questa funzione è stata modificata per caricare dinamicamente la configurazione
    del modello STUDENTE da un file JSON specificato tramite riga di comando.
    """
    # --- LOGICA DI CARICAMENTO DINAMICO DEL JSON ---
    # 1. Carica la configurazione del modello STUDENTE da un file JSON.
    #    Questo sostituisce l'import statico di 'model_config'.
    student_model_config_path = kwargs.get("model_config_file")
    if not student_model_config_path:
        raise ValueError(
            "Per la distillazione è necessario specificare il file di configurazione del modello studente."
            " Usa l'argomento: --model_config_file 'percorso/al/tuo/config_studente.json'"
        )
    
    try:
        with open(student_model_config_path, "r") as f:
            # Carica il JSON e lo converte in un oggetto accessibile con il punto (es. model_configs.hidden_size)
            model_configs = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Il file di configurazione specificato non è stato trovato: {student_model_config_path}")
    # --- FINE LOGICA DI CARICAMENTO ---

    # Carica le altre configurazioni come faceva in origine
    train_config = TRAIN_CONFIG()
    fsdp_config = FSDP_CONFIG()
    lora_config_ = lora_config()
    llama_adapter_config_ = llama_adapter_config()
    prefix_config_ = prefix_config()
    
    dataset_config_map = {
        "samsum_dataset": samsum_dataset,
        "grammar_dataset": grammar_dataset,
        "alpaca_dataset": alpaca_dataset,
        "custom_dataset": custom_dataset,
        "lakhmidi_dataset": lakhmidi_dataset,
        "merge_dataset": merge_dataset,
        "emophia_con_gen_dataset": emophia_con_gen_dataset,
        "commu_con_gen_dataset": commu_con_gen_dataset,
    }
    
    # Aggiorna le configurazioni con gli argomenti passati da riga di comando
    update_config((train_config, fsdp_config, lora_config_, llama_adapter_config_, prefix_config_), **kwargs)
    
    # Seleziona il dataset corretto in base al nome fornito nella configurazione di training
    dataset_config = dataset_config_map.get(train_config.dataset)
    if dataset_config is None:
        raise ValueError(f"Dataset '{train_config.dataset}' non trovato nelle configurazioni.")
    dataset_config = dataset_config()
    
    # Gestisce la quantizzazione
    from llama_recipes.utils.train_utils import get_quantization_config
    quantization_config = get_quantization_config(train_config)
    
    # Crea la configurazione per wandb se richiesto
    wandb_config_ = wandb_config() if train_config.use_wandb else None

    return (
        model_configs,  # L'oggetto caricato dal nostro JSON
        train_config,
        fsdp_config,
        lora_config_,
        llama_adapter_config_,
        prefix_config_,
        quantization_config,
        dataset_config,
        wandb_config_,
    )


# Il resto del file rimane invariato...
def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
    kwargs = {}
    batch_size = train_config.val_batch_size if mode == "val" else train_config.batch_size_training
    if train_config.batching_strategy == "padding":
        if train_config.enable_fsdp:
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
            )
        else:
            kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
    elif train_config.batching_strategy == "packing":
        if train_config.enable_fsdp:
            kwargs["sampler"] = DistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=mode=="train",
            drop_last=True,
        )
        elif train_config.enable_ddp:
            kwargs["sampler"] = DistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=mode=="train",
            drop_last=True,
            )
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
        kwargs["collate_fn"] = default_data_collator
    else:
        raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")

    return kwargs

class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle


    def __iter__(self):
        # get the sequence length of each sample
        seq_lens = [len(x['input_ids']) for x in self.data_source]
        
        #
        # create indices and sort them by sequence length
        indices = list(range(len(self.data_source)))
        
        if self.shuffle:
            # similar to RandomSampler, we shuffle the indices
            random.shuffle(indices)

        # sort indices by sequence length
        indices.sort(key=lambda i: seq_lens[i], reverse=True)
        

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch
            
    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size -1) // self.batch_size


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int =0):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # get the sequence length of each sample
        seq_lens = [len(x['input_ids']) for x in self.data_source]
        
        # create indices and sort them by sequence length
        indices = list(range(len(self.data_source)))
        indices.sort(key=lambda i: seq_lens[i], reverse=True)
        
        # subsample
        indices = indices[self.rank:len(self.data_source):self.num_replicas]

        # shuffle
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.data_source) // (self.batch_size * self.num_replicas)
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
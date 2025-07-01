import json
import os
import fire
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

# --- 1. IMPORTAZIONI CRUCIALI ---
# Importa la configurazione e il modello custom
from transformers import LlamaForCausalLM, LlamaConfig

# Importazioni dal tuo progetto
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.datasets.lakh_dataset import LakhDataset
from llama_recipes.utils.config_utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Dataclass per la configurazione del dataset
@dataclass
class DatasetConfig:
    data_dir: str = None
    csv_file: str = None

def distillation_loss_fn(student_hidden_states, teacher_hidden_states, alpha, student_hard_loss):
    """
    Calcola la loss di distillazione per architetture Encoder-Decoder.
    - La soft loss è un MSE loss sugli stati nascosti intermedi dell'encoder.
    - La hard loss è quella già calcolata dal decoder dello student.
    """
    # Soft loss: Vogliamo che gli "hidden states" dello student siano simili a quelli del teacher
    soft_loss = F.mse_loss(student_hidden_states, teacher_hidden_states)
    
    # Combina le due loss
    total_loss = alpha * soft_loss + (1. - alpha) * student_hard_loss
    return total_loss, soft_loss

def train_distillation(student_model, teacher_model, train_dataloader, eval_dataloader, tokenizer, optimizer, scheduler, train_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Avvio training sul dispositivo: {device} ---")
    
    student_model.to(device)
    teacher_model.to(device)

    for epoch in range(train_config.num_epochs):
        print(f"\n{'='*25} Inizio Epoca {epoch+1}/{train_config.num_epochs} {'='*25}")
        student_model.train()
        teacher_model.eval()
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Training]", leave=False)
        for step, batch in enumerate(pbar):
            # Il batch ora contiene 'input_ids' composti e 'labels' per il decoder
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # --- Calcolo output del Teacher ---
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )
                # Estraiamo gli stati nascosti intermedi per la distillazione.
                # Nel tuo modello, questi sono salvati nell'attributo 'logits' dell'output.
                teacher_hidden_states = teacher_outputs.logits 

            # --- Calcolo output dello Student ---
            student_outputs = student_model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            student_hidden_states = student_outputs.logits
            student_hard_loss = student_outputs.loss # Il modello calcola già la hard loss (CrossEntropy)

            # Calcolo della loss combinata
            if student_hard_loss is not None:
                loss, soft_loss = distillation_loss_fn(
                    student_hidden_states,
                    teacher_hidden_states,
                    alpha=train_config.alpha,
                    student_hard_loss=student_hard_loss
                )
                
                loss = loss / train_config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % train_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                pbar.set_postfix({"Total Loss": loss.item(), "Hard Loss": student_hard_loss.item(), "Soft Loss": soft_loss.item()})
            else:
                 pbar.set_postfix({"Status": "Skipping batch, no loss"})


        scheduler.step()
        
        # --- Ciclo di validazione ---
        if eval_dataloader and len(eval_dataloader) > 0:
            student_model.eval()
            eval_hard_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc=f"Epoch {epoch+1} [Validation]", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    outputs = student_model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                    if outputs.loss is not None:
                        eval_hard_loss += outputs.loss.item()
            
            avg_eval_loss = eval_hard_loss / len(eval_dataloader)
            print(f"--- Fine Validazione Epoca {epoch+1} - Average Validation Hard Loss: {avg_eval_loss:.4f} ---")
        else:
            avg_eval_loss = float('inf')
            print(f"--- Saltata validazione per l'epoca {epoch+1} ---")

        if train_config.save_model:
            output_dir_epoch = os.path.join(train_config.output_dir, f"epoch_{epoch+1}")
            os.makedirs(output_dir_epoch, exist_ok=True)
            student_model.save_pretrained(output_dir_epoch)
            tokenizer_config_path = os.path.join(output_dir_epoch, "tokenizer_config.json")
            # Salva i parametri del tokenizer in un file json
            with open(tokenizer_config_path, 'w') as f:
                 json.dump(vars(tokenizer), f, indent=4)
            print(f"Modello e config del tokenizer salvati in {output_dir_epoch}")
            
    return {"final_validation_loss": avg_eval_loss}

def main(
    teacher_model_config: str,
    teacher_model_checkpoint: str,
    student_config_file: str,
    data_dir: str,
    csv_file: str,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    alpha: float = 0.5, # Peso per la soft loss (distillazione)
    seed: int = 42,
    save_model: bool = True,
    weight_decay: float = 0.01,
    gamma: float = 0.85,
    gradient_accumulation_steps: int = 1,
    num_workers_dataloader: int = 0,
):
    train_config = SimpleNamespace(
        num_epochs=num_epochs, batch_size=batch_size, lr=lr, alpha=alpha,
        seed=seed, save_model=save_model, output_dir=output_dir, weight_decay=weight_decay,
        gamma=gamma, gradient_accumulation_steps=gradient_accumulation_steps, 
        num_workers_dataloader=num_workers_dataloader
    )
    set_seed(train_config.seed)

    # Carica le configurazioni da file
    with open(teacher_model_config, 'r') as f:
        teacher_config_dict = json.load(f)
    teacher_config = LlamaConfig.from_dict(teacher_config_dict)

    with open(student_config_file, 'r') as f:
        student_config_dict = json.load(f)
    student_config = LlamaConfig.from_dict(student_config_dict)

    tokenizer = MusicTokenizer(
        timeshift_vocab_size=teacher_config.onset_vocab_size, 
        dur_vocab_size=teacher_config.dur_vocab_size,
        octave_vocab_size=teacher_config.octave_vocab_size,
        pitch_class_vocab_size=teacher_config.pitch_class_vocab_size,
        instrument_vocab_size=teacher_config.instrument_vocab_size,
        velocity_vocab_size=teacher_config.velocity_vocab_size,
        sos_token=getattr(teacher_config, 'sos_token', -1),
        eos_token=getattr(teacher_config, 'eos_token', -2),
        pad_token=getattr(teacher_config, 'pad_token', -3)
    )
    print("Tokenizer non condizionale caricato con successo.")

    print(f"Caricamento del Teacher model da: {teacher_model_checkpoint}")
    teacher_model = LlamaForCausalLM(teacher_config)
    
    # Aggiunto weights_only=True per sicurezza e per silenziare il warning
    model_checkpoint = torch.load(teacher_model_checkpoint, map_location="cpu", weights_only=True)
    state_dict = model_checkpoint.get('model_state_dict', model_checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    teacher_model.load_state_dict(new_state_dict, strict=False)
    teacher_model.eval()
    
    print("Teacher model caricato con successo.")
    
    print(f"Creazione del modello Student da configurazione: {student_config_file}")
    student_model = LlamaForCausalLM(student_config)
    print("Modello Student creato con successo.")
    
    print(f"Caricamento dataset da: {data_dir} usando l'indice {csv_file}")
    dataset_config = DatasetConfig(data_dir=data_dir, csv_file=csv_file)
    
    # Il LakhDataset ora verrà usato nella sua forma originale, come nella tua pipeline di fine-tuning
    train_dataset = LakhDataset(dataset_config=dataset_config, tokenizer=tokenizer, partition="train")
    eval_dataset = LakhDataset(dataset_config=dataset_config, tokenizer=tokenizer, partition="test")
    print(f"Dataset di training caricato: {len(train_dataset)} campioni.")
    print(f"Dataset di validazione caricato: {len(eval_dataset)} campioni.")

    # Il collate_fn ora deve gestire due tensori e fare padding su entrambi
    def collate_fn(batch):
        # Filtra eventuali item None che possono provenire dal dataset
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]
        
        # Padding per gli input composti (6 dimensioni)
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token)
        # Padding per le labels del decoder (7 dimensioni)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # La maschera si basa sui token di padding degli input
        attention_mask = (input_ids_padded[:, :, 0] != tokenizer.pad_token).long()
        
        return {"input_ids": input_ids_padded, "labels": labels_padded, "attention_mask": attention_mask}

    train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, collate_fn=collate_fn, num_workers=train_config.num_workers_dataloader, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=train_config.batch_size, collate_fn=collate_fn, num_workers=train_config.num_workers_dataloader)
    
    optimizer = optim.AdamW(student_model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    print("Avvio della distillazione non condizionata...")
    results = train_distillation(student_model, teacher_model, train_dataloader, eval_dataloader, tokenizer, optimizer, scheduler, train_config)
    
    print("Distillazione completata.")
    print(f"Risultati: {results}")

if __name__ == "__main__":
    fire.Fire(main)

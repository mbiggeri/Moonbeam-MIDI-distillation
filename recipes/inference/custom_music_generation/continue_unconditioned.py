import fire
import os
from generation import MusicLlama
import random

# Funzione per continuare e "migliorare" un brano musicale da un file MIDI
def continue_and_improve_midi(
    midi_input_path: str,
    output_path: str,
    ckpt_dir: str,
    tokenizer_path: str,
    model_config_path: str,
    max_gen_len: int = 512,
    prompt_len: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    finetuned_PEFT_weight_path: str = None,
):
    """
    Carica un file MIDI, lo usa come prompt per un modello MusicLlama,
    e salva la continuazione generata in un nuovo file MIDI.

    Args:
        midi_input_path (str): Percorso del file MIDI di input da continuare.
        output_path (str): Percorso dove salvare il file MIDI generato.
        ckpt_dir (str): Percorso al checkpoint del modello pre-addestrato.
        tokenizer_path (str): Percorso al file del tokenizer.
        model_config_path (str): Percorso al file di configurazione del modello.
        max_gen_len (int): Lunghezza massima della musica da generare dopo il prompt.
        prompt_len (int): Numero massimo di token da usare dal MIDI di input come prompt.
        temperature (float): Controlla la casualità. Valori più alti = più casuale.
        top_p (float): Parametro per il nucleus sampling.
        max_seq_len (int): Lunghezza massima della sequenza per il modello.
        max_batch_size (int): Dimensione del batch (usare 1 per un singolo file).
        finetuned_PEFT_weight_path (str): Percorso opzionale ai pesi di fine-tuning PEFT.
    """
    # Imposta un seed per la riproducibilità
    seed = random.randint(0, 10000)
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("Step 1: Inizializzazione del modello MusicLlama...")
    # Costruisci il generatore caricando il modello e il tokenizer
    generator = MusicLlama.build(
        ckpt_dir=ckpt_dir,
        model_config_path=model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        finetuned_PEFT_weight_path=finetuned_PEFT_weight_path
    )
    print("Modello caricato con successo.")

    print(f"Step 2: Processamento del file MIDI di input: {midi_input_path}")
    # Usa il tokenizer del modello per convertire il MIDI in "compound tokens"
    # Questa è la stessa logica di data_preprocess.py
    try:
        # Il metodo midi_to_compound è il cuore della tokenizzazione
        compound_tokens = generator.tokenizer.midi_to_compound(midi_input_path)
        if not compound_tokens:
            print("Errore: Il MIDI non ha prodotto token. Potrebbe essere vuoto o non valido.")
            return
    except Exception as e:
        print(f"Errore durante la lettura del file MIDI: {e}")
        return
    
    print(f"MIDI tokenizzato in {len(compound_tokens)} token.")

    # Prepara il prompt per il modello
    # Aggiunge il token di inizio (SOS) e taglia alla lunghezza desiderata
    prompt_tokens = generator.tokenizer.encode_series(
        compound_tokens, if_add_sos=True, if_add_eos=False
    )
    
    # Se il prompt è più lungo di prompt_len, lo accorciamo
    if len(prompt_tokens) > prompt_len:
        print(f"Il prompt è troppo lungo ({len(prompt_tokens)} token), verrà accorciato a {prompt_len} token.")
        prompt_tokens = prompt_tokens[:prompt_len]

    print(f"Step 3: Generazione della continuazione musicale (max {max_gen_len} nuovi token)...")
    # Esegui la generazione
    results = generator.music_completion(
        [prompt_tokens],  # La funzione si aspetta una lista di prompt
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    print("Generazione completata.")

    # Salva il risultato
    # L'output 'content' contiene l'intero brano (prompt + generazione)
    generated_midi = results[0]['generation']['content']
    
    # Assicurati che la cartella di output esista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generated_midi.save(output_path)
    
    print(f"Step 4: Brano completato e salvato con successo in: {output_path}")


if __name__ == "__main__":
    fire.Fire(continue_and_improve_midi)


'''
USAGE:
python continue_and_improve.py \
  --midi_input_path /percorso/del/tuo/brano.mid \
  --output_path /percorso/di/output/brano_continuato.mid \
  --ckpt_dir /percorso/del/tuo/checkpoint.pt \
  --tokenizer_path /percorso/del/tuo/tokenizer.model \
  --model_config_path src/llama_recipes/configs/model_config.json \
  --max_gen_len 1024 \
  --prompt_len 512 \
  --temperature 0.8
'''
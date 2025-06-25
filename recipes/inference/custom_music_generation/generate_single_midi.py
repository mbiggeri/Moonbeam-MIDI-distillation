# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import fire
import numpy as np
import os
import random
import ast
import json
import torch
from typing import List, Optional

# Assicurati che queste librerie siano installate
import miditoolkit
from mido import MidiFile, MidiTrack, Message, bpm2tempo, MetaMessage
from music21 import chord, harmony

# La classe MusicLlamaConditional deve essere importata dal tuo file
# Esempio: from generation_conditioned import MusicLlamaConditional
# Visto che non ho il file, lo lascio come commento. Assicurati che sia importabile.
from generation_conditioned import MusicLlamaConditional

def chord_to_midi(chord_symbol):
    try:
        chord_obj = harmony.ChordSymbol(chord_symbol)
        pitches = chord_obj.pitches
        return [p.midi for p in pitches]
    except Exception as e:
        print(f"Attenzione: Impossibile parsare l'accordo '{chord_symbol}'. Errore: {e}. L'accordo verrà saltato.")
        return []

def create_midi_from_chords(chord_list, is_incomplete_measure, ticks_per_bar, bpm=120):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    tempo = bpm2tempo(bpm)
    start_time = is_incomplete_measure * ticks_per_bar
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(Message('program_change', program=0, time=start_time))

    for chord_symbol in chord_list:
        notes = chord_to_midi(chord_symbol)
        if not notes:
            continue
        for note in notes:
            track.append(Message('note_on', note=note, velocity=64, time=0))

        ts = mid.ticks_per_beat // 2
        for note in notes:
            track.append(Message('note_off', note=note, velocity=64, time=ts))
            ts = 0
    return mid

def generate_music(
    generator: MusicLlamaConditional,
    # --- Metadati della musica ---
    audio_key: str,
    pitch_range: str,
    num_measures: int,
    bpm: int,
    genre: str,
    track_role: str,
    inst: str,
    sample_rhythm: str,
    time_signature: str,
    min_velocity: int,
    max_velocity: int,
    # MODIFICA 1: La progressione di accordi ora può essere opzionale (None)
    chord_progression: Optional[List[str]],
    # --- Parametri di generazione ---
    temperature: float,
    top_p: float,
    max_gen_len: int,
    # --- Impostazioni del modello ---
    if_add_chords_in_transformer: bool,
    if_add_metadata_in_transformer: bool,
    additional_token_dict: dict,
    chord_dict: dict,
    # --- Percorso di output ---
    output_path: str,
):
    print("--- Inizio della generazione MIDI ---")
    
    print("1. Preparazione dei metadati e dei token di condizione...")
    metadata_id = [
        additional_token_dict[f"audio_key_{audio_key}"],
        additional_token_dict[f"pitch_range_{pitch_range}"],
        additional_token_dict[f"num_measures_{num_measures}"],
        additional_token_dict[f"bpm_{bpm}"],
        additional_token_dict[f"genre_{genre}"],
        additional_token_dict[f"track_role_{track_role}"],
        additional_token_dict[f"inst_{inst.split('-')[0]}"],
        additional_token_dict[f"sample_rhythm_{sample_rhythm}"],
        additional_token_dict[f"time_signature_{time_signature}"],
        additional_token_dict[f"min_velocity_{min_velocity}"],
        additional_token_dict[f"max_velocity_{max_velocity}"]
    ]

    metadata_condition_decoder = [metadata_id] if generator.model.if_add_metadata_in_decoder else None
    # La condizione degli accordi per il decoder sarà None se non vengono forniti
    chord_condition_decoder = [chord_progression] if generator.model.if_add_chord_in_decoder and chord_progression else None

    # MODIFICA 2: Costruiamo il prompt con gli accordi solo se sono stati forniti
    raw_tokens_chord = []
    if chord_progression:
        print("Generazione condizionata da accordi...")
        if num_measures % 4 != 0: is_incomplete_measure = True
        else: is_incomplete_measure = False
        numerator, denominator = map(int, time_signature.split('/'))
        ticks_per_beat = 480
        beats_per_bar = (numerator / denominator) * 4
        ticks_per_bar = int(ticks_per_beat * beats_per_bar)
        temp_chord_midi_path = "temp_chord_prompt.mid"
        chord_midi = create_midi_from_chords(chord_progression, is_incomplete_measure, ticks_per_bar)
        chord_midi.save(temp_chord_midi_path)
        print(f"2. Creato MIDI temporaneo per gli accordi in: {temp_chord_midi_path}")
        raw_tokens_chord = generator.tokenizer.midi_to_compound(temp_chord_midi_path, calibate_to_default_tempo=True)
        os.remove(temp_chord_midi_path)
    else:
        print("Generazione non condizionata da accordi (improvvisazione libera)...")

    prompt = generator.tokenizer.encode_series_con_gen_commu(
        None,
        raw_tokens_chord, # Sarà una lista vuota se non ci sono accordi
        metadata_tokens=[[x for _ in range(6)] for x in metadata_id],
        if_only_keep_condition_tokens=True,
        if_add_chords_in_transformer=if_add_chords_in_transformer,
        if_add_metadata_in_transformer=if_add_metadata_in_transformer
    )
    prompts = [prompt]
    print(f"3. Prompt per il modello creato (lunghezza: {len(prompt)} token).")
    
    chord_token_indices = None
    if chord_progression and if_add_chords_in_transformer:
        try:
            soc_idx = prompt.index(generator.tokenizer.soc_token_compound)
            eoc_idx = prompt.index(generator.tokenizer.eoc_token_compound)
            chord_token_indices = [[soc_idx, eoc_idx]]
        except ValueError:
            print("Attenzione: Token SOC/EOC non trovati nel prompt, la condizione sugli accordi potrebbe non funzionare.")


    print("4. Chiamata al modello MusicLlama per la generazione...")
    results = generator.music_completion(
        prompts,
        bpm_condition=[bpm],
        time_signature_condition=[time_signature],
        num_measures_condition=[num_measures],
        metadata_condition=metadata_condition_decoder,
        chord_condition=chord_condition_decoder,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        condition_token_lengths=[len(p) for p in prompts],
        chord_token_indices=chord_token_indices,
        chord_dict=chord_dict,
        if_return_chords=False
    )
    print("5. Generazione completata.")
    
    generated_midi = results[0]['generation']['content'][0]
    generated_midi.save(output_path)
    print(f"✅ Successo! File MIDI generato e salvato in: {output_path}")


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    model_config_path: str,
    additional_token_dict_path: str,
    finetuned_PEFT_weight_path: str,
    chord_dict_path: str,
    output_path: str = "generated_music.mid",
    
    # --- Metadati per la generazione ---
    audio_key: str = "cmajor",
    pitch_range: str = "mid",
    num_measures: int = 8,
    bpm: int = 120,
    genre: str = "cinematic",
    track_role: str = "main_melody",
    inst: str = "acoustic_piano",
    sample_rhythm: str = "standard",
    time_signature: str = "4/4",
    min_velocity: int = 60,
    max_velocity: int = 100,
    
    # MODIFICA 3: Argomento completamente cambiato per semplicità. Ora è opzionale.
    chords: Optional[str] = None,
    
    # --- Parametri di generazione ---
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: int = 1024,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
    
    # --- Flag del modello ---
    if_add_chords_in_transformer: bool = True,
    if_add_metadata_in_transformer: bool = False,
):
    """
    Genera un singolo file MIDI basato su metadati forniti come argomenti.
    """
    # Imposta il seed casuale per la riproducibilità
    seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"Random seed impostato a: {seed}")
    with open(additional_token_dict_path, "r") as f:
        additional_token_dict = json.load(f)
    with open(chord_dict_path, "r") as f:
        chord_dict = json.load(f)

    print("Caricamento del modello MusicLlama...")
    generator = MusicLlamaConditional.build_commu_con_gen(
        ckpt_dir=ckpt_dir,
        model_config_path=model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        finetuned_PEFT_weight_path=finetuned_PEFT_weight_path,
        additional_token_dict=additional_token_dict
    )
    print("Modello caricato con successo.")

    # Logica di parsing degli accordi
    chord_progression = None
    if chords and chords.strip(): # Aggiunto .strip() per gestire stringhe vuote o con solo spazi
        # Trasforma la stringa "C G Am F" nella lista ['C', 'G', 'Am', 'F']
        chord_progression = chords.strip().split()
        print(f"Accordi forniti: {chord_progression}")
    else:
        print("Nessun accordo fornito, si procederà con l'improvvisazione libera.")
    
    # Avvia la generazione
    generate_music(
        generator=generator,
        audio_key=audio_key,
        pitch_range=pitch_range,
        num_measures=num_measures,
        bpm=bpm,
        genre=genre,
        track_role=track_role,
        inst=inst,
        sample_rhythm=sample_rhythm,
        time_signature=time_signature,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        chord_progression=chord_progression, # Sarà la lista di accordi o None
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
        if_add_chords_in_transformer=if_add_chords_in_transformer,
        if_add_metadata_in_transformer=if_add_metadata_in_transformer,
        additional_token_dict=additional_token_dict,
        chord_dict=chord_dict,
        output_path=output_path
    )

if __name__ == "__main__":
    fire.Fire(main)
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
import multiprocessing

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

'''
# ORIGINAL:
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
'''

# MODIFIED VERSION TO TEST:
def create_midi_from_chords(chord_list, is_incomplete_measure, ticks_per_bar, bpm=120):
    """
    Creates a MIDI file from a list of chord symbols, placing them sequentially in time.
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480

    tempo = bpm2tempo(bpm)
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(Message('program_change', program=0, time=0))

    # Each chord will last for 2 beats (a half note).
    TICKS_PER_CHORD = 2 * mid.ticks_per_beat
    current_delay = is_incomplete_measure * ticks_per_bar

    for chord_symbol in chord_list:
        notes = chord_to_midi(chord_symbol)

        if not notes:
            current_delay += TICKS_PER_CHORD
            continue

        # Add 'note_on' messages for the chord.
        for i, note_val in enumerate(notes):
            track.append(Message(
                'note_on', 
                note=note_val, 
                velocity=64, 
                time=(current_delay if i == 0 else 0)
            ))

        current_delay = TICKS_PER_CHORD

        # Add 'note_off' messages for the chord.
        for i, note_val in enumerate(notes):
            track.append(Message(
                'note_off', 
                note=note_val, 
                velocity=64, 
                time=(current_delay if i == 0 else 0)
            ))

        current_delay = 0

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
    chord_progression: Optional[List[str]],
    # --- Parametri di generazione ---
    temperature: float,
    top_p: float,
    max_gen_len: int,
    min_gen_len: int,
    # --- Impostazioni del modello ---
    if_add_chords_in_transformer: bool,
    if_add_metadata_in_transformer: bool,
    additional_token_dict: dict,
    chord_dict: dict,
    # --- Percorso di output ---
    output_path: str,
    save_chord_track: bool = True,
):
    # --- NEW DEBUG PRINT BLOCK ---
    print("\n" + "="*50)
    print(f"--- AVVIO GENERAZIONE PER: {os.path.basename(output_path)} ---")
    print("--- PARAMETRI DI GENERAZIONE IN USO ---")
    params = {
        "Audio Key": audio_key,
        "Pitch Range": pitch_range,
        "Number of Measures": num_measures,
        "BPM": bpm,
        "Genre": genre,
        "Track Role": track_role,
        "Instrument": inst,
        "Time Signature": time_signature,
        "Min Velocity": min_velocity,
        "Max Velocity": max_velocity,
        "Chord Progression": chord_progression if chord_progression else "Nessuna",
        "Temperature": temperature,
        "Top P": top_p,
        "Max Gen Length": max_gen_len,
        "Min Gen Length": min_gen_len,
    }
    for key, value in params.items():
        print(f"- {key:<25}: {value}")
    print("="*50 + "\n")
    # --- END OF DEBUG PRINT BLOCK ---

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
    chord_condition_decoder = [chord_progression] if generator.model.if_add_chord_in_decoder and chord_progression else None

    raw_tokens_chord = []
    if chord_progression:
        print("Generazione condizionata da accordi...")
        is_incomplete_measure = num_measures % 4 != 0
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
        raw_tokens_chord,
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
        min_gen_len=min_gen_len,
        temperature=temperature,
        top_p=top_p,
        condition_token_lengths=[len(p) for p in prompts],
        chord_token_indices=chord_token_indices,
        chord_dict=chord_dict,
        if_return_chords=True
    )
    print("5. Generazione completata.")
    
    generated_midi = results[0]['generation']['content'][0]
    generated_midi.save(output_path)
    print(f"✅ Successo! File MIDI generato e salvato in: {output_path}")
    
    if save_chord_track:
        generated_chords_midi = results[0]['generation']['chord']
        if generated_chords_midi and generated_chords_midi[0]:
            base_path, extension = os.path.splitext(output_path)
            chord_output_path = f"{base_path}_chords{extension}"
            generated_chords_midi[0].save(chord_output_path)
            print(f"✅ Traccia di accordi salvata in: {chord_output_path}")


# --- NEW WORKER FUNCTION ---
# This function will be executed by each parallel process.
# It loads its own instance of the model and runs one generation task.
def generation_worker(worker_args: dict):
    """
    A worker function for multiprocessing. It loads a model and runs generation.
    """
    try:
        # Unpack all arguments
        ckpt_dir = worker_args["ckpt_dir"]
        model_config_path = worker_args["model_config_path"]
        tokenizer_path = worker_args["tokenizer_path"]
        max_seq_len = worker_args["max_seq_len"]
        max_batch_size = worker_args["max_batch_size"]
        finetuned_PEFT_weight_path = worker_args["finetuned_PEFT_weight_path"]
        additional_token_dict = worker_args["additional_token_dict"]
        output_path = worker_args["output_path"]
        generation_index = worker_args["generation_index"]

        # Each process should have a different seed for varied outputs
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        random.seed(seed)
        
        print(f"[Process {generation_index}]: Loading model with seed {seed}...")
        
        generator = MusicLlamaConditional.build_commu_con_gen(
            ckpt_dir=ckpt_dir,
            model_config_path=model_config_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            finetuned_PEFT_weight_path=finetuned_PEFT_weight_path,
            additional_token_dict=additional_token_dict,
            seed=seed
        )
        print(f"[Process {generation_index}]: Model loaded.")

        # Call the main generation logic
        generate_music(
            generator=generator,
            output_path=worker_args["output_path"],
            save_chord_track=False, # This prevents the worker from saving the chord file
            **worker_args["music_metadata"]
        )
        return f"[Process {generation_index}]: Successfully generated {output_path}"
    except Exception as e:
        return f"[Process {generation_index}]: FAILED with error: {e}"

# --- MAIN FUNCTION ---
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    model_config_path: str,
    additional_token_dict_path: str,
    finetuned_PEFT_weight_path: str,
    chord_dict_path: str,
    output_path: str = "generated_music.mid",
    
    number_generations: int = 1,

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
    chords: Optional[str] = None,
    
    # --- Parametri di generazione ---
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: int = 1024,
    min_gen_len: int = 50,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
    
    if_add_chords_in_transformer: bool = True,
    if_add_metadata_in_transformer: bool = True,
):
    """
    Genera uno o più file MIDI basato su metadati, usando il multiprocessing.
    """
    with open(additional_token_dict_path, "r") as f:
        additional_token_dict = json.load(f)
    with open(chord_dict_path, "r") as f:
        chord_dict = json.load(f)

    chord_progression = chords.strip().split() if chords and chords.strip() else None

    # --- Prepare arguments for workers ---
    tasks = []
    base_path, extension = os.path.splitext(output_path)
    
    for i in range(number_generations):
        current_output_path = f"{base_path}_{i}{extension}" if number_generations > 1 else output_path

        music_metadata_args = {
            "audio_key": audio_key, "pitch_range": pitch_range, "num_measures": num_measures,
            "bpm": bpm, "genre": genre, "track_role": track_role, "inst": inst,
            "sample_rhythm": sample_rhythm, "time_signature": time_signature,
            "min_velocity": min_velocity, "max_velocity": max_velocity,
            "chord_progression": chord_progression, "temperature": temperature,
            "top_p": top_p, "max_gen_len": max_gen_len, "min_gen_len": min_gen_len,
            "if_add_chords_in_transformer": if_add_chords_in_transformer,
            "if_add_metadata_in_transformer": if_add_metadata_in_transformer,
            "additional_token_dict": additional_token_dict, "chord_dict": chord_dict,
        }

        task_args = {
            "ckpt_dir": ckpt_dir, "tokenizer_path": tokenizer_path,
            "model_config_path": model_config_path,
            "finetuned_PEFT_weight_path": finetuned_PEFT_weight_path,
            "additional_token_dict": additional_token_dict,
            "output_path": current_output_path, "max_seq_len": max_seq_len,
            "max_batch_size": max_batch_size, "generation_index": i,
            "music_metadata": music_metadata_args
        }
        tasks.append(task_args)

    # --- Execute generation tasks ---
    if number_generations <= 1:
        print("Avvio della generazione singola...")
        generation_worker(tasks[0])
    else:
        num_processes = min(number_generations, os.cpu_count(), 8) # Limit to 8 processes to avoid VRAM issues
        print(f"Avvio di {number_generations} generazioni in parallelo usando {num_processes} processi...")
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(generation_worker, tasks):
                print(result)

    # --- Save the single chord track (if applicable) ---
    if chord_progression:
        print("\nCreazione del file MIDI per la traccia di accordi singola...")
        try:
            is_incomplete_measure = num_measures % 4 != 0
            numerator, denominator = map(int, time_signature.split('/'))
            ticks_per_beat = 480
            beats_per_bar = (numerator / denominator) * 4
            ticks_per_bar = int(ticks_per_beat * beats_per_bar)
            chord_midi = create_midi_from_chords(
                chord_list=chord_progression,
                is_incomplete_measure=is_incomplete_measure,
                ticks_per_bar=ticks_per_bar,
                bpm=bpm
            )
            chord_output_path = f"{base_path}_chords{extension}"
            chord_midi.save(chord_output_path)
            print(f"✅ Traccia di accordi singola salvata in: {chord_output_path}")
        except Exception as e:
            print(f"ATTENZIONE: Errore durante la creazione della traccia di accordi: {e}")

    print("\n--- Tutte le generazioni sono state completate. ---")

if __name__ == "__main__":
    fire.Fire(main)
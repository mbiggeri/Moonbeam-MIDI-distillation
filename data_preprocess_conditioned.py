import os
import matplotlib.pyplot as plt
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from collections import Counter
import numpy as np
from collections import defaultdict
import argparse
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
import json
from mido import Message, MidiFile, MidiTrack
from music21 import chord
from music21 import harmony

import miditoolkit
import math
import bisect
import shutil

tokenizer = None  # Definisci il tokenizer come variabile globale per i worker
num_cores = multiprocessing.cpu_count()

class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return "Item(name={}, start={}, end={}, velocity={}, pitch={})".format(
            self.name, self.start, self.end, self.velocity, self.pitch
        )

class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return "Event(name={}, time={}, value={}, text={})".format(
            self.name, self.time, self.value, self.text
        )

def init_tokenizer_worker(config):
    """
    Funzione di inizializzazione per ogni processo worker.
    Crea un'istanza del tokenizer per processo.
    """
    global tokenizer
    tokenizer = MusicTokenizer(
        timeshift_vocab_size=config["onset_vocab_size"],
        dur_vocab_size=config["dur_vocab_size"],
        octave_vocab_size=config["octave_vocab_size"],
        pitch_class_vocab_size=config["pitch_class_vocab_size"],
        instrument_vocab_size=config["instrument_vocab_size"],
        velocity_vocab_size=config["velocity_vocab_size"]
    )

def detect_chord(chord_progression, beats_per_bar):
    chords_per_bar = beats_per_bar * 2
    num_measures = int(len(chord_progression)/chords_per_bar)
    split_by_bar = np.array_split(np.array(chord_progression), num_measures)
    chord_idx = []
    chord_name = []
    for bar_idx, bar in enumerate(split_by_bar):
        for c_idx, chord in enumerate(bar):
            if c_idx == 0 or chord != chord_name[-1]:
                chord_idx.append(bar_idx + c_idx / chords_per_bar)
                chord_name.append(chord)
    return chord_idx, chord_name

def find_chord_bar_position(
    chord_progression,
    tick_per_bar,
    num_measures,
    is_incomplete_measure,
    beats_per_bar,
    position_resolution
):
    chord_idx_lst, chords = detect_chord(chord_progression, beats_per_bar)
    start_time = tick_per_bar * is_incomplete_measure

    chord_event_deciphered = []
    for i in range(num_measures):
        while chord_idx_lst and chord_idx_lst[0] < i + 1 - is_incomplete_measure:
            chord_position = chord_idx_lst.pop(0)
            chord_time = int(chord_position * tick_per_bar + start_time)
            chord = chords.pop(0)
            chord_event_deciphered.append((i, int((chord_position - i + is_incomplete_measure) * position_resolution), chord)) #bar, position, chord_name

    return chord_event_deciphered

def item2event(groups, duration_bins, position_resolution):

    n_downbeat = 0
    output = []
    for i in range(len(groups)):
        bar_st, bar_et = groups[i][0], groups[i][-1]
        if "NN" in [item.pitch for item in groups[i][1:-1]]: #1:-1 remove first and last downbeat
            n_downbeat += 1
            continue
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, position_resolution, endpoint=False)
            index = np.argmin(abs(flags - item.start)) 
            output.append((n_downbeat, index))
        n_downbeat += 1

    return output

def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(
            Item(
                name="Note",
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch,
            )
        )
    note_items.sort(key=lambda x: x.start)
    return note_items

def group_items(items, max_time, ticks_per_bar):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        if not insiders:
            insiders.append(Item(name="None", start=None, end=None, velocity=None, pitch="NN"))
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

def extract_events(
    input_path,
    duration_bins,
    ticks_per_bar=None,
    ticks_per_beat=None,
    chord_progression=None,
    num_measures=None,
    is_incomplete_measure=None,
    position_resolution = None):
    note_items = read_items(input_path)
    max_time = note_items[-1].end
    if not chord_progression[0]:
        return None
    else:
        items = note_items
    groups = group_items(items, max_time, ticks_per_bar) #insert downbeats between events, each group is a bar of events
    bar_and_beats = item2event(groups, duration_bins, position_resolution) #quantize events to the nearest position
    
    beats_per_bar = int(ticks_per_bar/ticks_per_beat)
    if chord_progression:
        chord_idx_lst= find_chord_bar_position(
            chord_progression,
            ticks_per_bar,
            num_measures,
            is_incomplete_measure,
            beats_per_bar,
            position_resolution
        )

    # Organize chords into a dictionary by bar and sort positions within each bar
    chord_dict = {}
    for bar, pos, name in chord_idx_lst:
        if bar not in chord_dict:
            chord_dict[bar] = []
        chord_dict[bar].append((pos, name))

    # Sort chords within each bar by their position
    for bar in chord_dict:
        chord_dict[bar].sort(key=lambda x: x[0])

    # Determine the chord for each note
    note_chords = []
    bar_beat_chord_2 = [[0, 0, "s"]] #for sos token
    for note in bar_and_beats:
        note_bar, note_pos = note
        chords = chord_dict.get(note_bar, [])
        positions = [pos for pos, _ in chords]
        # Find the rightmost chord position <= note_pos
        index = bisect.bisect_right(positions, note_pos) - 1
        if index >= 0:
            chord_name = chords[index][1]
        else:
            chord_name = "s"  #when there are no chords, assign a special token "s"
        note_chords.append(chord_name)
        bar_beat_chord_2.append(list((note_bar, note_pos, chord_name)))
    
    return bar_beat_chord_2

def find_bar_beat_chord(midi_paths, chord_progression, num_measures, time_signature, is_incomplete_measure, sample_info=None):
    midi_file = miditoolkit.MidiFile(midi_paths)
    ticks_per_beat = midi_file.ticks_per_beat
    num_measures = math.ceil(num_measures)
    numerator = int(time_signature.split("/")[0])
    denominator = int(time_signature.split("/")[1])
    position_resolution = int(numerator * 32 / denominator)
    beats_per_bar = numerator / denominator * 4
    ticks_per_bar = int(ticks_per_beat * beats_per_bar)
    duration_bins = np.arange(
        int(ticks_per_bar / position_resolution),
        ticks_per_bar + 1,
        int(ticks_per_bar / position_resolution),
        dtype=int,
    )

    bar_beat_chord = extract_events(
        midi_paths,
        duration_bins,
        ticks_per_bar=ticks_per_bar,
        ticks_per_beat=ticks_per_beat,
        chord_progression=chord_progression,
        num_measures=num_measures,
        is_incomplete_measure=is_incomplete_measure,
        position_resolution=position_resolution
    )
    return bar_beat_chord

def chord_to_midi(chord_symbol):
    # Create a ChordSymbol object
    chord_obj = harmony.ChordSymbol(chord_symbol)
    # Get the pitches of the chord
    pitches = chord_obj.pitches
    # Return the pitch names
    out = [p.midi  for p in pitches]
    return out

# Function to create a MIDI file from a chord progression
def create_midi_from_chords(chord_list, is_incomplete_measure, ticks_per_bar, filename, bpm=220):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set the tempo (optional)
    start_time = is_incomplete_measure*ticks_per_bar
    if is_incomplete_measure:
        print(f"incomplete measure, start_time: {start_time}")
    track.append(Message('program_change', program=0, time=start_time))
    for chord_symbol in chord_list:
        notes = chord_to_midi(chord_symbol)
        for note in notes:
            track.append(Message('note_on', note=note, velocity=64, time=0))
        ts = 240
        for note in notes:
            track.append(Message('note_off', note=note, velocity=64, time=ts))  # 480 ticks for a beat
            ts = 0
    mid.save(filename)

def process_chord_progression(args):
    chord_list_str, file_id, split, output_path, num_measures, time_signature, dataset_folder = args
    chord_list = ast.literal_eval(chord_list_str)
    flat_chord_list = [chord for sublist in chord_list for chord in sublist]

    if num_measures%4==0:
        is_incomplete_measure = False
    else:
        is_incomplete_measure = True
    numerator = int(time_signature.split("/")[0])
    denominator = int(time_signature.split("/")[1])
    ticks_per_beat = miditoolkit.MidiFile(os.path.join(dataset_folder, split, "raw", f"{file_id}.mid")).ticks_per_beat
    beats_per_bar = numerator / denominator * 4
    ticks_per_bar = int(ticks_per_beat * beats_per_bar)
    # Create a MIDI file from the chord progression
    create_midi_from_chords(flat_chord_list,is_incomplete_measure, ticks_per_bar, f'{output_path}/{split}/{file_id}_chord.mid')

def process_single_file(args):
    # Unpack all arguments needed for processing a single file
    chord_progressions_str, num_measures, time_signature, split, file_id, dataset_folder, output_path = args
    
    # Construct MIDI file path
    midi_path = os.path.join(dataset_folder, split, "raw", f"{file_id}.mid")
    
    # Generate bar/beat/chord structure using MIDI analysis
    bar_beat_chord = find_bar_beat_chord(midi_path, ast.literal_eval(chord_progressions_str)[0], num_measures, time_signature, num_measures%4!=0)
    
    # Create output path and ensure directory exists
    output_filename = os.path.join(output_path, split, f"{file_id}_bar_beat_chord.npy")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    # Save processed data
    np.save(output_filename, np.array(bar_beat_chord))
    print("bar beat chord saved at", output_filename)
    return True

def process_midi_file_wrapper(args):
    """Wrapper per chiamare la funzione di processamento principale."""
    # Ora questa funzione non ha più bisogno di inizializzare il tokenizer,
    # perché è già stato fatto da init_tokenizer_worker.
    return process_midi_file_safe_v3_commu(*args)

def find_midi_files_from_file_commu(dataset_name, split_file, dataset_folder, output_folder):
    if dataset_name == "commu_con_gen":
        output_path = f"{dataset_folder}/commu_midi_chords"
        output_path_bar_beat_chord = f"{dataset_folder}/commu_midi_bar_beat_chord"
        os.makedirs(output_path+"/train", exist_ok=True)
        os.makedirs(output_path+"/val", exist_ok=True)
        os.makedirs(output_path_bar_beat_chord+"/train", exist_ok=True)
        os.makedirs(output_path_bar_beat_chord+"/val", exist_ok=True)
        df = pd.read_csv(split_file, index_col=False)

        df["inst"] = df["inst"].apply(lambda x: x.split("-")[0])

        #step1: construct tokens_dict and save it to local json file
        tokens_dict = {}

        for column in df.columns:
            # Skip 'split' and 'id' columns
            if column in ['split_data', 'id', "chord_progressions"]:
                continue
            
            # Extract unique values for the column
            unique_values = df[column].unique()
            
            # Store the sorted unique values in the dictionary
            tokens_dict[column] = sorted(set(unique_values))

        tokens_dict.pop("Unnamed: 0", None) # Remove the 'Unnamed: 0' key if it exists
        # Create an indexed dictionary
        indexed_tokens_dict = {"soc_token_compound": -4, "eoc_token_compound": -5}

        index = -6

        for column, tokens in tokens_dict.items():
            for token in tokens:
                indexed_tokens_dict[f"{column}_{token}"] = index
                index -= 1

        with open(f'{output_folder}/indexed_tokens_dict.json', 'w') as f:
            json.dump(indexed_tokens_dict, f)
            print(f"Indexed tokens dict saved at {output_folder}/indexed_tokens_dict.json")
        # Step3: tokenize labels
        label_lists = []
        for column, tokens in tokens_dict.items():
            label_lists.append(df[column].apply(lambda x: indexed_tokens_dict[f"{column}_{x}"]).tolist())
        
        labels = list(zip(*label_lists))
        chord_lists = df['chord_progressions'].tolist()
        file_id_lists = df['id'].tolist()
        splits = df['split_data'].tolist()
        num_measure_lists = df['num_measures'].tolist()
        time_signature_lists = df['time_signature'].tolist()

        # Step4: find all corresponding files
        midi_files = [os.path.join(dataset_folder, split,"raw" , x)+ ".mid" for split, x in zip(splits, file_id_lists)]

        args_list = [
            (chord_progressions, num_measures, time_signature, split, file_id, dataset_folder, output_path_bar_beat_chord)
            for chord_progressions, num_measures, time_signature, split, file_id in zip(
                chord_lists, num_measure_lists, time_signature_lists, splits, file_id_lists
            )
        ]

        # Process files in parallel with progress tracking
        with multiprocessing.Pool(processes=num_cores) as pool:
            for _ in tqdm(pool.imap_unordered(process_single_file, args_list), total=len(args_list)):
                pass
        # Step2: generate chord midi files
        # Use multiprocessing to process the chord progressions
        # args_list = [(chord_list_str, file_id, split,output_path) for chord_list_str, file_id, split in zip(chord_lists, file_id_lists, splits)]
        args_list = [(chord_list_str, file_id, split,output_path, num_measures, time_signature, dataset_folder) 
                    for chord_list_str, file_id, split, num_measures, time_signature in zip(
                        chord_lists, file_id_lists, splits, num_measure_lists, time_signature_lists)]

        with multiprocessing.Pool(processes=num_cores) as pool:
            for _ in tqdm(pool.imap_unordered(process_chord_progression, args_list), total=len(args_list)):
                pass
        chord_midi_files = [os.path.join(output_path, split, x )+ "_chord.mid" for split, x in zip(splits, file_id_lists)]
        bar_beat_chord_files = [os.path.join(output_path_bar_beat_chord, split, x )+ "_bar_beat_chord.npy" for split, x in zip(splits, file_id_lists)]
        # labels = zipped_label_lists
    assert len(midi_files) == len(chord_midi_files) == len(splits) == len(labels) == len(bar_beat_chord_files)
    return midi_files, chord_midi_files, bar_beat_chord_files, splits, labels

def detect_large_timeshifts_and_durations(compounds, onset_vocab_size, dur_vocab_size):
    #check if file contains large timeshits and durations
    onsets = [c[0] for c in compounds]
    onsets_padded = [0] + onsets
    timeshift_counter = Counter([onsets_padded[i+1] - onsets_padded[i] for i in range(len(onsets_padded) - 1)]) 
    duration_counter = Counter([c[1] for c in compounds])
    onsets_exceed_vocab_size = any(key > onset_vocab_size-3 for key in timeshift_counter.keys()) #why -3? onset_vocab_size -1, onset_vocab_size -2 are assigned to sos and eos, max value is onset_vocab_size - 3
    duration_exceed_vocab_size = any(key > dur_vocab_size-3 for key in duration_counter.keys()) #if dur_vocab_size=1026, 1024 and 1025 are reserved for sos and eos, the largest value becomes 1023
    return onsets_exceed_vocab_size, duration_exceed_vocab_size, timeshift_counter, duration_counter

def filter_large_ts_dur_commu(compounds, chord_compounds, output_file_path, chord_output_file_path, beat_bar_chord_output_file_path, split, label, onset_vocab_size, dur_vocab_size, log_file):
    """detect large timeshifts and durations, return None if any of them exceed the vocab size else return the processed dictionary"""

    onsets_exceed_vocab_size, duration_exceed_vocab_size, timeshift_counter, duration_counter = detect_large_timeshifts_and_durations(compounds, onset_vocab_size, dur_vocab_size)

    if onsets_exceed_vocab_size or duration_exceed_vocab_size:
        with open(log_file, 'a') as log:
            log.write(f'Failed to process {output_file_path} but during finetuning these files will be kept:\n')
            log.write(f'{output_file_path} contains large onsets: {onsets_exceed_vocab_size}, large durations: {duration_exceed_vocab_size}\n')
            # Optionally, print the largest onset and duration exceeding the vocab size
            if onsets_exceed_vocab_size:
                largest_onset = max(key for key in timeshift_counter.keys() if key > onset_vocab_size - 3)
                log.write(f'Largest onset: {largest_onset}\n')
            if duration_exceed_vocab_size:
                largest_duration = max(key for key in duration_counter.keys() if key > dur_vocab_size - 3)
                log.write(f'Largest duration: {largest_duration}\n')

        return None

    else:

        return {
            'file': output_file_path,
            'compounds': compounds,
            'chord_file': chord_output_file_path,
            'beat_bar_chord_file': beat_bar_chord_output_file_path,
            'chord_compounds': chord_compounds,
            'split': split,
            'label':label,
            'timeshifts':dict(timeshift_counter),
            'durations':dict(duration_counter),           
            'length_token': len(compounds),
            'length_duration': compounds[-1][0]+compounds[-1][1],
        }

def process_midi_file_v3_commu(midi_file, chord_midi_file, bar_beat_chord_file ,split, label, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold = None):
    #convert midi to compounds
    compounds = tokenizer.midi_to_compound(midi_file, calibate_to_default_tempo = True)
    chord_compounds = tokenizer.midi_to_compound(chord_midi_file, calibate_to_default_tempo = True)
    output_file_name = midi_file.replace("/", "_").replace(".midi", ".npy").replace(".mid", ".npy") #TODO: this will avoid duplicates
    output_file_path = os.path.join(output_folder, output_file_name)

    chord_output_file_path = output_file_path.replace(".npy", "_chord.npy")

    #copy the processed beats, pos, chord to the output folder
    beat_bar_chord_output_file_path = output_file_path.replace(".npy", "_bar_beat_chord.npy")
    shutil.copyfile(bar_beat_chord_file, beat_bar_chord_output_file_path)
    return [filter_large_ts_dur_commu(compounds, chord_compounds, output_file_path, chord_output_file_path, beat_bar_chord_output_file_path, split, label, onset_vocab_size, dur_vocab_size, log_file)]

def process_midi_file_safe_v3_commu(midi_file, chord_midi_file, bar_beat_chord_file, split, label, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold = None):
    """
    input: midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold

    output: a list of processed chunked dictionaries [{file: file_path, split: split, onsets: onset_counter, durations: duration_counter, length: length}, ...] or None
    """
    try:
        out = process_midi_file_v3_commu(midi_file, chord_midi_file, bar_beat_chord_file, split, label, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold)
        if out is not None:
            for out_chunk in out:
                if out_chunk is not None:
                    np.save(out_chunk['file'], out_chunk['compounds'])
                    np.save(out_chunk['chord_file'], out_chunk['chord_compounds'])
        return out
    except Exception as e:
        # Log failure with full traceback
        with open(log_file, 'a') as log:
            log.write(f'Failed to process {midi_file}:\n')
            log.write(traceback.format_exc())
            log.write('\n')
        return [None]

def convert_chord_symbol_to_indices(directory):
    files = [f for f in os.listdir(directory) if f.endswith('bar_beat_chord.npy')]
    
    # 1. Collect unique chords
    unique_chords = set()
    
    # First pass: collect all unique chords
    for file in tqdm(files, desc="Scanning files"):
        data = np.load(os.path.join(directory, file), allow_pickle=True)
        for row in data:
            unique_chords.add(row[2])
    
    # Sort chords alphabetically
    alphabetically_sorted_chords = sorted(unique_chords)
    
    # Create dictionary with alphabetical ordering
    chord_to_idx = {chord: idx for idx, chord in enumerate(alphabetically_sorted_chords)}
    
    # Save dictionary as JSON
    with open(os.path.join(os.path.dirname(directory), 'chord_dictionary.json'), 'w') as f:
        json.dump(chord_to_idx, f, indent=4, sort_keys=False)  # Explicitly disable key sorting
    
    # 2. Convert and overwrite files
    for file in tqdm(files, desc="Processing files"):
        filepath = os.path.join(directory, file)
        data = np.load(filepath, allow_pickle=True)
        
        # Convert last column to indices
        new_data = []
        for row in data:
            new_row = [int(row[0]), int(row[1]), chord_to_idx[row[2]]]
            new_data.append(new_row)
        
        # Overwrite original file
        np.save(filepath, np.array(new_data, dtype=np.int32))

def setup_arg_parser() -> argparse.Namespace:
    """Configura e restituisce il parser per gli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(description='Process MIDI files for a conditional dataset.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset Name (e.g., "commu_con_gen").')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to the root dataset folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder where processed files will be saved.')
    parser.add_argument('--model_config', type=str, required=True, help='Model configuration file that decides the vocab size.')
    parser.add_argument('--train_test_split_file', type=str, required=True, help='Path to the CSV file defining splits and metadata.')
    parser.add_argument('--ts_threshold', type=lambda x: None if x.lower() == "none" else int(x), default=None, help='If Timeshift exceeds this value, chunk the file. Default: None.')
    return parser.parse_args()

def load_conditional_dataset(dataset_name: str, split_file: str, dataset_folder: str, output_folder: str) -> tuple:
    """
    Carica i metadati dal file di split, prepara i percorsi e i task per il processamento.
    Sostituisce la vecchia funzione `find_midi_files_from_file_commu`.
    """
    if dataset_name == "commu_con_gen":
        print(f"Loading and preparing '{dataset_name}' dataset...")
        
        output_path_chords = os.path.join(dataset_folder, "commu_midi_chords")
        output_path_bar_beat_chord = os.path.join(dataset_folder, "commu_midi_bar_beat_chord")
        os.makedirs(os.path.join(output_path_chords, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path_chords, "val"), exist_ok=True)
        os.makedirs(os.path.join(output_path_bar_beat_chord, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path_bar_beat_chord, "val"), exist_ok=True)
        
        df = pd.read_csv(split_file, index_col=False)
        df["inst"] = df["inst"].apply(lambda x: x.split("-")[0])

        # --- Creazione dizionario token (logica originale) ---
        tokens_dict = {}
        for column in df.columns:
            if column in ['split_data', 'id', "chord_progressions", "Unnamed: 0"]:
                continue
            tokens_dict[column] = sorted(set(df[column].unique()))
        
        indexed_tokens_dict = {"soc_token_compound": -4, "eoc_token_compound": -5}
        index = -6
        for column, tokens in tokens_dict.items():
            for token in tokens:
                indexed_tokens_dict[f"{column}_{token}"] = index
                index -= 1
        
        dict_path = os.path.join(output_folder, 'indexed_tokens_dict.json')
        with open(dict_path, 'w') as f:
            json.dump(indexed_tokens_dict, f)
            print(f"Indexed tokens dict saved at {dict_path}")

        # --- Preparazione liste di dati dal DataFrame ---
        label_lists = [df[col].apply(lambda x: indexed_tokens_dict[f"{col}_{x}"]).tolist() for col, _ in tokens_dict.items()]
        labels = list(zip(*label_lists))
        chord_lists = df['chord_progressions'].tolist()
        file_id_lists = df['id'].tolist()
        splits = df['split_data'].tolist()
        num_measure_lists = df['num_measures'].tolist()
        time_signature_lists = df['time_signature'].tolist()

        # --- Processamento preliminare (creazione MIDI accordi e struttura bar/beat) ---
        print("Preprocessing chord progressions and bar/beat structures...")
        bar_beat_args = [(cl, nm, ts, s, fid, dataset_folder, output_path_bar_beat_chord) for cl, nm, ts, s, fid in zip(chord_lists, num_measure_lists, time_signature_lists, splits, file_id_lists)]
        chord_prog_args = [(cl, fid, s, output_path_chords, nm, ts, dataset_folder) for cl, fid, s, nm, ts in zip(chord_lists, file_id_lists, splits, num_measure_lists, time_signature_lists)]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(process_single_file, bar_beat_args), total=len(bar_beat_args), desc="Extracting bar/beat structure"))
            list(tqdm(pool.imap_unordered(process_chord_progression, chord_prog_args), total=len(chord_prog_args), desc="Generating chord MIDIs"))

        # --- Costruzione percorsi finali ---
        midi_files = [os.path.join(dataset_folder, s, "raw", f"{fid}.mid") for s, fid in zip(splits, file_id_lists)]
        chord_midi_files = [os.path.join(output_path_chords, s, f"{fid}_chord.mid") for s, fid in zip(splits, file_id_lists)]
        bar_beat_chord_files = [os.path.join(output_path_bar_beat_chord, s, f"{fid}_bar_beat_chord.npy") for s, fid in zip(splits, file_id_lists)]
        
        assert len(midi_files) == len(chord_midi_files) == len(splits) == len(labels) == len(bar_beat_chord_files)
        return midi_files, chord_midi_files, bar_beat_chord_files, splits, labels

    else:
        raise ValueError(f"Dataset '{dataset_name}' non supportato.")

def save_results_to_csv(results: list, csv_path: str) -> int:
    """Salva i risultati del processamento in un file CSV e restituisce il numero di successi."""
    csv_rows = []
    # Appiattisce la lista di liste e rimuove i None
    successful_results = [item for sublist in results if sublist for item in sublist if item]
    
    for result in successful_results:
        csv_rows.append([
            os.path.basename(result['file']),
            os.path.basename(result['chord_file']),
            os.path.basename(result['beat_bar_chord_file']),
            result['split'],
            result['label']
        ])

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file_base_name', 'chord_file_base_name', 'beat_bar_base_name', 'split', 'label'])
        csv_writer.writerows(csv_rows)
    
    print(f'CSV file with {len(csv_rows)} entries saved to {csv_path}')
    return len(successful_results)

# -----------------------------------------------------------------------------
# FUNZIONE MAIN
# -----------------------------------------------------------------------------
def main():
    """Funzione principale che orchestra il processo di preprocessing."""
    args = setup_arg_parser()
    
    # 1. Configura percorsi e parametri
    midi_output_folder = os.path.join(args.output_folder, "processed")
    log_file = os.path.join(args.output_folder, "failed_midi_files.log")
    csv_file_path = os.path.join(args.output_folder, "train_test_split.csv")
    os.makedirs(midi_output_folder, exist_ok=True)
    
    with open(args.model_config, 'r') as f:
        config_data = json.load(f)
    
    tokenizer_config = {
        "onset_vocab_size": config_data.get("onset_vocab_size"),
        "dur_vocab_size": config_data.get("dur_vocab_size"),
        "octave_vocab_size": config_data.get("octave_vocab_size"),
        "pitch_class_vocab_size": config_data.get("pitch_class_vocab_size"),
        "instrument_vocab_size": config_data.get("instrument_vocab_size"),
        "velocity_vocab_size": config_data.get("velocity_vocab_size")
    }
    assert all(tokenizer_config.values()), "Una o più configurazioni del tokenizer mancano nel file JSON."

    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} CPUs. Max timeshift: {tokenizer_config['onset_vocab_size']-3}, max duration: {tokenizer_config['dur_vocab_size']-3}")

    # 2. Carica i dati e prepara i task di elaborazione
    midi_files, chord_midi_files, bar_beat_chord_files, splits, labels = load_conditional_dataset(
        args.dataset_name, args.train_test_split_file, args.dataset_folder, args.output_folder
    )
    
    tasks = [
        (midi_file, chord_midi_file, bar_beat_chord_file, split, label, tokenizer_config['onset_vocab_size'], tokenizer_config['dur_vocab_size'], midi_output_folder, log_file, args.ts_threshold)
        for midi_file, chord_midi_file, bar_beat_chord_file, split, label in zip(midi_files, chord_midi_files, bar_beat_chord_files, splits, labels)
    ]
    print(f"{len(tasks)} MIDI files found and prepared for tokenization.")
    
    # 3. Esegui il processamento in parallelo
    results = []
    with multiprocessing.Pool(processes=num_cpus, initializer=init_tokenizer_worker, initargs=(tokenizer_config,)) as pool:
        # Usa imap_unordered per processare i file man mano che sono pronti
        results_iterator = pool.imap_unordered(process_midi_file_wrapper, tasks)
        # Usa tqdm per mostrare una barra di avanzamento
        for result in tqdm(results_iterator, total=len(tasks), desc='Tokenizing MIDI files'):
            results.append(result)

    # 4. Salva i risultati e le statistiche
    success_count = save_results_to_csv(results, csv_file_path)
    failure_count = len(tasks) - success_count
    print(f'Processed {len(tasks)} files: {success_count} succeeded, {failure_count} failed.')
    
    # 5. Esegui la conversione finale degli accordi in indici
    print("Converting chord symbols to indices for all processed files...")
    convert_chord_symbol_to_indices(midi_output_folder)
    
    print("\nPreprocessing complete!")

if __name__ == '__main__':
    main()





    
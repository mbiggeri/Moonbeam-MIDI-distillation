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

# USAGE: python data_preprocess.py --dataset_folder "C:\Users\Michael\Desktop\MusicDatasets\Datasets\Moonbeam_Distillation\raw_midi_data" 
# --output_folder "C:\Users\Michael\Desktop\MusicDatasets\Datasets\Moonbeam_Distillation\processed_data" 
# --model_config "src/llama_recipes/configs/model_config.json" --train_ratio 0.9

num_cores = multiprocessing.cpu_count()
tokenizer = None  # Definisci il tokenizer come variabile globale

def init_tokenizer_worker(onset_vs, dur_vs, octave_vs, pitch_class_vs, instrument_vs, velocity_vs):
    """
    Funzione di inizializzazione per ogni processo worker.
    Crea un'istanza del tokenizer per processo.
    """
    global tokenizer
    tokenizer = MusicTokenizer(
        timeshift_vocab_size=onset_vs,
        dur_vocab_size=dur_vs,
        octave_vocab_size=octave_vs,
        pitch_class_vocab_size=pitch_class_vs,
        instrument_vocab_size=instrument_vs,
        velocity_vocab_size=velocity_vs
    )

def chunk_compounds(compounds, threshold=1024):
    """chunk the compounds such that long silences in between are not treated as long timeshifts"""
    if not compounds:
        return []

    onsets = [c[0] for c in compounds]
    onsets_padded = [0] + onsets
    timeshifts = [onsets_padded[i+1] - onsets_padded[i] for i in range(len(onsets_padded) - 1)]
    cur_pos = 0
    out = []
    for pointer in range(len(onsets)):
        if timeshifts[pointer] > threshold:
            out.append(compounds[cur_pos:pointer])
            cur_pos = pointer
    out.append(compounds[cur_pos:])

    # Rimuovi eventuali chunk vuoti
    out = [chunk for chunk in out if chunk]

    for i, chunk in enumerate(out): #shift the onsets to 0
        if not chunk: continue
        first_onset = chunk[0][0]
        out[i] = [[comps[0]-first_onset] + comps[1:]  for comps in chunk]
    assert sum([len(chunk) for chunk in out]) == len(compounds)
    return out

def find_midi_files(folder):
    midi_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid') or file.endswith('.MID'):
                midi_files.append(os.path.join(root, file))
    return midi_files

def find_midi_files_from_file(dataset_name, split_file, dataset_folder):
    if dataset_name =="GAPS":
        df = pd.read_csv(split_file)
        train_files = df[df['split'] == 'train_annotation']['filename'].tolist()
        test_files = df[df['split'] == 'test_annotation']['filename'].tolist()
        midi_files = train_files + test_files
        midi_files = [os.path.join(dataset_folder, f) for f in midi_files]
        splits = ['train']*len(train_files) + ['test']*len(test_files)

    elif dataset_name == "GuitarSet":
        import json
        with open(split_file, 'r') as file:
            splits = json.load(file)
        train_files = [os.path.join(dataset_folder, os.path.basename(f)) for f in splits['train_annotation']]
        test_files = [os.path.join(dataset_folder, os.path.basename(f)) for f in splits['test_annotation']]
        midi_files = train_files + test_files
        splits = ['train']*len(train_files) + ['test']*len(test_files)

    assert len(midi_files) == len(splits)
    return midi_files, splits

def process_midi_file_safe_v2(midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold = None):
    try:
        out = process_midi_file_v2(midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold)
        if out is not None:
            for out_chunk in out:
                if out_chunk is not None:
                    # Salva i file .npy solo se il chunk è valido
                    np.save(out_chunk['file'], out_chunk['compounds'])
        return out
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f'Failed to process {midi_file}:\n')
            log.write(traceback.format_exc())
            log.write('\n')
        return [None] # Ritorna una lista con None per segnalare il fallimento

def detect_large_timeshifts_and_durations(compounds, onset_vocab_size, dur_vocab_size):
    if not compounds:
        return False, False, Counter(), Counter()
    onsets = [c[0] for c in compounds]
    onsets_padded = [0] + onsets
    timeshift_counter = Counter([onsets_padded[i+1] - onsets_padded[i] for i in range(len(onsets_padded) - 1)]) 
    duration_counter = Counter([c[1] for c in compounds])
    onsets_exceed_vocab_size = any(key > onset_vocab_size-3 for key in timeshift_counter.keys())
    duration_exceed_vocab_size = any(key > dur_vocab_size-3 for key in duration_counter.keys())
    return onsets_exceed_vocab_size, duration_exceed_vocab_size, timeshift_counter, duration_counter

def filter_large_ts_dur(compounds, output_file_path, split, onset_vocab_size, dur_vocab_size, log_file):
    if not compounds:
        return None
        
    onsets_exceed_vocab_size, duration_exceed_vocab_size, timeshift_counter, duration_counter = detect_large_timeshifts_and_durations(compounds, onset_vocab_size, dur_vocab_size)

    if onsets_exceed_vocab_size or duration_exceed_vocab_size:
        # Logica di logging...
        return None
    else:
        return {
            'file': output_file_path,
            'compounds': compounds,
            'split': split,
            'timeshifts':dict(timeshift_counter),
            'durations':dict(duration_counter),           
            'length_token': len(compounds),
            'length_duration': compounds[-1][0]+compounds[-1][1] if compounds else 0,
        }

def process_midi_file_v2(midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold = None):
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not initialized in this worker process.")
    
    compounds = tokenizer.midi_to_compound(midi_file)
    if not compounds:
        return []

    output_file_name = os.path.basename(midi_file).replace(".midi", ".npy").replace(".mid", ".npy")
    output_file_path = os.path.join(output_folder, output_file_name)
    
    list_of_compounds = []
    if silence_threshold:
        list_of_compounds = chunk_compounds(compounds, threshold=silence_threshold)
    else:
        list_of_compounds = [compounds]

    processed_chunks = []
    if len(list_of_compounds) == 1:
        chunk_data = filter_large_ts_dur(list_of_compounds[0], output_file_path, split, onset_vocab_size, dur_vocab_size, log_file)
        if chunk_data:
            processed_chunks.append(chunk_data)
    else:
        for i, chunk in enumerate(list_of_compounds):
            chunk_file_path = output_file_path.replace('.npy', f'_{i}.npy')
            chunk_data = filter_large_ts_dur(chunk, chunk_file_path, split, onset_vocab_size, dur_vocab_size, log_file)
            if chunk_data:
                processed_chunks.append(chunk_data)
    
    return processed_chunks if processed_chunks else None


# Main script execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MIDI files for the MIDI dataset.')
    parser.add_argument('--dataset_name', type=str, help='Dataset Name')
    parser.add_argument('--dataset_folder', type=str, help='Path to the dataset folder containing MIDI files.')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where processed files will be saved.')
    parser.add_argument('--model_config', type=str, help='Model configuration file that decides the vocab size')
    parser.add_argument('--train_test_split_file', type=lambda x: None if x == "None" else str(x), help='Path to the split file.')
    parser.add_argument('--train_ratio', type=float, help='Training/Total')
    parser.add_argument('--ts_threshold', type=lambda x: None if x == "None" else int(x), help='If Timeshift exceeds this value, chunk the file')
    args = parser.parse_args()

    midi_output_folder = os.path.join(args.output_folder, "processed")
    log_file = os.path.join(args.output_folder, "failed_midi_files.log")
    csv_file_path = os.path.join(args.output_folder, "train_test_split.csv")
    os.makedirs(midi_output_folder, exist_ok=True)
    
    if args.train_ratio and args.train_ratio == 1:
        midi_files = find_midi_files(args.dataset_folder)
        splits = ['train']*len(midi_files)
    elif args.train_ratio is not None and args.train_ratio == 0:
        midi_files = find_midi_files(args.dataset_folder)
        splits = ['test']*len(midi_files)
    else:
        if args.train_test_split_file:
            midi_files, splits = find_midi_files_from_file(args.dataset_name, args.train_test_split_file, args.dataset_folder)
        else:
            midi_files = find_midi_files(args.dataset_folder)
            train_files, test_files = train_test_split(midi_files, train_size=args.train_ratio, random_state=42)
            splits = ['train'] * len(train_files) + ['test'] * len(test_files)
            midi_files = train_files + test_files

    with open(args.model_config, 'r') as file:
        data = json.load(file)
        onset_vocab_size = data.get("onset_vocab_size")
        dur_vocab_size = data.get("dur_vocab_size")
        octave_vocab_size = data.get("octave_vocab_size")
        pitch_class_vocab_size = data.get("pitch_class_vocab_size")
        instrument_vocab_size = data.get("instrument_vocab_size")
        velocity_vocab_size = data.get("velocity_vocab_size")
        assert onset_vocab_size and dur_vocab_size

    print(f"Processing {len(midi_files)} files using {num_cores} CPUs. Max timeshift: {onset_vocab_size-3}, max duration: {dur_vocab_size-3}")
    
    initializer_args = (onset_vocab_size, dur_vocab_size, octave_vocab_size, pitch_class_vocab_size, instrument_vocab_size, velocity_vocab_size)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file_base_name', 'split', 'length', 'duration'])

        success_count = 0
        fail_count = 0

        with ProcessPoolExecutor(max_workers=num_cores, initializer=init_tokenizer_worker, initargs=initializer_args) as executor:
            # Sottometti tutti i task all'executor
            futures = {
                executor.submit(process_midi_file_safe_v2, midi_file, split, onset_vocab_size, dur_vocab_size, midi_output_folder, log_file, args.ts_threshold): midi_file 
                for midi_file, split in zip(midi_files, splits)
            }
            
            # Usa tqdm per creare la barra di avanzamento
            pbar = tqdm(total=len(midi_files), desc="Processing MIDI files")

            for future in as_completed(futures):
                result = future.result()
                
                if result == [None] or result is None:
                    fail_count += 1
                else:
                    success_count += 1
                    for sublist in result:
                        if sublist is not None:
                            csv_writer.writerow([os.path.basename(sublist['file']), sublist['split'], sublist['length_token'], sublist['length_duration']])
                
                # Aggiorna la barra di avanzamento e i contatori
                pbar.update(1)
                pbar.set_postfix({'Success': success_count, 'Failed': fail_count})

            pbar.close()

    processed_count = success_count
    total_files = len(midi_files)
    success_ratio = (processed_count / total_files * 100) if total_files > 0 else 0
    print(f'\nProcessing complete. Results saved to {csv_file_path}')
    print(f'Successfully processed: {success_count}, Failed: {fail_count}')
    print(f'Success ratio: {success_ratio:.2f}%')
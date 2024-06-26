
# input: dataset name, config, working_folder, output_folder (decide npy or T5) 
# a folder of midi --> filter based on whether the file contains large onsets etc --> process, save and append to csv --> sample a portion and get stats 
# from src.llama_recipes.datasets.lakh_dataset import midi_to_compound


# midi_files = []
# #1. find all midi_files under a folder 
# #2.  for each file analyze using this function: midi_to_compound. it will return a list of compound tokens (onset, duration, octave, pitch_class, instrument, velocity) 1. Use try and except keep track of how many files failed log the failure message in a seperate file 2. For all success file:  Do a data analysis of a. distribution of file lengths, onsets, duration
import os
import matplotlib.pyplot as plt
# from src.llama_recipes.datasets.lakh_dataset import midi_to_compound
from src.llama_recipes.datasets.music_tokenizer import MusicTokenizer
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
import time
import math
import json
import multiprocessing
from collections import Counter
import numpy as np
from collections import defaultdict
import argparse
import csv
from sklearn.model_selection import train_test_split
import h5py
num_cores = multiprocessing.cpu_count()

save_folder = "/data/home/acw753/musicllama/dataset_analysis"
# Directory containing the MIDI files
midi_folder = '/data/scratch/acw753/lakhmidi'
# Log file for failed analyses

progress_log_file = f'{save_folder}/progress.txt'
dataset_name = "Lakhmidi"

# Function to find all MIDI files under a folder
def find_midi_files(folder):
    midi_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    return midi_files

def process_midi_file(midi_file, onset_vocab_size, dur_vocab_size, output_folder, log_file, save_format = "npy"):
    try:
        # Analyze the MIDI file
        compounds = MusicTokenizer().midi_to_compound(midi_file)

        #Filter out files with large onsets or durations
        onsets_counter = Counter([math.ceil(math.log2(c[0])) if c[0] != 0 else 1 for c in compounds])
        duration_counter = Counter([c[1] for c in compounds])

        # Check if any key in onsets_counter is larger than onset_vocab_size
        onsets_exceed_vocab_size = any(key > onset_vocab_size-1 for key in onsets_counter.keys())

        # Check if any key in duration_counter is larger than dur_vocab_size
        duration_exceed_vocab_size = any(key > dur_vocab_size-1 for key in duration_counter.keys()) #if dur_vocab_size=1026, largest value allowed should be 1025

        if onsets_exceed_vocab_size or duration_exceed_vocab_size:
            with open(log_file, 'a') as log:
                log.write(f'Failed to process {midi_file}:\n')
                log.write(f'{midi_file} contains large onsets: {onsets_exceed_vocab_size}, large durations: {duration_exceed_vocab_size}\n')
            return None
        #save to npy or flat t5
        # output_file_folder = os.path.join(output_folder, midi_file.split("/")[1:-1])
        # os.makedirs(output_file_folder, exist_ok = True)                              
        output_file_name = midi_file.split("/")[-1].replace('.mid', f'.{save_format}')
        output_file_path = os.path.join(output_folder, output_file_name)
        if os.path.isfile(output_file_path):
            print(f"warning... {output_file_path} already exists")
        if save_format == "npy":
            np.save(output_file_path, np.array(compounds))
        elif save_format == "h5":
            with h5py.File(output_file_path, 'w') as hf:
                hf.create_dataset('compounds', data=np.array(compounds))
        return {
            'file': output_file_path,
            'onsets':onsets_counter,
            'durations':duration_counter,           
            'length': len(compounds)
        }
    except Exception as e:
        # Log failure with full traceback
        with open(log_file, 'a') as log:
            log.write(f'Failed to process {midi_file}:\n')
            log.write(traceback.format_exc())
            log.write('\n')
        return None






def analyze_and_plot_beta0(success_data):

    # Merge dictionaries and aggregate counts
    onset_counts = {}
    duration_counts = {}
    file_length_counts = {}
    # file_lengths = [data['length'] for data in success_data]
    for data in success_data:
        # Update onset counts
        for onset, count in data['onsets'].items():
            if onset in onset_counts:
                onset_counts[onset] += count
            else:
                onset_counts[onset] = count

        # Update duration counts
        for duration, count in data['durations'].items():
            if duration in duration_counts:
                duration_counts[duration] += count
            else:
                duration_counts[duration] = count
        if data['length'] in file_length_counts:
            file_length_counts[data['length']] +=1
        else:
            file_length_counts[data['length']] = 1
    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(list(file_length_counts.keys()), bins=range(0, max(file_length_counts.keys()) + 100, 100), weights=list(file_length_counts.values()), edgecolor='black')
    plt.xscale('log')  # Set the y-axis to logarithmic scale

    plt.title('File Len Histogram')
    plt.xlabel('File Len')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{save_folder}/file_lengths_distribution.png', dpi=200)
    plt.close()

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(list(onset_counts.keys()), bins=range(0, max(onset_counts.keys()) + 100, 100), weights=list(onset_counts.values()), edgecolor='black')
    plt.xscale('log')  # Set the y-axis to logarithmic scale

    plt.title('Onset Histogram')
    plt.xlabel('Onset')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{save_folder}/onsets_distribution.png', dpi=200)
    plt.close()


    plt.figure(figsize=(12, 6))
    plt.hist(list(duration_counts.keys()), bins=range(0, max(duration_counts.keys()) + 10, 10), weights=list(duration_counts.values()), edgecolor='black')
    plt.xscale('log')  # Set the y-axis to logarithmic scale
    plt.title('Duration Histogram')
    plt.xlabel('Duration')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{save_folder}/durations_distribution.png', dpi=200)
    plt.close()


def analyze_and_plot_beta1(success_data):

    # Initialize count dictionaries
    onset_counts = defaultdict(int)
    duration_counts = defaultdict(int)
    file_length_counts = defaultdict(int)

    def update_counts(data):
        local_onset_counts = defaultdict(int)
        local_duration_counts = defaultdict(int)
        local_file_length_counts = defaultdict(int)

        # Update onset counts
        for onset, count in data['onsets'].items():
            local_onset_counts[onset] += count

        # Update duration counts
        for duration, count in data['durations'].items():
            local_duration_counts[duration] += count

        # Update file length counts
        local_file_length_counts[data['length']] += 1

        return local_onset_counts, local_duration_counts, local_file_length_counts
    # Measure start time
    start_time = time.time()
    # Use ThreadPoolExecutor to process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(update_counts, data) for data in success_data]
        
        for future in concurrent.futures.as_completed(futures):
            local_onset_counts, local_duration_counts, local_file_length_counts = future.result()
            
            # Merge local counts into global counts
            for key, value in local_onset_counts.items():
                onset_counts[key] += value
            for key, value in local_duration_counts.items():
                duration_counts[key] += value
            for key, value in local_file_length_counts.items():
                file_length_counts[key] += value

    # Convert defaultdicts to regular dictionaries if needed
    onset_counts = dict(onset_counts)
    duration_counts = dict(duration_counts)
    file_length_counts = dict(file_length_counts)


    # Measure end time
    end_time = time.time()

    # Calculate and print execution time
    execution_time = end_time - start_time
    print(f"Parallel processing executed in {execution_time:.2f} seconds.")

    start_time = time.time()
    plt.figure(figsize=(12, 6))
    plt.hist(list(duration_counts.keys()), bins=range(0, max(duration_counts.keys()) + 10, 10), weights=list(duration_counts.values()), edgecolor='black')
    plt.xscale('log')  # Set the y-axis to logarithmic scale
    plt.title('Duration Histogram')
    plt.xlabel('Duration')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{save_folder}/durations_distribution2.png', dpi=200)
    plt.close()
    end_time = time.time()
    print(f"Duration histogram executed in {execution_time:.2f} seconds.")
    
    start_time = time.time()
    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(list(file_length_counts.keys()), bins=range(0, max(file_length_counts.keys()) + 100, 100), weights=list(file_length_counts.values()), edgecolor='black')
    plt.xscale('log')  # Set the y-axis to logarithmic scale

    plt.title('File Len Histogram')
    plt.xlabel('File Len')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{save_folder}/file_lengths_distribution2.png', dpi=200)
    plt.close()
    end_time = time.time()
    print(f"Length histogram executed in {execution_time:.2f} seconds.")

    # print(f"onset_counts:{onset_counts}, file_lengths:{file_lengths}")
    # Plot distribution of file lengths
    # plt.figure(figsize=(12, 6))
    # plt.hist(file_lengths, bins=50, alpha=0.75, color='blue', edgecolor='black')
    # # plt.xscale('log')  # Set the y-axis to logarithmic scale
    # plt.title('Distribution of MIDI File Lengths')
    # plt.xlabel('Number of Compounds')
    # plt.ylabel('Frequency')
    # plt.savefig(f'{save_folder}/file_lengths_distribution.png', dpi=200)
    # plt.close()

    # # Create a list of onsets and their corresponding counts
    # onsets = []
    # counts = []
    # for onset, count in onset_counts.items():
    #     onsets.append(onset)
    #     counts.append(count)

    # # Plot the histogram
    # plt.figure(figsize=(12, 6))
    # plt.bar(onsets, counts, width=100, align='edge', edgecolor='black', alpha=0.75)

    # plt.xlabel('Onset')
    # plt.ylabel('Count')
    # plt.title('Histogram of Onset Counts')
    # plt.grid(True)
    # plt.savefig(f'{save_folder}/onsets_distribution.png')
    # plt.close()
    start_time = time.time()
    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(list(onset_counts.keys()), bins=range(0, max(onset_counts.keys()) + 100, 100), weights=list(onset_counts.values()), edgecolor='black')
    plt.xscale('log')  # Set the y-axis to logarithmic scale

    plt.title('Onset Histogram')
    plt.xlabel('Onset')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{save_folder}/onsets_distribution2.png', dpi=200)
    plt.close()
    end_time = time.time()
    print(f"Onset histogram executed in {execution_time:.2f} seconds.")

    # plt.figure(figsize=(12, 6))
    # plt.hist(list(onset_counts.keys()), bins=100, weights=list(onset_counts.values()), edgecolor='black')
    # plt.title('Onset Histogram')
    # plt.xlabel('Onset')
    # plt.ylabel('Count')
    # plt.savefig(f'{save_folder}/onsets_distribution.png')
    # plt.close()


    # Create a list of onsets and their corresponding counts
    # durations = []
    # counts = []
    # for duration, count in duration_counts.items():
    #     durations.append(onset)
    #     counts.append(count)

    # # Plot the histogram
    # plt.figure(figsize=(12, 6))
    # plt.bar(durations, counts, width=100, align='edge', edgecolor='black', alpha=0.75)

    # plt.xlabel('Duration')
    # plt.ylabel('Count')
    # plt.title('Histogram of Duration Counts')
    # plt.grid(True)
    # plt.savefig(f'{save_folder}/durations_distribution.png')
    # plt.close()
    # plt.figure(figsize=(12, 6))
    # plt.hist(list(duration_counts.keys()), bins=100, weights=list(duration_counts.values()), edgecolor='black')
    # plt.title('Duration Histogram')
    # plt.xlabel('Duration')
    # plt.ylabel('Count')
    # plt.savefig(f'{save_folder}/durations_distribution.png')
    # plt.close()


def update_counts(data):
    onset_counts = {}
    duration_counts = {}
    file_length_counts = {}

    for onset, count in data['onsets'].items():
        if onset in onset_counts:
            onset_counts[onset] += count
        else:
            onset_counts[onset] = count

    for duration, count in data['durations'].items():
        if duration in duration_counts:
            duration_counts[duration] += count
        else:
            duration_counts[duration] = count

    if data['length'] in file_length_counts:
        file_length_counts[data['length']] += 1
    else:
        file_length_counts[data['length']] = 1

    return onset_counts, duration_counts, file_length_counts

def merge_dictionaries(dicts):
    result = {}
    for d in dicts:
        
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
    
    return result



def merge_dicts_chunk(chunk):
    result = {}
    for d in chunk:
        for key, value in d.items():
            result[key] = result.get(key, 0) + value
    return result

def merge_dictionaries_parallel(dicts):
    # Split dicts into chunks for parallel processing
    num_chunks = len(dicts)
    chunk_size = max(num_chunks // num_cores, 1)
    # chunk_size = 300
    chunks = [dicts[i:i + chunk_size] for i in range(0, num_chunks, chunk_size)]

    # Merge chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # merged_dicts = list(executor.map(merge_dicts_chunk, chunks))
        merged_dicts = list(tqdm(executor.map(merge_dicts_chunk, chunks), total=len(chunks), desc='Merging Dict'))
    print(f"merged_dicts length{len(merged_dicts)}")
    # Combine results from parallel execution
    result = {}
    for merged_dict in merged_dicts:
        for key, value in merged_dict.items():
            result[key] = result.get(key, 0) + value
    return result

def plot_histogram_old(data, bins, xlabel, ylabel, title, filename, xscale='linear'):
    plt.figure(figsize=(12, 6))
    keys = list(data.keys())  # Convert keys to list
    values = list(data.values())  # Convert values to list
    plt.hist(keys, bins=bins, weights=values, edgecolor='black')
    plt.xscale(xscale)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, filename), dpi=200)
    plt.close()

def calculate_histogram_func(keys, data, bins):
    hist, _ = np.histogram(keys, bins=bins, weights=[data[key] for key in keys])
    return hist

def plot_histogram_serial(data, bins, xlabel, ylabel, title, filename, xscale='linear', threshold=0.9, if_plot=True):
    # Define batch size for sequential processing
    batch_size = 300
    
    # Initialize an empty list to store histograms
    histograms = []
    
    # Process data sequentially in batches
    for i in range(0, len(data), batch_size):
        hist = process_chunk(data, list(data.keys()), bins, i, batch_size)  # Process current batch
        print(f"hist:{hist}")
        histograms.append(hist)  # Append histogram for current batch
    
    # Combine histograms from all batches
    combined_hist = np.sum(histograms, axis=0)


    # Calculate the cumulative histogram
    cumulative_hist = np.cumsum(combined_hist)

    # Find the total count of all data points
    total_count = cumulative_hist[-1]

    # Find the value corresponding to the desired percentile
    target_count = int(total_count * threshold)
    output = bins[np.argmax(cumulative_hist >= target_count)]

    if if_plot:
        plt.figure(figsize=(12, 6))
        plt.bar(bins[:-1], combined_hist, width=np.diff(bins), edgecolor='black')
        plt.axvline(x=output, color='r', linestyle='--', label=f'x = {output}, percentile: {threshold:.2f}')
        plt.xscale(xscale)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_folder, filename), dpi=200)
        print(f"file saved to: ", os.path.join(save_folder, filename))
        plt.close()
    return output
def plot_histogram(data, bins, xlabel, ylabel, title, filename, xscale='linear', threshold = 0.9,if_plot = True):
    len_data = len(data)
    batch_size = 300
    # Create a shared dictionary
    manager = multiprocessing.Manager()
    shared_data = manager.dict(data)
    shared_keys = manager.list(data.keys())
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        # for _ in range(num_cores):
        for i in range(0, len_data, batch_size):
            future = executor.submit(process_chunk, shared_data, shared_keys, bins, i, batch_size)
            futures.append(future)
        histograms = []
        for future in tqdm(futures, total=num_cores):
            out = future.result()
            print("out!", out)
            histograms.append(out)
        # histograms = [future.result() for future in tqdm(futures, total=num_cores)]

    combined_hist = np.sum(histograms, axis=0)


    # Calculate the cumulative histogram
    cumulative_hist = np.cumsum(combined_hist)

    # Find the total count of all data points
    total_count = cumulative_hist[-1]

    # Find the value corresponding to the desired percentile
    target_count = int(total_count * threshold)
    output = bins[np.argmax(cumulative_hist >= target_count)]

    if if_plot:
        plt.figure(figsize=(12, 6))
        plt.bar(bins[:-1], combined_hist, width=np.diff(bins), edgecolor='black')
        plt.axvline(x=output, color='r', linestyle='--', label=f'x = {output}, percentile: {threshold:.2f}')
        plt.xscale(xscale)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_folder, filename), dpi=200)
        print(f"file saved to: ", os.path.join(save_folder, filename))
        plt.close()
    return output
def process_chunk(data, keys, bins, starting_idx, batch_size):
    end_idx = min(starting_idx + batch_size, len(keys)) 
    chunks = keys[starting_idx:end_idx]
    hist, _ = np.histogram(chunks, bins=bins, weights=[data[key] for key in chunks])
    return hist


def analyze(success_data):
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # futures = [executor.submit(update_counts, data) for data in success_data]
        # results = [future.result() for future in as_completed(futures)]
        futures = [executor.submit(update_counts, data) for data in success_data]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing data'):
            results.append(future.result())
    onset_counts_list, duration_counts_list, file_length_counts_list = zip(*results)
    # print(f"check onset_counts_list:{onset_counts_list}, file_length_counts_list:{file_length_counts_list}")
    onset_counts = merge_dictionaries_parallel(onset_counts_list)
    duration_counts = merge_dictionaries_parallel(duration_counts_list)
    file_length_counts = merge_dictionaries_parallel(file_length_counts_list)
    return onset_counts, duration_counts, file_length_counts


# Main script execution
if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process MIDI files for the Lakh MIDI dataset.')

    # Define and Parse the arguments
    parser.add_argument('--dataset_folder', type=str, help='Path to the dataset folder containing MIDI files.')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where processed files will be saved.')
    parser.add_argument('--if_sample', type=bool, help='Whether to sample the dataset.')
    parser.add_argument('--model_config', type=str, help='Model configuration file that decides the vocab size')
    parser.add_argument('--train_ratio', type=float, help='Training/Total')
    parser.add_argument('--save_format', type=str, help='File save format', choices=['npy', 'h5'])

    args = parser.parse_args()

    #get file paths
    midi_output_folder = args.output_folder+"/processed"
    log_file = args.output_folder + "/failed_midi_files.log"
    csv_file_path = args.output_folder+"/train_test_split.csv"
    train_stats_file = args.output_folder+"/train_tokens_stats.json"
    test_stats_file = args.output_folder+"/test_tokens_stats.json"
    os.makedirs(midi_output_folder, exist_ok=True)
    
    #find all midi files
    midi_files = find_midi_files(args.dataset_folder)
    print(f"{len(midi_files)} midi files found!")
    
    #determine vocab size
    with open(args.model_config, 'r') as file:
        data = json.load(file)
        onset_vocab_size = data.get("onset_vocab_size", None)
        dur_vocab_size = data.get("dur_vocab_size", None)
        assert onset_vocab_size and dur_vocab_size
    print(f"tokenizer config: onset_vocab_size: {onset_vocab_size}, dur_vocab_size: {dur_vocab_size}")
    print(f"processing using {num_cores} cpus")
    
    #process all midi files
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_midi_file, midi_files, [onset_vocab_size]*len(midi_files), [dur_vocab_size]*len(midi_files), [midi_output_folder]*len(midi_files), [log_file]*len(midi_files), [args.save_format]*len(midi_files)), total=len(midi_files), desc='Processing MIDI files'))

    success_data = []
    failures = 0
    for result in results: 
        if result is not None:
            success_data.append(result)
        else:
            failures += 1
    print(f'Processed {len(midi_files)} files with {failures} failures.')

    # Aggregate results, train test split, save to csv

    train_results, test_results = train_test_split(success_data, test_size=1-args.train_ratio, random_state=42)

    # Create a list to store the CSV rows
    csv_rows = []

    # Add rows to the CSV for training results
    for result in train_results:
        file_base_name = os.path.basename(result['file'])
        csv_rows.append([file_base_name, 'train'])

    # Add rows to the CSV for testing results
    for result in test_results:
        file_base_name = os.path.basename(result['file'])
        csv_rows.append([file_base_name, 'test'])

    # Write CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file_base_name', 'split'])  # Write header
        csv_writer.writerows(csv_rows)


    # Perform data analysis and plotting
    print("analyzing train data stats...")
    onset_stats_train, duration_stats_train, file_length_stats_train = analyze(train_results)
    print("analyzing test data stats...")
    onset_stats_test, duration_stats_test, file_length_stats_test = analyze(test_results)

    # Save the dictionaries to JSON files
    with open(train_stats_file, "w") as f:
        json.dump({"num_files":len(train_results), "onset_stats": onset_stats_train, "duration_stats": duration_stats_train, "len_stats": file_length_stats_train}, f)

    with open(test_stats_file, "w") as f:
        json.dump({"num_files":len(test_results), "onset_stats": onset_stats_test, "duration_stats": duration_stats_test, "len_stats": file_length_stats_test}, f)

    print("train and test stats saved!")

    
    # dur_val = plot_histogram_serial(duration_counts, bins=range(0, max(duration_counts.keys()) + 20, 20), xlabel='Duration', ylabel='Count', title='Duration Histogram', filename=f'{save_folder}/durations_distribution3.png', xscale='log')
    # onset_val = plot_histogram(onset_counts, bins=range(0, max(onset_counts.keys()) + 2, 2), xlabel='Onset', ylabel='Count', title='Onset Histogram', filename=f'{save_folder}/onsets_distribution3.png', xscale='log')
    # file_len_val = plot_histogram(file_length_counts, bins=range(0, max(file_length_counts.keys()) + 100, 100), xlabel='File Len', ylabel='Count', title='File Len Histogram', filename=f'{save_folder}/file_lengths_distribution3.png', xscale='log')

    # #save statistics: 
    # stats = {"onsets": onset_val, "durations": dur_val, "file_len": file_len_val, "failure_ratio": round(failures/len(midi_files),2)}
    # stats_filename = f"{save_folder}/stats.json"

    # with open(stats_filename, "w") as f:
    #     json.dump(stats_filename, f)

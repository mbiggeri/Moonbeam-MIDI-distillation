# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class lakhmidi_dataset:
    dataset: str = "lakhmidi_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/data/scratch/acw753/lakhmidi_processed"
    csv_file: str = "/data/scratch/acw753/lakhmidi_processed/train_test_split.csv"

@dataclass
class merge_dataset:
    dataset: str = "merge_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/data/scratch/acw753/processed_midi_onset_4099_dur_4099"
    csv_file: str = "/data/scratch/acw753/processed_midi_onset_4099_dur_4099/merged_train_test_split.csv"

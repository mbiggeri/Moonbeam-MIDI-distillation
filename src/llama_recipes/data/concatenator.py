# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain
import multiprocessing as mp
import mmap
import concurrent.futures
from torch.utils.data import Dataset
import numpy as np
import os
import pickle

class ConcatDataset_hybrid_padding_concatenating(Dataset):
    def __init__(self, dataset, chunk_size=4096, split = "train",data_dir = None):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        self._process_samples()
    
    def _process_samples(self):
        buffer = []  # Holds temporary samples before forming a chunk
        buffer_length = 0  # Tracks current buffer length
        sample_count = 1  # Unique counter for attention mask

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            length = len(sample['input_ids'])
            
            if length == self.chunk_size:
                sample['attention_mask'] = [sample_count] * length
                self.samples.append(sample)
                sample_count += 1
            elif length > self.chunk_size:
                sample['input_ids'] = sample['input_ids'][:self.chunk_size]
                sample['labels'] = sample['labels'][:self.chunk_size]
                sample['attention_mask'] = [sample_count] * self.chunk_size
                self.samples.append(sample)
                sample_count += 1
            else:
                if buffer_length + length <= self.chunk_size:
                    buffer.append((sample, sample_count))  # Store sample with its sample count
                    buffer_length += length
                    sample_count += 1
                else:
                    # Finalize current buffer before adding new sample
                    self._finalize_chunk(buffer, buffer_length)
                    buffer = [(sample, sample_count)]
                    buffer_length = length
                    sample_count += 1
        
        if buffer:  # Handle remaining buffer
            self._finalize_chunk(buffer, buffer_length)
    
    def _finalize_chunk(self, buffer, buffer_length):
        concatenated_sample = {
            'input_ids': [],
            'labels': [],
            'attention_mask': []
        }
        
        for sample, sample_id in buffer:
            concatenated_sample['input_ids'].extend(sample['input_ids'])
            concatenated_sample['labels'].extend(sample['labels'])
            concatenated_sample['attention_mask'].extend([sample_id] * len(sample['input_ids']))
        
        # Apply padding only if necessary
        pad_length = self.chunk_size - buffer_length
        if pad_length > 0:
            concatenated_sample['input_ids'].extend([[0, 0, 0, 0, 0, 0]] * pad_length)
            concatenated_sample['labels'].extend([[0, 0, 0, 0, 0, 0, 0]] * pad_length)
            concatenated_sample['attention_mask'].extend([max(sample_id for _, sample_id in buffer) * 2] * pad_length)
        
        self.samples.append(concatenated_sample)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)
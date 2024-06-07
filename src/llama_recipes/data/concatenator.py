# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        sample_count = 0  # Initialize a counter for the samples
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            sample['attention_mask'] = [sample_count]*len(sample['attention_mask']) #In the model, each sample will only attend to itself according to the "sample_count" value
            buffer = {k: v + sample[k] for k, v in buffer.items()}            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
            sample_count += 1  # Increment the sample counter

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

import copy
import json

import torch
from torch.utils.data import Dataset
import glob
import mido
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
import tqdm
import numpy as np
import pandas as pd
import os
def decimal_to_binary_batch(dec, bits=20):
    """
    Converts a batch of decimal numbers to their binary representations.

    Args:
        dec (torch.Tensor): A tensor containing decimal numbers.
        bits (int, optional): The desired number of bits in the binary representation. Default is 4.

    Returns:
        torch.Tensor: A tensor containing the binary representations of the input decimal numbers.
    """
    # Convert the input tensor to a long tensor for bitwise operations
    dec = dec.long()

    # Create a mask to extract the binary digits
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(dec.device)

    # Apply the mask to extract the binary digits
    binary = (dec.unsqueeze(-1) & mask).bool().to(torch.int8)

    return binary

class LakhDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        assert partition=="train" or partition=="test"
        self.data_dir = dataset_config.data_dir
        split_data = pd.read_csv(dataset_config.csv_file)
        file_basenames = split_data['file_base_name'].values
        splits = split_data['split'].values        
        self.file_basenames = [f for f, s in zip(file_basenames, splits) if s == partition] #TODO: this is only a toy dataset! remove in the future
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, index):
        raw_tokens = np.load(os.path.join(self.data_dir, "processed",self.file_basenames[index]))
        
        
        encoded_tokens = torch.tensor(self.tokenizer.encode_series(raw_tokens)) #SOS and EOS are added in the tokenizer
        #Label is constructed as follows: SOS token is encoded seperately: onset_SOS = [10000000...000], dur_SOS = dur_vocab_size - 1, ... 
        sos_label, eos_label = self.tokenizer.sos_label, self.tokenizer.eos_label
        encoded_tokens_label_wo_sos = torch.concat([decimal_to_binary_batch(encoded_tokens[1:-1, 0], bits=self.tokenizer.onset_vocab_size), encoded_tokens[1:-1, 1:]], dim = -1)

        encoded_tokens_label = torch.concat([sos_label.unsqueeze(0), encoded_tokens_label_wo_sos, eos_label.unsqueeze(0)], dim = 0)
        encoded_tokens, encoded_tokens_label = encoded_tokens.tolist(), encoded_tokens_label.tolist()

        return {
            "input_ids": encoded_tokens,
            "labels": encoded_tokens_label,
            "attention_mask":[True for _ in range(len(encoded_tokens))] #all True if no padding in the end
        }


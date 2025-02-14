# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire
import pandas as pd
import numpy as np
import os
import re
# from llama import Dialog, Llama
from generation import MusicLlama
import random

def main(
    ckpt_dir: str,
    csv_file: str,
    tokenizer_path: str,
    model_config_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    prompt_len: int = 5,
    num_test_data: int = 50,
    max_gen_len: Optional[int] = None,
    finetuned_PEFT_weight_path: Optional[str] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    # Set the random seed for CPU and GPU
    seed = 42
    import torch
    torch.manual_seed(seed)
    random.seed(seed)  # You can choose any seed value, 42 is commonly used
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.    


    generator = MusicLlama.build_emo_con_gen(
        ckpt_dir=ckpt_dir,
        model_config_path = model_config_path, #TODO: test whether this is successful
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        finetuned_PEFT_weight_path = finetuned_PEFT_weight_path
    ) # this is a model actually

    #1. read the csv file; 2. read the header with split = test and get the npy file path; 3. load the npy file; 4. get the first 10 tokens;
    prompts = []
    emotion_labels = []
    emotion_ids2token = {0:generator.tokenizer.emotion_token_4Q1, 1:generator.tokenizer.emotion_token_4Q2, 2:generator.tokenizer.emotion_token_4Q3, 3:generator.tokenizer.emotion_token_4Q4}
    for _ in range(num_test_data):
        rand_id = random.randint(0, 3)
        encoded_tokens = generator.tokenizer.encode_series_con_gen_emotion([], if_add_sos = True, if_add_eos = False, emotion_token_4Q = emotion_ids2token[rand_id]) #SOS and EOS are added in the tokenizer
        prompts.append(encoded_tokens)
        emotion_labels.append(rand_id)
    print(f"prompts: {prompts[:20]}") #TODO: FIND IN THE ORIGINAL TRAINING SCRIPT WHERE emotion_token_pos_map IS CALLED!! THEN ADD IT TO THE 
    """
    df = pd.read_csv(csv_file)

    # 2. Find the row where 'split' is 'test' and get the corresponding .npy file path
    test_filenames = df[df['split'] == 'train']['file_base_name'].tolist() #change from test to train, since there's really no test data in LLM
    test_filenames_sampled = random.sample(test_filenames, num_test_data)

    for filename in test_filenames_sampled:
        test_data = np.load(os.path.join(os.path.dirname(csv_file), filename))
        test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
        prompts.append(test_data_with_sos[:prompt_len])"""
    
    #1. Try unconditional generation without any prompt 


    results = generator.music_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        condition_token_length = 2 #remove sos token and emotion token
    )
    
    save_folder = os.path.join(os.path.dirname(finetuned_PEFT_weight_path), os.path.basename(ckpt_dir), f"temperature_{temperature}_top_p_{top_p}")
    os.makedirs(save_folder, exist_ok=True)
    for i, (dialog, result, label) in enumerate(zip(prompts, results, emotion_labels)):
        try:
            for msg in dialog:
                print(f"msg: {msg}")

            epoch_step = re.search(r'(\d+-\d+)\.pt$', ckpt_dir).group(1)
            for j, sub_mid in enumerate(result['generation']['content']):
                sub_mid.save(f'{save_folder}/{epoch_step}_{str(i)}_{str(j)}_emotion_{str(label+1)}.mid')
            print(f"midis saved at {save_folder}/{epoch_step}_{str(i)}_x.mid")
            #Also save the prompt 
            print("\n==================================\n")
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    fire.Fire(main)

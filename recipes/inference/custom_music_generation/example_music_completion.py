# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

# from llama import Dialog, Llama
from generation import MusicLlama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    model_config_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.    


    generator = MusicLlama.build(
        ckpt_dir=ckpt_dir,
        model_config_path = model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    ) # this is a model actually


    #TODO: WRITE PROMPTS -- accept a couple of formats: 1. midi + num_token or length 2. npy file 
    # prompts = [[[0, 0, 0, 0, 0, 0],
    #             [0, 384, 2, 11, 0, 100],
    #             [0, 384, 3, 6, 0, 100],
    #             [0, 384, 2, 11, 0, 100],
    #             [0, 384, 3, 6, 0, 100],
    #             [0, 48, 4, 6, 0, 100],
    #             [0, 192, 3, 11, 0, 100]],
    #            [[0, 0, 0, 0, 0, 0],
    #             [111, 16, 3, 5, 128, 100],
    #             [111, 16, 2, 11, 128, 100],
    #             [126, 16, 3, 5, 128, 100],
    #             [126, 16, 2, 11, 128, 100]]
    #            ] #[[0, 0, 0, 0, 0, 0]]
    
    # prompts = [[[0, 0, 0, 0, 0, 0],
    #             [111, 16, 3, 5, 128, 100],
    #             [111, 16, 2, 11, 128, 100],
    #             [126, 16, 3, 5, 128, 100],
    #             [126, 16, 2, 11, 128, 100]]
    #            ] #[[0, 0, 0, 0, 0, 0]]

    prompts = [[[0, 0, 0, 0, 0, 0], 
        [  0,   4,   2,   7, 128,  81],
       [ 50,   4,   2,   7, 128,  69],
       [100,   4,   2,   7, 128,  68],
       [150,   4,   2,   7, 128,  71],
       [200,  25,   5,   2,  61, 100],
       [200,  25,   5,   6,  61, 100],
       [200,  25,   5,   9,  61, 100],
       [200,  75,   3,   7,  27, 100],
       [200,  75,   2,   7,  33, 109],
       [200,  73,   4,   7,  56,  81]]]


    """
    from training set:



array([[  0,   4,   2,   7, 128,  81],
       [ 50,   4,   2,   7, 128,  69],
       [100,   4,   2,   7, 128,  68],
       [150,   4,   2,   7, 128,  71],
       [200,  25,   5,   2,  61, 100],
       [200,  25,   5,   6,  61, 100],
       [200,  25,   5,   9,  61, 100],
       [200,  75,   3,   7,  27, 100],
       [200,  75,   2,   7,  33, 109],
       [200,  73,   4,   7,  56,  81]])
 np.load("/data/scratch/acw753/lakhmidi_processed/processed/1513db2093f3ae5b191545a9718f4f56.npy")[:40]
array([[  0,   4,   2,   7, 128,  81],
       [ 50,   4,   2,   7, 128,  69],
       [100,   4,   2,   7, 128,  68],
       [150,   4,   2,   7, 128,  71],
       [200,  25,   5,   2,  61, 100],
       [200,  25,   5,   6,  61, 100],
       [200,  25,   5,   9,  61, 100],
       [200,  75,   3,   7,  27, 100],
       [200,  75,   2,   7,  33, 109],
       [200,  73,   4,   7,  56,  81],
       [200,  73,   4,  11,  56,  81],
       [200,  74,   5,   2,  56,  81],
       [200,  49,   3,   7,  27,  84],
       [200,  48,   4,   2,  27,  84],
       [200,  49,   4,   7,  27,  84],
       [200,  73,   4,   7,  27,  69],
       [200,  73,   4,  11,  27,  69],
       [200,  74,   5,   2,  27,  69],
       [200,   4,   2,  11, 128, 109],
       [200,   4,   3,   6, 128, 111],
       [225, 125,   5,   2,  61, 100],
       [225, 125,   5,   7,  61, 100],
       [225, 125,   5,  11,  61, 100],
       [250,  24,   3,   7,  27,  76],
       [250,  24,   4,   2,  27,  76],
       [250,  24,   4,   7,  27,  76],
       [250,   4,   3,   4, 128, 111],
       [250,   4,   3,   6, 128,  89],
       [275,  25,   4,   2,  27, 100],
       [275,  25,   2,   7,  33,  92],
       [275,  23,   4,   7,  56,  75],
       [275,  23,   4,  11,  56,  75],
       [275,  24,   5,   2,  56,  75],
       [275,  24,   3,   7,  27,  76],
       [275,  24,   4,   2,  27,  76],
       [275,  24,   4,   7,  27,  76],
       [275,  24,   4,   7,  27,  86],
       [275,  24,   4,  11,  27,  86],
       [275,  24,   5,   2,  27,  86],
       [275,   4,   2,  11, 128,  72]]) 
    
    """



    # prompts = [[[0, 0, 0, 0, 0, 0],
    #             [0, 384, 2, 11, 0, 100]],
    #            [[0, 0, 0, 0, 0, 0],
    #             [111, 16, 3, 5, 128, 100]],
    #            [[0, 0, 0, 0, 0, 0],
    #             [386, 371, 5, 7, 48, 88]],
    #            ] #[[0, 0, 0, 0, 0, 0]]

    results = generator.music_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    #TODO: Detokenizer + midi + render 
    for dialog, result in zip(prompts, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)

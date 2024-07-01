# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from accelerate.utils import is_xpu_available
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class MusicLlama:
    @staticmethod
    def build_original(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        print("night safari model_args", model_args)

        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)
    
    @staticmethod
    def build(
        ckpt_dir: str,
        model_config_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "MusicLlama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """

        # Set the seeds for reproducibility
        if is_xpu_available():
            torch.xpu.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        llama_config = LlamaConfig.from_pretrained(model_config_path)
        model = LlamaForCausalLM(llama_config) 
        start_time = time.time()
        checkpoint = torch.load(ckpt_dir)
        
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'): # Check if the keys have 'module.' prefix and remove it if necessary
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict) # Load the weights
        
        if is_xpu_available():
            model.to("xpu")
        else:
            model.to("cuda")
        model.eval()

        tokenizer = MusicTokenizer(onset_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size)
        
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)  #this throws me exeptions!  TODO: think what to do about it

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return MusicLlama(model, tokenizer, llama_config)

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[List[int]]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """

        bsz = len(prompt_tokens)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.max_len 
        total_len = min(self.config.max_len, max_gen_len + max_prompt_len) 

        pad_id = self.tokenizer.pad_token_compound
        pad_tensor = torch.tensor(pad_id, dtype=torch.long, device="cuda").unsqueeze(0).unsqueeze(0) #create a tensor with shape: (bsz, total_len, 6) filled with pad_id
        tokens = pad_tensor.expand(bsz, total_len, -1).clone() #6, --> bsz, total_len, 6

        for k, t in enumerate(prompt_tokens): 
            t_tensor = torch.tensor(t, dtype=torch.long, device="cuda")  # (len_t, 6) 
            tokens[k, :len(t)] = t_tensor  #tokens[k, :len(t)] --> len_t, 6


        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_mask = torch.all(tokens != pad_tensor, dim=-1).unsqueeze(-1) #(batch, len, 1)

        #Test out inference without KV cache
        for cur_pos in range(min_prompt_len, total_len): 
            print(f"input ids: {tokens[:, prev_pos:cur_pos]}, attn mask: {input_mask.long()[:, prev_pos:cur_pos, 0]}")
            output = self.model.forward(input_ids = tokens[:, prev_pos:cur_pos], use_cache = None, attention_mask = torch.zeros_like(tokens[:, prev_pos:cur_pos]).unsqueeze(-1))
            if temperature > 0:
                onset_probs = torch.sigmoid(output.onset_logits[:, -1]/ temperature) 
                duration_probs = torch.softmax(output.duration_logits[:, -1]/ temperature, dim=-1) 
                octave_probs = torch.softmax(output.octave_logits[:, -1]/ temperature, dim=-1) 
                pitch_probs = torch.softmax(output.pitch_logits[:, -1]/ temperature, dim=-1) 
                instrument_probs = torch.softmax(output.instrument_logits[:, -1]/ temperature, dim=-1)  
                velocity_probs = torch.softmax(output.velocity_logits[:, -1]/ temperature, dim=-1) 

                next_onset_token_bits = torch.bernoulli(onset_probs)
                next_duration_token = sample_top_p(duration_probs, top_p)
                next_onset_token = self.tokenizer.binary_to_decimal_batch(next_onset_token_bits).to(next_duration_token)

                next_octave_token = sample_top_p(octave_probs, top_p)
                next_pitch_token = sample_top_p(pitch_probs, top_p)
                next_instrument_token = sample_top_p(instrument_probs, top_p)
                next_velocity_token = sample_top_p(velocity_probs, top_p)
                next_token = torch.cat([next_onset_token, next_duration_token, next_octave_token, next_pitch_token,next_instrument_token,next_velocity_token  ], dim=-1) #batch, 1, 6, 
                
            else:
                onset_probs = torch.sigmoid(output.onset_logits[:, -1]) 
                duration_probs = torch.softmax(output.duration_logits[:, -1], dim=-1) 
                octave_probs = torch.softmax(output.octave_logits[:, -1], dim=-1) 
                pitch_probs = torch.softmax(output.pitch_logits[:, -1], dim=-1) 
                instrument_probs = torch.softmax(output.instrument_logits[:, -1], dim=-1)  
                velocity_probs = torch.softmax(output.velocity_logits[:, -1], dim=-1) 
                # print(f"onset_probs:{onset_probs}, duration_probs:{duration_probs}, octave_probs:{octave_probs}, pitch_probs:{pitch_probs}, instrument_probs:{instrument_probs}, velocity_probs:{velocity_probs}")
                # Sample the token with the highest probability (greedy sampling)
                next_onset_token_bits = (onset_probs > 0.5).float()  # Assuming binary decision for onset
                next_duration_token = duration_probs.argmax(dim=-1, keepdim=True)
                next_onset_token = self.tokenizer.binary_to_decimal_batch(next_onset_token_bits).to(next_duration_token)

                next_octave_token = octave_probs.argmax(dim=-1, keepdim=True)
                next_pitch_token = pitch_probs.argmax(dim=-1, keepdim=True)
                next_instrument_token = instrument_probs.argmax(dim=-1, keepdim=True)
                next_velocity_token = velocity_probs.argmax(dim=-1, keepdim=True)
                # print(f"next_onset_token_bits:{next_onset_token_bits}, next_duration_token:{next_duration_token}, next_onset_token:{next_onset_token}, next_octave_token:{next_octave_token}, next_pitch_token:{next_pitch_token}, next_instrument_token:{next_instrument_token}. next_velocity_token:{next_velocity_token}")

                # Concatenate the sampled tokens
                next_token = torch.cat([
                    next_onset_token,   # assuming next_onset_token is scalar
                    next_duration_token,  # assuming next_duration_token is scalar
                    next_octave_token,   # assuming next_octave_token is scalar
                    next_pitch_token,   # assuming next_pitch_token is scalar
                    next_instrument_token,   # assuming next_instrument_token is scalar
                    next_velocity_token,   # assuming next_velocity_token is scalar
                ], dim=-1)

                # next_token = torch.argmax(logits[:, -1], dim=-1)
    

            next_token = torch.where(
                input_mask[:, cur_pos], tokens[:, cur_pos], next_token
            ) 
            tokens[:, cur_pos] = next_token
            # print("check next token",next_onset_token, next_duration_token, next_octave_token, next_pitch_token, next_instrument_token, next_velocity_token)

        assert 1==2


        past_key_values = None
        for cur_pos in range(min_prompt_len, total_len): #recursively generate new tokens in parallel
            # if past_key_values is not None:
            #     print(f"cur_pos:{cur_pos}, prev_pos:{prev_pos}, input_ids: {tokens[:, prev_pos:cur_pos]},past_key_values:{past_key_values[0][0]}") #layer 0 key
            # else:
            #     print(f"cur_pos:{cur_pos}, prev_pos:{prev_pos}, input_ids: {tokens[:, prev_pos:cur_pos]} with shape:{tokens[:, prev_pos:cur_pos].shape},past_key_values:{past_key_values}")
            print(f"input ids: {tokens[:, prev_pos:cur_pos]}")
            output = self.model.forward(input_ids = tokens[:, prev_pos:cur_pos], past_key_values = past_key_values, use_cache = True, attention_mask = None)
            
            if temperature > 0:
                onset_probs = torch.sigmoid(output.onset_logits[:, -1]/ temperature) 
                duration_probs = torch.softmax(output.duration_logits[:, -1]/ temperature, dim=-1) 
                octave_probs = torch.softmax(output.octave_logits[:, -1]/ temperature, dim=-1) 
                pitch_probs = torch.softmax(output.pitch_logits[:, -1]/ temperature, dim=-1) 
                instrument_probs = torch.softmax(output.instrument_logits[:, -1]/ temperature, dim=-1)  
                velocity_probs = torch.softmax(output.velocity_logits[:, -1]/ temperature, dim=-1) 

                next_onset_token_bits = torch.bernoulli(onset_probs)
                next_duration_token = sample_top_p(duration_probs, top_p)
                next_onset_token = self.tokenizer.binary_to_decimal_batch(next_onset_token_bits).to(next_duration_token)

                next_octave_token = sample_top_p(octave_probs, top_p)
                next_pitch_token = sample_top_p(pitch_probs, top_p)
                next_instrument_token = sample_top_p(instrument_probs, top_p)
                next_velocity_token = sample_top_p(velocity_probs, top_p)
                next_token = torch.cat([next_onset_token, next_duration_token, next_octave_token, next_pitch_token,next_instrument_token,next_velocity_token  ], dim=-1) #batch, 1, 6, 
                
            else:
                onset_probs = torch.sigmoid(output.onset_logits[:, -1]) 
                duration_probs = torch.softmax(output.duration_logits[:, -1], dim=-1) 
                octave_probs = torch.softmax(output.octave_logits[:, -1], dim=-1) 
                pitch_probs = torch.softmax(output.pitch_logits[:, -1], dim=-1) 
                instrument_probs = torch.softmax(output.instrument_logits[:, -1], dim=-1)  
                velocity_probs = torch.softmax(output.velocity_logits[:, -1], dim=-1) 
                
                # Sample the token with the highest probability (greedy sampling)
                next_onset_token_bits = (onset_probs > 0.5).float()  # Assuming binary decision for onset
                next_duration_token = duration_probs.argmax(dim=-1, keepdim=True)
                next_onset_token = self.tokenizer.binary_to_decimal_batch(next_onset_token_bits).to(next_duration_token)

                next_octave_token = octave_probs.argmax(dim=-1, keepdim=True)
                next_pitch_token = pitch_probs.argmax(dim=-1, keepdim=True)
                next_instrument_token = instrument_probs.argmax(dim=-1, keepdim=True)
                next_velocity_token = velocity_probs.argmax(dim=-1, keepdim=True)
                print(f"check raw:next_onset_token_bits {next_onset_token_bits.shape}, {next_onset_token_bits},next_duration_token {next_duration_token.shape}, {next_duration_token}, next_onset_token {next_onset_token.shape}, {next_onset_token},  next_octave_token{next_octave_token.shape}")
                # Concatenate the sampled tokens
                next_token = torch.cat([
                    next_onset_token,   # assuming next_onset_token is scalar
                    next_duration_token,  # assuming next_duration_token is scalar
                    next_octave_token,   # assuming next_octave_token is scalar
                    next_pitch_token,   # assuming next_pitch_token is scalar
                    next_instrument_token,   # assuming next_instrument_token is scalar
                    next_velocity_token,   # assuming next_velocity_token is scalar
                ], dim=-1)

                
                print("TODO, implement greedy sampling")
                # next_token = torch.argmax(logits[:, -1], dim=-1)

            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_mask[:, cur_pos], tokens[:, cur_pos], next_token
            ) 
            tokens[:, cur_pos] = next_token
            # print("check next token",next_onset_token, next_duration_token, next_octave_token, next_pitch_token, next_instrument_token, next_velocity_token)
            # print("check eos condition", torch.isin(next_duration_token, torch.tensor(self.tokenizer.eos_dur)),\
            #        torch.isin(next_octave_token, torch.tensor(self.tokenizer.eos_octave)),\
            #         torch.isin(next_pitch_token, torch.tensor(self.tokenizer.eos_pitch_class)) ,\
            #             torch.isin(next_instrument_token, torch.tensor(self.tokenizer.eos_instrument)), \
            #             torch.isin(next_velocity_token, torch.tensor(self.tokenizer.eos_velocity))   )
            
            eos_conditions = (
                next_onset_token_bits == torch.tensor(self.tokenizer.eos_onset) |
                torch.isin(next_duration_token, torch.tensor(self.tokenizer.eos_dur)) |
                torch.isin(next_octave_token, torch.tensor(self.tokenizer.eos_octave)) |
                torch.isin(next_pitch_token, torch.tensor(self.tokenizer.eos_pitch_class)) |
                torch.isin(next_instrument_token, torch.tensor(self.tokenizer.eos_instrument)) |
                torch.isin(next_velocity_token, torch.tensor(self.tokenizer.eos_velocity))
            )

            # Ensure all operands are tensors and correct type
            eos_conditions = torch.tensor(eos_conditions, dtype=torch.bool, device="cuda")
            tmp  = ~input_mask[:, cur_pos]
            # print(f"eos_conditions:{eos_conditions.shape}, {eos_conditions},(~input_mask[:, cur_pos]):{tmp.shape}{(~input_mask[:, cur_pos])} ")
            # Update eos_reached based on the mask and EOS conditions
            # eos_reached |= (~input_mask[:, cur_pos]) & eos_conditions   
            #          
            eos_reached = False
            
            # eos_reached |= (~input_mask[:, cur_pos]) & (
            #     next_onset_token_bits == self.tokenizer.eos_onset | next_duration_token == self.tokenizer.eos_dur | next_octave_token == self.tokenizer.eos_octave | next_pitch_token == self.tokenizer.eos_pitch_class | next_instrument_token == self.tokenizer.eos_instrument | next_velocity_token == self.tokenizer.eos_velocity
            # ) #at pad position and eos 


            prev_pos = cur_pos
            past_key_values = output.past_key_values
            """if all(eos_reached): #wait until all sequences reach eos
                break """
        

        assert 1==2
        #TODO: Check below later
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def music_completion(
        self,
        prompt_tokens: List[List[List[int]]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        if max_gen_len is None:
            max_gen_len = self.config.max_len - 1 

        # prompt_tokens = [
        #     self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        # ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
            }
            for t in generation_tokens
        ]


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Dict
from accelerate.utils import is_xpu_available
import time
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaForCausalLM_Conditional_Generation,
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

# from llama.model import ModelArgs, Transformer
# from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required



class MusicLlamaConditional:
    # @staticmethod
    # def build_original(
    #     ckpt_dir: str,
    #     tokenizer_path: str,
    #     max_seq_len: int,
    #     max_batch_size: int,
    #     model_parallel_size: Optional[int] = None,
    #     seed: int = 1,
    # ) -> "Llama":
    #     """
    #     Build a Llama instance by initializing and loading a model checkpoint.

    #     Args:
    #         ckpt_dir (str): Path to the directory containing checkpoint files.
    #         tokenizer_path (str): Path to the tokenizer file.
    #         max_seq_len (int): Maximum sequence length for input text.
    #         max_batch_size (int): Maximum batch size for inference.
    #         model_parallel_size (Optional[int], optional): Number of model parallel processes.
    #             If not provided, it's determined from the environment. Defaults to None.

    #     Returns:
    #         Llama: An instance of the Llama class with the loaded model and tokenizer.

    #     Raises:
    #         AssertionError: If there are no checkpoint files in the specified directory,
    #             or if the model parallel size does not match the number of checkpoint files.

    #     Note:
    #         This method initializes the distributed process group, sets the device to CUDA,
    #         and loads the pre-trained model and tokenizer.
    #     """
    #     if not torch.distributed.is_initialized():
    #         torch.distributed.init_process_group("nccl")
    #     if not model_parallel_is_initialized():
    #         if model_parallel_size is None:
    #             model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    #         initialize_model_parallel(model_parallel_size)

    #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #     torch.cuda.set_device(local_rank)

    #     # seed must be the same in all processes
    #     torch.manual_seed(seed)

    #     if local_rank > 0:
    #         sys.stdout = open(os.devnull, "w")

    #     start_time = time.time()
    #     checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    #     assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    #     assert model_parallel_size == len(
    #         checkpoints
    #     ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
    #     ckpt_path = checkpoints[get_model_parallel_rank()]
    #     checkpoint = torch.load(ckpt_path, map_location="cpu")
    #     with open(Path(ckpt_dir) / "params.json", "r") as f:
    #         params = json.loads(f.read())

    #     model_args: ModelArgs = ModelArgs(
    #         max_seq_len=max_seq_len,
    #         max_batch_size=max_batch_size,
    #         **params,
    #     )
    #     print("night safari model_args", model_args)

    #     tokenizer = Tokenizer(model_path=tokenizer_path)
    #     assert model_args.vocab_size == tokenizer.n_words
    #     if torch.cuda.is_bf16_supported():
    #         torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    #     else:
    #         torch.set_default_tensor_type(torch.cuda.HalfTensor)
    #     model = Transformer(model_args)
    #     model.load_state_dict(checkpoint, strict=False)
    #     print(f"Loaded in {time.time() - start_time:.2f} seconds")

    #     return Llama(model, tokenizer)
    
    @staticmethod
    def build(
        ckpt_dir: str,
        model_config_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "MusicLlamaConditional":
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
        checkpoint = checkpoint['model_state_dict']
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

        tokenizer = MusicTokenizer(timeshift_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size)
        
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            model = model.to(torch.bfloat16)  # Explicitly cast the entire model to BF16 precision. TODO: whether or not cast in bf16
            print("model precision set to BF16")
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)  #this throws me exeptions!  TODO: think what to do about it

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return MusicLlamaConditional(model, tokenizer, llama_config)

    @staticmethod
    def build_commu_con_gen(
        ckpt_dir: str,
        model_config_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
        finetuned_PEFT_weight_path: Optional[str] = None,
        additional_token_dict: Optional[Dict] = None,
    ) -> "MusicLlamaConditional":
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
        model = LlamaForCausalLM_Conditional_Generation(llama_config) 
        start_time = time.time()

        #if peft model then load differently
        checkpoint = torch.load(ckpt_dir)
        checkpoint = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'): # Check if the keys have 'module.' prefix and remove it if necessary
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"when loading checkpoint, encounter missing keys: {missing_keys}; unexpected_keys:{unexpected_keys}")
        # model.load_state_dict(new_state_dict) # Load the weights
        
        if finetuned_PEFT_weight_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, finetuned_PEFT_weight_path)
            print("PEFT model loaded successfully")

        if is_xpu_available():
            model.to("xpu")
        elif torch.cuda.is_available():
            model.to("cuda")
        else:
            model.to("cpu")
        model.eval()

        tokenizer = MusicTokenizer(timeshift_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size)
    

        for key, value in additional_token_dict.items():
            tokenizer.add_new_tokens(token_name = key, token_val = value)
            print(f"added {key} to tokenizer")
        
        
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            model = model.to(torch.bfloat16)  # Explicitly cast the entire model to BF16 precision. TODO: whether or not cast in bf16
            print("model precision set to BF16")
        elif torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)  #this throws me exeptions!  TODO: think what to do about it

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return MusicLlamaConditional(model, tokenizer, llama_config)

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[List[int]]],
        bpm_condition: List[int],
        time_signature_condition: List[str],
        num_measures_condition: List[int],
        max_gen_len: int,
        min_gen_len: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        metadata_condition: List = None, 
        chord_condition: List = None,
        condition_token_lengths: List[int] = None,
        chord_dict: str = None,
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Usando il dispositivo: {device} ---")
        
        bsz = len(prompt_tokens)
        if metadata_condition is not None:
            metadata_tokens = torch.tensor(metadata_condition)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.max_len 
        total_len = min(self.config.max_len, max_gen_len + max_prompt_len) 

        pad_id = self.tokenizer.pad_token_compound
        pad_tensor = torch.tensor(pad_id, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0) #create a tensor with shape: (bsz, total_len, 6) filled with pad_id
        tokens = pad_tensor.expand(bsz, total_len, -1).clone() #6, --> bsz, total_len, 6

        for k, t in enumerate(prompt_tokens): 
            t_tensor = torch.tensor(t, dtype=torch.long, device=device)  # (len_t, 6) 
            tokens[k, :len(t)] = t_tensor  #tokens[k, :len(t)] --> len_t, 6

        #fill another pad tensor with start bar, beat, chord 
        if chord_condition is not None:
            bar_beat_chord_pad_id = [0, 0, 61] #bar = 0, beat = 0, chord = "s"
            bar_beat_chord_pad_tensor = torch.tensor(bar_beat_chord_pad_id, dtype=torch.long, device="cuda").unsqueeze(0).unsqueeze(0) #create a tensor with shape: (bsz, total_len, 6) filled with pad_id
            bar_beat_chord_condition = bar_beat_chord_pad_tensor.expand(bsz, total_len, -1).clone() # 3, --> bsz, total_len, 3
        else:
            bar_beat_chord_condition = None
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_mask = torch.all(tokens != pad_tensor, dim=-1).unsqueeze(-1) #(batch, len, 1)

        """KV Cache"""
        past_key_values = None
        for cur_pos in range(min_prompt_len, total_len): #recursively generate new tokens in parallel
            print(f"{cur_pos}/{total_len} generated")
            print(f"prev_pos:{prev_pos}, cur_pos:{cur_pos}")
            output = self.model.forward(input_ids = tokens[:, prev_pos:cur_pos], past_key_values = past_key_values, use_cache = True, attention_mask = None) #output logtis: (batch, len, dim 
            #KV cache 
            next_decoder_token = torch.tensor(self.tokenizer.sos_out).to(tokens).expand(tokens.shape[0]*(cur_pos - prev_pos), 1) #batch*len_x, len_y = 1
            next_decoder_token_out = next_decoder_token
            hidden_state = output.logits  #first forward pass: batch, len_x, dim --> batch*len_x, dim  --> num_layer, batch*len_x, dim;     
            hidden_state = hidden_state.view(hidden_state.shape[0]*hidden_state.shape[1], hidden_state.shape[2]).unsqueeze(0).expand(self.model.decoder.num_hidden_layers, -1, -1).contiguous() #batch, len_x, dim --> num_layer, batch*len_x, dim;
            
            if metadata_condition is not None:
                metadata_tokens_expanded = metadata_tokens.to(tokens).unsqueeze(1).expand(-1, cur_pos - prev_pos, -1).reshape(tokens.shape[0]*(cur_pos - prev_pos), -1) #batch, 11 --> batch, 1, 11 --> batch*cur_pos - prev_pos, 11
            else:
                metadata_tokens_expanded = None
            if chord_condition is not None:
                bar_beat_chord_condition_expanded = bar_beat_chord_condition[:, prev_pos:cur_pos].reshape(-1, 3) #batch, len, 3
            else:
                bar_beat_chord_condition_expanded = None

            for attribute in ["timeshift_dict_decode", "duration_dict_decode", "octave_dict_decode", "pitch_dict_decode", "instrument_dict_decode", "velocity_dict_decode"]:
                output_decoder = self.model.forward(decoded_hidden_state = hidden_state, decoded_language_tokens = next_decoder_token, attention_mask = None, metadata_condition = metadata_tokens_expanded, bar_beat_chord_condition = bar_beat_chord_condition_expanded)
                generation_logits = output_decoder.generation_logits ##batch*len_x, len_y, decode_vocab_size
                
                # NEW EOS PREVENT LOGIC (if min_gen_len is > 0)
                num_generated_tokens = cur_pos - min_prompt_len
                if num_generated_tokens < min_gen_len:
                    eos_token_id = -1 # Default to invalid
                    if attribute == "timeshift_dict_decode":
                        eos_token_id = self.tokenizer.timeshift_dict[self.tokenizer.eos_timeshift]
                    elif attribute == "duration_dict_decode":
                        eos_token_id = self.tokenizer.duration_dict[self.tokenizer.eos_dur]
                    elif attribute == "octave_dict_decode":
                        eos_token_id = self.tokenizer.octave_dict[self.tokenizer.eos_octave]
                    elif attribute == "pitch_dict_decode":
                        eos_token_id = self.tokenizer.pitch_dict[self.tokenizer.eos_pitch_class]
                    elif attribute == "instrument_dict_decode":
                        eos_token_id = self.tokenizer.instrument_dict[self.tokenizer.eos_instrument]
                    elif attribute == "velocity_dict_decode":
                        eos_token_id = self.tokenizer.velocity_dict[self.tokenizer.eos_velocity]
                    
                    if eos_token_id != -1:
                        # Set the probability of the EOS token to zero by setting its logit to negative infinity
                        generation_logits[:, -1, eos_token_id] = -float('inf')
                
                hidden_state = output_decoder.generation_hidden_state ##num_layers, batch*len_x, dim

                sample_indices = list(getattr(self.tokenizer, attribute).keys())
                sample_indices_set = set(sample_indices)
                if temperature > 0:
                    probs = torch.softmax(generation_logits[:, -1, : ]/ temperature, dim=-1) 
                    
                    '''
                    # --- START OF NEW DEBUG BLOCK FOR PRINTING TOP-5 TOKEN PROBABILITY ---
                    # Get the top 5 probabilities and their indices
                    top_probs, top_indices = torch.topk(probs, 5, dim=-1)

                    # Get the correct decoding dictionary for the current attribute
                    decode_dict = getattr(self.tokenizer, attribute)
                    
                    print(f"\n--- Top 5 choices for: {attribute.replace('_dict_decode', '')} (at step {cur_pos}) ---")
                    # Assuming a batch size of 1 for readable debug output
                    for k in range(5):
                        prob = top_probs[0, k].item()
                        vocab_idx = top_indices[0, k].item()
                        
                        # Find the musical value corresponding to the vocab index
                        musical_value = decode_dict.get(vocab_idx, "N/A (Not in this attribute's vocab)")
                        
                        print(f"  {k+1}. Value: {str(musical_value):<10} (Vocab Idx: {vocab_idx:<5}) | Probability: {prob:.6f}")
                    # --- END OF NEW DEBUG BLOCK ---
                    '''
                    
                    next_decoder_token = sample_top_p(probs, top_p) 
                    
                    for i in range(next_decoder_token.size(0)):  # Ensure that all next_decoder_token values are in sample_indices
                        start_time = time.time()
                        while next_decoder_token[i, 0].item() not in sample_indices_set:  # Check if token is valid
                            if time.time() - start_time > 15:  # If sampling takes too long, mask invalid indices
                                print(f"Warning: Resampling for token {i} exceeded 15 seconds. Masking invalid logits and Resampling...")
                                
                                # Set logits of invalid indices to -inf
                                mask = torch.full_like(probs, float('-inf'))
                                mask[:, sample_indices] = probs[:, sample_indices]  

                                # Recompute probabilities with the mask
                                probs = torch.softmax(mask, dim=-1)

                            next_decoder_token[i, 0] = sample_top_p(probs, top_p)[i, 0]  
                else:
                    probs = torch.softmax(generation_logits[:, -1, :], dim=-1)  #batch*len_x, len_y (last), decode_vocab_size
                    # next_decoder_token_greedy = probs.argmax(dim=-1, keepdim=True) #batch*len_x, 1

                    sample_indices_tensor = torch.tensor(sample_indices, device=probs.device)  # Ensure it's on the same device as probs
                    probs_at_sample_indices = probs[:, sample_indices_tensor]  # Shape: [batch_size, num_sample_indices]
                    next_token_index_in_subset = probs_at_sample_indices.argmax(dim=-1, keepdim=True)  # Shape: [batch_size, 1]
                    next_decoder_token = sample_indices_tensor[next_token_index_in_subset.squeeze(-1)].unsqueeze(-1)   # Shape: [batch_size]

                # Get the cumulative probability at sample_indices, if c_p is smaller than 0.8, print warning
                probs_at_sample_indices = probs[:, sample_indices]  # Extract probabilities for the sampled indices
                cumulative_prob = probs_at_sample_indices.sum(dim=-1)  # Sum over the sampled indices to get cumulative prob
                num_samples_below_threshold = (cumulative_prob < 0.8).sum().item()  # Count the number of True values
                if num_samples_below_threshold > 0:
                    print(f"{num_samples_below_threshold} / {cumulative_prob.shape[0]} samples have a cumulative probability < 0.8 at the allowed indices")

                next_decoder_token_out = torch.cat([next_decoder_token_out, next_decoder_token], dim=-1) #batch*len_x, len_y
            
            # 4. Decode the generated language tokens back to musical values
            next_decoder_token_out_reshaped = next_decoder_token_out[:, 1:].view(tokens.shape[0], -1 ,6) # batch*len_x, 6 -> batch, len_x, 6
            next_decoder_token_lang = self.tokenizer.convert_from_language_tokens(next_decoder_token_out_reshaped) # batch, lenx, 6

            # 5. Check if the newly generated token is an EOS token.
            # This is the single, authoritative check for EOS.
            eos_conditions_onset = next_decoder_token_lang.clone().detach()[:, -1, 0] == self.tokenizer.eos_timeshift
            eos_conditions_dur = next_decoder_token_lang.clone().detach()[:, -1, 1] == self.tokenizer.eos_dur
            eos_conditions_oct = next_decoder_token_lang.clone().detach()[:, -1, 2] == self.tokenizer.eos_octave
            eos_conditions_pitch = next_decoder_token_lang.clone().detach()[:, -1, 3] == self.tokenizer.eos_pitch_class
            eos_conditions_instr = next_decoder_token_lang.clone().detach()[:, -1, 4] == self.tokenizer.eos_instrument
            eos_conditions_vel = next_decoder_token_lang.clone().detach()[:, -1, 5] == self.tokenizer.eos_velocity
            eos_conditions_all_attr = torch.stack([eos_conditions_onset, eos_conditions_dur, eos_conditions_oct, eos_conditions_pitch, eos_conditions_instr, eos_conditions_vel], dim = -1)
            eos_conditions = torch.any(eos_conditions_all_attr, dim = -1) # Result is a boolean tensor of shape (bsz,)

            # 6. Calculate the potential next musical token (if not EOS).
            previous_onset = tokens[:, cur_pos-1, 0]
            # Handle cases where the previous token was symbolic (e.g., SOS < 0) by treating its onset as 0.
            previous_onset = torch.where(previous_onset < 0, torch.zeros_like(previous_onset), previous_onset)
            # Calculate the new absolute onset.
            new_onset = previous_onset + next_decoder_token_lang.clone().detach()[:, -1, 0].to(previous_onset)
            # Construct the full musical token [onset, duration, ...].
            next_musical_token = torch.cat([new_onset.unsqueeze(-1), next_decoder_token_lang.clone().detach()[:, -1, 1:]],dim=-1).to(tokens)

            # 7. Prepare the symbolic EOS token for insertion.
            eos_compound_tensor = torch.tensor(self.tokenizer.eos_token_compound, device=tokens.device, dtype=tokens.dtype)

            # 8. Write the correct token (musical or symbolic EOS) to the main tokens tensor.
            for i in range(bsz):
                # Only modify the token if the sequence has not already ended.
                if not eos_reached[i]:
                    if eos_conditions[i]:
                        # If an EOS was just generated, write the symbolic EOS token.
                        tokens[i, cur_pos] = eos_compound_tensor
                    else:
                        # Otherwise, write the normally generated musical token.
                        tokens[i, cur_pos] = next_musical_token[i]

            # 9. Calculate the next chord conditioning based on the new onsets.
            if chord_condition is not None:
                current_onsets = tokens[:, cur_pos, 0]
                bar_beat_chord_new_onset = onset2bar_beat_chord(current_onsets, chord_condition, time_signature_condition, bpm_condition, num_measures_condition, chord_dict)
                # Update the chord condition tensor for sequences that have not yet ended.
                for i in range(bsz):
                    if not eos_reached[i]:
                        bar_beat_chord_condition[i, cur_pos] = bar_beat_chord_new_onset[i]

            # 10. Update the eos_reached flag for the entire batch.
            eos_reached |= (~input_mask[:, cur_pos].squeeze(-1)) & eos_conditions

            # Update eos_reached based on the mask and EOS conditions
            eos_reached |= (~input_mask[:, cur_pos].squeeze(-1)) & eos_conditions   
            prev_pos = cur_pos
            past_key_values = output.past_key_values
            if any(eos_reached): #wait until all sequences reach eos
                eos_count = eos_reached.sum().item()  # Number of sequences with EOS
                total_count = len(eos_reached)        # Total number of sequences
                print(f"{eos_count}/{total_count} sequences have reached EOS.")

            if all(eos_reached): #wait until all sequences reach eos
                print("eos reached!")
                break 
        # tokens = tokens[:, condition_token_length:, :] #remove SOS token

        # --- IMPROVED OUTPUT TRUNCATION ---
        out_tokens = []
        out_logprobs = [] # Not implemented, but placeholder remains

        # Find the first occurrence of EOS for each item in the batch
        eos_indices = []
        for i in range(bsz):
            # Find the first index where eos_reached was True for this sequence
            eos_idx_tensor = (tokens[i, :, 0] == self.tokenizer.eos_token).nonzero(as_tuple=True)[0]
            if len(eos_idx_tensor) > 0:
                eos_indices.append(eos_idx_tensor[0].item())
            else:
                # If no EOS token was found, use the full generated length
                eos_indices.append(total_len)

        for i, toks in enumerate(tokens.tolist()):
            # Determine the start and end points for slicing
            start_index = 0 if echo else len(prompt_tokens[i])
            end_index = eos_indices[i]

            # Slice the tokens to get the final output
            final_toks = toks[start_index:end_index]
            
            out_tokens.append(final_toks)
            # When logprobs are implemented, they should be sliced here too
            # out_logprobs.append(token_logprobs[i][start_index:end_index] if logprobs else None)

        out_logprobs_no_cond_tokens = [] # Placeholder
        out_tokens_no_cond_tokens = []
        if condition_token_lengths:
            for i, condition_token_length in enumerate(condition_token_lengths):
                # This assumes the condition tokens are part of the prompt and handled by the `echo` flag.
                # If `echo=True`, the slicing above already handles it. Let's adjust for your specific use case.
                out_tokens_no_cond_tokens.append(out_tokens[i][condition_token_length:])
        else:
             out_tokens_no_cond_tokens = out_tokens


        print(f"after cutting condition tokens: {out_tokens_no_cond_tokens}")
        return (out_tokens_no_cond_tokens, out_logprobs_no_cond_tokens if logprobs else None)

    def music_completion(
        self,
        prompt_tokens: List[List[List[int]]],
        bpm_condition: List[int],
        time_signature_condition: List[str],
        num_measures_condition: List[int],
        metadata_condition: List = None, 
        chord_condition: List = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        min_gen_len: int = 0,
        logprobs: bool = False,
        condition_token_lengths: List[int] = None,
        chord_token_indices: List[List[int]] = None,
        chord_dict: str = None,
        if_return_chords: bool = True
    ):
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
            bpm_condition = bpm_condition,
            time_signature_condition = time_signature_condition,
            num_measures_condition = num_measures_condition,
            metadata_condition=metadata_condition, 
            chord_condition = chord_condition,
            max_gen_len=max_gen_len,
            min_gen_len=min_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo = True,
            condition_token_lengths = condition_token_lengths,
            chord_dict = chord_dict
        )
        if chord_token_indices is not None:
            chord_tokens = [prompt_tokens[i][chord_token_indices[i][0]+1:chord_token_indices[i][1]] for i in range(len(prompt_tokens))]
        else:
            chord_tokens = [generation_tokens[0] for _ in range(len(prompt_tokens))] #dummy output
        #Post processing: if the generated tokens have more than 15 channels, split the generation tokens into multiple parts of at most 15 channels 
        generation_tokens_post_proc = []
        for t in generation_tokens: #check if generated midi has more than 16 channels (instruments)
            if len(set(row[4] for row in t))>15:
                generation_tokens_post_proc.append(self.postprocess_split(t)) # postprocess_split returns a compounded list, where each element is a tensor with shape (len, 6) with max 16 instrument
            else:
                generation_tokens_post_proc.append([t])    

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
        if if_return_chords:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.compound_to_midi_multi(t, bpm=b), #a list of midi objects each contains at most 15 instruments
                        "chord": self.tokenizer.compound_to_midi_multi([chord], bpm=b),
                        "tokens": t,
                        "chord_tokens": [chord],
                    },
                }
                for chord,t, b in zip(chord_tokens, generation_tokens_post_proc, bpm_condition) #t is a list of tensors with shape (len, 6)
            ]
        else:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.compound_to_midi_multi(t, bpm=b), #a list of midi objects each contains at most 15 instruments
                        "chord": None,
                        "tokens": t,
                        "chord_tokens": None,
                    },
                }
                for t, b in zip(generation_tokens_post_proc, bpm_condition) #t is a list of tensors with shape (len, 6)
            ]   
    @staticmethod
    def postprocess_split(tokens):
        """Processes tokens to group them into sublists, each containing up to 16 unique instruments.
        
        Args:
            tokens (np.ndarray): An array with shape (len, 6), where tokens[:, 4] contains instrument information.
        
        Returns:
            List[List[np.ndarray]]: A list of lists, where each inner list contains tokens for up to 16 unique instruments.
        """
        split2instrument = dict() # split_id1: [instr1, instr2, ...], split_id2: [instr17, instr18, ...]
        instrument2split = dict() # instr1: split_id1, instr2: split_id1, instr17: split_id2, instr18: split_id2
        split2token = dict() # split_id1: [token1, token2, ...], split_id2: [token17, token18, ...]
        
        # Iterate through each token and append it to the current list
        for token in tokens:
            instrument = int(token[4])

            if instrument in instrument2split:
                split_id = instrument2split[instrument]
                split2token[split_id].append(token)
            else: 
                #if instrument not in instrument2split, check if the latest split has less than 15 instruments, if so, append the token to the latest split, otherwise create a new split and append the token to it
                if len(split2instrument)==0:
                    split2token[0] = [token]
                    split2instrument[0] = [instrument]
                    instrument2split[instrument] = 0
                else:
                    last_split = list(split2instrument.keys())[-1]
                    if len(split2instrument[last_split]) < 15:
                        split2token[last_split].append(token)
                        split2instrument[last_split].append(instrument)
                        instrument2split[instrument] = last_split
                    else:
                        new_split = last_split + 1
                        split2token[new_split]=[token]
                        split2instrument[new_split]=[instrument]
                        instrument2split[instrument] = new_split

        return [value for _, value in split2token.items()]


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

def onset2bar_beat_chord(onsets, chord_condition, time_signature_condition, bpm_condition, num_measures_condition, chord_dict):
    batch_size = onsets.size(0)
    output = []
    
    for i in range(batch_size):
        # Extract data for current batch element
        onset_abs = onsets[i].item()
        numerator = int(time_signature_condition[i].split("/")[0])
        denominator = int(time_signature_condition[i].split("/")[1])
        bpm = bpm_condition[i]
        is_incomplete_measure = num_measures_condition[i]%4!=0 
        if is_incomplete_measure:
            chords = ["s"]*8 + chord_condition[i]  # List of chord symbols for this batch
        else:
            chords = chord_condition[i]  # List of chord symbols for this batch
        # Convert absolute time to seconds
        onset_sec = onset_abs / 100.0 * (120 / bpm) #data normalized to 120 bpm now need to convert it back
        
        # Compute total beats
        total_beats = (onset_sec * bpm ) / 60.0
        
        # Compute bar number (0-based)
        # bar_number = int(total_beats // numerator)
        bar_len_in_beats = numerator * (4 / denominator)
        bar_number = int(total_beats // bar_len_in_beats)

        bar_number = min(bar_number, num_measures_condition[i]-1) #happens when one of the sequences in the batch reaches max num measures
        # Position within the bar in beats
        position_in_bar_beats = total_beats % numerator
        
        # Convert position to 32nd notes within the bar
        # Each beat has (32 / denominator) 32nd notes --> each beat has 8 32nd notes 
        # position_in_32nd = position_in_bar_beats * (32.0 / denominator) #this seems to be wrong 
        position_in_32nd = position_in_bar_beats * 8 #8 32nd notes per beat

        quantized_32nd = round(position_in_32nd)
        """# Handle wrap-around for bar's 32nd note capacity
        max_32nd_per_bar = numerator * (32 // denominator)
        quantized_32nd = quantized_32nd % max_32nd_per_bar"""
        
        # Calculate maximum 32nd notes per bar
        # max_32nd_per_bar = numerator * (32 // denominator)
        max_32nd_per_bar = bar_len_in_beats * 8 #8 32nd notes per beat
        # Handle bar overflow from quantization
        quantized_32nd = quantized_32nd % max_32nd_per_bar
        
        """# Check if quantization pushed us to next bar
        if quantized_32nd == 0 and position_in_32nd >= (1 + max_32nd_per_bar - 0.5):
            bar_number += 1"""

        # Find chord index based on 8th note divisions
        total_8th_notes = total_beats * 2.0  # Each beat has 2 8th notes
        chord_index = int(total_8th_notes) % len(chords) #this is an ugly fix!
        chord_symbol = chord_dict[chords[chord_index]]
        # Append [bar, quantized_32nd, chord_midi]
        output.append([bar_number, quantized_32nd, chord_symbol])
    
    return torch.tensor(output)
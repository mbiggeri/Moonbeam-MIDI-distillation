# Moonbeam: A MIDI Foundation Model Using Both Absolute and Relative Music Attributes

[Paper Link](https://TODO)  
[Demo Website](https://TODO)  
[Checkpoint](https://TODO) 

## Introduction

Moonbeam is a transformer-based foundation model for symbolic music, pretrained on a large and diverse collection of MIDI data totaling 81.6K and 18 billion tokens. Unlike most existing transformer-based models for symbolic music, Moonbeam incorporates music-domain inductive biases by capturing both absolute and relative musical attributes through the introduction of a novel domain-knowledge-inspired tokenization method and a novel attention mechanism

## Instructions

### 1. Install Dependencies
```bash
conda create --name moonbeam_test python=3.12
pip install . 
pip install src/llama_recipes/transformers_minimal/.
```

### 2. Download Checkpoints 
Download the pre-trained checkpoints from the following link:
- [Pre-trained Checkpoint](https://TODO)
- [Unconditional Generation Checkpoint (ATEPP-Bach)](https://TODO)
- [Conditional Generation Checkpoint (Commu)](https://TODO)

## Finetuning

### 1. Unconditional Music Generation 
Data Preprocessing:
Finetuning: 
Inferencing: 
```bash
torchrun --nproc_per_node 1 recipes/inference/custom_music_generation/unconditional_music_generation.py \
  --csv_file /PATH/TO/CSV \
  --top_p 0.95 \
  --temperature 0.9 \
  --model_config_path src/llama_recipes/configs/model_config.json \
  --ckpt_dir /PATH/TO/PRETRAINED/CHECKPOINT \
  --finetuned_PEFT_weight_path /PATH/TO/PEFT/WEIGHT \
  --tokenizer_path tokenizer.model \
  --max_seq_len 512 \
  --max_gen_len 512 \
  --max_batch_size 6 \
  --num_test_data 20 \
  --prompt_len 50
```
### 2. Conditional Music Generation and Music Infilling 
Switch to the branch for conditional music generation and music infilling: 
```bash
git checkout conditional_gen_commu
```
Data Preprocessing:
Finetuning: 
Inferencing:

### 3. Music Classification
Switch to the branch for music classification: 
```bash
git checkout finetune_player_classification
```
Data Preprocessing:
Finetuning: 
Inferencing:
## License

## Bibtex

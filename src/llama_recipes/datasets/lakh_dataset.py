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
def decimal_to_binary(dec, bits = 10): 
    """20 -- > 0b10100 --> 10100 --> 000...10100 (str format) """
    decimal = int(dec) 
    binary_str = bin(decimal)[2:]
    if not len(binary_str)<=bits:
        print("increase number of bits!")
    assert len(binary_str)<=bits
    padded_binary_str = binary_str.zfill(bits)  # Left-pad with zeros
    padded_binary = torch.tensor([int(bit) for bit in padded_binary_str])
    return padded_binary

def binary_to_decimal(binary_str):
    """
    Convert binary string to decimal.
    """
    decimal = int(binary_str, 2)  # Convert binary string to decimal
    return decimal

def pitch_to_octave_pitch_class(pitch):
    return pitch//12, pitch%12

def octave_pitch_class_to_pitch(octave, pitch_class):
    return int(octave*12+pitch_class)

def convert_midi(filename, debug=False):
    
    # try:
    tokens = midi_to_compound(filename, debug=debug)
    print("length", len(tokens))
    return tokens
    # except Exception:
    #     if debug:
    #         print('Failed to process: ', filename)
    #         print(traceback.format_exc())

    #     return 1

    # with open(f"{filename}.compound.txt", 'w') as f:
    #     f.write(' '.join(str(tok) for tok in tokens))

    # return 0

def midi_to_compound(midifile,TIME_RESOLUTION=100, debug=False):

    """
    modified from: https://github.com/jthickstun/anticipation/blob/main/anticipation/convert.py#L128 
    midifile: a midi file path or a mido object
    output: a compound list, each item contains: [onset, duration, pitch, instrument, velocity]
    """

    if type(midifile) == str:
        midi = mido.MidiFile(midifile)
    else:
        midi = midifile

    tokens = [] #output contains tuples of (onset, duration, pitch, instr, velocity)
    note_idx = 0
    open_notes = defaultdict(list)

    time = 0
    instruments = defaultdict(int) # default to code 0 = piano
    tempo = 500000 # default tempo: 500000 microseconds per beat = 0.5 seconds
    for message in midi:
        time += message.time

        # sanity check: negative time?
        if message.time < 0:
            raise ValueError
        #messages: program_change, note_on, note_off
        if message.type == 'program_change':
            instruments[message.channel] = message.program #assign an instrument name to that channel
        elif message.type in ['note_on', 'note_off']:
            # special case: channel 9 is drums!
            instr = 128 if message.channel == 9 else instruments[message.channel] #if drum--> instru = 128; else instrument in the program change message
            if message.type == 'note_on' and message.velocity > 0: # onset
                # time quantization --> convert to binary
                time_in_ticks = round(TIME_RESOLUTION*time)
                # time_in_ticks_binary = decimal_to_binary(time_in_ticks, bits = onset_bits)
                # pitch--> octave, pitch_class
                octave, pitch_class = pitch_to_octave_pitch_class(message.note)

                # tokens.append([time_in_ticks, -1, [octave, pitch_class], instr, message.velocity]) #not stackable! 
                tokens.append([time_in_ticks, -1, octave, pitch_class, instr, message.velocity]) #stackable! 
                open_notes[(instr,message.note,message.channel)].append((note_idx, time))
                note_idx += 1
            else: # offset
                try:
                    open_idx, onset_time = open_notes[(instr,message.note,message.channel)].pop(0)
                except IndexError:
                    if debug:
                        print('WARNING: ignoring bad offset') #note with offset but without onset
                else:
                    duration_ticks = round(TIME_RESOLUTION*(time-onset_time))
                    tokens[open_idx][1] = duration_ticks
                    #del open_notes[(instr,message.note,message.channel)]
        elif message.type == 'set_tempo':
            tempo = message.tempo
        elif message.type == 'time_signature':
            pass # we use real time
        elif message.type in ['aftertouch', 'polytouch', 'pitchwheel', 'sequencer_specific']:
            pass # we don't attempt to model these
        elif message.type == 'control_change':
            pass # this includes pedal and per-track volume: ignore for now
        elif message.type in ['track_name', 'text', 'end_of_track', 'lyrics', 'key_signature',
                              'copyright', 'marker', 'instrument_name', 'cue_marker',
                              'device_name', 'sequence_number']:
            pass # possibly useful metadata but ignore for now
        elif message.type == 'channel_prefix':
            pass # relatively common, but can we ignore this?
        elif message.type in ['midi_port', 'smpte_offset', 'sysex']:
            pass # I have no idea what this is
        else:
            if debug:
                print('UNHANDLED MESSAGE', message.type, message)

    #assign a hard stop timing (dur = 250ms) for all open notes.
    unclosed_count = 0
    for (instr,note,channel) ,v in open_notes.items():
        unclosed_count += len(v)
        for (open_idx, onset_time) in v:
            tokens[open_idx][1] = TIME_RESOLUTION//4



    if debug and unclosed_count > 0:
        print(f'WARNING: {unclosed_count} unclosed notes')
        print('  ', midifile) #TODO: sort based on onset and pitch?

    return tokens



class MusicTokenizer():
    def __init__(self, max_instr = 128, onset_vocab_size = 21, dur_vocab_size = 1001, octave_vocab_size = 9, pitch_class_vocab_size = 12, instrument_vocab_size = 128, velocity_vocab_size = 129):
        self.max_instr = max_instr #TODO: change here definitely
        self.onset_vocab_size = onset_vocab_size
        self.dur_vocab_size = dur_vocab_size
        self.octave_vocab_size = octave_vocab_size
        self.pitch_class_vocab_size = pitch_class_vocab_size
        self.instrument_vocab_size = instrument_vocab_size
        self.velocity_vocab_size = velocity_vocab_size
        self.sos_label = torch.tensor([0, 1]+[0 for _ in range(self.onset_vocab_size-2)]+ [self.dur_vocab_size-2, self.octave_vocab_size-2, self.pitch_class_vocab_size-2,self.instrument_vocab_size-2 , self.velocity_vocab_size-2]) #this need to move to the tokenizer
        self.eos_label = torch.tensor([1, 0]+[0 for _ in range(self.onset_vocab_size-2)]+ [self.dur_vocab_size-1, self.octave_vocab_size-1, self.pitch_class_vocab_size-1,self.instrument_vocab_size-1 , self.velocity_vocab_size-1]) #this need to move to the tokenizer

    
    def encode_single(self, raw_token):
        """
        each raw token looks like: [onset_binary_str, duration, [octave, pitch_class],instrument, velocity], ['00000001101100111101', 25, [4, 11], 24, 58] 
        """

        onset = raw_token[0]

        duration = raw_token[1]

        octave = raw_token[2]

        pitch_class = raw_token[3]

        instrument = raw_token[4]

        velocity = raw_token[5]

        return [onset, duration, octave, pitch_class, instrument, velocity]

    def encode_series(self, raw_token_series):
        return [[0 for _ in range(6)]]+[self.encode_single(x) for x in raw_token_series]+[[0 for _ in range(6)]] #during training [0 for _ in range(6)] will become the "position" of the SOS token. EOS wont be position-encoded. 


class LakhDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        #TODO: add partition; 
        #dataset_config should be a config file but now use as a list of midi file
        print("check dataset config", dataset_config)
        paths = sorted(glob.glob(dataset_config.data_path+"/**/*.mid"))
        paths = paths[:6]
        self.all_mid_paths = paths
        print("all mid path", len(self.all_mid_paths))
        self.tokenizer = MusicTokenizer() #TODO: check tokenizer

    def __len__(self):
        return len(self.all_mid_paths)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        mid_path = self.all_mid_paths[index] 
        raw_tokens = convert_midi(mid_path)  #[[onset, dur, octave, pitch_class, instr, velocity], ...]   #TODO: in the future, write preprocessing script and save it beforehand
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

if __name__=="__main__":

    #Lakh Tokenize Config:
    ONSET_BITS = 20
    TIME_RESOLUTION = 100              # 10ms time resolution = 100 bins/second

    #Lakh Dataset Config
    dataset_config = glob.glob("/data/scratch/acw753/lakhmidi/lmd_full/**/*.mid")
    tokenizer = None
    partition = 'train'
    lakh = LakhDataset(dataset_config= dataset_config, tokenizer = tokenizer, partition = partition)
    for mid in lakh:
        print(mid)
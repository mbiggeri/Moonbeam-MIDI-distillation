import torch
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

def pitch_to_octave_pitch_class(pitch):
    return pitch//12, pitch%12

def octave_pitch_class_to_pitch(octave, pitch_class):
    return int(octave*12+pitch_class)


class MusicTokenizer():
    def __init__(self, onset_vocab_size = 21, dur_vocab_size = 1001, octave_vocab_size = 9, pitch_class_vocab_size = 12, instrument_vocab_size = 128, velocity_vocab_size = 129, sos_token = -1, eos_token = -2, pad_token = -3):
        self.onset_vocab_size = onset_vocab_size
        self.dur_vocab_size = dur_vocab_size
        self.octave_vocab_size = octave_vocab_size
        self.pitch_class_vocab_size = pitch_class_vocab_size
        self.instrument_vocab_size = instrument_vocab_size
        self.velocity_vocab_size = velocity_vocab_size
        
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sos_token_compound = [self.sos_token for _ in range(6)]
        self.eos_token_compound = [self.eos_token for _ in range(6)]
        self.pad_token_compound = [self.pad_token for _ in range(6)] 


        self.sos_onset = [0 for _ in range(self.onset_vocab_size-2)] + [1, 0]
        self.eos_onset = [0 for _ in range(self.onset_vocab_size-2)] + [0, 1]

        self.sos_dur, self.eos_dur = self.dur_vocab_size-2, self.dur_vocab_size-1
        self.sos_octave, self.eos_octave = self.octave_vocab_size-2, self.octave_vocab_size-1
        self.sos_pitch_class, self.eos_pitch_class = self.pitch_class_vocab_size-2, self.pitch_class_vocab_size-1
        self.sos_instrument, self.eos_instrument = self.instrument_vocab_size-2, self.instrument_vocab_size-1
        self.sos_velocity, self.eos_velocity = self.velocity_vocab_size-2, self.velocity_vocab_size-1


        self.sos_label = self.sos_onset+ [self.sos_dur, self.sos_octave, self.sos_pitch_class, self.sos_instrument, self.sos_velocity] 
        self.eos_label = self.eos_onset+ [self.eos_dur, self.eos_octave, self.eos_pitch_class, self.eos_instrument, self.eos_velocity]
        

    
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

    def encode_series(self, raw_token_series, if_add_sos, if_add_eos):
        out = [self.encode_single(x) for x in raw_token_series]
        if if_add_sos:
            out = [self.sos_token_compound] + out
        if if_add_eos:
            out = out + [self.eos_token_compound]
        return out #archive: during training [0 for _ in range(6)] will become the "position" of the SOS token. EOS wont be position-encoded. 
    def encode_series_labels(self, encoded_tokens, if_added_sos, if_added_eos):
        encoded_tokens = torch.tensor(encoded_tokens) #temporarily convert to tensor for easier slicing (len, 6)

        if if_added_sos:
            encoded_tokens = encoded_tokens[1:]
        if if_added_eos:
            encoded_tokens = encoded_tokens[:-1]
        
        #now the encoded tokens does not contain sos and eos
        #1. Convert Onset to bits 
        onsets_labels = self.decimal_to_binary_batch(encoded_tokens[:, 0], bits=self.onset_vocab_size) #(len, ) -> (len, onset_vocab_size)
        other_labels = encoded_tokens[:, 1:] #(len, 5)

        #2. Concat the labels 
        output = torch.concat([onsets_labels, other_labels], dim = -1) #(len, onset_vocab_size+5)

        #3. Add sos and eos label if necessary
        if if_added_sos: 
            output = torch.concat([torch.tensor(self.sos_label).unsqueeze(0), output], dim = 0) 
        if if_added_eos:
            output = torch.concat([output, torch.tensor(self.eos_label).unsqueeze(0)], dim = 0) 
        return output.tolist()

    @staticmethod
    def binary_to_decimal_batch(binary):
        """
        Converts a batch of binary representations to their decimal equivalents.

        Args:
            binary (torch.Tensor): A tensor containing binary numbers with each binary number in a row.

        Returns:
            torch.Tensor: A tensor containing the decimal equivalents of the input binary numbers.
        """
        # Create a mask to apply the binary weights
        bits = binary.size(-1) #2 bits are reserved for sos and eos TODO: add exception handling script, e.g., no eos and sos token is added, return dtype?
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(binary.device)

        # Apply the mask and sum up the results to get the decimal numbers
        dec = (binary[..., :] * mask).sum(dim=-1, keepdim=True)

        return dec
    
    @staticmethod
    def decimal_to_binary_batch(dec, bits):
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

    @staticmethod
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

    @staticmethod
    def compound_to_midi(tokens, TIME_RESOLUTION = 100, debug=False):
        #TODO: double check and add doc string
        mid = mido.MidiFile()
        mid.ticks_per_beat = TIME_RESOLUTION // 2 # 2 beats/second at quarter=120

        it = iter(tokens)
        time_index = defaultdict(list)
        for _, (time_in_ticks,duration,note,instrument,velocity) in enumerate(zip(it,it,it,it,it)):
            time_index[(time_in_ticks,0)].append((note, instrument, velocity)) # 0 = onset
            time_index[(time_in_ticks+duration,1)].append((note, instrument, velocity)) # 1 = offset

        track_idx = {} # maps instrument to (track number, current time)
        num_tracks = 0
        for time_in_ticks, event_type in sorted(time_index.keys()):
            for (note, instrument, velocity) in time_index[(time_in_ticks, event_type)]:
                if event_type == 0: # onset
                    try:
                        track, previous_time, idx = track_idx[instrument]
                    except KeyError:
                        idx = num_tracks
                        previous_time = 0
                        track = mido.MidiTrack()
                        mid.tracks.append(track)
                        if instrument == 128: # drums always go on channel 9
                            idx = 9
                            message = mido.Message('program_change', channel=idx, program=0)
                        else:
                            message = mido.Message('program_change', channel=idx, program=instrument)
                        track.append(message)
                        num_tracks += 1
                        if num_tracks == 9:
                            num_tracks += 1 # skip the drums track

                    track.append(mido.Message(
                        'note_on', note=note, channel=idx, velocity=velocity,
                        time=time_in_ticks-previous_time))
                    track_idx[instrument] = (track, time_in_ticks, idx)
                else: # offset
                    try:
                        track, previous_time, idx = track_idx[instrument]
                    except KeyError:
                        # shouldn't happen because we should have a corresponding onset
                        if debug:
                            print('IGNORING bad offset')

                        continue

                    track.append(mido.Message(
                        'note_off', note=note, channel=idx,
                        time=time_in_ticks-previous_time))
                    track_idx[instrument] = (track, time_in_ticks, idx)

        return mid


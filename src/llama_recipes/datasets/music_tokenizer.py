import torch
class MusicTokenizer():
    def __init__(self, onset_vocab_size = 21, dur_vocab_size = 1001, octave_vocab_size = 9, pitch_class_vocab_size = 12, instrument_vocab_size = 128, velocity_vocab_size = 129):
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

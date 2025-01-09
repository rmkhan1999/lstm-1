import torch
from torch.utils.data import Dataset
import numpy as np

class NextWordDataset(Dataset):
    def __init__(self, sequences, max_sequence_length=20):
        self.sequences = sequences
        self.max_sequence_length = max_sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Find first EOL token
        eol_idx = None
        for i, token in enumerate(sequence):
            if token == 1:  # EOL token
                eol_idx = i
                break
        
        if eol_idx is None:
            eol_idx = len(sequence)
        
        # Get meaningful part of sequence (before first EOL)
        meaningful_seq = sequence[:eol_idx]
        
        # Ensure sequence is at least 2 tokens long
        if len(meaningful_seq) < 2:
            meaningful_seq = np.pad(meaningful_seq, (0, 2 - len(meaningful_seq)), 
                                  constant_values=1)  # Pad with EOL
        
        # Truncate or pad sequence to max_sequence_length
        if len(meaningful_seq) > self.max_sequence_length:
            meaningful_seq = meaningful_seq[:self.max_sequence_length]
        else:
            meaningful_seq = np.pad(meaningful_seq, 
                                  (0, self.max_sequence_length - len(meaningful_seq)),
                                  constant_values=1)  # Pad with EOL
        
        # Create input and target sequences
        input_seq = meaningful_seq[:-1]  # All tokens except last
        target_seq = meaningful_seq[1:]   # All tokens except first
        
        return torch.LongTensor(input_seq), torch.LongTensor(target_seq)
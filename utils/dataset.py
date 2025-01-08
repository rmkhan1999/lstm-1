import torch
from torch.utils.data import Dataset

class NextWordDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Input is the first part of the sequence
        input_seq = sequence[:-1]  # all but last token
        
        # Target is the sequence shifted by one position
        target_seq = sequence[1:]  # all but first token
        
        return torch.LongTensor(input_seq), torch.LongTensor(target_seq) 
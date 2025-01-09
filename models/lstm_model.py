import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(LSTMLanguageModel, self).__init__()
        
        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        embeds = self.embed_dropout(embeds)
        
        lstm_out, _ = self.lstm(embeds)  # (batch_size, seq_length, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        
        # Output layer for all time steps
        logits = self.fc(lstm_out)  # (batch_size, seq_length, vocab_size)
        return logits
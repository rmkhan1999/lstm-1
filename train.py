import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_preprocessing import TextPreprocessor, load_and_preprocess_data, create_word2vec_embeddings, prepare_datasets
from utils.dataset import NextWordDataset
from models.lstm_model import LSTMLanguageModel

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, preprocessor, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                val_loss += criterion(outputs, targets).item()
        
        avg_train_loss = total_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'preprocessor_word2idx': preprocessor.word2idx,
                'preprocessor_idx2word': preprocessor.idx2word,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'models/lstm_language_model_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return train_losses, val_losses

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    max_length = 20
    embed_dim = 256  # Reduced from 300
    hidden_dim = 256  # Reduced from 512
    num_layers = 2
    dropout = 0.5
    batch_size = 32   # Reduced from 64
    num_epochs = 50
    learning_rate = 0.001
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_length=max_length)
    
    # Load and preprocess data
    train_sequences, val_sequences, test_sequences = prepare_datasets(
        'data/train.txt',
        'data/val.txt',
        'data/test.txt',
        preprocessor
    )
    
    # Create datasets and dataloaders
    train_dataset = NextWordDataset(train_sequences)
    val_dataset = NextWordDataset(val_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LSTMLanguageModel(
        vocab_size=preprocessor.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=preprocessor.word2idx['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5, verbose=True
    )
    
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, 
        optimizer, scheduler, num_epochs, device, 
        preprocessor
    )
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessor_word2idx': preprocessor.word2idx,
        'preprocessor_idx2word': preprocessor.idx2word,
    }, 'models/lstm_language_model.pth')
    
if __name__ == '__main__':
    main() 
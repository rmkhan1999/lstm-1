import torch
from models.lstm_model import LSTMLanguageModel
from utils.data_preprocessing import (
    TextPreprocessor, 
    load_and_preprocess_data, 
    prepare_datasets
)
from torch.utils.data import DataLoader
from utils.dataset import NextWordDataset

def predict_next_words(model, preprocessor, input_text, device, max_words=5):
    model.eval()
    
    # Convert input text to sequence
    tokens = input_text.split()
    current_sequence = [preprocessor.word2idx.get(token, preprocessor.word2idx['<EOL>']) for token in tokens]
    
    generated_words = []
    
    with torch.no_grad():
        while len(generated_words) < max_words:
            # Prepare input tensor
            input_tensor = torch.LongTensor([current_sequence]).to(device)
            
            # Get predictions
            outputs = model(input_tensor)
            next_word_logits = outputs[0, -1, :]
            
            # Get top 5 most likely words
            top_k = 5
            top_indices = torch.topk(next_word_logits, top_k).indices
            
            # Find first non-EOL, non-PAD token
            next_word_idx = None
            for idx in top_indices:
                idx = idx.item()
                if idx not in [preprocessor.word2idx['<PAD>'], preprocessor.word2idx['<EOL>']]:
                    next_word_idx = idx
                    break
            
            if next_word_idx is None or next_word_idx == preprocessor.word2idx['<EOL>']:
                break
                
            # Add the predicted word
            next_word = preprocessor.idx2word[next_word_idx]
            generated_words.append(next_word)
            
            # Update sequence for next prediction
            current_sequence = current_sequence + [next_word_idx]
    
    return generated_words

def calculate_accuracy(model, test_loader, preprocessor, device):
    model.eval()
    total_correct = 0
    total_words = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Get predictions
            predictions = outputs.argmax(dim=-1)
            
            # Calculate accuracy (excluding PAD tokens)
            mask = targets != preprocessor.word2idx['<PAD>']
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_words += mask.sum().item()
    
    return total_correct / total_words

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model and preprocessor state
    checkpoint = torch.load('models/lstm_language_model_best.pth')
    
    # Initialize preprocessor with saved state
    preprocessor = TextPreprocessor(max_length=20)
    preprocessor.word2idx = checkpoint['preprocessor_word2idx']
    preprocessor.idx2word = checkpoint['preprocessor_idx2word']
    preprocessor.vocab_size = len(preprocessor.word2idx)
    
    # Load test data
    _, _, test_sequences, _ = prepare_datasets(
        'data/train.txt',
        'data/val.txt',
        'data/test.txt',
        preprocessor
    )
    
    # Initialize model with same parameters as training
    model = LSTMLanguageModel(
        vocab_size=len(preprocessor.word2idx),
        embed_dim=300,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Example prediction
    input_text = "I eat"
    predicted_words = predict_next_words(model, preprocessor, input_text, device)
    print(f"Input: {input_text}")
    print(f"Predicted completion: {' '.join(predicted_words)}")
    
    # Debug info
    print("\nDebug Info:")
    print(f"Vocabulary size: {len(preprocessor.word2idx)}")
    print(f"Sample vocabulary: {list(preprocessor.word2idx.items())[:10]}")
    
    # Calculate accuracy on test set
    max_target_length = 20  # Same as training
    test_dataset = NextWordDataset(test_sequences, max_target_length)
    test_loader = DataLoader(test_dataset, batch_size=32)
    accuracy = calculate_accuracy(model, test_loader, preprocessor, device)
    print(f"\nTest Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main() 
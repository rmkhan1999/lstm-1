import torch
from models.lstm_model import LSTMLanguageModel
from utils.data_preprocessing import TextPreprocessor, load_and_preprocess_data

def predict_next_words(model, preprocessor, input_text, device, max_words=5):
    model.eval()
    
    # Convert input text to sequence
    current_sequence = preprocessor.text_to_sequence(input_text)
    
    # Remove padding/EOL tokens from input
    input_tokens = input_text.split()
    input_len = len(input_tokens)
    
    generated_words = []
    
    with torch.no_grad():
        for _ in range(max_words):
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
            
            if next_word_idx is None:
                break
                
            # Add the predicted word
            next_word = preprocessor.idx2word[next_word_idx]
            generated_words.append(next_word)
            
            # Update sequence for next prediction
            current_sequence = current_sequence[1:] + [next_word_idx]
    
    return generated_words

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model and preprocessor state
    checkpoint = torch.load('models/lstm_language_model_best.pth')
    
    # Initialize preprocessor with saved state
    preprocessor = TextPreprocessor(max_length=20)
    preprocessor.word2idx = checkpoint['preprocessor_word2idx']
    preprocessor.idx2word = checkpoint['preprocessor_idx2word']
    preprocessor.vocab_size = len(preprocessor.word2idx)
    
    # Initialize model with same parameters as training
    model = LSTMLanguageModel(
        vocab_size=len(preprocessor.word2idx),
        embed_dim=256,  # Match training dimensions
        hidden_dim=256,  # Match training dimensions
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

if __name__ == '__main__':
    main() 
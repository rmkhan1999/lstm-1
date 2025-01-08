import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    def __init__(self, max_length=20):
        self.max_length = max_length
        self.word2idx = {'<PAD>': 0, '<EOL>': 1}
        self.idx2word = {0: '<PAD>', 1: '<EOL>'}
        self.vocab_size = 2  # Starting with PAD and EOL tokens
        
    def fit(self, sentences):
        # Build vocabulary from sentences
        words = set()
        for sentence in sentences:
            words.update(sentence.split())
            
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
    def pad_sequence(self, tokens):
        if len(tokens) > self.max_length:
            return tokens[:self.max_length]
        return tokens + ['<EOL>'] * (self.max_length - len(tokens))
    
    def text_to_sequence(self, text):
        tokens = text.split()
        padded = self.pad_sequence(tokens)
        return [self.word2idx.get(token, self.word2idx['<EOL>']) for token in padded]

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the emotion-labeled dataset
    Format: sentence;emotion
    """
    # Read the file and split by semicolon
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Split each line by the last semicolon to separate sentence and emotion
    sentences = [line.strip().rsplit(';', 1)[0] for line in lines]
    
    return sentences

def prepare_datasets(train_file, val_file, test_file, preprocessor):
    """
    Prepare train, validation and test datasets
    """
    # Load raw sentences
    train_sentences = load_and_preprocess_data(train_file)
    val_sentences = load_and_preprocess_data(val_file)
    test_sentences = load_and_preprocess_data(test_file)
    
    # Fit preprocessor on training data
    preprocessor.fit(train_sentences)
    
    # Convert sentences to sequences
    train_sequences = [preprocessor.text_to_sequence(sent) for sent in train_sentences]
    val_sequences = [preprocessor.text_to_sequence(sent) for sent in val_sentences]
    test_sequences = [preprocessor.text_to_sequence(sent) for sent in test_sentences]
    
    return np.array(train_sequences), np.array(val_sequences), np.array(test_sequences)

def create_word2vec_embeddings(sentences, embed_dim=300):
    # Train Word2Vec model
    tokenized_sentences = [sent.split() for sent in sentences]
    model = Word2Vec(sentences=tokenized_sentences, 
                    vector_size=embed_dim, 
                    window=5, 
                    min_count=1)
    
    return model 
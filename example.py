#!/usr/bin/env python3
"""
Simple example of using the Sequential Pattern Transformer for predictions.
This script demonstrates how to load a trained model and make predictions.
"""

import os
import sys
import torch
import pickle
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.models.model import TransformerDecoder

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load trained model and tokenizer."""
    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab_size = len(tokenizer.token2idx)
    
    # Try to infer max_len from the checkpoint or use default
    try:
        # Get max_len from positional encoding shape
        max_len = checkpoint['model_state_dict']['pos_encoding.pe'].shape[1]
    except:
        max_len = 512  # Default fallback
    
    model = TransformerDecoder(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        tie_weights=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def predict_next_tokens(model, tokenizer, input_sequence, top_k=5, device='cpu'):
    """Predict the top-k most likely next tokens for a given sequence."""
    model.to(device)
    
    # Tokenize and encode the input
    tokens = tokenizer.tokenize_sequence(input_sequence.split())
    encoded = tokenizer.encode_sequence(tokens, add_special_tokens=False)
    
    if not encoded:
        print("Warning: Empty sequence after encoding")
        return []
    
    # Convert to tensor
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    
    # Generate key padding mask
    pad_id = tokenizer.token2idx[tokenizer.special_tokens['PAD']]
    key_padding_mask = (input_ids == pad_id)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, key_padding_mask=key_padding_mask)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        # Get logits for the last token
        last_token_logits = logits[0, -1, :]
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(last_token_logits, k=top_k)
        probabilities = torch.softmax(last_token_logits, dim=-1)
        top_k_probs = probabilities[top_k_indices]
        
        # Convert to tokens
        predictions = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            token = tokenizer.idx2token.get(idx.item(), '[UNK]')
            predictions.append((token, prob.item()))
    
    return predictions

def main():
    # Example usage
    model_path = "outputs/best_model.pth"
    tokenizer_path = "outputs/tokenizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print(f"Model or tokenizer not found. Please train a model first using main.py")
        print(f"Expected files:")
        print(f"  {model_path}")
        print(f"  {tokenizer_path}")
        return
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    
    print(f"Model loaded with vocabulary size: {len(tokenizer.token2idx)}")
    print("\nExample usage:")
    
    # Example medical sequence
    example_sequences = [
        "END002 CIR011",  # Diabetes + Heart disease
        "GEN002 INF002",   # Renal failure + Sepsis  
        "END002 END003",   # Diabetes without/with complications
        "MUS009 END002"    # Osteoarthritis + Diabetes
    ]
    
    for seq in example_sequences:
        print(f"\nInput sequence: '{seq}'")
        predictions = predict_next_tokens(model, tokenizer, seq, top_k=5)
        
        print("Top 5 predictions:")
        for i, (token, prob) in enumerate(predictions, 1):
            if token not in tokenizer.special_tokens.values():
                print(f"  {i}. {token} (probability: {prob:.4f})")

if __name__ == "__main__":
    main()
